"""Orkiestracja runów Task 07: trzy ramiona, jedna pętla, wspólny pomiar.

Ramiona kontrolne są tu policzone **tą samą pętlą** co DPO — ta sama tokenizacja,
to samo maskowanie completion, ten sam optymalizator, ten sam licznik kroków i
tokenów. Użycie do continued SFT gotowego `training/sft.py` (HF `Trainer`) byłoby
wygodniejsze, ale wprowadziłoby drugą implementację maskowania i inny scheduler, a
wtedy różnica między ramionami przestałaby być różnicą metody: kontrola
przy „przybliżonym budżecie" (`tasks/07_dpo_training.md`) ma się różnić stratą, a
nie infrastrukturą.

Trzy straty na jednej pętli:

* `dpo` — klasyczna sigmoid na precomputowanych logprobach referencji;
* `continued_sft` — NLL `chosen`, uśredniony po tokenach;
* `score_weighted_continued_sft` — ten sam NLL, przemnożony przez wagę pary.

Metryki dev są **wspólne dla wszystkich trzech ramion**: referencją jest punkt
startowy, a policzyć go można bez dodatkowego artefaktu, bo w kroku 0 polityka
jeszcze nim jest. Dlatego `dev_metrics` liczy logproby dev na modelu przed
pierwszym krokiem i porównuje z nimi stan końcowy — implicit reward accuracy jest
wtedy zdefiniowana identycznie dla DPO i dla obu kontrol.

Runy są wznawialne: co `checkpoint_every` kroków optymalizatora leci checkpoint z
wagami trenowalnymi **i stanem AdamW**, żeby wznowienie nie gubiło momentów.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from doc2query.training.dpo import (
    ContinuedSFTRecord,
    DPOArm,
    DPOPlanManifest,
    PreferenceRecord,
    ScoreWeightedContinuedSFTRecord,
    ValidatedDPODataset,
    canonical_fingerprint,
    file_sha256,
    ordered_ids_fingerprint,
)
from doc2query.training.dpo_runtime import (
    RUN_CONTRACT,
    save_adapter_atomically,
    torch_dpo_loss,
)
from doc2query.utils.records import write_json

CHECKPOINT_DIR = "checkpoint"
STATE_NAME = "state.json"
TRAINABLE_NAME = "trainable.pt"
OPTIMIZER_NAME = "optimizer.pt"
HISTORY_NAME = "history.jsonl"
MANIFEST_NAME = "run_manifest.json"
ADAPTER_DIR = "adapter"


@dataclass(frozen=True)
class ArmExample:
    """Jedna jednostka pracy ramienia; `weight` jest 1.0 tam, gdzie nie waży."""

    preference_id: str
    prompt: str
    chosen: str
    rejected: str | None
    weight: float


def _weighted_records(dataset: ValidatedDPODataset, arm: DPOArm, split: str) -> list[ArmExample]:
    """Zbuduj kohortę ramienia w kolejności zamrożonej przez packager."""
    if arm == DPOArm.DPO:
        rows: Sequence[PreferenceRecord] = (
            dataset.preference_train if split == "train" else dataset.preference_dev
        )
        return [
            ArmExample(row.preference_id, row.prompt, row.chosen, row.rejected, 1.0) for row in rows
        ]
    if arm == DPOArm.CONTINUED_SFT:
        plain: Sequence[ContinuedSFTRecord] = (
            dataset.continued_sft_train if split == "train" else dataset.continued_sft_dev
        )
        return [
            ArmExample(row.preference_id, row.prompt, row.completion, None, 1.0) for row in plain
        ]
    weighted: Sequence[ScoreWeightedContinuedSFTRecord] = (
        dataset.weighted_sft_train if split == "train" else dataset.weighted_sft_dev
    )
    return [
        ArmExample(row.preference_id, row.prompt, row.completion, None, float(row.sample_weight))
        for row in weighted
    ]


def _completion_logprob(
    model: Any, tokenizer: Any, prompt: str, completion: str, max_length: int
) -> tuple[torch.Tensor, int, int]:
    """Suma logprobów completion z gradientem; prompt ucinany z lewej, nigdy completion."""
    prompt_ids = cast(list[int], tokenizer(prompt, add_special_tokens=False)["input_ids"])
    completion_ids = cast(list[int], tokenizer(completion, add_special_tokens=False)["input_ids"])
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        completion_ids = [*completion_ids, int(eos)]
    room = max_length - len(completion_ids)
    if room < 1:
        raise ValueError("completion nie mieści się w max_length; nie ucinamy completion")
    kept_prompt = prompt_ids[-room:]
    input_ids = torch.tensor(
        [[*kept_prompt, *completion_ids]], dtype=torch.long, device=next(model.parameters()).device
    )
    logits = model(input_ids=input_ids).logits.float()
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    completion_scores = gathered[:, len(kept_prompt) - 1 :]
    return completion_scores.sum(), len(completion_ids), len(kept_prompt)


def dev_metrics(
    *,
    model: Any,
    tokenizer: Any,
    records: Sequence[ArmExample],
    max_length: int,
    beta: float,
    reference: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Zmierz dev bez gradientu; z `reference` dodaje implicit reward accuracy."""
    if not records:
        raise ValueError("dev nie może być pusty: bez niego nie ma na czym wybierać")
    chosen_logprobs: dict[str, float] = {}
    rejected_logprobs: dict[str, float] = {}
    chosen_nll: list[float] = []
    margins: list[float] = []
    was_training = getattr(model, "training", False)
    model.eval()
    with torch.no_grad():
        for row in records:
            chosen, chosen_tokens, _ = _completion_logprob(
                model, tokenizer, row.prompt, row.chosen, max_length
            )
            chosen_value = float(chosen.item())
            chosen_logprobs[row.preference_id] = chosen_value
            chosen_nll.append(-chosen_value / max(1, chosen_tokens))
            if row.rejected is not None:
                rejected, _, _ = _completion_logprob(
                    model, tokenizer, row.prompt, row.rejected, max_length
                )
                rejected_value = float(rejected.item())
                rejected_logprobs[row.preference_id] = rejected_value
                margins.append(chosen_value - rejected_value)
    if was_training:
        model.train()

    metrics: dict[str, Any] = {
        "examples": len(records),
        "mean_chosen_logprob": sum(chosen_logprobs.values()) / len(chosen_logprobs),
        "mean_chosen_nll_per_token": sum(chosen_nll) / len(chosen_nll),
    }
    if margins:
        metrics["mean_policy_margin"] = sum(margins) / len(margins)
        metrics["policy_margin_accuracy"] = sum(1 for value in margins if value > 0) / len(margins)
    if reference is not None and rejected_logprobs:
        rewards = [
            beta
            * (
                (chosen_logprobs[key] - reference[key][0])
                - (rejected_logprobs[key] - reference[key][1])
            )
            for key in rejected_logprobs
            if key in reference
        ]
        if rewards:
            metrics["mean_implicit_reward_margin"] = sum(rewards) / len(rewards)
            metrics["implicit_reward_accuracy"] = sum(1 for value in rewards if value > 0) / len(
                rewards
            )
    return metrics


def _dev_reference(
    *, model: Any, tokenizer: Any, records: Sequence[ArmExample], max_length: int
) -> dict[str, tuple[float, float]]:
    """Logproby punktu startowego na dev; w kroku 0 polityka jest jeszcze referencją."""
    reference: dict[str, tuple[float, float]] = {}
    model.eval()
    with torch.no_grad():
        for row in records:
            if row.rejected is None:
                continue
            chosen, _, _ = _completion_logprob(model, tokenizer, row.prompt, row.chosen, max_length)
            rejected, _, _ = _completion_logprob(
                model, tokenizer, row.prompt, row.rejected, max_length
            )
            reference[row.preference_id] = (float(chosen.item()), float(rejected.item()))
    return reference


def _trainable(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def _save_checkpoint(
    *,
    directory: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    state: Mapping[str, Any],
) -> None:
    """Checkpoint atomowo: wagi trenowalne, stan AdamW i kursor kohorty."""
    staging = directory.parent / f".{directory.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        torch.save(
            {name: parameter.detach().cpu() for name, parameter in _trainable(model).items()},
            staging / TRAINABLE_NAME,
        )
        torch.save(optimizer.state_dict(), staging / OPTIMIZER_NAME)
        (staging / STATE_NAME).write_text(
            json.dumps(dict(state), ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        if directory.exists():
            shutil.rmtree(directory)
        os.replace(staging, directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_checkpoint(
    *, directory: Path, model: Any, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    """Wznów z checkpointu; brak choćby jednego pliku to odmowa, nie cichy restart."""
    for name in (TRAINABLE_NAME, OPTIMIZER_NAME, STATE_NAME):
        if not (directory / name).is_file():
            raise ValueError(f"checkpoint {directory} jest niekompletny: brak {name}")
    tensors = cast(
        dict[str, torch.Tensor], torch.load(directory / TRAINABLE_NAME, map_location="cpu")
    )
    trainable = _trainable(model)
    missing = sorted(set(trainable) - set(tensors))
    orphan = sorted(set(tensors) - set(trainable))
    if missing or orphan:
        raise ValueError(
            f"checkpoint nie pasuje do modelu: brakuje {len(missing)}, nadmiarowych {len(orphan)}"
        )
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(tensors[name].to(parameter.device, dtype=parameter.dtype))
    optimizer.load_state_dict(torch.load(directory / OPTIMIZER_NAME, map_location="cpu"))
    state = cast(dict[str, Any], json.loads((directory / STATE_NAME).read_text(encoding="utf-8")))
    return state


def _arm_loss(
    *,
    arm: DPOArm,
    model: Any,
    tokenizer: Any,
    batch: Sequence[ArmExample],
    reference: Mapping[str, tuple[float, float]],
    beta: float,
    max_length: int,
) -> tuple[torch.Tensor, dict[str, float], int]:
    """Strata mikro-batcha wraz z metrykami i liczbą zużytych tokenów."""
    tokens = 0
    if arm == DPOArm.DPO:
        policy_chosen: list[torch.Tensor] = []
        policy_rejected: list[torch.Tensor] = []
        ref_chosen: list[float] = []
        ref_rejected: list[float] = []
        for row in batch:
            if row.rejected is None:
                raise ValueError("ramię DPO wymaga strony rejected")
            missing = row.preference_id not in reference
            if missing:
                raise ValueError(
                    f"brak logprobu referencji dla {row.preference_id}; DPO nie zgaduje referencji"
                )
            chosen, chosen_tokens, prompt_tokens = _completion_logprob(
                model, tokenizer, row.prompt, row.chosen, max_length
            )
            rejected, rejected_tokens, _ = _completion_logprob(
                model, tokenizer, row.prompt, row.rejected, max_length
            )
            policy_chosen.append(chosen)
            policy_rejected.append(rejected)
            reference_row = reference[row.preference_id]
            ref_chosen.append(reference_row[0])
            ref_rejected.append(reference_row[1])
            tokens += 2 * prompt_tokens + chosen_tokens + rejected_tokens
        device = policy_chosen[0].device
        loss, metrics = torch_dpo_loss(
            torch.stack(policy_chosen),
            torch.stack(policy_rejected),
            torch.tensor(ref_chosen, device=device),
            torch.tensor(ref_rejected, device=device),
            beta,
        )
        return loss, metrics, tokens

    losses: list[torch.Tensor] = []
    weights: list[float] = []
    for row in batch:
        chosen, chosen_tokens, prompt_tokens = _completion_logprob(
            model, tokenizer, row.prompt, row.chosen, max_length
        )
        nll = -chosen / chosen_tokens
        weight = row.weight if arm == DPOArm.SCORE_WEIGHTED_CONTINUED_SFT else 1.0
        losses.append(nll * weight)
        weights.append(weight)
        tokens += prompt_tokens + chosen_tokens
    loss = torch.stack(losses).mean()
    return (
        loss,
        {
            "nll_per_token": float(loss.item()),
            "mean_weight": sum(weights) / len(weights),
        },
        tokens,
    )


def train_arm(
    *,
    arm: DPOArm,
    dataset: ValidatedDPODataset,
    plan: DPOPlanManifest,
    reference: Mapping[str, tuple[float, float]],
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    seed: int | None = None,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    max_steps: int | None = None,
    checkpoint_every: int = 25,
    progress_every: int = 10,
    resume: bool = True,
    authorization_decision: str | None = None,
) -> dict[str, Any]:
    """Wytrenuj jedno ramię planu, wznawialnie, i zamroź manifest runu."""
    if arm not in plan.arms:
        raise ValueError(f"plan nie zawiera ramienia {arm.value}")
    budget = plan.arms[arm]
    run_seed = budget.seeds[0] if seed is None else seed
    if run_seed not in budget.seeds:
        raise ValueError(f"seed {run_seed} nie jest w planie {budget.seeds}")
    target_steps = budget.target_optimizer_steps if max_steps is None else max_steps
    train_rows = _weighted_records(dataset, arm, "train")
    dev_rows = _weighted_records(dataset, arm, "dev")
    dev_pairs = _weighted_records(dataset, DPOArm.DPO, "dev")
    if len(train_rows) != budget.train_example_count:
        raise ValueError(
            f"kohorta {len(train_rows)} nie zgadza się z planem {budget.train_example_count}"
        )
    if ordered_ids_fingerprint([row.preference_id for row in train_rows]) != (
        budget.cohort_fingerprint
    ):
        raise ValueError("kolejność kohorty rozjechała się z planem")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / CHECKPOINT_DIR
    history_path = output_dir / HISTORY_NAME

    torch.manual_seed(run_seed)
    order = list(range(len(train_rows)))
    random.Random(run_seed).shuffle(order)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model nie ma parametrów trenowalnych; adapter musi być trainable")
    optimizer = torch.optim.AdamW(trainable, lr=plan.learning_rate)

    started_step = 0
    cursor = 0
    tokens_consumed = 0
    dev_start: dict[str, Any] | None = None
    dev_reference: dict[str, tuple[float, float]] = {}
    if resume and checkpoint_dir.is_dir():
        state = _load_checkpoint(directory=checkpoint_dir, model=model, optimizer=optimizer)
        if str(state.get("arm")) != arm.value or str(state.get("plan_fingerprint")) != (
            plan.plan_fingerprint
        ):
            raise ValueError("checkpoint pochodzi z innego ramienia albo innego planu")
        started_step = int(state["optimizer_step"])
        cursor = int(state["cursor"])
        tokens_consumed = int(state["tokens_consumed"])
        dev_start = cast(dict[str, Any] | None, state.get("dev_start"))
        dev_reference = {
            key: (float(value[0]), float(value[1]))
            for key, value in cast(
                Mapping[str, Sequence[float]], state.get("dev_reference", {})
            ).items()
        }
        print(f"[{arm.value}] wznowienie od kroku {started_step}/{target_steps}", flush=True)
    elif checkpoint_dir.is_dir():
        raise FileExistsError(
            f"{checkpoint_dir} istnieje, a resume=False; nie nadpisuję cudzej pracy"
        )

    if not dev_reference:
        dev_reference = _dev_reference(
            model=model, tokenizer=tokenizer, records=dev_pairs, max_length=plan.max_length
        )
    if dev_start is None:
        dev_start = dev_metrics(
            model=model,
            tokenizer=tokenizer,
            records=dev_rows,
            max_length=plan.max_length,
            beta=plan.beta,
            reference=dev_reference,
        )

    micro = batch_size * gradient_accumulation_steps
    history: list[dict[str, Any]] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started_at = time.perf_counter()
    step = started_step
    epochs = cursor // max(1, len(order))

    while step < target_steps:
        if cursor >= len(order):
            cursor = 0
            epochs += 1
            random.Random(run_seed + epochs).shuffle(order)
        window = order[cursor : cursor + micro]
        if not window:
            break
        cursor += len(window)
        step_metrics: list[dict[str, float]] = []
        step_loss = 0.0
        for start in range(0, len(window), batch_size):
            batch = [train_rows[index] for index in window[start : start + batch_size]]
            loss, metrics, tokens = _arm_loss(
                arm=arm,
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                reference=reference,
                beta=plan.beta,
                max_length=plan.max_length,
            )
            tokens_consumed += tokens
            scaled = loss / max(1, len(window) // batch_size)
            scaled.backward()  # type: ignore[no-untyped-call]
            step_loss += float(loss.item()) * len(batch) / len(window)
            step_metrics.append(metrics)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        row = {
            "step": step,
            "loss": step_loss,
            "tokens_consumed": tokens_consumed,
            "epoch": epochs,
            **{
                key: sum(entry[key] for entry in step_metrics) / len(step_metrics)
                for key in step_metrics[0]
            },
        }
        history.append(row)
        if progress_every and step % progress_every == 0:
            print(
                f"[{arm.value}] krok {step}/{target_steps} loss {step_loss:.4f} "
                f"tokeny {tokens_consumed}",
                flush=True,
            )
        if checkpoint_every and step % checkpoint_every == 0 and step < target_steps:
            _append_history(history_path, history)
            history = []
            _save_checkpoint(
                directory=checkpoint_dir,
                model=model,
                optimizer=optimizer,
                state={
                    "arm": arm.value,
                    "plan_fingerprint": plan.plan_fingerprint,
                    "optimizer_step": step,
                    "cursor": cursor,
                    "tokens_consumed": tokens_consumed,
                    "dev_start": dev_start,
                    "dev_reference": {
                        key: list(value) for key, value in sorted(dev_reference.items())
                    },
                },
            )
    _append_history(history_path, history)
    elapsed = time.perf_counter() - started_at

    dev_end = dev_metrics(
        model=model,
        tokenizer=tokenizer,
        records=dev_rows,
        max_length=plan.max_length,
        beta=plan.beta,
        reference=dev_reference,
    )
    adapter_path = save_adapter_atomically(model, tokenizer, output_dir / ADAPTER_DIR)

    payload: dict[str, Any] = {
        "contract": RUN_CONTRACT,
        "arm": arm.value,
        "plan_id": plan.plan_id,
        "plan_fingerprint": plan.plan_fingerprint,
        "dataset_fingerprint": plan.dataset_fingerprint,
        "cohort_fingerprint": budget.cohort_fingerprint,
        "seed": run_seed,
        "beta": plan.beta,
        "learning_rate": plan.learning_rate,
        "loss_type": plan.loss_type,
        "max_length": plan.max_length,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "target_optimizer_steps": target_steps,
        "completed_optimizer_steps": step,
        "resumed_from_step": started_step,
        "epochs_started": epochs + 1,
        "target_token_budget": budget.target_token_budget,
        "tokens_consumed": tokens_consumed,
        "train_example_count": budget.train_example_count,
        "weight_policy_id": budget.weight_policy_id,
        "dev": {"start": dev_start, "end": dev_end},
        "history": {
            "path": history_path.name,
            "sha256": file_sha256(history_path),
            # Liczymy wiersze pliku, a nie kroki tego procesu: po wznowieniu historia
            # zawiera także kroki policzone przed przerwaniem, a manifest ma opisywać
            # plik, który wskazuje.
            "record_count": sum(
                1 for line in history_path.read_text(encoding="utf-8").split("\n") if line.strip()
            ),
            "steps_this_process": step - started_step,
        },
        "adapter": {
            "path": str(adapter_path.relative_to(output_dir)),
            "fingerprint": _adapter_fingerprint(adapter_path),
        },
        "seconds_total": round(elapsed, 1),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
        "training_started": True,
        "task07_training_authorized": True,
        "authorization_decision": authorization_decision,
        "final_tests_used": [],
    }
    payload["artifact_fingerprint"] = canonical_fingerprint(payload)
    write_json(output_dir / MANIFEST_NAME, payload)
    return payload


def _append_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Historia rośnie dopisywaniem, żeby przerwany run zostawił to, co policzył."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.touch(exist_ok=True)
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _adapter_fingerprint(directory: Path) -> str:
    files = {
        str(path.relative_to(directory)): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    return canonical_fingerprint({"kind": "trained_adapter", "files": files})


__all__ = [
    "ADAPTER_DIR",
    "CHECKPOINT_DIR",
    "MANIFEST_NAME",
    "ArmExample",
    "dev_metrics",
    "train_arm",
]
