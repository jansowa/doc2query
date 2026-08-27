"""Runtime DPO: precompute reference logprobs, potem trening polityki z adaptera SFT.

`training/dpo.py` trzyma kontrakty, walidatory i czystą funkcję straty; brakowało
wyłącznie warstwy wykonawczej. Ten moduł ją dodaje w **dwóch fazach**, dokładnie
takich, jakich wymagają istniejące kontrakty (`ReferenceLogprobManifest` istnieje
właśnie po to):

1. **precompute** — model referencyjny (baza + adapter SFT, zamrożony) liczy
   logprob `chosen` i `rejected` dla każdej pary, wznawialnie po journalu, i zwalnia
   pamięć przed treningiem. To pierwszy krok oszczędzania pamięci z §„Memory
   strategy" Task 07 i jednocześnie jedyny sposób, żeby referencja **dowodliwie**
   odpowiadała punktowi startowemu;
2. **trening** — polityka startuje z tego samego adaptera i uczy się klasyczną stratą
   sigmoid DPO na precomputowanych logprobach referencji.

Dlaczego nie `trl.DPOTrainer`: w TRL 0.29.1 `ref_model=None` przy modelu PEFT liczy
referencję jako bazę z **wyłączonym** adapterem, czyli nie punkt startowy, a Task 07
wymaga wprost „walidacji, że model referencyjny odpowiada dokładnie punktowi
startowemu". Drugi pełny model referencyjny nie mieści się obok polityki na 8 GB.
Dwufazowy precompute rozwiązuje oba problemy naraz, a strata jest tą samą funkcją,
którą repo ma już przetestowaną — tutaj w wariancie torchowym, sprawdzanym w teście
wobec skalarnej wersji referencyjnej.

Moduł nie autoryzuje niczego: nie czyta testów finalnych, nie promuje wyników i nie
zmienia flagi `task07_training_authorized`.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from doc2query.preferences.diversity_gate import _reject_final_test_path
from doc2query.training.dpo import (
    PreferenceRecord,
    canonical_fingerprint,
    file_sha256,
    ordered_ids_fingerprint,
    sigmoid_dpo_loss,
)
from doc2query.utils.records import read_records, write_json

PRECOMPUTE_CONTRACT = "task07-precomputed-reference-logprobs-v1"
RUN_CONTRACT = "task07-dpo-run-v1"
JOURNAL_NAME = "reference_logprobs.journal.jsonl"
RECORDS_NAME = "reference_logprobs.jsonl"


def load_preference_records(path: Path) -> list[PreferenceRecord]:
    """Odczytaj pary preferencji w kolejności pliku; kolejność jest częścią kontraktu."""
    _reject_final_test_path(path)
    rows = [PreferenceRecord.model_validate(row) for row in read_records(path)]
    if not rows:
        raise ValueError(f"{path}: zbiór preferencji jest pusty")
    seen: set[str] = set()
    for row in rows:
        if row.preference_id in seen:
            raise ValueError(f"duplikat preference_id: {row.preference_id}")
        seen.add(row.preference_id)
        if row.split == "test":
            raise ValueError("DPO nie przyjmuje rekordów ze splitu test")
    return rows


@dataclass(frozen=True)
class SequenceScore:
    """Logprob completion dla jednej pary prompt/completion."""

    logprob: float
    prompt_tokens: int
    completion_tokens: int
    truncated: bool


def sequence_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    completion: str,
    *,
    max_length: int,
) -> SequenceScore:
    """Policz sumaryczny logprob tokenów completion, maskując prompt.

    Truncation jest **od lewej po stronie promptu**: gubienie tokenów completion
    zmieniłoby to, co model ocenia, więc completion jest nienaruszalny, a nadmiar
    ucinany z początku promptu. Fakt ucięcia jest raportowany, nie ukrywany.
    """
    if max_length < 2:
        raise ValueError("max_length musi zostawić miejsce na prompt i completion")
    prompt_ids = cast(list[int], tokenizer(prompt, add_special_tokens=False)["input_ids"])
    completion_ids = cast(list[int], tokenizer(completion, add_special_tokens=False)["input_ids"])
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        completion_ids = [*completion_ids, int(eos)]
    if not completion_ids:
        raise ValueError("completion nie może być pusty po tokenizacji")
    if len(completion_ids) >= max_length:
        raise ValueError(
            f"completion ma {len(completion_ids)} tokenów przy max_length={max_length}; "
            "DPO nie ucina completion"
        )
    room = max_length - len(completion_ids)
    truncated = len(prompt_ids) > room
    kept_prompt = prompt_ids[-room:] if truncated else prompt_ids
    input_ids = torch.tensor([[*kept_prompt, *completion_ids]], dtype=torch.long)
    input_ids = input_ids.to(next(model.parameters()).device)
    logits = model(input_ids=input_ids).logits.float()
    # Predykcja tokenu t jest w logits[t-1]; oceniamy wyłącznie ogon completion.
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    completion_slice = gathered[:, len(kept_prompt) - 1 :]
    return SequenceScore(
        logprob=float(completion_slice.sum().item()),
        prompt_tokens=len(kept_prompt),
        completion_tokens=len(completion_ids),
        truncated=truncated,
    )


def _journal_prefix(path: Path) -> dict[str, dict[str, Any]]:
    """Odczytaj trwały prefiks journala; niepełna ostatnia linia jest pomijana."""
    if not path.is_file():
        return {}
    done: dict[str, dict[str, Any]] = {}
    # Dzielimy TYLKO po "\n": pasaże msmarco_pl zawierają U+0085 i U+2028, które
    # `splitlines()` traktuje jako łamanie linii, a JSON ich nie escapuje.
    for line in path.read_text(encoding="utf-8").split("\n"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        done[str(row["preference_id"])] = row
    return done


def _append_journal(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def precompute_reference_logprobs(
    *,
    records: Sequence[PreferenceRecord],
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    max_length: int,
    dataset_fingerprint: str,
    plan_fingerprint: str,
    reference_model: Mapping[str, Any],
    tokenizer_fingerprint: str,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Policz logprob referencji dla każdej pary, wznawialnie, i zamroź manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = output_dir / JOURNAL_NAME
    done = _journal_prefix(journal_path)
    resumed_records = len(done)
    truncated_prompts = 0
    model.eval()
    with torch.no_grad():
        for position, record in enumerate(records):
            previous = done.get(record.preference_id)
            if previous is not None:
                if int(previous["position"]) != position:
                    raise ValueError(
                        f"journal zapisał {record.preference_id} na innej pozycji "
                        f"({previous['position']} vs {position}); wznowienie odmawia "
                        "mieszania kolejności datasetu"
                    )
                truncated_prompts += int(bool(previous.get("prompt_truncated")))
                continue
            chosen = sequence_logprob(
                model, tokenizer, record.prompt, record.chosen, max_length=max_length
            )
            rejected = sequence_logprob(
                model, tokenizer, record.prompt, record.rejected, max_length=max_length
            )
            truncated = chosen.truncated or rejected.truncated
            truncated_prompts += int(truncated)
            row = {
                "preference_id": record.preference_id,
                "position": position,
                "chosen_logprob": chosen.logprob,
                "rejected_logprob": rejected.logprob,
                "chosen_completion_tokens": chosen.completion_tokens,
                "rejected_completion_tokens": rejected.completion_tokens,
                "prompt_tokens": chosen.prompt_tokens,
                "prompt_truncated": truncated,
            }
            _append_journal(journal_path, row)
            done[record.preference_id] = row
            if progress_every and (position + 1) % progress_every == 0:
                print(f"[precompute] {position + 1}/{len(records)}", flush=True)

    ordered = [done[record.preference_id] for record in records]
    records_path = output_dir / RECORDS_NAME
    staging = records_path.with_suffix(f".staging-{uuid.uuid4().hex}")
    try:
        with staging.open("w", encoding="utf-8") as handle:
            for row in ordered:
                handle.write(
                    json.dumps(
                        {
                            "preference_id": row["preference_id"],
                            "position": row["position"],
                            "chosen_logprob": row["chosen_logprob"],
                            "rejected_logprob": row["rejected_logprob"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        os.replace(staging, records_path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise

    payload: dict[str, Any] = {
        "contract": PRECOMPUTE_CONTRACT,
        "records": {
            "path": records_path.name,
            "sha256": file_sha256(records_path),
            "record_count": len(ordered),
        },
        "ordered_preference_ids_fingerprint": ordered_ids_fingerprint(
            [record.preference_id for record in records]
        ),
        "dataset_fingerprint": dataset_fingerprint,
        "plan_fingerprint": plan_fingerprint,
        "reference_model": dict(reference_model),
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "final_tests_used": [],
    }
    payload["artifact_fingerprint"] = canonical_fingerprint(payload)
    manifest_path = output_dir / "reference_logprobs.manifest.json"
    write_json(manifest_path, payload)
    return {
        "manifest": payload,
        "manifest_path": str(manifest_path),
        "prompt_truncated_count": truncated_prompts,
        "record_count": len(ordered),
        "resumed_records": resumed_records,
        "computed_records": len(ordered) - resumed_records,
    }


def torch_dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Klasyczna strata sigmoid DPO wraz z metrykami nagród; różniczkowalna."""
    if beta <= 0.0:
        raise ValueError("beta musi być dodatnia")
    chosen_reward = beta * (policy_chosen - reference_chosen)
    rejected_reward = beta * (policy_rejected - reference_rejected)
    logits = chosen_reward - rejected_reward
    loss = -torch.nn.functional.logsigmoid(logits).mean()
    metrics = {
        "chosen_reward": float(chosen_reward.mean().item()),
        "rejected_reward": float(rejected_reward.mean().item()),
        "reward_margin": float(logits.mean().item()),
        "reward_accuracy": float((logits > 0).float().mean().item()),
    }
    return loss, metrics


def _batches(count: int, size: int) -> Iterator[range]:
    for start in range(0, count, size):
        yield range(start, min(start + size, count))


def run_dpo_steps(
    *,
    records: Sequence[PreferenceRecord],
    reference: Mapping[str, tuple[float, float]],
    model: Any,
    tokenizer: Any,
    beta: float,
    learning_rate: float,
    max_length: int,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Wykonaj optymalizację DPO na precomputowanych logprobach referencji."""
    missing = [row.preference_id for row in records if row.preference_id not in reference]
    if missing:
        raise ValueError(
            f"brak precomputowanych logprobów referencji dla {len(missing)} par "
            f"(pierwsza: {missing[0]}); DPO nie zgaduje referencji"
        )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    model.train()
    history: list[dict[str, Any]] = []
    completed = 0
    lengths: dict[str, list[int]] = {"chosen": [], "rejected": []}
    for indices in _batches(len(records), batch_size):
        if max_steps is not None and completed >= max_steps:
            break
        policy_chosen: list[torch.Tensor] = []
        policy_rejected: list[torch.Tensor] = []
        ref_chosen: list[float] = []
        ref_rejected: list[float] = []
        for index in indices:
            record = records[index]
            chosen = _policy_logprob(model, tokenizer, record.prompt, record.chosen, max_length)
            rejected = _policy_logprob(
                model, tokenizer, record.prompt, record.rejected, max_length
            )
            policy_chosen.append(chosen[0])
            policy_rejected.append(rejected[0])
            lengths["chosen"].append(chosen[1])
            lengths["rejected"].append(rejected[1])
            reference_row = reference[record.preference_id]
            ref_chosen.append(reference_row[0])
            ref_rejected.append(reference_row[1])
        device = policy_chosen[0].device
        loss, metrics = torch_dpo_loss(
            torch.stack(policy_chosen),
            torch.stack(policy_rejected),
            torch.tensor(ref_chosen, device=device),
            torch.tensor(ref_rejected, device=device),
            beta,
        )
        scaled = loss / gradient_accumulation_steps
        scaled.backward()  # type: ignore[no-untyped-call]
        completed += 1
        if completed % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        history.append({"step": completed, "loss": float(loss.item()), **metrics})
    if completed % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {
        "steps": completed,
        "history": history,
        "final_loss": history[-1]["loss"] if history else None,
        "mean_reward_accuracy": (
            sum(row["reward_accuracy"] for row in history) / len(history) if history else None
        ),
        "completion_length_mean": {
            role: (sum(values) / len(values) if values else None)
            for role, values in lengths.items()
        },
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
    }


def _policy_logprob(
    model: Any, tokenizer: Any, prompt: str, completion: str, max_length: int
) -> tuple[torch.Tensor, int]:
    """Logprob completion z gradientem; ta sama maska co w precompute."""
    prompt_ids = cast(list[int], tokenizer(prompt, add_special_tokens=False)["input_ids"])
    completion_ids = cast(list[int], tokenizer(completion, add_special_tokens=False)["input_ids"])
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        completion_ids = [*completion_ids, int(eos)]
    room = max_length - len(completion_ids)
    if room < 1:
        raise ValueError("completion nie mieści się w max_length; DPO nie ucina completion")
    kept_prompt = prompt_ids[-room:]
    input_ids = torch.tensor(
        [[*kept_prompt, *completion_ids]], dtype=torch.long, device=next(model.parameters()).device
    )
    logits = model(input_ids=input_ids).logits.float()
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return gathered[:, len(kept_prompt) - 1 :].sum(), len(completion_ids)


def load_reference_logprobs(manifest_path: Path) -> dict[str, tuple[float, float]]:
    """Odczytaj precomputowane logproby, weryfikując SHA-256 z manifestu."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("contract") != PRECOMPUTE_CONTRACT:
        raise ValueError("manifest logprobów referencji ma inny kontrakt")
    artifact = cast(Mapping[str, Any], payload["records"])
    records_path = manifest_path.parent / str(artifact["path"])
    if file_sha256(records_path) != str(artifact["sha256"]):
        raise ValueError("plik logprobów referencji rozjechał się z manifestem")
    rows = list(read_records(records_path))
    if len(rows) != int(artifact["record_count"]):
        raise ValueError("liczba logprobów referencji rozjechała się z manifestem")
    return {
        str(row["preference_id"]): (float(row["chosen_logprob"]), float(row["rejected_logprob"]))
        for row in rows
    }


def save_adapter_atomically(model: Any, tokenizer: Any, destination: Path) -> Path:
    """Zapisz adapter przez staging i `os.replace`, żeby nie zostawić połowy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        model.save_pretrained(staging)
        tokenizer.save_pretrained(staging)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


__all__ = [
    "PRECOMPUTE_CONTRACT",
    "RUN_CONTRACT",
    "SequenceScore",
    "load_preference_records",
    "load_reference_logprobs",
    "precompute_reference_logprobs",
    "run_dpo_steps",
    "save_adapter_atomically",
    "sequence_logprob",
    "sigmoid_dpo_loss",
    "torch_dpo_loss",
]
