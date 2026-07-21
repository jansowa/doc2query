"""Read-only completion audit for the 1.5B base/instruct campaign."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from doc2query.config import load_config
from doc2query.schemas import AppConfig

ArmState = Literal["completed", "deferred", "running", "failed-retriable", "missing", "invalid"]

CAMPAIGN_COMPLETION_CONTRACT = Path("configs/evaluation/task03_campaign_completion_v1.json")

BASE_ARM_CONFIGS = (
    "b01_1_5b_10k_l768_lr2e4_s42.yaml",
    "b02_1_5b_10k_l1024_lr2e4_s42.yaml",
    "b03_1_5b_10k_r16_lr2e4_s42.yaml",
    "b04_1_5b_10k_r32_lr2e4_s42.yaml",
    "b05_1_5b_10k_attention_lr2e4_s42.yaml",
    "b06_1_5b_10k_eb32_lr2e4_s42.yaml",
    "b07_1_5b_10k_dropout0_lr2e4_s42.yaml",
)
INSTRUCT_ARM_CONFIGS = (
    "i01_1_5b_instruct_10k_lr1e4_s42.yaml",
    "i02_1_5b_instruct_10k_lr5e5_s42.yaml",
    "i03_1_5b_instruct_10k_lr2e4_s42.yaml",
    "i04_1_5b_instruct_10k_lr1e4_s43.yaml",
    "i05_1_5b_instruct_50k_lr1e4_s42.yaml",
)
MATCHED_BASE_CONFIGS = (
    "w01_1_5b_10k_lr1e4_seed42.yaml",
    "w02_1_5b_10k_lr5e5_seed42.yaml",
    "w03_1_5b_10k_lr2e4_seed42.yaml",
    "w04_1_5b_10k_lr1e4_seed43.yaml",
    "w05_1_5b_50k_8gb.yaml",
)
_LOG_START = re.compile(r"\] START (?P<name>.+)$")
_LOG_END = re.compile(r"\] END (?P<name>.+) rc=-?\d+$")


@dataclass(frozen=True)
class StatusAttempt:
    """One append-only status record; later records supersede earlier attempts."""

    line: int
    started_at: str
    finished_at: str
    name: str
    exit_code: int | None


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _research_config_fingerprint(value: Mapping[str, Any]) -> str:
    """Fingerprint fields that define the run, excluding the discovered data hash."""
    return _canonical_fingerprint(
        {key: item for key, item in value.items() if key != "dataset_fingerprint"}
    )


def parse_status_tsv(path: Path) -> tuple[list[StatusAttempt], list[str]]:
    """Parse completed, running and retry records without mutating the status file."""
    if not path.is_file():
        return [], [f"missing status file: {path}"]
    attempts: list[StatusAttempt] = []
    errors: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["started_at", "finished_at", "name", "exit_code"]:
            return [], ["status.tsv header must be started_at, finished_at, name, exit_code"]
        for line, row in enumerate(reader, start=2):
            name = (row.get("name") or "").strip()
            started = (row.get("started_at") or "").strip()
            finished = (row.get("finished_at") or "").strip()
            raw_code = (row.get("exit_code") or "").strip()
            if not name or not started:
                errors.append(f"line {line}: status record requires started_at and name")
                continue
            code: int | None = None
            if raw_code:
                try:
                    code = int(raw_code)
                except ValueError:
                    errors.append(f"line {line}: invalid exit_code {raw_code!r}")
                    continue
            if bool(finished) != (code is not None):
                errors.append(
                    f"line {line}: finished_at and exit_code must both be present or absent"
                )
                continue
            attempts.append(StatusAttempt(line, started, finished, name, code))
    return attempts, errors


def _lock_is_held(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return False


def _active_log_step(path: Path) -> str | None:
    if not path.is_file():
        return None
    open_steps: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        start = _LOG_START.search(line)
        if start:
            open_steps.append(start.group("name"))
            continue
        end = _LOG_END.search(line)
        if end:
            name = end.group("name")
            for index in range(len(open_steps) - 1, -1, -1):
                if open_steps[index] == name:
                    open_steps.pop(index)
                    break
    return open_steps[-1] if open_steps else None


def _json_mapping(path: Path, errors: list[str], label: str) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {label}: {path}: {exc}")
        return None
    if not isinstance(raw, dict):
        errors.append(f"invalid {label}: expected JSON object: {path}")
        return None
    return raw


def _adapter_complete(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "adapter_config.json").is_file()
        and (
            (path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()
        )
    )


def _expected_contract(config: AppConfig) -> dict[str, Any]:
    effective_batch = (
        config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps
    )
    contract = {
        "experiment_id": config.run.experiment_id,
        "dataset_fingerprint": config.data.fingerprint,
        "model_name": config.model.name_or_path,
        "model_revision": config.model.revision,
        "seed": config.run.seed,
        "pair_count": config.data.max_train_examples,
        "max_length": config.training.max_length,
        "learning_rate": config.training.learning_rate,
        "lora_rank": config.lora.r,
        "lora_modules": config.lora.target_modules,
        "lora_dropout": config.lora.dropout,
        "effective_batch": effective_batch,
        "max_steps": config.training.max_steps,
    }
    return {**contract, "research_config_fingerprint": _research_config_fingerprint(contract)}


def _artifact_contract(config: AppConfig, run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    manifest = _json_mapping(run_dir / "run_manifest.json", errors, "run_manifest")
    summary = _json_mapping(run_dir / "sft_summary.json", errors, "sft_summary")
    identity = _json_mapping(run_dir / "resume_identity.json", errors, "resume_identity")
    if not _adapter_complete(run_dir / "adapter"):
        errors.append(f"missing or incomplete adapter: {run_dir / 'adapter'}")

    expected = _expected_contract(config)
    observed: dict[str, Any] = {}
    if manifest is None or summary is None or identity is None:
        return observed, errors
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, Mapping):
        errors.append("run_manifest.config must be an object")
        return observed, errors
    observed_contract = {
        "experiment_id": manifest.get("experiment_id"),
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "model_name": _nested(manifest_config, "model", "name_or_path"),
        "model_revision": _nested(manifest_config, "model", "revision"),
        "seed": manifest.get("seed"),
        "pair_count": _nested(manifest_config, "data", "max_train_examples"),
        "max_length": _nested(manifest_config, "training", "max_length"),
        "learning_rate": _nested(manifest_config, "training", "learning_rate"),
        "lora_rank": _nested(manifest_config, "lora", "r"),
        "lora_modules": _nested(manifest_config, "lora", "target_modules"),
        "lora_dropout": _nested(manifest_config, "lora", "dropout"),
        "effective_batch": _effective_batch(manifest_config),
        "max_steps": _nested(manifest_config, "training", "max_steps"),
    }
    observed = {
        **observed_contract,
        "research_config_fingerprint": _research_config_fingerprint(observed_contract),
    }
    for field, expected_value in expected.items():
        if field == "dataset_fingerprint" and expected_value is None:
            continue
        if observed.get(field) != expected_value:
            errors.append(
                f"contract drift for {field}: expected {expected_value!r}, "
                f"observed {observed.get(field)!r}"
            )
    manifest_fingerprint = manifest.get("dataset_fingerprint")
    if not isinstance(manifest_fingerprint, str) or not manifest_fingerprint.strip():
        errors.append("run_manifest dataset_fingerprint is missing")
    for source_name, source in (("sft_summary", summary), ("resume_identity", identity)):
        if source.get("experiment_id") != expected["experiment_id"]:
            errors.append(f"{source_name} experiment_id does not match config")
        if source.get("dataset_fingerprint") != manifest_fingerprint:
            errors.append(f"{source_name} dataset_fingerprint does not match run_manifest")
    if summary.get("train_examples") != expected["pair_count"]:
        errors.append("sft_summary train_examples does not match configured pair count")
    global_step = summary.get("global_step")
    if not isinstance(global_step, int) or global_step < 1:
        errors.append("sft_summary global_step does not prove completed training")
    elif expected["max_steps"] > 0 and global_step != expected["max_steps"]:
        errors.append("sft_summary global_step does not match configured max_steps")
    expected_adapter_values = {
        str(run_dir / "adapter"),
        str(config.run.output_dir / "adapter"),
    }
    if summary.get("adapter_path") not in expected_adapter_values:
        errors.append("sft_summary adapter_path does not match run directory")
    manifest_artifacts = manifest.get("artifacts")
    if isinstance(manifest_artifacts, Mapping):
        if manifest_artifacts.get("adapter") not in expected_adapter_values:
            errors.append("run_manifest adapter artifact does not match run directory")
        expected_summary_values = {
            str(run_dir / "sft_summary.json"),
            str(config.run.output_dir / "sft_summary.json"),
        }
        if manifest_artifacts.get("summary") not in expected_summary_values:
            errors.append("run_manifest summary artifact does not match run directory")
    else:
        errors.append("run_manifest artifacts mapping is missing")
    signature = identity.get("signature")
    identity_without_signature = {
        key: value for key, value in identity.items() if key != "signature"
    }
    if signature != _canonical_fingerprint(identity_without_signature):
        errors.append("resume_identity signature is invalid")
    return observed, errors


def _nested(value: Mapping[str, Any], first: str, second: str) -> Any:
    child = value.get(first)
    return child.get(second) if isinstance(child, Mapping) else None


def _effective_batch(config: Mapping[str, Any]) -> int | None:
    micro = _nested(config, "training", "per_device_train_batch_size")
    accumulation = _nested(config, "training", "gradient_accumulation_steps")
    if isinstance(micro, int) and isinstance(accumulation, int):
        return micro * accumulation
    return None


def _trajectory(config: AppConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json")
    payload.pop("run", None)
    return payload


def _remove_path(payload: dict[str, Any], path: tuple[str, ...]) -> None:
    target: dict[str, Any] = payload
    for part in path[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            return
        target = child
    target.pop(path[-1], None)


def _single_factor_errors(configs: Sequence[AppConfig], reference: AppConfig) -> list[str]:
    allowed = (
        ("training", "max_length"),
        ("training", "max_length"),
        ("lora", "r"),
        ("lora", "r"),
        ("lora", "target_modules"),
        ("training", "gradient_accumulation_steps"),
        ("lora", "dropout"),
    )
    errors: list[str] = []
    for index, (config, varying) in enumerate(zip(configs, allowed, strict=True), start=1):
        actual = _trajectory(config)
        expected = _trajectory(reference)
        _remove_path(actual, varying)
        _remove_path(expected, varying)
        if varying == ("lora", "r"):
            _remove_path(actual, ("lora", "alpha"))
            _remove_path(expected, ("lora", "alpha"))
        if varying == ("lora", "target_modules"):
            for dependent in (
                ("lora", "minimum_target_modules"),
                ("lora", "expected_layer_patterns"),
            ):
                _remove_path(actual, dependent)
                _remove_path(expected, dependent)
        if actual != expected:
            errors.append(f"B{index:02d} is not a single-factor ablation of W03")
    return errors


def _matched_instruct_errors(
    instruct: Sequence[AppConfig], matched_base: Sequence[AppConfig]
) -> list[str]:
    errors: list[str] = []
    for index, (candidate, base) in enumerate(zip(instruct, matched_base, strict=True), start=1):
        left = _trajectory(candidate)
        right = _trajectory(base)
        left.pop("model", None)
        right.pop("model", None)
        if left != right:
            errors.append(f"I{index:02d} is not budget/config matched to its base counterpart")
    return errors


def audit_campaign(root: Path, *, status_path: Path | None = None) -> dict[str, Any]:
    """Audit expected arms and artifacts without selecting or ranking any model."""
    root = root.resolve()
    status = status_path or root / "reports/base_1_5b_campaign/status.tsv"
    attempts, status_errors = parse_status_tsv(status)
    queue_active = _lock_is_held(root / "reports/base_1_5b_campaign/queue.lock")
    active_step = _active_log_step(root / "logs/base_1_5b_campaign.log") if queue_active else None
    by_name: dict[str, list[StatusAttempt]] = {}
    for attempt in attempts:
        by_name.setdefault(attempt.name, []).append(attempt)

    config_dir = root / "configs/experiments"
    configs = [load_config(config_dir / name) for name in BASE_ARM_CONFIGS]
    instruct = [load_config(config_dir / name) for name in INSTRUCT_ARM_CONFIGS]
    matched = [load_config(config_dir / name) for name in MATCHED_BASE_CONFIGS]
    completion_contract_path = root / CAMPAIGN_COMPLETION_CONTRACT
    completion_errors: list[str] = []
    completion_contract = _json_mapping(
        completion_contract_path, completion_errors, "campaign completion contract"
    )
    required_arms: set[str] = set()
    deferred_arms: set[str] = set()
    decision_provenance: dict[str, Any] = {}
    if completion_contract is not None:
        if completion_contract.get("schema_version") != 1:
            completion_errors.append("campaign completion contract schema_version must be 1")
        if completion_contract.get("contract_id") != "task03-campaign-completion-v1":
            completion_errors.append("unexpected campaign completion contract_id")
        if completion_contract.get("final_tests_used") != []:
            completion_errors.append(
                "campaign completion contract must declare final_tests_used=[]"
            )
        required_raw = completion_contract.get("required_arms")
        deferred_raw = completion_contract.get("deferred_arms")
        if not isinstance(required_raw, list) or not all(
            isinstance(item, str) for item in required_raw
        ):
            completion_errors.append("campaign completion contract requires string required_arms")
        else:
            required_arms = set(required_raw)
        if not isinstance(deferred_raw, list) or not all(
            isinstance(item, str) for item in deferred_raw
        ):
            completion_errors.append("campaign completion contract requires string deferred_arms")
        else:
            deferred_arms = set(deferred_raw)
        expected_arms = {f"B{index:02d}" for index in range(1, 8)} | {
            f"I{index:02d}" for index in range(1, 6)
        }
        if required_arms & deferred_arms or required_arms | deferred_arms != expected_arms:
            completion_errors.append("required/deferred arms must partition all 12 campaign arms")
        decision_path_raw = completion_contract.get("decision_path")
        decision_sha = completion_contract.get("decision_sha256")
        if not isinstance(decision_path_raw, str) or not isinstance(decision_sha, str):
            completion_errors.append(
                "campaign completion contract requires decision path and SHA-256"
            )
        else:
            decision_path = root / decision_path_raw
            observed_sha = (
                hashlib.sha256(decision_path.read_bytes()).hexdigest()
                if decision_path.is_file()
                else None
            )
            if observed_sha != decision_sha:
                completion_errors.append("campaign early-stop decision SHA-256 drift")
            decision_provenance = {
                "path": decision_path_raw,
                "sha256": observed_sha,
                "expected_sha256": decision_sha,
            }
    global_errors = [
        *status_errors,
        *completion_errors,
        *_single_factor_errors(configs, load_config(config_dir / "w03_1_5b_10k_lr2e4_seed42.yaml")),
        *_matched_instruct_errors(instruct, matched),
    ]

    arms: list[dict[str, Any]] = []
    for config_name, config in zip(
        (*BASE_ARM_CONFIGS, *INSTRUCT_ARM_CONFIGS), (*configs, *instruct), strict=True
    ):
        step_name = f"train-{Path(config_name).stem}"
        arm_id = config.run.experiment_id.split("-", 1)[0]
        history = by_name.get(step_name, [])
        latest = history[-1] if history else None
        state: ArmState
        errors: list[str] = []
        observed: dict[str, Any] = {}
        if arm_id in deferred_arms:
            if any(attempt.exit_code == 0 for attempt in history):
                state = "invalid"
                errors.append("arm completed despite pinned early-stop deferral")
            else:
                state = "deferred"
        elif active_step == step_name and (latest is None or latest.exit_code is not None):
            state = "running"
        elif latest is None:
            state = "missing"
        elif latest.exit_code is None:
            state = "running"
        elif latest.exit_code != 0:
            state = "failed-retriable"
        else:
            observed, errors = _artifact_contract(config, root / config.run.output_dir)
            state = "invalid" if errors else "completed"
        arms.append(
            {
                "arm_id": arm_id,
                "experiment_id": config.run.experiment_id,
                "config_path": str(Path("configs/experiments") / config_name),
                "run_dir": str(config.run.output_dir),
                "step_name": step_name,
                "state": state,
                "attempt_count": len(history),
                "latest_exit_code": latest.exit_code if latest else None,
                "latest_status_line": latest.line if latest else None,
                "contract": observed or _expected_contract(config),
                "errors": errors,
            }
        )
    states_by_arm = {str(arm["arm_id"]): arm["state"] for arm in arms}
    complete = (
        not queue_active
        and not global_errors
        and all(states_by_arm.get(arm_id) == "completed" for arm_id in required_arms)
        and all(states_by_arm.get(arm_id) == "deferred" for arm_id in deferred_arms)
    )
    return {
        "schema_version": 1,
        "campaign_id": "base-instruct-1.5b-technical-v1",
        "selection_performed": False,
        "selection_metric": None,
        "complete": complete,
        "completion_contract": {
            "path": str(CAMPAIGN_COMPLETION_CONTRACT),
            "sha256": (
                hashlib.sha256(completion_contract_path.read_bytes()).hexdigest()
                if completion_contract_path.is_file()
                else None
            ),
            "required_arms": sorted(required_arms),
            "deferred_arms": sorted(deferred_arms),
            "decision": decision_provenance,
        },
        "status_path": str(status),
        "expected_arm_count": 12,
        "queue_active": queue_active,
        "active_step": active_step,
        "state_counts": {
            state: sum(arm["state"] == state for arm in arms)
            for state in (
                "completed",
                "deferred",
                "running",
                "failed-retriable",
                "missing",
                "invalid",
            )
        },
        "global_errors": global_errors,
        "arms": arms,
    }


def campaign_audit_markdown(report: Mapping[str, Any]) -> str:
    """Render a deterministic, loss-free completion report."""
    lines = [
        "# Base/instruct 1.5B campaign completion audit",
        "",
        f"Complete: **{str(bool(report['complete'])).lower()}**",
        "",
        "This audit checks completion and contract integrity only. It does not rank models,",
        "and evaluation loss is deliberately absent from the decision surface.",
        "",
        "| arm | state | attempts | artifact/contract errors |",
        "|---|---|---:|---|",
    ]
    arms = report.get("arms", [])
    if isinstance(arms, list):
        for arm in arms:
            if not isinstance(arm, Mapping):
                continue
            errors = arm.get("errors")
            rendered_errors = (
                "; ".join(str(item) for item in errors) if isinstance(errors, list) else ""
            )
            lines.append(
                f"| {arm.get('arm_id')} | {arm.get('state')} | "
                f"{arm.get('attempt_count')} | {rendered_errors or '—'} |"
            )
    global_errors = report.get("global_errors")
    if isinstance(global_errors, list) and global_errors:
        lines.extend(["", "## Global blockers", ""])
        lines.extend(f"- {error}" for error in global_errors)
    return "\n".join(lines) + "\n"


def write_campaign_audit(
    report: Mapping[str, Any], *, json_path: Path | None = None, markdown_path: Path | None = None
) -> None:
    """Write only explicitly requested report paths."""
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(campaign_audit_markdown(report), encoding="utf-8")
