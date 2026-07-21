from __future__ import annotations

import copy
import fcntl
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from doc2query.config import load_config
from doc2query.evaluation.campaign_audit import (
    BASE_ARM_CONFIGS,
    INSTRUCT_ARM_CONFIGS,
    audit_campaign,
    write_campaign_audit,
)


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "campaign"
    shutil.copytree(Path("configs"), root / "configs")
    decision = Path("docs/decisions/task03_instruct_campaign_early_stop_2026-07-21.md")
    target = root / decision
    target.parent.mkdir(parents=True)
    shutil.copy2(decision, target)
    return root


def _identity(config: dict[str, Any], dataset_fingerprint: str) -> dict[str, Any]:
    training = copy.deepcopy(config["training"])
    training.pop("resume_if_available", None)
    for field in (
        "logging_steps",
        "eval_steps",
        "save_steps",
        "save_total_limit",
        "dataloader_num_workers",
        "early_stopping_patience",
    ):
        training.pop(field, None)
    payload = {
        "schema_version": 1,
        "experiment_id": config["run"]["experiment_id"],
        "seed": config["run"]["seed"],
        "dataset_fingerprint": dataset_fingerprint,
        "model": config["model"],
        "quantization": config["quantization"],
        "lora": config["lora"],
        "training": training,
    }
    import hashlib

    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**payload, "signature": hashlib.sha256(canonical.encode()).hexdigest()}


def _complete_arm(root: Path, config_name: str) -> tuple[str, Path]:
    config = load_config(root / "configs/experiments" / config_name)
    payload = config.model_dump(mode="json")
    run_dir = root / config.run.output_dir
    adapter = run_dir / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    fingerprint = "fixture-dataset-fingerprint"
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": config.run.experiment_id,
                "seed": config.run.seed,
                "dataset_fingerprint": fingerprint,
                "config": payload,
                "artifacts": {
                    "adapter": str(config.run.output_dir / "adapter"),
                    "summary": str(config.run.output_dir / "sft_summary.json"),
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "sft_summary.json").write_text(
        json.dumps(
            {
                "experiment_id": config.run.experiment_id,
                "dataset_fingerprint": fingerprint,
                "train_examples": config.data.max_train_examples,
                "global_step": 625,
                "adapter_path": str(config.run.output_dir / "adapter"),
                "loss": {"last_eval_loss": 999.0},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "resume_identity.json").write_text(
        json.dumps(_identity(payload, fingerprint)), encoding="utf-8"
    )
    return f"train-{Path(config_name).stem}", run_dir


def _write_status(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["started_at\tfinished_at\tname\texit_code"]
    lines.extend("\t".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_campaign_retry_success_supersedes_old_failure_and_loss_is_not_selected(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    rows: list[tuple[str, str, str, str]] = []
    required = (*BASE_ARM_CONFIGS, INSTRUCT_ARM_CONFIGS[0], INSTRUCT_ARM_CONFIGS[2])
    for index, name in enumerate(required):
        step, _ = _complete_arm(root, name)
        if index == 0:
            rows.append(("2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z", step, "1"))
        rows.append(("2026-01-02T00:00:00Z", "2026-01-02T00:01:00Z", step, "0"))
    status = root / "status.tsv"
    _write_status(status, rows)

    report = audit_campaign(root, status_path=status)

    assert report["complete"] is True
    assert report["arms"][0]["attempt_count"] == 2
    assert report["arms"][0]["state"] == "completed"
    assert report["state_counts"]["deferred"] == 3
    assert report["selection_performed"] is False
    assert "loss" not in report["arms"][0]["contract"]


def test_generation_schema_additions_do_not_invalidate_research_contract(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    step, run_dir = _complete_arm(root, BASE_ARM_CONFIGS[0])
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["generation"].pop("controlled")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    status = root / "status.tsv"
    _write_status(status, [("2026-01-01", "2026-01-02", step, "0")])

    report = audit_campaign(root, status_path=status)

    assert report["arms"][0]["state"] == "completed"


def test_missing_running_failed_and_config_drift_are_fail_closed(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    first_step, first_dir = _complete_arm(root, BASE_ARM_CONFIGS[0])
    manifest_path = first_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["training"]["learning_rate"] = 0.123
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    second_step = f"train-{Path(BASE_ARM_CONFIGS[1]).stem}"
    third_step = f"train-{Path(BASE_ARM_CONFIGS[2]).stem}"
    status = root / "status.tsv"
    _write_status(
        status,
        [
            ("2026-01-01", "2026-01-02", first_step, "0"),
            ("2026-01-02", "", second_step, ""),
            ("2026-01-02", "2026-01-03", third_step, "17"),
        ],
    )

    report = audit_campaign(root, status_path=status)

    assert report["complete"] is False
    states = {arm["arm_id"]: arm["state"] for arm in report["arms"]}
    assert states["B01"] == "invalid"
    assert states["B02"] == "running"
    assert states["B03"] == "failed-retriable"
    assert states["B04"] == "missing"
    assert any("learning_rate" in error for error in report["arms"][0]["errors"])


def test_running_arm_is_detected_from_read_only_lock_and_unclosed_log(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    status = root / "reports/base_1_5b_campaign/status.tsv"
    _write_status(status, [])
    lock = root / "reports/base_1_5b_campaign/queue.lock"
    lock.touch()
    log = root / "logs/base_1_5b_campaign.log"
    log.parent.mkdir(parents=True)
    step = f"train-{Path(BASE_ARM_CONFIGS[0]).stem}"
    log.write_text(f"[2026-01-01] START {step}\n", encoding="utf-8")
    with lock.open("r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = audit_campaign(root, status_path=status)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    assert report["queue_active"] is True
    assert report["active_step"] == step
    assert report["arms"][0]["state"] == "running"


def test_auditor_writes_nothing_without_explicit_output_and_check_complete_fails(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    status = root / "status.tsv"
    _write_status(status, [])
    report = audit_campaign(root, status_path=status)
    write_campaign_audit(report)
    assert sorted(path.name for path in root.iterdir()) == ["configs", "docs", "status.tsv"]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_base_1_5b_campaign.py",
            "--root",
            str(root),
            "--status",
            str(status),
            "--check-complete",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_reports_are_deterministic_and_only_use_requested_paths(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    status = root / "status.tsv"
    _write_status(status, [])
    report = audit_campaign(root, status_path=status)
    json_a = tmp_path / "a" / "audit.json"
    json_b = tmp_path / "b" / "audit.json"
    markdown = tmp_path / "audit.md"
    write_campaign_audit(report, json_path=json_a, markdown_path=markdown)
    write_campaign_audit(report, json_path=json_b)
    assert json_a.read_bytes() == json_b.read_bytes()
    assert "winner" not in markdown.read_text(encoding="utf-8").lower()
