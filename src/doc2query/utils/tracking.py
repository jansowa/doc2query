"""Tracker-independent local run manifests."""

import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from doc2query.utils.hardware import collect_hardware_report


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _serialize_config(config: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, BaseModel):
        return config.model_dump(mode="json")
    return dict(config)


def write_run_manifest(
    run_dir: Path,
    *,
    experiment_id: str,
    seed: int,
    config: BaseModel | Mapping[str, Any],
    dataset_fingerprint: str | None = None,
    artifacts: Mapping[str, str] | None = None,
) -> Path:
    """Always persist a local manifest, independent of online tracker state."""
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    status = _git_value("status", "--porcelain")
    packages: dict[str, str | None] = {}
    for package in ("torch", "transformers", "trl", "peft", "bitsandbytes"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": experiment_id,
        "seed": seed,
        "dataset_fingerprint": dataset_fingerprint,
        "git": {
            "commit": _git_value("rev-parse", "HEAD"),
            "dirty": bool(status) if status is not None else None,
        },
        "runtime": {
            "python": platform.python_version(),
            "packages": packages,
            "hardware": collect_hardware_report(),
        },
        "config": _serialize_config(config),
        "artifacts": dict(artifacts or {}),
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest_path
