#!/usr/bin/env python3
"""Validate the prospective S07 contract without loading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from doc2query.config import load_config
from doc2query.utils.records import write_json


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = _mapping(yaml.safe_load(args.contract.read_text(encoding="utf-8")), "contract")
    selected = _mapping(_mapping(raw["model_selection"], "model_selection")["selected"], "selected")
    alignment = _mapping(raw["w05_alignment"], "w05_alignment")
    access = _mapping(raw["data_access"], "data_access")
    gates = _mapping(raw["gates"], "gates")
    config = load_config(Path(str(_mapping(raw["training"], "training")["config"])))
    reference = json.loads(Path(str(alignment["reference_summary"])).read_text(encoding="utf-8"))
    checks = {
        "contract_is_prospective": raw.get("status") == "prospective",
        "selected_model_matches_config": (
            config.model.name_or_path == selected["name_or_path"]
            and config.model.revision == selected["revision"]
            and config.model.architecture == "seq2seq_lm"
        ),
        "full_finetuning": config.training.finetuning == "full",
        "same_paths": (
            str(config.data.input_path) == alignment["train_path"]
            and str(config.data.eval_path) == alignment["eval_path"]
        ),
        "same_pair_caps": (
            config.data.max_train_examples == alignment["train_pairs"]
            and config.data.max_eval_examples == alignment["eval_pairs"]
        ),
        "same_seed": config.run.seed == alignment["seed"] == 42,
        "same_epoch_and_effective_batch": (
            config.training.num_train_epochs == alignment["epochs"]
            and config.training.per_device_train_batch_size
            * config.training.gradient_accumulation_steps
            == alignment["effective_batch_size"]
        ),
        "same_token_budgets": (
            config.training.max_length == alignment["source_max_tokens"]
            and config.training.max_completion_tokens == alignment["target_max_tokens"]
        ),
        "w05_fingerprint_pinned": reference["dataset_fingerprint"]
        == alignment["dataset_fingerprint"],
        "w05_pair_counts_match": (
            reference["train_examples"] == alignment["train_pairs"]
            and reference["eval_examples"] == alignment["eval_pairs"]
        ),
        "w05_optimizer_budget_matches": reference["global_step"] == alignment["optimizer_steps"],
        "development_only": access.get("development_only") is True
        and access.get("final_tests_used") == [],
        "closed_scopes": all(
            gates.get(name) is True for name in ("no_dev_confirm", "no_final_tests", "no_p06")
        ),
        "input_artifacts_present": all(
            Path(str(alignment[name])).is_file() for name in ("train_path", "eval_path")
        ),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    cuda = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    report = {
        "schema_version": 1,
        "contract_version": raw["contract_version"],
        "status": "pass" if not failures else "failed",
        "checks": checks,
        "failures": failures,
        "cuda": cuda,
        "cheap_preflight_pass": not failures,
        "full_train_authorized": False,
        "full_train_pending_gates": ["unit_tests", "tiny_smoke", "memory_probe"],
        "final_tests_used": [],
        "dev_confirm_opened": False,
        "p06_opened": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
