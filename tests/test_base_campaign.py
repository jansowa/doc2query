from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from doc2query.config import load_config

CAMPAIGN_CONFIGS = {
    "b01_1_5b_10k_l768_lr2e4_s42.yaml": {
        "max_length": 768,
        "rank": 8,
        "effective_batch": 16,
    },
    "b02_1_5b_10k_l1024_lr2e4_s42.yaml": {
        "max_length": 1024,
        "rank": 8,
        "effective_batch": 16,
    },
    "b03_1_5b_10k_r16_lr2e4_s42.yaml": {
        "max_length": 512,
        "rank": 16,
        "effective_batch": 16,
    },
    "b04_1_5b_10k_r32_lr2e4_s42.yaml": {
        "max_length": 512,
        "rank": 32,
        "effective_batch": 16,
    },
    "b05_1_5b_10k_attention_lr2e4_s42.yaml": {
        "max_length": 512,
        "rank": 8,
        "effective_batch": 16,
    },
    "b06_1_5b_10k_eb32_lr2e4_s42.yaml": {
        "max_length": 512,
        "rank": 8,
        "effective_batch": 32,
    },
    "b07_1_5b_10k_dropout0_lr2e4_s42.yaml": {
        "max_length": 512,
        "rank": 8,
        "effective_batch": 16,
    },
}


def test_base_campaign_configs_are_pinned_single_factor_runs() -> None:
    root = Path("configs/experiments")
    for name, expected in CAMPAIGN_CONFIGS.items():
        config = load_config(root / name)
        assert config.model.name_or_path == "speakleash/Bielik-1.5B-v3"
        assert config.model.revision == "4b25049621bf3952a1fc9314c89773102eda0333"
        assert config.data.max_train_examples == 10_000
        assert config.training.learning_rate == 2e-4
        assert config.run.seed == 42
        assert config.training.max_length == expected["max_length"]
        assert config.lora.r == expected["rank"]
        assert config.lora.alpha == 2 * config.lora.r
        effective_batch = (
            config.training.per_device_train_batch_size
            * config.training.gradient_accumulation_steps
        )
        assert effective_batch == expected["effective_batch"]
    attention = load_config(root / "b05_1_5b_10k_attention_lr2e4_s42.yaml")
    assert attention.lora.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    dropout = load_config(root / "b07_1_5b_10k_dropout0_lr2e4_s42.yaml")
    assert dropout.lora.dropout == 0.0


def test_base_campaign_help_dry_run_and_order() -> None:
    for argument in ("--help", "--dry-run"):
        result = subprocess.run(
            ["bash", "scripts/run_base_1_5b_campaign.sh", argument],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    source = Path("scripts/run_base_1_5b_campaign.sh").read_text(encoding="utf-8")
    p03 = source.index("record_step p03-w05-sensitivity")
    memory = source.index("record_step memory-probe-768-1024")
    first_training = source.index('for config in "${configs[@]}"')
    assert p03 < memory < first_training


def test_memory_probe_reuses_completed_measurement(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    run_dir = root / "S01-1.5B-8GB" / "length-768"
    (run_dir / "adapter").mkdir(parents=True)
    (run_dir / "sft_summary.json").write_text(
        json.dumps(
            {
                "peak_vram_allocated_bytes": 10,
                "peak_vram_reserved_bytes": 20,
                "throughput_examples_per_second": 3.0,
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_memory_probe.py",
            "--config",
            "configs/experiments/s01_1_5b_8gb_smoke.yaml",
            "--lengths",
            "768",
            "--output-dir",
            str(root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((root / "S01-1.5B-8GB" / "memory_probe.json").read_text(encoding="utf-8"))
    assert report["probes"][0]["status"] == "already_complete"
    assert report["probes"][0]["peak_vram_reserved_bytes"] == 20
