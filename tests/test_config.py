from pathlib import Path

import pytest
from pydantic import ValidationError

from doc2query.config import load_config


def test_base_config_is_valid() -> None:
    config = load_config(Path("configs/base.yaml"))
    assert config.run.experiment_id == "bootstrap-smoke"
    assert config.training.max_length == 1024


def test_hydra_experiment_composes_hierarchical_groups() -> None:
    config = load_config(Path("configs/experiments/e00_prompting.yaml"))
    assert config.run.experiment_id == "E00"
    assert config.data.input_format == "jsonl"
    assert config.training.gradient_accumulation_steps == 16


def test_invalid_precision_fails_before_run(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
run: {experiment_id: test, seed: 42, output_dir: runs/test}
data: {input_path: data/test.jsonl, input_format: jsonl}
model: {name_or_path: tiny, revision: main}
training: {bf16: true, fp16: true}
generation: {}
tracking: {backend: offline, online: false}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="cannot both be enabled"):
        load_config(path)


def test_unknown_config_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
run: {experiment_id: test, seed: 42, output_dir: runs/test, typo: true}
data: {input_path: data/test.jsonl, input_format: jsonl}
model: {name_or_path: tiny}
training: {}
generation: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config(path)
