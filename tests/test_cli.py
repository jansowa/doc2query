from typer.testing import CliRunner

from doc2query.cli import app

runner = CliRunner()


def test_root_help_lists_public_groups() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "config", "data", "train", "generate", "preferences", "evaluate"):
        assert command in result.stdout


def test_nested_help_does_not_load_a_model() -> None:
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "sft" in result.stdout
    assert "reranker" in result.stdout
    assert "dpo" in result.stdout
    assert "grpo" in result.stdout


def test_config_validate_cli() -> None:
    result = runner.invoke(app, ["config", "validate", "--config", "configs/base.yaml"])
    assert result.exit_code == 0
    assert "Configuration valid" in result.stdout


def test_pending_command_validates_then_explains_scope() -> None:
    result = runner.invoke(app, ["train", "sft", "--config", "configs/base.yaml"])
    assert result.exit_code == 3
    assert "not implemented in task 00" in result.stdout


def test_reranker_training_stub_is_permanently_disabled() -> None:
    result = runner.invoke(app, ["train", "reranker", "--config", "configs/base.yaml"])
    assert result.exit_code == 3
    assert "Reranker training is disabled by AGENTS.md" in result.stdout
    assert "benchmarking and calibration only" in result.stdout
