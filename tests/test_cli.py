from pathlib import Path

from typer.testing import CliRunner

from doc2query.cli import app
from doc2query.utils.records import JsonlWriter

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


def test_sft_command_invokes_task03_pipeline(monkeypatch: object) -> None:
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    monkeypatch.setattr(
        "doc2query.cli.run_sft",
        lambda _config, **_kwargs: {"experiment_id": "bootstrap-smoke", "global_step": 1},
    )
    result = runner.invoke(
        app,
        [
            "train",
            "sft",
            "--config",
            "configs/base.yaml",
            "--resume-if-available",
        ],
    )
    assert result.exit_code == 0
    assert '"global_step": 1' in result.stdout


def test_sft_help_documents_automatic_resume() -> None:
    result = runner.invoke(app, ["train", "sft", "--help"])
    assert result.exit_code == 0
    assert "--resume-if-available" in result.stdout


def test_data_validate_cli_runs_task01_pipeline(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    with JsonlWriter(input_path) as writer:
        writer.write(
            {
                "example_id": "q-1",
                "query": "Jak działa pompa ciepła?",
                "positives": [
                    {
                        "doc_id": "p-1",
                        "text": "Pompa ciepła pobiera energię z otoczenia i ogrzewa budynek.",
                    }
                ],
                "hard_negatives": [
                    {
                        "doc_id": f"n-{index}",
                        "text": (
                            f"Negatywny dokument numer {index} opisuje inne urządzenie grzewcze."
                        ),
                    }
                    for index in range(10)
                ],
                "metadata": {"language": "pl"},
            }
        )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
run: {{experiment_id: data-test, seed: 42, output_dir: {tmp_path / "run"}}}
data: {{input_path: {input_path}, input_format: jsonl}}
model: {{name_or_path: tiny, revision: main}}
training: {{}}
generation: {{}}
tracking: {{backend: offline, online: false}}
""",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["data", "validate", "--config", str(config)])
    assert result.exit_code == 0
    assert (tmp_path / "run" / "data_validation" / "report.json").is_file()


def test_reranker_training_stub_is_permanently_disabled() -> None:
    result = runner.invoke(app, ["train", "reranker", "--config", "configs/base.yaml"])
    assert result.exit_code == 3
    assert "Reranker training is disabled by AGENTS.md" in result.stdout
    assert "benchmarking and calibration only" in result.stdout
