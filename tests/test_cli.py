import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from doc2query.cli import app
from doc2query.utils.records import JsonlWriter

# Rich zawija pomoc do szerokości terminala; w CI to 80 kolumn, więc długie flagi
# (np. --resume-if-available) łamią się w środku i asercje podłańcuchowe padają.
runner = CliRunner(env={"COLUMNS": "200", "TERM": "dumb"})


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


def test_generator_script_help_exposes_runtime_optimization_controls() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_generator.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for option in (
        "--generation-batch-size",
        "--primary-judge-device",
        "--shadow-judge-device",
        "--archive-incompatible-scoring",
    ):
        assert option in result.stdout
    benchmark = subprocess.run(
        [sys.executable, "scripts/benchmark_evaluation_runtime.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--component" in benchmark.stdout
    assert "--bm25-workers" in benchmark.stdout
    probe = subprocess.run(
        [sys.executable, "scripts/train_probe_embedder.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--retrieval-query-batch-size" in probe.stdout
    assert "--retrieval-device" in probe.stdout
    probe_benchmark = subprocess.run(
        [sys.executable, "scripts/benchmark_probe_retrieval.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--query-batch-sizes" in probe_benchmark.stdout
    artifact_benchmark = subprocess.run(
        [sys.executable, "scripts/benchmark_probe_artifacts.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--embedding-cache" in artifact_benchmark.stdout
    assert "--parity-queries" in artifact_benchmark.stdout


def test_generate_command_invokes_task05_pipeline(monkeypatch: object) -> None:
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    monkeypatch.setattr(
        "doc2query.cli.run_controlled_generation",
        lambda _config, **_kwargs: {"experiment_id": "task05-smoke", "generated": 4},
    )
    result = runner.invoke(app, ["generate", "--config", "configs/base.yaml"])
    assert result.exit_code == 0
    assert '"generated": 4' in result.stdout


def test_embedder_help_exposes_native_holdout_profiles_without_loading_model() -> None:
    result = runner.invoke(app, ["evaluate", "embedder", "--help"])
    assert result.exit_code == 0
    assert "--holdout-manifest" in result.stdout
    assert "--native-corpus" in result.stdout
    assert "--holdout-profile" in result.stdout
    assert "--primary-judge-conf" in result.stdout
    assert "--bm25-index" in result.stdout
    assert "--generator-id" in result.stdout


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
