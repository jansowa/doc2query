"""Stable command-line interface for the doc2query research pipeline."""

import json
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError
from rich.console import Console

from doc2query.config import load_config
from doc2query.data.validate import ValidationPolicy, validate_dataset
from doc2query.utils.hardware import collect_hardware_report, write_hardware_report

app = typer.Typer(help="Bielik doc2query research toolkit.", no_args_is_help=True)
config_app = typer.Typer(help="Validate and inspect run configurations.", no_args_is_help=True)
data_app = typer.Typer(help="Data pipeline commands.", no_args_is_help=True)
train_app = typer.Typer(
    help="Generator training commands and the required disabled reranker stub.",
    no_args_is_help=True,
)
preferences_app = typer.Typer(help="Preference dataset commands.", no_args_is_help=True)
evaluate_app = typer.Typer(help="Evaluation commands.", no_args_is_help=True)

app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(preferences_app, name="preferences")
app.add_typer(evaluate_app, name="evaluate")

console = Console()
ConfigPath = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]
OutputPath = Annotated[Path | None, typer.Option("--output", help="Optional JSON report path.")]


def _validated(path: Path) -> None:
    try:
        load_config(path)
    except (ValueError, ValidationError) as exc:
        console.print(f"[red]Invalid configuration:[/red] {exc}", highlight=False)
        raise typer.Exit(code=2) from exc


def _pending(path: Path, module: str) -> NoReturn:
    _validated(path)
    console.print(
        f"Configuration valid. Module '{module}' is not implemented in task 00; "
        "run the corresponding later task."
    )
    raise typer.Exit(code=3)


@app.command()
def doctor(
    output: OutputPath = None,
) -> None:
    """Report hardware capabilities; absence of a GPU is supported."""
    report = collect_hardware_report()
    if output is not None:
        write_hardware_report(output, report)
    console.print_json(json.dumps(report))


@config_app.command("validate")
def validate_config(
    config: ConfigPath,
) -> None:
    """Resolve and validate a complete run configuration."""
    parsed = load_config(config)
    console.print(f"Configuration valid: {parsed.run.experiment_id}")


@data_app.command("validate")
def validate_data(
    config: ConfigPath,
) -> None:
    """Validate a local canonical dataset and write accepted/rejected audit artifacts."""
    parsed = load_config(config)
    if parsed.data.input_path is None:
        console.print("[red]Data validation requires a materialized local input_path.[/red]")
        raise typer.Exit(code=2)
    output_dir = parsed.run.output_dir / "data_validation"
    report = validate_dataset(
        parsed.data.input_path,
        accepted_path=output_dir / "accepted.jsonl",
        rejected_path=output_dir / "rejected.jsonl",
        report_path=output_dir / "report.json",
        policy=ValidationPolicy.defaults(),
    )
    console.print_json(json.dumps(report))
    if report["contains_error_policy_violations"]:
        raise typer.Exit(code=2)


@train_app.command("sft")
def train_sft(config: ConfigPath) -> None:
    """Train an SFT adapter (implemented by task 03)."""
    _pending(config, "training.sft")


@train_app.command("reranker")
def train_reranker(
    config: ConfigPath,
) -> None:
    """Compatibility stub: reranker training is prohibited by project policy."""
    _validated(config)
    console.print(
        "Configuration valid. Reranker training is disabled by AGENTS.md: use frozen "
        "primary and shadow judges. Task 02 provides benchmarking and calibration only."
    )
    raise typer.Exit(code=3)


@app.command()
def generate(config: ConfigPath) -> None:
    """Generate queries (implemented by a later task)."""
    _pending(config, "generation")


@preferences_app.command("build")
def build_preferences(
    config: ConfigPath,
) -> None:
    """Build preference pairs (implemented by task 06)."""
    _pending(config, "preferences.build")


@train_app.command("dpo")
def train_dpo(config: ConfigPath) -> None:
    """Train with DPO (implemented by task 07)."""
    _pending(config, "training.dpo")


@train_app.command("grpo")
def train_grpo(config: ConfigPath) -> None:
    """Train with GRPO (implemented by task 08)."""
    _pending(config, "training.grpo")


@evaluate_app.command("generator")
def evaluate_generator(
    config: ConfigPath,
) -> None:
    """Evaluate a generator (implemented by task 04)."""
    _pending(config, "evaluation.generator")


@evaluate_app.command("embedder")
def evaluate_embedder(
    config: ConfigPath,
) -> None:
    """Evaluate a probe embedder (implemented by task 04)."""
    _pending(config, "evaluation.embedder")


if __name__ == "__main__":
    app()
