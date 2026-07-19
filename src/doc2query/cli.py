"""Stable command-line interface for the doc2query research pipeline."""

import json
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError
from rich.console import Console

from doc2query.config import load_config
from doc2query.data.validate import ValidationPolicy, validate_dataset
from doc2query.evaluation.embedder_probe import ProbeRecipe, run_probe_experiment
from doc2query.evaluation.generator import run_checkpoint_evaluation
from doc2query.training.sft import run_sft
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
def train_sft(
    config: ConfigPath,
    resume_if_available: Annotated[
        bool,
        typer.Option(
            "--resume-if-available",
            help="Start fresh or resume the newest compatible complete checkpoint.",
        ),
    ] = False,
) -> None:
    """Train an ordinary, balanced, or weighted completion-only SFT adapter."""
    parsed = load_config(config)
    summary = run_sft(parsed, resume_if_available=resume_if_available)
    console.print_json(json.dumps(summary))


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
    frozen_manifest: Annotated[
        Path, typer.Option("--frozen-manifest", exists=True, dir_okay=False)
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    subset: Annotated[str, typer.Option("--subset")] = "test_generator_panel_rank10",
    adapter: Annotated[Path | None, typer.Option("--adapter", exists=True)] = None,
    primary_judge: Annotated[
        Path | None, typer.Option("--primary-judge", exists=True, dir_okay=False)
    ] = None,
    shadow_judge: Annotated[
        Path | None, typer.Option("--shadow-judge", exists=True, dir_okay=False)
    ] = None,
    judge_device: Annotated[str | None, typer.Option("--judge-device")] = None,
    max_examples: Annotated[int | None, typer.Option("--max-examples", min=1)] = None,
    generations: Annotated[
        Path | None, typer.Option("--generations", exists=True, dir_okay=False)
    ] = None,
    generation_only: Annotated[bool, typer.Option("--generation-only")] = False,
    corpus_index: Annotated[
        Path | None, typer.Option("--corpus-index", exists=True, file_okay=False)
    ] = None,
) -> None:
    """Generate deterministic/diverse queries, score, slice and report a checkpoint."""
    result = run_checkpoint_evaluation(
        config,
        frozen_manifest=frozen_manifest,
        subset=subset,
        output_dir=output_dir,
        adapter_path=adapter,
        primary_config=primary_judge,
        shadow_config=shadow_judge,
        judge_device=judge_device,
        max_examples=max_examples,
        generations_path=generations,
        generation_only=generation_only,
        corpus_index_path=corpus_index,
    )
    console.print_json(json.dumps(result))


@evaluate_app.command("embedder")
def evaluate_embedder(
    config: ConfigPath,
    frozen_manifest: Annotated[
        Path, typer.Option("--frozen-manifest", exists=True, dir_okay=False)
    ],
    recipe_path: Annotated[Path, typer.Option("--recipe", exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    corpus: Annotated[Path, typer.Option("--corpus", exists=True, dir_okay=False)],
    query_source: Annotated[str, typer.Option("--query-source")] = "natural",
    test_subset: Annotated[str, typer.Option("--test-subset")] = "test_embedder",
    holdout_manifest: Annotated[
        Path | None, typer.Option("--holdout-manifest", exists=True, dir_okay=False)
    ] = None,
    native_corpus: Annotated[
        Path | None, typer.Option("--native-corpus", exists=True, dir_okay=False)
    ] = None,
    holdout_profile: Annotated[str, typer.Option("--holdout-profile")] = "quick",
    synthetic_generations: Annotated[
        Path | None, typer.Option("--synthetic-generations", exists=True, dir_okay=False)
    ] = None,
    train_limit: Annotated[int | None, typer.Option("--train-limit", min=1)] = None,
) -> None:
    """Train the frozen-budget probe and evaluate natural-query retrieval."""
    import yaml

    parsed = load_config(config)
    if parsed.data.input_path is None:
        raise typer.BadParameter("probe training requires a local data.input_path")
    raw = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise typer.BadParameter("probe recipe must be a YAML mapping")
    if query_source not in {"natural", "copy_control", "synthetic"}:
        raise typer.BadParameter("query-source must be natural, copy_control, or synthetic")
    if holdout_profile not in {"quick", "medium", "full"}:
        raise typer.BadParameter("holdout-profile must be quick, medium, or full")
    result = run_probe_experiment(
        train_path=parsed.data.input_path,
        frozen_manifest=frozen_manifest,
        test_subset=test_subset,
        output_dir=output_dir,
        recipe=ProbeRecipe(**raw),
        query_source=query_source,  # type: ignore[arg-type]
        synthetic_generations=synthetic_generations,
        train_limit=train_limit,
        documents_path=corpus,
        holdout_manifest=holdout_manifest,
        native_documents_path=native_corpus,
        holdout_profile=holdout_profile,  # type: ignore[arg-type]
    )
    console.print_json(json.dumps(result))


if __name__ == "__main__":
    app()
