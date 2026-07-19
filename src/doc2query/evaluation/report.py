"""Self-contained Markdown/HTML reports with explicit missing-measurement labels."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from doc2query.utils.records import read_records


def _display(value: Any) -> str:
    if value is None:
        return "NOT MEASURED"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _flatten(prefix: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        result = []
        for key, nested in value.items():
            result.extend(_flatten(f"{prefix}.{key}" if prefix else str(key), nested))
        return result
    return [(prefix, value)]


def _write_distribution_svg(summary: dict[str, Any], path: Path) -> None:
    metrics = [
        ("Lemma Jaccard", summary.get("lexical", {}).get("content_jaccard")),
        ("Copy density", summary.get("lexical", {}).get("copy_density")),
        ("Margin (scaled)", summary.get("reranker_margin")),
    ]
    colors = {"p05": "#93c5fd", "p50": "#2563eb", "p95": "#1e3a8a"}
    rows = []
    for row_index, (name, values) in enumerate(metrics):
        if not isinstance(values, dict):
            continue
        scale = max(abs(float(values.get("p95", 0))), abs(float(values.get("p05", 0))), 1.0)
        for value_index, percentile_name in enumerate(("p05", "p50", "p95")):
            value = float(values.get(percentile_name, 0))
            width = min(280.0, abs(value) / scale * 280.0)
            y = 35 + row_index * 90 + value_index * 18
            rows.append(
                f'<rect x="170" y="{y}" width="{width:.1f}" height="12" '
                f'fill="{colors[percentile_name]}"/>'
                f'<text x="455" y="{y + 11}" font-size="11">'
                f"{percentile_name}: {value:.4f}</text>"
            )
        rows.append(
            f'<text x="10" y="{55 + row_index * 90}" font-size="13">{html.escape(name)}</text>'
        )
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="310">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="10" y="18" font-size="14" font-weight="bold">'
        "Distribution percentiles (p05 / p50 / p95)</text>" + "".join(rows) + "</svg>\n",
        encoding="utf-8",
    )


def build_generator_report(
    summary_path: Path,
    per_generation_path: Path,
    *,
    markdown_path: Path,
    html_path: Path,
    max_examples: int = 100,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    examples = list(read_records(per_generation_path))[:max_examples]
    protocols = summary.get("protocols", {})
    pool = protocols.get("candidate_pool_ranking", {})
    corpus = protocols.get("corpus_retrieval", {})
    pool_metrics = pool.get("metrics", {})
    corpus_metrics = corpus.get("metrics", {})
    headline = [
        ("Pool Recall@1", pool_metrics.get("pool_recall_at_1")),
        ("Pool Recall@5", pool_metrics.get("pool_recall_at_5")),
        ("Pool MRR", pool_metrics.get("pool_mrr")),
        ("Pool nDCG@10", pool_metrics.get("pool_ndcg_at_10")),
        ("Corpus round-trip@1", corpus_metrics.get("corpus_round_trip_at_1")),
        ("Corpus round-trip@100", corpus_metrics.get("corpus_round_trip_at_100")),
        ("Format valid", summary.get("format", {}).get("valid_rate")),
        (
            "Probe embedder",
            summary.get("probe_embedder"),
        ),
    ]
    chart_path = html_path.with_name("metric_distributions.svg")
    _write_distribution_svg(summary, chart_path)
    copy_mean = summary.get("lexical", {}).get("copy_density", {}).get("mean")
    duplicate_mean = summary.get("diversity", {}).get("duplicate_rate", {}).get("mean")
    pareto = [
        ("Pool grounding (R@1)", pool_metrics.get("pool_recall_at_1")),
        ("Corpus round-trip@20", corpus_metrics.get("corpus_round_trip_at_20")),
        (
            "Non-copying (1-copy density)",
            1 - float(copy_mean) if isinstance(copy_mean, (int, float)) else None,
        ),
        (
            "Diversity (1-duplicate rate)",
            1 - float(duplicate_mean) if isinstance(duplicate_mean, (int, float)) else None,
        ),
        ("Probe embedder nDCG@10", summary.get("probe_embedder")),
    ]
    lines = [
        f"# Evaluation report: {summary.get('experiment_id', 'unknown')}",
        "",
        "## Executive summary",
        "",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Test fingerprint: `{summary.get('test_fingerprint', 'missing')}`",
        f"- Generations: {summary.get('generation_count', 0)}",
        f"- Primary judge: `{summary.get('judges', {}).get('primary', 'NOT MEASURED')}`",
        f"- Shadow judge: `{summary.get('judges', {}).get('shadow') or 'NOT MEASURED'}`",
        "- Generator comparison retrieval basis: `corpus_retrieval`",
        f"- Candidate-pool size: `{_display(pool.get('candidate_count'))}`",
        f"- Corpus size: `{_display(corpus.get('candidate_count'))}`",
        f"- Corpus index fingerprint: "
        f"`{_display((corpus.get('index') or {}).get('index_fingerprint'))}`",
        "",
        "## Key metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        *[f"| {name} | {_display(value)} |" for name, value in headline],
        "",
        "## Distribution plots",
        "",
        f"![Distribution percentiles]({chart_path.name})",
        "",
        "## Pareto coordinates",
        "",
        "| Objective | Value |",
        "|---|---:|",
        *[f"| {name} | {_display(value)} |" for name, value in pareto],
        "",
        "A Pareto frontier is not declared until the probe-embedder objective is measured "
        "for comparable variants.",
        "",
        "## Measurements not executed",
        "",
        *[f"- {value}" for value in summary.get("unmeasured", [])],
        "",
        "## Distribution and slice artifact",
        "",
        "The machine-readable `summary.json` contains all distributions and slices. "
        "Missing observations remain `null`; they are never replaced with zero.",
        "",
        "## Reward hacking / failure modes",
        "",
        "- Inspect low/negative reranker margins together with copying and format flags.",
        "- High source score is a proxy, not proof of answerability.",
        "- Generic, copied, answer-leaking and first-sentence-only generations "
        "require human review.",
        "",
        "## Side-by-side examples",
        "",
        "| # | Passage | Natural query | Generated query | Mode | Pool rank/size | "
        "Pool margin | Corpus RT@20 |",
        "|---:|---|---|---|---|---:|---:|---:|",
    ]
    for index, row in enumerate(examples, 1):
        passage = str(row.get("positive", {}).get("text", "")).replace("|", "\\|")
        reference = str(row.get("reference", "")).replace("|", "\\|")
        generated = str(row.get("generated", "")).replace("|", "\\|")
        lines.append(
            f"| {index} | {passage[:500]} | {reference} | {generated} | "
            f"{row.get('mode', '')} | {_display(row.get('pool_rank'))}/"
            f"{_display(row.get('pool_candidate_count'))} | "
            f"{_display(row.get('pool_margin'))} | "
            f"{_display(row.get('corpus_round_trip_at_20'))} |"
        )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metric_rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{html.escape(_display(value))}</td></tr>"
        for name, value in headline
    )
    pareto_rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{html.escape(_display(value))}</td></tr>"
        for name, value in pareto
    )
    example_rows = "".join(
        "<tr>"
        f"<td>{index}</td>"
        f"<td>{html.escape(str(row.get('positive', {}).get('text', '')))}</td>"
        f"<td>{html.escape(str(row.get('reference', '')))}</td>"
        f"<td>{html.escape(str(row.get('generated', '')))}</td>"
        f"<td>{html.escape(str(row.get('mode', '')))}</td>"
        f"<td>{html.escape(_display(row.get('pool_rank')))}/"
        f"{html.escape(_display(row.get('pool_candidate_count')))}</td>"
        f"<td>{html.escape(_display(row.get('corpus_round_trip_at_20')))}</td>"
        "</tr>"
        for index, row in enumerate(examples, 1)
    )
    unmeasured = "".join(
        f"<li>{html.escape(str(value))}</li>" for value in summary.get("unmeasured", [])
    )
    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>doc2query evaluation</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1500px;margin:auto;padding:2rem}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:.4rem;vertical-align:top}}
th{{background:#f3f3f3}}.missing{{color:#9b1c1c}}
</style></head><body>
<h1>{html.escape(str(summary.get("experiment_id", "unknown")))}</h1>
<p>Test fingerprint: <code>{html.escape(str(summary.get("test_fingerprint", "missing")))}</code></p>
<h2>Key metrics</h2><table>{metric_rows}</table>
<h2>Distribution plots</h2><img src="{html.escape(chart_path.name)}"
alt="p05 p50 p95 distribution plot">
<h2>Pareto coordinates</h2><table>{pareto_rows}</table>
<p>A frontier is not declared until comparable probe-embedder scores are measured.</p>
<h2>Not measured</h2><ul class="missing">{unmeasured}</ul>
<h2>Reward hacking / failure modes</h2>
<p>Review generic, copied, answer-leaking, negative-margin and first-sentence-only outputs.
Reranker scores are proxies and do not establish logical answerability.</p>
<h2>Side-by-side examples</h2>
<table><tr><th>#</th><th>Passage</th><th>Natural</th><th>Generated</th><th>Mode</th>
<th>Pool rank/size</th><th>Corpus RT@20</th></tr>
{example_rows}</table></body></html>
"""
    html_path.write_text(html_text, encoding="utf-8")
    return {
        "markdown": str(markdown_path),
        "html": str(html_path),
        "distribution_plot": str(chart_path),
        "examples": len(examples),
        "summary_fields": len(_flatten("", summary)),
    }
