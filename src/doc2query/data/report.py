"""Streaming JSON/HTML data audit report with compact exact histograms."""

from __future__ import annotations

import hashlib
import heapq
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from doc2query.data.invert import lexical_overlap, query_style
from doc2query.data.token_lengths import TokenLengthCounter, load_tokenizer_specs
from doc2query.utils.records import read_records, write_json
from doc2query.utils.tracking import collect_code_provenance

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _percentiles(histogram: Counter[int], probabilities: tuple[float, ...]) -> dict[str, float]:
    total = sum(histogram.values())
    if not total:
        return {f"p{int(probability * 100)}": 0.0 for probability in probabilities}
    ordered = sorted(histogram.items())
    result: dict[str, float] = {}
    for probability in probabilities:
        target = max(1, round(probability * total))
        cumulative = 0
        selected = 0
        for candidate, count in ordered:
            selected = candidate
            cumulative += count
            if cumulative >= target:
                break
        result[f"p{int(probability * 100)}"] = float(selected)
    return result


def _sample_push(
    heap: list[tuple[int, str]], key: str, value: dict[str, Any], size: int = 50
) -> None:
    priority = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    item = (-priority, payload)
    if len(heap) < size:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _load_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def build_data_report(
    input_paths: list[Path],
    *,
    json_path: Path,
    html_path: Path,
    validation_report: Path | None = None,
    dedup_report: Path | None = None,
    split_manifest: Path | None = None,
    tokenizer_config: Path | None = None,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    styles: Counter[str] = Counter()
    query_chars: Counter[int] = Counter()
    query_words: Counter[int] = Counter()
    passage_chars: Counter[int] = Counter()
    passage_words: Counter[int] = Counter()
    whitespace_tokens: Counter[int] = Counter()
    tokenizer_counters = (
        [TokenLengthCounter(spec) for spec in load_tokenizer_specs(tokenizer_config)]
        if tokenizer_config is not None
        else []
    )
    model_token_histograms: dict[str, dict[str, Counter[int]]] = {
        counter.spec.label: {"queries": Counter(), "passages": Counter()}
        for counter in tokenizer_counters
    }
    typical: list[tuple[int, str]] = []
    suspicious: list[tuple[int, str]] = []
    for input_path in input_paths:
        for record in read_records(input_path):
            counts["records"] += 1
            query = str(record["query"])
            query_chars[len(query)] += 1
            query_words[len(_TOKEN.findall(query))] += 1
            style = query_style(query)
            styles[style] += 1
            metadata = record.get("metadata", {})
            domain = (
                str(metadata.get("domain", "unknown")) if isinstance(metadata, dict) else "unknown"
            )
            domains[domain] += 1
            max_overlap = 0.0
            flags: list[str] = []
            documents = [*record["positives"], *record["hard_negatives"]]
            texts = [query, *(str(document["text"]) for document in documents)]
            for counter in tokenizer_counters:
                lengths = counter.count(texts)
                model_token_histograms[counter.spec.label]["queries"].update(lengths[:1])
                model_token_histograms[counter.spec.label]["passages"].update(lengths[1:])
            for document in documents:
                text = str(document["text"])
                words = len(_TOKEN.findall(text))
                passage_chars[len(text)] += 1
                passage_words[words] += 1
                whitespace_tokens[words] += 1
                counts["document_occurrences"] += 1
                document_metadata = document.get("metadata", {})
                if isinstance(document_metadata, dict):
                    flags.extend(
                        str(flag) for flag in document_metadata.get("text_quality_flags", [])
                    )
            for positive in record["positives"]:
                max_overlap = max(max_overlap, lexical_overlap(query, str(positive["text"])))
            sample = {
                "example_id": str(record["example_id"]),
                "query": query,
                "positive_preview": str(record["positives"][0]["text"])[:500],
                "max_overlap": max_overlap,
                "flags": sorted(set(flags)),
            }
            _sample_push(typical, str(record["example_id"]), sample)
            if flags or max_overlap >= 0.85 or not query.strip():
                _sample_push(suspicious, "suspicious:" + str(record["example_id"]), sample)
            if max_overlap >= 0.85:
                counts["high_overlap_queries"] += 1
    probabilities = (0.5, 0.9, 0.95, 0.97, 0.99)
    report = {
        "counts": dict(counts),
        "length_percentiles": {
            "query_chars": _percentiles(query_chars, probabilities),
            "query_words": _percentiles(query_words, probabilities),
            "passage_chars": _percentiles(passage_chars, probabilities),
            "passage_words": _percentiles(passage_words, probabilities),
        },
        "tokenizer_percentiles": {
            "whitespace_diagnostic_not_model_tokenizer": _percentiles(
                whitespace_tokens, probabilities
            ),
            **{
                label: {
                    kind: _percentiles(histogram, probabilities)
                    for kind, histogram in histograms.items()
                }
                for label, histograms in model_token_histograms.items()
            },
        },
        "tokenizer_note": (
            "Pinned model tokenizers were measured."
            if tokenizer_counters
            else "Model-tokenizer percentiles were not requested; do not choose max_length "
            "from the whitespace diagnostic alone. Pass --tokenizer-config."
        ),
        "style_distribution": dict(styles),
        "domain_distribution": dict(domains),
        "high_overlap_rate": (
            counts["high_overlap_queries"] / counts["records"] if counts["records"] else 0.0
        ),
        "typical_examples": [json.loads(item[1]) for item in sorted(typical, reverse=True)],
        "suspicious_examples": [json.loads(item[1]) for item in sorted(suspicious, reverse=True)],
        "validation": _load_optional(validation_report),
        "deduplication": _load_optional(dedup_report),
        "splits": _load_optional(split_manifest),
        "code": collect_code_provenance(),
    }
    write_json(json_path, report)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    payload = html.escape(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    html_path.write_text(
        "<!doctype html><html lang='pl'><meta charset='utf-8'><title>Data audit</title>"
        "<style>body{font-family:system-ui;max-width:1200px;margin:2rem auto}"
        "pre{white-space:pre-wrap}</style>"
        f"<h1>doc2query data audit</h1><pre>{payload}</pre></html>\n",
        encoding="utf-8",
    )
    return report
