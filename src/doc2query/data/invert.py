"""Invert canonical retrieval records into passage-to-query training pairs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from doc2query.data.validate import polish_confidence
from doc2query.utils.records import JsonParquetWriter, read_records, write_json
from doc2query.utils.tracking import collect_code_provenance

_TOKEN = re.compile(r"\w+", re.UNICODE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def query_style(query: str) -> str:
    lowered = query.lower().strip()
    words = _TOKEN.findall(lowered)
    if lowered.startswith(("jak ", "w jaki sposób")):
        return "how_to"
    if lowered.startswith(("co to", "czym jest", "co oznacza")):
        return "definition"
    if any(marker in lowered for marker in (" czy ", " a ", "różnica", "porówn")):
        return "comparison"
    if query.rstrip().endswith("?"):
        return "full_question"
    if len(words) <= 5:
        return "keyword_query"
    return "fact_lookup"


def lexical_overlap(query: str, passage: str) -> float:
    query_tokens = {token.lower() for token in _TOKEN.findall(query) if len(token) > 1}
    passage_tokens = {token.lower() for token in _TOKEN.findall(passage) if len(token) > 1}
    return (
        len(query_tokens & passage_tokens) / len(query_tokens | passage_tokens)
        if query_tokens
        else 0.0
    )


def invert_doc2query_pairs(
    input_path: Path,
    *,
    output_path: Path,
    report_path: Path,
    split: str | None = None,
    max_positives_per_query: int | None = None,
) -> dict[str, Any]:
    if max_positives_per_query is not None and max_positives_per_query < 1:
        raise ValueError("max_positives_per_query must be positive")
    records = pairs = skipped = 0
    styles: Counter[str] = Counter()
    fingerprint = hashlib.sha256()
    with JsonParquetWriter(output_path) as writer:
        for record in read_records(input_path):
            records += 1
            positives = sorted(record["positives"], key=lambda item: str(item["doc_id"]))
            positive_count = len(positives)
            selected = (
                positives[:max_positives_per_query]
                if max_positives_per_query is not None
                else positives
            )
            skipped += positive_count - len(selected)
            record_split = split or str(record.get("metadata", {}).get("split", "unknown"))
            negative_ids = [str(item["doc_id"]) for item in record["hard_negatives"]]
            for positive_index, positive in enumerate(selected):
                query = str(record["query"])
                passage = str(positive["text"])
                style = query_style(query)
                styles[style] += 1
                pair = {
                    "pair_id": f"{record['example_id']}::{positive['doc_id']}",
                    "example_id": str(record["example_id"]),
                    "doc_id": str(positive["doc_id"]),
                    "passage": passage,
                    "query": query,
                    "query_style": style,
                    "focus_sentence_id": None,
                    "focus_bucket": None,
                    "content_lemma_overlap": lexical_overlap(query, passage),
                    "negative_doc_ids": negative_ids,
                    "positive_count": positive_count,
                    "positive_index": positive_index,
                    "split": record_split,
                    "query_char_length": len(query),
                    "query_word_length": len(_TOKEN.findall(query)),
                    "query_whitespace_token_length": len(query.split()),
                    "query_language_confidence": polish_confidence(query),
                    "passage_char_length": len(passage),
                    "passage_word_length": len(_TOKEN.findall(passage)),
                    "passage_whitespace_token_length": len(passage.split()),
                    "passage_sentence_count": len(_SENTENCE.split(passage.strip())),
                    "metadata": {
                        "record": record.get("metadata", {}),
                        "positive": positive.get("metadata", {}),
                    },
                }
                writer.write(pair)
                fingerprint.update(
                    json.dumps(
                        pair, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ).encode()
                )
                pairs += 1
    report = {
        "input_records": records,
        "output_pairs": pairs,
        "skipped_positives_due_to_cap": skipped,
        "max_positives_per_query": max_positives_per_query,
        "style_distribution": dict(styles),
        "fingerprint": fingerprint.hexdigest(),
        "output_path": str(output_path),
        "code": collect_code_provenance(),
    }
    write_json(report_path, report)
    return report
