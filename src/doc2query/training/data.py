"""Prompt-completion dataset preparation and completion-only collation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from doc2query.data.style_labels import intent_applicable, label_query
from doc2query.models.templates import (
    BaselineName,
    normalize_completion,
    render_controlled_prompt,
    render_prompt,
)
from doc2query.schemas import FocusMode, GenerationConfig, QueryControl, QueryForm, QueryIntent
from doc2query.utils.records import read_records

IGNORE_INDEX = -100
_BALANCE_FIELDS = ("query_style", "focus_bucket", "overlap_quantile", "passage_length_bucket")


class PromptCompletionDataset(Dataset[dict[str, Any]]):
    def __init__(self, examples: Sequence[dict[str, Any]]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


@dataclass(frozen=True)
class PreparedDatasets:
    train: list[dict[str, Any]]
    evaluation: list[dict[str, Any]]
    fingerprint: str
    weight_report: dict[str, Any]


def _rank_buckets(values: list[float], bucket_count: int = 4) -> list[str]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = ["q0"] * len(values)
    for rank, index in enumerate(order):
        result[index] = f"q{min(bucket_count - 1, rank * bucket_count // max(1, len(values)))}"
    return result


def add_balance_buckets(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add deterministic quantile buckets used by both balanced and weighted SFT."""
    if not examples:
        return []
    overlaps = [float(example.get("content_lemma_overlap", 0.0) or 0.0) for example in examples]
    lengths = [float(example.get("passage_word_length", 0.0) or 0.0) for example in examples]
    overlap_buckets = _rank_buckets(overlaps)
    length_buckets = _rank_buckets(lengths)
    enriched: list[dict[str, Any]] = []
    for index, source in enumerate(examples):
        item = dict(source)
        item["query_style"] = str(item.get("query_style") or "unknown")
        item["focus_bucket"] = str(item.get("focus_bucket") or "unknown")
        item["overlap_quantile"] = overlap_buckets[index]
        item["passage_length_bucket"] = length_buckets[index]
        enriched.append(item)
    return enriched


def _normalize_bounded(raw: list[float], minimum: float, maximum: float) -> list[float]:
    if not raw:
        return []
    low, high = 0.0, max(1.0, 2.0 / min(raw))
    for _ in range(80):
        scale = (low + high) / 2.0
        mean = sum(min(maximum, max(minimum, value * scale)) for value in raw) / len(raw)
        if mean < 1.0:
            low = scale
        else:
            high = scale
    scale = (low + high) / 2.0
    return [min(maximum, max(minimum, value * scale)) for value in raw]


def compute_example_weights(
    examples: list[dict[str, Any]],
    *,
    minimum: float,
    maximum: float,
) -> tuple[list[float], dict[str, Any]]:
    """Average inverse-frequency factors, then normalize to mean one under hard bounds."""
    if not minimum <= 1.0 <= maximum:
        raise ValueError("weight bounds must contain 1.0")
    if not examples:
        return [], {"count": 0}
    counts = {field: Counter(str(item[field]) for item in examples) for field in _BALANCE_FIELDS}
    raw = [
        sum(len(examples) / counts[field][str(item[field])] for field in _BALANCE_FIELDS)
        / len(_BALANCE_FIELDS)
        for item in examples
    ]
    weights = _normalize_bounded(raw, minimum, maximum)
    report = {
        "count": len(weights),
        "minimum": min(weights),
        "maximum": max(weights),
        "mean": sum(weights) / len(weights),
        "bounds": [minimum, maximum],
        "bucket_counts": {
            field: dict(sorted(field_counts.items())) for field, field_counts in counts.items()
        },
    }
    return weights, report


class BalancedBatchSampler(Sampler[list[int]]):
    """Deterministically oversample rare style/focus/overlap/length buckets by batch."""

    def __init__(
        self,
        examples: list[dict[str, Any]],
        *,
        batch_size: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not examples:
            raise ValueError("balanced sampling requires at least one example")
        self.examples = examples
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.weights, self.report = compute_example_weights(
            examples, minimum=0.01, maximum=max(4.0, float(len(examples)))
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.examples) // self.batch_size
        return math.ceil(len(self.examples) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            torch.tensor(self.weights, dtype=torch.double),
            len(self.examples),
            replacement=True,
            generator=generator,
        ).tolist()
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch


class CompletionOnlyCollator:
    """Tokenize prompt/completion separately and mask every prompt/padding token."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        max_completion_tokens: int,
        min_prompt_tokens: int,
        pad_to_max_length: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_completion_tokens = max_completion_tokens
        self.min_prompt_tokens = min_prompt_tokens
        self.pad_to_max_length = pad_to_max_length
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer requires a pad_token_id")
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer requires an eos_token_id")

    def _encode(self, example: dict[str, Any]) -> tuple[list[int], list[int]]:
        prompt_ids = list(self.tokenizer.encode(example["prompt"], add_special_tokens=False))
        completion_ids = list(
            self.tokenizer.encode(example["completion"], add_special_tokens=False)
        )
        completion_ids = completion_ids[: self.max_completion_tokens - 1]
        completion_ids.append(int(self.tokenizer.eos_token_id))
        maximum_completion = self.max_length - self.min_prompt_tokens
        if len(completion_ids) > maximum_completion:
            completion_ids = [
                *completion_ids[: maximum_completion - 1],
                int(self.tokenizer.eos_token_id),
            ]
        prompt_budget = self.max_length - len(completion_ids)
        if len(prompt_ids) > prompt_budget:
            prefix = min(self.min_prompt_tokens, prompt_budget)
            suffix = prompt_budget - prefix
            prompt_ids = prompt_ids[:prefix] + (prompt_ids[-suffix:] if suffix else [])
        if not completion_ids or len(prompt_ids) + len(completion_ids) > self.max_length:
            raise RuntimeError("completion-preserving truncation invariant failed")
        return prompt_ids, completion_ids

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode(example) for example in examples]
        width = (
            self.max_length
            if self.pad_to_max_length
            else max(len(prompt) + len(completion) for prompt, completion in encoded)
        )
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for prompt, completion in encoded:
            ids = prompt + completion
            padding = width - len(ids)
            input_ids.append(ids + [int(self.tokenizer.pad_token_id)] * padding)
            labels.append([IGNORE_INDEX] * len(prompt) + completion + [IGNORE_INDEX] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "sample_weight": torch.tensor(
                [float(example.get("sample_weight", 1.0)) for example in examples],
                dtype=torch.float,
            ),
        }


class Seq2SeqCompletionCollator:
    """Tokenize encoder input and completion labels with independent budgets."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        max_completion_tokens: int,
        pad_to_max_length: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_completion_tokens = max_completion_tokens
        self.pad_to_max_length = pad_to_max_length
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("seq2seq tokenizer requires pad_token_id and eos_token_id")

    def _encode(self, example: dict[str, Any]) -> tuple[list[int], list[int]]:
        source = list(self.tokenizer.encode(example["prompt"], add_special_tokens=False))
        source = [*source[: self.max_length - 1], int(self.tokenizer.eos_token_id)]
        target = list(self.tokenizer.encode(example["completion"], add_special_tokens=False))
        target = [
            *target[: self.max_completion_tokens - 1],
            int(self.tokenizer.eos_token_id),
        ]
        if not source or not target:
            raise RuntimeError("seq2seq source/target encoding cannot be empty")
        return source, target

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode(example) for example in examples]
        source_width = (
            self.max_length if self.pad_to_max_length else max(len(x[0]) for x in encoded)
        )
        target_width = (
            self.max_completion_tokens
            if self.pad_to_max_length
            else max(len(x[1]) for x in encoded)
        )
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for source, target in encoded:
            source_padding = source_width - len(source)
            target_padding = target_width - len(target)
            input_ids.append(source + [int(self.tokenizer.pad_token_id)] * source_padding)
            attention_mask.append([1] * len(source) + [0] * source_padding)
            labels.append(target + [IGNORE_INDEX] * target_padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "sample_weight": torch.tensor(
                [float(example.get("sample_weight", 1.0)) for example in examples],
                dtype=torch.float,
            ),
        }


def _convert_record(
    record: dict[str, Any],
    baseline: BaselineName,
    generation: GenerationConfig | None,
) -> dict[str, Any] | None:
    if "passage" not in record or "query" not in record:
        raise ValueError("SFT input must contain inverted passage and query fields")
    item = dict(record)
    if generation is not None and generation.controlled:
        labels = label_query(str(record["query"]))
        try:
            form = QueryForm(str(record.get("query_form", labels.form.value)))
        except ValueError:
            form = QueryForm.UNKNOWN
        try:
            intent = QueryIntent(str(record.get("query_intent", labels.intent.value)))
        except ValueError:
            intent = QueryIntent.UNKNOWN
        if record.get("intent_applicable") is False and intent != QueryIntent.UNKNOWN:
            return None
        focus_mode = generation.focus_modes[0]
        focus_bucket = record.get("focus_bucket")
        focus_sentence_id = record.get("focus_sentence_id")
        if focus_mode == FocusMode.BUCKET and focus_bucket not in {
            "beginning",
            "middle",
            "end",
        }:
            return None
        if focus_mode in {FocusMode.MARKED_SENTENCE, FocusMode.SENTENCE_ID} and not isinstance(
            focus_sentence_id, int
        ):
            return None
        control = QueryControl(
            form=form,
            intent=intent,
            intent_applicable=record.get("intent_applicable")
            if isinstance(record.get("intent_applicable"), bool)
            else intent_applicable(intent, str(record["passage"])),
            focus_mode=focus_mode,
            focus_bucket=focus_bucket if focus_mode == FocusMode.BUCKET else None,
            focus_sentence_id=(
                focus_sentence_id
                if focus_mode in {FocusMode.MARKED_SENTENCE, FocusMode.SENTENCE_ID}
                else None
            ),
        )
        item["prompt"] = render_controlled_prompt(str(record["passage"]), control)
        item["control"] = control.model_dump(mode="json")
    else:
        item["prompt"] = render_prompt(str(record["passage"]), baseline)
    item["completion"] = normalize_completion(str(record["query"]))
    return item


def prepare_datasets(
    input_path: Path,
    *,
    eval_path: Path | None,
    train_split: str,
    eval_split: str,
    baseline: BaselineName,
    strategy: str,
    weight_min: float,
    weight_max: float,
    seed: int,
    batch_size: int,
    generation: GenerationConfig | None = None,
    max_train_examples: int | None = None,
    max_eval_examples: int | None = None,
) -> PreparedDatasets:
    """Read canonical inverted pairs and prepare deterministic train/dev examples."""
    source_records = list(read_records(input_path))
    if eval_path is None:
        train_records = [item for item in source_records if str(item.get("split")) == train_split]
        eval_records = [item for item in source_records if str(item.get("split")) == eval_split]
        if not train_records and not eval_records:
            train_records = source_records
    else:
        train_records = source_records
        eval_records = list(read_records(eval_path))

    def capped(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
        if limit is None or len(records) <= limit:
            return records
        return sorted(
            records,
            key=lambda item: hashlib.sha256(
                f"{seed}:{item.get('pair_id', item.get('doc_id', ''))}".encode()
            ).digest(),
        )[:limit]

    def eligible(
        records: list[dict[str, Any]],
    ) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[str], int]:
        converted: list[tuple[dict[str, Any], dict[str, Any]]] = []
        invalid_completion_ids: list[str] = []
        focus_ineligible = 0
        for source in records:
            if "query" in source:
                query = source["query"]
                if (
                    not isinstance(query, str)
                    or not query.strip()
                    or "\n" in query
                    or "\r" in query
                ):
                    invalid_completion_ids.append(
                        str(source.get("pair_id", source.get("example_id", "unknown")))
                    )
                    continue
            item = _convert_record(source, baseline, generation)
            if item is not None:
                converted.append((source, item))
            else:
                focus_ineligible += 1
        return converted, invalid_completion_ids, focus_ineligible

    train_pairs, invalid_train_ids, focus_ineligible_train = eligible(train_records)
    eval_pairs, invalid_eval_ids, focus_ineligible_eval = eligible(eval_records)
    train_sources = capped([source for source, _ in train_pairs], max_train_examples)
    eval_sources = capped([source for source, _ in eval_pairs], max_eval_examples)
    train_ids = {id(source) for source in train_sources}
    eval_ids = {id(source) for source in eval_sources}
    train = add_balance_buckets([item for source, item in train_pairs if id(source) in train_ids])
    evaluation = add_balance_buckets(
        [item for source, item in eval_pairs if id(source) in eval_ids]
    )
    train_records = train_sources
    eval_records = eval_sources
    if not train:
        raise ValueError(f"no training records found for split {train_split!r}")
    weights, report = compute_example_weights(train, minimum=weight_min, maximum=weight_max)
    report["focus_ineligible_train_records"] = focus_ineligible_train
    report["focus_ineligible_eval_records"] = focus_ineligible_eval
    report["invalid_completion_train_records"] = len(invalid_train_ids)
    report["invalid_completion_eval_records"] = len(invalid_eval_ids)
    report["invalid_completion_examples"] = {
        "train": invalid_train_ids[:20],
        "evaluation": invalid_eval_ids[:20],
    }
    for item, weight in zip(train, weights, strict=True):
        item["sample_weight"] = weight if strategy == "weighted" else 1.0
    if strategy == "balanced":
        sampler = BalancedBatchSampler(train, batch_size=batch_size, seed=seed)
        indices = [index for batch in sampler for index in batch]
        train = [dict(train[index], sample_weight=1.0) for index in indices]
        report["balanced_resampled_examples"] = len(train)
    fingerprint = hashlib.sha256()
    for item in train_records + eval_records:
        fingerprint.update(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        )
    return PreparedDatasets(train, evaluation, fingerprint.hexdigest(), report)
