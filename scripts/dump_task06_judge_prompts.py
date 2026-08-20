#!/usr/bin/env python3
"""Zrzut faktycznie wysyłanych requestów sędziego + diagnostyka prefiksów.

Powstał do rozmowy o `prefix cache hit rate`: pokazuje bajtowo, co idzie na serwer, w
jakiej kolejności i ile prefiksu dzielą kolejne requesty w tym samym pasie. Nie wysyła
niczego i nie dotyka żadnego artefaktu — tylko czyta pakiet i pisze pliki diagnostyczne.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

SCRIPT = Path(__file__).resolve().parent / "task06_judge_remote.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("task06_judge_remote", SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - blad srodowiska
        raise RuntimeError(f"nie udalo sie zaladowac {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def common_prefix(left: str, right: str) -> int:
    count = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        count += 1
    return count


def send_order(runner: ModuleType, items: list[dict[str, Any]], lanes: int) -> list[dict[str, Any]]:
    """Kolejność, w jakiej requesty faktycznie wychodzą: pasy pracują równolegle.

    Pas wysyła kolejny request dopiero po odpowiedzi na poprzedni, więc realna sekwencja
    to przeplot pasów. Dla cache'u prefiksów liczy się poprzednik **w tym samym pasie**.
    """
    buckets = runner.lanes_by_passage(items, lanes)
    order: list[dict[str, Any]] = []
    for round_index in range(max((len(bucket) for bucket in buckets), default=0)):
        for lane_index, bucket in enumerate(buckets):
            if round_index < len(bucket):
                order.append(
                    {"lane": lane_index, "round": round_index, "item": bucket[round_index]}
                )
    return order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet-dir", type=Path, default=Path("artifacts/task06/answerability_pool_authorized_v1")
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("artifacts/task06/judge_cache_diagnostics")
    )
    args = parser.parse_args()

    runner = _load_runner()
    items = runner.load_items(str(args.packet_dir / "items.jsonl"))
    fake_args = SimpleNamespace(
        model=args.model, seed=20260817, max_tokens=24, decoding="json_schema_enum"
    )
    order = send_order(runner, items, args.lanes)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    last_in_lane: dict[int, str] = {}
    rows: list[dict[str, Any]] = []
    for position, entry in enumerate(order[: args.count]):
        item = entry["item"]
        payload = runner.build_payload(item, fake_args)
        body = json.dumps(payload, ensure_ascii=False)
        previous = last_in_lane.get(entry["lane"])
        rows.append(
            {
                "send_position": position,
                "lane": entry["lane"],
                "item_id": item["item_id"],
                "passage_sha256_prefix": runner.hashlib.sha256(
                    item["passage"].encode()
                ).hexdigest()[:12],
                "same_passage_as_previous_in_lane": previous == item["passage"],
                "common_prefix_chars_with_previous_in_lane": (
                    common_prefix(body, last_in_lane.get(str(entry["lane"]) + "_body", ""))
                ),
                "request_body_chars": len(body),
                "request_body": payload,
            }
        )
        last_in_lane[entry["lane"]] = item["passage"]
        last_in_lane[str(entry["lane"]) + "_body"] = body

    dump_path = args.out_dir / f"first_{args.count}_requests.jsonl"
    with dump_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_passage = Counter(item["passage"] for item in items)
    buckets = runner.lanes_by_passage(items, args.lanes)
    reuse = total = 0
    for bucket in buckets:
        previous = None
        for item in bucket:
            total += 1
            reuse += 1 if previous == item["passage"] else 0
            previous = item["passage"]
    shared = [row["common_prefix_chars_with_previous_in_lane"] for row in rows[args.lanes :]]
    system_prompt = runner.SYSTEM_PROMPT
    stats = {
        "packet": str(args.packet_dir),
        "items": len(items),
        "passages": len(per_passage),
        "items_per_passage_mean": len(items) / len(per_passage),
        "requests_with_same_passage_predecessor_in_lane": reuse,
        "requests_total": total,
        "same_passage_predecessor_share": reuse / total if total else None,
        "lanes": args.lanes,
        "system_prompt_chars": len(system_prompt),
        "system_prompt_sha256": runner.EXPECTED_SYSTEM_PROMPT_SHA256,
        "request_body_chars_median": statistics.median(row["request_body_chars"] for row in rows),
        "common_prefix_chars_median_after_first_round": (
            statistics.median(shared) if shared else None
        ),
        "decoding": "json_schema_enum",
        "sampling_params_constant": {
            "temperature": 0.0,
            "seed": 20260817,
            "max_tokens": 24,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    stats_path = args.out_dir / "prefix_stats.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nzrzut requestow: {dump_path}")
    print(f"statystyki: {stats_path}")


if __name__ == "__main__":
    main()
