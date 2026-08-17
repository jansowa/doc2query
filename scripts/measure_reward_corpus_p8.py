#!/usr/bin/env python
"""Zmierz predykcję P8 na ocenionym korpusie walidacyjnym nagrody.

P8 jest zapisana prospektywnie w ADR
`reports/decisions/task06_reward_validation_corpus_v1.md`:

- **P8a**: `ungrounded` ma niższy primary score niż `good_specific`
  w ≥ 85% grup (próg zamrożony w
  `configs/rewards/reward_validation_corpus_v1.yaml`);
- **P8b**: `too_general` ma słabszy corpus round-trip niż `good_specific`.
  Ta połowa została zapisana **kierunkowo, bez progu liczbowego**, więc skrypt
  raportuje wskaźniki i nie orzeka PASS/FAIL — dopisanie progu po zobaczeniu
  danych byłoby kalibracją po fakcie.

Pomiar jest czytelnikiem artefaktów: nie liczy żadnych score'ów sam, tylko
zestawia wyjście zamrożonego pipeline'u scoringu z etykietami konstrukcyjnymi.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

PRIMARY_SCORE = "pool_positive_score"
ROUND_TRIP = "corpus_round_trip_at_20"


def _rate(hits: int, total: int) -> float | None:
    return hits / total if total else None


def load_groups(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = str(row["construction_label"])
            groups[str(row["example_id"])][label] = row
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scoring",
        type=Path,
        default=Path("artifacts/task06/reward_validation_corpus_v1/scoring/per_generation.jsonl"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rewards/reward_validation_corpus_v1.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/task06/reward_validation_corpus_v1/measurement_p8.json"),
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    threshold = float(config["predictions"]["thresholds"]["P8a_ungrounded_below_good_rate_min"])

    groups = load_groups(args.scoring)
    complete = {
        key: value
        for key, value in groups.items()
        if {"good_specific", "ungrounded", "too_general"} <= set(value)
    }

    p8a_hits = p8a_ties = 0
    p8b_hits = p8b_ties = 0
    shadow_hits = 0
    good_primary: list[float] = []
    ungrounded_primary: list[float] = []
    good_round_trip: list[float] = []
    general_round_trip: list[float] = []
    good_margin: list[float] = []
    ungrounded_margin: list[float] = []

    for group in complete.values():
        good, ungrounded, general = (
            group["good_specific"],
            group["ungrounded"],
            group["too_general"],
        )
        good_primary.append(float(good[PRIMARY_SCORE]))
        ungrounded_primary.append(float(ungrounded[PRIMARY_SCORE]))
        good_margin.append(float(good["pool_margin"]))
        ungrounded_margin.append(float(ungrounded["pool_margin"]))
        good_round_trip.append(float(good[ROUND_TRIP]))
        general_round_trip.append(float(general[ROUND_TRIP]))

        if float(ungrounded[PRIMARY_SCORE]) < float(good[PRIMARY_SCORE]):
            p8a_hits += 1
        elif float(ungrounded[PRIMARY_SCORE]) == float(good[PRIMARY_SCORE]):
            p8a_ties += 1
        if float(ungrounded["shadow_score"]) < float(good["shadow_score"]):
            shadow_hits += 1
        if float(general[ROUND_TRIP]) < float(good[ROUND_TRIP]):
            p8b_hits += 1
        elif float(general[ROUND_TRIP]) == float(good[ROUND_TRIP]):
            p8b_ties += 1

    total = len(complete)
    p8a_rate = _rate(p8a_hits, total)
    results: dict[str, Any] = {
        "contract": config["contract"],
        "adr": config["adr"],
        "scoring": str(args.scoring),
        "complete_groups": total,
        "P8a": {
            "prediction": "ungrounded ma niższy primary score niż good_specific",
            "metric": PRIMARY_SCORE,
            "rate": p8a_rate,
            "threshold": threshold,
            "passed": p8a_rate is not None and p8a_rate >= threshold,
            "ties": p8a_ties,
            "mean_good": mean(good_primary) if good_primary else None,
            "mean_ungrounded": mean(ungrounded_primary) if ungrounded_primary else None,
            "mean_pool_margin_good": mean(good_margin) if good_margin else None,
            "mean_pool_margin_ungrounded": mean(ungrounded_margin) if ungrounded_margin else None,
        },
        "P8b": {
            "prediction": "too_general ma słabszy corpus round-trip niż good_specific",
            "metric": ROUND_TRIP,
            "rate_strictly_weaker": _rate(p8b_hits, total),
            "ties": p8b_ties,
            "threshold": None,
            "verdict": "kierunkowa, bez prerejestrowanego progu — nie orzekam PASS/FAIL",
            "mean_good": mean(good_round_trip) if good_round_trip else None,
            "mean_too_general": mean(general_round_trip) if general_round_trip else None,
        },
        "shadow_control": {
            "note": "niezależna kontrola, nie część P8",
            "rate_ungrounded_below_good": _rate(shadow_hits, total),
        },
        "thresholds_calibrated_here": False,
        "final_tests_used": [],
    }

    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status = "PASS" if results["P8a"]["passed"] else "FAIL"
    print(f"grup kompletnych: {total}")
    print(f"{status} P8a: {p8a_rate} (próg {threshold}, remisy {p8a_ties})")
    print(
        f"---- P8b (kierunkowa): {results['P8b']['rate_strictly_weaker']} "
        f"słabszych, remisy {p8b_ties}, "
        f"średnie {results['P8b']['mean_good']:.4f} vs {results['P8b']['mean_too_general']:.4f}"
    )
    print(f"kontrola shadow: {results['shadow_control']['rate_ungrounded_below_good']}")
    print(f"zapisano {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
