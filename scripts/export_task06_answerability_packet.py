#!/usr/bin/env python3
"""Zamroź label-free pakiet itemów dla sędziego odpowiadalności uruchamianego zdalnie.

Pakiet zawiera wyłącznie (item_id, zapytanie, pasaż) i manifest z SHA-256. Etykiety
(referencje Groq, klasy konstrukcyjne korpusu) **zostają na tej maszynie**, więc zdalny
sędzia nie ma czego dostrajać pod wynik, a import weryfikuje, że oceniono dokładnie ten
zamrożony zbiór itemów.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.answerability_judge import (
    calibration_items_from_audit,
    calibration_items_from_reward_corpus,
)
from doc2query.preferences.answerability_remote import write_packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-dir", type=Path, default=Path("artifacts/task06/preference_audit_v2")
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/task06/reward_validation_corpus_v1/corpus.jsonl"),
    )
    parser.add_argument(
        "--cohort-records",
        type=Path,
        default=Path("artifacts/task06/candidate_pilot_v1/cohort.records.jsonl"),
    )
    parser.add_argument(
        "--packet-dir", type=Path, default=Path("artifacts/task06/answerability_packet_v1")
    )
    args = parser.parse_args()

    items = calibration_items_from_audit(args.export_dir)
    items += calibration_items_from_reward_corpus(args.corpus, args.cohort_records)
    unique = {item.item_id: item for item in items}
    ordered = [unique[item_id] for item_id in sorted(unique)]
    manifest = write_packet(ordered, args.packet_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
