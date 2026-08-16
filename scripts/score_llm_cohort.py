#!/usr/bin/env python
"""Oceń kohortę pisaną tokenami modelu zamrożonym kontraktem scoringu Task 06.

Używa tej samej ścieżki co kandydaci lokalnego generatora
(`score_generation_artifact`: primary builder, shadow niezależna kontrola,
corpus round-trip na zamrożonym BM25, batch 8), więc wyniki są porównywalne
bez żadnego nowego kodu oceniającego.

Wznawialność: `evaluate_intrinsic_records` prowadzi `scoring.journal.jsonl`
i `scoring.resume.json`, wznawia się od dokładnego prefiksu fsyncowanego
dziennika, a niezgodność tożsamości wejścia zgłasza jako błąd zamiast po cichu
nadpisać stan. Maksymalna strata przy zabiciu procesu to jeden batch.

ADR: reports/decisions/task06_llm_cohort_gpu_scoring_amendment_2026-08-16.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import yaml

from doc2query.evaluation.generator import score_generation_artifact
from doc2query.preferences.execution_design import sha256_file

DESIGN = Path("configs/preferences/task06_candidate_execution_design_v1.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--design", type=Path, default=DESIGN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=16)
    args = parser.parse_args()

    design = yaml.safe_load(args.design.read_text(encoding="utf-8"))
    scoring = cast(dict[str, Any], design["scoring"])
    batch_size = args.batch_size or int(scoring.get("max_batch_size", 8))
    if batch_size > int(scoring.get("max_batch_size", 8)):
        raise SystemExit("batch scoringu przekracza kontraktowy limit")

    result = score_generation_artifact(
        args.generations,
        primary_config=Path(str(scoring["primary"]["config"])),
        shadow_config=Path(str(scoring["shadow"]["config"])),
        judge_device=args.device,
        output_dir=args.output_dir,
        test_fingerprint=sha256_file(args.generations),
        experiment_id=args.experiment_id,
        corpus_index_path=Path("data/processed/v1/evaluation/corpus-bm25-v1"),
        scoring_batch_size=batch_size,
        bm25_workers=8,
        progress_every=args.progress_every,
        minimum_hard_negatives=10,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
