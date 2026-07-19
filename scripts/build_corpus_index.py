#!/usr/bin/env python3
"""Build a frozen BM25 or auxiliary bi-encoder corpus index for Harness v1.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from doc2query.evaluation.corpus import (
    BM25IndexConfig,
    FrozenBiEncoderConfig,
    build_biencoder_index,
    build_bm25_index,
)
from doc2query.text.normalization import SimplePolishNormalizer, SpacyPolishNormalizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--backend", choices=("bm25", "auxiliary_biencoder"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-cache", type=Path)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("protocol") != "corpus_retrieval":
        raise ValueError("corpus config must declare protocol: corpus_retrieval")
    if args.backend == "bm25":
        backend = raw.get("bm25")
        if not isinstance(backend, dict):
            raise ValueError("corpus config requires a bm25 mapping")
        normalizer_config = raw.get("bm25_normalizer")
        if not isinstance(normalizer_config, dict):
            raise ValueError("corpus config requires a bm25_normalizer mapping")
        normalizer_backend = normalizer_config.get("backend")
        if normalizer_backend == "spacy_pl":
            normalizer = SpacyPolishNormalizer(str(normalizer_config["model_name"]))
        elif normalizer_backend == "simple":
            normalizer = SimplePolishNormalizer()
        else:
            raise ValueError("bm25_normalizer.backend must be spacy_pl or simple")
        result = build_bm25_index(
            args.documents,
            output_dir=args.output_dir,
            config=BM25IndexConfig(**backend),
            normalizer=normalizer,
            analysis_cache_path=args.analysis_cache,
        )
    else:
        backend = raw.get("auxiliary_biencoder")
        if not isinstance(backend, dict):
            raise ValueError("corpus config requires an auxiliary_biencoder mapping")
        result = build_biencoder_index(
            args.documents,
            output_dir=args.output_dir,
            config=FrozenBiEncoderConfig(**backend),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
