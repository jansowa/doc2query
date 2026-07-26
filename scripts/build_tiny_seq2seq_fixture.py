#!/usr/bin/env python3
"""Build a deterministic local T5 checkpoint for the S07 CPU smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from tokenizers.pre_tokenizers import Whitespace  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "config.json").is_file():
        print(args.output)
        return
    vocabulary = {"<pad>": 0, "</s>": 1, "<unk>": 2}
    words = (
        "Wygeneruj jedno polskie zapytanie wyszukiwawcze na które można odpowiedzieć "
        "na podstawie pasażu Nie kopiuj długich fragmentów Zachowaj konieczne nazwy "
        "własne liczby i terminy Pasaż Zapytanie jak co gdzie kiedy dlaczego ile"
    ).split()
    for word in words:
        vocabulary.setdefault(word, len(vocabulary))
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    model = T5ForConditionalGeneration(
        T5Config(  # type: ignore[call-arg]
            vocab_size=len(tokenizer),
            d_model=32,
            d_kv=8,
            d_ff=64,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            decoder_start_token_id=tokenizer.pad_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    )
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(args.output)
    model.save_pretrained(args.output, safe_serialization=True)
    print(args.output)


if __name__ == "__main__":
    main()
