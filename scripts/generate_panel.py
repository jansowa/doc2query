#!/usr/bin/env python3
"""Generate the fixed 100-passage panel from a base model or trained adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.config import load_config
from doc2query.models.load_generator import load_generator, load_tokenizer
from doc2query.training.panel import generate_panel
from doc2query.utils.records import read_records, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--input",
        type=Path,
        help="explicit inverted dataset; when set, all records are eligible for the panel",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["config", "greedy", "sampling"], default="config")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode != "config":
        config = config.model_copy(
            update={
                "generation": config.generation.model_copy(
                    update={"do_sample": args.mode == "sampling", "num_return_sequences": 1}
                )
            }
        )
    input_path = args.input or config.data.input_path
    if input_path is None:
        raise ValueError("panel generation requires a materialized input_path")
    tokenizer = load_tokenizer(config)
    model, _ = load_generator(config, for_training=False)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    if args.input is not None:
        records = list(read_records(input_path))
    else:
        records = [
            record
            for record in read_records(input_path)
            if str(record.get("split")) == config.data.eval_split
        ]
        if not records:
            records = list(read_records(input_path))
    report = generate_panel(model, tokenizer, records, output_path=args.output, config=config)
    report_path = args.output.with_suffix(".report.json")
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
