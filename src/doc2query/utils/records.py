"""Small streaming JSONL/Parquet helpers used by offline scoring commands."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{line_number}: expected object")
                    yield value
        return
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the data dependency group for Parquet") from exc
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches():
            yield from batch.to_pylist()
        return
    raise ValueError("input must have .jsonl or .parquet suffix")


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *_args: object) -> None:
        self._handle.close()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
