"""Small streaming JSONL/Parquet helpers used by offline scoring commands."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    if ".jsonl" in path.suffixes:
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
            for row in batch.to_pylist():
                if set(row) == {"record_json"}:
                    value = json.loads(row["record_json"])
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}: record_json must contain an object")
                    yield value
                else:
                    yield row
        return
    raise ValueError("input must have .jsonl or .parquet suffix")


def read_durable_jsonl_prefix(path: Path) -> list[dict[str, Any]]:
    """Read a journal prefix and repair only a crash-truncated final line."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    valid_bytes = 0
    with path.open("rb") as handle:
        while line := handle.readline():
            if not line.endswith(b"\n"):
                break
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}: malformed durable JSONL row") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}: durable JSONL row must be an object")
            rows.append(value)
            valid_bytes = handle.tell()
    if path.stat().st_size != valid_bytes:
        with path.open("r+b") as handle:
            handle.truncate(valid_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    return rows


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


class JsonParquetWriter:
    """Streaming Parquet container for heterogeneous canonical JSON records."""

    def __init__(self, path: Path, batch_size: int = 1000) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the data dependency group for Parquet") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        self._pa = pa
        self._pq = pq
        self._path = path
        self._batch_size = batch_size
        self._buffer: list[str] = []
        self._writer: Any = None

    def write(self, record: dict[str, Any]) -> None:
        self._buffer.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
        if len(self._buffer) >= self._batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = self._pa.table({"record_json": self._buffer})
        if self._writer is None:
            self._writer = self._pq.ParquetWriter(self._path, table.schema, compression="zstd")
        self._writer.write_table(table)
        self._buffer = []

    def close(self) -> None:
        self._flush()
        if self._writer is None:
            table = self._pa.table({"record_json": self._pa.array([], type=self._pa.string())})
            self._pq.write_table(table, self._path, compression="zstd")
        else:
            self._writer.close()

    def __enter__(self) -> JsonParquetWriter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
