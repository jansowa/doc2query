from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from doc2query.preferences.task06_smoke import _load_design


def _design(path: Path, *, smoke: bool, pilot: bool, pilot_passages: int = 512) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "contract": "task06-candidate-execution-design-v1",
                "final_tests_used": [],
                "authorization": {
                    "smoke_authorized": smoke,
                    "pilot_authorized": pilot,
                    "pilot_passages_authorized": pilot_passages,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_smoke_design_requires_explicit_scoped_authorization(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not explicitly owner-authorized"):
        _load_design(_design(tmp_path / "closed.yaml", smoke=False, pilot=False))
    opened = _load_design(_design(tmp_path / "open.yaml", smoke=True, pilot=False))
    assert opened["authorization"]["smoke_authorized"] is True


def test_pilot_design_requires_exact_scoped_authorization(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not explicitly owner-authorized"):
        _load_design(
            _design(tmp_path / "closed.yaml", smoke=True, pilot=False), stage="pilot"
        )
    with pytest.raises(ValueError, match="pin exactly 512"):
        _load_design(
            _design(tmp_path / "wrong.yaml", smoke=True, pilot=True, pilot_passages=1024),
            stage="pilot",
        )
    opened = _load_design(
        _design(tmp_path / "open.yaml", smoke=True, pilot=True), stage="pilot"
    )
    assert opened["authorization"]["pilot_passages_authorized"] == 512


def test_smoke_design_rejects_final_test_provenance(tmp_path: Path) -> None:
    path = _design(tmp_path / "design.yaml", smoke=True, pilot=False)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["final_tests_used"] = ["test"]
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot use final tests"):
        _load_design(path)
