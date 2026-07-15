import json
from pathlib import Path

from doc2query.utils.tracking import write_run_manifest


def test_manifest_is_always_written_locally(tmp_path: Path) -> None:
    path = write_run_manifest(
        tmp_path / "run",
        experiment_id="E-test",
        seed=7,
        config={"tracking": {"online": False}},
        dataset_fingerprint="sha256:test",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "run_manifest.json"
    assert payload["experiment_id"] == "E-test"
    assert payload["seed"] == 7
    assert payload["dataset_fingerprint"] == "sha256:test"
    assert payload["config"]["tracking"]["online"] is False
    assert "hardware" in payload["runtime"]
