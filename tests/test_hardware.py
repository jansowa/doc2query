import json
from pathlib import Path

import pytest
import torch

from doc2query.utils.hardware import collect_hardware_report, write_hardware_report


def test_cpu_hardware_report_is_supported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = collect_hardware_report()
    assert report["status"] == "ok"
    assert report["cpu_only_supported"] is True
    assert report["cuda_available"] is False
    assert report["gpu_count"] == 0

    path = write_hardware_report(tmp_path / "nested" / "hardware.json", report)
    assert json.loads(path.read_text(encoding="utf-8")) == report
