from __future__ import annotations

from pathlib import Path

import pytest
import torch

from doc2query.config import load_config
from doc2query.models.load_generator import _place_inference_model


class _MovableModel:
    def __init__(self) -> None:
        self.device: torch.device | None = None

    def to(self, device: torch.device) -> _MovableModel:
        self.device = device
        return self


def test_nonquantized_inference_model_is_explicitly_placed_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(Path("configs/experiments/s07_tiny_smoke.yaml"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    model = _MovableModel()
    assert _place_inference_model(model, config, for_training=False) is model
    assert model.device == torch.device("cuda", 0)


def test_training_and_quantized_loaders_keep_existing_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(Path("configs/experiments/s07_tiny_smoke.yaml"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = _MovableModel()
    assert _place_inference_model(model, config, for_training=True) is model
    assert model.device is None
    quantized = config.model_copy(
        update={"quantization": config.quantization.model_copy(update={"load_in_4bit": True})}
    )
    assert _place_inference_model(model, quantized, for_training=False) is model
    assert model.device is None
