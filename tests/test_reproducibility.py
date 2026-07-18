import os
import random

import numpy as np
import pytest
import torch

from doc2query.utils.reproducibility import set_seed


def test_set_seed_repeats_all_rngs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    set_seed(123)
    first = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
    set_seed(123)
    second = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
    assert first == second
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
