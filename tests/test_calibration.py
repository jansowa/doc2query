import pytest

from doc2query.reranker.calibrate import PercentileCalibrator, RobustZCalibrator
from doc2query.rewards.calibration import overlap_band_reward, pearson_matrix
from doc2query.rewards.grounding import grounding_reward


def test_calibrators_have_expected_scale() -> None:
    robust = RobustZCalibrator.fit([1, 2, 3, 4, 100])
    assert robust.transform(3) == 0
    percentile = PercentileCalibrator.fit([1, 2, 3])
    assert percentile.transform(1) == 0
    assert percentile.transform(2) == 0.5
    assert percentile.transform(3) == 1


def test_overlap_band_rejects_unrelated_and_copy() -> None:
    assert overlap_band_reward(0.3, low=0.2, high=0.4) == 1
    assert overlap_band_reward(0.0, low=0.2, high=0.4) == 0
    assert overlap_band_reward(1.0, low=0.2, high=0.4) == 0
    assert 0 < overlap_band_reward(0.1, low=0.2, high=0.4) < 1


def test_grounding_calibrates_absolute_and_margin_separately() -> None:
    score = RobustZCalibrator.fit([0, 1, 2])
    margin = RobustZCalibrator.fit([-2, 0, 2])
    reward = grounding_reward(2, 0, score_calibrator=score, margin_calibrator=margin)
    assert reward.absolute > 0
    assert reward.margin > 0
    assert reward.total == pytest.approx(reward.absolute + reward.margin)


def test_correlation_matrix() -> None:
    result = pearson_matrix([{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}])
    assert result["x"]["y"] == pytest.approx(1)
