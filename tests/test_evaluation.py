import numpy as np
import pytest

from anomaly.evaluation import image_metrics, split_indices
from anomaly.threshold import optimise_threshold


def test_extreme_costs_and_tied_scores():
    labels, scores = np.array([0, 0, 1, 1]), np.ones(4)
    cheap_miss = optimise_threshold(labels, scores, 0.001, 0.02)
    expensive_miss = optimise_threshold(labels, scores, 100000, 0.02)
    assert not (scores >= cheap_miss.threshold).any()
    assert (scores >= expensive_miss.threshold).all()


def test_optimiser_matches_exhaustive_cost():
    rng = np.random.default_rng(9)
    labels = np.array([0] * 20 + [1] * 15)
    scores = rng.normal(size=len(labels))
    candidates = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)]
    for ratio in [0.01, 1, 10, 10000]:
        point = optimise_threshold(labels, scores, ratio, 0.02)
        costs = []
        for threshold in candidates:
            pred = scores >= threshold
            costs.append(
                ratio * 0.02 * np.mean(~pred[labels == 1]) + 0.98 * np.mean(pred[labels == 0])
            )
        assert point.expected_cost == pytest.approx(min(costs))


@pytest.mark.parametrize(
    "labels,scores,ratio,base",
    [
        ([0, 0], [1, 2], 1, 0.02),
        ([0, 1], [1, np.nan], 1, 0.02),
        ([0, 1], [1, 2], -1, 0.02),
        ([0, 1], [1, 2], 1, 0),
    ],
)
def test_invalid_threshold_inputs(labels, scores, ratio, base):
    with pytest.raises(ValueError):
        optimise_threshold(np.array(labels), np.array(scores), ratio, base)


def test_split_disjoint_exhaustive_and_reproducible():
    types = ["good"] * 20 + ["crack"] * 10 + ["dent"] * 6
    calibration, evaluation = split_indices(types, 0.3, 42)
    assert not set(calibration) & set(evaluation)
    assert sorted([*calibration, *evaluation]) == list(range(len(types)))
    assert np.array_equal(calibration, split_indices(types, 0.3, 42)[0])
    assert {types[i] for i in calibration} == {types[i] for i in evaluation} == set(types)


def test_prevalence_adjustment_and_confusion_order():
    metrics = image_metrics(np.array([0, 0, 1, 1]), np.array([0.0, 0.8, 0.7, 0.9]), 0.75, 10, 0.02)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["precision"] == 0.5
    assert metrics["scenario_precision"] == pytest.approx(0.02)
    assert metrics["scenario_average_precision"] < metrics["average_precision"]
