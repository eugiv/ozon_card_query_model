import numpy as np
import pandas as pd

from evaluation import (
    PolicyThresholds,
    apply_model_policy,
    choose_policy_thresholds,
    cluster_bootstrap,
    expected_calibration_error,
    policy_metrics,
)


def test_perfect_probabilities_have_zero_calibration_error():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, scores) == 0.0


def test_thresholds_separate_confident_keep_and_remove():
    y = np.array([0] * 60 + [1] * 60)
    scores = np.concatenate([np.linspace(0.01, 0.35, 60), np.linspace(0.65, 0.99, 60)])
    thresholds = choose_policy_thresholds(
        y,
        scores,
        remove_eligible=np.ones(len(y), dtype=bool),
        keep_precision_target=0.95,
        remove_precision_target=0.95,
        min_rows=10,
    )
    assert thresholds.remove is not None
    assert thresholds.remove < thresholds.keep


def test_policy_never_removes_without_enough_exposure():
    frame = pd.DataFrame(
        {
            "orders_past": [1, 0, 0, 0],
            "freq_past": [5, 500, 20, 500],
            "views_past": [1, 20, 20, 1],
            "label": [1, 0, 0, 0],
        }
    )
    thresholds = PolicyThresholds(
        keep=0.8,
        remove=0.2,
        keep_precision_target=0.6,
        remove_precision_target=0.95,
        min_remove_searches=100,
        min_remove_views=3,
    )
    result = apply_model_policy(frame, np.array([0.1, 0.1, 0.1, 0.1]), thresholds)
    assert result["decision"].tolist() == [
        "keep",
        "remove_candidate",
        "review",
        "review",
    ]


def test_policy_metrics_and_cluster_bootstrap_include_removal_risk():
    frame = pd.DataFrame(
        {
            "sku": ["a", "a", "b", "b"],
            "label": [1, 0, 1, 0],
            "score": [0.9, 0.1, np.nan, np.nan],
            "decision": ["keep", "remove_candidate", "review", "review"],
        }
    )
    metrics = policy_metrics(frame)
    assert metrics["remove_precision"] == 1.0
    assert metrics["positive_removed"] == 0
    interval = cluster_bootstrap(frame, iterations=20, seed=1)
    assert interval["positive_removed"]["value"] == 0.0
