from datetime import date

import pandas as pd

from features import MODEL_FEATURES
from model_pipeline import (
    PAIR_SQL,
    build_windows,
    eligible_skus,
    evaluate_fold,
    prepare_frame,
)


def test_windows_are_frozen_and_do_not_overlap():
    windows = build_windows(date(2026, 9, 25), block_count=3)
    latest = windows["block_0"]
    previous = windows["block_1"]
    assert latest.feat_start == "2026-08-01"
    assert latest.feat_end == "2026-08-28"
    assert latest.label_start == "2026-08-29"
    assert latest.label_end == "2026-09-25"
    assert previous.label_end < latest.feat_start


def test_sql_selects_cohort_only_from_past_columns():
    final_filter = PAIR_SQL.split("SELECT *", maxsplit=1)[1]
    assert "orders_past > 0" in final_filter
    assert "freq_past >=" in final_filter
    assert "views_past >=" in final_filter
    assert "orders_y >" not in final_filter
    assert "freq_y >=" not in final_filter


def test_model_eligibility_uses_past_orders():
    frame = pd.DataFrame(
        {
            "sku": ["a", "a", "a", "b", "b", "b"],
            "orders_past": [1, 1, 1, 0, 0, 0],
            "orders_y": [0, 0, 0, 1, 1, 1],
        }
    )
    assert eligible_skus(frame) == {"a"}


def test_historical_features_do_not_require_card_text():
    frame = pd.DataFrame(
        {
            "sku": ["1"],
            "query": ["крем для ног"],
            "orders_past": [1],
            "freq_past": [100],
            "views_past": [20],
            "days_past": [10],
            "orders_y": [0],
            "freq_y": [80],
            "views_y": [10],
        }
    )
    prepared = prepare_frame(frame)
    assert all(feature in prepared.columns for feature in MODEL_FEATURES)
    assert prepared.loc[0, "label"] == 0
    assert prepared.loc[0, "future_exposed"]


def _synthetic_frame(seed: int) -> pd.DataFrame:
    rows = []
    for sku_index in range(6):
        for query_index in range(16):
            strength = query_index / 15
            label = int((query_index + sku_index + seed) % 10 < 2 + int(5 * strength))
            rows.append(
                {
                    "sku": f"sku-{sku_index}",
                    "query": f"запрос {query_index}",
                    "orders_past": int(query_index < 4),
                    "freq_past": 20 + query_index * 30,
                    "views_past": 3 + query_index,
                    "days_past": 10,
                    "orders_y": label,
                    "freq_y": 20 + query_index,
                    "views_y": 3 + query_index,
                }
            )
    return prepare_frame(pd.DataFrame(rows))


def test_full_fold_evaluates_final_policy():
    metrics, policy, predictions = evaluate_fold(
        _synthetic_frame(1),
        _synthetic_frame(2),
        _synthetic_frame(3),
        {"depth": 3, "learning_rate": 0.1, "iterations": 25},
    )
    assert metrics["score"]["pr_auc"] > 0
    assert metrics["policy"]["rows"] == len(policy)
    assert len(predictions) > 0
    assert set(policy["decision"]) <= {"keep", "review", "remove_candidate"}
