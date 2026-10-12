from datetime import date

import pandas as pd
import pytest

from cluster_data import (
    aggregate_clusters,
    dictionary_snapshot,
    future_positives,
    period_pairs,
    recommendation_windows,
)


def dictionary():
    return pd.DataFrame(
        [
            [" МЫЛО ", "мыло", "2026-01-01", 200, ""],
            ["мыло", "мыло другое", "2026-03-01", 200, ""],
            ["мыло", "мыло", "2026-01-01", 200, ""],
            ["мыло белое", "мыло", "2026-01-02", 200, ""],
            ["bad", "ошибка", "2026-01-01", 500, "failed"],
        ],
        columns=["query", "normquery", "loaded_at", "status_code", "error"],
    )


def facts():
    return pd.DataFrame(
        [
            ["2026-01-10", "1", "мыло", 100, 3, 1],
            ["2026-01-10", "2", "Мыло", 100, 5, 2],
            ["2026-01-10", "1", "мыло белое", 20, 2, 0],
            ["2026-01-11", "1", "мыло", 50, 1, 0],
            ["2026-01-10", "1", "missing", 30, 1, 2],
            ["2026-01-10", "1", " ", 30, 1, 4],
        ],
        columns=[
            "date",
            "sku",
            "query",
            "unique_search_users",
            "unique_view_users",
            "order_count",
        ],
    )


def test_snapshot_no_future_and_deterministic_ties():
    records = dictionary()
    records.loc[len(records)] = ["мыло", "я мыло", "2026-01-01", 200, ""]
    snap, audit = dictionary_snapshot(records, pd.Timestamp("2026-01-31"))
    shuffled, _ = dictionary_snapshot(
        records.sample(frac=1, random_state=7), pd.Timestamp("2026-01-31")
    )
    pd.testing.assert_frame_equal(snap, shuffled)
    assert snap.set_index("clean_query").loc["мыло", "normquery"] == "я мыло"
    assert audit["future_rows_ignored"] == 1
    assert audit["ambiguous_queries"] == 1
    assert "bad" not in snap["clean_query"].tolist()


def test_aggregate_conserves_orders_and_deduplicates_searches_across_sku():
    snap, _ = dictionary_snapshot(dictionary(), pd.Timestamp("2026-01-31"))
    daily, audit = aggregate_clusters(facts(), snap)
    first = daily.loc[daily["date"].eq(pd.Timestamp("2026-01-10"))].set_index("sku")
    assert first.loc["1", "searches"] == 120
    assert first.loc["2", "searches"] == 100
    assert first["cluster_searches"].tolist() == [120, 120]
    assert first["orders"].sum() == 3
    assert first.loc["1", "views"] == 5
    assert audit["covered_orders"] == 3 and audit["orders"] == 5
    assert audit["empty_query_orders"] == 4
    assert audit["unmapped_queries"] == ["missing"]
    pairs = period_pairs(daily, date(2026, 1, 10), date(2026, 1, 11))
    assert pairs.set_index("sku").loc["1", "days"] == 2


@pytest.mark.parametrize(
    "column,value",
    [
        ("order_count", -1),
        ("unique_view_users", float("nan")),
        ("unique_search_users", float("inf")),
        ("order_count", 0.5),
    ],
)
def test_invalid_counts_are_rejected(column, value):
    data = facts()
    data[column] = data[column].astype(float)
    data.loc[0, column] = value
    snap, _ = dictionary_snapshot(dictionary(), pd.Timestamp("2026-01-31"))
    with pytest.raises(ValueError, match=column):
        aggregate_clusters(data, snap)


def test_ambiguous_join_is_rejected():
    snap, _ = dictionary_snapshot(dictionary(), pd.Timestamp("2026-01-31"))
    with pytest.raises(ValueError, match="ровно одно"):
        aggregate_clusters(facts(), pd.concat([snap, snap]))


def test_duplicate_daily_records_cannot_double_orders():
    snap, _ = dictionary_snapshot(dictionary(), pd.Timestamp("2026-01-31"))
    with pytest.raises(ValueError, match="Дубли"):
        aggregate_clusters(pd.concat([facts(), facts().iloc[:1]]), snap)


def test_three_chronological_windows_do_not_share_target_dates():
    windows = recommendation_windows(date(2026, 9, 25))
    assert windows["train"].target_end < windows["validation"].target_start
    assert windows["validation"].target_end < windows["test"].target_start
    assert windows["test"].target_end == date(2026, 9, 25)
    for window in windows.values():
        assert (window.history_end - window.history_start).days == 27
        assert window.history_end < window.target_start


def test_future_positive_is_new_cluster_and_only_seeded_sku():
    hist = pd.DataFrame(
        [["1", "a", 1], ["1", "b", 0], ["2", "a", 0]],
        columns=["sku", "normquery", "orders"],
    )
    future = pd.DataFrame(
        [["1", "a", 3], ["1", "b", 1], ["2", "c", 1]],
        columns=["sku", "normquery", "orders"],
    )
    assert future_positives(hist, future).to_dict("records") == [
        {"sku": "1", "normquery": "b"}
    ]
