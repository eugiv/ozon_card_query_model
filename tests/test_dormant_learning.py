import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRanker

from cluster_recommender import RANK_FEATURES
from dormant_learning import (
    LEARNING_FEATURES,
    build_pairs,
    features,
    fit_ranker,
    learning_rows,
    predict_ranker,
)


def candidates(pairs):
    return pd.DataFrame([
        {"sku": sku, "normquery": query, **{name: 0.0 for name in RANK_FEATURES}}
        for sku, query in pairs
    ], columns=["sku", "normquery", *RANK_FEATURES])


def positives(pairs):
    return pd.DataFrame(pairs, columns=["sku", "normquery"])


def test_missing_targets_and_skus_are_audited_without_injection():
    pool = candidates([
        ("a", "p1"), ("a", "p2"), ("a", "u1"), ("a", "u2"),
        ("b", "solo"), ("c", "unknown"),
    ])
    targets = positives([
        ("a", "p1"), ("a", "p2"), ("a", "missing"),
        ("b", "solo"), ("missing-sku", "absent"),
    ])
    rows, audit = learning_rows(pool, targets)
    assert set(rows["sku"]) == {"a"}
    assert set(rows.loc[rows["label"].eq(1), "normquery"]) == {"p1", "p2"}
    assert set(rows.loc[rows["label"].eq(0), "normquery"]) == {"u1", "u2"}
    assert not rows.duplicated(["sku", "normquery"]).any()
    assert rows["group_id"].eq("train:a").all()
    assert audit["target_count"] == 5
    assert audit["found"] == 3
    assert audit["missing"] == 2
    assert audit["unlabelled_sampled"] == 2
    assert audit["training_rows"] == 4
    assert audit["training_groups"] == 1
    assert audit["confirmed_negative"] == 0


def test_future_exposure_never_confirms_negatives_or_changes_training():
    pool = candidates([("a", "positive"), ("a", "unseen"), ("a", "exposed")])
    targets = positives([("a", "positive")])
    future = pd.DataFrame([
        ["a", "positive", 1, 1], ["a", "exposed", 0, 200],
    ], columns=["sku", "normquery", "orders", "views"])
    before = pool.copy(deep=True)
    rows, audit = learning_rows(pool, targets, future)
    expected, expected_audit = learning_rows(pool, targets)
    pd.testing.assert_frame_equal(rows, expected)
    assert audit == expected_audit
    pd.testing.assert_frame_equal(pool, before)
    assert rows.loc[rows["label"].eq(0), "label_status"].eq(
        "unlabelled_comparison"
    ).all()
    assert rows.loc[rows["label"].eq(1), "label_status"].eq("positive").all()
    assert "observed_negative" not in set(rows["label_status"])


def test_sampling_is_capped_deterministic_and_retains_all_positives():
    pool = candidates([
        (sku, query)
        for sku in ["a", "b"]
        for query in ["p1", "p2", *[f"u{i:03}" for i in range(80)]]
    ])
    pool["lexical_max"] = np.tile(np.arange(82), 2)
    targets = positives([(sku, query) for sku in ["a", "b"] for query in ["p1", "p2"]])
    rows, audit = learning_rows(pool, targets, group_prefix="masked", seed=7)
    shuffled, shuffled_audit = learning_rows(
        pool.sample(frac=1, random_state=3),
        targets.sample(frac=1, random_state=9),
        group_prefix="masked", seed=7,
    )
    pd.testing.assert_frame_equal(rows, shuffled)
    assert audit == shuffled_audit
    assert rows.groupby("sku")["label"].sum().eq(2).all()
    assert rows.loc[rows["label"].eq(0)].groupby("sku").size().eq(24).all()
    assert not rows.duplicated(["sku", "normquery"]).any()
    assert rows["group_id"].str.startswith("masked:").all()
    selected = set(rows.loc[rows["sku"].eq("a"), "normquery"])
    assert {f"u{i:03}" for i in range(68, 80)}.issubset(selected)
    other, _ = learning_rows(pool, targets, group_prefix="masked", seed=8)
    assert set(zip(rows["sku"], rows["normquery"])) != set(zip(other["sku"], other["normquery"]))
    only_a, _ = learning_rows(pool.loc[pool["sku"].eq("a")], targets, group_prefix="masked")
    pd.testing.assert_frame_equal(rows.loc[rows["sku"].eq("a")].reset_index(drop=True), only_a)


@pytest.mark.parametrize("channel", ["lexical_max", "embedding_max", "graph_score"])
def test_hard_sampling_uses_each_retrieval_channel(channel):
    pool = candidates([("a", "p"), *[("a", f"u{i:03}") for i in range(60)]])
    pool.loc[pool["normquery"].eq("u059"), channel] = 10
    rows, _ = learning_rows(pool, positives([("a", "p")]), unlabelled_per_positive=2)
    assert "u059" in set(rows["normquery"])
    assert len(rows) == 3


def test_pairs_use_positions_not_index_and_normalize_each_sku():
    pool = candidates([
        ("a", "p1"), ("a", "p2"), ("a", "u1"), ("a", "u2"), ("a", "u3"),
        ("b", "p"), ("b", "u"),
    ])
    rows, _ = learning_rows(pool, positives([("a", "p1"), ("a", "p2"), ("b", "p")]))
    rows = rows.sample(frac=1, random_state=8)
    rows.index = [100] * len(rows)
    pairs, weights = build_pairs(rows)
    assert len(pairs) == 7
    totals = {"a": 0.0, "b": 0.0}
    unique = set()
    for (winner, loser), weight in zip(pairs, weights):
        positive, unknown = rows.iloc[winner], rows.iloc[loser]
        assert positive["sku"] == unknown["sku"]
        assert positive["group_id"] == unknown["group_id"]
        assert positive["label"] == 1 and unknown["label"] == 0
        assert unknown["label_status"] == "unlabelled_comparison"
        assert weight == pytest.approx(1 / 6 if positive["sku"] == "a" else 1)
        totals[positive["sku"]] += weight
        unique.add((winner, loser))
    assert len(unique) == len(pairs)
    assert totals == pytest.approx({"a": 1.0, "b": 1.0})


def test_features_are_finite_numeric_and_ignore_own_exposure():
    pool = candidates([("a", "z"), ("a", "a"), ("b", "q")])
    pool.index = [50, 50, 80]
    pool["embedding_mean"] = [np.nan, np.inf, -np.inf]
    pool["graph_score"] = ["bad", "2", None]
    computed = features(pool)
    assert computed.index.equals(pool.index)
    assert list(computed.columns) == LEARNING_FEATURES
    assert len(LEARNING_FEATURES) == 14
    assert np.isfinite(computed.to_numpy()).all()
    assert all(pd.api.types.is_numeric_dtype(computed[name]) for name in LEARNING_FEATURES)
    prohibited = {
        "log_views", "log_searches", "history_days", "seed_count", "log_seed_orders",
        "text_source", "embedding_source", "graph_source", "popular_source",
    }
    assert prohibited.isdisjoint(LEARNING_FEATURES)
    changed = pool.copy()
    for column in prohibited:
        changed[column] = [1000, 999999, -500]
    pd.testing.assert_frame_equal(computed, features(changed))
    assert computed.iloc[0]["lexical_max_rr"] == pytest.approx(1 / 12)
    assert computed.iloc[1]["lexical_max_rr"] == pytest.approx(1 / 11)
    assert computed.iloc[2]["lexical_max_rr"] == pytest.approx(1 / 11)


def test_feature_ties_and_ranks_are_shuffle_invariant():
    pool = candidates([("a", "z"), ("a", "a"), ("a", "m"), ("b", "z")])
    pool["lexical_max"] = [2, 2, 3, 1]
    computed = features(pool)
    shuffled = pool.sample(frac=1, random_state=7)
    pd.testing.assert_frame_equal(computed, features(shuffled).loc[pool.index])
    np.testing.assert_allclose(computed["lexical_max_rr"], [1 / 13, 1 / 12, 1 / 11, 1 / 11])


def test_training_preserves_ranks_from_full_candidate_pool():
    pool = candidates([("a", "p"), *[("a", f"u{i:03}") for i in range(60)]])
    pool["lexical_max"] = np.arange(len(pool))
    computed = features(pool)
    rows, _ = learning_rows(pool, positives([("a", "p")]), unlabelled_per_positive=2)
    expected = computed.set_axis(pool["normquery"])
    np.testing.assert_allclose(
        rows[LEARNING_FEATURES], expected.loc[rows["normquery"], LEARNING_FEATURES]
    )


@pytest.mark.parametrize("target_pairs", [[], [("missing", "p")]])
def test_empty_training_has_stable_schema_and_audit(target_pairs):
    pool = candidates([("a", "u")])
    rows, audit = learning_rows(pool, positives(target_pairs))
    assert rows.empty
    assert {"group_id", "label", "label_status", *LEARNING_FEATURES}.issubset(rows.columns)
    assert audit["found"] == 0
    assert audit["missing"] == len(target_pairs)
    assert audit["training_groups"] == 0
    assert build_pairs(rows) == ([], [])
    assert features(pool.iloc[:0]).empty
    with pytest.raises(ValueError, match="обуч"):
        fit_ranker(rows, iterations=5)


@pytest.mark.parametrize("size", [0, -1, True, 1.5, 13])
def test_invalid_sampling_budget_is_rejected(size):
    pool = candidates([("a", "p"), ("a", "u")])
    with pytest.raises(ValueError):
        learning_rows(pool, positives([("a", "p")]), unlabelled_per_positive=size)


def test_duplicate_candidates_are_rejected_and_targets_are_a_set():
    pool = candidates([("a", "p"), ("a", "u")])
    targets = positives([("a", "p"), ("a", "p")])
    rows, audit = learning_rows(pool, targets)
    assert rows["label"].sum() == 1
    assert audit["target_count"] == 1
    with pytest.raises(ValueError, match="уник"):
        learning_rows(pd.concat([pool, pool]), targets)


def test_pairs_reject_mixed_sku_groups_and_confirmed_negatives():
    pool = candidates([("a", "p"), ("a", "u"), ("b", "p"), ("b", "u")])
    rows, _ = learning_rows(pool, positives([("a", "p"), ("b", "p")]))
    mixed = rows.assign(group_id="one-group")
    with pytest.raises(ValueError, match="SKU"):
        build_pairs(mixed)
    invalid = rows.copy()
    invalid.loc[invalid["label"].eq(0), "label_status"] = "observed_negative"
    with pytest.raises(ValueError, match="unlabelled_comparison"):
        build_pairs(invalid)


def test_genuine_small_pairlogit_fit_and_prediction():
    pool = candidates([
        (f"sku-{sku}", query)
        for sku in range(4)
        for query in ["p1", "p2", "u1", "u2", "u3"]
    ])
    positive = pool["normquery"].str.startswith("p")
    for name in ["lexical_max", "lexical_mean", "embedding_max", "embedding_mean", "graph_score"]:
        pool[name] = positive.astype(float)
    pool["log_cluster_orders"] = np.arange(len(pool)) % 5
    rows, _ = learning_rows(pool, pool.loc[positive, ["sku", "normquery"]])
    model = fit_ranker(rows.sample(frac=1, random_state=1), iterations=25, seed=7)
    assert isinstance(model, CatBoostRanker)
    parameters = model.get_params()
    assert parameters["loss_function"] == "PairLogit"
    assert parameters["depth"] == 4
    assert parameters["thread_count"] == 2
    assert parameters["allow_writing_files"] is False
    assert model.feature_names_ == LEARNING_FEATURES
    scores = predict_ranker(model, pool)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (len(pool),)
    assert np.isfinite(scores).all()
    assert scores[positive].min() > scores[~positive].max()
    changed = pool.assign(log_views=1e8, log_searches=1e8, history_days=1e8, seed_count=1e8)
    np.testing.assert_array_equal(scores, predict_ranker(model, changed))
    shuffled = pool.sample(frac=1, random_state=2)
    np.testing.assert_allclose(scores[shuffled.index], predict_ranker(model, shuffled))
    assert predict_ranker(model, pool.iloc[:0]).shape == (0,)