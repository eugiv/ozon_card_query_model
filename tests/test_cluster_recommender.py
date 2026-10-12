import numpy as np
import pandas as pd
import pytest

from cluster_recommender import (
    RANK_FEATURES,
    HybridRetriever,
    load_embeddings,
    mask_selling_edges,
)
from dormant_learning import learning_rows


def history():
    return pd.DataFrame(
        [
            ["1", "мыло ланолиновое", 2, 5, 100, 1],
            ["2", "мыло ланолиновое", 3, 10, 100, 1],
            ["2", "мыло невская косметика", 1, 4, 100, 1],
            ["3", "крем", 0, 0, 100, 1],
        ],
        columns=["sku", "normquery", "orders", "views", "searches", "days"],
    )


def catalog():
    return sorted(["мыло ланолиновое", "мыло невская косметика", "крем", "мыло новое"])


def test_retrieval_can_find_unobserved_pair_and_never_recommends_seeds():
    rows = HybridRetriever(catalog(), history(), pd.Series(dtype=float)).generate(3)
    first = rows.loc[rows["sku"].eq("1")].set_index("normquery")
    assert "мыло невская косметика" in first.index
    assert first.loc["мыло невская косметика", "graph_score"] > 0
    assert first.loc["мыло невская косметика", "log_views"] == 0
    assert "мыло ланолиновое" not in first.index
    assert "3" not in set(rows["sku"])
    assert not rows.duplicated(["sku", "normquery"]).any()
    assert np.isfinite(rows[RANK_FEATURES].to_numpy()).all()


def test_candidates_do_not_change_when_future_changes():
    retriever = HybridRetriever(catalog(), history(), pd.Series(dtype=float))
    before = retriever.generate(3)
    future = pd.DataFrame(
        [["1", "мыло новое", 1, 100]], columns=["sku", "normquery", "orders", "views"]
    )
    learning_rows(before, future[["sku", "normquery"]], future)
    pd.testing.assert_frame_equal(before, retriever.generate(3))


def test_low_traffic_is_not_negative_and_search_frequency_is_not_sku_exposure():
    candidates = HybridRetriever(catalog(), history(), pd.Series(dtype=float)).generate(
        10
    )
    future = pd.DataFrame(
        [
            ["1", "мыло невская косметика", 1, 1],
            ["1", "мыло новое", 0, 25],
            ["2", "мыло новое", 0, 1],
        ],
        columns=["sku", "normquery", "orders", "views"],
    )
    positives = future.loc[future["orders"].gt(0), ["sku", "normquery"]]
    training, audit = learning_rows(candidates, positives, future)
    assert training["label"].sum() == 1
    assert set(training["sku"]) == {"1"}
    assert "мыло новое" in set(training.loc[training["label"].eq(0), "normquery"])
    assert training.loc[training["label"].eq(0), "label_status"].eq("unlabelled_comparison").all()
    assert audit["confirmed_negative"] == 0
    assert audit["unlabelled_sampled"] > 0


def test_mask_removes_edges_from_graph_and_features_not_only_seeds():
    masked, positives = mask_selling_edges(history())
    hidden = positives.iloc[0]
    assert hidden["sku"] == "2"
    assert masked.merge(positives, on=["sku", "normquery"]).empty
    retriever = HybridRetriever(catalog(), masked, pd.Series(dtype=float))
    candidates = retriever.generate(10)
    recovered = candidates.loc[
        (candidates["sku"] == hidden["sku"])
        & (candidates["normquery"] == hidden["normquery"])
    ].iloc[0]
    assert recovered["log_views"] == 0
    assert recovered["log_searches"] == 0
    assert recovered["history_days"] == 0
    pd.testing.assert_frame_equal(
        masked,
        mask_selling_edges(history().sample(frac=1, random_state=3))[0]
        .sort_values(["sku", "normquery"])
        .reset_index(drop=True),
    )


def test_embeddings_reorder_by_names_and_reject_missing_catalog(tmp_path):
    path = tmp_path / "vectors.npz"
    names = catalog()[::-1]
    vectors = np.eye(len(names))
    np.savez(
        path,
        normquery=np.asarray(names),
        embeddings=vectors,
        model_name=np.asarray("test"),
        model_revision=np.asarray("commit"),
    )
    loaded, metadata = load_embeddings(path, catalog())
    np.testing.assert_array_equal(loaded, vectors[::-1])
    assert metadata["model_revision"] == "commit"
    with pytest.raises(ValueError, match="точно соответствовать"):
        load_embeddings(path, catalog()[:-1])


def test_neural_vectors_are_a_separate_channel():
    vectors = np.ones((4, 2))
    candidates = HybridRetriever(
        catalog(), history(), pd.Series(dtype=float), vectors
    ).generate(3)
    np.testing.assert_allclose(candidates["embedding_mean"], 1)
    assert "embedding_mean" in RANK_FEATURES


def test_dense_similarity_matches_seedwise_cosines_including_ties():
    rng = np.random.default_rng(7)
    names = [f"cluster-{number:02}" for number in range(40)]
    past = pd.DataFrame([["1", names[0], 1, 2, 3, 1]], columns=history().columns)
    retriever = HybridRetriever(names, past, pd.Series(dtype=float))
    matrix = rng.normal(size=(40, 12)).astype(np.float32)
    matrix[39] = 0
    indices = list(range(20))
    weights = np.ones(20) / 20
    maximum, mean, closest = retriever._similarity(indices, weights, matrix)
    expected = np.column_stack([matrix @ matrix[index] for index in indices])
    np.testing.assert_allclose(maximum, expected.max(axis=1), atol=2e-6)
    np.testing.assert_allclose(mean, expected @ weights, atol=2e-6)
    np.testing.assert_array_equal(closest, np.asarray(indices)[expected.argmax(axis=1)])


def test_neural_retrieval_preserves_independent_lexical_candidates():
    base = HybridRetriever(catalog(), history(), pd.Series(dtype=float)).generate(1)
    neural = HybridRetriever(catalog(), history(), pd.Series(dtype=float),
                             np.ones((4, 2))).generate(1)
    expected = base.loc[base["text_source"].eq(1), ["sku", "normquery"]]
    actual = neural.loc[neural["text_source"].eq(1), ["sku", "normquery"]]
    pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True))


def test_dormant_filter_precedes_source_budget_and_uses_recent_views():
    names = sorted(["мыло", "мыло белое", "мыло белое новое"])
    past = pd.DataFrame([
        ["1", "мыло", 2, 20, 100, 1],
        ["1", "мыло белое", 0, 90, 100, 1],
    ], columns=history().columns)
    recent = past.copy()
    recent.loc[recent["normquery"].eq("мыло белое"), "views"] = 6
    retriever = HybridRetriever(names, past, pd.Series(dtype=float))
    rows = retriever.generate(1, max_history_views=5, traffic_history=recent)
    assert set(rows["normquery"]) == {"мыло белое новое"}
    recent.loc[recent["normquery"].eq("мыло белое"), "views"] = 0
    rows = retriever.generate(3, max_history_views=0, traffic_history=recent)
    row = rows.set_index("normquery").loc["мыло белое"]
    assert row["log_views"] == 0
    assert "мыло" not in set(rows["normquery"])


@pytest.mark.parametrize("limit", [True, -1, 1.5])
def test_dormant_limit_validation(limit):
    with pytest.raises(ValueError):
        HybridRetriever(catalog(), history(), pd.Series(dtype=float)).generate(
            max_history_views=limit
        )


def test_support_history_does_not_force_recommendations_for_inactive_skus():
    rows = HybridRetriever(catalog(), history(), pd.Series(dtype=float)).generate(
        3, target_skus={"1"}
    )
    assert set(rows["sku"]) == {"1"}
    assert rows.set_index("normquery").loc["мыло невская косметика", "graph_score"] > 0


def test_truncated_anchors_still_exclude_every_sold_cluster():
    names = sorted(["мыло", "мыло белое", "мыло белое новое"])
    past = pd.DataFrame([
        ["1", "мыло", 10, 20, 100, 1],
        ["1", "мыло белое", 1, 20, 100, 1],
    ], columns=history().columns)
    rows = HybridRetriever(names, past, pd.Series(dtype=float)).generate(3, max_seeds=1)
    assert set(rows["normquery"]) == {"мыло белое новое"}


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
def test_anchor_limit_validation(limit):
    with pytest.raises(ValueError):
        HybridRetriever(catalog(), history(), pd.Series(dtype=float)).generate(max_seeds=limit)
