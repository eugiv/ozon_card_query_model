from math import log2

import numpy as np
import pandas as pd
import pytest

from dormant_learning import ranking_metrics


def predictions(rows=()):
    return pd.DataFrame(rows, columns=["sku", "normquery", "score"])


def positives(rows=()):
    return pd.DataFrame(rows, columns=["sku", "normquery"])


def test_macro_metrics_include_missing_targets_and_skus():
    candidates = predictions([
        ("a", "negative", 4.0),
        ("a", "first", 3.0),
        ("a", "second", 2.0),
        ("b", "only", 1.0),
        ("unlabelled", "candidate", 10.0),
    ])
    targets = positives([
        ("a", "first"), ("a", "second"), ("a", "not_retrieved"),
        ("b", "only"), ("missing_sku", "future"),
    ])

    result = ranking_metrics(candidates, targets, ks=(2, 5))

    assert result["evaluated_skus"] == 3
    assert result["positive_pairs"] == 5
    assert result["candidate_pairs"] == 5
    assert result["recall@2"] == pytest.approx((1 / 3 + 1 + 0) / 3)
    assert result["recall@5"] == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert result["precision@2"] == pytest.approx((1 / 2 + 1 / 2 + 0) / 3)
    assert result["precision@5"] == pytest.approx((2 / 5 + 1 / 5 + 0) / 3)
    dcg_at_2 = 1 / log2(3)
    ideal_at_2 = 1 + 1 / log2(3)
    dcg_at_5 = 1 / log2(3) + 1 / log2(4)
    ideal_at_5 = 1 + 1 / log2(3) + 1 / log2(4)
    assert result["ndcg@2"] == pytest.approx((dcg_at_2 / ideal_at_2 + 1) / 3)
    assert result["ndcg@5"] == pytest.approx((dcg_at_5 / ideal_at_5 + 1) / 3)
    assert result["mrr"] == pytest.approx((1 / 2 + 1 + 0) / 3)
    assert result["retrieval_recall"] == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert all(type(result[key]) is int for key in (
        "evaluated_skus", "positive_pairs", "candidate_pairs",
    ))
    assert all(type(value) is float for key, value in result.items()
               if key not in {"evaluated_skus", "positive_pairs", "candidate_pairs"})


def test_short_list_precision_uses_k_and_ndcg_uses_all_positives():
    result = ranking_metrics(
        predictions([("a", "hit", -2.0)]),
        positives([("a", "hit"), ("a", "missing")]),
    )

    assert result["recall@5"] == result["recall@10"] == 0.5
    assert result["precision@5"] == 0.2
    assert result["precision@10"] == 0.1
    assert result["ndcg@5"] == result["ndcg@10"] == pytest.approx(1 / (1 + 1 / log2(3)))
    assert result["mrr"] == 1.0
    assert result["retrieval_recall"] == 0.5


def test_mrr_and_retrieval_recall_use_candidates_beyond_k():
    result = ranking_metrics(
        predictions([("a", "negative", 2), ("a", "hit", 1)]),
        positives([("a", "hit")]),
        ks=(1,),
    )

    assert result["recall@1"] == result["precision@1"] == result["ndcg@1"] == 0.0
    assert result["mrr"] == 0.5
    assert result["retrieval_recall"] == 1.0


def test_score_ties_use_query_ascending_independent_of_input_order():
    candidates = predictions([
        ("a", "z", 2), ("a", "b", 2), ("a", "a", 2), ("a", "top", 3),
    ])
    targets = positives([("a", "a"), ("a", "z")])
    original_candidates = candidates.copy(deep=True)
    original_targets = targets.copy(deep=True)

    result = ranking_metrics(candidates, targets, ks=(1, 2, 4))

    for seed in range(5):
        assert ranking_metrics(candidates.sample(frac=1, random_state=seed),
                               targets.iloc[::-1], ks=(1, 2, 4)) == result
    assert result["recall@1"] == 0.0
    assert result["recall@2"] == 0.5
    assert result["mrr"] == 0.5
    assert result["ndcg@4"] == pytest.approx(
        (1 / log2(3) + 1 / log2(5)) / (1 + 1 / log2(3))
    )
    pd.testing.assert_frame_equal(candidates, original_candidates)
    pd.testing.assert_frame_equal(targets, original_targets)


def test_categorical_queries_use_lexical_order_not_category_order():
    candidates = predictions([("a", "z", 1), ("a", "a", 1)])
    candidates["normquery"] = pd.Categorical(
        candidates["normquery"], categories=["z", "a"], ordered=True,
    )
    candidates["sku"] = pd.Categorical(candidates["sku"], categories=["unused", "a"])
    targets = positives([("a", "a")])
    targets["sku"] = pd.Categorical(targets["sku"], categories=["unused", "a"])

    result = ranking_metrics(candidates, targets, ks=(1,))

    assert result["evaluated_skus"] == 1
    assert result["recall@1"] == result["ndcg@1"] == result["mrr"] == 1.0


def test_categorical_scores_use_numeric_order_not_category_order():
    candidates = predictions([("a", "hit", 2), ("a", "negative", 1)])
    candidates["score"] = pd.Categorical(
        candidates["score"], categories=[2, 1], ordered=True,
    )

    result = ranking_metrics(candidates, positives([("a", "hit")]), ks=(1,))

    assert result["recall@1"] == result["mrr"] == 1.0


def test_rejects_blank_categorical_identifiers():
    candidates = predictions([("a", "hit", 1), ("a", "", 2)])
    candidates["normquery"] = pd.Categorical(candidates["normquery"])

    with pytest.raises(ValueError, match="normquery"):
        ranking_metrics(candidates, positives([("a", "hit")]))


@pytest.mark.parametrize("candidates", [
    predictions(), predictions([("other", "negative", 1.0)]),
    predictions([("a", "negative", 1.0)]),
])
def test_no_relevant_candidates_produces_zero_metrics(candidates):
    result = ranking_metrics(candidates, positives([("a", "hit")]))

    assert result["evaluated_skus"] == result["positive_pairs"] == 1
    assert result["candidate_pairs"] == len(candidates)
    assert all(value == 0.0 for key, value in result.items()
               if key not in {"evaluated_skus", "positive_pairs", "candidate_pairs"})


@pytest.mark.parametrize("candidates", [predictions(), predictions([("a", "x", 1.0)])])
def test_empty_positives_returns_none_not_zero(candidates):
    result = ranking_metrics(candidates, positives())

    assert result == {
        "evaluated_skus": 0, "positive_pairs": 0, "candidate_pairs": len(candidates),
        "recall@5": None, "precision@5": None, "ndcg@5": None,
        "recall@10": None, "precision@10": None, "ndcg@10": None,
        "mrr": None, "retrieval_recall": None,
    }


def test_custom_ks_order_extra_columns_and_numeric_skus():
    candidates = predictions([(123, "hit", 1.0)]).assign(extra="ignored")
    targets = positives([(123, "hit")]).assign(extra="ignored")

    result = ranking_metrics(candidates, targets, ks=(10, 1))

    assert result["recall@10"] == result["recall@1"] == 1.0
    assert result["precision@10"] == 0.1
    assert result["precision@1"] == 1.0
    assert "recall@5" not in result


def test_empty_ks_keeps_untruncated_metrics():
    assert ranking_metrics(predictions([("a", "hit", 1)]),
                           positives([("a", "hit")]), ks=()) == {
        "evaluated_skus": 1, "positive_pairs": 1, "candidate_pairs": 1,
        "mrr": 1.0, "retrieval_recall": 1.0,
    }


@pytest.mark.parametrize("table,column", [
    ("predictions", "sku"), ("predictions", "normquery"), ("predictions", "score"),
    ("positives", "sku"), ("positives", "normquery"),
])
def test_rejects_missing_columns_even_for_empty_tables(table, column):
    candidates, targets = predictions(), positives()
    if table == "predictions":
        candidates = candidates.drop(columns=column)
    else:
        targets = targets.drop(columns=column)

    with pytest.raises(ValueError, match=column):
        ranking_metrics(candidates, targets)


@pytest.mark.parametrize("table", ["predictions", "positives"])
@pytest.mark.parametrize("column", ["sku", "normquery"])
@pytest.mark.parametrize("value", [None, np.nan, pd.NA, "", " \t\n"])
def test_rejects_missing_or_blank_identifiers(table, column, value):
    candidates = predictions([("a", "hit", 1.0)])
    targets = positives([("a", "hit")])
    frame = candidates if table == "predictions" else targets
    frame.loc[0, column] = value

    with pytest.raises(ValueError, match=column):
        ranking_metrics(candidates, targets)


@pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf, None, pd.NA,
                                   "1.0", "invalid", True, 1 + 2j])
def test_rejects_invalid_scores(score):
    with pytest.raises(ValueError, match="score"):
        ranking_metrics(predictions([("a", "hit", score)]), positives())


@pytest.mark.parametrize("table", ["predictions", "positives"])
def test_rejects_duplicate_pairs(table):
    candidates = predictions([("a", "hit", 2.0)])
    targets = positives([("a", "hit")])
    if table == "predictions":
        candidates = predictions([("a", "hit", 2.0), ("a", "hit", 1.0)])
    else:
        targets = positives([("a", "hit"), ("a", "hit")])

    with pytest.raises(ValueError, match="duplicate"):
        ranking_metrics(candidates, targets)


def test_same_query_for_different_skus_is_not_a_duplicate():
    result = ranking_metrics(predictions([("a", "hit", 1), ("b", "hit", 2)]),
                             positives([("a", "hit"), ("b", "hit")]), ks=(1,))
    assert result["evaluated_skus"] == 2
    assert result["recall@1"] == 1.0


@pytest.mark.parametrize("ks", [(0,), (-1,), (1.0,), ("5",), (True,), (False,),
                                (np.bool_(True),), (None,), (5, 5), [5], None])
def test_rejects_invalid_ks(ks):
    with pytest.raises(ValueError, match="ks"):
        ranking_metrics(predictions(), positives(), ks=ks)


@pytest.mark.parametrize("table", ["predictions", "positives"])
def test_rejects_duplicate_column_names(table):
    candidates = predictions()
    targets = positives()
    if table == "predictions":
        candidates = pd.DataFrame(columns=["sku", "normquery", "score", "score"])
    else:
        targets = pd.DataFrame(columns=["sku", "normquery", "normquery"])

    with pytest.raises(ValueError, match="column"):
        ranking_metrics(candidates, targets)


@pytest.mark.parametrize("table", ["predictions", "positives"])
def test_rejects_non_dataframe_inputs(table):
    with pytest.raises(TypeError, match="DataFrame"):
        ranking_metrics([] if table == "predictions" else predictions(),
                        [] if table == "positives" else positives())
