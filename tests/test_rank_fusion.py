import numpy as np
import pandas as pd
import pytest

from dormant_learning import fit_fusion, fusion_scores, reciprocal_ranks


def candidates():
    return pd.DataFrame([
        ["1", "good", .9, 0., .8], ["1", "bad", .1, 1., .1],
        ["2", "good", .9, 0., .8], ["2", "bad", .1, 1., .1],
    ], columns=["sku", "normquery", "lexical_mean", "graph_score", "embedding_mean"])


def test_fusion_zero_channel_support_is_zero_not_arbitrary_rank():
    values = reciprocal_ranks(candidates(), ["co_sales"])["co_sales"]
    np.testing.assert_allclose(values, [0, 1 / 61, 0, 1 / 61])


def test_validation_fitting_prefers_useful_channel_without_test_labels():
    frame = candidates()
    positives = frame.loc[frame["normquery"].eq("good"), ["sku", "normquery"]]
    fitted = fit_fusion(frame, positives, ["lexical", "co_sales"])
    assert fitted["weights"]["lexical"] >= .75
    assert len(fitted["trials"]) == 5
    assert fitted["trials"][0]["metrics"]["positive_pairs"] == 2
    before = fusion_scores(frame, fitted["weights"])
    frame["observed_future_positive"] = True
    np.testing.assert_array_equal(before, fusion_scores(frame, fitted["weights"]))


def test_fusion_ties_and_input_order_are_deterministic():
    frame = candidates().assign(lexical_mean=1.)
    scores = fusion_scores(frame, {"lexical": 1.})
    assert scores[1] > scores[0]
    shuffled = frame.sample(frac=1, random_state=7)
    shuffled_scores = fusion_scores(shuffled, {"lexical": 1.})
    np.testing.assert_array_equal(pd.Series(shuffled_scores, index=shuffled.index).sort_index(), scores)


@pytest.mark.parametrize("weights", [{}, {"lexical": -1.}, {"lexical": float("nan")},
                                    {"lexical": .5}, {"unknown": 1.}])
def test_fusion_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        fusion_scores(candidates(), weights)


def test_missing_validation_targets_are_not_dropped():
    frame = candidates()
    positives = pd.DataFrame([["1", "absent"], ["2", "good"]], columns=["sku", "normquery"])
    fitted = fit_fusion(frame, positives, ["lexical", "co_sales"])
    best = max(row["metrics"]["ndcg@10"] for row in fitted["trials"])
    assert best == .5


def test_empty_validation_does_not_silently_choose_weights():
    with pytest.raises(ValueError, match="validation"):
        fit_fusion(candidates(), pd.DataFrame(columns=["sku", "normquery"]), ["lexical"])


def test_fusion_does_not_claim_zero_views_without_a_supplied_definition():
    rows = pd.DataFrame([["1", "a", 1., 1.], ["1", "b", .2, .3]],
                        columns=["sku", "normquery", "lexical_mean", "graph_score"])
    result = fit_fusion(rows, rows.iloc[:1][["sku", "normquery"]], ["lexical", "co_sales"])
    assert result["selection"] == "validation NDCG@10 on supplied positives"


def test_real_pipeline_preserves_text_null_and_tracks_fusion_artifacts(tmp_path, monkeypatch):
    import json

    from sample_data import write_dataset
    from test_full_history_recommender import small_config

    import full_history_recommender
    from cluster_data import sha256
    from dormant_learning import ranking_metrics
    from full_history_recommender import run_full_history

    facts, dictionary = write_dataset(tmp_path)
    records = pd.read_csv(dictionary, keep_default_na=False)
    records = pd.concat([records, pd.DataFrame([["null", "null", "2025-12-31T12:00:00Z", 200, ""]],
                                              columns=records.columns)])
    records.to_csv(dictionary, index=False)
    catalogs = []
    original_index = full_history_recommender.lexical_index

    def track_catalog(names):
        catalogs.append(list(names))
        return original_index(names)

    monkeypatch.setattr(full_history_recommender, "lexical_index", track_catalog)
    result = run_full_history(facts, dictionary, None, tmp_path / "base", small_config())
    assert result["test"]["validation_fusion"]["positive_pairs"] == 1
    assert catalogs and "null" in catalogs[0]
    pool = pd.read_csv(tmp_path / "base" / "test_candidates.csv", dtype={"sku": str},
                       float_precision="round_trip", keep_default_na=False)
    targets = pd.read_csv(tmp_path / "base" / "test_positives.csv", dtype={"sku": str},
                          keep_default_na=False)
    fitted = result["fusion"]
    predictions = pool[["sku", "normquery"]].assign(
        score=fusion_scores(pool, fitted["weights"], fitted["offset"])
    )
    assert ranking_metrics(predictions, targets) == result["test"]["validation_fusion"]
    manifest = json.loads((tmp_path / "base" / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert sha256(tmp_path / "base" / name) == digest
    with pytest.raises(ValueError, match="перезапись"):
        run_full_history(facts, dictionary, None, tmp_path / "base", small_config())


def test_paired_bootstrap_resamples_skus_not_candidate_rows():
    from dormant_learning import paired_metric_interval

    frame = candidates()
    positives = frame.loc[frame["normquery"].eq("good"), ["sku", "normquery"]]
    predictions = frame[["sku", "normquery"]].assign(score=frame["lexical_mean"])
    baseline = frame[["sku", "normquery"]].assign(score=frame["graph_score"])
    result = paired_metric_interval(predictions, baseline, positives, "mrr", samples=50)
    assert result["difference"] == .5
    assert result["95_interval"] == [.5, .5]
    assert result["skus"] == 2
    assert result == paired_metric_interval(predictions, baseline, positives, "mrr", samples=50)


def test_paired_bootstrap_includes_sku_without_predictions_and_rejects_empty():
    from dormant_learning import paired_metric_interval

    frame = candidates()
    positives = frame.loc[frame["normquery"].eq("good"), ["sku", "normquery"]]
    predictions = frame[["sku", "normquery"]].assign(score=frame["lexical_mean"])
    empty = predictions.iloc[:0]
    result = paired_metric_interval(predictions, empty, positives, samples=50)
    assert result["difference"] == 1.
    assert result["95_interval"] == [1., 1.]
    with pytest.raises(ValueError, match="SKU"):
        paired_metric_interval(predictions, empty, positives.iloc[:0])