import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dormant_learning import fusion_scores, ranking_metrics
from ranking_snapshot import snapshot_metrics, target_ranks


ROOT = Path(__file__).resolve().parents[1]


def test_snapshot_reproduces_original_metrics_and_keeps_missing_targets():
    candidates = pd.DataFrame({
        "sku": ["a", "a", "a", "b"], "normquery": ["z", "c", "a", "x"],
        "embedding_mean": [0.5, 0.5, 0.5, 0.7], "graph_score": [0., 2., 1., 0.],
    })
    positives = pd.DataFrame({"sku": ["a", "a", "b", "c"],
                              "normquery": ["a", "missing", "x", "no_candidates"]})
    weights = {"baseline": {"pretrained_embeddings": 1.}, "system": {"co_sales": 1.}}
    original_candidates = candidates.copy(deep=True)
    original_positives = positives.copy(deep=True)
    ranks = target_ranks(candidates, positives, weights)
    assert ranks.loc[ranks.normquery.eq("a"), "baseline_rank"].item() == 1
    assert ranks.loc[ranks.normquery.eq("a"), "system_rank"].item() == 2
    assert ranks.loc[ranks.normquery.isin(["missing", "no_candidates"]), "system_rank"].isna().all()
    pairs = ranks.rename(columns={"sku": "product_id", "normquery": "query_id"})
    actual = snapshot_metrics(pairs, k=1)
    for name, channels in weights.items():
        expected = ranking_metrics(candidates[["sku", "normquery"]].assign(
            score=fusion_scores(candidates, channels)), positives, ks=(1,))
        for key, value in expected.items():
            if key in actual[name]:
                assert actual[name][key] == pytest.approx(value)
    assert actual["baseline"]["recall@1"] == pytest.approx(0.5)
    pd.testing.assert_frame_equal(candidates, original_candidates)
    pd.testing.assert_frame_equal(positives, original_positives)


@pytest.mark.parametrize("rank", [0, -1, 1.5, float("inf"), "bad"])
def test_snapshot_rejects_invalid_positions(rank):
    pairs = pd.DataFrame({"product_id": ["P1"], "query_id": ["Q1"],
                          "baseline_rank": [rank], "system_rank": [1]})
    with pytest.raises(ValueError):
        snapshot_metrics(pairs)


def test_snapshot_rejects_duplicates_and_invalid_k():
    pairs = pd.DataFrame({"product_id": ["P1"], "query_id": ["Q1"],
                          "baseline_rank": [1], "system_rank": [1]})
    for k in [True, 0, -1, 1.5]:
        with pytest.raises(ValueError):
            snapshot_metrics(pairs, k=k)
    with pytest.raises(ValueError):
        snapshot_metrics(pd.concat([pairs, pairs], ignore_index=True))
    with pytest.raises(ValueError):
        snapshot_metrics(pd.concat([pairs, pairs.assign(query_id="Q2")], ignore_index=True))
    with pytest.raises(ValueError):
        snapshot_metrics(pairs.assign(product_id=""))


def test_empty_predictions_keep_all_targets():
    pairs = pd.DataFrame({"product_id": ["P1", "P2"], "query_id": ["Q1", "Q2"],
                          "baseline_rank": [np.nan, np.nan], "system_rank": [np.nan, np.nan]})
    for measured in snapshot_metrics(pairs).values():
        assert measured["recall@10"] == measured["ndcg@10"] == measured["retrieval_recall"] == 0
        assert measured["positive_pairs"] == 2
        assert measured["evaluated_skus"] == 2


def test_public_snapshot_is_complete_anonymized_and_matches_report():
    path = ROOT / "reports/notebook_ranking_pairs.csv"
    manifest = json.loads((ROOT / "reports/notebook_ranking_manifest.json").read_text())
    report_path = ROOT / "reports/experiment_summary.json"
    report = json.loads(report_path.read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["pairs_sha256"]
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == manifest["experiment_summary_sha256"]
    pairs = pd.read_csv(path)
    assert list(pairs.columns) == ["period", "product_id", "query_id", "baseline_rank", "system_rank"]
    assert pairs.product_id.str.fullmatch(r"P\d{4}").all()
    assert pairs.query_id.str.fullmatch(r"Q\d{4}").all()
    assert set(pairs.period) == {"test", "new_days"}
    assert len(pairs) == 392
    for period in ["test", "new_days"]:
        measured = snapshot_metrics(pairs.loc[pairs.period.eq(period)])
        for name, reference in [("baseline", "e5_co_sales"), ("system", "all_four")]:
            for metric in ["recall@10", "precision@10", "ndcg@10", "retrieval_recall",
                           "evaluated_skus", "positive_pairs"]:
                assert measured[name][metric] == pytest.approx(report["ablation"][period][reference][metric])
    test = pairs.loc[pairs.period.eq("test")]
    baseline = test.baseline_rank.le(10)
    system = test.system_rank.le(10)
    assert (system & ~baseline).any()
    assert (baseline & ~system).any()
    assert test.baseline_rank.isna().any()