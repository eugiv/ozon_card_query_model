import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from sample_data import write_dataset
from sklearn.preprocessing import normalize

from cluster_data import aggregate_clusters, dictionary_snapshot, period_pairs, sha256
from cluster_recommender import CANDIDATE_COLUMNS, HybridRetriever, lexical_index
from dormant_learning import ranking_metrics
from finalize_recommender import run_finalize
from full_history_recommender import (
    _RESULT_METRICS,
    _RESULT_SCENARIOS,
    DormantConfig,
    _results_candidates,
    _results_interval,
    _results_overlap,
    _results_rank_distribution,
    _results_recall,
    _results_stress,
    enrich_candidates,
    graph_vectors,
    main,
    run_full_history,
    run_fusion_ablation,
    run_refinement,
    run_results,
    verify_folder,
)
from graph_learning import train_graph_from_files
from semantic_adapter import main as adapt


@pytest.fixture
def artifacts(tmp_path, request):
    facts, dictionary = write_dataset(tmp_path)
    snapshot, _ = dictionary_snapshot(pd.read_csv(dictionary), pd.Timestamp("2026-01-01", tz="UTC"))
    # Явно синтетические векторы используются только для проверки алгоритмов.
    catalog = sorted(snapshot["normquery"].unique())
    frame = pd.read_csv(facts, dtype={"sku": "string"})
    extra = pd.DataFrame([
        [f"2026-01-{day:02}", "005", name, 100, 5, 1]
        for day in range(1, 9) for name in ["мыло дегтярное", "мыло белое"]
    ], columns=frame.columns)
    frame = pd.concat([frame, extra], ignore_index=True)
    for day in range(9, getattr(request, "param", 8) + 1):
        frame = pd.concat([frame, frame.loc[frame["date"].eq("2026-01-08")].assign(
            date=f"2026-01-{day:02}"
        )], ignore_index=True)
    if getattr(request, "param", 8) > 8:
        # Новый положительный ответ только в конце дополнительного toy-периода.
        unseen = next(name for name in catalog if name not in {"мыло дегтярное", "мыло белое"})
        raw_dictionary = pd.read_csv(dictionary, keep_default_na=False)
        raw_query = raw_dictionary.loc[raw_dictionary["normquery"].eq(unseen), "query"].iloc[0]
        frame = pd.concat([frame, pd.DataFrame([
            [f"2026-01-{request.param:02}", "005", raw_query, 100, 0, 1],
        ], columns=frame.columns)], ignore_index=True)
    frame["sku"] = frame["sku"].astype(int).astype(str)
    frame.to_csv(facts, index=False)
    vectors = normalize(np.random.default_rng(7).normal(size=(len(catalog), 384))).astype(np.float32)
    embeddings = tmp_path / "base.npz"
    np.savez_compressed(embeddings, normquery=np.array(catalog), embeddings=vectors,
                        model_name=np.array("synthetic-test-only"), model_revision=np.array("toy-revision"),
                        dictionary_as_of=np.array("2025-12-31T23:59:59.999999999+00:00"),
                        text_prefix=np.array("query: "))
    config = DormantConfig(date(2025, 12, 31), date(2026, 1, 2), date(2026, 1, 4),
                           date(2026, 1, 6), window_days=2, per_source=2, iterations=(3,))
    baseline, adapter, graph = [tmp_path / name for name in ["baseline", "adapter", "graph"]]
    run_full_history(facts, dictionary, embeddings, baseline, config)
    adapt(["--facts", str(facts), "--dictionary", str(dictionary), "--embeddings", str(embeddings),
           "--dictionary-as-of", "2025-12-31", "--train-end", "2026-01-04", "--epochs", "2",
           "--output", str(adapter)])
    model, _ = train_graph_from_files(facts=facts, dictionary=dictionary, dictionary_as_of="2025-12-31",
                                      train_end="2026-01-04", output=graph, epochs=2, dimensions=4)
    return baseline, adapter, graph, facts, dictionary, config, vectors, model


def test_genuine_three_models_future_evaluation_and_immutable_output(artifacts, tmp_path):
    baseline, adapter, graph, *_ = artifacts
    output = tmp_path / "evaluation"
    result = run_refinement(baseline, adapter, graph, output)
    assert result["test"]["horizon_days"] == 2
    assert set(result["test"]["models"]) == {
        "popularity", "lexical", "co_sales", "pretrained_embeddings", "semantic_adapter",
        "lightgcn", "catboost_pu", "learned_fusion",
    }
    assert all(item["positive_pairs"] == 1 for item in result["test"]["models"].values())
    assert result["selected_method"] == max(result["validation"]["models"],
                                            key=lambda name: result["validation"]["models"][name]["ndcg@10"])
    verify_folder(output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["code"]) == {
        "cluster_data.py", "cluster_recommender.py", "dormant_learning.py",
        "full_history_recommender.py", "graph_learning.py",
    }
    ablated = run_fusion_ablation(output, tmp_path / "ablation")
    assert {value["candidate_pairs"] for value in ablated["test"].values()} == {
        result["test"]["models"]["learned_fusion"]["candidate_pairs"]
    }
    assert ablated["test"]["all_four"] == result["test"]["models"]["learned_fusion"]
    with pytest.raises(ValueError, match="существует"):
        run_refinement(baseline, adapter, graph, output)


def test_future_trained_adapter_rejected_before_scoring(artifacts, tmp_path):
    baseline, adapter, graph, *_ = artifacts
    manifest = json.loads((adapter / "manifest.json").read_text())
    manifest["training_cutoff"] = "2026-01-05"
    (adapter / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="validation/test"):
        run_refinement(baseline, adapter, graph, tmp_path / "invalid")


def test_adapter_with_different_source_rejected(artifacts, tmp_path):
    baseline, adapter, graph, facts, *_ = artifacts
    manifest = json.loads((adapter / "manifest.json").read_text())
    manifest["input_sha256"][str(facts)] = "0" * 64
    (adapter / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="других входных"):
        run_refinement(baseline, adapter, graph, tmp_path / "different_source")


def test_extra_candidates_obey_exposure_and_preserve_original_features(artifacts):
    _, _, _, facts, dictionary, config, vectors, model = artifacts
    snapshot, _ = dictionary_snapshot(pd.read_csv(dictionary), pd.Timestamp("2026-01-01", tz="UTC"))
    daily, _ = aggregate_clusters(pd.read_csv(facts, dtype={"sku": "string"}), snapshot)
    history = period_pairs(daily, date(2026, 1, 1), date(2026, 1, 4))
    recent = period_pairs(daily, date(2026, 1, 3), date(2026, 1, 4))
    catalog = sorted(snapshot["normquery"].unique())
    text = lexical_index(catalog)
    demand = pd.Series(1., index=catalog)
    base = HybridRetriever(catalog, history, demand, vectors, text).generate(
        2, max_history_views=5, traffic_history=recent
    )
    extended = enrich_candidates(base, history, recent, demand, catalog, vectors, vectors,
                                 graph_vectors(model, catalog), text, config, set(recent["sku"]))
    joined = base.merge(extended, on=["sku", "normquery"], suffixes=("_old", "_new"), validate="one_to_one")
    assert len(joined) == len(base)
    for feature in ["lexical_mean", "embedding_mean", "graph_score", "log_views"]:
        np.testing.assert_allclose(joined[f"{feature}_old"], joined[f"{feature}_new"])
    forbidden = history.loc[history["orders"].gt(0), ["sku", "normquery"]]
    assert extended.merge(forbidden, on=["sku", "normquery"]).empty
    high = recent.loc[recent["views"].gt(5), ["sku", "normquery"]]
    assert extended.merge(high, on=["sku", "normquery"]).empty
    assert not extended.duplicated(["sku", "normquery"]).any()
    unseen = extended.loc[~extended["normquery"].isin(model.queries)]
    assert (unseen["lightgcn_score"] == 0).all()
    assert not unseen["lightgcn_available"].any()


def test_full_refit_is_separate_from_evaluation_and_expert_labels_are_empty(artifacts, tmp_path):
    baseline, adapter, graph, facts, dictionary, _, _, _ = artifacts
    evaluation = tmp_path / "evaluation"
    result = run_refinement(baseline, adapter, graph, evaluation)
    before = (baseline / "ranker.cbm").read_bytes()
    final_adapter, final_graph = tmp_path / "final_adapter", tmp_path / "final_graph"
    adapt(["--facts", str(facts), "--dictionary", str(dictionary), "--embeddings", str(tmp_path / "base.npz"),
           "--dictionary-as-of", "2025-12-31", "--train-end", "2026-01-08", "--epochs", "2",
           "--output", str(final_adapter)])
    train_graph_from_files(facts=facts, dictionary=dictionary, dictionary_as_of="2025-12-31",
                           train_end="2026-01-08", output=final_graph, epochs=2, dimensions=4)
    products = tmp_path / "products.csv"
    pd.DataFrame([
        [str(sku), "Мыло", "Описание товара", "", "[]", "2026-01-08T00:00:00Z", "ok"]
        for sku in range(1, 6)
    ], columns=["sku", "name", "description", "brand", "attributes_json", "snapshot_at", "attributes_status"]).to_csv(products, index=False)
    release = tmp_path / "release"
    summary = run_finalize(evaluation, final_adapter, final_graph, products, release,
                           pd.Timestamp("2026-01-09", tz="UTC"), top_k=2)
    assert summary["source_end"] == "2026-01-08"
    assert summary["selected_method"] == result["selected_method"]
    assert summary["status"] == "full_refit_for_review_not_independent_evaluation"
    assert summary["training_rows"] > json.loads((baseline / "metrics.json").read_text())["training"]["rows"]
    assert before == (baseline / "ranker.cbm").read_bytes()
    verify_folder(release)
    review = pd.read_csv(release / "recommendations_for_review.csv")
    assert review["rank"].max() <= 2
    assert review["decision"].eq("review").all()
    assert not (release / "expert_review_template.csv").exists()
    assert summary["expert_labels_collected"] == 0
    assert "test" not in summary


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_final_refit_invalid_budget_rejected_before_inputs(tmp_path, k):
    with pytest.raises(ValueError, match="top_k"):
        run_finalize(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / "new", "2026-01-09T00:00:00Z", k)


@pytest.fixture
def stress_candidates():
    frame = pd.DataFrame({column: np.ones(100) for column in CANDIDATE_COLUMNS})
    frame["sku"] = ["001"] * 50 + ["nan"] * 50
    frame["normquery"] = [f"q{i:03}" for i in range(100)]
    frame["closest_seed"] = "seed"
    frame["retrieval_sources"] = "text"
    frame["lexical_mean"] = np.linspace(.2, .5, 100)
    frame["lexical_max"] = frame["lexical_mean"] + .1
    frame["embedding_mean"] = np.linspace(-.2, .2, 100)
    frame["embedding_max"] = frame["embedding_mean"] + .1
    frame["adapter_score"] = np.linspace(-.1, .3, 100)
    frame["lightgcn_score"] = np.linspace(-.3, .1, 100)
    frame["lightgcn_available"] = True
    frame["log_cluster_orders"] = np.log1p(3.)
    frame["log_demand"] = np.log1p(12.)
    return frame


@pytest.mark.parametrize("scenario", ["score_noise_15", "counter_outliers", "candidate_loss_30"])
def test_results_stress_canonical_deterministic_label_independent_and_no_mutation(stress_candidates, scenario):
    before = stress_candidates.copy(deep=True)
    first, counts = _results_stress(stress_candidates, scenario, 7)
    shuffled = stress_candidates.sample(frac=1, random_state=19).assign(label=1, label_status="positive")
    second, other_counts = _results_stress(shuffled, scenario, 7)
    pd.testing.assert_frame_equal(first, second[first.columns])
    pd.testing.assert_frame_equal(before, stress_candidates)
    assert counts == other_counts
    third, _ = _results_stress(stress_candidates, scenario, 8)
    assert not first.equals(third)


def test_results_clean_no_changes_and_loss_zero(stress_candidates):
    for scenario, fraction in [("clean", None), ("candidate_loss_10", 0.)]:
        actual, counts = _results_stress(stress_candidates, scenario, 7, fraction)
        pd.testing.assert_frame_equal(actual, _results_candidates(stress_candidates))
        assert counts["affected_pairs"] == counts["dropped_pairs"] == 0


def test_results_all_candidates_lost_preserves_all_positive_skus(stress_candidates):
    positives = pd.DataFrame({"sku": ["001", "nan", "missing"], "normquery": ["q000", "q050", "nan"]})
    lost, counts = _results_stress(stress_candidates, "candidate_loss_10", 7, 1.)
    actual = ranking_metrics(lost[["sku", "normquery"]].assign(score=np.empty(0)), positives)
    assert actual["evaluated_skus"] == actual["positive_pairs"] == 3
    assert counts["dropped_pairs"] == 100
    assert all(actual[metric] == 0. for metric in _RESULT_METRICS)
    np.testing.assert_array_equal(_results_recall(lost, positives, []), np.zeros(3))
    distribution = _results_rank_distribution(lost, positives, [])
    assert distribution["macro_fractions"]["missing_from_pool"] == 1.
    assert distribution["pair_counts"]["missing_from_pool"] == 3


def test_results_dropped_positive_not_removed_from_targets(stress_candidates):
    lost, _ = _results_stress(stress_candidates, "candidate_loss_30", 7)
    removed = stress_candidates.merge(lost[["sku", "normquery"]], how="left", indicator=True)
    positives = removed.loc[removed["_merge"].eq("left_only"), ["sku", "normquery"]].copy()
    before = positives.copy(deep=True)
    actual = ranking_metrics(lost[["sku", "normquery"]].assign(score=1.), positives)
    assert actual["positive_pairs"] == 30
    assert actual["retrieval_recall"] == actual["recall@10"] == 0.
    pd.testing.assert_frame_equal(positives, before)


def test_results_outliers_transform_underlying_counts_and_keep_eligibility(stress_candidates):
    actual, counts = _results_stress(stress_candidates, "counter_outliers", 7)
    changed = actual["graph_score"].ne(stress_candidates["graph_score"])
    assert changed.sum() == counts["affected_pairs"] == 1
    np.testing.assert_allclose(actual.loc[changed, "log_cluster_orders"], np.log1p(100 * 3))
    np.testing.assert_allclose(actual.loc[changed, "log_demand"], np.log1p(100 * 12))
    assert actual.loc[changed, "graph_score"].eq(100.).all()
    for column in ["log_views", "log_searches", "history_days", "lightgcn_available"]:
        pd.testing.assert_series_equal(actual[column], stress_candidates[column])


def test_results_extreme_finite_counters_and_noise_clipping(stress_candidates):
    stress_candidates["log_cluster_orders"] = 1e308
    stress_candidates["log_demand"] = 1000.
    stress_candidates["graph_score"] = 1e308
    stress_candidates["lexical_max"] = 1.
    stress_candidates["embedding_max"] = 1.
    for scenario in ["counter_outliers", "score_noise_15"]:
        actual, _ = _results_stress(stress_candidates, scenario, 7)
        assert np.isfinite(actual.select_dtypes(include="number").to_numpy()).all()
        if scenario.startswith("score_noise"):
            assert actual[["lexical_mean", "lexical_max"]].ge(0).all().all()
            assert actual[["lexical_mean", "lexical_max"]].le(1).all().all()
            for column in ["embedding_mean", "embedding_max", "adapter_score", "lightgcn_score"]:
                assert actual[column].between(-1, 1).all()
        else:
            assert actual["log_demand"].max() == pytest.approx(1000. + np.log(100))


@pytest.mark.parametrize("sigma", [.05, .15])
def test_results_score_noise_exact_population_std_and_shared_max_delta(stress_candidates, sigma):
    rng = np.random.default_rng(7)
    actual, _ = _results_stress(stress_candidates, f"score_noise_{round(sigma * 100):02}", 7)
    for mean, maximum, low, high in [
        ("lexical_mean", "lexical_max", 0, 1), ("embedding_mean", "embedding_max", -1, 1),
        ("adapter_score", None, -1, 1), ("lightgcn_score", None, -1, 1),
    ]:
        std = stress_candidates.groupby("sku")[mean].transform(lambda values: np.std(values, ddof=0))
        delta = rng.normal(size=100) * sigma * std
        for column in [mean, *([maximum] if maximum else [])]:
            np.testing.assert_allclose(actual[column], np.clip(stress_candidates[column] + delta, low, high), atol=1e-15)
    np.testing.assert_allclose(actual["graph_score"], rng.lognormal(0, sigma, 100))


def test_results_overlap_missing_short_and_empty_skus():
    clean = pd.DataFrame({"sku": ["a", "a", "b"], "normquery": ["x", "y", "x"]})
    positives = pd.DataFrame({"sku": ["a", "b", "c"], "normquery": ["x", "x", "z"]})
    actual = clean.iloc[:1]
    overlap = _results_overlap(actual, [1.], clean, [1., 1., 1.], positives)
    assert overlap["intersection_over_k"] == pytest.approx(.1 / 3)
    assert overlap["jaccard"] == pytest.approx((.5 + 0 + 1) / 3)
    unchanged = _results_overlap(clean, [1., 1., 1.], clean, [1., 1., 1.], positives)
    assert unchanged["jaccard"] == 1.
    assert unchanged["intersection_over_k"] == pytest.approx(.3 / 3)


def test_results_rank_distribution_macro_all_targets_and_tie_order():
    frame = pd.DataFrame({"sku": ["a"] * 52, "normquery": [f"q{i:02}" for i in range(52)]})
    positives = pd.DataFrame({"sku": ["a"] * 5 + ["b"],
                              "normquery": ["q00", "q05", "q10", "q50", "absent", "nan"]})
    distribution = _results_rank_distribution(frame.iloc[::-1], positives, np.ones(52))
    assert distribution["pair_counts"] == {"1-5": 1, "6-10": 1, "11-50": 1, ">50": 1, "missing_from_pool": 2}
    assert distribution["macro_fractions"] == {"1-5": .1, "6-10": .1, "11-50": .1, ">50": .1, "missing_from_pool": .6}
    assert sum(distribution["macro_fractions"].values()) == pytest.approx(1.)
    assert sum(distribution["pair_counts"].values()) == len(positives)


def test_results_bootstrap_uses_sku_means_not_independent_repetitions():
    deltas = np.array([[-1., 1., 0.], [1., 1., -1.]])
    mean_per_sku = deltas.mean(axis=0)
    interval = _results_interval(mean_per_sku, 7, 25)
    indices = np.random.default_rng(7).integers(0, 3, size=(25, 3))
    samples = mean_per_sku[indices].mean(axis=1)
    assert interval["skus"] == 3
    assert interval["difference"] == pytest.approx(mean_per_sku.mean())
    assert interval["lower"] == pytest.approx(np.quantile(samples, .025))
    assert interval["upper"] == pytest.approx(np.quantile(samples, .975))


def test_results_candidate_validation(stress_candidates):
    invalid = [stress_candidates.drop(columns="adapter_score"),
               pd.concat([stress_candidates, stress_candidates.iloc[:1]]),
               stress_candidates.assign(embedding_mean=np.inf),
               stress_candidates.assign(graph_score=-1.), stress_candidates.assign(sku=""),
               stress_candidates.assign(score=np.nan)]
    for frame in invalid:
        with pytest.raises(ValueError):
            _results_candidates(frame)
    assert "nan" in _results_candidates(stress_candidates)["sku"].unique()


@pytest.mark.parametrize("name,bad_values", [
    ("seed", [-1, True, 1.5]), ("repetitions", [0, -1, True]),
    ("bootstrap_iterations", [0, -1, True, 1.5]),
])
def test_results_invalid_parameters_before_inputs(tmp_path, name, bad_values):
    for value in bad_values:
        with pytest.raises(ValueError, match=name):
            run_results(tmp_path / "absent", tmp_path / "output", **{name: value})


def test_results_existing_output_and_final_release_rejected(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="перезапись"):
        run_results(tmp_path / "absent", output)
    release = tmp_path / "release"
    release.mkdir()
    (release / "metrics.json").write_text(json.dumps({"status": "full_refit_for_review_not_independent_evaluation"}))
    (release / "manifest.json").write_text(json.dumps({"files": {"metrics.json": sha256(release / "metrics.json")}}))
    with pytest.raises(ValueError, match="evaluation, не final release"):
        run_results(release, tmp_path / "results")


@pytest.fixture
def results_evaluation(artifacts, tmp_path):
    baseline, adapter, graph, *_ = artifacts
    evaluation = tmp_path / "evaluation"
    saved = run_refinement(baseline, adapter, graph, evaluation)
    return evaluation, saved, baseline


@pytest.mark.parametrize("artifacts", [10], indirect=True)
def test_results_real_artifacts_clean_parity_determinism_new_days_and_cli(results_evaluation, tmp_path):
    evaluation, saved, baseline = results_evaluation
    before = {str(path): sha256(path) for folder in [evaluation, baseline] for path in folder.iterdir() if path.is_file()}
    actual = run_results(evaluation, tmp_path / "results", repetitions=2, bootstrap_iterations=25)
    assert actual["status"] == "fixed_pool_diagnostic_not_independent_evaluation"
    assert actual["selected_method"] == saved["selected_method"]
    assert actual["fusion"] == {key: saved["fusion"][key] for key in ["weights", "offset"]}
    assert actual["new_days"]["horizon_days"] == 1
    assert actual["test"]["horizon_days"] == 2
    assert actual["new_days"]["clean_comparisons"]["lexical"]["positive_pairs"] > 0
    for split in ["test", "new_days"]:
        assert len(actual[split]["clean_comparisons"]) == 8
        for method, metrics in actual[split]["clean_comparisons"].items():
            assert metrics == pytest.approx(saved[split]["models"][method], abs=1e-12)
        assert set(actual[split]["clean_paired"]) == {"selected_minus_lexical", "selected_minus_co_sales"}
        assert set(actual[split]["scenarios"]) == set(_RESULT_SCENARIOS)
        for name, scenario in actual[split]["scenarios"].items():
            assert scenario["repetitions"] == (1 if name == "clean" else 2)
            assert set(scenario["models"]) == {"selected", "catboost_pu"}
            assert scenario["selected_delta_vs_clean"]["skus"] == saved[split]["models"]["lexical"]["evaluated_skus"]
        clean = actual[split]["scenarios"]["clean"]
        assert clean["selected_delta_vs_clean"]["difference"] == 0.
        assert clean["models"]["selected"]["recall@10"]["std"] == 0.
        distribution = actual[split]["rank_distribution"]
        assert sum(distribution["pair_counts"].values()) == saved[split]["models"]["lexical"]["positive_pairs"]
        assert sum(distribution["macro_fractions"].values()) == pytest.approx(1.)
    assert {path.name for path in (tmp_path / "results").iterdir()} == {"metrics.json", "manifest.json"}
    verify_folder(tmp_path / "results")
    main(["results", "--evaluation", str(evaluation), "--output", str(tmp_path / "cli"),
          "--seed", "7", "--repetitions", "2", "--bootstrap-iterations", "25"])
    assert json.loads((tmp_path / "cli" / "metrics.json").read_text()) == actual
    assert before == {str(path): sha256(path) for folder in [evaluation, baseline] for path in folder.iterdir() if path.is_file()}
    with pytest.raises(ValueError, match="перезапись"):
        run_results(evaluation, tmp_path / "results", repetitions=2, bootstrap_iterations=25)


def test_results_checksum_parent_and_clean_parity_tampering(results_evaluation, tmp_path):
    evaluation, _, baseline = results_evaluation
    for path in [evaluation / "test_candidates.csv", evaluation / "test_positives.csv", baseline / "ranker.cbm"]:
        original = path.read_bytes()
        path.write_bytes(original + b"tamper")
        with pytest.raises(ValueError, match="Контрольная сумма"):
            run_results(evaluation, tmp_path / "results", repetitions=2, bootstrap_iterations=25)
        path.write_bytes(original)
    manifest_path = baseline / "manifest.json"
    original = manifest_path.read_bytes()
    manifest_path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="parent baseline"):
        run_results(evaluation, tmp_path / "results", repetitions=2, bootstrap_iterations=25)
    manifest_path.write_bytes(original)
    metrics_path = evaluation / "metrics.json"
    saved = json.loads(metrics_path.read_text())
    saved["test"]["models"]["lexical"]["recall@10"] += .01
    metrics_path.write_text(json.dumps(saved))
    manifest_path = evaluation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["metrics.json"] = sha256(metrics_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Clean parity"):
        run_results(evaluation, tmp_path / "results", repetitions=2, bootstrap_iterations=25)
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("change", ["input", "output"])
def test_results_precommit_recheck_and_no_concurrent_overwrite(results_evaluation, tmp_path, monkeypatch, change):
    import full_history_recommender as module

    evaluation, _, _ = results_evaluation
    output = tmp_path / "results"
    original_write = module.write_json

    def write_and_change(path, value):
        original_write(path, value)
        if path.name == "metrics.json":
            if change == "input":
                with (evaluation / "test_positives.csv").open("a") as stream:
                    stream.write("tamper")
            else:
                output.mkdir()
                (output / "marker").write_text("keep")

    monkeypatch.setattr(module, "write_json", write_and_change)
    with pytest.raises(ValueError, match="изменился|появился"):
        run_results(evaluation, output, repetitions=2, bootstrap_iterations=25)
    if change == "output":
        assert (output / "marker").read_text() == "keep"
    else:
        assert not output.exists()