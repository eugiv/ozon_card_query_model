import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRanker

import finalize_recommender
from cluster_data import sha256, write_json
from full_history_recommender import refinement_scores, run_refinement
from graph_learning import train_graph_from_files
from semantic_adapter import main as adapt

pytest_plugins = ["test_neural_refinement"]


def _reseal(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    for name in manifest.get("parents", {}):
        parent = Path(name)
        _reseal(parent)
        manifest["parents"][name] = sha256(parent / "manifest.json")
    for field in ["files", "artifact_sha256", "checksums"]:
        for name in manifest.get(field, {}):
            manifest[field][name] = sha256(folder / name)
    for item in manifest.get("inputs", {}).values():
        item["sha256"] = sha256(Path(item["path"]))
    for name in manifest.get("input_sha256", {}):
        manifest["input_sha256"][name] = sha256(Path(name))
    if "products" in manifest:
        manifest["products"]["sha256"] = sha256(Path(manifest["products"]["path"]))
    write_json(folder / "manifest.json", manifest)


def _edit_json(path, change):
    data = json.loads(path.read_text())
    change(data)
    write_json(path, data)


@pytest.fixture
def saved_release(artifacts, tmp_path):
    baseline, adapter, graph, facts, dictionary, *_ = artifacts
    evaluation = tmp_path / "evaluation"
    run_refinement(baseline, adapter, graph, evaluation)
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
    finalize_recommender.run_finalize(evaluation, final_adapter, final_graph, products, release,
                                      "2026-01-09T00:00:00Z")
    return release


def test_saved_release_reproduces_refit_without_training(saved_release, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Inference не должен обучать модели или создавать обучающие пары")

    monkeypatch.setattr(finalize_recommender, "fit_ranker", forbidden)
    monkeypatch.setattr(finalize_recommender, "learning_rows", forbidden)
    before = (saved_release / "ranker.cbm").read_bytes()
    output = tmp_path / "inference"
    summary = finalize_recommender.recommend_from_release(saved_release, output, "2026-01-09T00:00:00Z")
    for name in ["recommendations_for_review.csv", "filtered_candidates.csv"]:
        assert (output / name).read_bytes() == (saved_release / name).read_bytes()
    assert (saved_release / "ranker.cbm").read_bytes() == before
    assert not (output / "ranker.cbm").exists()
    assert not (saved_release / "expert_review_template.csv").exists()
    assert not (output / "expert_review_template.csv").exists()
    assert summary["status"] == "saved_release_inference_no_training"
    assert summary["score_semantics"] == "ranking_priority_not_probability"
    assert summary["training_performed"] is False
    assert json.loads((tmp_path / "final_adapter" / "metrics.json").read_text())["optimizer_steps"] > 0
    graph_audit = json.loads((tmp_path / "final_graph" / "metrics.json").read_text())
    assert graph_audit["epochs_completed"] == 2
    assert graph_audit["parameter_delta_l2"] > 0
    review = pd.read_csv(output / "recommendations_for_review.csv", keep_default_na=False)
    assert not review.empty
    np.testing.assert_allclose(review["history_views"], np.expm1(review["log_views"]))
    assert review["selling_seed"].equals(review["closest_seed"])
    assert review["recommendation_basis"].str.contains("not_verified").all()
    for field in ["source_start", "source_end", "selected_method", "fusion_weights"]:
        assert summary[field] == json.loads((saved_release / "metrics.json").read_text())[field]
    with pytest.raises(ValueError, match="существует"):
        finalize_recommender.recommend_from_release(saved_release, output, "2026-01-09T00:00:00Z")


@pytest.mark.parametrize("folder,field,name", [
    ("baseline", "files", "metrics.json"),
    ("evaluation", "files", "metrics.json"),
    ("final_adapter", "artifact_sha256", "adapter.pt"),
    ("final_adapter", "artifact_sha256", "vectors.npz"),
    ("adapter", "artifact_sha256", "vectors.npz"),
    ("final_graph", "checksums", "catalog.json"),
    ("graph", "checksums", "graph.pt"),
    ("release", "files", "ranker.cbm"),
])
def test_consumed_files_require_manifest_checksums(saved_release, tmp_path, folder, field, name):
    _edit_json(tmp_path / folder / "manifest.json", lambda manifest: manifest[field].pop(name))
    _reseal(saved_release)
    output = tmp_path / "untracked"
    with pytest.raises(ValueError):
        finalize_recommender.recommend_from_release(saved_release, output, "2026-01-09T00:00:00Z")
    assert not output.exists()


@pytest.mark.parametrize("folder,name", [
    ("baseline", "training_pairs.csv"),
    ("evaluation", "validation_candidates.csv"),
    ("evaluation", "test_positives.csv"),
])
def test_refit_consumed_training_files_require_checksums(saved_release, tmp_path, folder, name):
    _edit_json(tmp_path / folder / "manifest.json", lambda manifest: manifest["files"].pop(name))
    _reseal(saved_release)
    with pytest.raises(ValueError, match="обязательные контрольные суммы"):
        finalize_recommender.run_finalize(tmp_path / "evaluation", tmp_path / "final_adapter", tmp_path / "final_graph",
                                          tmp_path / "products.csv", tmp_path / "untracked_refit", "2026-01-09T00:00:00Z")


@pytest.mark.parametrize("name", ["facts", "dictionary", "embeddings"])
def test_required_baseline_inputs(saved_release, tmp_path, name):
    _edit_json(tmp_path / "baseline" / "manifest.json", lambda manifest: manifest["inputs"].pop(name))
    _reseal(saved_release)
    with pytest.raises(ValueError, match="обязательные входы"):
        finalize_recommender.recommend_from_release(saved_release, tmp_path / "missing", "2026-01-09T00:00:00Z")


@pytest.mark.parametrize("name", [
    "release/ranker.cbm", "evaluation/metrics.json", "final_graph/graph.pt",
    "final_adapter/vectors.npz", "facts.csv", "products.csv", "final_adapter/manifest.json",
])
def test_recursive_checksum_tampering(saved_release, tmp_path, name):
    path = tmp_path / name
    path.write_bytes(path.read_bytes() + b"\nchanged")
    output = tmp_path / "tampered"
    with pytest.raises(ValueError, match="Контрольная сумма"):
        finalize_recommender.recommend_from_release(saved_release, output, "2026-01-09T00:00:00Z")
    assert not output.exists()


@pytest.mark.parametrize("folder,field,value", [
    ("final_adapter", "training_cutoff", "2026-01-10"),
    ("final_adapter", "training_cutoff", None),
    ("final_adapter", "dictionary_as_of", "2025-12-30T23:59:59.999999999Z"),
    ("final_graph", "train_end", "2026-01-07T23:59:59.999999999Z"),
    ("final_graph", "dictionary_as_of", "2025-12-30T23:59:59.999999999Z"),
    ("adapter", "training_cutoff", "2026-01-05"),
])
def test_inconsistent_model_dates(saved_release, tmp_path, folder, field, value):
    _edit_json(tmp_path / folder / "manifest.json", lambda manifest: manifest.update({field: value}))
    _reseal(saved_release)
    with pytest.raises(ValueError):
        finalize_recommender.recommend_from_release(saved_release, tmp_path / "dates", "2026-01-09T00:00:00Z")


@pytest.mark.parametrize("cutoff", ["2026-01-07", "2026-01-10"])
def test_consistently_saved_adapter_must_end_at_last_facts(saved_release, tmp_path, cutoff):
    folder = tmp_path / "final_adapter"
    adapt(["--facts", str(tmp_path / "facts.csv"), "--dictionary", str(tmp_path / "dictionary.csv"),
           "--embeddings", str(tmp_path / "base.npz"), "--dictionary-as-of", "2025-12-31",
           "--train-end", cutoff, "--epochs", "2", "--output", str(tmp_path / "other_adapter")])
    _edit_json(saved_release / "manifest.json", lambda manifest: manifest["parents"].pop(str(folder)))
    _edit_json(saved_release / "manifest.json", lambda manifest: manifest["parents"].update(
        {str(tmp_path / "other_adapter"): sha256(tmp_path / "other_adapter" / "manifest.json")}))
    _reseal(saved_release)
    with pytest.raises(ValueError, match="последний день фактов|позже as_of"):
        finalize_recommender.recommend_from_release(saved_release, tmp_path / "adapter_date", "2026-01-09T00:00:00Z")


def test_consistently_saved_graph_must_end_at_last_facts(saved_release, tmp_path):
    folder = tmp_path / "other_graph"
    train_graph_from_files(facts=tmp_path / "facts.csv", dictionary=tmp_path / "dictionary.csv",
                           dictionary_as_of="2025-12-31", train_end="2026-01-07", output=folder,
                           epochs=2, dimensions=4)
    _edit_json(saved_release / "manifest.json", lambda manifest: manifest["parents"].pop(str(tmp_path / "final_graph")))
    _edit_json(saved_release / "manifest.json", lambda manifest: manifest["parents"].update({str(folder): sha256(folder / "manifest.json")}))
    _reseal(saved_release)
    with pytest.raises(ValueError, match="последний день фактов"):
        finalize_recommender.recommend_from_release(saved_release, tmp_path / "graph_date", "2026-01-09T00:00:00Z")


def test_as_of_requires_last_fact_day_to_be_complete(saved_release, tmp_path):
    with pytest.raises(ValueError, match="не завершённый день"):
        finalize_recommender.recommend_from_release(saved_release, tmp_path / "incomplete", "2026-01-08T23:59:59Z")


@pytest.mark.parametrize("change,match", [
    ("future_facts", "не завершённый день"),
    ("missing_window", "Неполное окно"),
    ("future_products", "snapshot_at > as_of"),
    ("naive_products", "timezone"),
    ("embeddings_snapshot", "Снимок"),
    ("graph_catalog", "Каталог"),
])
def test_temporal_and_catalog_inputs(saved_release, tmp_path, change, match):
    if change in {"future_facts", "missing_window"}:
        path = tmp_path / "facts.csv"
        facts = pd.read_csv(path, dtype={"sku": "string"})
        if change == "future_facts":
            facts.loc[facts["date"].eq("2026-01-08"), "date"] = "2026-01-09"
        else:
            facts = facts.loc[~facts["date"].eq("2026-01-07")]
        facts.to_csv(path, index=False)
    elif change in {"future_products", "naive_products"}:
        path = tmp_path / "products.csv"
        products = pd.read_csv(path, keep_default_na=False)
        products.loc[0, "snapshot_at"] = "2026-01-10T00:00:00Z" if change == "future_products" else "2026-01-08"
        products.to_csv(path, index=False)
    elif change == "embeddings_snapshot":
        path = tmp_path / "base.npz"
        with np.load(path, allow_pickle=False) as stored:
            values = {name: stored[name] for name in stored.files}
        values["dictionary_as_of"] = np.array("2025-12-30T23:59:59.999999999+00:00")
        np.savez_compressed(path, **values)
    else:
        _edit_json(tmp_path / "final_graph" / "catalog.json", lambda data: data["normquery"].append("новый кластер"))
        _edit_json(tmp_path / "final_graph" / "metrics.json", lambda data: data.update(
            catalog_queries=data["catalog_queries"] + 1))
    _reseal(saved_release)
    output = tmp_path / "invalid_inputs"
    with pytest.raises(ValueError, match=match):
        finalize_recommender.recommend_from_release(saved_release, output, "2026-01-09T00:00:00Z")
    assert not output.exists()


@pytest.mark.parametrize("folder", ["final_adapter", "final_graph"])
def test_other_model_input_rejected(saved_release, tmp_path, folder):
    other = tmp_path / "other_facts.csv"
    other.write_bytes((tmp_path / "facts.csv").read_bytes())
    if folder == "final_adapter":
        _edit_json(tmp_path / folder / "manifest.json", lambda manifest: manifest["input_sha256"].update(
            {str(other): manifest["input_sha256"].pop(str(tmp_path / "facts.csv"))}))
    else:
        _edit_json(tmp_path / folder / "manifest.json", lambda manifest: manifest["inputs"]["facts"].update(path=str(other)))
    _reseal(saved_release)
    with pytest.raises(ValueError, match="других входных"):
        finalize_recommender.recommend_from_release(saved_release, tmp_path / "other_inputs", "2026-01-09T00:00:00Z")


@pytest.mark.parametrize("name", ["../outside.txt", "/outside.txt", "nested/file.txt", "..\\outside.txt"])
def test_artifact_path_traversal_rejected(saved_release, tmp_path, name):
    _edit_json(saved_release / "manifest.json", lambda manifest: manifest["files"].update({name: "0" * 64}))
    with pytest.raises(ValueError, match="путь|Путь"):
        finalize_recommender.recommend_from_release(saved_release, tmp_path / "traversal", "2026-01-09T00:00:00Z")


def test_symlink_artifact_rejected(saved_release, tmp_path):
    path = tmp_path / "final_graph" / "graph.pt"
    copy = tmp_path / "graph_copy.pt"
    copy.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(copy)
    with pytest.raises(ValueError, match="Символические ссылки"):
        finalize_recommender.recommend_from_release(saved_release, tmp_path / "symlink", "2026-01-09T00:00:00Z")


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
def test_inference_budget_validation_before_loading(tmp_path, limit):
    with pytest.raises(ValueError, match="top_k"):
        finalize_recommender.recommend_from_release(tmp_path / "missing", tmp_path / "output", "2026-01-09T00:00:00Z", limit)


@pytest.mark.parametrize("as_of", [None, "", "2026-01-09", "2026-01-09T00:00:00"])
def test_inference_requires_explicit_timezone_before_loading(tmp_path, as_of):
    with pytest.raises(ValueError, match="timezone"):
        finalize_recommender.recommend_from_release(tmp_path / "missing", tmp_path / "output", as_of)


@pytest.mark.parametrize("method", ["catboost_pu", "learned_fusion"])
def test_actual_saved_score_channels_reproduce_refit(saved_release, tmp_path, method):
    _edit_json(tmp_path / "evaluation" / "metrics.json", lambda metrics: metrics.update(selected_method=method))
    _reseal(saved_release)
    release = tmp_path / f"release_{method}"
    finalize_recommender.run_finalize(tmp_path / "evaluation", tmp_path / "final_adapter", tmp_path / "final_graph",
                                      tmp_path / "products.csv", release, "2026-01-09T00:00:00Z")
    output = tmp_path / f"inference_{method}"
    finalize_recommender.main(["--release", str(release), "--output", str(output), "--as-of", "2026-01-09T03:00:00+03:00"])
    for name in ["recommendations_for_review.csv", "filtered_candidates.csv"]:
        assert (release / name).read_bytes() == (output / name).read_bytes()
    model = CatBoostRanker()
    model.load_model(str(release / "ranker.cbm"))
    assert model.tree_count_ == 3
    prepared = finalize_recommender._prepare(tmp_path / "evaluation", tmp_path / "final_adapter", tmp_path / "final_graph",
                                             tmp_path / "products.csv", pd.Timestamp("2026-01-09", tz="UTC"), {}, {})
    candidates, _ = finalize_recommender._candidates(prepared)
    scores = refinement_scores(candidates, model, prepared.evaluation_metrics["fusion"])[method]
    expected = candidates.assign(expected_score=scores)
    actual = pd.read_csv(output / "recommendations_for_review.csv", dtype={"sku": "string"})
    joined = actual.merge(expected[["sku", "normquery", "expected_score"]], on=["sku", "normquery"], validate="one_to_one")
    assert len(joined) == len(actual) > 0
    np.testing.assert_allclose(joined["score"], joined["expected_score"], rtol=1e-12, atol=1e-15)


def test_seo_blocks_before_budget_and_empty_review(saved_release, tmp_path):
    path = tmp_path / "products.csv"
    products = pd.read_csv(path, keep_default_na=False, dtype={"sku": "string"})
    products["description"] = "мыло ланолиновое"
    products["brand"] = "Acme"
    extra = products.iloc[[0]].copy()
    extra["sku"], extra["brand"] = "999", "белое"
    products = pd.concat([products, extra], ignore_index=True)
    products.to_csv(path, index=False)
    _reseal(saved_release)
    output = tmp_path / "blocked"
    summary = finalize_recommender.recommend_from_release(saved_release, output, "2026-01-09T00:00:00Z", top_k=1)
    filtered = pd.read_csv(output / "filtered_candidates.csv")
    assert filtered["decision"].eq("already_present").any()
    assert filtered["compatibility"].eq("brand_conflict").any()
    review = pd.read_csv(output / "recommendations_for_review.csv", dtype={"sku": "string"})
    assert not review.empty
    assert review["rank"].eq(1).all()
    full = tmp_path / "all_review"
    finalize_recommender.recommend_from_release(saved_release, full, "2026-01-09T00:00:00Z", top_k=100)
    expected = pd.read_csv(full / "recommendations_for_review.csv", dtype={"sku": "string"}).groupby("sku", sort=False).head(1)
    pd.testing.assert_frame_equal(review, expected.reset_index(drop=True))
    assert summary["review_pairs"] == len(review)
    products["brand"] = ""
    products["description"] = "мыло ланолиновое; мыло невская; мыло дегтярное; мыло белое; мыло детское"
    products.to_csv(path, index=False)
    _reseal(saved_release)
    empty = tmp_path / "empty_review"
    summary = finalize_recommender.recommend_from_release(saved_release, empty, "2026-01-09T00:00:00Z")
    assert summary["review_pairs"] == summary["review_skus"] == 0
    assert summary["candidate_pairs"] > 0
    assert pd.read_csv(empty / "recommendations_for_review.csv").empty
    assert len(pd.read_csv(empty / "filtered_candidates.csv")) == summary["candidate_pairs"]


def test_full_unmasked_history_is_preserved(saved_release, tmp_path):
    prepared = finalize_recommender._prepare(tmp_path / "evaluation", tmp_path / "final_adapter", tmp_path / "final_graph",
                                             tmp_path / "products.csv", pd.Timestamp("2026-01-09", tz="UTC"), {}, {})
    candidates, _ = finalize_recommender._candidates(prepared)
    assert prepared.daily["date"].min() == pd.Timestamp("2026-01-01")
    assert prepared.daily["sku"].nunique() == prepared.facts["sku"].nunique() == 5
    historical_sales = prepared.daily.loc[prepared.daily["orders"].gt(0), ["sku", "normquery"]].drop_duplicates()
    old_only = prepared.daily.loc[prepared.daily["date"].le(pd.Timestamp("2026-01-04")) & prepared.daily["orders"].gt(0), ["sku", "normquery"]]
    recent_sales = prepared.daily.loc[prepared.daily["date"].ge(pd.Timestamp("2026-01-07")) & prepared.daily["orders"].gt(0), ["sku", "normquery"]]
    assert not old_only.merge(recent_sales, how="left", on=["sku", "normquery"], indicator=True).loc[lambda frame: frame["_merge"].eq("left_only")].empty
    assert candidates.merge(historical_sales, on=["sku", "normquery"]).empty
    assert not candidates.empty
    model = CatBoostRanker()
    model.load_model(str(saved_release / "ranker.cbm"))
    annotated, shortlist = finalize_recommender._queue(candidates.iloc[:0], model, prepared, pd.Timestamp("2026-01-09", tz="UTC"), 10)
    assert annotated.empty and shortlist.empty
    assert "history_views" in annotated and "recommendation_basis" in annotated


def test_cli_help_and_modes(capsys, tmp_path):
    with pytest.raises(SystemExit) as help_exit:
        finalize_recommender.main(["--help"])
    assert help_exit.value.code == 0
    assert "--release" in capsys.readouterr().out
    for args in [[], ["--release", "release", "--evaluation", "evaluation"]]:
        with pytest.raises(SystemExit) as invalid:
            finalize_recommender.main([*args, "--output", str(tmp_path / "output"), "--as-of", "2026-01-09T00:00:00Z"])
        assert invalid.value.code == 2
    assert "--release нельзя сочетать" in capsys.readouterr().err