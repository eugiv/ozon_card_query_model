import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sample_data import write_dataset
from sklearn.preprocessing import normalize

from recommender import load_config, run_pipeline, verify_pipeline


def small_config(tmp_path):
    facts, dictionary = write_dataset(tmp_path)
    frame = pd.read_csv(facts, dtype={"sku": "string"})
    extra = pd.DataFrame([
        [f"2026-01-{day:02}", "005", query, 100, 5, 1]
        for day in range(1, 9) for query in ["мыло дегтярное", "мыло белое"]
    ], columns=frame.columns)
    frame = pd.concat([frame, extra], ignore_index=True)
    frame["sku"] = frame["sku"].astype(int).astype(str)
    frame.to_csv(facts, index=False)
    catalog = sorted(pd.read_csv(dictionary).normquery.unique())
    # Синтетический encoder input, настоящие обучения PyTorch и CatBoost ниже.
    vectors = normalize(np.random.default_rng(7).normal(size=(len(catalog), 384))).astype(np.float32)
    np.savez_compressed(tmp_path / "vectors.npz", normquery=np.array(catalog), embeddings=vectors,
                        model_name=np.array("synthetic-test-only"), model_revision=np.array("toy"),
                        text_prefix=np.array("query: "),
                        dictionary_as_of=np.array("2025-12-31T23:59:59.999999999+00:00"))
    products = tmp_path / "products.csv"
    pd.DataFrame([
        [str(sku), "Мыло", "Описание товара", "", "[]", "2026-01-08T00:00:00Z", "ok"]
        for sku in range(1, 6)
    ], columns=["sku", "name", "description", "brand", "attributes_json", "snapshot_at", "attributes_status"]).to_csv(products, index=False)
    value = {
        "inputs": {"facts": facts.name, "dictionary": dictionary.name,
                   "embeddings": "vectors.npz", "products": "products.csv"},
        "outputs": {name: f"runs/{name}" for name in
                    ["baseline", "adapter", "graph", "evaluation", "ablation", "final_adapter", "final_graph", "release"]},
        "method": {"dictionary_as_of": "2025-12-31", "first_decision": "2026-01-02",
                   "validation_decision": "2026-01-04", "test_decision": "2026-01-06",
                   "window_days": 2, "per_source": 2, "iterations": [3], "seed": 7},
        "neural": {"adapter_epochs": 2, "graph_epochs": 2, "graph_dimensions": 4, "graph_layers": 2},
        "as_of": "2026-01-09T00:00:00Z", "top_k": 2,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    return path, value


def test_configuration_paths_are_relative_to_config_not_shell(tmp_path, monkeypatch):
    path, _ = small_config(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    config = load_config(path)
    assert config["inputs"]["facts"] == tmp_path / "facts.csv"
    assert config["outputs"]["release"] == tmp_path / "runs/release"
    assert config["method"].window_days == 2


@pytest.mark.parametrize("change", [
    {"top_k": 0}, {"as_of": "2026-01-09"}, {"ignored_parameter": 1},
    {"neural": {"adapter_epochs": 0}},
    {"method": {"dictionary_as_of": "2026-01-05", "first_decision": "2026-01-02",
                "validation_decision": "2026-01-04", "test_decision": "2026-01-06", "window_days": 2}},
])
def test_invalid_configuration_stops_before_work(tmp_path, change):
    path, value = small_config(tmp_path)
    value.update(change)
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, TypeError)):
        load_config(path)
    assert not (tmp_path / "runs").exists()


def test_outputs_must_not_alias_each_other_or_input_files(tmp_path):
    path, value = small_config(tmp_path)
    value["outputs"]["release"] = value["outputs"]["baseline"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="пути"):
        load_config(path)
    value["outputs"]["release"] = "facts.csv"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="пути"):
        load_config(path)


def test_complete_workflow_trains_real_models_and_resumes_without_overwriting(tmp_path):
    path, _ = small_config(tmp_path)
    config = load_config(path)
    result = run_pipeline(config)
    assert result["status"] == "verified"
    assert result["source_rows"] > 0
    assert result["training_rows"] > 0
    assert result["review_pairs"] > 0
    assert result["validation_before_test"] is True
    release = config["outputs"]["release"]
    before = {p.name: p.read_bytes() for p in release.iterdir() if p.is_file()}
    resumed = run_pipeline(config)
    assert resumed == result
    assert before == {p.name: p.read_bytes() for p in release.iterdir() if p.is_file()}
    assert verify_pipeline(config) == result
    assert not (release / "expert_review_template.csv").exists()
    model = config["outputs"]["graph"] / "graph.pt"
    model.write_bytes(model.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="сумм"):
        verify_pipeline(config)
    assert before == {p.name: p.read_bytes() for p in release.iterdir() if p.is_file()}


def test_missing_stage_is_not_silently_reported_as_success(tmp_path):
    path, _ = small_config(tmp_path)
    with pytest.raises((FileNotFoundError, ValueError)):
        verify_pipeline(load_config(path))


def test_changed_configuration_cannot_resume_an_unrelated_experiment(tmp_path):
    path, value = small_config(tmp_path)
    run_pipeline(load_config(path))
    value["method"]["per_source"] = 3
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Параметры существующего baseline отличаются"):
        run_pipeline(load_config(path))


def test_future_final_snapshot_rejected_without_mutating_artifacts(tmp_path):
    path, value = small_config(tmp_path)
    value["as_of"] = "2026-01-08T00:00:00Z"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="as_of"):
        run_pipeline(load_config(path))
    assert not Path(tmp_path / "runs").exists()