import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import graph_learning
from cluster_data import sha256
from graph_learning import (
    _adjacency,
    _propagate,
    _sample_unknown,
    fit_graph,
    load_graph,
    main,
    save_graph,
    train_graph_from_files,
)


def history():
    return pd.DataFrame(
        [["1", "a", 2], ["1", "b", 1], ["2", "b", 1], ["2", "c", 3],
         ["3", "c", 1], ["3", "d", 1], ["zero", "cold", 0], ["1", "a", 4]],
        columns=["sku", "normquery", "orders"],
    )


def catalog():
    return pd.DataFrame({"normquery": ["d", "cold", "b", "a", "c"]})


@pytest.fixture
def trained():
    return fit_graph(history(), catalog(), epochs=8, dimensions=8)


def test_real_parameters_change_losses_are_finite_and_nodes_are_only_sold(trained):
    model, audit = trained
    initial = torch.randn((7, 8), generator=torch.Generator().manual_seed(7)) * 0.1
    assert not np.allclose(initial.numpy(), model.initial_embeddings)
    assert audit["parameter_delta_l2"] > 0.1
    assert audit["trained"] and audit["epochs_completed"] == 8
    assert len(audit["losses"]) == 8 and np.isfinite(audit["losses"]).all()
    assert audit["losses"][-1] < audit["losses"][0]
    assert audit["edge_visits"] == 8 * 6
    assert audit["sold_edges"] == 6 and audit["sold_input_rows"] == 7
    assert model.skus == ("1", "2", "3")
    assert model.queries == ("a", "b", "c", "d")
    assert model.query_embeddings.shape == (4, 8)
    assert audit["cold_catalog_queries"] == 1
    assert torch.get_num_threads() == 2
    assert all(np.isfinite(a).all() for a in [model.initial_embeddings, model.query_embeddings, model.sku_embeddings])


def test_cpu_determinism_and_input_order_invariance(trained):
    model, audit = trained
    again, again_audit = fit_graph(history().sample(frac=1, random_state=11),
                                   catalog().iloc[::-1], epochs=8, dimensions=8)
    np.testing.assert_allclose(model.initial_embeddings, again.initial_embeddings, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(model.query_embeddings, again.query_embeddings, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(audit["losses"], again_audit["losses"], rtol=1e-6, atol=1e-7)


def test_sparse_bipartite_normalization_and_layer_average(trained):
    users = np.array([0, 0, 1, 1, 2, 2])
    items = np.array([0, 1, 1, 2, 2, 3])
    sparse = _adjacency(torch, users, items, 3, 4)
    assert sparse.is_sparse and sparse.device.type == "cpu"
    dense = np.zeros((7, 7), dtype=np.float32)
    for u, q in zip(users, items):
        dense[u, q + 3] = dense[q + 3, u] = 1
    degree = dense.sum(axis=1)
    expected = dense / np.sqrt(degree[:, None] * degree[None, :])
    np.testing.assert_allclose(sparse.to_dense().numpy(), expected, atol=1e-7)
    model, _ = trained
    initial = model.initial_embeddings
    average = (initial + expected @ initial + expected @ expected @ initial) / 3
    np.testing.assert_allclose(np.concatenate([model.sku_embeddings, model.query_embeddings]),
                               average, atol=2e-7)
    propagated = _propagate(torch, sparse, torch.from_numpy(initial), 0)
    np.testing.assert_array_equal(propagated.numpy(), initial)


def test_complement_sampler_never_returns_known_edge_including_dense_skus():
    for item_count in range(1, 7):
        for degree in range(item_count + 1):
            for positives in itertools.combinations(range(item_count), degree):
                rng = np.random.default_rng(7)
                sampled = _sample_unknown(np.array(positives), item_count, rng, 128)
                unknown = set(range(item_count)) - set(positives)
                assert set(sampled).issubset(unknown)
                assert len(sampled) == (128 if unknown else 0)
                if unknown:
                    assert set(sampled) == unknown
    np.testing.assert_array_equal(
        _sample_unknown(np.array([0, 2, 3]), 4, np.random.default_rng(7), 5), [1] * 5
    )


def test_all_edges_participate_each_epoch_with_real_sampler(monkeypatch):
    original = graph_learning._sample_unknown
    calls = []

    def record(positives, item_count, rng, size):
        result = original(positives, item_count, rng, size)
        calls.append((tuple(positives), tuple(result)))
        return result

    monkeypatch.setattr(graph_learning, "_sample_unknown", record)
    _, audit = fit_graph(history(), catalog(), epochs=3, dimensions=4)
    assert len(calls) == 3 * 3
    for epoch in range(3):
        positives = [p for p, _ in calls[epoch * 3:(epoch + 1) * 3]]
        assert positives == [(0, 1), (1, 2), (2, 3)]
    assert all(not set(p) & set(n) and len(p) == len(n) for p, n in calls)
    assert audit["eligible_edges_per_epoch"] == 6


def test_exact_inverse_degree_bpr_and_l2_on_initial_layer():
    data = pd.DataFrame([["1", "a", 1], ["2", "b", 1], ["2", "c", 1]],
                        columns=["sku", "normquery", "orders"])
    _, audit = fit_graph(data, ["a", "b", "c"], epochs=1, dimensions=5, layers=1)
    initial = torch.randn((5, 5), generator=torch.Generator().manual_seed(7)) * 0.1
    adjacency = _adjacency(torch, np.array([0, 1, 1]), np.array([0, 1, 2]), 2, 3)
    embedding = (initial + adjacency.to_dense() @ initial) / 2
    rng = np.random.default_rng(7)
    negative = np.concatenate([_sample_unknown(np.array([0]), 3, rng, 1),
                               _sample_unknown(np.array([1, 2]), 3, rng, 2)])
    margin = (embedding[[0, 1, 1]] * (embedding[[2, 3, 4]] - embedding[negative + 2])).sum(dim=1)
    bpr = (torch.nn.functional.softplus(-margin) * torch.tensor([0.5, 0.25, 0.25])).sum()
    loss = bpr + graph_learning.L2_REGULARIZATION * initial.square().sum() / 5
    assert audit["bpr_losses"][0] == pytest.approx(float(bpr), abs=1e-7)
    assert audit["losses"][0] == pytest.approx(float(loss), abs=1e-7)


def test_default_30_epoch_fit_is_real_on_tiny_data():
    model, audit = fit_graph(history(), catalog())
    assert audit["epochs_completed"] == 30 and audit["edge_visits"] == 180
    assert model.query_embeddings.shape == (4, 32)
    assert audit["losses"][-1] < audit["losses"][0]


def test_seed_fold_in_cosine_is_independent_of_sku_embeddings(trained):
    model, audit = trained
    scores, available = model.score_from_seeds(["a", "b", "cold", "a"], catalog())
    unit = model.query_embeddings / np.linalg.norm(model.query_embeddings, axis=1, keepdims=True)
    profile = unit[:2].mean(axis=0)
    expected = unit @ (profile / np.linalg.norm(profile))
    np.testing.assert_allclose(scores[[3, 2, 4, 0]], expected, atol=1e-7)
    assert available.tolist() == [True, False, True, True, True]
    assert scores[1] == 0 and not audit["fold_in_uses_sku_embedding"]
    model.sku_embeddings[:] = 999
    repeated, repeated_mask = model.score_from_seeds(["b", "a"], catalog())
    np.testing.assert_array_equal(scores, repeated)
    np.testing.assert_array_equal(available, repeated_mask)


@pytest.mark.parametrize("seeds", [[], ["cold"], ["does-not-exist"]])
def test_empty_or_unknown_seeds_are_unavailable(trained, seeds):
    model, _ = trained
    scores, available = model.score_from_seeds(seeds, catalog())
    assert not scores.any() and not available.any()


def test_cold_items_and_cold_skus_are_zero_not_negative_relevance(trained):
    model, _ = trained
    scores, available = model.score("1", ["cold", "a", "unseen"])
    assert scores[0] == scores[2] == 0
    assert available.tolist() == [False, True, False]
    scores, available = model.score("zero", catalog())
    assert not scores.any() and not available.any()
    scores, available = model.score_from_seeds(["a"], [])
    assert scores.shape == available.shape == (0,)


def test_all_item_sku_is_in_graph_but_skipped_by_bpr():
    data = pd.DataFrame([["all", "a", 1], ["all", "b", 1], ["some", "a", 1]],
                        columns=["sku", "normquery", "orders"])
    model, audit = fit_graph(data, ["a", "b", "cold"], epochs=2, dimensions=4)
    assert model.skus == ("all", "some")
    assert audit["skus_without_unknown_items"] == ["all"]
    assert audit["skipped_edges_per_epoch"] == 2
    assert audit["eligible_edges_per_epoch"] == 1 and audit["edge_visits"] == 2
    assert audit["trained"]
    only, audit = fit_graph(data.loc[data["sku"] == "all"], ["a", "b", "cold"], epochs=2, dimensions=4)
    assert not audit["trained"] and audit["epochs_completed"] == 0
    assert audit["losses"] == [] and audit["parameter_delta_l2"] == 0
    assert only.queries == ("a", "b")


@pytest.mark.parametrize("data", [pd.DataFrame(columns=["sku", "normquery", "orders"]),
                                  pd.DataFrame([["zero", "cold", 0]], columns=["sku", "normquery", "orders"])])
def test_no_sold_edges_is_finite_and_not_claimed_trained(data, tmp_path):
    model, audit = fit_graph(data, ["cold"], epochs=2, dimensions=4)
    assert model.skus == model.queries == ()
    assert model.initial_embeddings.shape == (0, 4)
    assert audit["cold_catalog_queries"] == 1 and not audit["trained"]
    assert audit["edge_visits"] == 0
    loaded = load_graph(save_graph(model, tmp_path / "empty"))
    scores, available = loaded.score_from_seeds(["cold"], ["cold"])
    assert scores.tolist() == [0] and available.tolist() == [False]


def test_roundtrip_safe_tensors_manifest_and_immutable_output(trained, tmp_path):
    model, audit = trained
    path = save_graph(model, tmp_path / "run")
    payload = torch.load(path / "graph.pt", weights_only=True, map_location="cpu")
    assert payload["query_embeddings"].shape == (4, 8)
    manifest = json.loads((path / "manifest.json").read_text())
    assert set(manifest["checksums"]) == {"graph.pt", "catalog.json", "metrics.json"}
    assert set(manifest["packages"]) == {"numpy", "torch", "pandas"}
    assert "graph_learning.py" in manifest["code_sha256"]
    loaded = load_graph(path)
    assert loaded.audit == audit and loaded.query_to_index == model.query_to_index
    for method, argument in [("score", "1"), ("score_from_seeds", ["a", "b"])]:
        before = getattr(model, method)(argument, catalog())
        after = getattr(loaded, method)(argument, catalog())
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)
    with pytest.raises(FileExistsError):
        save_graph(model, path)
    (path / "metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="Checksum"):
        load_graph(path)


@pytest.mark.parametrize("corruption", ["nan", "shape", "mapping", "cutoff"])
def test_loader_validates_even_when_checksum_is_updated(trained, tmp_path, corruption):
    model, _ = trained
    path = save_graph(model, tmp_path / "bad")
    payload = torch.load(path / "graph.pt", weights_only=True)
    if corruption == "nan":
        payload["query_embeddings"][0, 0] = float("nan")
    elif corruption == "shape":
        payload["query_embeddings"] = payload["query_embeddings"][:, :1]
    elif corruption == "mapping":
        payload["queries"][0] = payload["queries"][1]
    else:
        payload["train_end"] = "2099-01-01"
    torch.save(payload, path / "graph.pt")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["checksums"]["graph.pt"] = sha256(path / "graph.pt")
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_graph(path)


def test_aggregate_date_provenance_is_honest_and_future_dates_rejected():
    data = history()
    _, audit = fit_graph(data, catalog(), epochs=1, dimensions=4)
    assert audit["train_end"] is None and audit["source_date_max"] is None
    assert not audit["cutoff_verified"]
    data.attrs = {"train_end": "2026-01-31", "source_date_min": "2025-01-01",
                  "source_date_max": "2026-01-30"}
    model, audit = fit_graph(data, catalog(), epochs=1, dimensions=4)
    assert model.train_end.startswith("2026-01-31") and audit["cutoff_verified"]
    assert audit["source_date_min"].startswith("2025-01-01")
    data.attrs["source_date_max"] = "2026-02-01"
    with pytest.raises(ValueError, match="будущие"):
        fit_graph(data, catalog(), epochs=1)
    data.attrs = {}
    data["date"] = "2026-02-01"
    with pytest.raises(ValueError, match="будущие"):
        fit_graph(data, catalog(), epochs=1, train_end="2026-01-31")


def files(tmp_path, *, future=True):
    data = pd.DataFrame(
        [["2025-01-01", "001", "qa", 10, 0, 1],
         ["2026-01-31", "002", "qb", 10, 0, 1],
         ["2026-02-01", "future-sku", "qc", 10, 30, 50]],
        columns=["date", "sku", "query", "unique_search_users", "unique_view_users", "order_count"],
    )
    dictionary = pd.DataFrame(
        [["qa", "a", "2025-01-01", 200, ""], ["qb", "b", "2025-01-01", 200, ""],
         ["qc", "cold", "2025-01-01", 200, ""], ["qa", "future-cluster", "2026-02-01", 200, ""]],
        columns=["query", "normquery", "loaded_at", "status_code", "error"],
    )
    if not future:
        data = data.iloc[:-1]
        dictionary = dictionary.iloc[:-1]
    fact_path = tmp_path / ("facts.csv" if future else "past.csv")
    dict_path = tmp_path / ("dict.csv" if future else "past_dict.csv")
    data.to_csv(fact_path, index=False)
    dictionary.to_csv(dict_path, index=False)
    return {"facts": fact_path, "dictionary": dict_path, "dictionary_as_of": "2026-01-31",
            "train_end": "2026-01-31", "epochs": 3, "dimensions": 4, "layers": 2}


def test_file_training_all_past_cutoff_before_aggregate_and_future_isolation(tmp_path):
    args = files(tmp_path)
    model, audit = train_graph_from_files(**args, output=tmp_path / "full")
    past, _ = train_graph_from_files(**files(tmp_path, future=False), output=tmp_path / "past_run")
    np.testing.assert_allclose(model.initial_embeddings, past.initial_embeddings, atol=1e-7)
    assert model.skus == ("001", "002") and model.queries == ("a", "b")
    assert audit["source_date_min"].startswith("2025-01-01")
    assert audit["source_date_max"].startswith("2026-01-31")
    assert audit["cutoff_verified"] and audit["future_fact_rows_ignored"] == 1
    assert audit["facts_input_rows"] == 3 and audit["facts_train_rows"] == 2
    assert audit["dictionary_audit"]["future_rows_ignored"] == 1
    assert audit["aggregation_audit"]["input_rows"] == 2
    assert model.audit == audit
    assert model.catalog_queries == ("a", "b", "cold")
    assert json.loads((tmp_path / "full" / "metrics.json").read_text()) == audit
    manifest = json.loads((tmp_path / "full" / "manifest.json").read_text())
    assert manifest["inputs"]["facts"]["sha256"] == sha256(args["facts"])
    assert manifest["dictionary_as_of"] == audit["dictionary_as_of"]
    assert load_graph(tmp_path / "full").train_end == model.train_end
    assert load_graph(tmp_path / "full").audit == audit
    with pytest.raises(FileExistsError):
        train_graph_from_files(**args, output=tmp_path / "full")


def test_file_training_preserves_the_configured_seed(tmp_path):
    args = files(tmp_path)
    args.update(epochs=1, dimensions=4)
    model, audit = train_graph_from_files(**args, output=tmp_path / "seeded-graph", seed=23)
    assert audit["seed"] == 23
    assert model.audit["seed"] == 23


def test_literal_nan_query_is_not_a_missing_csv_value(tmp_path):
    args = files(tmp_path)
    dictionary = pd.read_csv(args["dictionary"], keep_default_na=False)
    records = dictionary.iloc[:1].copy()
    records["query"] = "nan"
    records["normquery"] = "nan"
    pd.concat([dictionary, records], ignore_index=True).to_csv(args["dictionary"], index=False)
    model, _ = train_graph_from_files(**args, output=tmp_path / "literal-query")
    assert "nan" in model.catalog_queries
    assert "nan" not in model.queries


def test_cli_arguments_and_reject_future_dictionary_and_existing_output(tmp_path, capsys):
    args = files(tmp_path)
    output = tmp_path / "cli"
    argv = ["--facts", str(args["facts"]), "--dictionary", str(args["dictionary"]),
            "--dictionary-as-of", "2026-01-31", "--train-end", "2026-01-31",
            "--output", str(output), "--epochs", "2", "--dimensions", "4", "--layers", "1"]
    assert main(argv) == 0
    assert load_graph(output).audit["epochs_completed"] == 2
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
    argv[argv.index("--dictionary-as-of") + 1] = "2026-02-01"
    argv[argv.index("--output") + 1] = str(tmp_path / "future")
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2 and not (tmp_path / "future").exists()
    assert "dictionary-as-of" in capsys.readouterr().err


@pytest.mark.parametrize("column,value", [("orders", -1), ("orders", float("nan")),
                                           ("orders", float("inf")), ("sku", ""), ("normquery", None)])
def test_invalid_history_rejected(column, value):
    data = history()
    if column == "orders":
        data[column] = data[column].astype(float)
    data.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        fit_graph(data, catalog(), epochs=1)


def test_import_and_cli_help_do_not_import_torch():
    source = Path(__file__).resolve().parents[1] / "src"
    code = f"import sys; sys.path.insert(0, {str(source)!r}); import graph_learning; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, str(source / "graph_learning.py"), "--help"], capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0 and "--train-end" in result.stdout


@pytest.mark.parametrize("best_epoch", [0, 1])
def test_validation_restores_best_weights_including_epoch_zero(best_epoch):
    snapshots, scores = [], []

    def validate(model):
        assert not torch.is_grad_enabled()
        epoch = len(snapshots)
        assert model.audit["epochs_completed"] == epoch
        snapshots.append((model.initial_embeddings.copy(), model.sku_embeddings.copy(),
                          model.query_embeddings.copy()))
        scores.append(model.score_from_seeds(["a"], catalog()))
        expected = torch.from_numpy(model.initial_embeddings)
        adjacency = _adjacency(torch, np.array([0, 0, 1, 1, 2, 2]),
                               np.array([0, 1, 1, 2, 2, 3]), 3, 4)
        propagated = _propagate(torch, adjacency, expected, 2).numpy()
        np.testing.assert_array_equal(model.query_embeddings, propagated[3:])
        vectors, available = model.query_vectors(model.queries)
        np.testing.assert_allclose(vectors, propagated[3:] / np.linalg.norm(propagated[3:], axis=1, keepdims=True))
        assert available.all()
        if epoch:
            assert not np.array_equal(snapshots[-1][0], snapshots[-2][0])
            assert not np.array_equal(scores[-1][0], scores[-2][0])
        return 1.0 if epoch == best_epoch else 0.0

    model, audit = fit_graph(history(), catalog(), epochs=3, dimensions=4,
                             validation_callback=validate, selection_metric="NDCG@5")
    assert audit["best_epoch"] == best_epoch and audit["best_validation_score"] == 1.0
    assert audit["validation_per_epoch"] == [float(epoch == best_epoch) for epoch in range(4)]
    assert audit["epochs_completed"] == audit["optimizer_steps"] == 3
    assert audit["edge_visits"] == 18
    assert not audit["stopped_early"] and audit["selection_metric"] == "NDCG@5"
    for actual, expected in zip((model.initial_embeddings, model.sku_embeddings, model.query_embeddings),
                                snapshots[best_epoch]):
        np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(model.initial_embeddings, snapshots[-1][0])
    for actual, expected in zip(model.score_from_seeds(["a"], catalog()), scores[best_epoch]):
        np.testing.assert_array_equal(actual, expected)
    assert audit["parameter_delta_l2"] == pytest.approx(
        np.linalg.norm(snapshots[best_epoch][0] - snapshots[0][0]), abs=1e-7)


@pytest.mark.parametrize("scores,min_delta,best_epoch", [
    ([0.5, 0.5, 0.5], 0.0, 0),
    ([0.0, 0.5, 0.5, 0.75, 0.75, 0.75], 0.0, 3),
    ([0.0, 0.0625, 0.125], 0.125, 2),
])
def test_validation_plateau_stops_and_counts_actual_work(scores, min_delta, best_epoch):
    values = iter(scores)
    _, audit = fit_graph(history(), catalog(), epochs=10, dimensions=4,
                         validation_callback=lambda model: next(values), patience=2, min_delta=min_delta)
    completed = len(scores) - 1
    assert audit["validation_per_epoch"] == scores
    assert audit["epochs_completed"] == audit["optimizer_steps"] == completed
    assert len(audit["losses"]) == len(audit["bpr_losses"]) == completed
    assert audit["edge_visits"] == completed * 6
    assert audit["best_epoch"] == best_epoch and audit["best_validation_score"] == max(scores)
    assert audit["stopped_early"] and audit["epochs_requested"] == 10


def test_validation_default_preserves_training_and_final_epoch_selection():
    model, audit = fit_graph(history(), catalog(), epochs=3, dimensions=4)
    explicit, explicit_audit = fit_graph(history(), catalog(), epochs=3, dimensions=4,
                                        validation_callback=None, patience=None, min_delta=0.0)
    values = iter([0.0, 0.25, 0.5, 0.75])
    selected, selected_audit = fit_graph(history(), catalog(), epochs=3, dimensions=4,
                                        validation_callback=lambda current: next(values))
    assert audit == explicit_audit
    assert audit["validation_per_epoch"] == [] and audit["selection_metric"] is None
    assert audit["best_epoch"] is audit["best_validation_score"] is None
    assert audit["epochs_completed"] == audit["optimizer_steps"] == 3 and not audit["stopped_early"]
    assert selected_audit["losses"] == audit["losses"] and selected_audit["best_epoch"] == 3
    for candidate in [explicit, selected]:
        np.testing.assert_array_equal(model.initial_embeddings, candidate.initial_embeddings)
        np.testing.assert_array_equal(model.query_embeddings, candidate.query_embeddings)


@pytest.mark.parametrize("options", [{"validation_callback": 42}, {"patience": 0},
                                     {"patience": 2}, {"min_delta": -1}, {"seed": 2**63}])
def test_invalid_validation_parameters_rejected(options):
    with pytest.raises(ValueError):
        fit_graph(history(), catalog(), epochs=1, **options)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), None, True, "0.5"])
def test_invalid_validation_metric_rejected(score):
    with pytest.raises(ValueError, match="validation_callback.*epoch 0.*finite float"):
        fit_graph(history(), catalog(), epochs=1, validation_callback=lambda model: score)


@pytest.mark.parametrize("error_epoch", [0, 1])
def test_callback_error_reports_epoch_and_preserves_cause(error_epoch):
    calls = []
    error = RuntimeError("no held-out targets")

    def validate(model):
        calls.append(model)
        if len(calls) - 1 == error_epoch:
            raise error
        return 0.5

    with pytest.raises(ValueError, match=f"epoch {error_epoch}.*no held-out targets") as caught:
        fit_graph(history(), catalog(), epochs=2, validation_callback=validate)
    assert caught.value.__cause__ is error
    assert torch.is_grad_enabled()


def test_query_vectors_align_catalog_mask_cold_queries_and_return_copies(trained):
    model, _ = trained
    requested = ["d", "cold", "a", "missing", "a"]
    vectors, available = model.query_vectors(requested)
    assert vectors.shape == (5, 8) and vectors.dtype == np.float32
    assert available.tolist() == [True, False, True, False, True]
    assert not vectors[[1, 3]].any()
    np.testing.assert_allclose(np.linalg.norm(vectors[available], axis=1), 1, atol=1e-7)
    np.testing.assert_array_equal(vectors[2], vectors[4])
    expected_scores = vectors @ vectors[2]
    np.testing.assert_allclose(model.score_from_seeds(["a"], requested)[0], expected_scores, atol=1e-7)
    default, default_available = model.query_vectors()
    explicit, explicit_available = model.query_vectors(model.catalog_queries)
    np.testing.assert_array_equal(default, explicit)
    np.testing.assert_array_equal(default_available, explicit_available)
    vectors[:] = 0
    assert model.query_vectors(["a"])[0].any()
    empty, empty_available = model.query_vectors([])
    assert empty.shape == (0, 8) and empty_available.shape == (0,)


@pytest.mark.parametrize("data", [pd.DataFrame(columns=["sku", "normquery", "orders"]),
                                  pd.DataFrame([["all", "a", 1]], columns=["sku", "normquery", "orders"])])
def test_validation_without_eligible_edges_only_evaluates_epoch_zero(data):
    calls = []

    def validate(model):
        calls.append(model.query_vectors())
        return 0.5

    model, audit = fit_graph(data, ["a", "cold"], epochs=5, dimensions=4,
                             validation_callback=validate, patience=1)
    assert len(calls) == 1 and audit["validation_per_epoch"] == [0.5]
    assert audit["best_epoch"] == audit["epochs_completed"] == audit["optimizer_steps"] == 0
    assert not audit["trained"] and not audit["stopped_early"]
    assert audit["parameter_delta_l2"] == 0
    np.testing.assert_array_equal(model.query_vectors()[0], calls[0][0])