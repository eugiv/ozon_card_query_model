import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cluster_recommender import load_embeddings
from semantic_adapter import (
    apply_adapter,
    build_training_pairs,
    denominator_mask,
    fit_adapter,
    load_adapter,
    main,
    save_adapter,
)


def tiny_data():
    catalog = ["a", "b", "c", "d", "e", "f", "g"]
    history = pd.DataFrame(
        [["s1", "a", 2], ["s1", "b", 1], ["s1", "c", 1],
         ["s2", "d", 1], ["s2", "e", 1], ["s3", "f", 1],
         ["s3", "g", 1], ["singleton", "a", 1], ["unsold", "f", 0]],
        columns=["sku", "normquery", "orders"],
    )
    vectors = np.random.default_rng(7).normal(size=(len(catalog), 384)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return history, catalog, vectors


def test_module_import_does_not_import_torch():
    src = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", (f"import sys; sys.path.insert(0, {src!r}); "
                               "import semantic_adapter; assert 'torch' not in sys.modules")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_actual_fit_has_finite_gradients_changing_loss_and_frozen_inputs():
    history, catalog, vectors = tiny_data()
    original = vectors.copy()
    model, audit = fit_adapter(history, catalog, vectors, epochs=6, batch_size=128)
    result = apply_adapter(model, vectors)
    assert result.shape == vectors.shape
    assert np.isfinite(result).all()
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(vectors, original)
    assert not np.allclose(result, original)
    assert len(audit["loss_per_epoch"]) == 6
    assert np.isfinite(audit["loss_per_epoch"]).all()
    assert audit["loss_per_epoch"][-1] < audit["loss_per_epoch"][0]
    assert audit["optimizer_steps"] == 6
    assert audit["finite_gradients"] is True
    assert audit["confirmed_negatives"] == 0
    assert audit["comparison_status"] == "unknown_weak_comparison"
    assert audit["transformer_fine_tuned"] is False
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert model.mlp[0].in_features == 384
    assert model.mlp[0].out_features == model.mlp[2].in_features == 128
    assert model.mlp[2].out_features == 384
    tensor = torch.from_numpy(vectors)
    with torch.no_grad():
        expected = torch.nn.functional.normalize(tensor + 0.1 * model.mlp(tensor), dim=1)
    np.testing.assert_allclose(result, expected.numpy(), atol=1e-7)


def test_all_selling_skus_edges_and_balanced_deterministic_pair_cap():
    history, catalog, _ = tiny_data()
    extra = [f"z{index:02}" for index in range(30)]
    catalog += extra
    history = pd.concat([history, pd.DataFrame(
        [["large", name, 1] for name in extra], columns=history.columns,
    )], ignore_index=True)
    pairs, support, audit = build_training_pairs(history, catalog)
    shuffled = build_training_pairs(history.sample(frac=1, random_state=99), catalog)
    pd.testing.assert_frame_equal(pairs, shuffled[0])
    assert support == shuffled[1]
    counts = pairs.groupby("sku").size().to_dict()
    assert counts == {"large": 64, "s1": 3, "s2": 1, "s3": 1}
    np.testing.assert_allclose(pairs.groupby("sku")["weight"].sum(), 1)
    assert (pairs["left"] < pairs["right"]).all()
    assert not pairs.duplicated(["sku", "left", "right"]).any()
    assert all("large" in support[catalog.index(name)] for name in extra)
    assert "singleton" in support[catalog.index("a")]
    assert audit["sold_skus"] == 5
    assert audit["train_skus"] == 4
    assert audit["sold_edges"] == 38
    assert audit["singleton_skus"] == ["singleton"]
    sampled = set(pairs.loc[pairs["sku"].eq("large"), ["left", "right"]].itertuples(index=False, name=None))
    missed = next((left, right) for left in range(7, len(catalog))
                  for right in range(left + 1, len(catalog)) if (left, right) not in sampled)
    mask = denominator_mask([missed[0], missed[1], 7, 8], [2, 3, 0, 1], support, catalog)
    assert not mask[0, 1]  # Пара не выбрана в max64, но общий продающий SKU всё равно известен.


def test_mask_uses_all_shared_skus_and_same_phrases_except_designated_positive():
    catalog = ["a", "b", "c", "d", "e", " A "]
    history = pd.DataFrame(
        [["x", "a", 1], ["x", "b", 1], ["y", "b", 1], ["y", "c", 1],
         ["z", "d", 1], ["z", "e", 1], ["future", "a", 0]],
        columns=["sku", "normquery", "orders"],
    )
    _, support, _ = build_training_pairs(history, catalog)
    indices = np.array([0, 1, 1, 2, 3, 4, 5, 1])
    positive = np.array([1, 0, 3, 2, 5, 4, 7, 6])
    mask = denominator_mask(indices, positive, support, catalog)
    assert mask.dtype == bool
    assert mask[np.arange(len(indices)), positive].all()
    assert not np.diag(mask).any()
    assert not mask[0, 2]  # Другая копия положительной фразы.
    assert not mask[1, 3]  # Общий SKU y, хотя designated positive относится к x.
    assert not mask[0, 6]  # Та же фраза после нормализации пробелов и регистра.
    assert mask[0, 4]      # Нет co-sale: unknown, а не confirmed negative.


def test_daily_cutoff_precedes_validation_and_future_never_affects_fit():
    history, catalog, vectors = tiny_data()
    history["date"] = "2026-01-09"
    history.attrs["train_end"] = "2026-01-10"
    future = pd.DataFrame([["bad", "outside_catalog", -100, "2026-01-11"]],
                          columns=history.columns)
    extended = pd.concat([history, future], ignore_index=True)
    extended.attrs["train_end"] = "2026-01-10"
    model, audit = fit_adapter(history, catalog, vectors, 3)
    other, other_audit = fit_adapter(extended, catalog, vectors, 3)
    np.testing.assert_array_equal(apply_adapter(model, vectors), apply_adapter(other, vectors))
    assert audit["loss_per_epoch"] == other_audit["loss_per_epoch"]
    assert other_audit["future_rows_ignored"] == 1
    pairs, _, _ = build_training_pairs(extended, catalog, train_end="2026-01-10")
    assert "bad" not in set(pairs["sku"])
    no_cutoff = history.copy()
    no_cutoff.attrs.clear()
    with pytest.raises(ValueError, match="train_end"):
        fit_adapter(no_cutoff, catalog, vectors, 1)
    with pytest.raises(ValueError, match="пуст"):
        fit_adapter(extended, catalog, vectors, 1, train_end="2026-01-08")


def test_safe_roundtrip_and_existing_weight_file_is_immutable(tmp_path, monkeypatch):
    history, catalog, vectors = tiny_data()
    model, _ = fit_adapter(history, catalog, vectors, 2)
    path = tmp_path / "adapter.pt"
    save_adapter(model, path)
    original = path.read_bytes()
    torch_load = torch.load
    calls = []

    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        return torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, torch_load.__name__, checked_load)
    restored = load_adapter(path)
    assert calls[0]["weights_only"] is True
    np.testing.assert_array_equal(apply_adapter(model, vectors), apply_adapter(restored, vectors))
    with pytest.raises(FileExistsError):
        save_adapter(model, path)
    assert path.read_bytes() == original


def test_rejects_corrupt_checkpoint_without_unsafe_loading(tmp_path):
    history, catalog, vectors = tiny_data()
    model, _ = fit_adapter(history, catalog, vectors, 1)
    path = tmp_path / "adapter.pt"
    save_adapter(model, path)
    payload = torch.load(path, weights_only=True)
    payload["state_dict"]["mlp.0.weight"][0, 0] = torch.nan
    bad = tmp_path / "nonfinite.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="веса"):
        load_adapter(bad)
    payload["format_version"] = 999
    bad_format = tmp_path / "unknown-format.pt"
    torch.save(payload, bad_format)
    with pytest.raises(ValueError, match="формат"):
        load_adapter(bad_format)


def test_minibatches_merge_singleton_tail_and_restore_deterministic_settings():
    history, catalog, vectors = tiny_data()
    previous = torch.are_deterministic_algorithms_enabled()
    rng = torch.get_rng_state().clone()
    model, audit = fit_adapter(history, catalog, vectors, 3, batch_size=4)
    assert audit["optimizer_steps"] == 3
    assert audit["zero_comparison_skus"] == []
    assert set(audit["informative_pairs_per_sku"]) == {"s1", "s2", "s3"}
    assert torch.are_deterministic_algorithms_enabled() == previous
    assert torch.equal(torch.get_rng_state(), rng)
    other, repeated = fit_adapter(history.sample(frac=1, random_state=99), catalog,
                                  vectors, 3, batch_size=4)
    np.testing.assert_array_equal(apply_adapter(model, vectors), apply_adapter(other, vectors))
    assert audit["loss_per_epoch"] == repeated["loss_per_epoch"]


def cli_inputs(tmp_path):
    _, catalog, vectors = tiny_data()
    facts = pd.DataFrame(
        [["2026-01-09", sku, name, 10, 2, 1]
         for sku, names in [("s1", ["a", "b"]), ("s2", ["c", "d"]),
                            ("s3", ["e", "f", "g"])] for name in names]
        + [["2026-01-11", "future", "outside", -1, -2, -3]],
        columns=["date", "sku", "query", "unique_search_users", "unique_view_users", "order_count"],
    )
    dictionary = pd.DataFrame(
        [[name, name, "2026-01-08", 200, ""] for name in catalog],
        columns=["query", "normquery", "loaded_at", "status_code", "error"],
    )
    facts.to_csv(tmp_path / "facts.csv", index=False)
    dictionary.to_csv(tmp_path / "dictionary.csv", index=False)
    np.savez(tmp_path / "e5.npz", normquery=np.array(catalog[::-1]),
             embeddings=vectors[::-1], model_name=np.array("intfloat/multilingual-e5-small"),
             model_revision=np.array("a" * 40), text_prefix=np.array("query: "),
             dictionary_as_of=np.array("2026-01-08T23:59:59.999999999+00:00"))
    args = ["--facts", str(tmp_path / "facts.csv"), "--dictionary", str(tmp_path / "dictionary.csv"),
            "--embeddings", str(tmp_path / "e5.npz"), "--dictionary-as-of", "2026-01-08",
            "--train-end", "2026-01-10", "--epochs", "2", "--device", "cpu",
            "--output", str(tmp_path / "trained")]
    return args, catalog, vectors


@pytest.mark.parametrize("base_model", ["intfloat/multilingual-e5-small", "another-384d-encoder"])
def test_cli_exact_catalog_metadata_hashes_roundtrip_and_immutable_output(tmp_path, base_model):
    args, catalog, vectors = cli_inputs(tmp_path)
    path = tmp_path / "e5.npz"
    with np.load(path, allow_pickle=False) as stored:
        payload = {name: stored[name] for name in stored.files}
    payload["model_name"] = np.array(base_model)
    np.savez(path, **payload)
    main(args)
    output = tmp_path / "trained"
    with np.load(output / "vectors.npz", allow_pickle=False) as stored:
        expected = "e5-small" if base_model == "intfloat/multilingual-e5-small" else base_model
        assert stored["model_name"].item() == f"{expected}+trained-residual-adapter"
    assert {path.name for path in output.iterdir()} == {
        "adapter.pt", "vectors.npz", "metrics.json", "manifest.json"}
    loaded, metadata = load_embeddings(output / "vectors.npz", catalog)
    adapted = apply_adapter(load_adapter(output / "adapter.pt"), vectors)
    np.testing.assert_allclose(loaded, adapted, atol=1e-6)
    weight_hash = hashlib.sha256((output / "adapter.pt").read_bytes()).hexdigest()
    assert metadata["model_name"] == f"{expected}+trained-residual-adapter"
    assert metadata["model_revision"] == weight_hash
    assert metadata["text_prefix"] == "query: "
    assert metadata["dictionary_as_of"] == "2026-01-08T23:59:59.999999999+00:00"
    with np.load(output / "vectors.npz", allow_pickle=False) as stored:
        assert stored["training_cutoff"].item() == "2026-01-10"
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["future_rows_ignored"] == 1
    assert metrics["train_skus"] == 3
    manifest = json.loads((output / "manifest.json").read_text())
    source = Path(__file__).resolve().parents[1] / "src" / "semantic_adapter.py"
    assert manifest["code_hash"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["package_versions"] == {
        "numpy": np.__version__, "pandas": pd.__version__, "torch": str(torch.__version__),
    }
    for name, digest in manifest["artifact_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["input_sha256"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(SystemExit):
        main(args)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(ValueError, match="точно соответствовать"):
        load_embeddings(output / "vectors.npz", catalog[:-1])


@pytest.mark.parametrize("option,value", [
    ("--dictionary-as-of", "2026-01-11"), ("--train-end", "bad"),
    ("--train-end", "2026-01-07"), ("--epochs", "0"),
])
def test_cli_rejects_bad_dates_and_parameters_without_output(tmp_path, option, value):
    args, _, _ = cli_inputs(tmp_path)
    args[args.index(option) + 1] = value
    with pytest.raises(SystemExit):
        main(args)
    assert not (tmp_path / "trained").exists()


def test_cli_rejects_empty_past_and_mismatched_embedding_snapshot(tmp_path):
    args, _, _ = cli_inputs(tmp_path)
    facts = pd.read_csv(tmp_path / "facts.csv")
    facts["date"] = "2026-01-11"
    facts.to_csv(tmp_path / "facts.csv", index=False)
    with pytest.raises(SystemExit):
        main(args)
    assert not (tmp_path / "trained").exists()
    cli_inputs(tmp_path)
    args[args.index("--dictionary-as-of") + 1] = "2026-01-09"
    with pytest.raises(SystemExit):
        main(args)
    assert not (tmp_path / "trained").exists()


def test_invalid_vectors_and_no_contrastive_signal_fail_honestly():
    history, catalog, vectors = tiny_data()
    for bad in [vectors[:, :128], vectors * np.nan, vectors * 0]:
        with pytest.raises(ValueError):
            fit_adapter(history, catalog, bad, 1)
    with pytest.raises(ValueError, match="co-sale"):
        fit_adapter(history.loc[history["sku"].eq("singleton")], catalog, vectors, 1)
    with pytest.raises(ValueError, match="comparison"):
        fit_adapter(history.loc[history["sku"].eq("s1")], catalog, vectors, 1)