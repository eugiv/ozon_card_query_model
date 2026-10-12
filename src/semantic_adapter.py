"""Обучаемый residual-адаптер замороженных E5-векторов, не fine-tuning Transformer."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_data import (
    aggregate_clusters,
    dictionary_snapshot,
    require_columns,
    sha256,
    write_json,
)


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} должен быть целым числом >= {minimum}")


def _seed(seed):
    _positive_integer(seed, "seed", 0)
    if seed >= 2**63:
        raise ValueError("seed должен быть меньше 2**63")


def _catalog(catalog):
    if (not len(catalog) or len(set(catalog)) != len(catalog)
            or any(not isinstance(name, str) or not name.strip() for name in catalog)):
        raise ValueError("Каталог должен быть непустым и содержать уникальные фразы")


def _vectors(vectors):
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != 384:
        raise ValueError("Ожидаются E5-векторы формы (N, 384)")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Векторы должны быть конечными и ненулевыми")
    return np.ascontiguousarray(matrix / norms)


def _history_before(history, train_end=None):
    cutoff = train_end if train_end is not None else history.attrs.get("train_end")
    cutoff = date.fromisoformat(str(cutoff)) if cutoff is not None else None
    if "date" in history.columns:
        if cutoff is None:
            raise ValueError("Для daily history нужен явный train_end или history.attrs['train_end']")
        dates = pd.to_datetime(history["date"], utc=True, errors="raise").dt.normalize()
        if dates.isna().any():
            raise ValueError("Дата обязательна в каждой строке")
        selected = dates <= pd.Timestamp(cutoff, tz="UTC")
        part = history.loc[selected].copy()
        part["date"] = dates.loc[selected].dt.tz_localize(None)
        ignored = int((~selected).sum())
    else:
        part, ignored = history.copy(), 0
    if part.empty:
        raise ValueError("История до train_end пуста")
    return part, {
        "training_cutoff": cutoff.isoformat() if cutoff is not None else None,
        "future_rows_ignored": ignored,
        "historical_rows": len(part),
        "history_is_preaggregated": "date" not in history.columns,
    }


def build_training_pairs(history, catalog, seed=7, *, train_end=None):
    """Все sold edges для маски; <=64 unordered co-sale пары/SKU с весом 1/count.

    Daily-история сначала отсекается по train_end. История без date должна быть
    предварительно агрегирована вызывающим кодом только на обучающем интервале.
    Возвращает (pairs[sku,left,right,weight], selling_skus_by_cluster, audit).
    """
    _catalog(catalog)
    _seed(seed)
    part, audit = _history_before(history, train_end)
    require_columns(part, ["sku", "normquery", "orders"])
    part["sku"] = part["sku"].astype("string").fillna("").str.strip()
    orders = pd.to_numeric(part["orders"], errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(orders).all() or (orders < 0).any() or (orders % 1 != 0).any()
            or part["sku"].eq("").any()):
        raise ValueError("Ожидаются непустые SKU и конечные неотрицательные целые orders")
    if not set(part["normquery"]).issubset(catalog):
        raise ValueError("В истории есть кластеры вне замороженного каталога")
    sold = part.loc[orders > 0, ["sku", "normquery"]].drop_duplicates()
    index = {name: number for number, name in enumerate(catalog)}
    support = [set() for _ in catalog]
    records, singletons, counts = [], [], {}
    for sku, group in sold.groupby("sku", sort=True):
        clusters = sorted(index[name] for name in group["normquery"])
        for cluster in clusters:
            support[cluster].add(str(sku))
        size = len(clusters)
        if size < 2:
            singletons.append(str(sku))
            continue
        total = size * (size - 1) // 2
        count = min(total, 64)
        sku_seed = int.from_bytes(hashlib.sha256(f"{seed}:{sku}".encode()).digest()[:8], "big")
        rng = np.random.default_rng(sku_seed)
        ranks = np.arange(total) if total <= 64 else np.sort(rng.choice(total, 64, replace=False))
        rows = np.arange(size, dtype=np.int64)
        starts = rows * (2 * size - rows - 1) // 2
        for rank in ranks:
            left = int(np.searchsorted(starts, rank, side="right") - 1)
            right = left + 1 + int(rank - starts[left])
            records.append((str(sku), clusters[left], clusters[right], 1.0 / count))
        counts[str(sku)] = count
    pairs = pd.DataFrame(records, columns=["sku", "left", "right", "weight"])
    audit.update({
        "sold_edges": len(sold), "sold_skus": int(sold["sku"].nunique()),
        "train_skus": len(counts), "singleton_skus": singletons,
        "positive_pairs": len(pairs), "pairs_per_sku": counts,
        "max_pairs_per_sku": 64, "sku_weighting": "each eligible SKU has total pair weight 1",
        "confirmed_negatives": 0, "comparison_status": "unknown_weak_comparison",
    })
    return pairs, [frozenset(skus) for skus in support], audit


def denominator_mask(indices, positive_indices, selling_skus, catalog):
    """False для одинаковых фраз и общих train-SKU; designated positive всегда True."""
    indices = np.asarray(indices, dtype=np.int64)
    positive_indices = np.asarray(positive_indices, dtype=np.int64)
    size = len(indices)
    if (indices.ndim != 1 or positive_indices.shape != (size,)
            or (indices < 0).any() or (indices >= len(catalog)).any()
            or (positive_indices < 0).any() or (positive_indices >= size).any()
            or len(selling_skus) != len(catalog)):
        raise ValueError("Некорректные индексы contrastive batch")
    mask = np.ones((size, size), dtype=bool)
    by_phrase, by_sku = {}, {}
    for position, cluster in enumerate(indices):
        phrase = " ".join(catalog[cluster].casefold().split())
        by_phrase.setdefault(phrase, []).append(position)
        for sku in selling_skus[cluster]:
            by_sku.setdefault(sku, []).append(position)
    for positions in [*by_phrase.values(), *by_sku.values()]:
        mask[np.ix_(positions, positions)] = False
    mask[np.arange(size), positive_indices] = True
    return mask


def _new_adapter(device="cpu"):
    import torch

    class ResidualAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(384, 128), torch.nn.ReLU(), torch.nn.Linear(128, 384),
            )
            torch.nn.init.zeros_(self.mlp[2].weight)
            torch.nn.init.zeros_(self.mlp[2].bias)

        def forward(self, x):
            return torch.nn.functional.normalize(x + 0.1 * self.mlp(x), dim=-1)

    return ResidualAdapter().to(device)


def fit_adapter(history, catalog, vectors, epochs, batch_size=128, seed=7, device="cpu",
                *, train_end=None):
    """Symmetric masked in-batch InfoNCE; обучается только MLP, E5-векторы заморожены.

    Для daily history передайте train_end (или одноимённый attrs); история без date
    считается уже отфильтрованной. Нет co-sale не означает подтверждённый негатив.
    """
    _positive_integer(epochs, "epochs")
    _positive_integer(batch_size, "batch_size", 2)
    pairs, support, audit = build_training_pairs(history, catalog, seed, train_end=train_end)
    matrix = _vectors(vectors)
    if len(matrix) != len(catalog):
        raise ValueError("Векторы должны точно соответствовать каталогу")
    if pairs.empty:
        raise ValueError("Нет положительных co-sale пар для обучения")
    import torch

    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    losses, optimizer_steps, informative_anchors = [], 0, 0
    masked_candidates, weak_comparisons = 0, 0
    informative_pairs = {sku: 0 for sku in audit["pairs_per_sku"]}
    rng = np.random.default_rng(seed)
    try:
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = _new_adapter(device)
            base = torch.from_numpy(matrix).to(device).detach()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            endpoints = pairs[["left", "right"]].to_numpy(dtype=np.int64)
            pair_weights = pairs["weight"].to_numpy(dtype=np.float32)
            pair_skus = pairs["sku"].tolist()
            for _ in range(epochs):
                order = rng.permutation(len(pairs))
                batches = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
                if len(batches) > 1 and len(batches[-1]) == 1:
                    tail = batches.pop()
                    batches[-1] = np.concatenate((batches[-1], tail))
                epoch_loss, epoch_steps = 0.0, 0
                for selected in batches:
                    count = len(selected)
                    indices = np.concatenate((endpoints[selected, 0], endpoints[selected, 1]))
                    positive = np.concatenate((np.arange(count, 2 * count), np.arange(count)))
                    allowed = denominator_mask(indices, positive, support, catalog)
                    comparisons = int(allowed.sum() - len(indices))
                    masked_candidates += int((~allowed).sum())
                    weak_comparisons += comparisons
                    informative = allowed.sum(axis=1) > 1
                    informative_anchors += int(np.count_nonzero(informative))
                    for pair, active in zip(selected, informative[:count] | informative[count:]):
                        if active:
                            informative_pairs[pair_skus[pair]] += 1
                    if not comparisons:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    embeddings = model(base[torch.as_tensor(indices, device=device)])
                    logits = (embeddings @ embeddings.T) / 0.1
                    logits = logits.masked_fill(~torch.as_tensor(allowed, device=device), -torch.inf)
                    per_anchor = torch.nn.functional.cross_entropy(
                        logits, torch.as_tensor(positive, device=device), reduction="none",
                    )
                    per_pair = (per_anchor[:count] + per_anchor[count:]) / 2
                    weights = torch.as_tensor(pair_weights[selected], device=device)
                    weighted = (per_pair * weights).sum()
                    loss = weighted * (len(pairs) / (count * audit["train_skus"]))
                    if not torch.isfinite(loss):
                        raise ValueError("Неконечный contrastive loss")
                    loss.backward()
                    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                           for parameter in model.parameters()):
                        raise ValueError("Неконечные gradients адаптера")
                    optimizer.step()
                    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                        raise ValueError("Неконечные веса адаптера")
                    epoch_loss += float(weighted.detach().cpu())
                    epoch_steps += 1
                if not epoch_steps:
                    raise ValueError("В batch нет unknown comparison: contrastive обучение невозможно")
                optimizer_steps += epoch_steps
                losses.append(epoch_loss / audit["train_skus"])
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    model.eval()
    model.training_cutoff = audit["training_cutoff"]
    audit.update({
        "architecture": "normalize(x + 0.1 * MLP(x)); 384->128->384; ReLU",
        "transformer_fine_tuned": False, "frozen_base_vectors": True,
        "epochs": epochs, "batch_size": batch_size, "seed": seed, "device": str(device),
        "torch_version": str(torch.__version__), "temperature": 0.1, "learning_rate": 1e-3,
        "loss_per_epoch": losses, "optimizer_steps": optimizer_steps,
        "finite_gradients": True, "informative_anchors": informative_anchors,
        "informative_pairs_per_sku": informative_pairs,
        "zero_comparison_skus": sorted(sku for sku, count in informative_pairs.items() if not count),
        "singleton_batch_policy": "merge final singleton into previous batch (at most batch_size + 1)",
        "masked_denominator_entries": masked_candidates, "weak_comparison_entries": weak_comparisons,
        "quality_claim": "training objective only; relevance improvement is not established",
    })
    return model, audit


def apply_adapter(model, vectors):
    """Нормализованные numpy-векторы в исходном порядке, без изменения входа."""
    import torch

    matrix = _vectors(vectors)
    device = next(model.parameters()).device
    was_training = model.training
    outputs = []
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(matrix), 1024):
                transformed = model(torch.from_numpy(matrix[start:start + 1024]).to(device))
                outputs.append(transformed.cpu().numpy())
    finally:
        model.train(was_training)
    return _vectors(np.concatenate(outputs))


def save_adapter(model, path):
    """Только state_dict + простые metadata, файл никогда не перезаписывается."""
    import torch

    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Нельзя сохранить неконечные веса")
    payload = {"format_version": 1, "dimension": 384, "hidden": 128, "residual_scale": 0.1,
               "training_cutoff": getattr(model, "training_cutoff", None), "state_dict": state}
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)


def load_adapter(path, device="cpu"):
    """Без pickle-моделей и небезопасного fallback: всегда weights_only=True."""
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    expected = {"format_version": 1, "dimension": 384, "hidden": 128, "residual_scale": 0.1}
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Неизвестный формат residual adapter")
    state = payload.get("state_dict")
    if (not isinstance(state, dict) or any(not isinstance(value, torch.Tensor)
            or not torch.isfinite(value).all() for value in state.values())):
        raise ValueError("Некорректные веса адаптера")
    model = _new_adapter()
    model.load_state_dict(state, strict=True)
    model.training_cutoff = payload.get("training_cutoff")
    return model.to(device).eval()


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Обучить residual-адаптер замороженных E5; не fine-tuning E5")
    for name in ["facts", "dictionary", "embeddings", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dictionary-as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--train-end", type=date.fromisoformat, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("Выходной каталог уже существует")
    if args.dictionary_as_of > args.train_end:
        parser.error("dictionary-as-of не может быть позже train-end")
    try:
        _positive_integer(args.epochs, "epochs")
        _positive_integer(args.batch_size, "batch_size", 2)
        _seed(args.seed)
        input_paths = [args.facts, args.dictionary, args.embeddings]
        input_hashes = {str(path): sha256(path) for path in input_paths}
        facts = pd.read_csv(args.facts, keep_default_na=False, dtype={"sku": str})
        facts, temporal_audit = _history_before(facts, args.train_end)
        as_of = (pd.Timestamp(args.dictionary_as_of, tz="UTC")
                 + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1))
        records = pd.read_csv(args.dictionary, keep_default_na=False)
        snapshot, dictionary_audit = dictionary_snapshot(records, as_of)
        catalog = sorted(snapshot["normquery"].unique().tolist())
        _catalog(catalog)
        from cluster_recommender import load_embeddings

        vectors, metadata = load_embeddings(args.embeddings, catalog)
        if ("text_prefix" not in metadata or "dictionary_as_of" not in metadata
                or pd.Timestamp(metadata["dictionary_as_of"]) != as_of):
            raise ValueError("Embeddings должны иметь text_prefix и тот же dictionary_as_of")
        daily, aggregation_audit = aggregate_clusters(facts, snapshot)
        model, audit = fit_adapter(daily, catalog, vectors, args.epochs, args.batch_size,
                                   args.seed, args.device, train_end=args.train_end)
        audit["future_rows_ignored"] = temporal_audit["future_rows_ignored"]
        audit["dictionary"] = dictionary_audit
        audit["aggregation"] = aggregation_audit
        adapted = apply_adapter(model, vectors)
        if any(sha256(path) != input_hashes[str(path)] for path in input_paths):
            raise ValueError("Входные файлы изменились во время обучения")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".semantic-adapter-") as temporary:
            folder = Path(temporary)
            save_adapter(model, folder / "adapter.pt")
            adapter_hash = sha256(folder / "adapter.pt")
            np.savez_compressed(
                folder / "vectors.npz", normquery=np.asarray(catalog), embeddings=adapted,
                model_name=np.asarray(
                    ("e5-small" if metadata["model_name"] == "intfloat/multilingual-e5-small"
                     else metadata["model_name"]) + "+trained-residual-adapter"
                ),
                model_revision=np.asarray(adapter_hash), text_prefix=np.asarray(metadata["text_prefix"]),
                dictionary_as_of=np.asarray(metadata["dictionary_as_of"]),
                training_cutoff=np.asarray(args.train_end.isoformat()), device=np.asarray(args.device),
                torch_version=np.asarray(audit["torch_version"]),
            )
            write_json(folder / "metrics.json", audit)
            artifacts = ["adapter.pt", "vectors.npz", "metrics.json"]
            write_json(folder / "manifest.json", {
                "format_version": 1, "training_cutoff": args.train_end.isoformat(),
                "dictionary_as_of": metadata["dictionary_as_of"], "catalog_size": len(catalog),
                "base_embeddings": metadata, "transformer_fine_tuned": False,
                "input_sha256": input_hashes,
                "code_hash": sha256(Path(__file__)),
                "package_versions": {
                    "numpy": np.__version__, "pandas": pd.__version__, "torch": audit["torch_version"],
                },
                "artifact_sha256": {name: sha256(folder / name) for name in artifacts},
                "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
            })
            args.output.mkdir(exist_ok=False)
            published = []
            try:
                for name in [*artifacts, "manifest.json"]:
                    os.link(folder / name, args.output / name)
                    published.append(args.output / name)
            except OSError:
                for path in published:
                    path.unlink()
                args.output.rmdir()
                raise
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()