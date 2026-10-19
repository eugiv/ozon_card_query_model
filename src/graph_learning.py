"""Обучаемый LightGCN по проданным SKU ↔ normquery, без будущих рёбер.

fit_graph принимает уже отфильтрованную историю. Для агрегатов без date
передавайте train_end и source_date_min/source_date_max через DataFrame.attrs
или train_end явно: неизвестные даты не выдаются за проверенный cutoff.
score использует обученный SKU только для диагностики; для восстановления
скрытых рёбер нужен score_from_seeds, НЕ использующий embedding SKU.

CLI: python src/graph_learning.py --facts facts.parquet --dictionary dict.parquet
--dictionary-as-of 2026-01-31 --train-end 2026-01-31 --output graph_run
--epochs 30 --dimensions 32 --layers 2
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import metadata
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
from training_validation import TrainingValidation

FORMAT_VERSION = 1
LEARNING_RATE = 0.03
L2_REGULARIZATION = 1e-4


def _torch():
    import torch

    return torch


def _timestamp(value, *, end_of_day: bool = False) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Дата обязательна")
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    if end_of_day and len(str(value)) == 10:
        stamp += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return stamp


def _catalog_queries(catalog) -> list[str]:
    if isinstance(catalog, pd.DataFrame):
        require_columns(catalog, ["normquery"])
        values = catalog["normquery"]
    elif isinstance(catalog, str):
        values = [catalog]
    else:
        values = catalog
    result = []
    for value in values:
        if pd.isna(value) or not str(value).strip():
            raise ValueError("catalog: normquery не может быть пустым")
        result.append(str(value).strip())
    return result


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm > 0)


@dataclass
class GraphModel:
    skus: tuple[str, ...]
    queries: tuple[str, ...]
    catalog_queries: tuple[str, ...]
    initial_embeddings: np.ndarray
    sku_embeddings: np.ndarray
    query_embeddings: np.ndarray
    audit: dict

    def __post_init__(self) -> None:
        self.sku_to_index = {value: index for index, value in enumerate(self.skus)}
        self.query_to_index = {value: index for index, value in enumerate(self.queries)}
        self._unit_queries = _normalize(self.query_embeddings)

    @property
    def train_end(self) -> str | None:
        return self.audit["train_end"]

    def query_vectors(self, catalog=None) -> tuple[np.ndarray, np.ndarray]:
        """Копия unit-векторов в порядке catalog (по умолчанию catalog_queries).

        Возвращает (vectors, available); cold/unknown queries имеют нулевую
        строку и False. Повторы и порядок каталога сохраняются.
        """
        requested = self.catalog_queries if catalog is None else _catalog_queries(catalog)
        vectors = np.zeros((len(requested), self.query_embeddings.shape[1]), dtype=np.float32)
        available = np.zeros(len(requested), dtype=bool)
        for row, query in enumerate(requested):
            index = self.query_to_index.get(query)
            if index is not None:
                vectors[row] = self._unit_queries[index]
                available[row] = True
        return vectors, available

    def _score_profile(self, profile: np.ndarray | None, catalog) -> tuple[np.ndarray, np.ndarray]:
        requested = _catalog_queries(catalog)
        scores = np.zeros(len(requested), dtype=np.float32)
        available = np.zeros(len(requested), dtype=bool)
        if profile is None or not np.any(profile):
            return scores, available
        positions = [(row, self.query_to_index[q]) for row, q in enumerate(requested)
                     if q in self.query_to_index]
        if positions:
            rows, indices = zip(*positions)
            scores[list(rows)] = self._unit_queries[list(indices)] @ _normalize(profile)
            available[list(rows)] = True
        return scores, available

    def score(self, sku, catalog) -> tuple[np.ndarray, np.ndarray]:
        """Диагностический cosine по SKU; не применять к hidden-edge оценке."""
        index = self.sku_to_index.get(str(sku).strip())
        profile = None if index is None else self.sku_embeddings[index]
        return self._score_profile(profile, catalog)

    def score_from_seeds(self, seedqueries: Iterable[str], catalog) -> tuple[np.ndarray, np.ndarray]:
        """LightGCN item embeddings with seed fold-in; никакого SKU ID.

        Среднее unit-embedding известных seed-кластеров нормируется ещё раз.
        Неизвестные seeds игнорируются; пустой/полностью cold профиль недоступен.
        """
        seeds = [seedqueries] if isinstance(seedqueries, str) else seedqueries
        indices = sorted({self.query_to_index[str(q).strip()] for q in seeds
                          if str(q).strip() in self.query_to_index})
        profile = self._unit_queries[indices].mean(axis=0) if indices else None
        return self._score_profile(profile, catalog)


def _sample_unknown(positives: np.ndarray, item_count: int, rng: np.random.Generator,
                    size: int) -> np.ndarray:
    """Равномерный complement sampling без rejection loop и известных рёбер."""
    positives = np.unique(np.asarray(positives, dtype=np.int64))
    if item_count < 0 or np.any(positives < 0) or np.any(positives >= item_count):
        raise ValueError("Некорректные индексы positive items")
    count = item_count - len(positives)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    ranks = rng.integers(0, count, size=size)
    return ranks + np.searchsorted(positives - np.arange(len(positives)), ranks, side="right")


def _adjacency(torch, sku_indices, query_indices, sku_count, query_count):
    node_count = sku_count + query_count
    left = torch.tensor(sku_indices, dtype=torch.int64, device="cpu")
    right = torch.tensor(query_indices, dtype=torch.int64, device="cpu") + sku_count
    rows = torch.cat((left, right))
    cols = torch.cat((right, left))
    degrees = torch.bincount(rows, minlength=node_count).to(torch.float32)
    values = (degrees[rows] * degrees[cols]).rsqrt()
    return torch.sparse_coo_tensor(torch.stack((rows, cols)), values,
                                   (node_count, node_count), check_invariants=True).coalesce()


def _propagate(torch, adjacency, initial, layers):
    current = initial
    total = initial
    for _ in range(layers):
        current = torch.sparse.mm(adjacency, current)
        total = total + current
    return total / (layers + 1)


def _history_dates(history: pd.DataFrame, train_end) -> dict:
    cutoff = train_end if train_end is not None else history.attrs.get("train_end")
    cutoff = _timestamp(cutoff, end_of_day=True) if cutoff is not None else None
    if "date" in history:
        dates = pd.to_datetime(history["date"], utc=True, errors="raise")
        if dates.isna().any():
            raise ValueError("history: date обязательна")
        first, last = (dates.min(), dates.max()) if len(dates) else (None, None)
        provenance = "history.date"
    else:
        first = history.attrs.get("source_date_min")
        last = history.attrs.get("source_date_max")
        first = _timestamp(first) if first is not None else None
        last = _timestamp(last) if last is not None else None
        provenance = "upstream attrs" if first is not None or last is not None else "unknown aggregate dates"
    if cutoff is not None and last is not None and last > cutoff:
        raise ValueError("history содержит будущие даты относительно train_end")
    if first is not None and last is not None and first > last:
        raise ValueError("Некорректный диапазон source_date")
    return {
        "train_end": cutoff.isoformat() if cutoff is not None else None,
        "source_date_min": first.isoformat() if first is not None else None,
        "source_date_max": last.isoformat() if last is not None else None,
        "source_date_provenance": provenance,
        "cutoff_verified": cutoff is not None and last is not None,
        "upstream_filter_required": True,
    }


def fit_graph(history: pd.DataFrame, catalog, epochs: int = 30, dimensions: int = 32,
              layers: int = 2, seed: int = 7, *, train_end=None, validation_callback=None,
              patience=None, min_delta=0.0, selection_metric=None) -> tuple[GraphModel, dict]:
    """Настоящие sparse CPU LightGCN + BPR по orders > 0.

    Каждый eligible sold edge участвует в каждой full-batch эпохе. SKU без
    неизвестных train-seen items остаётся в графе, но не участвует в BPR.
    Negative здесь означает unknown interaction, не доказанную нерелевантность.
    Callback(GraphModel) -> finite float максимизируется на эпохах 0..N по
    текущим propagated embeddings. Лучшие веса восстанавливаются; patience=None
    отключает только остановку. Без callback сохраняется последняя эпоха.
    """
    for name, value, minimum in [("epochs", epochs, 1), ("dimensions", dimensions, 1),
                                 ("layers", layers, 0), ("seed", seed, 0)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name}: ожидается целое число >= {minimum}")
    if seed >= 2**63:
        raise ValueError("seed должен быть меньше 2**63")
    validation = TrainingValidation(validation_callback, patience, min_delta, selection_metric)
    require_columns(history, ["sku", "normquery", "orders"])
    dates = _history_dates(history, train_end)
    catalog_queries = tuple(sorted(set(_catalog_queries(catalog))))
    data = history[["sku", "normquery", "orders"]].copy()
    for column in ["sku", "normquery"]:
        data[column] = data[column].astype("string").fillna("").str.strip()
        if data[column].eq("").any():
            raise ValueError(f"history: {column} обязательна")
    orders = pd.to_numeric(data["orders"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(orders).all() or np.any(orders < 0):
        raise ValueError("orders: ожидаются конечные неотрицательные значения")
    data["orders"] = orders
    sold = data.loc[data["orders"] > 0].drop_duplicates(["sku", "normquery"])
    outside = ~sold["normquery"].isin(catalog_queries)
    pairs = sold.loc[~outside].sort_values(["sku", "normquery"]).reset_index(drop=True)
    skus = tuple(sorted(pairs["sku"].unique().tolist()))
    queries = tuple(sorted(pairs["normquery"].unique().tolist()))
    sku_map = {value: index for index, value in enumerate(skus)}
    query_map = {value: index for index, value in enumerate(queries)}
    users = pairs["sku"].map(sku_map).to_numpy(dtype=np.int64)
    items = pairs["normquery"].map(query_map).to_numpy(dtype=np.int64)
    degree = np.bincount(users, minlength=len(skus))
    eligible = degree[users] < len(queries)
    eligible_users, eligible_items = users[eligible], items[eligible]
    saturated = [sku for index, sku in enumerate(skus) if degree[index] == len(queries)]
    offsets = np.concatenate(([0], np.cumsum(degree)))
    groups = [(index, items[offsets[index]:offsets[index + 1]]) for index in range(len(skus))
              if degree[index] < len(queries)]
    audit = {
        "method": "LightGCN item embeddings with seed fold-in",
        "format_version": FORMAT_VERSION,
        "epochs_requested": epochs, "epochs_completed": 0,
        "dimensions": dimensions, "layers": layers, "seed": seed, "torch_threads": 2,
        "learning_rate": LEARNING_RATE, "l2_regularization": L2_REGULARIZATION,
        "weighting": "inverse SKU sold degree; equal total weight per eligible SKU",
        "negative_semantics": "uniform unknown train-seen item, never a known sold edge; not claimed irrelevant",
        "input_rows": len(history), "sold_input_rows": int((orders > 0).sum()),
        "sold_edges": len(pairs), "sold_edges_outside_catalog": int(outside.sum()),
        "sku_nodes": len(skus), "query_nodes": len(queries),
        "catalog_queries": len(catalog_queries), "cold_catalog_queries": len(catalog_queries) - len(queries),
        "eligible_edges_per_epoch": int(eligible.sum()),
        "skus_without_unknown_items": saturated,
        "skipped_edges_per_epoch": int((~eligible).sum()),
        "edge_visits": 0, "optimizer_steps": 0, "losses": [], "bpr_losses": [],
        "trained": False, "parameter_delta_l2": 0.0,
        "cold_item_score": 0.0, "cold_item_available": False,
        "fold_in_uses_sku_embedding": False,
        **dates,
    }
    torch = _torch()
    torch.set_num_threads(2)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.nn.Parameter(torch.randn((len(skus) + len(queries), dimensions),
                                             generator=generator, device="cpu", dtype=torch.float32) * 0.1)
    before = initial.detach().clone()
    adjacency = _adjacency(torch, users, items, len(skus), len(queries))
    rng = np.random.default_rng(seed)

    def inference_model():
        with torch.no_grad():
            embeddings = _propagate(torch, adjacency, initial, layers).numpy().copy()
            trained_initial = initial.detach().numpy().copy()
        if not np.isfinite(embeddings).all() or not np.isfinite(trained_initial).all():
            raise ValueError("LightGCN embeddings не конечны")
        return GraphModel(skus, queries, catalog_queries, trained_initial,
                          embeddings[:len(skus)], embeddings[len(skus):], audit.copy())

    def snapshot():
        return initial.detach().clone()

    if validation.enabled:
        validation.evaluate(inference_model(), 0, snapshot)
    if len(eligible_users):
        optimizer = torch.optim.Adam([initial], lr=LEARNING_RATE)
        u = torch.from_numpy(eligible_users)
        positive = torch.from_numpy(eligible_items) + len(skus)
        weights = torch.from_numpy((1.0 / degree[eligible_users]).astype(np.float32))
        weights = weights / weights.sum()
        for epoch in range(1, epochs + 1):
            negatives = np.concatenate([_sample_unknown(pos, len(queries), rng, len(pos))
                                        for _, pos in groups])
            negative = torch.from_numpy(negatives) + len(skus)
            optimizer.zero_grad()
            embedding = _propagate(torch, adjacency, initial, layers)
            margin = (embedding[u] * (embedding[positive] - embedding[negative])).sum(dim=1)
            bpr = (torch.nn.functional.softplus(-margin) * weights).sum()
            regularization = initial.square().sum() / initial.shape[0]
            loss = bpr + L2_REGULARIZATION * regularization
            if not torch.isfinite(loss):
                raise ValueError("LightGCN loss не конечен")
            loss.backward()
            if initial.grad is None or not torch.isfinite(initial.grad).all():
                raise ValueError("LightGCN gradients не конечны")
            optimizer.step()
            if not torch.isfinite(initial).all():
                raise ValueError("LightGCN weights не конечны")
            audit["losses"].append(float(loss.detach()))
            audit["bpr_losses"].append(float(bpr.detach()))
            audit["epochs_completed"] = epoch
            audit["optimizer_steps"] += 1
            audit["edge_visits"] += len(eligible_users)
            audit["trained"] = True
            if validation.enabled and validation.evaluate(inference_model(), epoch, snapshot):
                break
    with torch.no_grad():
        validation.restore(initial.copy_)
        audit["parameter_delta_l2"] = float(torch.linalg.vector_norm(initial - before))
    audit.update(validation.audit(audit["epochs_completed"], epochs))
    audit["status"] = "trained" if audit["trained"] else "no eligible BPR edges"
    model = inference_model()
    model.audit = audit
    return model, audit


def save_graph(model: GraphModel, path, *, inputs: dict | None = None) -> Path:
    """Создать новый неизменяемый run directory; существующий не перезаписывать."""
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"output уже существует: {output}")
    torch = _torch()
    packages = {name: metadata.version(name) for name in ["torch", "numpy", "pandas"]}
    payload = {
        "format_version": FORMAT_VERSION,
        "skus": list(model.skus), "queries": list(model.queries),
        "initial_embeddings": torch.from_numpy(model.initial_embeddings.copy()),
        "sku_embeddings": torch.from_numpy(model.sku_embeddings.copy()),
        "query_embeddings": torch.from_numpy(model.query_embeddings.copy()),
        "dimensions": model.audit["dimensions"], "layers": model.audit["layers"],
        "train_end": model.train_end,
    }
    output.mkdir(parents=True, exist_ok=False)
    torch.save(payload, output / "graph.pt")
    write_json(output / "catalog.json", {"normquery": list(model.catalog_queries)})
    write_json(output / "metrics.json", model.audit)
    checksums = {name: sha256(output / name) for name in ["graph.pt", "catalog.json", "metrics.json"]}
    code_paths = [Path(__file__), Path(__file__).with_name("cluster_data.py")]
    write_json(output / "manifest.json", {
        "format_version": FORMAT_VERSION, "method": model.audit["method"],
        "train_end": model.train_end, "source_date_min": model.audit["source_date_min"],
        "source_date_max": model.audit["source_date_max"],
        "dictionary_as_of": model.audit.get("dictionary_as_of"),
        "checksums": checksums, "inputs": inputs or {},
        "code_sha256": {file.name: sha256(file) for file in code_paths},
        "packages": packages, "weights_only_load": True,
    })
    return output


def load_graph(path) -> GraphModel:
    """Проверить checksum/schema/cutoff и загрузить только безопасные tensor weights."""
    import json

    output = Path(path)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected = {"graph.pt", "catalog.json", "metrics.json"}
    if manifest.get("format_version") != FORMAT_VERSION or set(manifest.get("checksums", {})) != expected:
        raise ValueError("Некорректный manifest/schema")
    for name in expected:
        if sha256(output / name) != manifest["checksums"][name]:
            raise ValueError(f"Checksum mismatch: {name}")
    audit = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    catalog = json.loads((output / "catalog.json").read_text(encoding="utf-8"))["normquery"]
    torch = _torch()
    payload = torch.load(output / "graph.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Некорректный graph schema")
    dimensions = payload.get("dimensions")
    if type(dimensions) is not int or dimensions < 1 or type(payload.get("layers")) is not int or payload["layers"] < 0:
        raise ValueError("Некорректные dimensions/layers")
    for name in ["skus", "queries"]:
        values = payload.get(name)
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in values):
            raise ValueError(f"Некорректные mappings: {name}")
        if values != sorted(set(values)):
            raise ValueError(f"Mappings не уникальны/не отсортированы: {name}")
    if not isinstance(catalog, list) or _catalog_queries(catalog) != catalog or catalog != sorted(set(catalog)):
        raise ValueError("Некорректный catalog")
    if not set(payload["queries"]).issubset(catalog):
        raise ValueError("Query mappings отсутствуют в catalog")
    shapes = {"initial_embeddings": (len(payload["skus"]) + len(payload["queries"]), dimensions),
              "sku_embeddings": (len(payload["skus"]), dimensions),
              "query_embeddings": (len(payload["queries"]), dimensions)}
    arrays = {}
    for name, shape in shapes.items():
        tensor = payload.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or tensor.dtype != torch.float32:
            raise ValueError(f"Некорректный tensor: {name}")
        if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
            raise ValueError(f"Некорректный shape/finite: {name}")
        arrays[name] = tensor.numpy().copy()
    if (audit.get("dimensions") != dimensions or audit.get("layers") != payload["layers"]
            or audit.get("sku_nodes") != len(payload["skus"]) or audit.get("query_nodes") != len(payload["queries"])
            or audit.get("catalog_queries") != len(catalog)):
        raise ValueError("Audit не соответствует weights/mappings")
    if manifest.get("train_end") != payload.get("train_end") or audit.get("train_end") != payload.get("train_end"):
        raise ValueError("Несогласованный train_end")
    for name in ["source_date_min", "source_date_max"]:
        if manifest.get(name) != audit.get(name):
            raise ValueError("Несогласованные source dates")
    if (audit.get("train_end") and audit.get("source_date_max")
            and _timestamp(audit["source_date_max"]) > _timestamp(audit["train_end"])):
        raise ValueError("Будущие graph edges")
    if manifest.get("dictionary_as_of") != audit.get("dictionary_as_of"):
        raise ValueError("Несогласованный dictionary_as_of")
    if (audit.get("dictionary_as_of") and audit.get("train_end")
            and _timestamp(audit["dictionary_as_of"]) > _timestamp(audit["train_end"])):
        raise ValueError("Будущий dictionary_as_of")
    return GraphModel(tuple(payload["skus"]), tuple(payload["queries"]), tuple(catalog),
                      arrays["initial_embeddings"], arrays["sku_embeddings"], arrays["query_embeddings"], audit)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"sku": "string", "query": "string", "normquery": "string"},
                           keep_default_na=False)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Ожидается CSV или Parquet")


def train_graph_from_files(*, facts, dictionary, dictionary_as_of, train_end, output,
                           epochs=30, dimensions=32, layers=2, seed=7) -> tuple[GraphModel, dict]:
    """Читать полные факты, фильтровать cutoff ДО агрегации; никаких downloads."""
    if Path(output).exists():
        raise FileExistsError(f"output уже существует: {output}")
    cutoff = _timestamp(train_end, end_of_day=True)
    as_of = _timestamp(dictionary_as_of, end_of_day=True)
    if as_of > cutoff:
        raise ValueError("dictionary-as-of не может быть позже train-end")
    facts_path, dictionary_path = Path(facts), Path(dictionary)
    inputs = {name: {"path": str(path), "sha256": sha256(path)}
              for name, path in [("facts", facts_path), ("dictionary", dictionary_path)]}
    raw = _read_table(facts_path)
    require_columns(raw, ["date"])
    dates = pd.to_datetime(raw["date"], utc=True, errors="raise")
    if dates.isna().any():
        raise ValueError("facts: date обязательна")
    past = raw.loc[dates <= cutoff].copy()
    past["date"] = dates.loc[dates <= cutoff]
    snapshot, dictionary_audit = dictionary_snapshot(_read_table(dictionary_path), as_of)
    daily, aggregation_audit = aggregate_clusters(past, snapshot)
    history = daily.groupby(["sku", "normquery"], as_index=False)["orders"].sum()
    history.attrs.update({"train_end": cutoff,
                          "source_date_min": past["date"].min() if len(past) else None,
                          "source_date_max": past["date"].max() if len(past) else None})
    catalog = snapshot[["normquery"]].drop_duplicates().sort_values("normquery")
    model, audit = fit_graph(history, catalog, epochs, dimensions, layers, seed=seed, train_end=cutoff)
    audit.update({"facts_input_rows": len(raw), "facts_train_rows": len(past),
                  "future_fact_rows_ignored": len(raw) - len(past),
                  "dictionary_as_of": as_of.isoformat(),
                  "dictionary_audit": dictionary_audit, "aggregation_audit": aggregation_audit})
    for name, path in [("facts", facts_path), ("dictionary", dictionary_path)]:
        if sha256(path) != inputs[name]["sha256"]:
            raise ValueError(f"Вход изменился во время обучения: {name}")
    save_graph(model, output, inputs=inputs)
    return model, audit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train sold-edge CPU LightGCN; immutable artifacts")
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--dictionary-as-of", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dimensions", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        _, audit = train_graph_from_files(**vars(args))
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"LightGCN: {audit['sold_edges']} sold edges, {audit['epochs_completed']} epochs; {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())