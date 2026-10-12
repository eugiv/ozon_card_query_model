"""Гибридный baseline: текст, совместные продажи и обучаемый ранжировщик."""

from __future__ import annotations

import argparse
import hashlib
import re
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from cluster_data import dictionary_snapshot, require_columns, rolling_training_windows

RANK_FEATURES = [
    "lexical_max",
    "lexical_mean",
    "embedding_max",
    "embedding_mean",
    "graph_score",
    "log_cluster_orders",
    "log_demand",
    "log_views",
    "log_searches",
    "history_days",
    "log_seed_orders",
    "seed_count",
    "text_source",
    "embedding_source",
    "graph_source",
    "popular_source",
]
CANDIDATE_COLUMNS = [
    "sku",
    "normquery",
    "closest_seed",
    "retrieval_sources",
    *RANK_FEATURES,
]


def load_embeddings(path: Path, catalog: list[str]) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as stored:
        names = stored["normquery"].astype(str).tolist()
        vectors = np.asarray(stored["embeddings"], dtype=np.float32)
        metadata = {
            key: str(stored[key].item()) for key in ["model_name", "model_revision"]
        }
        for key in ["text_prefix", "dictionary_as_of", "device", "torch_version"]:
            if key in stored.files:
                metadata[key] = str(stored[key].item())
    if len(set(names)) != len(names) or set(names) != set(catalog):
        raise ValueError(
            "Embeddings должны точно соответствовать уникальному каталогу снимка"
        )
    if vectors.ndim != 2 or vectors.shape[0] != len(names) or vectors.shape[1] < 1:
        raise ValueError("Некорректная форма embeddings")
    if not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) == 0).any():
        raise ValueError("Embeddings содержат нечисловые или нулевые векторы")
    if not metadata["model_name"] or not metadata["model_revision"]:
        raise ValueError("Для воспроизводимости нужны модель и revision embeddings")
    by_name = {name: index for index, name in enumerate(names)}
    return normalize(vectors[[by_name[name] for name in catalog]]), metadata


def lexical_index(catalog):
    return normalize(TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), dtype=np.float32
    ).fit_transform(catalog))


def encoding_as_of(cutoff: date, window_days: int, training_windows: int,
                   dictionary_as_of: date | None = None) -> pd.Timestamp:
    if dictionary_as_of is not None:
        if dictionary_as_of > cutoff:
            raise ValueError("Снимок словаря не может быть позже cutoff")
        day = dictionary_as_of
    else:
        day = rolling_training_windows(cutoff, window_days, training_windows)[0].history_end
    return pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Закодировать WB-кластеры предобученной embedding-моделью"
    )
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--cutoff", type=date.fromisoformat, required=True)
    parser.add_argument("--dictionary-as-of", type=date.fromisoformat,
                        help="Явная дата словаря для full-history; без неё используется прежняя схема окон")
    parser.add_argument("--window-days", type=int, default=28)
    parser.add_argument("--training-windows", type=int, default=1)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument(
        "--revision", required=True, help="Полный SHA коммита модели Hugging Face"
    )
    parser.add_argument(
        "--text-prefix",
        default="query: ",
        help="Префикс для E5; для другой модели проверьте её инструкцию",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        parser.error("--revision должен быть полным неизменяемым SHA, а не main")
    if args.batch_size < 1:
        parser.error("--batch-size должен быть положительным")
    if args.output.exists():
        parser.error("Выходной файл уже существует")
    try:
        as_of = encoding_as_of(args.cutoff, args.window_days, args.training_windows,
                               args.dictionary_as_of)
    except ValueError as error:
        parser.error(str(error))
    records = pd.read_csv(args.dictionary, keep_default_na=False)
    snapshot, _ = dictionary_snapshot(records, as_of)
    catalog = sorted(snapshot["normquery"].unique().tolist())
    if not catalog:
        parser.error("Исторический каталог пуст")
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError:
        parser.error(
            "Для нейросетевой ветки установите requirements.txt в подходящем окружении"
        )
    torch.manual_seed(7)
    model = SentenceTransformer(
        args.model, revision=args.revision, device=args.device, trust_remote_code=False
    )
    vectors = model.encode(
        [args.text_prefix + name for name in catalog],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    if not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) == 0).any():
        raise ValueError("Модель вернула некорректные embeddings")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=args.output.parent, prefix=".cluster-embeddings-"
    ) as temporary:
        path = Path(temporary) / "vectors.npz"
        np.savez_compressed(
            path,
            normquery=np.asarray(catalog),
            embeddings=vectors,
            model_name=np.asarray(args.model),
            model_revision=np.asarray(args.revision),
            text_prefix=np.asarray(args.text_prefix),
            dictionary_as_of=np.asarray(as_of.isoformat()),
            device=np.asarray(args.device),
            torch_version=np.asarray(torch.__version__),
        )
        if args.output.exists():
            raise ValueError(
                "Выходной файл появился во время кодирования; публикация отменена"
            )
        path.replace(args.output)


class HybridRetriever:
    """История только до даты решения; каталог задаётся замороженным словарём WB."""

    def __init__(
        self,
        catalog: list[str],
        history: pd.DataFrame,
        demand: pd.Series,
        embeddings: np.ndarray | None = None,
        lexical_vectors=None,
    ) -> None:
        if (
            not catalog
            or catalog != sorted(set(catalog))
            or any(not str(item).strip() for item in catalog)
        ):
            raise ValueError(
                "Каталог должен быть непустым, уникальным и отсортированным"
            )
        require_columns(
            history, ["sku", "normquery", "orders", "views", "searches", "days"]
        )
        if history.duplicated(["sku", "normquery"]).any():
            raise ValueError("История должна содержать одну строку на SKU + normquery")
        if not set(history["normquery"]).issubset(catalog):
            raise ValueError("В истории есть кластеры вне замороженного каталога")
        self.catalog = catalog
        self.history = history
        self.index = {name: index for index, name in enumerate(catalog)}
        self.skus = sorted(history["sku"].unique())
        self.sku_index = {name: index for index, name in enumerate(self.skus)}
        self.text = lexical_index(catalog) if lexical_vectors is None else lexical_vectors
        if self.text.shape[0] != len(catalog):
            raise ValueError("Текстовый индекс не соответствует каталогу")
        self.embeddings = embeddings
        if embeddings is not None:
            if (
                embeddings.ndim != 2
                or len(embeddings) != len(catalog)
                or not np.isfinite(embeddings).all()
            ):
                raise ValueError("Embeddings не соответствуют каталогу")
            if (
                embeddings.shape[1] < 1
                or (np.linalg.norm(embeddings, axis=1) == 0).any()
            ):
                raise ValueError("Embeddings содержат нулевые векторы")
            self.embeddings = normalize(embeddings)
        sold = history.loc[history["orders"] > 0]
        self.sales = normalize(
            sparse.csr_matrix(
                (
                    np.log1p(sold["orders"]),
                    (
                        sold["sku"].map(self.sku_index),
                        sold["normquery"].map(self.index),
                    ),
                ),
                shape=(len(self.skus), len(catalog)),
            )
        )
        self.popularity = (
            history.groupby("normquery")["orders"]
            .sum()
            .reindex(catalog, fill_value=0)
            .to_numpy()
        )
        self.demand = demand.reindex(catalog, fill_value=0).to_numpy()

    def _similarity(
        self, indices: list[int], weights: np.ndarray, matrix
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        maximum = np.full(len(self.catalog), -np.inf)
        mean = np.zeros(len(self.catalog))
        closest = np.zeros(len(self.catalog), dtype=int)
        for start in range(0, len(indices), 16):
            block = indices[start:start + 16]
            values = matrix @ matrix[block].T
            values = values.toarray() if sparse.issparse(values) else np.asarray(values)
            best = values.argmax(axis=1)
            block_max = values[np.arange(len(self.catalog)), best]
            better = block_max > maximum
            closest[better] = np.asarray(block)[best[better]]
            maximum = np.maximum(maximum, block_max)
            mean += values @ weights[start:start + len(block)]
        return maximum, mean, closest

    def generate(
        self, per_source: int = 100, *, max_history_views: int | None = None,
        traffic_history: pd.DataFrame | None = None, target_skus: set[str] | None = None,
        max_seeds: int | None = None,
    ) -> pd.DataFrame:
        if (
            isinstance(per_source, bool)
            or not isinstance(per_source, int)
            or per_source < 1
        ):
            raise ValueError("per_source должен быть положительным целым числом")
        if max_history_views is not None and (
            isinstance(max_history_views, bool)
            or not isinstance(max_history_views, int) or max_history_views < 0
        ):
            raise ValueError("max_history_views должен быть неотрицательным целым")
        if max_seeds is not None and (
            isinstance(max_seeds, bool) or not isinstance(max_seeds, int) or max_seeds < 1
        ):
            raise ValueError("max_seeds должен быть положительным целым")
        exposure = self.history if traffic_history is None else traffic_history
        require_columns(exposure, ["sku", "normquery", "views", "searches", "days"])
        if exposure.duplicated(["sku", "normquery"]).any():
            raise ValueError("Трафик должен содержать одну строку на SKU + normquery")
        numeric = exposure[["views", "searches", "days"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric < 0).any():
            raise ValueError("Трафик должен быть конечным и неотрицательным")
        traffic_by_sku = {sku: part for sku, part in exposure.groupby("sku", sort=False)}
        records = []
        for sku, own in self.history.groupby("sku", sort=True):
            if target_skus is not None and sku not in target_skus:
                continue
            seeds = own.loc[own["orders"] > 0].sort_values("normquery")
            if seeds.empty:
                continue
            anchors = seeds if max_seeds is None else seeds.sort_values(
                ["orders", "normquery"], ascending=[False, True], kind="stable"
            ).head(max_seeds).sort_values("normquery")
            indices = [self.index[name] for name in anchors["normquery"]]
            weights = anchors["orders"].to_numpy(dtype=float)
            weights = weights / weights.sum()
            lexical_max, lexical_mean, closest = self._similarity(
                indices, weights, self.text
            )
            if self.embeddings is None:
                embedding_max = embedding_mean = np.zeros(len(self.catalog))
                text_score = lexical_mean
            else:
                embedding_max, embedding_mean, closest = self._similarity(
                    indices, weights, self.embeddings
                )
                text_score = embedding_mean
            neighbours = self.sales @ self.sales[self.sku_index[sku]].T
            graph = np.asarray((self.sales.T @ neighbours).toarray()).ravel()
            allowed = np.ones(len(self.catalog), dtype=bool)
            allowed[[self.index[name] for name in seeds["normquery"]]] = False
            recent = traffic_by_sku.get(sku, exposure.iloc[:0])
            if max_history_views is not None:
                blocked = recent.loc[recent["views"] > max_history_views, "normquery"]
                allowed[[self.index[name] for name in blocked if name in self.index]] = False

            def top(values, allowed=allowed):
                eligible = np.flatnonzero(allowed & (values > 0))
                return eligible[
                    np.argsort(-values[eligible], kind="stable")[:per_source]
                ]

            sources = {
                "text": set(top(lexical_mean)),
                "embedding": set(top(text_score)) if self.embeddings is not None else set(),
                "graph": set(top(graph)),
                "popular": set(top(self.popularity)),
            }
            candidates = sorted(set().union(*sources.values()))
            own_by_query = recent.set_index("normquery")
            for index in candidates:
                name = self.catalog[index]
                pair = own_by_query.loc[name] if name in own_by_query.index else None
                records.append(
                    {
                        "sku": sku,
                        "normquery": name,
                        "closest_seed": self.catalog[closest[index]],
                        "retrieval_sources": ",".join(
                            key for key, values in sources.items() if index in values
                        ),
                        "lexical_max": lexical_max[index],
                        "lexical_mean": lexical_mean[index],
                        "embedding_max": embedding_max[index],
                        "embedding_mean": embedding_mean[index],
                        "graph_score": graph[index],
                        "log_cluster_orders": np.log1p(self.popularity[index]),
                        "log_demand": np.log1p(self.demand[index]),
                        "log_views": np.log1p(pair["views"])
                        if pair is not None
                        else 0.0,
                        "log_searches": np.log1p(pair["searches"])
                        if pair is not None
                        else 0.0,
                        "history_days": pair["days"] if pair is not None else 0,
                        "log_seed_orders": np.log1p(seeds["orders"].sum()),
                        "seed_count": len(seeds),
                        **{
                            f"{key}_source": int(index in values)
                            for key, values in sources.items()
                        },
                    }
                )
        return pd.DataFrame(records, columns=CANDIDATE_COLUMNS)


def mask_selling_edges(
    history: pd.DataFrame, seed: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sold = history.loc[history["orders"] > 0]
    hidden = []
    for sku, rows in sold.groupby("sku", sort=True):
        if len(rows) < 2:
            continue
        names = sorted(rows["normquery"])
        number = int.from_bytes(
            hashlib.sha256(f"{seed}:{sku}".encode()).digest()[:8], "big"
        )
        hidden.append({"sku": sku, "normquery": names[number % len(names)]})
    positives = pd.DataFrame(hidden, columns=["sku", "normquery"])
    masked = history.merge(
        positives.assign(_hidden=True), on=["sku", "normquery"], how="left"
    )
    return masked.loc[masked["_hidden"].isna(), history.columns].reset_index(
        drop=True
    ), positives
