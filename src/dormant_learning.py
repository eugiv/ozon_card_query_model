"""Retrieval новых кластеров: positives против слабых unlabelled-компараторов.

Label 0 не означает отрицательную релевантность. PairLogit оптимизирует порядок
retrieval, не вероятность продажи, рост traffic или доказанную релевантность.
Eligibility (не более пяти просмотров SKU) задаёт upstream retriever.
"""

from __future__ import annotations

import hashlib
import itertools
from bisect import bisect_right
from math import fsum, log2
from numbers import Integral, Real

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from cluster_data import require_columns
from cluster_recommender import RANK_FEATURES

_BASE_FEATURES = [
    "lexical_max",
    "lexical_mean",
    "embedding_max",
    "embedding_mean",
    "graph_score",
    "log_cluster_orders",
    "log_demand",
]
LEARNING_FEATURES = [*_BASE_FEATURES, *[f"{name}_rr" for name in _BASE_FEATURES]]
_HARD_FEATURES = _BASE_FEATURES[:5]


CHANNELS = {"lexical": "lexical_mean", "co_sales": "graph_score",
            "pretrained_embeddings": "embedding_mean", "semantic_adapter": "adapter_score",
            "lightgcn": "lightgcn_score", "catboost_pu": "catboost_score"}


def _validate_pairs(frame: pd.DataFrame, name: str, required: tuple[str, ...]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} has duplicate column names")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
    for column in ("sku", "normquery"):
        values = frame[column]
        if values.isna().any() or any(
            isinstance(value, str) and not value.strip() for value in values
        ):
            raise ValueError(f"{name}.{column} must not contain missing or blank values")
    if frame.duplicated(subset=["sku", "normquery"]).any():
        raise ValueError(f"{name} contains duplicate sku + normquery pairs")


def ranking_metrics(
    predictions: pd.DataFrame,
    positives: pd.DataFrame,
    ks: tuple[int, ...] = (5, 10),
) -> dict:
    """Macro по всем SKU с положительными ответами, включая потерянные при поиске.

    Порядок: score убывает, normquery возрастает. Precision делится на K,
    ideal DCG учитывает полный набор positives. Пустые positives дают None.
    Входы не изменяются; некорректные схемы, дубли и scores отклоняются.
    """
    if not isinstance(ks, tuple) or any(
        isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k <= 0
        for k in ks
    ):
        raise ValueError("ks must be a tuple of unique positive integers, not booleans")
    if len(set(ks)) != len(ks):
        raise ValueError("ks must contain unique positive integers")
    ks = tuple(int(k) for k in ks)

    _validate_pairs(predictions, "predictions", ("sku", "normquery", "score"))
    _validate_pairs(positives, "positives", ("sku", "normquery"))
    scores = predictions["score"]
    if not all(
        isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
        for value in scores
    ):
        raise ValueError("predictions.score must contain finite real numbers")
    try:
        finite_scores = np.isfinite(scores.to_numpy(dtype=float)).all()
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("predictions.score must contain finite real numbers") from error
    if not finite_scores:
        raise ValueError("predictions.score must contain finite real numbers")

    evaluated_skus = int(positives["sku"].nunique())
    result = {
        "evaluated_skus": evaluated_skus,
        "positive_pairs": len(positives),
        "candidate_pairs": len(predictions),
    }
    metric_values = {}
    for k in ks:
        for metric in ("recall", "precision", "ndcg"):
            metric_values[f"{metric}@{k}"] = []
    metric_values["mrr"] = []
    metric_values["retrieval_recall"] = []
    if not evaluated_skus:
        result.update({metric: None for metric in metric_values})
        return result

    ranked = predictions[["sku", "normquery", "score"]].assign(
        score=scores.astype(object), normquery=predictions["normquery"].astype(object)
    ).sort_values(
        ["score", "normquery"], ascending=[False, True], kind="mergesort"
    )
    ranked_queries = {
        sku: group["normquery"].tolist()
        for sku, group in ranked.groupby("sku", sort=False, observed=True)
    }
    for sku, group in positives.groupby("sku", sort=False, observed=True):
        target_queries = set(group["normquery"])
        total_positive = len(target_queries)
        hit_ranks = [
            rank for rank, query in enumerate(ranked_queries.get(sku, ()), start=1)
            if query in target_queries
        ]
        for k in ks:
            hits = bisect_right(hit_ranks, k)
            dcg = fsum(1 / log2(rank + 1) for rank in hit_ranks[:hits])
            ideal_dcg = fsum(1 / log2(rank + 1)
                             for rank in range(1, min(k, total_positive) + 1))
            metric_values[f"recall@{k}"].append(hits / total_positive)
            metric_values[f"precision@{k}"].append(hits / k)
            metric_values[f"ndcg@{k}"].append(dcg / ideal_dcg)
        metric_values["mrr"].append(1 / hit_ranks[0] if hit_ranks else 0.0)
        metric_values["retrieval_recall"].append(len(hit_ranks) / total_positive)

    result.update({metric: fsum(values) / evaluated_skus
                   for metric, values in metric_values.items()})
    return result


def reciprocal_ranks(candidates: pd.DataFrame, channels, offset: int = 60):
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 1:
        raise ValueError("offset должен быть положительным целым")
    # Общая валидация идентификаторов и уникальности, без использования ответов.
    checked = candidates[["sku", "normquery"]].assign(score=0.)
    ranking_metrics(checked, checked.iloc[:0][["sku", "normquery"]])
    values = {}
    for channel in channels:
        if channel not in CHANNELS:
            raise ValueError("Неизвестный канал")
        column = CHANNELS[channel]
        if column not in candidates:
            raise ValueError(f"Нет признака {column}")
        scores = candidates[column].to_numpy(dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("Неконечные оценки канала")
        ordered = candidates[["sku", "normquery", column]].reset_index(drop=True)
        ordered = ordered.loc[scores > 0].sort_values(
            ["sku", column, "normquery"], ascending=[True, False, True], kind="stable"
        )
        ranks = ordered.groupby("sku").cumcount() + 1
        array = np.zeros(len(candidates))
        array[ordered.index] = 1. / (offset + ranks)
        values[channel] = array
    return values


def fusion_scores(candidates, weights, offset=60):
    if not weights or any(not np.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("Нужны конечные неотрицательные веса")
    if not np.isclose(sum(weights.values()), 1.):
        raise ValueError("Сумма весов должна быть 1")
    ranks = reciprocal_ranks(candidates, weights, offset)
    return sum(weights[channel] * ranks[channel] for channel in weights)


def fit_fusion(candidates, positives, channels, offset=60):
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("Нужны различные каналы")
    ranks = reciprocal_ranks(candidates, channels, offset)
    trials = []
    for units in itertools.product(range(5), repeat=len(channels)):
        if sum(units) != 4:
            continue
        weights = dict(zip(channels, [unit / 4 for unit in units], strict=True))
        prediction = candidates[["sku", "normquery"]].copy()
        prediction["score"] = sum(weights[channel] * ranks[channel] for channel in channels)
        metrics = ranking_metrics(prediction, positives)
        if metrics["ndcg@10"] is None:
            raise ValueError("Нет положительных validation-пар")
        trials.append({"weights": weights, "metrics": metrics})
    # Детерминированное разрешение ничьих в заранее фиксированной сетке.
    best = max(trials, key=lambda trial: trial["metrics"]["ndcg@10"])
    return {"weights": best["weights"], "offset": offset,
            "selection": "validation NDCG@10 on supplied positives", "trials": trials}


def paired_metric_interval(predictions, baseline, positives, metric="recall@10",
                           samples=2000, seed=7):
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("Нужно положительное число bootstrap-выборок")
    actual = ranking_metrics(predictions, positives)
    ranking_metrics(baseline, positives)
    if metric not in actual or metric in {"evaluated_skus", "positive_pairs", "candidate_pairs"}:
        raise ValueError("Неизвестная метрика")
    skus = sorted(positives["sku"].unique())
    if not skus:
        raise ValueError("Нет SKU для bootstrap")
    differences = []
    for sku in skus:
        target = positives.loc[positives["sku"].eq(sku)]
        left = ranking_metrics(predictions.loc[predictions["sku"].eq(sku)], target)[metric]
        right = ranking_metrics(baseline.loc[baseline["sku"].eq(sku)], target)[metric]
        differences.append(left - right)
    values = np.asarray(differences)
    indices = np.random.default_rng(seed).integers(0, len(skus), size=(samples, len(skus)))
    return {"metric": metric, "difference": float(values.mean()),
            "95_interval": np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist(),
            "method": "paired SKU bootstrap", "samples": samples, "seed": seed, "skus": len(skus)}


def _validate_keys(frame: pd.DataFrame, unique: bool = True) -> None:
    require_columns(frame, ["sku", "normquery"])
    keys = frame[["sku", "normquery"]]
    if keys.isna().any().any() or any(
        keys[name].astype(str).str.strip().eq("").any() for name in keys
    ):
        raise ValueError("SKU и normquery должны быть заполнены")
    if unique and keys.duplicated().any():
        raise ValueError("Кандидаты должны быть уникальны по SKU + normquery")


def features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Семь сигналов и 1/(10 + rank), ties по normquery, исходный index.

    Неконечные/нечисловые сигналы заменяются нулём. Ранги считаются отдельно
    внутри каждого SKU по полному candidate pool, а не по sampled train rows.
    """
    _validate_keys(candidates)
    require_columns(candidates, _BASE_FEATURES)
    result = (
        candidates[_BASE_FEATURES].apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], 0).fillna(0).astype(float)
    )
    keys = candidates[["sku", "normquery"]].reset_index(drop=True)
    for name in _BASE_FEATURES:
        ordered = keys.assign(_value=result[name].to_numpy()).sort_values(
            ["sku", "_value", "normquery"], ascending=[True, False, True]
        )
        ranks = ordered.groupby("sku", sort=False).cumcount().to_numpy() + 1
        reciprocal = np.empty(len(candidates), dtype=float)
        reciprocal[ordered.index.to_numpy()] = 1.0 / (10 + ranks)
        result[f"{name}_rr"] = reciprocal
    return result[LEARNING_FEATURES]


def learning_rows(
    candidates: pd.DataFrame,
    positives: pd.DataFrame,
    future: pd.DataFrame | None = None,
    group_prefix: str = "train",
    unlabelled_per_positive: int = 12,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict]:
    """Сохраняет найденные targets без injection; unknown остаются unknown.

    Половина бюджета — hard lexical/embedding/graph, остаток — random из
    остальных кандидатов. RNG отдельный для SKU, не зависит от порядка строк.
    Полный набор positives нужен для аудита missing, включая отсутствующие SKU.
    future намеренно не определяет labels: даже 20+ views не доказывают negative.
    label_status: positive / unlabelled_comparison. Обучаются только группы
    с найденными positives и хотя бы одним компаратором.
    """
    if (
        isinstance(unlabelled_per_positive, bool)
        or not isinstance(unlabelled_per_positive, int)
        or not 1 <= unlabelled_per_positive <= 12
    ):
        raise ValueError("unlabelled_per_positive должен быть целым числом от 1 до 12")
    _validate_keys(candidates)
    _validate_keys(positives, unique=False)
    require_columns(candidates, RANK_FEATURES)
    targets = positives[["sku", "normquery"]].drop_duplicates()
    data = candidates.reset_index(drop=True).copy()
    data[LEARNING_FEATURES] = features(data)
    target_keys = pd.MultiIndex.from_frame(targets)
    found = pd.MultiIndex.from_frame(data[["sku", "normquery"]]).isin(target_keys)
    data["label"] = found.astype(int)
    data["label_status"] = np.where(found, "positive", "unlabelled_comparison")
    data["group_id"] = group_prefix + ":" + data["sku"].astype(str)
    hard_scores = np.column_stack([
        np.where(data[name].to_numpy() > 0, data[f"{name}_rr"].to_numpy(), 0)
        for name in _HARD_FEATURES
    ]).max(axis=1)
    chosen = []
    for sku, group in data.groupby("sku", sort=True):
        positive = group.loc[group["label"].eq(1)]
        unknown = group.loc[group["label"].eq(0)].sort_values("normquery")
        if positive.empty or unknown.empty:
            continue
        limit = min(len(unknown), unlabelled_per_positive * len(positive))
        hard = unknown.assign(_hardness=hard_scores[unknown.index]).sort_values(
            ["_hardness", "normquery"], ascending=[False, True]
        ).head((limit + 1) // 2)
        remaining = unknown.loc[~unknown.index.isin(hard.index)]
        number = int.from_bytes(
            hashlib.sha256(f"{seed}:{sku}".encode()).digest()[:8], "big"
        )
        rng = np.random.default_rng(number)
        random = rng.choice(remaining.index.to_numpy(), size=limit - len(hard), replace=False)
        chosen.extend(positive.index)
        chosen.extend(hard.index)
        chosen.extend(random)
    rows = data.loc[chosen].sort_values(["sku", "normquery"]).reset_index(drop=True)
    return rows, {
        "target_count": len(targets),
        "found": int(found.sum()),
        "missing": len(targets) - int(found.sum()),
        "unlabelled_sampled": int(rows["label"].eq(0).sum()),
        "training_rows": len(rows),
        "training_groups": int(rows["group_id"].nunique()),
        "confirmed_negative": 0,
    }


def build_pairs(rows: pd.DataFrame) -> tuple[list[tuple[int, int]], list[float]]:
    """Все positive -> unlabelled пары, positional indices, вес SKU суммарно 1.

    Вес каждой слабой пары равен 1/(число positives * число компараторов).
    Это нормировка влияния SKU, не мера достоверности отрицательного label.
    """
    require_columns(rows, ["sku", "normquery", "group_id", "label", "label_status"])
    if rows["group_id"].isna().any() or not rows["label"].isin([0, 1]).all():
        raise ValueError("Для пар нужны group_id и labels 0/1")
    if not rows.loc[rows["label"].eq(0), "label_status"].eq("unlabelled_comparison").all():
        raise ValueError("Label 0 должен иметь status unlabelled_comparison")
    if not rows.loc[rows["label"].eq(1), "label_status"].eq("positive").all():
        raise ValueError("Label 1 должен иметь status positive")
    pairs = []
    weights = []
    for _, group in rows.reset_index(drop=True).groupby("group_id", sort=False):
        if group["sku"].nunique() != 1:
            raise ValueError("Одна группа должна соответствовать одному SKU")
        if group.duplicated(["sku", "normquery"]).any():
            raise ValueError("Обучающие пары SKU + normquery должны быть уникальны")
        positive = group.index[group["label"].eq(1)].tolist()
        unknown = group.index[group["label"].eq(0)].tolist()
        if not positive or not unknown:
            raise ValueError("Обучающей группе нужны positive и unlabelled_comparison")
        weight = 1.0 / (len(positive) * len(unknown))
        for winner in positive:
            for loser in unknown:
                pairs.append((winner, loser))
                weights.append(weight)
    return pairs, weights


def fit_ranker(rows: pd.DataFrame, iterations: int, seed: int = 7) -> CatBoostRanker:
    """Настоящий PairLogit по слабым парам; rows получены из learning_rows.

    Материализованные ranks сохраняют полный candidate pool до sampling.
    Scores модели являются retrieval priority, не вероятностью релевантности.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations должен быть положительным целым числом")
    if rows.empty:
        raise ValueError("Нет обучающих групп с positive и unlabelled_comparison")
    require_columns(rows, ["group_id", "normquery", *LEARNING_FEATURES])
    ordered = rows.sort_values(["group_id", "normquery"]).reset_index(drop=True)
    pairs, weights = build_pairs(ordered)
    values = ordered[LEARNING_FEATURES].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Обучающие признаки должны быть конечными числами")
    pool = Pool(
        values,
        label=ordered["label"],
        group_id=ordered["group_id"].tolist(),
        pairs=pairs,
        pairs_weight=weights,
    )
    model = CatBoostRanker(
        loss_function="PairLogit",
        iterations=iterations,
        depth=4,
        learning_rate=0.05,
        random_seed=seed,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(pool)
    return model


def predict_ranker(model: CatBoostRanker, candidates: pd.DataFrame) -> np.ndarray:
    """Retrieval scores в исходном порядке кандидатов, не probabilities."""
    values = features(candidates)
    if values.empty:
        return np.empty(0, dtype=float)
    return np.asarray(model.predict(values), dtype=float).reshape(-1)