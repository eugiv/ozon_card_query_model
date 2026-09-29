"""Оценка ранжирования, калибровки и итоговой политики рекомендаций."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class PolicyThresholds:
    keep: float
    remove: float | None
    keep_precision_target: float
    remove_precision_target: float
    min_remove_searches: int
    min_remove_views: int

    def to_dict(self) -> dict:
        return asdict(self)


class IsotonicScoreCalibrator:
    """Калибрует score на отдельном временном окне."""

    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.fitted = False

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> IsotonicScoreCalibrator:
        self.model.fit(np.asarray(scores, dtype=float), np.asarray(y_true, dtype=int))
        self.fitted = True
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("calibrator is not fitted")
        return np.asarray(
            self.model.predict(np.asarray(scores, dtype=float)), dtype=float
        )


def expected_calibration_error(
    y_true: np.ndarray,
    scores: np.ndarray,
    bins: int = 10,
) -> float:
    y = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(y), 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (probability >= lower) & (probability <= upper)
        else:
            mask = (probability >= lower) & (probability < upper)
        if not mask.any():
            continue
        error += (
            mask.sum()
            / total
            * abs(float(y[mask].mean()) - float(probability[mask].mean()))
        )
    return float(error)


def score_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> dict:
    y = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    prediction = probability >= threshold
    roc_auc = None
    if len(np.unique(y)) > 1:
        roc_auc = float(roc_auc_score(y, probability))
    return {
        "rows": len(y),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "roc_auc": roc_auc,
        "pr_auc": float(average_precision_score(y, probability)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(
            log_loss(y, np.column_stack([1 - probability, probability]), labels=[0, 1])
        ),
        "ece_10": expected_calibration_error(y, probability, bins=10),
        "threshold": float(threshold),
    }


def choose_policy_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    remove_eligible: np.ndarray,
    keep_precision_target: float = 0.60,
    remove_precision_target: float = 0.95,
    min_rows: int = 50,
    min_remove_searches: int = 100,
    min_remove_views: int = 3,
) -> PolicyThresholds:
    """Выбрать безопасные пороги на validation-окне.

    Keep получает самый широкий охват при заданной точности положительного решения.
    Remove получает самый широкий охват при заданной точности отрицательного решения.
    """

    y = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    eligible = np.asarray(remove_eligible, dtype=bool)
    grid = np.linspace(0.01, 0.99, 99)

    keep_threshold = 0.99
    for threshold in grid:
        selected = probability >= threshold
        if selected.sum() < min_rows:
            continue
        precision = float(y[selected].mean())
        if precision >= keep_precision_target:
            keep_threshold = float(threshold)
            break

    remove_threshold = None
    for threshold in grid:
        selected = eligible & (probability <= threshold)
        if selected.sum() < min_rows:
            continue
        negative_precision = float((y[selected] == 0).mean())
        if negative_precision >= remove_precision_target and threshold < keep_threshold:
            remove_threshold = float(threshold)

    return PolicyThresholds(
        keep=keep_threshold,
        remove=remove_threshold,
        keep_precision_target=keep_precision_target,
        remove_precision_target=remove_precision_target,
        min_remove_searches=min_remove_searches,
        min_remove_views=min_remove_views,
    )


def apply_model_policy(
    frame: pd.DataFrame,
    scores: np.ndarray,
    thresholds: PolicyThresholds,
) -> pd.DataFrame:
    out = frame.copy()
    out["score"] = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    out["decision"] = "review"
    out["decision_reason"] = "Недостаточно уверенности для автоматического решения"

    past_order = out["orders_past"] > 0
    predicted_keep = out["score"] >= thresholds.keep
    out.loc[past_order | predicted_keep, "decision"] = "keep"
    out.loc[past_order, "decision_reason"] = "По запросу уже были заказы"
    out.loc[~past_order & predicted_keep, "decision_reason"] = (
        "Высокая оценка будущего заказа"
    )

    remove_eligible = (
        ~past_order
        & (out["freq_past"] >= thresholds.min_remove_searches)
        & (out["views_past"] >= thresholds.min_remove_views)
    )
    if thresholds.remove is not None:
        candidate = remove_eligible & (out["score"] <= thresholds.remove)
        out.loc[candidate, "decision"] = "remove_candidate"
        out.loc[candidate, "decision_reason"] = (
            "Достаточно поисков и переходов, но низкая оценка будущего заказа"
        )
    return out


def policy_metrics(frame: pd.DataFrame) -> dict:
    y = frame["label"].to_numpy(dtype=int)
    decision = frame["decision"].astype(str)
    keep = decision.eq("keep").to_numpy()
    remove = decision.eq("remove_candidate").to_numpy()
    review = decision.eq("review").to_numpy()
    positives = int(y.sum())
    negatives = int((y == 0).sum())
    return {
        "rows": len(frame),
        "positives": positives,
        "prevalence": float(y.mean()) if len(y) else None,
        "keep": int(keep.sum()),
        "review": int(review.sum()),
        "remove_candidate": int(remove.sum()),
        "automatic_coverage": float((keep | remove).mean()) if len(frame) else 0.0,
        "review_share": float(review.mean()) if len(frame) else 0.0,
        "keep_precision": float(y[keep].mean()) if keep.any() else None,
        "keep_recall": float(y[keep].sum() / positives) if positives else None,
        "remove_precision": float((y[remove] == 0).mean()) if remove.any() else None,
        "negative_coverage": float((y[remove] == 0).sum() / negatives)
        if negatives
        else None,
        "positive_removed": int(y[remove].sum()),
        "positive_removed_share": float(y[remove].sum() / positives)
        if positives
        else None,
    }


def _bootstrap_summary(frame: pd.DataFrame) -> dict[str, float]:
    scored = frame.loc[frame["score"].notna()]
    result = policy_metrics(frame)
    if len(scored) and scored["label"].nunique() > 1:
        result["pr_auc"] = float(
            average_precision_score(scored["label"], scored["score"])
        )
        result["brier"] = float(brier_score_loss(scored["label"], scored["score"]))
    else:
        result["pr_auc"] = np.nan
        result["brier"] = np.nan
    return result


def cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    cluster_column: str = "sku",
    iterations: int = 300,
    seed: int = 7,
) -> dict:
    """Доверительные интервалы с пересэмплированием целых SKU."""

    if frame.empty:
        return {}
    clusters = frame[cluster_column].drop_duplicates().astype(str).to_numpy()
    groups = {
        str(key): value for key, value in frame.groupby(cluster_column, sort=False)
    }
    point = _bootstrap_summary(frame)
    samples: dict[str, list[float]] = {key: [] for key in point}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        sampled = pd.concat(
            [groups[str(cluster)] for cluster in selected], ignore_index=True
        )
        metrics = _bootstrap_summary(sampled)
        for key, value in metrics.items():
            if value is not None and np.isfinite(value):
                samples[key].append(float(value))

    output = {}
    for key, value in point.items():
        values = samples.get(key, [])
        if value is None or not values:
            output[key] = {"value": value, "low": None, "high": None}
        else:
            output[key] = {
                "value": float(value),
                "low": float(np.quantile(values, 0.025)),
                "high": float(np.quantile(values, 0.975)),
            }
    return output


def curve_rows(y_true: np.ndarray, scores: np.ndarray, model_name: str) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(scores, dtype=float)
    false_positive, true_positive, _ = roc_curve(y, probability)
    precision, recall, _ = precision_recall_curve(y, probability)
    roc_rows = pd.DataFrame(
        {
            "curve": "roc",
            "model_name": model_name,
            "x": false_positive,
            "y": true_positive,
        }
    )
    pr_rows = pd.DataFrame(
        {
            "curve": "precision_recall",
            "model_name": model_name,
            "x": recall,
            "y": precision,
        }
    )
    return pd.concat([roc_rows, pr_rows], ignore_index=True)


def rolling_metric(
    frames: list[pd.DataFrame],
    evaluator: Callable[[pd.DataFrame], dict],
) -> dict:
    values = [evaluator(frame) for frame in frames if not frame.empty]
    return {"folds": values, "count": len(values)}
