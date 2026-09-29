"""Честный временной benchmark и безопасные рекомендации по поисковым запросам."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import joblib
import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from dotenv import dotenv_values
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from artifacts import write_report_artifacts
from compose import compose_recommendations
from evaluation import (
    IsotonicScoreCalibrator,
    PolicyThresholds,
    apply_model_policy,
    choose_policy_thresholds,
    cluster_bootstrap,
    curve_rows,
    policy_metrics,
    score_metrics,
)
from features import MODEL_FEATURES, attach_features
from sparse_rules import decide_sparse_card

logger = logging.getLogger("ozon_card_query_model")

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
ENV_PATH = Path(
    os.environ.get("OZON_CARD_ENV", ROOT.parent / "clickhouse_dbt" / ".env")
)

RANDOM_SEED = 7
WINDOW_DAYS = 28
BLOCK_COUNT = 7
DEFAULT_EVALUATION_CUTOFF = date(2026, 9, 25)
MIN_MODEL_POSITIVE_PAIRS = 3
MIN_CANDIDATE_SEARCHES = 10
MIN_CANDIDATE_VIEWS = 3
MIN_REMOVE_SEARCHES = 100
MIN_REMOVE_VIEWS = 3
KEEP_PRECISION_TARGET = 0.60
REMOVE_PRECISION_TARGET = 0.95

GRID = [
    {"depth": 4, "learning_rate": 0.08, "iterations": 400},
    {"depth": 6, "learning_rate": 0.05, "iterations": 400},
    {"depth": 6, "learning_rate": 0.1, "iterations": 300},
]

PAIR_SQL = """
WITH daily AS (
    SELECT
        date,
        sku,
        lowerUTF8(query) AS query,
        order_count AS orders,
        unique_search_users AS freq,
        unique_view_users AS views
    FROM default_staging.stg_ozon_product_queries_details
    WHERE query != ''
      AND date >= toDate('{feat_start}')
      AND date <= toDate('{label_end}')
),
pairs AS (
    SELECT
        sku,
        query,
        sumIf(orders, date BETWEEN toDate('{feat_start}') AND toDate('{feat_end}')) AS orders_past,
        sumIf(freq, date BETWEEN toDate('{feat_start}') AND toDate('{feat_end}')) AS freq_past,
        sumIf(views, date BETWEEN toDate('{feat_start}') AND toDate('{feat_end}')) AS views_past,
        countIf(date BETWEEN toDate('{feat_start}') AND toDate('{feat_end}')) AS days_past,
        sumIf(orders, date BETWEEN toDate('{label_start}') AND toDate('{label_end}')) AS orders_y,
        sumIf(freq, date BETWEEN toDate('{label_start}') AND toDate('{label_end}')) AS freq_y,
        sumIf(views, date BETWEEN toDate('{label_start}') AND toDate('{label_end}')) AS views_y
    FROM daily
    GROUP BY sku, query
)
SELECT *
FROM pairs
WHERE orders_past > 0
   OR freq_past >= {min_candidate_searches}
   OR views_past >= {min_candidate_views}
"""

SCORE_SQL = """
SELECT
    sku,
    lowerUTF8(query) AS query,
    sum(order_count) AS orders_past,
    sum(unique_search_users) AS freq_past,
    sum(unique_view_users) AS views_past,
    count() AS days_past,
    0 AS orders_y,
    0 AS freq_y,
    0 AS views_y
FROM default_staging.stg_ozon_product_queries_details
WHERE query != ''
  AND date >= toDate('{feat_start}')
  AND date <= toDate('{feat_end}')
GROUP BY sku, query
HAVING orders_past > 0
    OR freq_past >= {min_candidate_searches}
    OR views_past >= {min_candidate_views}
"""


@dataclass(frozen=True)
class Window:
    feat_start: str
    feat_end: str
    label_start: str
    label_end: str

    def to_dict(self) -> dict[str, str]:
        return {
            "feat_start": self.feat_start,
            "feat_end": self.feat_end,
            "label_start": self.label_start,
            "label_end": self.label_end,
        }


def span(end: date, shift_days: int) -> tuple[date, date]:
    window_end = end - timedelta(days=shift_days)
    window_start = window_end - timedelta(days=WINDOW_DAYS - 1)
    return window_start, window_end


def build_windows(cutoff: date, block_count: int = BLOCK_COUNT) -> dict[str, Window]:
    windows = {}
    for index in range(block_count):
        label_start, label_end = span(cutoff, 2 * index * WINDOW_DAYS)
        feat_start, feat_end = span(cutoff, (2 * index + 1) * WINDOW_DAYS)
        windows[f"block_{index}"] = Window(
            feat_start=feat_start.isoformat(),
            feat_end=feat_end.isoformat(),
            label_start=label_start.isoformat(),
            label_end=label_end.isoformat(),
        )
    return windows


def score_window(max_date: date) -> dict[str, str]:
    feat_start, feat_end = span(max_date, 0)
    return {"feat_start": feat_start.isoformat(), "feat_end": feat_end.isoformat()}


def client():
    import clickhouse_connect

    env = dotenv_values(ENV_PATH)
    secure = str(env.get("DBT_CLICKHOUSE_SECURE", "false")).lower() == "true"
    verify = str(env.get("DBT_CLICKHOUSE_VERIFY", "false")).lower() == "true"
    return clickhouse_connect.get_client(
        host=env["DBT_CLICKHOUSE_HOST"],
        port=int(env.get("DBT_CLICKHOUSE_PORT", "8123")),
        username=env.get("DBT_CLICKHOUSE_USER", "default"),
        password=env.get("DBT_CLICKHOUSE_PASSWORD", ""),
        secure=secure,
        verify=verify,
        connect_timeout=30,
        send_receive_timeout=900,
    )


def load_frame(ch, sql: str) -> pd.DataFrame:
    frame = ch.query_df(sql)
    if frame.empty:
        return frame
    frame["sku"] = frame["sku"].astype(str)
    frame["query"] = frame["query"].fillna("").astype(str)
    return frame.sort_values(["sku", "query"]).reset_index(drop=True)


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = attach_features(frame)
    result["future_exposed"] = (
        (result["orders_y"] > 0)
        | (result["freq_y"] >= MIN_CANDIDATE_SEARCHES)
        | (result["views_y"] >= MIN_CANDIDATE_VIEWS)
    )
    return result


def eligible_skus(frame: pd.DataFrame) -> set[str]:
    if frame.empty:
        return set()
    positive_pairs = frame.loc[frame["orders_past"] > 0].groupby("sku").size()
    return set(
        positive_pairs[positive_pairs >= MIN_MODEL_POSITIVE_PAIRS].index.astype(str)
    )


def model_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["sku"].isin(eligible_skus(frame))].copy()


def concat_nonempty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    available = [frame for frame in frames if not frame.empty]
    if not available:
        return pd.DataFrame()
    return pd.concat(available, ignore_index=True)


def fit_catboost(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    params: dict,
    val_x: pd.DataFrame | None = None,
    val_y: pd.Series | None = None,
) -> CatBoostClassifier:
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        **params,
    )
    if val_x is not None and val_y is not None:
        model.fit(
            train_x,
            train_y,
            eval_set=(val_x, val_y),
            early_stopping_rounds=40,
            use_best_model=True,
        )
    else:
        model.fit(train_x, train_y)
    return model


def tune_model(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    best_params = None
    best_pr_auc = -1.0
    best_iterations = 100
    for params in GRID:
        model = fit_catboost(
            train[MODEL_FEATURES],
            train["label"],
            params,
            validation[MODEL_FEATURES],
            validation["label"],
        )
        scores = model.predict_proba(validation[MODEL_FEATURES])[:, 1]
        pr_auc = float(average_precision_score(validation["label"], scores))
        best_iteration = model.get_best_iteration()
        iterations = max(
            int(
                best_iteration
                if best_iteration is not None and best_iteration >= 0
                else params["iterations"]
            ),
            50,
        )
        logger.info(
            "grid params=%s val_pr_auc=%.4f iterations=%d", params, pr_auc, iterations
        )
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_params = params
            best_iterations = iterations
    if best_params is None:
        raise RuntimeError("could not select CatBoost parameters")
    return {**best_params, "iterations": best_iterations}


def remove_eligible_mask(frame: pd.DataFrame) -> np.ndarray:
    return (
        (frame["orders_past"] <= 0)
        & (frame["freq_past"] >= MIN_REMOVE_SEARCHES)
        & (frame["views_past"] >= MIN_REMOVE_VIEWS)
    ).to_numpy()


def fit_calibrator_and_thresholds(
    model,
    validation: pd.DataFrame,
) -> tuple[IsotonicScoreCalibrator, PolicyThresholds, np.ndarray]:
    raw = model.predict_proba(validation[MODEL_FEATURES])[:, 1]
    calibrator = IsotonicScoreCalibrator().fit(raw, validation["label"].to_numpy())
    calibrated = calibrator.predict(raw)
    thresholds = choose_policy_thresholds(
        validation["label"].to_numpy(),
        calibrated,
        remove_eligible=remove_eligible_mask(validation),
        keep_precision_target=KEEP_PRECISION_TARGET,
        remove_precision_target=REMOVE_PRECISION_TARGET,
        min_remove_searches=MIN_REMOVE_SEARCHES,
        min_remove_views=MIN_REMOVE_VIEWS,
    )
    return calibrator, thresholds, calibrated


def route_policy(
    frame: pd.DataFrame,
    model,
    calibrator: IsotonicScoreCalibrator,
    thresholds: PolicyThresholds,
) -> pd.DataFrame:
    eligible = eligible_skus(frame)
    model_frame = frame.loc[frame["sku"].isin(eligible)].copy()
    sparse_frame = frame.loc[~frame["sku"].isin(eligible)].copy()

    pieces = []
    if len(model_frame):
        raw = model.predict_proba(model_frame[MODEL_FEATURES])[:, 1]
        scored = apply_model_policy(model_frame, calibrator.predict(raw), thresholds)
        scored["source"] = "model"
        pieces.append(scored)
    if len(sparse_frame):
        sparse_frame["score"] = np.nan
        sparse_frame["decision"] = [
            decide_sparse_card(
                row.query, row.orders_past, row.freq_past, row.views_past
            )
            for row in sparse_frame.itertuples(index=False)
        ]
        sparse_frame["decision_reason"] = np.where(
            sparse_frame["orders_past"] > 0,
            "По запросу уже были заказы",
            "Для карточки мало положительных примеров; решение оставлено человеку",
        )
        sparse_frame["source"] = "sparse_rule"
        pieces.append(sparse_frame)
    if not pieces:
        return frame.copy()
    routed = pd.concat(pieces, ignore_index=True)
    return compose_recommendations(routed)


def evaluate_fold(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    params: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    train_model = model_rows(train)
    validation_model = model_rows(validation)
    test_model = model_rows(test)
    if min(len(train_model), len(validation_model), len(test_model)) == 0:
        raise ValueError("rolling fold has an empty model cohort")
    if any(
        frame["label"].nunique() < 2
        for frame in (train_model, validation_model, test_model)
    ):
        raise ValueError("rolling fold requires both target classes")

    model = fit_catboost(train_model[MODEL_FEATURES], train_model["label"], params)
    calibrator, thresholds, _ = fit_calibrator_and_thresholds(model, validation_model)
    raw_test = model.predict_proba(test_model[MODEL_FEATURES])[:, 1]
    calibrated_test = calibrator.predict(raw_test)
    test_policy = route_policy(test, model, calibrator, thresholds)
    metrics = {
        "score": score_metrics(test_model["label"].to_numpy(), calibrated_test),
        "policy": policy_metrics(test_policy),
        "thresholds": thresholds.to_dict(),
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "model_test": len(test_model),
        },
    }
    test_model_predictions = test_model.copy()
    test_model_predictions["score"] = calibrated_test
    return metrics, test_policy, test_model_predictions


def rolling_backtests(blocks: dict[int, pd.DataFrame], params: dict) -> list[dict]:
    results = []
    for test_index in (2, 1):
        validation_index = test_index + 1
        train_indices = [index for index in sorted(blocks) if index > validation_index]
        train = concat_nonempty([blocks[index] for index in train_indices])
        validation = blocks.get(validation_index, pd.DataFrame())
        test = blocks.get(test_index, pd.DataFrame())
        if min(len(train), len(validation), len(test)) == 0:
            continue
        try:
            metrics, _, _ = evaluate_fold(train, validation, test, params)
        except ValueError as exc:
            logger.warning("skip rolling fold test_index=%d: %s", test_index, exc)
            continue
        metrics["test_block"] = test_index
        metrics["validation_block"] = validation_index
        metrics["train_blocks"] = train_indices
        results.append(metrics)
    return results


def baseline_metrics(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict, dict[str, np.ndarray]]:
    train_model = model_rows(train)
    test_model = model_rows(test)
    y_test = test_model["label"].to_numpy()
    past_order = (test_model["orders_past"] > 0).to_numpy(dtype=float)
    always_keep = np.ones(len(test_model), dtype=float)

    logistic = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    logistic.fit(train_model[MODEL_FEATURES], train_model["label"])
    logistic_scores = logistic.predict_proba(test_model[MODEL_FEATURES])[:, 1]

    past_order_policy = test.copy()
    past_order_policy["decision"] = np.where(
        past_order_policy["orders_past"] > 0,
        "keep",
        "remove_candidate",
    )
    always_keep_policy = test.copy()
    always_keep_policy["decision"] = "keep"
    metrics = {
        "past_order": score_metrics(y_test, past_order),
        "always_keep": score_metrics(y_test, always_keep),
        "logistic": score_metrics(y_test, logistic_scores),
        "past_order_full_policy": policy_metrics(past_order_policy),
        "always_keep_full_policy": policy_metrics(always_keep_policy),
    }
    return metrics, {
        "past_order": past_order,
        "logistic": logistic_scores,
    }


def _window_key(window: Window) -> tuple[str, str, str, str]:
    return (
        window.feat_start,
        window.feat_end,
        window.label_start,
        window.label_end,
    )


def load_blocks(
    ch, windows: dict[str, Window], cache: dict | None = None
) -> dict[int, pd.DataFrame]:
    cache = cache if cache is not None else {}
    result = {}
    for name, window in windows.items():
        key = _window_key(window)
        if key not in cache:
            sql = PAIR_SQL.format(
                **window.to_dict(),
                min_candidate_searches=MIN_CANDIDATE_SEARCHES,
                min_candidate_views=MIN_CANDIDATE_VIEWS,
            )
            cache[key] = prepare_frame(load_frame(ch, sql))
        result[int(name.split("_")[1])] = cache[key]
    return result


def _count_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"pairs": 0, "positives": 0, "sku": 0, "future_exposed": 0}
    return {
        "pairs": len(frame),
        "positives": int(frame["label"].sum()),
        "sku": int(frame["sku"].nunique()),
        "future_exposed": int(frame["future_exposed"].sum()),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        return {
            "commit": commit,
            "dirty": bool(diff),
            "diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": type(exc).__name__}


def _package_versions() -> dict[str, str]:
    names = ["catboost", "mlflow", "numpy", "pandas", "scikit-learn"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    versions["python"] = sys.version.split()[0]
    return versions


def _publish_latest(run_dir: Path, artifact_id: str) -> None:
    compatibility_files = [
        "metrics.json",
        "recommendations.csv",
        "recommendations.xlsx",
        "test_predictions.csv",
        "model_curves.csv",
        "deployment_model.cbm",
        "deployment_calibrator.joblib",
    ]
    for name in compatibility_files:
        source = run_dir / name
        destination = REPORT_DIR / name
        temporary = destination.with_name(destination.name + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    latest = {
        "artifact_id": artifact_id,
        "run_dir": str(run_dir),
        "manifest_sha256": _sha256(run_dir / "manifest.json"),
    }
    temporary_latest = REPORT_DIR / "latest.tmp.json"
    _write_json(temporary_latest, latest)
    temporary_latest.replace(REPORT_DIR / "latest.json")


def run(cutoff: date, bootstrap_iterations: int = 300) -> dict:
    os.chdir(ROOT)
    REPORT_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("ozon_card_query")

    ch = client()
    max_date = pd.Timestamp(
        ch.query_df(
            "SELECT max(date) AS max_date "
            "FROM default_staging.stg_ozon_product_queries_details"
        )["max_date"].iloc[0]
    ).date()
    if cutoff > max_date:
        raise ValueError(
            f"evaluation cutoff {cutoff} is after source max_date {max_date}"
        )

    cache: dict = {}
    benchmark_windows = build_windows(cutoff)
    benchmark_blocks = load_blocks(ch, benchmark_windows, cache)
    train = concat_nonempty(
        [benchmark_blocks[index] for index in sorted(benchmark_blocks) if index >= 2]
    )
    validation = benchmark_blocks[1]
    test = benchmark_blocks[0]
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        if frame.empty:
            raise RuntimeError(f"{name} cohort is empty")

    params = tune_model(model_rows(train), model_rows(validation))
    benchmark_model = fit_catboost(
        model_rows(train)[MODEL_FEATURES],
        model_rows(train)["label"],
        params,
    )
    benchmark_calibrator, benchmark_thresholds, _ = fit_calibrator_and_thresholds(
        benchmark_model,
        model_rows(validation),
    )
    raw_test = benchmark_model.predict_proba(model_rows(test)[MODEL_FEATURES])[:, 1]
    calibrated_test = benchmark_calibrator.predict(raw_test)
    test_policy = route_policy(
        test,
        benchmark_model,
        benchmark_calibrator,
        benchmark_thresholds,
    )

    baselines, baseline_scores = baseline_metrics(train, test)
    main_metrics = {
        "score": score_metrics(model_rows(test)["label"].to_numpy(), calibrated_test),
        "policy": policy_metrics(test_policy),
        "thresholds": benchmark_thresholds.to_dict(),
    }
    bootstrap = cluster_bootstrap(
        test_policy,
        iterations=bootstrap_iterations,
        seed=RANDOM_SEED,
    )
    rolling = rolling_backtests(benchmark_blocks, params)

    production_windows = build_windows(max_date)
    production_blocks = load_blocks(ch, production_windows, cache)
    production_train = concat_nonempty(
        [production_blocks[index] for index in sorted(production_blocks) if index >= 1]
    )
    production_calibration = production_blocks[0]
    production_model = fit_catboost(
        model_rows(production_train)[MODEL_FEATURES],
        model_rows(production_train)["label"],
        params,
    )
    production_calibrator, production_thresholds, _ = fit_calibrator_and_thresholds(
        production_model,
        model_rows(production_calibration),
    )

    current_window = score_window(max_date)
    score_sql = SCORE_SQL.format(
        **current_window,
        min_candidate_searches=MIN_CANDIDATE_SEARCHES,
        min_candidate_views=MIN_CANDIDATE_VIEWS,
    )
    score_frame = prepare_frame(load_frame(ch, score_sql))
    recommendations = route_policy(
        score_frame,
        production_model,
        production_calibrator,
        production_thresholds,
    )

    curves = pd.concat(
        [
            curve_rows(
                model_rows(test)["label"].to_numpy(),
                calibrated_test,
                "Модель по истории",
            ),
            curve_rows(
                model_rows(test)["label"].to_numpy(),
                baseline_scores["logistic"],
                "Логистическая модель",
            ),
            curve_rows(
                model_rows(test)["label"].to_numpy(),
                baseline_scores["past_order"],
                "Только прошлые заказы",
            ),
        ],
        ignore_index=True,
    )

    artifact_id = f"{cutoff.isoformat()}-{uuid.uuid4().hex[:8]}"
    recommendation_counts = (
        recommendations["decision"].value_counts().sort_index().to_dict()
    )
    summary = {
        "schema_version": 2,
        "artifact_id": artifact_id,
        "evaluation_cutoff": cutoff.isoformat(),
        "source_max_date": max_date.isoformat(),
        "window_days": WINDOW_DAYS,
        "candidate_rule": {
            "min_searches": MIN_CANDIDATE_SEARCHES,
            "min_views": MIN_CANDIDATE_VIEWS,
            "cohort_uses_future": False,
        },
        "historical_text_features": False,
        "model_features": MODEL_FEATURES,
        "params": params,
        "windows": {
            name: window.to_dict() for name, window in benchmark_windows.items()
        },
        "counts": {
            "train": _count_summary(train),
            "validation": _count_summary(validation),
            "test": _count_summary(test),
            "score": {
                "pairs": len(score_frame),
                "sku": int(score_frame["sku"].nunique()),
            },
        },
        "benchmark": {
            "catboost_history": main_metrics,
            "baselines": baselines,
            "bootstrap_by_sku": bootstrap,
            "rolling_backtests": rolling,
        },
        "deployment": {
            "training_blocks": [
                index for index in sorted(production_blocks) if index >= 1
            ],
            "calibration_block": 0,
            "thresholds": production_thresholds.to_dict(),
            "score_window": current_window,
        },
        "recommendations": recommendation_counts,
        "limitations": [
            "Модель прогнозирует будущий заказ, а не причинный эффект текста карточки.",
            "Исторические снимки текста отсутствуют, поэтому текстовые признаки отключены.",
            "Кандидат на исключение требует ручной проверки и не применяется автоматически.",
        ],
    }

    test_predictions = test_policy.sort_values(["sku", "query"]).reset_index(drop=True)
    with mlflow.start_run(run_name=f"honest_pipeline_{artifact_id}") as run:
        summary["mlflow_run_id"] = run.info.run_id
        runs_dir = REPORT_DIR / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        for stale_dir in runs_dir.glob(".*.tmp"):
            if stale_dir.is_dir():
                shutil.rmtree(stale_dir)
        run_dir = runs_dir / artifact_id
        temporary_run_dir = runs_dir / f".{artifact_id}.tmp"
        if run_dir.exists() or temporary_run_dir.exists():
            raise FileExistsError(f"artifact directory already exists: {artifact_id}")
        temporary_run_dir.mkdir()

        model_path = temporary_run_dir / "deployment_model.cbm"
        calibrator_path = temporary_run_dir / "deployment_calibrator.joblib"
        production_model.save_model(model_path)
        joblib.dump(production_calibrator, calibrator_path)
        artifact_paths = write_report_artifacts(
            temporary_run_dir,
            summary=summary,
            recommendations=recommendations,
            test_predictions=test_predictions,
            curves=curves,
        )
        _write_json(
            temporary_run_dir / "config.json",
            {
                "evaluation_cutoff": cutoff.isoformat(),
                "bootstrap_iterations": bootstrap_iterations,
                "random_seed": RANDOM_SEED,
                "window_days": WINDOW_DAYS,
                "candidate_searches": MIN_CANDIDATE_SEARCHES,
                "candidate_views": MIN_CANDIDATE_VIEWS,
                "keep_precision_target": KEEP_PRECISION_TARGET,
                "remove_precision_target": REMOVE_PRECISION_TARGET,
            },
        )
        _write_json(
            temporary_run_dir / "windows.json",
            {name: window.to_dict() for name, window in benchmark_windows.items()},
        )
        _write_json(
            temporary_run_dir / "data_manifest.json",
            {
                "source": "default_staging.stg_ozon_product_queries_details",
                "source_max_date": max_date.isoformat(),
                "counts": summary["counts"],
                "pair_sql_sha256": hashlib.sha256(PAIR_SQL.encode()).hexdigest(),
                "score_sql_sha256": hashlib.sha256(SCORE_SQL.encode()).hexdigest(),
            },
        )
        _write_json(
            temporary_run_dir / "feature_schema.json",
            {
                "features": MODEL_FEATURES,
                "historical_text_features": False,
            },
        )
        _write_json(
            temporary_run_dir / "policy.json",
            {
                "benchmark": benchmark_thresholds.to_dict(),
                "deployment": production_thresholds.to_dict(),
                "allow_automatic_card_edit": False,
                "remove_decision": "remove_candidate",
            },
        )

        file_hashes = {
            path.name: _sha256(path)
            for path in temporary_run_dir.iterdir()
            if path.is_file()
        }
        _write_json(
            temporary_run_dir / "manifest.json",
            {
                "artifact_id": artifact_id,
                "git": _git_metadata(),
                "packages": _package_versions(),
                "files": file_hashes,
            },
        )
        mlflow.log_params(
            {
                "artifact_id": artifact_id,
                "evaluation_cutoff": cutoff.isoformat(),
                "source_max_date": max_date.isoformat(),
                "features": ",".join(MODEL_FEATURES),
                **{f"catboost_{key}": value for key, value in params.items()},
            }
        )
        mlflow.log_metrics(
            {
                "test_pr_auc": main_metrics["score"]["pr_auc"],
                "test_brier": main_metrics["score"]["brier"],
                "policy_remove_precision": main_metrics["policy"]["remove_precision"]
                or 0.0,
                "policy_positive_removed_share": (
                    main_metrics["policy"]["positive_removed_share"] or 0.0
                ),
                "policy_review_share": main_metrics["policy"]["review_share"],
            }
        )
        for path in [*artifact_paths.values(), str(model_path), str(calibrator_path)]:
            mlflow.log_artifact(path)
        mlflow.log_artifact(str(temporary_run_dir / "manifest.json"))
        temporary_run_dir.replace(run_dir)
        _publish_latest(run_dir, artifact_id)

    logger.info(
        "pipeline complete artifact_id=%s run_dir=%s recommendations=%s",
        artifact_id,
        run_dir,
        recommendation_counts,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Ozon card-query model without future cohort leakage."
    )
    parser.add_argument(
        "--cutoff",
        type=date.fromisoformat,
        default=DEFAULT_EVALUATION_CUTOFF,
        help="Frozen end date of the final evaluation label window (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=300,
        help="Number of SKU-cluster bootstrap iterations.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)
    summary = run(args.cutoff, bootstrap_iterations=args.bootstrap_iterations)
    logger.info(
        "test PR-AUC=%.4f policy=%s",
        summary["benchmark"]["catboost_history"]["score"]["pr_auc"],
        json.dumps(
            summary["benchmark"]["catboost_history"]["policy"],
            ensure_ascii=False,
        ),
    )
