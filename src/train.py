"""Обучение CatBoost и сравнение с правилом «по фразе уже был заказ».

Разделение по времени, не случайное.
Признаки частоты и конверсии — прошлые недели.
Ответ «был заказ или нет» — следующие недели.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from dotenv import dotenv_values
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit

from features import SEMANTIC_FEATURES, STATS_FEATURES, SemanticEncoder, attach_features

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = Path(
    os.environ.get("OZON_CARD_ENV", ROOT.parent / "clickhouse_dbt" / ".env")
)
RANDOM_SEED = 7

TRAIN = {
    "feat_start": "2026-06-05",
    "feat_end": "2026-07-02",
    "label_start": "2026-07-03",
    "label_end": "2026-07-30",
}
TEST = {
    "feat_start": "2026-07-31",
    "feat_end": "2026-08-27",
    "label_start": "2026-08-28",
    "label_end": "2026-09-24",
}

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
        query,
        argMax(ifNull(order_count, 0), loaded_at) AS orders,
        argMax(ifNull(unique_search_users, 0), loaded_at) AS freq,
        argMax(ifNull(unique_view_users, 0), loaded_at) AS views,
        argMax(ifNull(position, 0), loaded_at) AS position
    FROM raw.ozon_product_queries_details
    WHERE query != ''
      AND date >= toDate('{feat_start}')
      AND date <= toDate('{label_end}')
    GROUP BY date, sku, query
),
pairs AS (
    SELECT
        sku,
        query,
        sumIf(orders, date >= toDate('{feat_start}') AND date <= toDate('{feat_end}')) AS orders_past,
        sumIf(freq, date >= toDate('{feat_start}') AND date <= toDate('{feat_end}')) AS freq_past,
        sumIf(views, date >= toDate('{feat_start}') AND date <= toDate('{feat_end}')) AS views_past,
        countIf(date >= toDate('{feat_start}') AND date <= toDate('{feat_end}')) AS days_past,
        avgIf(position, date >= toDate('{feat_start}') AND date <= toDate('{feat_end}') AND position > 0) AS position_past,
        sumIf(orders, date >= toDate('{label_start}') AND date <= toDate('{label_end}')) AS orders_y,
        sumIf(freq, date >= toDate('{label_start}') AND date <= toDate('{label_end}')) AS freq_y
    FROM daily
    GROUP BY sku, query
),
eligible AS (
    SELECT sku
    FROM pairs
    GROUP BY sku
    HAVING countIf(orders_y > 0) >= 3
)
SELECT
    p.sku AS sku,
    p.query AS query,
    p.orders_past AS orders_past,
    p.freq_past AS freq_past,
    p.views_past AS views_past,
    p.days_past AS days_past,
    p.position_past AS position_past,
    p.orders_y AS orders_y,
    p.freq_y AS freq_y
FROM pairs AS p
INNER JOIN eligible AS e ON p.sku = e.sku
WHERE p.orders_y > 0 OR p.freq_y >= 10
"""

CARD_SQL = """
SELECT
    sku,
    substringUTF8(arrayStringConcat(groupArray(val), ' '), 1, 1200) AS card_text
FROM (
    SELECT
        toString(sku) AS sku,
        JSONExtractString(value_raw, 'value') AS val
    FROM (
        SELECT
            sku,
            arrayJoin(JSONExtractArrayRaw(JSONExtractRaw(item, 'values'))) AS value_raw,
            JSONExtractInt(item, 'id') AS attr_id
        FROM (
            SELECT
                sku,
                arrayJoin(JSONExtractArrayRaw(attributes)) AS item
            FROM raw.ozon_product_info_attributes
            WHERE isValidJSON(attributes)
        )
        WHERE attr_id != 11254
          AND lengthUTF8(JSONExtractString(value_raw, 'value')) BETWEEN 1 AND 300
    )
)
GROUP BY sku
"""

BRAND_SQL = """
SELECT DISTINCT brand
FROM (
    SELECT replaceAll(lowerUTF8(trimBoth(JSONExtractString(value_raw, 'value'))), 'ё', 'е') AS brand
    FROM (
        SELECT arrayJoin(JSONExtractArrayRaw(JSONExtractRaw(item, 'values'))) AS value_raw
        FROM (
            SELECT arrayJoin(JSONExtractArrayRaw(attributes)) AS item
            FROM raw.ozon_product_info_attributes
            WHERE isValidJSON(attributes)
        )
        WHERE JSONExtractInt(item, 'id') = 85
    )
)
WHERE lengthUTF8(brand) >= 3
  AND brand NOT IN ('нет бренда', 'пустой бренд')
"""


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


def load_pairs(ch, window: dict) -> pd.DataFrame:
    frame = ch.query_df(PAIR_SQL.format(**window))
    frame["sku"] = frame["sku"].astype(str)
    return frame


def load_cards(ch) -> pd.DataFrame:
    cards = ch.query_df(CARD_SQL)
    cards["sku"] = cards["sku"].astype(str)
    cards["card_text"] = cards["card_text"].fillna("").astype(str)
    try:
        names = ch.query_df(
            "SELECT toString(sku) AS sku, ifNull(name, '') AS name "
            "FROM raw.ozon_product_info"
        )
        names["sku"] = names["sku"].astype(str)
        cards = cards.merge(names, on="sku", how="left")
        cards["card_text"] = (
            cards["name"].fillna("").astype(str) + " " + cards["card_text"]
        ).str.strip()
        cards = cards.drop(columns=["name"])
    except Exception as exc:
        print(f"product names skipped: {type(exc).__name__}")
    return cards.groupby("sku", as_index=False)["card_text"].agg(
        lambda series: " ".join(dict.fromkeys(series))
    )


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    best_t = 0.5
    best_f = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        score = f1_score(y_true, scores >= threshold, zero_division=0)
        if score > best_f:
            best_f = score
            best_t = float(threshold)
    return best_t


def classification_metrics(y_true, scores, threshold: float) -> dict:
    pred = scores >= threshold
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
    }


def fit_model(train_x, train_y, val_x, val_y, params: dict) -> CatBoostClassifier:
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_SEED,
        verbose=False,
        **params,
    )
    model.fit(
        train_x,
        train_y,
        eval_set=(val_x, val_y),
        early_stopping_rounds=40,
        use_best_model=True,
    )
    return model


def refit(train_x, train_y, params: dict, iterations: int) -> CatBoostClassifier:
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_SEED,
        verbose=False,
        **{**params, "iterations": iterations},
    )
    model.fit(train_x, train_y)
    return model


def typo_text(text: str, rng: np.random.Generator) -> str:
    chars = list(text)
    if len(chars) < 5:
        return text
    index = int(rng.integers(1, len(chars) - 1))
    if rng.random() < 0.5:
        chars.pop(index)
    else:
        chars[index - 1], chars[index] = chars[index], chars[index - 1]
    return "".join(chars)


def log_run(name: str, params: dict, metrics: dict, features: list[str], model=None) -> str:
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({key: str(value) for key, value in params.items() if value is not None})
        numeric = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        mlflow.log_metrics(numeric)
        mlflow.log_param("decision_threshold", metrics.get("threshold", 0))
        feature_path = ROOT / "reports" / f"{name}_features.txt"
        feature_path.write_text("\n".join(features) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(feature_path))
        if model is not None:
            model_path = ROOT / "reports" / f"{name}.cbm"
            model.save_model(str(model_path))
            mlflow.log_artifact(str(model_path))
        return run.info.run_id


def main() -> None:
    os.chdir(ROOT)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "mlartifacts").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("ozon_card_query")

    print("loading clickhouse")
    ch = client()
    cards = load_cards(ch)
    brands = sorted({str(item) for item in ch.query_df(BRAND_SQL)["brand"].tolist()})
    train_raw = load_pairs(ch, TRAIN).sort_values(["sku", "query"]).reset_index(drop=True)
    test_raw = load_pairs(ch, TEST).sort_values(["sku", "query"]).reset_index(drop=True)
    print(f"train rows {len(train_raw)} test rows {len(test_raw)} cards {len(cards)}")

    encoder = SemanticEncoder()
    encoder.fit(
        train_raw["query"].astype(str).tolist()
        + cards.loc[cards["sku"].isin(train_raw["sku"]), "card_text"].tolist()
    )
    train = attach_features(train_raw, cards, brands, encoder)
    test = attach_features(test_raw, cards, brands, encoder)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
    fit_idx, val_idx = next(splitter.split(train, train["label"], train["sku"]))
    fit_df = train.iloc[fit_idx]
    val_df = train.iloc[val_idx]

    best_params = None
    best_ap = -1.0
    best_iterations = 100
    for params in GRID:
        model = fit_model(
            fit_df[STATS_FEATURES],
            fit_df["label"],
            val_df[STATS_FEATURES],
            val_df["label"],
            params,
        )
        scores = model.predict_proba(val_df[STATS_FEATURES])[:, 1]
        ap = float(average_precision_score(val_df["label"], scores))
        iterations = int(model.get_best_iteration() or params["iterations"])
        iterations = max(iterations, 50)
        print(f"grid {params} val_pr_auc {ap:.4f} iterations {iterations}")
        if ap > best_ap:
            best_ap = ap
            best_params = params
            best_iterations = iterations

    stats_model = refit(train[STATS_FEATURES], train["label"], best_params, best_iterations)
    semantic_model = refit(
        train[SEMANTIC_FEATURES], train["label"], best_params, best_iterations
    )

    # Порог F1 берём с отложенных карточек обучения, не с будущего теста.
    stats_for_threshold = fit_model(
        fit_df[STATS_FEATURES], fit_df["label"], val_df[STATS_FEATURES], val_df["label"],
        {**best_params, "iterations": best_iterations},
    )
    semantic_for_threshold = fit_model(
        fit_df[SEMANTIC_FEATURES],
        fit_df["label"],
        val_df[SEMANTIC_FEATURES],
        val_df["label"],
        {**best_params, "iterations": best_iterations},
    )
    stats_threshold = best_f1_threshold(
        val_df["label"].to_numpy(),
        stats_for_threshold.predict_proba(val_df[STATS_FEATURES])[:, 1],
    )
    semantic_threshold = best_f1_threshold(
        val_df["label"].to_numpy(),
        semantic_for_threshold.predict_proba(val_df[SEMANTIC_FEATURES])[:, 1],
    )

    y_test = test["label"].to_numpy()
    rule_scores = test["orders_past"].to_numpy(dtype=float)
    stats_scores = stats_model.predict_proba(test[STATS_FEATURES])[:, 1]
    semantic_scores = semantic_model.predict_proba(test[SEMANTIC_FEATURES])[:, 1]

    results = {
        "rule_past_order": classification_metrics(y_test, rule_scores, threshold=1.0),
        "catboost_stats": classification_metrics(y_test, stats_scores, stats_threshold),
        "catboost_semantic": classification_metrics(
            y_test, semantic_scores, semantic_threshold
        ),
    }
    # Правило — жёсткое: заказ в прошлом окне был или нет.
    results["rule_past_order"]["f1"] = float(
        f1_score(y_test, rule_scores > 0, zero_division=0)
    )
    results["rule_past_order"]["precision"] = float(
        precision_score(y_test, rule_scores > 0, zero_division=0)
    )
    results["rule_past_order"]["recall"] = float(
        recall_score(y_test, rule_scores > 0, zero_division=0)
    )
    results["rule_past_order"]["threshold_note"] = "orders_past > 0"

    rng = np.random.default_rng(RANDOM_SEED)
    typo_queries = [
        typo_text(query, rng) if rng.random() < 0.3 else query
        for query in test["query"].tolist()
    ]
    typo_frame = test_raw.copy()
    typo_frame["query"] = typo_queries
    typo_ready = attach_features(typo_frame, cards, brands, encoder)
    typo_scores = semantic_model.predict_proba(typo_ready[SEMANTIC_FEATURES])[:, 1]
    typo_metrics = classification_metrics(y_test, typo_scores, semantic_threshold)

    outlier = test.copy()
    top_query = outlier.loc[outlier["freq_past"].idxmax(), "query"]
    outlier.loc[outlier["query"] == top_query, "freq_past"] *= 100
    outlier["log_freq"] = np.log1p(outlier["freq_past"].clip(lower=0))
    freq = outlier["freq_past"].to_numpy(dtype=float)
    orders = outlier["orders_past"].to_numpy(dtype=float)
    outlier["conv_past"] = np.divide(orders, freq, out=np.zeros_like(orders), where=freq > 0)
    outlier_scores = semantic_model.predict_proba(outlier[SEMANTIC_FEATURES])[:, 1]
    outlier_metrics = classification_metrics(y_test, outlier_scores, semantic_threshold)

    position_noise = test.copy()
    position_noise["position_past"] = position_noise["position_past"].fillna(0) + rng.normal(
        0, 50, len(position_noise)
    )
    position_scores = semantic_model.predict_proba(position_noise[SEMANTIC_FEATURES])[:, 1]
    position_metrics = classification_metrics(y_test, position_scores, semantic_threshold)

    base_ap = results["catboost_semantic"]["pr_auc"]
    robustness = {
        "typos": {
            **typo_metrics,
            "share_changed": 0.3,
            "pr_auc_delta": typo_metrics["pr_auc"] - base_ap,
        },
        "freq_outlier": {
            **outlier_metrics,
            "query": str(top_query),
            "pr_auc_delta": outlier_metrics["pr_auc"] - base_ap,
        },
        "position_noise": {
            **position_metrics,
            "pr_auc_delta": position_metrics["pr_auc"] - base_ap,
            "scores_identical": bool(np.allclose(position_scores, semantic_scores)),
        },
    }

    common_params = {
        "train_features": f"{TRAIN['feat_start']}..{TRAIN['feat_end']}",
        "train_labels": f"{TRAIN['label_start']}..{TRAIN['label_end']}",
        "test_features": f"{TEST['feat_start']}..{TEST['feat_end']}",
        "test_labels": f"{TEST['label_start']}..{TEST['label_end']}",
        "depth": best_params["depth"],
        "learning_rate": best_params["learning_rate"],
        "iterations": best_iterations,
        "auto_class_weights": "Balanced",
        "random_seed": RANDOM_SEED,
        "semantic": "tfidf_char_wb_3_5",
        "val_pr_auc_stats": round(best_ap, 6),
        "param_source": "https://catboost.ai/docs/en/concepts/parameter-tuning",
    }
    run_ids = {
        "rule_past_order": log_run(
            "rule_past_order",
            {**common_params, "model": "orders_past"},
            results["rule_past_order"],
            ["orders_past"],
        ),
        "catboost_stats": log_run(
            "catboost_stats",
            {**common_params, "model": "catboost"},
            results["catboost_stats"],
            STATS_FEATURES,
            stats_model,
        ),
        "catboost_semantic": log_run(
            "catboost_semantic",
            {**common_params, "model": "catboost"},
            results["catboost_semantic"],
            SEMANTIC_FEATURES,
            semantic_model,
        ),
        "robustness_typos": log_run(
            "robustness_typos",
            {**common_params, "model": "catboost_semantic_typos"},
            typo_metrics,
            SEMANTIC_FEATURES,
        ),
        "robustness_freq_outlier": log_run(
            "robustness_freq_outlier",
            {**common_params, "model": "catboost_semantic_freq_outlier", "query": str(top_query)},
            outlier_metrics,
            SEMANTIC_FEATURES,
        ),
        "robustness_position_noise": log_run(
            "robustness_position_noise",
            {**common_params, "model": "catboost_semantic_position_unused"},
            position_metrics,
            SEMANTIC_FEATURES,
        ),
    }

    summary = {
        "windows": {"train": TRAIN, "test": TEST},
        "counts": {
            "train_pairs": int(len(train)),
            "train_positives": int(train["label"].sum()),
            "train_sku": int(train["sku"].nunique()),
            "val_pairs": int(len(val_df)),
            "test_pairs": int(len(test)),
            "test_positives": int(test["label"].sum()),
            "test_sku": int(test["sku"].nunique()),
            "brands": len(brands),
            "cards_with_text": int(cards["sku"].nunique()),
        },
        "params": common_params,
        "results": results,
        "robustness": robustness,
        "mlflow_runs": run_ids,
        "beats_rule": results["catboost_semantic"]["pr_auc"] > results["rule_past_order"]["pr_auc"],
    }
    (ROOT / "reports" / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(robustness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
