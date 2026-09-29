"""Единая запись метрик и рекомендаций одного запуска."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DECISION_LABELS = {
    "keep": "Оставить как приоритетный запрос",
    "review": "Проверить вручную",
    "remove_candidate": "Кандидат на исключение",
}
SOURCE_LABELS = {
    "model": "Модель",
    "sparse_rule": "Правило для карточки с малым числом заказов",
}


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _presentation_frame(recommendations: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "sku": "Карточка",
        "theme_id": "Тема",
        "theme_phrase": "Формулировка темы",
        "query": "Запрос",
        "orders_past": "Заказов за прошлые 28 дней",
        "freq_past": "Поисков за прошлые 28 дней",
        "views_past": "Переходов в карточку за прошлые 28 дней",
        "score": "Оценка будущего заказа",
        "decision": "Решение",
        "decision_reason": "Почему принято решение",
        "source": "Источник решения",
    }
    available = [column for column in columns if column in recommendations.columns]
    result = recommendations[available].rename(columns=columns).copy()
    if "Решение" in result:
        result["Решение"] = result["Решение"].map(DECISION_LABELS).fillna(result["Решение"])
    if "Источник решения" in result:
        result["Источник решения"] = (
            result["Источник решения"].map(SOURCE_LABELS).fillna(result["Источник решения"])
        )
    if "Оценка будущего заказа" in result:
        result["Оценка будущего заказа"] = result["Оценка будущего заказа"].round(4)
    return result


def write_report_artifacts(
    report_dir: Path,
    *,
    summary: dict,
    recommendations: pd.DataFrame,
    test_predictions: pd.DataFrame,
    curves: pd.DataFrame,
) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)

    expected = recommendations["decision"].value_counts().sort_index().to_dict()
    if summary.get("recommendations") != expected:
        raise ValueError(
            "recommendation counts in summary do not match row-level recommendations"
        )

    paths = {
        "metrics": report_dir / "metrics.json",
        "recommendations_csv": report_dir / "recommendations.csv",
        "recommendations_xlsx": report_dir / "recommendations.xlsx",
        "test_predictions": report_dir / "test_predictions.csv",
        "model_curves": report_dir / "model_curves.csv",
    }

    recommendations.to_csv(paths["recommendations_csv"], index=False)
    test_predictions.to_csv(paths["test_predictions"], index=False)
    curves.to_csv(paths["model_curves"], index=False)

    presentation = _presentation_frame(recommendations)
    guide = pd.DataFrame(
        [
            {
                "Поле": "Оставить как приоритетный запрос",
                "Что значит": "По запросу уже были заказы или модель уверенно ожидает заказ.",
            },
            {
                "Поле": "Проверить вручную",
                "Что значит": "Данных или уверенности недостаточно для автоматического решения.",
            },
            {
                "Поле": "Кандидат на исключение",
                "Что значит": (
                    "Было достаточно поисков и переходов, но оценка будущего заказа низкая. "
                    "Это не команда удалить фразу."
                ),
            },
            {
                "Поле": "Оценка будущего заказа",
                "Что значит": (
                    "Калиброванная оценка для приоритизации. Она не доказывает эффект "
                    "добавления или удаления текста."
                ),
            },
        ]
    )
    temporary_xlsx = paths["recommendations_xlsx"].with_name(
        paths["recommendations_xlsx"].stem + ".tmp.xlsx"
    )
    with pd.ExcelWriter(temporary_xlsx, engine="openpyxl") as writer:
        presentation.to_excel(writer, sheet_name="Рекомендации", index=False)
        guide.to_excel(writer, sheet_name="Как читать", index=False)
    temporary_xlsx.replace(paths["recommendations_xlsx"])

    temporary_json = paths["metrics"].with_suffix(".tmp.json")
    temporary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(paths["metrics"])
    return {name: str(path) for name, path in paths.items()}
