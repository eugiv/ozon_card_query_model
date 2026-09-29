import json

import pandas as pd
import pytest

from artifacts import write_report_artifacts


def _recommendations():
    return pd.DataFrame(
        {
            "sku": ["1", "1"],
            "query": ["крем", "лосьон"],
            "orders_past": [1, 0],
            "freq_past": [10, 100],
            "views_past": [3, 20],
            "score": [0.8, 0.1],
            "decision": ["keep", "remove_candidate"],
            "decision_reason": ["Заказы", "Низкая оценка"],
            "source": ["model", "model"],
        }
    )


def test_artifacts_share_one_set_of_counts(tmp_path):
    recommendations = _recommendations()
    summary = {
        "artifact_id": "test",
        "recommendations": {"keep": 1, "remove_candidate": 1},
    }
    paths = write_report_artifacts(
        tmp_path,
        summary=summary,
        recommendations=recommendations,
        test_predictions=recommendations.assign(label=[1, 0]),
        curves=pd.DataFrame(
            {"curve": ["roc"], "model_name": ["m"], "x": [0.0], "y": [1.0]}
        ),
    )
    stored = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    workbook = pd.read_excel(paths["recommendations_xlsx"], sheet_name="Рекомендации")
    assert stored["recommendations"] == {"keep": 1, "remove_candidate": 1}
    assert len(workbook) == 2


def test_artifacts_reject_mismatched_counts(tmp_path):
    with pytest.raises(ValueError, match="counts"):
        write_report_artifacts(
            tmp_path,
            summary={"recommendations": {"keep": 2}},
            recommendations=_recommendations(),
            test_predictions=_recommendations(),
            curves=pd.DataFrame({"curve": ["roc"], "model_name": ["m"], "x": [0], "y": [1]}),
        )
