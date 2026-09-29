import pandas as pd

from compose import compose_recommendations


def test_theme_rule_only_softens_decision():
    frame = pd.DataFrame(
        {
            "sku": ["1", "1", "1"],
            "query": [
                "крем для ног мочевина",
                "крем для ног с мочевиной",
                "шампунь",
            ],
            "orders_past": [2, 0, 0],
            "freq_past": [100, 80, 500],
            "decision": ["keep", "remove_candidate", "remove_candidate"],
            "decision_reason": ["Заказы", "Низкая оценка", "Низкая оценка"],
            "source": ["model", "model", "model"],
        }
    )
    result = compose_recommendations(frame).set_index("query")
    assert result.loc["крем для ног с мочевиной", "decision"] == "keep"
    assert result.loc["шампунь", "decision"] == "remove_candidate"
    assert result.loc["крем для ног с мочевиной", "raw_decision"] == "remove_candidate"


def test_chain_similarity_does_not_promote_unrelated_edge():
    frame = pd.DataFrame(
        {
            "sku": ["1", "1", "1"],
            "query": [
                "крем увлажняющий лицо",
                "крем увлажняющий тело",
                "лосьон увлажняющий тело",
            ],
            "orders_past": [1, 0, 0],
            "freq_past": [100, 100, 100],
            "decision": ["keep", "review", "remove_candidate"],
            "decision_reason": ["Заказы", "Проверка", "Низкая оценка"],
            "source": ["model", "model", "model"],
        }
    )
    result = compose_recommendations(frame).set_index("query")
    assert result.loc["крем увлажняющий тело", "decision"] == "keep"
    assert result.loc["лосьон увлажняющий тело", "decision"] == "remove_candidate"
