"""Маленький синтетический набор для настоящих обучающих проверок."""

import pandas as pd


def write_dataset(folder):
    names = [
        "мыло ланолиновое", "мыло невская", "мыло дегтярное",
        "мыло белое", "мыло детское",
    ]
    records = pd.DataFrame(
        [[name, name, "2025-12-31T12:00:00Z", 200, ""] for name in names],
        columns=["query", "normquery", "loaded_at", "status_code", "error"],
    )
    rows = []
    for day in range(8):
        current = names[[0, 1, 3, 4][day // 2]]
        for sku, query, views, orders in [
            ("001", names[0], 5, 1), ("001", names[2], 30, 0),
            ("002", names[0], 5, 1), ("002", names[1], 5, 1),
            ("003", names[2], 5, 1), ("004", names[3], 5, 1),
        ]:
            rows.append([f"2026-01-{day + 1:02}", sku, query, 100, views, orders])
        if current != names[0]:
            rows.append([f"2026-01-{day + 1:02}", "001", current, 100, 5, 1])
    facts = pd.DataFrame(rows, columns=[
        "date", "sku", "query", "unique_search_users", "unique_view_users", "order_count",
    ])
    facts_path, dict_path = folder / "facts.csv", folder / "dictionary.csv"
    facts.to_csv(facts_path, index=False)
    records.to_csv(dict_path, index=False)
    return facts_path, dict_path
