"""Сводит похожие фразы одной карточки в одну тему.

Решение «оставить» не затирается.
Если у близкой формулировки уже был заказ, запрос можно оставить.
Тематическое правило только смягчает решение и не создаёт кандидатов на удаление.
"""

from __future__ import annotations

import pandas as pd

from sparse_rules import tokens

_SUFFIXES = (
    "ами",
    "ями",
    "ого",
    "ему",
    "ыми",
    "ими",
    "ая",
    "яя",
    "ое",
    "ее",
    "ый",
    "ий",
    "ой",
    "ые",
    "ие",
    "ов",
    "ев",
    "ам",
    "ям",
    "ах",
    "ях",
    "ом",
    "ем",
    "а",
    "я",
    "ы",
    "и",
    "о",
    "е",
    "у",
    "ю",
)


def stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def stems(query: str) -> set[str]:
    return {stem(token) for token in tokens(query)}


def is_generic(query: str) -> bool:
    return len(tokens(query)) <= 2


def _same_theme(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return False
    shared = left & right
    if len(shared) >= 2:
        return True
    if len(shared) != 1:
        return False
    common = next(iter(shared))
    if len(common) < 6:
        return False
    shorter = min(len(left), len(right))
    return len(shared) / shorter >= 0.5


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _group_ids(queries: list[str]) -> list[int]:
    keys = [stems(query) for query in queries]
    forest = _UnionFind(len(queries))
    for left in range(len(queries)):
        for right in range(left + 1, len(queries)):
            if _same_theme(keys[left], keys[right]):
                forest.union(left, right)
    roots = [forest.find(index) for index in range(len(queries))]
    remap = {root: number + 1 for number, root in enumerate(dict.fromkeys(roots))}
    return [remap[root] for root in roots]


def compose_recommendations(frame: pd.DataFrame) -> pd.DataFrame:
    """Добавить тему и выровнять решение внутри неё."""
    pieces = []
    for sku, block in frame.groupby("sku", sort=False):
        block = block.copy()
        if "raw_decision" not in block:
            block["raw_decision"] = block["decision"]
        queries = block["query"].astype(str).tolist()
        query_stems = {
            index: stems(query) for index, query in zip(block.index, queries)
        }
        block["theme_id"] = _group_ids(queries)
        block["generic"] = [is_generic(query) for query in queries]
        for theme_id, theme in block.groupby("theme_id"):
            sold = bool((theme["orders_past"] > 0).any())
            evidence = theme.loc[
                (theme["orders_past"] > 0)
                | ((theme["decision"] == "keep") & ~theme["generic"])
            ]
            model_keep = bool(len(evidence))
            theme_on = sold or model_keep
            label = block.index.isin(theme.index)
            bare = label & block["generic"]
            if theme_on:
                for index in theme.index:
                    if bool(block.at[index, "generic"]):
                        continue
                    if any(
                        _same_theme(query_stems[index], query_stems[evidence_index])
                        for evidence_index in evidence.index
                    ):
                        block.at[index, "decision"] = "keep"
                        if "decision_reason" in block:
                            block.at[index, "decision_reason"] = (
                                "Близкая формулировка темы уже получила заказ "
                                "или уверенное решение"
                            )
                block.loc[bare & (block["orders_past"] > 0), "decision"] = "keep"
            sold_rows = theme[theme["orders_past"] > 0].sort_values(
                ["orders_past", "freq_past"],
                ascending=False,
            )
            if len(sold_rows):
                canonical = sold_rows["query"].iloc[0]
            else:
                canonical = theme.sort_values("freq_past", ascending=False)[
                    "query"
                ].iloc[0]
            block.loc[label, "theme_phrase"] = canonical
            block.loc[label, "theme_kept"] = theme_on
        block["sku"] = sku
        pieces.append(block)
    out = pd.concat(pieces, ignore_index=True)
    out["theme_id"] = out["theme_id"].astype(int)
    return out.sort_values(
        ["sku", "theme_id", "generic", "orders_past", "freq_past"],
        ascending=[True, True, True, False, False],
    )
