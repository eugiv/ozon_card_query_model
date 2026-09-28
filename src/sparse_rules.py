"""Правила для карточек, у которых в окне ответа меньше трёх пар с заказом.

Модель на таких карточках не учится: позитивов слишком мало.
Решение по новой фразе принимает эта функция.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[0-9a-zа-я]+", re.IGNORECASE)
_STOP = {"для", "при", "над", "под", "без", "или", "это", "как", "что", "уже", "еще", "про"}


def tokens(text: str) -> list[str]:
    raw = (text or "").lower().replace("ё", "е")
    return [
        item
        for item in _TOKEN.findall(raw)
        if len(item) >= 2 and item not in _STOP
    ]


def decide_sparse_card(query: str, card_text: str, brand: str, orders_past: float) -> str:
    """Вернуть keep, drop или review.

    keep — фразу можно оставить в тексте.
    drop — одно широкое слово без прошлого заказа.
    review — фраза узкая, но совпадение с карточкой слабое, её смотрит человек.
    """
    if orders_past > 0:
        return "keep"
    query_tokens = tokens(query)
    if not query_tokens:
        return "drop"
    brand_norm = (brand or "").lower().replace("ё", "е").strip()
    if len(brand_norm) >= 3 and brand_norm in " ".join(query_tokens):
        return "keep"
    card_tokens = set(tokens(card_text))
    overlap = sum(token in card_tokens for token in query_tokens) / len(query_tokens)
    if len(query_tokens) >= 3 and overlap >= 0.5:
        return "keep"
    if len(query_tokens) <= 1:
        return "drop"
    return "review"
