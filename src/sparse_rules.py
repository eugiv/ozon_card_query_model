"""Безопасные правила для карточек с малым числом прошлых заказов.

Модель на таких карточках не применяется: позитивных примеров слишком мало.
Без исторических снимков текста правило не выдаёт автоматическое удаление.
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


def decide_sparse_card(
    query: str,
    orders_past: float,
    freq_past: float,
    views_past: float,
) -> str:
    """Вернуть keep или review.

    Заказ в прошлом — достаточная причина не удалять запрос автоматически.
    Во всех остальных случаях решение остаётся человеку.
    """
    if orders_past > 0:
        return "keep"
    _ = (query, freq_past, views_past)
    return "review"
