"""Признаки пары «карточка + фраза». Ответ в признаки не входит."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

_TOKEN = re.compile(r"[0-9a-zа-я]+", re.IGNORECASE)
_STOP = {"для", "при", "над", "под", "без", "или", "это", "как", "что", "уже", "еще", "про"}

MODEL_FEATURES = [
    "log_freq",
    "log_views",
    "orders_past",
    "conv_past",
    "view_rate_past",
    "order_rate_from_view_past",
    "views_past",
    "days_past",
    "query_token_count",
]
# Старые имена оставлены для совместимости с локальными отчётами.
STATS_FEATURES = MODEL_FEATURES
SEMANTIC_FEATURES = MODEL_FEATURES + ["brand_hit", "word_overlap", "semantic_cosine"]


def tokens(text: str) -> list[str]:
    raw = str(text or "").lower().replace("ё", "е")
    return [
        item
        for item in _TOKEN.findall(raw)
        if len(item) >= 2 and item not in _STOP
    ]


def _brand_hit(query: str, brands: list[str]) -> int:
    query_norm = " " + " ".join(tokens(query)) + " "
    if query_norm.strip() == "":
        return 0
    return int(any(f" {brand} " in query_norm for brand in brands))


def _overlap(query: str, card_text: str) -> float:
    query_tokens = tokens(query)
    if not query_tokens:
        return 0.0
    card_tokens = set(tokens(card_text))
    return sum(token in card_tokens for token in query_tokens) / len(query_tokens)


class SemanticEncoder:
    """Косинус близости фразы и текста карточки по символьным n-граммам.

    Вектор — TF-IDF, n-граммы 3–5 символов. Словарь строится только
    по текстам обучения, тестовые фразы в словарь не попадают.
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=30000,
        )
        self._matrix = None
        self._index: dict[str, int] = {}

    def fit(self, texts: list[str]) -> None:
        unique = list(dict.fromkeys(texts))
        self._matrix = normalize(self.vectorizer.fit_transform(unique))
        self._index = {text: i for i, text in enumerate(unique)}

    def add_texts(self, texts: list[str]) -> None:
        missing = [text for text in dict.fromkeys(texts) if text not in self._index]
        if not missing:
            return
        extra = normalize(self.vectorizer.transform(missing))
        start = len(self._index)
        self._index.update({text: start + i for i, text in enumerate(missing)})
        if self._matrix is None:
            self._matrix = extra
        else:
            from scipy import sparse

            self._matrix = sparse.vstack([self._matrix, extra], format="csr")

    def cosine(self, left: list[str], right: list[str]) -> np.ndarray:
        self.add_texts(list(left) + list(right))
        left_idx = [self._index[text] for text in left]
        right_idx = [self._index[text] for text in right]
        left_vec = self._matrix[left_idx]
        right_vec = self._matrix[right_idx]
        return np.asarray(left_vec.multiply(right_vec).sum(axis=1)).ravel()


def attach_features(
    frame: pd.DataFrame,
    cards: pd.DataFrame | None = None,
    brands: list[str] | None = None,
    encoder: SemanticEncoder | None = None,
) -> pd.DataFrame:
    if cards is None:
        data = frame.copy()
        data["card_text"] = ""
    else:
        data = frame.merge(cards, on="sku", how="left")
    data["card_text"] = data["card_text"].fillna("")
    data["query"] = data["query"].fillna("").astype(str)
    data["log_freq"] = np.log1p(data["freq_past"].clip(lower=0))
    data["log_views"] = np.log1p(data["views_past"].clip(lower=0))
    freq = data["freq_past"].to_numpy(dtype=float)
    views = data["views_past"].to_numpy(dtype=float)
    orders = data["orders_past"].to_numpy(dtype=float)
    data["conv_past"] = np.divide(orders, freq, out=np.zeros_like(orders), where=freq > 0)
    data["view_rate_past"] = np.divide(
        views,
        freq,
        out=np.zeros_like(views),
        where=freq > 0,
    )
    data["order_rate_from_view_past"] = np.divide(
        orders,
        views,
        out=np.zeros_like(orders),
        where=views > 0,
    )
    data["brand_hit"] = [_brand_hit(query, brands or []) for query in data["query"]]
    data["word_overlap"] = [
        _overlap(query, card)
        for query, card in zip(data["query"], data["card_text"], strict=True)
    ]
    data["query_token_count"] = [len(tokens(query)) for query in data["query"]]
    if encoder is not None:
        data["semantic_cosine"] = encoder.cosine(
            data["query"].tolist(),
            data["card_text"].tolist(),
        )
    data["label"] = (data["orders_y"] > 0).astype(int)
    return data
