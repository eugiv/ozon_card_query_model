"""Публичная проверка сохранённых позиций всех будущих продающих пар."""

import numpy as np
import pandas as pd


def target_ranks(candidates, positives, weights):
    from dormant_learning import fusion_scores, ranking_metrics

    result = positives[["sku", "normquery"]].copy()
    for name, channels in weights.items():
        prediction = candidates[["sku", "normquery"]].assign(
            score=fusion_scores(candidates, channels)
        )
        ranking_metrics(prediction, positives)
        ordered = prediction.sort_values(
            ["sku", "score", "normquery"], ascending=[True, False, True], kind="stable"
        )
        ordered[f"{name}_rank"] = ordered.groupby("sku").cumcount() + 1
        result = result.merge(
            ordered[["sku", "normquery", f"{name}_rank"]],
            on=["sku", "normquery"], how="left", validate="one_to_one",
        )
        result[f"{name}_rank"] = result[f"{name}_rank"].astype("Int64")
    return result


def snapshot_metrics(pairs, k=10):
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k должен быть положительным целым")
    required = {"product_id", "query_id", "baseline_rank", "system_rank"}
    if not required.issubset(pairs.columns) or pairs.empty:
        raise ValueError("Нужны все будущие продающие пары и позиции двух методов")
    for column in ["product_id", "query_id"]:
        if pairs[column].isna().any() or pairs[column].astype(str).str.strip().eq("").any():
            raise ValueError("Пустой идентификатор")
    if pairs.duplicated(["product_id", "query_id"]).any():
        raise ValueError("Повторяющаяся продающая пара")
    counts = pairs.groupby("product_id").size()
    ideal = counts.map(lambda n: sum(1 / np.log2(r + 1) for r in range(1, min(k, n) + 1)))
    metrics = {}
    for name in ["baseline", "system"]:
        ranks = pd.to_numeric(pairs[f"{name}_rank"], errors="raise")
        present = ranks.dropna().to_numpy(dtype=float)
        if not np.isfinite(present).all() or (present < 1).any() or (present % 1 != 0).any():
            raise ValueError("Позиция должна быть положительным целым или пропуском")
        ranked = pairs.loc[ranks.notna(), ["product_id"]].assign(rank=ranks[ranks.notna()])
        if ranked.duplicated(["product_id", "rank"]).any():
            raise ValueError("Повторяющаяся позиция внутри товара")
        hit = ranks.le(k).fillna(False)
        hits = hit.groupby(pairs["product_id"]).sum()
        discount = pd.Series(0., index=pairs.index)
        discount.loc[hit] = 1 / np.log2(ranks.loc[hit].astype(float) + 1)
        dcg = discount.groupby(pairs["product_id"]).sum()
        retrieved = ranks.notna().groupby(pairs["product_id"]).sum()
        metrics[name] = {
            "evaluated_skus": len(counts), "positive_pairs": len(pairs),
            f"recall@{k}": float((hits / counts).mean()),
            f"precision@{k}": float((hits / k).mean()),
            f"ndcg@{k}": float((dcg / ideal).mean()),
            "retrieval_recall": float((retrieved / counts).mean()),
            "hits": int(hit.sum()),
        }
    return metrics