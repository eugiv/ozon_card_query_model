"""Экспорт позиций без реальных идентификаторов, текстов и таблицы соответствий."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cluster_data import sha256
from full_history_recommender import verify_folder
from ranking_snapshot import snapshot_metrics, target_ranks


def export():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    evaluation = ROOT / config["outputs"]["evaluation"]
    ablation = ROOT / config["outputs"]["ablation"]
    verify_folder(evaluation)
    ablation_manifest = verify_folder(ablation)
    if ablation_manifest["parent_sha256"] != sha256(evaluation / "manifest.json"):
        raise ValueError("Сравнение относится к другому эксперименту")
    experiment = json.loads((ROOT / "reports/experiment_summary.json").read_text(encoding="utf-8"))
    comparison = json.loads((ablation / "metrics.json").read_text(encoding="utf-8"))
    weights = {"baseline": comparison["weights"]["e5_co_sales"],
               "system": comparison["weights"]["all_four"]}
    frames = []
    sources = {}
    for period in ["test", "new_days"]:
        paths = {name: evaluation / f"{period}_{name}.csv" for name in ["candidates", "positives"]}
        candidates, positives = [pd.read_csv(paths[name], dtype={"sku": "string"},
                                keep_default_na=False) for name in ["candidates", "positives"]]
        frames.append(target_ranks(candidates, positives, weights).assign(period=period))
        sources[period] = {"candidate_pairs": len(candidates),
                           **{name + "_sha256": sha256(path) for name, path in paths.items()}}
    pairs = pd.concat(frames, ignore_index=True)
    for original, public, prefix in [("sku", "product_id", "P"), ("normquery", "query_id", "Q")]:
        mapping = {value: f"{prefix}{number:04d}" for number, value in
                   enumerate(sorted(pairs[original].unique()), start=1)}
        pairs[public] = pairs[original].map(mapping)
    pairs = pairs[["period", "product_id", "query_id", "baseline_rank", "system_rank"]].sort_values(
        ["period", "product_id", "query_id"], kind="stable"
    )
    for period in ["test", "new_days"]:
        measured = snapshot_metrics(pairs.loc[pairs.period.eq(period)])
        for name, reference in [("baseline", "e5_co_sales"), ("system", "all_four")]:
            for key in ["recall@10", "precision@10", "ndcg@10", "retrieval_recall",
                        "evaluated_skus", "positive_pairs"]:
                actual = measured[name][key]
                if not np.isclose(actual, comparison[period][reference][key], rtol=1e-10, atol=1e-12):
                    raise ValueError("Позиции не воспроизводят исходную оценку")
                if not np.isclose(actual, experiment["ablation"][period][reference][key], rtol=1e-10, atol=1e-12):
                    raise ValueError("Позиции не совпадают с публичным отчётом")
    output = ROOT / "reports/notebook_ranking_pairs.csv"
    pairs.to_csv(output, index=False)
    manifest = {
        "schema_version": 1,
        "contents": "all future selling pairs, including targets missing from the candidate pool",
        "privacy": "ordinal product/query aliases; no original IDs, text or lookup table",
        "evaluation": "saved ranking evaluation, not retraining or a sales uplift experiment",
        "rank_order": "score descending, original normquery ascending; ranks computed before anonymization",
        "missing_rank": "target absent from the candidate pool; retained in metric denominator",
        "weights": weights, "rrf_offset": 60, "top_k": 10,
        "pairs_sha256": sha256(output), "source_sha256": sources,
        "ablation_manifest_sha256": sha256(ablation / "manifest.json"),
        "experiment_summary_sha256": sha256(ROOT / "reports/experiment_summary.json"),
    }
    (ROOT / "reports/notebook_ranking_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Экспортировано {len(pairs)} обезличенных продающих пар; метрики совпали с отчётом.")


if __name__ == "__main__":
    export()