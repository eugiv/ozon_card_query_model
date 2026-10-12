"""Поиск малотрафиковых кластеров с накопленной историей и явными неизвестными."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from sklearn.preprocessing import normalize

from cluster_data import (
    aggregate_clusters,
    dictionary_snapshot,
    future_positives,
    period_pairs,
    sha256,
    write_json,
)
from cluster_recommender import (
    CANDIDATE_COLUMNS,
    HybridRetriever,
    lexical_index,
    load_embeddings,
    mask_selling_edges,
)
from dormant_learning import (
    CHANNELS,
    fit_fusion,
    fit_ranker,
    fusion_scores,
    learning_rows,
    predict_ranker,
    ranking_metrics,
)
from graph_learning import load_graph

logger = logging.getLogger("full_history_recommender")


@dataclass(frozen=True)
class DormantConfig:
    dictionary_as_of: date
    first_decision: date
    validation_decision: date
    test_decision: date
    window_days: int = 14
    max_views: int = 5
    per_source: int = 150
    iterations: tuple[int, ...] = (100, 300)
    mask_rounds: int = 1
    seed: int = 7
    max_seeds: int = 32
    training_stride_days: int | None = None

    def __post_init__(self):
        for value in [self.window_days, self.per_source, self.mask_rounds, self.max_seeds, *self.iterations]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Размер окна, бюджет и параметры обучения должны быть положительными")
        if isinstance(self.max_views, bool) or not isinstance(self.max_views, int) or self.max_views < 0:
            raise ValueError("max_views должен быть неотрицательным целым")
        if not self.iterations or len(set(self.iterations)) != len(self.iterations):
            raise ValueError("Нужны различные iterations")
        if self.training_stride_days is not None and (
            type(self.training_stride_days) is not int
            or not 1 <= self.training_stride_days <= self.window_days
        ):
            raise ValueError("Шаг обучения должен быть от 1 до window_days")
        if not self.dictionary_as_of <= self.first_decision < self.validation_decision < self.test_decision:
            raise ValueError("Нарушен порядок снимка и временных проверок")
        for start, end in [(self.first_decision, self.validation_decision),
                           (self.validation_decision, self.test_decision)]:
            if (end - start).days % self.window_days:
                raise ValueError("Границы должны совпадать с шагом окон")

    def training_decisions(self):
        return list(pd.date_range(self.first_decision,
                                  self.validation_decision - timedelta(days=self.window_days),
                                  freq=f"{self.training_stride_days or self.window_days}D").date)


def dormant_positives(positives, recent, max_views, active_skus):
    joined = positives.merge(recent[["sku", "normquery", "views"]],
                             on=["sku", "normquery"], how="left", validate="one_to_one")
    return joined.loc[joined["views"].fillna(0).le(max_views)
                      & joined["sku"].isin(active_skus), ["sku", "normquery"]].reset_index(drop=True)


def missing_window_days(source_days, decision, window_days, include_target=True):
    start = decision - timedelta(days=window_days - 1)
    end = decision + timedelta(days=window_days) if include_target else decision
    return pd.date_range(start, end).difference(source_days).strftime("%Y-%m-%d").tolist()


def generate_dormant(daily, catalog, decision, config, vectors, text, *, hidden=None):
    start = daily["date"].min().date()
    history = period_pairs(daily, start, decision)
    recent_start = decision - timedelta(days=config.window_days - 1)
    recent = period_pairs(daily, recent_start, decision)
    active = set(recent["sku"])
    if hidden is not None:
        def without(frame):
            joined = frame.merge(hidden.assign(_hidden=True), on=["sku", "normquery"], how="left")
            return joined.loc[joined["_hidden"].isna(), frame.columns].reset_index(drop=True)
        history, recent = without(history), without(recent)
    part = daily.loc[daily["date"].between(pd.Timestamp(recent_start), pd.Timestamp(decision))]
    demand = part.drop_duplicates(["date", "normquery"]).groupby("normquery")["cluster_searches"].sum()
    candidates = HybridRetriever(catalog, history, demand, vectors, text).generate(
        config.per_source, max_history_views=config.max_views,
        traffic_history=recent, target_skus=active, max_seeds=config.max_seeds,
    )
    return candidates, history, recent, active


def evaluate_scores(candidates, positives, methods):
    return {name: ranking_metrics(candidates[["sku", "normquery"]].assign(score=scores), positives)
            for name, scores in methods.items()}


def bootstrap_difference(candidates, positives, scores, baseline, seed=7, repeats=2000):
    values = []
    for sku in sorted(positives["sku"].unique()):
        target = positives.loc[positives["sku"].eq(sku)]
        selected = candidates["sku"].eq(sku)
        frame = candidates.loc[selected, ["sku", "normquery"]]
        left = ranking_metrics(frame.assign(score=np.asarray(scores)[selected]), target)
        right = ranking_metrics(frame.assign(score=np.asarray(baseline)[selected]), target)
        values.append(left["recall@10"] - right["recall@10"])
    if not values:
        return {"skus": 0, "difference": None, "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    samples = np.asarray(values)[rng.integers(0, len(values), size=(repeats, len(values)))].mean(axis=1)
    return {"skus": len(values), "difference": float(np.mean(values)),
            "lower": float(np.quantile(samples, .025)), "upper": float(np.quantile(samples, .975)),
            "repeats": repeats, "seed": seed, "unit": "SKU within one diagnostic period"}


def run_full_history(facts_path, dictionary_path, embeddings_path, output, config):
    if output.exists():
        raise ValueError("Результат уже существует; перезапись запрещена")
    paths = {"facts": facts_path, "dictionary": dictionary_path}
    if embeddings_path is not None:
        paths["embeddings"] = embeddings_path
    inputs = {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}
    facts = pd.read_csv(facts_path, dtype={"sku": "string", "query": "string"}, keep_default_na=False)
    records = pd.read_csv(dictionary_path, keep_default_na=False)
    dates = pd.to_datetime(facts["date"]).dt.normalize()
    source_days = pd.DatetimeIndex(dates.unique())
    as_of = pd.Timestamp(config.dictionary_as_of, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    snapshot, dict_audit = dictionary_snapshot(records, as_of)
    if snapshot.empty:
        raise ValueError("Нет исторического словаря на дату первого решения")
    daily, audit = aggregate_clusters(facts, snapshot)
    if daily.empty:
        raise ValueError("Словарь не покрывает факты")
    catalog = sorted(snapshot["normquery"].unique().tolist())
    vectors, embedding_meta = load_embeddings(embeddings_path, catalog) if embeddings_path else (None, None)
    if embedding_meta and pd.Timestamp(embedding_meta.get("dictionary_as_of", "NaT")) != as_of:
        raise ValueError("Дата embeddings не соответствует замороженному словарю")
    text = lexical_index(catalog)
    logger.info("Подготовлены %s фактов / %s SKU / %s кластеров", len(facts), facts["sku"].nunique(), len(catalog))
    cache_root = output.with_name(output.name + "_checkpoints")
    identity = {"inputs": inputs, "config": {key: str(value) for key, value in asdict(config).items()},
                "code": {path.name: sha256(path) for path in [Path(__file__), Path(__file__).with_name("cluster_recommender.py"),
                                                              Path(__file__).with_name("cluster_data.py")]}}

    def fetch(decision, hidden=None):
        request = {**identity, "decision": decision.isoformat(),
                   "hidden": None if hidden is None else hidden.sort_values(["sku", "normquery"]).to_dict("records")}
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        cached = cache_root / digest
        if cached.exists():
            manifest = json.loads((cached / "manifest.json").read_text())
            for name, checksum in manifest["files"].items():
                if sha256(cached / name) != checksum:
                    raise ValueError("Контрольная сумма checkpoint не совпадает")
            frames = [pd.read_csv(cached / f"{name}.csv", dtype={"sku": "string", "normquery": "string"},
                                  float_precision="round_trip", keep_default_na=False)
                      for name in ["candidates", "history", "recent"]]
            logger.info("Восстановлен поиск %s из checkpoint", decision)
            return *frames, set(manifest["active"])
        result = generate_dormant(daily, catalog, decision, config, vectors, text, hidden=hidden)
        cache_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache_root, prefix=".checkpoint-") as temporary:
            folder = Path(temporary) / "data"
            folder.mkdir()
            for name, frame in zip(["candidates", "history", "recent"], result[:3], strict=True):
                frame.to_csv(folder / f"{name}.csv", index=False)
            write_json(folder / "manifest.json", {"request": request, "active": sorted(result[3]),
                                                  "files": {path.name: sha256(path) for path in folder.iterdir()}})
            folder.replace(cached)
        return result

    frames, audits, skipped = [], [], []
    for index, decision in enumerate(config.training_decisions()):
        missing = missing_window_days(source_days, decision, config.window_days)
        if missing:
            skipped.append({"decision": decision.isoformat(), "missing_days": missing})
            continue
        logger.info("Обучающее окно %s", decision)
        candidates, history, recent, active = fetch(decision)
        future = period_pairs(daily, decision + timedelta(days=1), decision + timedelta(days=config.window_days))
        positives = dormant_positives(future_positives(history, future), recent, config.max_views, active)
        rows, training_audit = learning_rows(candidates, positives, group_prefix=f"{index}:future", seed=config.seed)
        frames.append(rows)
        audits.append({"decision": decision.isoformat(), "task": "future", **training_audit})
        for mask_round in range(config.mask_rounds):
            _, hidden = mask_selling_edges(recent, config.seed + mask_round)
            if hidden.empty:
                continue
            masked, _, _, _ = fetch(decision, hidden=hidden)
            rows, training_audit = learning_rows(masked, hidden,
                                                group_prefix=f"{index}:mask:{mask_round}", seed=config.seed)
            frames.append(rows)
            audits.append({"decision": decision.isoformat(), "task": "mask", **training_audit})
    if not frames or not any(len(frame) for frame in frames):
        raise ValueError("Нет обучающих сравнений")
    training = pd.concat(frames, ignore_index=True).sort_values(["group_id", "normquery"]).reset_index(drop=True)
    evaluations = {}
    for name, decision in [("validation", config.validation_decision), ("test", config.test_decision)]:
        missing = missing_window_days(source_days, decision, config.window_days)
        if missing:
            raise ValueError(f"Неполное окно {name}: {missing}")
        logger.info("Поиск %s: %s", name, decision)
        candidates, history, recent, active = fetch(decision)
        future = period_pairs(daily, decision + timedelta(days=1), decision + timedelta(days=config.window_days))
        positives = dormant_positives(future_positives(history, future), recent, config.max_views, active)
        if positives.empty:
            raise ValueError(f"Нет положительных малотрафиковых кластеров: {name}")
        evaluations[name] = {"candidates": candidates, "positives": positives, "recent": recent,
                             "history": history, "active_skus": len(active)}
    validation = evaluations["validation"]
    channels = ["lexical", "co_sales", "pretrained_embeddings"] if vectors is not None else ["lexical", "co_sales"]
    fitted = fit_fusion(validation["candidates"], validation["positives"], channels)
    models, trials = {}, []
    for iterations in config.iterations:
        logger.info("Обучение: %s строк, %s итераций", len(training), iterations)
        model = fit_ranker(training, iterations, seed=config.seed)
        metrics = ranking_metrics(validation["candidates"][["sku", "normquery"]].assign(
            score=predict_ranker(model, validation["candidates"])), validation["positives"])
        trials.append({"iterations": iterations, "metrics": metrics})
        models[iterations] = model
    selected_iterations = max(trials, key=lambda trial: trial["metrics"]["ndcg@10"])["iterations"]
    model = models[selected_iterations]

    def scores(candidates):
        result = {name: candidates[column].to_numpy() for name, column in CHANNELS.items()
                  if name in channels}
        result["popularity"] = candidates["log_cluster_orders"].to_numpy()
        result["catboost_pu"] = predict_ranker(model, candidates)
        result["validation_fusion"] = fusion_scores(candidates, fitted["weights"], fitted["offset"])
        return result

    validation_scores = scores(validation["candidates"])
    validation_metrics = evaluate_scores(validation["candidates"], validation["positives"], validation_scores)
    selected_method = max(validation_metrics, key=lambda name: validation_metrics[name]["ndcg@10"])
    test = evaluations["test"]
    test_scores = scores(test["candidates"])
    test_metrics = evaluate_scores(test["candidates"], test["positives"], test_scores)
    zero = dormant_positives(test["positives"], test["recent"], 0, set(test["recent"]["sku"]))
    _, hidden = mask_selling_edges(test["recent"], config.seed + 10000)
    masked, _, _, _ = fetch(config.test_decision, hidden=hidden)
    mask_metrics = evaluate_scores(masked, hidden, scores(masked))
    summary = {
        "experiment": "full_history_dormant_v1",
        "config": {key: value.isoformat() if isinstance(value, date) else value for key, value in asdict(config).items()},
        "source": {"fact_rows": len(facts), "skus": int(facts["sku"].nunique()),
                   "first_day": dates.min().date().isoformat(), "last_day": dates.max().date().isoformat(),
                   "days_present": len(source_days), "missing_days": pd.date_range(dates.min(), dates.max()).difference(source_days).strftime("%Y-%m-%d").tolist(),
                   "rows_after_test_target_not_used": int((dates > pd.Timestamp(config.test_decision + timedelta(days=config.window_days))).sum())},
        "dictionary": dict_audit, "data": audit, "embedding_model": embedding_meta,
        "training": {"rows": len(training), "skus": int(training["sku"].nunique()),
                     "groups": int(training["group_id"].nunique()), "per_task": audits},
        "skipped_training_windows": skipped, "selected_iterations": selected_iterations,
        "validation_trials": trials, "fusion": fitted,
        "selected_method": selected_method, "selection_metric": "validation NDCG@10 only",
        "validation": validation_metrics, "test": test_metrics,
        "zero_views_test": evaluate_scores(test["candidates"], zero, test_scores),
        "mask_and_recover": mask_metrics,
        "selected_minus_lexical_recall10": bootstrap_difference(test["candidates"], test["positives"],
                                                                test_scores[selected_method], test_scores["lexical"], config.seed),
        "limitations": [
            "Сентябрьский test уже изучался; это диагностическая, не новая независимая оценка.",
            "Положительная цель — будущий заказ по новому кластеру; отсутствие заказа не доказывает нерелевантность.",
            "Неизвестные пары используются как слабые сравнения PU, а не подтвержденные отрицательные ответы.",
            "Просмотры карточки не являются кликами; суммы пользователей по дням и фразам не уникальны за период.",
            "Порог малых просмотров — параметр эксперимента, не доказанная бизнес-граница.",
            "Нулевые просмотры не доказывают отсутствие фразы в SEO; исторических текстов карточки нет.",
            "Пропуски старой истории сохранены; нет гарантии, что новый кластер никогда не продавал до наблюдаемого периода.",
            "Товарная совместимость не проверена; выдача review, не готовое add_candidate. Фото не использовались.",
            "SKU без продающих ориентиров и без недавних наблюдений не входят в оценку; холодный старт остается открытым.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".dormant-") as temporary:
        folder = Path(temporary) / "run"
        folder.mkdir()
        write_json(folder / "metrics.json", summary)
        training.to_csv(folder / "training_pairs.csv", index=False)
        for name, part in evaluations.items():
            predictions = part["candidates"].copy()
            for method, values in scores(predictions).items():
                predictions[f"score_{method}"] = values
            predictions["score"] = predictions[f"score_{selected_method}"]
            predictions["decision"] = "review"
            predictions["seo_presence"] = "unknown"
            predictions.to_csv(folder / f"{name}_candidates.csv", index=False)
            part["positives"].to_csv(folder / f"{name}_positives.csv", index=False)
        masked.to_csv(folder / "masked_candidates.csv", index=False)
        hidden.to_csv(folder / "masked_positives.csv", index=False)
        model.save_model(str(folder / "ranker.cbm"))
        code = [Path(__file__), Path(__file__).with_name("dormant_learning.py"),
                Path(__file__).with_name("cluster_recommender.py"), Path(__file__).with_name("cluster_data.py"),
                Path(__file__).with_name("graph_learning.py")]
        for name, path in paths.items():
            if sha256(path) != inputs[name]["sha256"]:
                raise ValueError("Входные данные изменились во время эксперимента")
        write_json(folder / "manifest.json", {
            "inputs": inputs, "code": {path.name: sha256(path) for path in code},
            "python": sys.version,
            "packages": {name: importlib.metadata.version(name)
                         for name in ["numpy", "pandas", "scipy", "scikit-learn", "catboost"]},
            "files": {path.name: sha256(path) for path in folder.iterdir()},
        })
        if output.exists():
            raise ValueError("Результат появился во время обучения")
        folder.replace(output)
    return summary


def verify_folder(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    artifacts = manifest.get("files", manifest.get("artifact_sha256", {}))
    for name, digest in artifacts.items():
        if sha256(folder / name) != digest:
            raise ValueError("Контрольная сумма артефакта не совпадает")
    return manifest


def graph_vectors(model, catalog):
    vectors = np.zeros((len(catalog), model.query_embeddings.shape[1]), dtype=np.float32)
    indices = {query: position for position, query in enumerate(catalog)}
    for query, row in zip(model.queries, normalize(model.query_embeddings)):
        if query not in indices:
            raise ValueError("LightGCN содержит запрос вне каталога")
        vectors[indices[query]] = row
    return vectors


def enrich_candidates(base, history, recent, demand, catalog, vectors, adapted,
                      graph, text, config, active):
    """Расширяем пул двумя каналами, признаки CatBoost считаем в исходном E5.

    Явный запрет прежних продаж и >max_views применяется до top-N каждого канала.
    Cold LightGCN: нулевой вектор означает отсутствие сигнала, не отрицательный ответ.
    """
    retriever = HybridRetriever(catalog, history, demand, vectors, text)
    recent_parts = {sku: rows.set_index("normquery") for sku, rows in recent.groupby("sku")}
    base_parts = {sku: rows for sku, rows in base.groupby("sku")}
    records = []
    for sku, own in history.groupby("sku", sort=True):
        if sku not in active:
            continue
        seeds = own.loc[own["orders"].gt(0)].sort_values(
            ["orders", "normquery"], ascending=[False, True], kind="stable"
        )
        if seeds.empty:
            continue
        anchors = seeds.head(config.max_seeds).sort_values("normquery")
        indices = [retriever.index[name] for name in anchors["normquery"]]
        weights = anchors["orders"].to_numpy(dtype=float)
        weights = weights / weights.sum()
        adapter_scores = adapted @ (adapted[indices].T @ weights)
        known_seeds = [retriever.index[name] for name in seeds["normquery"]
                       if np.any(graph[retriever.index[name]])]
        profile = graph[known_seeds].mean(axis=0) if known_seeds else np.zeros(graph.shape[1])
        profile /= max(float(np.linalg.norm(profile)), 1e-12)
        graph_scores = graph @ profile
        allowed = np.ones(len(catalog), dtype=bool)
        allowed[[retriever.index[name] for name in seeds["normquery"]]] = False
        exposure = recent_parts.get(sku, recent.iloc[:0].set_index("normquery"))
        blocked = exposure.loc[exposure["views"].gt(config.max_views)].index
        allowed[[retriever.index[name] for name in blocked]] = False

        def top(scores, allowed=allowed):
            positions = np.flatnonzero(allowed & (scores > 0))
            return positions[np.argsort(-scores[positions], kind="stable")[:config.per_source]]

        extra = {"adapter": set(top(adapter_scores)), "lightgcn": set(top(graph_scores))}
        previous = base_parts.get(sku, base.iloc[:0])
        original = {row["normquery"]: row for row in previous.to_dict("records")}
        positions = sorted({retriever.index[name] for name in original} | set().union(*extra.values()))
        if not positions:
            continue
        lexical_values = (retriever.text[positions] @ retriever.text[indices].T).toarray()
        semantic_values = retriever.embeddings[positions] @ retriever.embeddings[indices].T
        neighbours = retriever.sales @ retriever.sales[retriever.sku_index[sku]].T
        co_sales = np.asarray((retriever.sales[:, positions].T @ neighbours).toarray()).ravel()
        for position, index in enumerate(positions):
            name = catalog[index]
            row = original.get(name)
            if row is None:
                pair = exposure.loc[name] if name in exposure.index else None
                row = {
                    "sku": sku, "normquery": name,
                    "closest_seed": catalog[indices[int(semantic_values[position].argmax())]],
                    "retrieval_sources": "", "lexical_max": lexical_values[position].max(),
                    "lexical_mean": lexical_values[position] @ weights,
                    "embedding_max": semantic_values[position].max(),
                    "embedding_mean": semantic_values[position] @ weights,
                    "graph_score": co_sales[position],
                    "log_cluster_orders": np.log1p(retriever.popularity[index]),
                    "log_demand": np.log1p(retriever.demand[index]),
                    "log_views": np.log1p(pair["views"]) if pair is not None else 0.,
                    "log_searches": np.log1p(pair["searches"]) if pair is not None else 0.,
                    "history_days": pair["days"] if pair is not None else 0,
                    "log_seed_orders": np.log1p(seeds["orders"].sum()), "seed_count": len(seeds),
                    **{f"{source}_source": 0 for source in ["text", "embedding", "graph", "popular"]},
                }
            else:
                row = {column: row[column] for column in CANDIDATE_COLUMNS}
            sources = [source for source, items in extra.items() if index in items]
            row["retrieval_sources"] = ",".join(filter(None, [row["retrieval_sources"], *sources]))
            row.update(adapter_score=float(adapter_scores[index]), lightgcn_score=float(graph_scores[index]),
                       lightgcn_available=bool(np.any(graph[index]) and known_seeds))
            records.append(row)
    return pd.DataFrame(records, columns=[*CANDIDATE_COLUMNS, "adapter_score", "lightgcn_score", "lightgcn_available"])


def refinement_scores(candidates, model, fusion=None):
    scores = {name: candidates[column].to_numpy() for name, column in CHANNELS.items()
              if name != "catboost_pu"}
    scores["popularity"] = candidates["log_cluster_orders"].to_numpy()
    scores["catboost_pu"] = predict_ranker(model, candidates)
    if fusion is not None:
        frame = candidates.assign(catboost_score=scores["catboost_pu"])
        scores["learned_fusion"] = fusion_scores(frame, fusion["weights"], fusion["offset"])
    return scores


def run_refinement(baseline, adapter, graph_path, output):
    if output.exists():
        raise ValueError("Результат уже существует")
    manifest = verify_folder(baseline)
    adapter_manifest = verify_folder(adapter)
    graph_manifest = verify_folder(graph_path)
    graph_model = load_graph(graph_path)
    metrics = json.loads((baseline / "metrics.json").read_text())
    settings = metrics["config"].copy()
    for field in ["dictionary_as_of", "first_decision", "validation_decision", "test_decision"]:
        settings[field] = date.fromisoformat(settings[field])
    settings["iterations"] = tuple(settings["iterations"])
    config = DormantConfig(**settings)
    training_cutoff = adapter_manifest["training_cutoff"]
    if pd.Timestamp(training_cutoff).date() > config.validation_decision or (
        pd.Timestamp(graph_model.train_end).date() > config.validation_decision
    ):
        raise ValueError("Обучение нейросетей использовало validation/test")
    for item in manifest["inputs"].values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("Изменён вход baseline")
        if adapter_manifest["input_sha256"].get(item["path"]) != item["sha256"]:
            raise ValueError("Адаптер обучен на других входных данных")
    for name in ["facts", "dictionary"]:
        if graph_manifest["inputs"][name]["sha256"] != manifest["inputs"][name]["sha256"]:
            raise ValueError("LightGCN обучен на других входных данных")
    facts = pd.read_csv(manifest["inputs"]["facts"]["path"], dtype={"sku": "string"}, keep_default_na=False)
    dictionary = pd.read_csv(manifest["inputs"]["dictionary"]["path"], keep_default_na=False)
    as_of = pd.Timestamp(config.dictionary_as_of, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    snapshot, _ = dictionary_snapshot(dictionary, as_of)
    daily, audit = aggregate_clusters(facts, snapshot)
    catalog = sorted(snapshot["normquery"].unique())
    vectors, _ = load_embeddings(Path(manifest["inputs"]["embeddings"]["path"]), catalog)
    adapted, adapter_meta = load_embeddings(adapter / "vectors.npz", catalog)
    if pd.Timestamp(adapter_meta["dictionary_as_of"]) != as_of:
        raise ValueError("Снимок адаптера отличается от baseline")
    graph = graph_vectors(graph_model, catalog)
    text = lexical_index(catalog)
    model = CatBoostRanker()
    model.load_model(str(baseline / "ranker.cbm"))
    source_days = pd.DatetimeIndex(pd.to_datetime(facts["date"]).unique())
    results, predictions, targets = {}, {}, {}
    decisions = {"validation": config.validation_decision, "test": config.test_decision}
    last_day = source_days.max().date()
    full_end = config.test_decision + timedelta(days=config.window_days)
    if last_day > full_end:
        decisions["new_days"] = full_end + timedelta(days=1)
    fitted, selected = None, None
    for split, decision in decisions.items():
        logger.info("Обученные каналы: %s / %s", split, decision)
        horizon = min(config.window_days, (last_day - decision).days)
        if split == "new_days" and horizon < 1:
            continue
        missing = missing_window_days(source_days, decision, config.window_days, include_target=False)
        future_days = pd.date_range(decision + timedelta(days=1), decision + timedelta(days=horizon))
        if missing or len(future_days.difference(source_days)):
            raise ValueError("Неполное окно проверки")
        history = period_pairs(daily, daily["date"].min().date(), decision)
        recent_start = decision - timedelta(days=config.window_days - 1)
        recent = period_pairs(daily, recent_start, decision)
        active = set(recent["sku"])
        part = daily.loc[daily["date"].between(pd.Timestamp(recent_start), pd.Timestamp(decision))]
        demand = part.drop_duplicates(["date", "normquery"]).groupby("normquery")["cluster_searches"].sum()
        if split in {"validation", "test"}:
            base = pd.read_csv(baseline / f"{split}_candidates.csv", dtype={"sku": "string"}, keep_default_na=False)
        else:
            base = HybridRetriever(catalog, history, demand, vectors, text).generate(
                config.per_source, max_history_views=config.max_views, traffic_history=recent,
                target_skus=active, max_seeds=config.max_seeds,
            )
        candidates = enrich_candidates(base, history, recent, demand, catalog, vectors, adapted,
                                       graph, text, config, active)
        future = period_pairs(daily, decision + timedelta(days=1), decision + timedelta(days=horizon))
        positives = dormant_positives(future_positives(history, future), recent, config.max_views, active)
        zero = dormant_positives(positives, recent, 0, active)
        scores = refinement_scores(candidates, model)
        if split == "validation":
            candidates["catboost_score"] = scores["catboost_pu"]
            fitted = fit_fusion(candidates, positives, ["pretrained_embeddings", "co_sales", "semantic_adapter", "lightgcn"])
        scores = refinement_scores(candidates, model, fitted)
        evaluation = evaluate_scores(candidates, positives, scores)
        if split == "validation":
            selected = max(evaluation, key=lambda name: evaluation[name]["ndcg@10"])
        results[split] = {
            "decision": decision.isoformat(), "target_end": (decision + timedelta(days=horizon)).isoformat(),
            "horizon_days": horizon, "models": evaluation,
            "zero_views": evaluate_scores(candidates, zero, scores),
            "selected_minus_lexical": bootstrap_difference(candidates, positives, scores[selected], scores["lexical"]),
        }
        predictions[split] = candidates.assign(**{f"score_{name}": score for name, score in scores.items()}, score=scores[selected])
        targets[split] = positives
    summary = {
        "baseline": str(baseline), "adapter": str(adapter), "graph": str(graph_path),
        "training_cutoff": training_cutoff, "selected_method": selected,
        "selection": "validation NDCG@10; before test and new_days", "fusion": fitted,
        "source": metrics["source"], "aggregation": {key: value for key, value in audit.items() if key != "unmapped_queries"},
        **results,
        "limitations": [
            "Сентябрьский период диагностический: он уже использовался ранее.",
            "Новые дни имеют короткий горизонт; показатели нельзя сравнивать с 14-дневным тестом как одинаковые.",
            "E5 заморожен: обучен residual-адаптер, не Transformer.",
            "LightGCN обучен на прошлых продажах; холодные кластеры не имеют графового сигнала.",
            "Mask-and-recover baseline сохранён отдельно; обученные представления не проверяются на скрытых ранее виденных рёбрах.",
            "Будущие заказы — косвенная цель, не экспертная релевантность и не эффект SEO.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for item in manifest["inputs"].values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("Вход изменился во время проверки")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".refinement-") as temporary:
        folder = Path(temporary) / "result"
        folder.mkdir()
        write_json(folder / "metrics.json", summary)
        for split, frame in predictions.items():
            frame.to_csv(folder / f"{split}_candidates.csv", index=False)
            targets[split].to_csv(folder / f"{split}_positives.csv", index=False)
        write_json(folder / "manifest.json", {
            "parents": {str(path): sha256(path / "manifest.json") for path in [baseline, adapter, graph_path]},
            "code": {name: sha256(Path(__file__).with_name(name)) for name in [
                "cluster_data.py", "cluster_recommender.py", "dormant_learning.py",
                "full_history_recommender.py", "graph_learning.py",
            ]},
            "packages": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "pandas", "catboost", "scikit-learn"]},
            "files": {path.name: sha256(path) for path in folder.iterdir()},
        })
        if output.exists():
            raise ValueError("Выход появился во время проверки")
        folder.replace(output)
    return summary


def run_fusion_ablation(evaluation, output):
    """Тот же расширенный пул: вклад каналов без изменения бюджета/знаменателя."""
    if output.exists():
        raise ValueError("Результат уже существует")
    verify_folder(evaluation)
    validation = pd.read_csv(evaluation / "validation_candidates.csv", dtype={"sku": "string"}, keep_default_na=False)
    positives = pd.read_csv(evaluation / "validation_positives.csv", dtype={"sku": "string"}, keep_default_na=False)
    controls = {
        "e5_co_sales": ["pretrained_embeddings", "co_sales"],
        "plus_adapter": ["pretrained_embeddings", "co_sales", "semantic_adapter"],
        "plus_lightgcn": ["pretrained_embeddings", "co_sales", "lightgcn"],
        "all_four": ["pretrained_embeddings", "co_sales", "semantic_adapter", "lightgcn"],
    }
    fitted = {name: fit_fusion(validation, positives, channels) for name, channels in controls.items()}
    result = {"comparison": "identical expanded candidate pool; each weight set selected on validation only",
              "weights": {name: model["weights"] for name, model in fitted.items()}}
    for split in ["test", "new_days"]:
        if not (evaluation / f"{split}_candidates.csv").exists():
            continue
        candidates = pd.read_csv(evaluation / f"{split}_candidates.csv", dtype={"sku": "string"}, keep_default_na=False)
        target = pd.read_csv(evaluation / f"{split}_positives.csv", dtype={"sku": "string"}, keep_default_na=False)
        scores = {name: fusion_scores(candidates, model["weights"], model["offset"])
                  for name, model in fitted.items()}
        result[split] = evaluate_scores(candidates, target, scores)
        result[f"{split}_full_minus_control"] = bootstrap_difference(
            candidates, target, scores["all_four"], scores["e5_co_sales"],
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".ablation-") as temporary:
        folder = Path(temporary) / "result"
        folder.mkdir()
        write_json(folder / "metrics.json", result)
        write_json(folder / "manifest.json", {
            "parent_sha256": sha256(evaluation / "manifest.json"),
            "code_sha256": sha256(Path(__file__)),
            "files": {"metrics.json": sha256(folder / "metrics.json")},
        })
        if output.exists():
            raise ValueError("Выход появился во время сравнения")
        folder.replace(output)
    return result


_RESULT_METRICS = ("recall@5", "recall@10", "precision@5", "precision@10",
                   "ndcg@5", "ndcg@10", "mrr", "retrieval_recall")
_RESULT_FEATURES = [column for column in CANDIDATE_COLUMNS
                    if column not in {"sku", "normquery", "closest_seed", "retrieval_sources"}]
_RESULT_SCENARIOS = ("clean", "score_noise_05", "score_noise_15", "counter_outliers",
                     "candidate_loss_10", "candidate_loss_30")


def _results_pairs(frame):
    if not frame.columns.is_unique or not {"sku", "normquery"}.issubset(frame.columns):
        raise ValueError("Нужны уникальные столбцы sku и normquery")
    for column in ["sku", "normquery"]:
        if frame[column].isna().any() or any(
            not isinstance(value, str) or not value.strip() for value in frame[column]
        ):
            raise ValueError("SKU и normquery должны быть заполненными строками")
    if frame.duplicated(["sku", "normquery"]).any():
        raise ValueError("Повторяющиеся пары SKU + normquery")
    return frame.sort_values(["sku", "normquery"], kind="stable").reset_index(drop=True).copy()


def _results_candidates(frame):
    ordered = _results_pairs(frame)
    required = [*CANDIDATE_COLUMNS, "adapter_score", "lightgcn_score", "lightgcn_available"]
    if not set(required).issubset(ordered.columns):
        raise ValueError("Отсутствуют обязательные столбцы кандидатов")
    numeric = [*_RESULT_FEATURES, "adapter_score", "lightgcn_score",
               *[name for name in ordered if name == "score" or name.startswith("score_")]]
    for column in numeric:
        try:
            values = pd.to_numeric(ordered[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Некорректный числовой столбец {column}") from error
        if not np.isfinite(values).all():
            raise ValueError(f"Неконечный столбец {column}")
        ordered[column] = values
    if (ordered[["graph_score", "log_cluster_orders", "log_demand"]] < 0).any().any():
        raise ValueError("Счётчики и co-sales должны быть неотрицательными")
    return ordered


def _results_stress(candidates, scenario, seed, loss_fraction=None):
    """Искусственный стресс сохранённых сигналов, без labels и перестройки графа.

    Для потери/выбросов выбираем ceil(fraction * rows) строк без возвращения.
    Canonical order фиксирует RNG независимо от порядка CSV и будущих positives.
    """
    frame = _results_candidates(candidates)
    if scenario not in _RESULT_SCENARIOS:
        raise ValueError("Неизвестный стресс-сценарий")
    rng = np.random.default_rng(seed)
    counts = {"input_pairs": len(frame), "affected_pairs": 0, "dropped_pairs": 0}
    if scenario.startswith("candidate_loss_"):
        fraction = loss_fraction if loss_fraction is not None else int(scenario[-2:]) / 100
        if isinstance(fraction, (bool, np.bool_)) or not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("loss_fraction должен быть от 0 до 1")
        count = int(np.ceil(len(frame) * fraction))
        dropped = rng.choice(len(frame), size=count, replace=False)
        frame = frame.drop(index=dropped).reset_index(drop=True)
        counts.update(affected_pairs=count, dropped_pairs=count)
    elif scenario.startswith("score_noise_"):
        sigma = int(scenario[-2:]) / 100
        for mean, maximum, low, high in [
            ("lexical_mean", "lexical_max", 0, 1),
            ("embedding_mean", "embedding_max", -1, 1),
            ("adapter_score", None, -1, 1), ("lightgcn_score", None, -1, 1),
        ]:
            spread = frame.groupby("sku")[mean].transform(
                lambda values: np.std(values.to_numpy(dtype=np.longdouble), ddof=0)
            ).to_numpy(dtype=np.longdouble)
            delta = rng.normal(size=len(frame)) * sigma * spread
            for column in [mean, *([maximum] if maximum else [])]:
                frame[column] = np.clip(frame[column].to_numpy(dtype=np.longdouble) + delta,
                                        low, high).astype(float)
        multiplier = rng.lognormal(0, sigma, len(frame))
        limit = np.finfo(float).max / np.maximum(multiplier, 1.)
        frame["graph_score"] = np.minimum(frame["graph_score"].to_numpy(), limit) * multiplier
        counts["affected_pairs"] = len(frame)
    elif scenario == "counter_outliers":
        count = int(np.ceil(len(frame) * .01))
        selected = rng.choice(len(frame), size=count, replace=False)
        for column in ["log_cluster_orders", "log_demand"]:
            values = frame.loc[selected, column].to_numpy(dtype=np.longdouble)
            # log1p(100 * expm1(x)), устойчиво даже при x > log(float_max).
            with np.errstate(divide="ignore"):
                inflated = np.logaddexp(0, np.log(100.) + values + np.log(-np.expm1(-values)))
            frame.loc[selected, column] = np.minimum(inflated, np.finfo(float).max).astype(float)
        graph = frame.loc[selected, "graph_score"].to_numpy()
        frame.loc[selected, "graph_score"] = np.minimum(graph, np.finfo(float).max / 100) * 100
        counts["affected_pairs"] = count
    counts["candidate_pairs"] = len(frame)
    return frame, counts


def _results_ranked(candidates, scores):
    frame = candidates[["sku", "normquery"]].assign(score=np.asarray(scores, dtype=float))
    if not np.isfinite(frame["score"]).all():
        raise ValueError("Scores должны быть конечными")
    return frame.sort_values(["sku", "score", "normquery"], ascending=[True, False, True], kind="stable")


def _results_recall(candidates, positives, scores):
    top = _results_ranked(candidates, scores).groupby("sku", sort=False).head(10)
    hits = positives[["sku", "normquery"]].merge(
        top[["sku", "normquery"]].assign(hit=1.), on=["sku", "normquery"], how="left", validate="one_to_one"
    )
    return hits.assign(hit=hits["hit"].fillna(0.)).groupby("sku", sort=True)["hit"].mean().to_numpy()


def _results_overlap(candidates, scores, clean, clean_scores, positives, k=10):
    def sets(frame, values):
        top = _results_ranked(frame, values).groupby("sku", sort=False).head(k)
        return {sku: set(group["normquery"]) for sku, group in top.groupby("sku", sort=False)}

    left, right = sets(candidates, scores), sets(clean, clean_scores)
    slots, jaccard = [], []
    for sku in sorted(positives["sku"].unique()):
        a, b = left.get(sku, set()), right.get(sku, set())
        slots.append(len(a & b) / k)
        jaccard.append(len(a & b) / len(a | b) if a | b else 1.)
    return {"intersection_over_k": float(np.mean(slots)) if slots else None,
            "jaccard": float(np.mean(jaccard)) if jaccard else None}


def _results_rank_distribution(candidates, positives, scores):
    ranked = _results_ranked(candidates, scores)
    ranked["rank"] = ranked.groupby("sku", sort=False).cumcount() + 1
    targets = positives[["sku", "normquery"]].merge(
        ranked[["sku", "normquery", "rank"]], on=["sku", "normquery"], how="left", validate="one_to_one"
    )
    rank = targets["rank"]
    buckets = {"1-5": rank.between(1, 5), "6-10": rank.between(6, 10),
               "11-50": rank.between(11, 50), ">50": rank.gt(50), "missing_from_pool": rank.isna()}
    return {"evaluated_skus": int(positives["sku"].nunique()), "positive_pairs": len(positives),
            "macro_fractions": {name: float(mask.groupby(targets["sku"]).mean().mean()) if len(targets) else None
                                for name, mask in buckets.items()},
            "pair_counts": {name: int(mask.sum()) for name, mask in buckets.items()}}


def _results_interval(values, seed, iterations):
    if not len(values):
        return {"skus": 0, "difference": None, "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    # Ограничиваем память; одна строка bootstrap всегда содержит полный набор SKU.
    samples = np.empty(iterations)
    batch = max(1, 1_000_000 // len(values))
    for start in range(0, iterations, batch):
        size = min(batch, iterations - start)
        samples[start:start + size] = values[rng.integers(0, len(values), (size, len(values)))].mean(axis=1)
    return {"skus": len(values), "difference": float(values.mean()),
            "lower": float(np.quantile(samples, .025)), "upper": float(np.quantile(samples, .975)),
            "repeats": iterations, "seed": seed, "unit": "SKU; scenario repeats averaged before bootstrap"}


def _results_mean_std(values):
    if not values or values[0] is None:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}


def run_results(evaluation: Path, output: Path, seed=7, repetitions=3, bootstrap_iterations=2000) -> dict:
    """Диагностика сохранённого пула: без обучения, поиска и подбора весов."""
    for name, value, minimum in [("seed", seed, 0), ("repetitions", repetitions, 1),
                                  ("bootstrap_iterations", bootstrap_iterations, 1)]:
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name}: нужен целый параметр >= {minimum}")
    evaluation, output = Path(evaluation), Path(output)
    if output.exists():
        raise ValueError("Результат уже существует; перезапись запрещена")
    inputs = {}

    def checked_folder(path, required):
        manifest_path = path / "manifest.json"
        digest = sha256(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        files = manifest.get("files", manifest.get("artifact_sha256", {}))
        if not set(required).issubset(files):
            raise ValueError("Manifest не содержит обязательные артефакты оценки")
        inputs[str(manifest_path)] = digest
        for name, expected in files.items():
            artifact = path / name
            if not artifact.resolve().is_relative_to(path.resolve()) or not artifact.is_file():
                raise ValueError("Некорректный путь артефакта")
            if sha256(artifact) != expected:
                raise ValueError("Контрольная сумма артефакта не совпадает")
            inputs[str(artifact)] = expected
        return manifest

    manifest = checked_folder(evaluation, ["metrics.json"])
    saved = json.loads((evaluation / "metrics.json").read_text())
    if saved.get("status") == "full_refit_for_review_not_independent_evaluation" or not {
        "baseline", "fusion", "selected_method", "test"
    }.issubset(saved):
        raise ValueError("Нужна сохранённая evaluation, не final release")
    baseline = Path(saved["baseline"])
    baseline_manifest = checked_folder(baseline, ["metrics.json", "ranker.cbm"])
    if manifest.get("parents", {}).get(str(baseline)) != inputs[str(baseline / "manifest.json")]:
        raise ValueError("Контрольная сумма parent baseline не совпадает")
    model = CatBoostRanker()
    model.load_model(str(baseline / "ranker.cbm"))
    fusion = saved["fusion"]
    if not isinstance(fusion, dict) or not isinstance(fusion.get("weights"), dict) or (
        isinstance(fusion.get("offset"), bool) or not isinstance(fusion.get("offset"), (int, float))
        or not np.isfinite(fusion["offset"]) or fusion["offset"] <= 0
    ):
        raise ValueError("Нужны сохранённые fusion weights и положительный offset")
    selected = saved["selected_method"]
    result = {
        "status": "fixed_pool_diagnostic_not_independent_evaluation", "selected_method": selected,
        "seed": seed, "repetitions": repetitions, "bootstrap_iterations": bootstrap_iterations,
        "fusion": {"weights": fusion["weights"], "offset": fusion["offset"]},
        "protocol": {
            "selection": saved.get("selection"), "scenario_seeds": [seed + i for i in range(repetitions)],
            "metrics": list(_RESULT_METRICS), "delta_metric": "recall@10", "std_ddof": 0,
            "bootstrap": "paired SKU; average per-SKU repeat deltas before resampling",
            "candidate_sampling": "ceil(fraction * rows), without replacement, independent of positives",
            "score_noise": "Gaussian .05/.15 * within-SKU population std; shared mean/max delta then clip; co-sales lognormal sigma=.05/.15",
            "counter_outliers": "1% rows: underlying global orders/demand and co-sales x100; log1p counters; own exposure unchanged",
            "top10": "macro all positive SKU; intersection/10; Jaccard(empty,empty)=1; empty slots=0",
        },
        "limitations": [
            "Фиксированный сохранённый пул: не end-to-end оценка и не шум обучения; retrieval и веса не обновляются.",
            "Шумы scores и выбросы признаков искусственные: не производственная порча, не опечатки и не перестройка графа.",
            "Test уже использовался: диагностическая, не новая независимая проверка.",
            "New_days оценивается отдельно; четыре дня недостаточны и не равноценны полному тестовому горизонту.",
            "Precision — proxy будущих заказов, не экспертная релевантность; отсутствие заказа не означает negative.",
            "Все future positives и SKU остаются в знаменателе, включая потерянные при поиске и стрессе.",
        ],
    }
    for split in ["test", "new_days"]:
        if split not in saved:
            if split == "new_days" and any(name.startswith("new_days_") for name in manifest["files"]):
                raise ValueError("New_days артефакты без метрик")
            continue
        required = [f"{split}_candidates.csv", f"{split}_positives.csv"]
        if not set(required).issubset(manifest["files"]):
            raise ValueError("Manifest не содержит обязательные артефакты split")
        read_options = {"dtype": {"sku": "string", "normquery": "string"},
                        "keep_default_na": False, "float_precision": "round_trip"}
        clean = _results_candidates(pd.read_csv(evaluation / required[0], **read_options))
        positives = _results_pairs(pd.read_csv(evaluation / required[1], **read_options))
        scores = refinement_scores(clean, model, fusion)
        if len(scores) != 8 or selected not in scores:
            raise ValueError("Нужны восемь методов и сохранённый selected_method")
        comparisons = evaluate_scores(clean, positives, scores)
        for method, actual in comparisons.items():
            expected = saved[split]["models"].get(method, {})
            for metric in [*_RESULT_METRICS, "evaluated_skus", "positive_pairs", "candidate_pairs"]:
                left, right = actual[metric], expected.get(metric)
                if (left is None) != (right is None) or (
                    left is not None and (not np.isfinite(right) or abs(left - right) > 1e-12)
                ):
                    raise ValueError(f"Clean parity нарушен: {split}/{method}/{metric}")
        section = {key: saved[split][key] for key in ["decision", "target_end", "horizon_days"]}
        section.update(
            clean_comparisons=comparisons,
            rank_distribution=_results_rank_distribution(clean, positives, scores[selected]),
            clean_paired={f"selected_minus_{name}": bootstrap_difference(
                clean, positives, scores[selected], scores[name], seed=seed, repeats=bootstrap_iterations
            ) for name in ["lexical", "co_sales"]},
        )
        clean_recall = _results_recall(clean, positives, scores[selected])
        scenarios = {}
        for scenario in _RESULT_SCENARIOS:
            runs, counts, overlaps, deltas = [], [], [], []
            for iteration in range(1 if scenario == "clean" else repetitions):
                frame, count = _results_stress(clean, scenario, seed + iteration)
                stressed_scores = scores if scenario == "clean" else refinement_scores(frame, model, fusion)
                runs.append(evaluate_scores(frame, positives, {
                    "selected": stressed_scores[selected], "catboost_pu": stressed_scores["catboost_pu"]
                }))
                counts.append(count)
                overlaps.append(_results_overlap(frame, stressed_scores[selected], clean, scores[selected], positives))
                deltas.append(_results_recall(frame, positives, stressed_scores[selected]) - clean_recall)
            scenarios[scenario] = {
                "repetitions": len(runs), "seeds": [seed + i for i in range(len(runs))],
                "models": {method: {metric: _results_mean_std([run[method][metric] for run in runs])
                                    for metric in _RESULT_METRICS} for method in ["selected", "catboost_pu"]},
                "counts": {key: _results_mean_std([count[key] for count in counts]) for key in counts[0]},
                "selected_top10": {key: _results_mean_std([overlap[key] for overlap in overlaps]) for key in overlaps[0]},
                "selected_delta_vs_clean": _results_interval(np.mean(deltas, axis=0), seed, bootstrap_iterations),
            }
        section["scenarios"] = scenarios
        result[split] = section
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".results-") as temporary:
        folder = Path(temporary) / "result"
        folder.mkdir()
        write_json(folder / "metrics.json", result)
        write_json(folder / "manifest.json", {
            "parents": {str(path): inputs[str(path / "manifest.json")] for path in [evaluation, baseline]},
            "inputs": inputs, "baseline_training_cutoff": baseline_manifest.get("training_cutoff"),
            "code": {name: sha256(Path(__file__).with_name(name)) for name in [
                "full_history_recommender.py", "dormant_learning.py", "cluster_recommender.py", "cluster_data.py"
            ]},
            "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "catboost", "scikit-learn"]},
            "files": {"metrics.json": sha256(folder / "metrics.json")},
        })
        for path, expected in inputs.items():
            if sha256(Path(path)) != expected:
                raise ValueError("Вход изменился во время проверки результатов")
        if output.exists():
            raise ValueError("Выход появился во время проверки результатов")
        folder.replace(output)
    return result


def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["results"]:
        parser = argparse.ArgumentParser(description="Диагностика качества и устойчивости сохранённой evaluation")
        parser.add_argument("--evaluation", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--seed", type=int, default=7)
        parser.add_argument("--repetitions", type=int, default=3)
        parser.add_argument("--bootstrap-iterations", type=int, default=2000)
        args = parser.parse_args(argv[1:])
        run_results(args.evaluation, args.output, args.seed, args.repetitions, args.bootstrap_iterations)
        return
    if argv[:1] == ["refine"]:
        parser = argparse.ArgumentParser(description="Проверка обученных каналов на будущем; test не используется при подборе")
        for name in ["baseline", "adapter", "graph", "evaluation"]:
            parser.add_argument(f"--{name}", type=Path)
        for name in ["output"]:
            parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument("--ablate-fusion", action="store_true")
        args = parser.parse_args(argv[1:])
        logging.basicConfig(level=logging.INFO)
        if args.ablate_fusion:
            if args.evaluation is None:
                parser.error("Нужен --evaluation")
            run_fusion_ablation(args.evaluation, args.output)
            return
        if any(value is None for value in [args.baseline, args.adapter, args.graph]):
            parser.error("Для проверки нужны --baseline, --adapter и --graph")
        result = run_refinement(args.baseline, args.adapter, args.graph, args.output)
        logger.info("Выбран на validation: %s", result["selected_method"])
        return
    parser = argparse.ArgumentParser(description="Полная история и рекомендатель малотрафиковых кластеров")
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    for name in ["dictionary-as-of", "first-decision", "validation-decision", "test-decision"]:
        parser.add_argument(f"--{name}", type=date.fromisoformat, required=True)
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument("--max-views", type=int, default=5)
    parser.add_argument("--per-source", type=int, default=150)
    parser.add_argument("--iterations", type=int, nargs="+", default=[100, 300])
    parser.add_argument("--mask-rounds", type=int, default=1)
    parser.add_argument("--max-seeds", type=int, default=32)
    parser.add_argument("--training-stride-days", type=int,
                        help="Шаг обучающих решений; перекрывающиеся окна не независимые наблюдения")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    config = DormantConfig(args.dictionary_as_of, args.first_decision, args.validation_decision,
                            args.test_decision, args.window_days, args.max_views, args.per_source,
                            tuple(args.iterations), args.mask_rounds, max_seeds=args.max_seeds,
                            training_stride_days=args.training_stride_days)
    result = run_full_history(args.facts, args.dictionary, args.embeddings, args.output, config)
    logger.info("Результат: %s; метод %s", args.output, result["selected_method"])


if __name__ == "__main__":
    main()