"""Отдельный полный refit и очередь review; не повторная оценка на обученном test."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import tempfile
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from catboost import CatBoostRanker

from cluster_data import (
    aggregate_clusters,
    dictionary_snapshot,
    explicit_timestamp,
    sha256,
    write_json,
)
from cluster_recommender import lexical_index, load_embeddings, mask_selling_edges
from dormant_learning import fit_ranker, learning_rows
from full_history_recommender import (
    DormantConfig,
    enrich_candidates,
    generate_dormant,
    graph_vectors,
    missing_window_days,
    refinement_scores,
)
from graph_learning import load_graph
from semantic_adapter import load_adapter
from seo_recommendation_policy import annotate_recommendations, review_top_k

logger = logging.getLogger(__name__)


def _local_path(value):
    path = Path(value)
    if ".." in path.parts or "\\" in str(value) or "://" in str(value):
        raise ValueError("Недопустимый локальный путь")
    path = path.absolute()
    if any(item.is_symlink() for item in [path, *path.parents]):
        raise ValueError("Символические ссылки в артефактах запрещены")
    return path


def _track(path, digest, hashes):
    path = _local_path(path)
    if not path.is_file() or sha256(path) != digest:
        raise ValueError(f"Контрольная сумма не совпадает: {path}")
    if path in hashes and hashes[path] != digest:
        raise ValueError("Несогласованные контрольные суммы входов")
    hashes[path] = digest


def _verify_tree(folder, manifests, hashes, visiting=None):
    folder = _local_path(folder)
    visiting = set() if visiting is None else visiting
    if folder in visiting:
        raise ValueError("Цикл родителей в manifest")
    if folder in manifests:
        return manifests[folder]
    visiting.add(folder)
    path = folder / "manifest.json"
    _track(path, sha256(path), hashes)
    manifest = json.loads(path.read_text())
    tables = [manifest[key] for key in ["files", "artifact_sha256", "checksums"] if key in manifest]
    if len(tables) != 1 or not isinstance(tables[0], dict) or not tables[0]:
        raise ValueError("Manifest не содержит контрольных сумм артефактов")
    artifacts = tables[0]
    for name, digest in artifacts.items():
        if Path(name).name != name or name in {".", "..", "manifest.json"}:
            raise ValueError("Путь артефакта выходит за папку")
        _track(folder / name, digest, hashes)
    for item in manifest.get("inputs", {}).values():
        _track(item["path"], item["sha256"], hashes)
    for name, digest in manifest.get("input_sha256", {}).items():
        _track(name, digest, hashes)
    if "products" in manifest:
        item = manifest["products"]
        _track(item["path"], item["sha256"], hashes)
    for name, digest in manifest.get("parents", {}).items():
        parent = _local_path(name)
        _track(parent / "manifest.json", digest, hashes)
        _verify_tree(parent, manifests, hashes, visiting)
    visiting.remove(folder)
    manifests[folder] = manifest
    return manifest


def _require_files(manifest, names, field="files"):
    if not set(names).issubset(manifest.get(field, {})):
        raise ValueError(f"Manifest не содержит обязательные контрольные суммы: {', '.join(names)}")


def _validate_request(output, as_of, top_k):
    output = _local_path(output)
    if output.exists():
        raise ValueError("Результат уже существует")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k должен быть положительным целым")
    return output, explicit_timestamp(as_of)


def _parent(manifest, path):
    if _local_path(path) not in {_local_path(name) for name in manifest.get("parents", {})}:
        raise ValueError("Родитель модели не соответствует manifest")


def _matching_inputs(manifest, inputs, adapter=False):
    found = {_local_path(path): digest for path, digest in manifest.get("input_sha256", {}).items()}
    for name, item in inputs.items():
        if adapter:
            if found.get(_local_path(item["path"])) != item["sha256"]:
                raise ValueError("Адаптер обучен на других входных данных")
        elif name in {"facts", "dictionary"}:
            other = manifest.get("inputs", {}).get(name, {})
            if (other.get("sha256") != item["sha256"] or not other.get("path")
                    or _local_path(other["path"]) != _local_path(item["path"])):
                raise ValueError("Граф обучен на других входных данных")


def _cutoff(value, as_of):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Дата обучения модели позже as_of или отсутствует")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    stamp = stamp.tz_convert("UTC")
    if pd.isna(stamp) or stamp > as_of:
        raise ValueError("Дата обучения модели позже as_of или отсутствует")
    return stamp


def _adapter_cutoff(folder, manifest, as_of):
    _require_files(manifest, ["adapter.pt", "vectors.npz", "metrics.json"], "artifact_sha256")
    cutoff = _cutoff(manifest.get("training_cutoff"), as_of)
    model = load_adapter(folder / "adapter.pt")
    with np.load(folder / "vectors.npz", allow_pickle=False) as stored:
        if ("training_cutoff" not in stored or _cutoff(str(stored["training_cutoff"].item()), as_of) != cutoff
                or _cutoff(model.training_cutoff, as_of) != cutoff):
            raise ValueError("Дата векторов/весов адаптера отличается от manifest")
    return cutoff


def _prepare(evaluation, adapter, graph_path, products_path, as_of, manifests, hashes):
    evaluation, adapter, graph_path, products_path = map(_local_path, [evaluation, adapter, graph_path, products_path])
    for folder in [evaluation, adapter, graph_path]:
        _verify_tree(folder, manifests, hashes)
    evaluation_manifest = manifests[evaluation]
    _require_files(evaluation_manifest, ["metrics.json"])
    evaluation_metrics = json.loads((evaluation / "metrics.json").read_text())
    baseline = _local_path(evaluation_metrics["baseline"])
    _parent(evaluation_manifest, baseline)
    baseline_manifest = manifests[baseline]
    _require_files(baseline_manifest, ["metrics.json"])
    if not {"facts", "dictionary", "embeddings"}.issubset(baseline_manifest.get("inputs", {})):
        raise ValueError("Manifest baseline не содержит обязательные входы")
    baseline_metrics = json.loads((baseline / "metrics.json").read_text())
    settings = baseline_metrics["config"].copy()
    for field in ["dictionary_as_of", "first_decision", "validation_decision", "test_decision"]:
        settings[field] = date.fromisoformat(settings[field])
    settings["iterations"] = tuple(settings["iterations"])
    config = DormantConfig(**settings)
    paths = {name: _local_path(item["path"]) for name, item in baseline_manifest["inputs"].items()}
    evaluation_models = {}
    for name in ["adapter", "graph"]:
        parent = _local_path(evaluation_metrics[name])
        _parent(evaluation_manifest, parent)
        _matching_inputs(manifests[parent], baseline_manifest["inputs"], adapter=name == "adapter")
        if name == "adapter":
            cutoff = _adapter_cutoff(parent, manifests[parent], as_of)
        else:
            _require_files(manifests[parent], ["graph.pt", "catalog.json", "metrics.json"], "checksums")
            evaluation_models[name] = load_graph(parent)
            cutoff = _cutoff(evaluation_models[name].train_end, as_of)
        if cutoff.date() > config.validation_decision:
            raise ValueError("Родитель evaluation использовал validation/test")
    product_hash = sha256(products_path)
    _track(products_path, product_hash, hashes)
    facts = pd.read_csv(paths["facts"], dtype={"sku": "string"}, keep_default_na=False)
    dates = pd.to_datetime(facts["date"], utc=True)
    if facts.empty or dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Факты должны содержать непустые календарные дни")
    days = pd.DatetimeIndex(dates.dt.tz_localize(None).unique())
    decision = days.max().date()
    if pd.Timestamp(decision, tz="UTC") + pd.Timedelta(days=1) > as_of:
        raise ValueError("Снимок содержит ещё не завершённый день относительно as_of")
    if missing_window_days(days, decision, config.window_days, include_target=False):
        raise ValueError("Неполное окно недавних просмотров")
    adapter_manifest = manifests[adapter]
    graph_manifest = manifests[graph_path]
    _matching_inputs(adapter_manifest, baseline_manifest["inputs"], adapter=True)
    _matching_inputs(graph_manifest, baseline_manifest["inputs"])
    _require_files(graph_manifest, ["graph.pt", "catalog.json", "metrics.json"], "checksums")
    graph_model = load_graph(graph_path)
    if _adapter_cutoff(adapter, adapter_manifest, as_of).date() != decision or (
        _cutoff(graph_model.train_end, as_of).date() != decision
    ):
        raise ValueError("Финальные нейросети должны быть обучены по последний день фактов")
    snapshot_time = pd.Timestamp(config.dictionary_as_of, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    if snapshot_time > as_of:
        raise ValueError("Снимок каталога позже as_of")
    snapshot, _ = dictionary_snapshot(pd.read_csv(paths["dictionary"], keep_default_na=False), snapshot_time)
    daily, _ = aggregate_clusters(facts, snapshot)
    catalog = sorted(snapshot["normquery"].unique())
    vectors, metadata = load_embeddings(paths["embeddings"], catalog)
    adapted, adapter_meta = load_embeddings(adapter / "vectors.npz", catalog)
    for item in [metadata, adapter_meta, adapter_manifest]:
        if pd.Timestamp(item.get("dictionary_as_of", "NaT")) != snapshot_time:
            raise ValueError("Снимок embeddings/адаптера отличается от каталога")
    if adapter_meta["model_revision"] != sha256(adapter / "adapter.pt"):
        raise ValueError("Векторы не соответствуют сохранённому адаптеру")
    evaluation_adapter = _local_path(evaluation_metrics["adapter"])
    _, evaluation_adapter_meta = load_embeddings(evaluation_adapter / "vectors.npz", catalog)
    if (pd.Timestamp(evaluation_adapter_meta.get("dictionary_as_of", "NaT")) != snapshot_time
            or pd.Timestamp(manifests[evaluation_adapter].get("dictionary_as_of", "NaT")) != snapshot_time
            or evaluation_adapter_meta["model_revision"] != sha256(evaluation_adapter / "adapter.pt")):
        raise ValueError("Снимок адаптера evaluation отличается от каталога/весов")
    for model in [graph_model, evaluation_models["graph"]]:
        if (list(model.catalog_queries) != catalog
                or _cutoff(model.audit.get("dictionary_as_of"), as_of) != snapshot_time):
            raise ValueError("Каталог LightGCN отличается от baseline")
    graph = graph_vectors(graph_model, catalog)
    text = lexical_index(catalog)
    products = pd.read_csv(products_path, dtype={"sku": "string"}, keep_default_na=False)
    for value in products["snapshot_at"]:
        if explicit_timestamp(value) > as_of:
            raise ValueError("snapshot_at > as_of: будущие метаданные запрещены")
    return SimpleNamespace(
        evaluation=evaluation, adapter=adapter, graph_path=graph_path, baseline=baseline,
        baseline_manifest=baseline_manifest, baseline_metrics=baseline_metrics,
        evaluation_metrics=evaluation_metrics, products_path=products_path, product_hash=product_hash,
        facts=facts, days=days, decision=decision, config=config, daily=daily, catalog=catalog,
        vectors=vectors, adapted=adapted, graph=graph, text=text, products=products,
    )


def _candidates(prepared, hidden=None):
    base, history, recent, active = generate_dormant(
        prepared.daily, prepared.catalog, prepared.decision, prepared.config,
        prepared.vectors, prepared.text, hidden=hidden,
    )
    recent_start = prepared.decision - timedelta(days=prepared.config.window_days - 1)
    part = prepared.daily.loc[prepared.daily["date"].between(pd.Timestamp(recent_start), pd.Timestamp(prepared.decision))]
    demand = part.drop_duplicates(["date", "normquery"]).groupby("normquery")["cluster_searches"].sum()
    candidates = enrich_candidates(base, history, recent, demand, prepared.catalog,
                                   prepared.vectors, prepared.adapted, prepared.graph,
                                   prepared.text, prepared.config, active)
    return candidates, recent


def _queue(candidates, model, prepared, as_of, top_k):
    scores = refinement_scores(candidates, model, prepared.evaluation_metrics["fusion"])
    method = prepared.evaluation_metrics["selected_method"]
    if method not in scores:
        raise ValueError("Неизвестный сохранённый метод ранжирования")
    candidates = candidates.assign(score=scores[method])
    candidates["history_views"] = np.expm1(candidates["log_views"].astype(float))
    candidates["selling_seed"] = candidates["closest_seed"]
    reasons = []
    for row in candidates.to_dict("records"):
        signals = []
        if row["embedding_max"] > 0 or row["adapter_score"] > 0:
            signals.append("semantic_similarity")
        if row["graph_score"] > 0 or (row["lightgcn_available"] and row["lightgcn_score"] > 0):
            signals.append("behavioral_similarity")
        if row["lexical_max"] > 0:
            signals.append("lexical_similarity")
        if row["log_cluster_orders"] > 0:
            signals.append("catalog_sales")
        reasons.append(f"ranked_by={method}; signals={','.join(signals) or 'none'}; compatibility_not_verified")
    candidates["recommendation_basis"] = pd.Series(reasons, index=candidates.index, dtype="string")
    annotated = annotate_recommendations(candidates, prepared.products, as_of)
    return annotated, review_top_k(annotated, top_k)


def _summary(prepared, annotated, shortlist, as_of, top_k):
    return {
        "as_of": as_of.isoformat(), "source_rows": len(prepared.facts),
        "source_skus": int(prepared.facts["sku"].nunique()),
        "source_start": prepared.days.min().date().isoformat(), "source_end": prepared.decision.isoformat(),
        "source": prepared.baseline_metrics["source"],
        "catalog_as_of": prepared.config.dictionary_as_of.isoformat(),
        "traffic_lag_calendar_days": (as_of.date() - prepared.decision).days,
        "history_views_window_days": prepared.config.window_days, "max_history_views": prepared.config.max_views,
        "top_k": top_k, "selected_method": prepared.evaluation_metrics["selected_method"],
        "fusion_weights": prepared.evaluation_metrics["fusion"]["weights"],
        "score_semantics": "ranking_priority_not_probability",
        "candidate_pairs": len(annotated), "review_pairs": len(shortlist),
        "review_skus": int(shortlist["sku"].nunique()),
        "decisions": {name: int(value) for name, value in annotated["decision"].value_counts().items()},
        "compatibility": {name: int(value) for name, value in annotated["compatibility"].value_counts().items()},
        "seo_presence": {name: int(value) for name, value in annotated["seo_presence"].value_counts().items()},
        "expert_labels_collected": 0,
        "limitations": [
            "Полное переобучение включает прежние ответы test; его качество не оценено на этом test.",
            "Текущая маскировка используется только для обучения; финальные нейросети уже видели скрываемые связи.",
            "E5 заморожен: обучен residual-адаптер; фото и OCR не используются.",
            "Отсев ограничен явно известными свойствами; отсутствие конфликта не подтверждает релевантность.",
            "Текущие поля не доказывают отсутствие фразы во всех SEO-полях, add_candidate не выдаётся.",
            "Ручная оценка отложена; экспертная разметка не требуется для применения релиза.",
            "history_views — сумма просмотров недавнего окна, не уникальные пользователи за весь период.",
            "Используется вся доступная история продаж всех SKU; пропуски старой истории не восстанавливаются.",
        ],
    }


def _write_output(output, summary, annotated, shortlist, prepared, parents, hashes, model=None, training=None):
    for path, digest in hashes.items():
        _track(path, digest, {})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".release-") as temporary:
        folder = Path(temporary) / "result"
        folder.mkdir()
        if model is not None:
            model.save_model(str(folder / "ranker.cbm"))
            training.to_csv(folder / "training_pairs.csv", index=False)
        write_json(folder / ("metrics.json" if model is not None else "summary.json"), summary)
        shortlist.to_csv(folder / "recommendations_for_review.csv", index=False)
        annotated.loc[annotated["decision"].ne("review")].to_csv(folder / "filtered_candidates.csv", index=False)
        write_json(folder / "manifest.json", {
            "parents": {str(path): hashes[_local_path(path) / "manifest.json"] for path in parents},
            "inputs": prepared.baseline_manifest["inputs"],
            "products": {"path": str(prepared.products_path), "sha256": prepared.product_hash},
            "code": {name: sha256(Path(__file__).with_name(name)) for name in ["finalize_recommender.py", "full_history_recommender.py", "seo_recommendation_policy.py", "dormant_learning.py", "cluster_data.py"]},
            "packages": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "pandas", "catboost"]},
            "files": {path.name: sha256(path) for path in folder.iterdir()},
        })
        for path, digest in hashes.items():
            _track(path, digest, {})
        if output.exists() or output.is_symlink():
            raise ValueError("Выход появился во время расчёта")
        folder.replace(output)


def run_finalize(evaluation, adapter, graph_path, products_path, output, as_of, top_k=10):
    output, as_of = _validate_request(output, as_of, top_k)
    manifests, hashes = {}, {}
    prepared = _prepare(evaluation, adapter, graph_path, products_path, as_of, manifests, hashes)
    evaluation_metrics = prepared.evaluation_metrics
    baseline_metrics = prepared.baseline_metrics
    config = prepared.config
    _require_files(prepared.baseline_manifest, ["training_pairs.csv"])
    training = pd.read_csv(prepared.baseline / "training_pairs.csv", dtype={"sku": "string"}, keep_default_na=False)
    frames, audits = [training], []
    for split in ["validation", "test", "new_days"]:
        if split not in evaluation_metrics:
            continue
        _require_files(manifests[prepared.evaluation], [f"{split}_candidates.csv", f"{split}_positives.csv"])
        candidates = pd.read_csv(prepared.evaluation / f"{split}_candidates.csv", dtype={"sku": "string"}, keep_default_na=False)
        positives = pd.read_csv(prepared.evaluation / f"{split}_positives.csv", dtype={"sku": "string"}, keep_default_na=False)
        rows, audit = learning_rows(candidates, positives, group_prefix=f"final:{split}")
        frames.append(rows)
        audits.append({"task": split, **audit})
    logger.info("Полный refit: восстановление связей на %s", prepared.decision)
    candidates, recent = _candidates(prepared)
    _, hidden = mask_selling_edges(recent, config.seed + 20000)
    masked, _ = _candidates(prepared, hidden)
    rows, audit = learning_rows(masked, hidden, group_prefix="final:current_mask")
    frames.append(rows)
    audits.append({"task": "current_mask", **audit})
    training = pd.concat(frames, ignore_index=True).sort_values(["group_id", "normquery"])
    model = fit_ranker(training, baseline_metrics["selected_iterations"], config.seed)
    annotated, shortlist = _queue(candidates, model, prepared, as_of, top_k)
    summary = {
        **_summary(prepared, annotated, shortlist, as_of, top_k),
        "status": "full_refit_for_review_not_independent_evaluation", "training_performed": True,
        "training_rows": len(training), "training_groups": int(training["group_id"].nunique()),
        "training_skus": int(training["sku"].nunique()), "training_tasks_added": audits,
        "ranker_iterations": baseline_metrics["selected_iterations"],
    }
    _write_output(output, summary, annotated, shortlist, prepared,
                  [prepared.evaluation, prepared.adapter, prepared.graph_path], hashes, model, training)
    return summary


def recommend_from_release(release: Path, output: Path, as_of, top_k=10):
    """Восстановить немаскированную выдачу только из сохранённого полного refit."""
    output, as_of = _validate_request(output, as_of, top_k)
    release = _local_path(release)
    manifests, hashes = {}, {}
    manifest = _verify_tree(release, manifests, hashes)
    if not {"ranker.cbm", "metrics.json"}.issubset(manifest.get("files", {})):
        raise ValueError("Ожидается сохранённый полный релиз с ranker.cbm")
    parents = {}
    for name in manifest.get("parents", {}):
        path = _local_path(name)
        parent = manifests[path]
        if "artifact_sha256" in parent and "training_cutoff" in parent:
            role = "adapter"
        elif "checksums" in parent and "graph.pt" in parent["checksums"]:
            role = "graph"
        elif "metrics.json" in parent.get("files", {}) and "baseline" in json.loads((path / "metrics.json").read_text()):
            role = "evaluation"
        else:
            raise ValueError("Неизвестный родитель релиза")
        if role in parents:
            raise ValueError("Неоднозначные родители релиза")
        parents[role] = path
    if set(parents) != {"evaluation", "adapter", "graph"}:
        raise ValueError("Релиз должен ссылаться на evaluation и финальные нейросети")
    prepared = _prepare(parents["evaluation"], parents["adapter"], parents["graph"],
                        manifest["products"]["path"], as_of, manifests, hashes)
    if manifest["inputs"] != prepared.baseline_manifest["inputs"]:
        raise ValueError("Входы релиза отличаются от родителя baseline")
    metrics = json.loads((release / "metrics.json").read_text())
    if (metrics["source_end"] != prepared.decision.isoformat()
            or metrics["source_start"] != prepared.days.min().date().isoformat()
            or metrics["selected_method"] != prepared.evaluation_metrics["selected_method"]
            or metrics["fusion_weights"] != prepared.evaluation_metrics["fusion"]["weights"]):
        raise ValueError("Релиз не соответствует датам или выбранному методу родителей")
    model = CatBoostRanker()
    model.load_model(str(release / "ranker.cbm"))
    candidates, _ = _candidates(prepared)
    annotated, shortlist = _queue(candidates, model, prepared, as_of, top_k)
    summary = {
        **_summary(prepared, annotated, shortlist, as_of, top_k),
        "status": "saved_release_inference_no_training", "training_performed": False,
        "release": str(release), "ranker_iterations": metrics["ranker_iterations"],
    }
    _write_output(output, summary, annotated, shortlist, prepared, [release], hashes)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, help="Применить сохранённый релиз без обучения")
    for name in ["evaluation", "adapter", "graph", "products"]:
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--as-of", type=explicit_timestamp, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.release is not None:
        if any(value is not None for value in [args.evaluation, args.adapter, args.graph, args.products]):
            parser.error("--release нельзя сочетать с входами полного refit")
        result = recommend_from_release(args.release, args.output, args.as_of, args.top_k)
    else:
        if any(value is None for value in [args.evaluation, args.adapter, args.graph, args.products]):
            parser.error("Для refit нужны --evaluation, --adapter, --graph, --products")
        result = run_finalize(args.evaluation, args.adapter, args.graph, args.products, args.output, args.as_of, args.top_k)
    logger.info("%s; очередь %s", result["status"], result["review_pairs"])
    return result


if __name__ == "__main__":
    main()