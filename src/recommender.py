"""Рекомендатель: полный запуск, проверка артефактов и выдача готовой модели."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_data import sha256

COMMANDS = {
    "train": ("full_history_recommender", []),
    "recommend": ("finalize_recommender", []),
    "refine": ("full_history_recommender", ["refine"]),
    "results": ("full_history_recommender", ["results"]),
    "adapt": ("semantic_adapter", []),
    "graph": ("graph_learning", []),
    "finalize": ("finalize_recommender", []),
    "export": ("cluster_data", []),
    "encode": ("cluster_recommender", []),
    "products": ("cluster_data", ["products"]),
}
STAGES = ("baseline", "adapter", "graph", "evaluation", "ablation", "final_adapter", "final_graph", "release")
ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def load_config(path):
    from cluster_data import explicit_timestamp
    from full_history_recommender import DormantConfig

    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {"inputs", "outputs", "method", "neural", "as_of", "top_k"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("Конфигурация должна содержать inputs, outputs, method, neural, as_of, top_k")
    if set(raw["inputs"]) != {"facts", "dictionary", "embeddings", "products"} or set(raw["outputs"]) != set(STAGES):
        raise ValueError("Нужны все входы и восемь выходных этапов")

    def paths(values):
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ValueError("Пути должны быть непустыми строками")
        return {key: (path.parent / value).resolve() for key, value in values.items()}

    inputs, outputs = paths(raw["inputs"]), paths(raw["outputs"])
    destinations = list(outputs.values())
    if (len(set(destinations)) != len(destinations) or set(destinations) & set(inputs.values())
            or any(a in b.parents or b in a.parents for i, a in enumerate(destinations) for b in destinations[i + 1:])
            or any(output in source.parents for output in destinations for source in inputs.values())):
        raise ValueError("Входные и выходные пути не должны совпадать или пересекаться")
    method = dict(raw["method"])
    for key in ("dictionary_as_of", "first_decision", "validation_decision", "test_decision"):
        method[key] = date.fromisoformat(method[key])
    if "iterations" in method:
        method["iterations"] = tuple(method["iterations"])
    config = DormantConfig(**method)
    neural = {"adapter_epochs": 15, "adapter_batch_size": 128, "device": "cpu",
              "graph_epochs": 30, "graph_dimensions": 32, "graph_layers": 2, **raw["neural"]}
    if set(neural) != {"adapter_epochs", "adapter_batch_size", "device", "graph_epochs", "graph_dimensions", "graph_layers"}:
        raise ValueError("Неизвестный параметр neural")
    for name, value in neural.items():
        if name == "device":
            if value not in {"cpu", "mps", "cuda"}:
                raise ValueError("device должен быть cpu, mps или cuda")
        elif type(value) is not int or value < (0 if name == "graph_layers" else 1):
            raise ValueError(f"Некорректный параметр {name}")
    if neural["adapter_batch_size"] < 2 or type(raw["top_k"]) is not int or raw["top_k"] < 1:
        raise ValueError("Некорректные batch_size или top_k")
    return {"inputs": inputs, "outputs": outputs, "method": config, "neural": neural,
            "as_of": explicit_timestamp(raw["as_of"]), "top_k": raw["top_k"]}


def _manifest(folder):
    """Проверяем также graph checksums, а не только files baseline."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    hashes = manifest.get("files", manifest.get("checksums", manifest.get("artifact_sha256")))
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Манифест не содержит контрольных сумм")
    for name, expected in hashes.items():
        if Path(name).name != name or sha256(folder / name) != expected:
            raise ValueError(f"Контрольная сумма артефакта не совпадает: {folder.name}/{name}")
    return manifest


def _read(folder, name="metrics.json"):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def _preflight(config):
    inputs, method = config["inputs"], config["method"]
    for path in inputs.values():
        if not path.is_file():
            raise ValueError(f"Нет входного файла {path.name}; сначала подготовьте данные")
    dates = pd.to_datetime(pd.read_csv(inputs["facts"], usecols=["date"])["date"], errors="raise")
    if dates.empty or dates.isna().any():
        raise ValueError("Пустые или неизвестные даты фактов")
    last_day = dates.max().date()
    if last_day < method.test_decision + pd.Timedelta(days=method.window_days):
        raise ValueError("Нет полного будущего окна test")
    if pd.Timestamp(last_day, tz="UTC") + pd.Timedelta(days=1) > config["as_of"]:
        raise ValueError("as_of должен быть после завершения последнего дня фактов")
    snapshots = pd.to_datetime(pd.read_csv(inputs["products"], usecols=["snapshot_at"])["snapshot_at"], utc=True)
    if snapshots.isna().any() or snapshots.gt(config["as_of"]).any():
        raise ValueError("Снимок карточки из будущего относительно as_of")
    return last_day


def _check_stage(name, config, last_day):
    folder, inputs = config["outputs"][name], config["inputs"]
    manifest = _manifest(folder)
    metrics = _read(folder)
    if name == "baseline":
        expected = {key: value.isoformat() if isinstance(value, date) else value
                    for key, value in asdict(config["method"]).items()}
        expected["iterations"] = list(expected["iterations"])
        if metrics["config"] != expected:
            raise ValueError("Параметры существующего baseline отличаются; выберите новые выходные папки")
    recorded = manifest.get("inputs", {})
    for key, item in recorded.items():
        if key not in inputs or (ROOT / item["path"]).resolve() != inputs[key] or sha256(inputs[key]) != item["sha256"]:
            raise ValueError("Входы существующего этапа не соответствуют конфигурации")
    if name in {"adapter", "final_adapter", "graph", "final_graph"}:
        cutoff = config["method"].validation_decision if name in {"adapter", "graph"} else last_day
        recorded_cutoff = manifest.get("training_cutoff", manifest.get("train_end"))
        if pd.Timestamp(recorded_cutoff).date() != cutoff:
            raise ValueError("Дата обучения существующей нейросети отличается")
        if name.endswith("adapter"):
            for key in ("facts", "dictionary", "embeddings"):
                if manifest["input_sha256"].get(str(inputs[key])) != sha256(inputs[key]):
                    # Исторические манифесты использовали относительные пути.
                    matching = [value for path, value in manifest["input_sha256"].items()
                                if (ROOT / path).resolve() == inputs[key]]
                    if matching != [sha256(inputs[key])]:
                        raise ValueError("Адаптер использует другие входы")
            if metrics["epochs"] != config["neural"]["adapter_epochs"]:
                raise ValueError("Параметр adapter_epochs отличается")
            if metrics["batch_size"] != config["neural"]["adapter_batch_size"] or metrics["seed"] != config["method"].seed:
                raise ValueError("Параметры обучения адаптера отличаются")
        else:
            for key, setting in (("epochs_completed", "graph_epochs"), ("dimensions", "graph_dimensions"), ("layers", "graph_layers")):
                if metrics[key] != config["neural"][setting]:
                    raise ValueError(f"Параметр {setting} отличается")
        if pd.Timestamp(manifest["dictionary_as_of"]).date() != config["method"].dictionary_as_of:
            raise ValueError("Нейросеть использует другую дату словаря")
    parents = {
        "evaluation": ("baseline", "adapter", "graph"),
        "release": ("evaluation", "final_adapter", "final_graph"),
    }.get(name)
    if parents is not None:
        expected = {config["outputs"][key]: sha256(config["outputs"][key] / "manifest.json") for key in parents}
        actual = {(ROOT / path).resolve(): digest for path, digest in manifest["parents"].items()}
        if actual != expected:
            raise ValueError("Родительские модели не соответствуют конфигурации")
    if name == "ablation" and manifest["parent_sha256"] != sha256(config["outputs"]["evaluation"] / "manifest.json"):
        raise ValueError("Сравнение относится к другому эксперименту")
    if name == "release":
        item = manifest["products"]
        if (ROOT / item["path"]).resolve() != inputs["products"] or item["sha256"] != sha256(inputs["products"]):
            raise ValueError("Изменён снимок карточек выпуска")
        if pd.Timestamp(metrics["as_of"]) != config["as_of"]:
            raise ValueError("Параметр as_of отличается")
    return metrics


def verify_pipeline(config):
    """Проверяем реальные файлы, заново считаем метрики, не обучаем на test."""
    from full_history_recommender import evaluate_scores

    last_day = _preflight(config)
    outputs = config["outputs"]
    stages = {name: _check_stage(name, config, last_day) for name in STAGES}
    evaluation = stages["evaluation"]
    if (pd.Timestamp(evaluation["training_cutoff"]).date() != config["method"].validation_decision
            or evaluation["test"]["decision"] != config["method"].test_decision.isoformat()
            or evaluation["validation"]["decision"] != config["method"].validation_decision.isoformat()
            or evaluation["test"]["horizon_days"] != config["method"].window_days):
        raise ValueError("Проверка использует другие временные границы")
    for split in ("validation", "test", "new_days"):
        if split not in evaluation:
            continue
        candidates = pd.read_csv(outputs["evaluation"] / f"{split}_candidates.csv", dtype={"sku": "string"},
                                 keep_default_na=False, float_precision="round_trip")
        positives = pd.read_csv(outputs["evaluation"] / f"{split}_positives.csv", dtype={"sku": "string"},
                                keep_default_na=False)
        scores = {name: candidates[f"score_{name}"].to_numpy() for name in evaluation[split]["models"]}
        recalculated = evaluate_scores(candidates, positives, scores)
        for name, values in recalculated.items():
            for key, value in values.items():
                expected = evaluation[split]["models"][name][key]
                if (value is None) != (expected is None) or (value is not None and not np.isclose(value, expected, rtol=1e-12, atol=1e-12)):
                    raise ValueError("Пересчитанные метрики не совпадают с отчётом")
        if candidates.log_views.gt(np.log1p(config["method"].max_views) + 1e-8).any():
            raise ValueError("Проверочная выдача содержит высокотрафиковую пару")
    selected = max(evaluation["validation"]["models"], key=lambda name: evaluation["validation"]["models"][name]["ndcg@10"])
    if selected != evaluation["selected_method"]:
        raise ValueError("Метод должен выбираться только по validation")
    release = stages["release"]
    queue = pd.read_csv(outputs["release"] / "recommendations_for_review.csv", dtype={"sku": "string"},
                       keep_default_na=False, float_precision="round_trip")
    if (queue.duplicated(["sku", "normquery"]).any() or not queue.decision.eq("review").all()
            or len(queue) != release["review_pairs"] or queue.sku.nunique() != release["review_skus"]
            or queue["rank"].gt(config["top_k"]).any()
            or queue.log_views.gt(np.log1p(config["method"].max_views) + 1e-8).any()
            or not np.isfinite(queue.score).all()):
        raise ValueError("Выдача нарушает правила рекомендателя")
    return {"status": "verified", "source_rows": release["source_rows"], "source_skus": release["source_skus"],
            "training_rows": release["training_rows"], "review_pairs": len(queue), "review_skus": int(queue.sku.nunique()),
            "selected_method": selected, "validation_before_test": True,
            "test": evaluation["test"]["models"][selected],
            "evaluation_status": "diagnostic_not_expert_relevance_or_seo_uplift",
            "stages_checked": len(STAGES), "manual_labels_required": False, "photos_used": False}


def run_pipeline(config):
    """Возобновляем только совпадающие артефакты; новые результаты неизменяемы."""
    from finalize_recommender import run_finalize
    from full_history_recommender import (
        run_full_history,
        run_fusion_ablation,
        run_refinement,
    )
    from graph_learning import train_graph_from_files
    from semantic_adapter import main as adapt

    last_day = _preflight(config)
    inputs, outputs, method, neural = (config[key] for key in ("inputs", "outputs", "method", "neural"))
    # Проверяем всё уже существующее до первого изменения.
    for name in STAGES:
        if outputs[name].exists():
            _check_stage(name, config, last_day)
    for name in STAGES:
        if outputs[name].exists():
            logger.info("%s: проверенный результат уже существует", name)
            continue
        logger.info("%s: запуск", name)
        if name == "baseline":
            run_full_history(inputs["facts"], inputs["dictionary"], inputs["embeddings"], outputs[name], method)
        elif name in {"adapter", "final_adapter"}:
            cutoff = method.validation_decision if name == "adapter" else last_day
            adapt(["--facts", str(inputs["facts"]), "--dictionary", str(inputs["dictionary"]),
                   "--embeddings", str(inputs["embeddings"]), "--dictionary-as-of", method.dictionary_as_of.isoformat(),
                   "--train-end", cutoff.isoformat(), "--epochs", str(neural["adapter_epochs"]),
                   "--batch-size", str(neural["adapter_batch_size"]), "--seed", str(method.seed),
                   "--device", neural["device"], "--output", str(outputs[name])])
        elif name in {"graph", "final_graph"}:
            train_graph_from_files(facts=inputs["facts"], dictionary=inputs["dictionary"],
                                   dictionary_as_of=method.dictionary_as_of.isoformat(),
                                   train_end=(method.validation_decision if name == "graph" else last_day).isoformat(),
                                   output=outputs[name], epochs=neural["graph_epochs"],
                                   dimensions=neural["graph_dimensions"], layers=neural["graph_layers"], seed=method.seed)
        elif name == "evaluation":
            run_refinement(outputs["baseline"], outputs["adapter"], outputs["graph"], outputs[name])
        elif name == "ablation":
            run_fusion_ablation(outputs["evaluation"], outputs[name])
        else:
            run_finalize(outputs["evaluation"], outputs["final_adapter"], outputs["final_graph"],
                         inputs["products"], outputs[name], config["as_of"], config["top_k"])
    return verify_pipeline(config)


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify", *COMMANDS], help="run — весь процесс; verify — проверка; recommend — готовая модель")
    if args and args[0] in {"run", "verify"}:
        parser.add_argument("--config", type=Path, required=True)
        parsed = parser.parse_args(args)
        logging.basicConfig(level=logging.INFO)
        config = load_config(parsed.config)
        result = run_pipeline(config) if parsed.command == "run" else verify_pipeline(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if not args or args[0] not in COMMANDS:
        parser.parse_args(args)
        return
    if args[0] == "recommend":
        from cluster_data import explicit_timestamp
        from finalize_recommender import recommend_from_release

        inference = argparse.ArgumentParser(description="Выдача сохранённого выпуска без обучения")
        inference.add_argument("--release", type=Path, required=True)
        inference.add_argument("--output", type=Path, required=True)
        inference.add_argument("--as-of", type=explicit_timestamp, required=True)
        inference.add_argument("--top-k", type=int, default=10)
        parsed = inference.parse_args(args[1:])
        logging.basicConfig(level=logging.INFO)
        result = recommend_from_release(parsed.release, parsed.output, parsed.as_of, parsed.top_k)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    module = importlib.import_module(COMMANDS[args[0]][0])
    module.main(COMMANDS[args[0]][1] + args[1:])


if __name__ == "__main__":
    main()