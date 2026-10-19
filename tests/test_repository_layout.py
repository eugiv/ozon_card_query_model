import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOVED_MODULES = {
    "train", "train_recommender", "model_pipeline", "artifacts", "compose",
    "evaluation", "features", "sparse_rules", "prepare_recommender_pilot",
    "publish_recommender_report", "expert_review", "expert_metrics", "expert_review_ui",
    "create_task_definition_presentation", "append_analysis_presentation",
    "append_methodology_presentation", "dormant_ablation",
    "export_cluster_data", "product_media", "recommender_io", "encode_clusters",
    "rank_fusion", "recommendation_metrics", "neural_refinement", "current_cluster_recommendations",
    "adapt_defense_presentation", "create_explanatory_note",
}


def test_remaining_python_files_have_no_imports_of_removed_modules():
    for folder in (ROOT / "src", ROOT / "tools", ROOT / "tests"):
        for path in folder.iterdir():
            if path.suffix != ".py":
                continue
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                if isinstance(node, ast.ImportFrom):
                    assert (node.module or "").split(".")[0] not in REMOVED_MODULES, path
                elif isinstance(node, ast.Import):
                    assert all(alias.name.split(".")[0] not in REMOVED_MODULES
                               for alias in node.names), path
    assert not any((ROOT / "src" / f"{name}.py").exists() for name in REMOVED_MODULES)


def test_public_results_are_small_aggregate_metrics_not_private_rows():
    path = ROOT / "reports" / "experiment_summary.json"
    assert path.stat().st_size < 30_000
    snapshot = json.loads(path.read_text())
    assert snapshot["evaluation_status"] == "reused_diagnostic_period"
    assert snapshot["selected_method"] == "learned_fusion"
    assert snapshot["training"]["rows"] == 37245
    assert snapshot["final_training"]["rows"] == 46787
    assert snapshot["source"]["fact_rows"] == 3203109
    assert snapshot["current_queue"]["recommended_review_pairs"] == 3530
    stage4 = snapshot["stage4"]
    assert stage4["scope"] == "fixed_candidate_pool_reranking"
    assert stage4["seed"] == 7 and stage4["repetitions"] == 3
    assert stage4["test"]["scenarios"]["clean"]["recall@10"] == snapshot["test"]["models"]["learned_fusion"]["recall@10"]
    assert stage4["test"]["horizon_days"] == 14
    assert stage4["new_days"]["horizon_days"] == 4
    for part in ("test", "zero_views_test", "new_days"):
        results = snapshot[part]
        assert results["positive_pairs"] >= results["evaluated_skus"] > 0
        assert results["candidate_pairs"] > 0
        assert 0 <= results["retrieval_recall"] <= 1
        assert len(results["models"]) == 8
        for metrics in results["models"].values():
            assert all(0 <= value <= 1 for value in metrics.values())
    assert snapshot["test"]["positive_pairs"] != snapshot["zero_views_test"]["positive_pairs"]
    assert all(not Path(path).is_absolute() for path in snapshot["provenance"])

    def check_keys(value):
        if isinstance(value, dict):
            assert not {"sku", "password", "attributes_json", "main_image_url", "unmapped_queries"} & value.keys()
            for item in value.values():
                check_keys(item)
        elif isinstance(value, list):
            for item in value:
                check_keys(item)

    check_keys(snapshot)


def test_minimal_source_layout_has_only_nine_ml_modules_and_two_document_tools():
    """Девять основных ML-модулей дополнены только явными помощниками ноутбука."""
    expected = {
        "recommender.py", "cluster_data.py", "cluster_recommender.py",
        "dormant_learning.py", "semantic_adapter.py", "graph_learning.py",
        "full_history_recommender.py", "finalize_recommender.py", "seo_recommendation_policy.py",
        "ranking_snapshot.py", "notebook_research.py", "training_validation.py",
    }
    assert {path.name for path in (ROOT / "src").iterdir() if path.suffix == ".py"} == expected
    assert {path.name for path in (ROOT / "tools").iterdir() if path.suffix == ".py"} == {
        "defense.py", "export_notebook_evaluation.py", "readable_docs.py",
    }
    assert {path.name for path in ROOT.iterdir() if path.name.startswith("requirements")} == {"requirements.txt"}
    assert {path.name for path in (ROOT / "docs").iterdir() if path.suffix == ".md"} == {
        "recommender_methodology.md", "explanatory_note.md", "answers_to_feedback.md", "project_manual.md",
    }
