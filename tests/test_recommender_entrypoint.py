import json
import subprocess
import sys
from pathlib import Path

import pytest
from sample_data import write_dataset

from recommender import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("command", [None, "train", "recommend", "run", "verify", "products", "export", "encode", "refine", "results"])
def test_help_is_available_without_data_or_connection(command, capsys):
    with pytest.raises(SystemExit) as error:
        main(([command] if command else []) + ["--help"])
    assert error.value.code == 0
    assert "--" in capsys.readouterr().out


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit) as error:
        main(["classify"])
    assert error.value.code == 2


def test_recommend_rejects_training_arguments(tmp_path, monkeypatch):
    import finalize_recommender

    def unexpected_training(*args, **kwargs):
        pytest.fail("Команда recommend не должна запускать обучение")

    monkeypatch.setattr(finalize_recommender, "run_finalize", unexpected_training)
    with pytest.raises(SystemExit) as error:
        main([
            "recommend", "--evaluation", str(tmp_path / "evaluation"),
            "--adapter", str(tmp_path / "adapter"), "--graph", str(tmp_path / "graph"),
            "--products", str(tmp_path / "products.csv"),
            "--output", str(tmp_path / "output"), "--as-of", "2026-10-11T01:28:00Z",
        ])
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def test_recommend_help_exposes_only_saved_inference(capsys):
    with pytest.raises(SystemExit) as error:
        main(["recommend", "--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--release" in help_text
    assert "--evaluation" not in help_text
    assert "--adapter" not in help_text


def test_active_modules_do_not_import_historical_training():
    result = subprocess.run(
        [sys.executable, "-c", ("import sys; sys.path.insert(0, sys.argv[1]); "
         "import full_history_recommender, seo_recommendation_policy, cluster_data; "
         "assert not {'train_recommender', 'model_pipeline', 'mlflow'} & set(sys.modules)"),
         str(ROOT / "src")],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_entrypoint_runs_real_small_training(tmp_path):
    facts, dictionary = write_dataset(tmp_path)
    output = tmp_path / "trained"
    main([
        "train", "--facts", str(facts), "--dictionary", str(dictionary),
        "--output", str(output), "--dictionary-as-of", "2025-12-31",
        "--first-decision", "2026-01-02", "--validation-decision", "2026-01-04",
        "--test-decision", "2026-01-06", "--window-days", "2",
        "--per-source", "5", "--iterations", "3", "6",
    ])
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["training"]["rows"] > 0
    assert metrics["test"]["catboost_pu"]["positive_pairs"] == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["python"]
    assert manifest["packages"]["catboost"]
    assert "cluster_data.py" in manifest["code"]
    with pytest.raises(ValueError, match="перезапись"):
        main([
            "train", "--facts", str(facts), "--dictionary", str(dictionary),
            "--output", str(output), "--dictionary-as-of", "2025-12-31",
            "--first-decision", "2026-01-02", "--validation-decision", "2026-01-04",
            "--test-decision", "2026-01-06", "--window-days", "2",
        ])