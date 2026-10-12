import json
from datetime import date
from itertools import pairwise

import pandas as pd
import pytest
from sample_data import write_dataset

import full_history_recommender as experiment
from cluster_data import sha256
from full_history_recommender import (
    DormantConfig,
    dormant_positives,
    missing_window_days,
    run_full_history,
)


def config(**kwargs):
    return DormantConfig(date(2026, 6, 26), date(2026, 6, 26), date(2026, 9, 4),
                         date(2026, 9, 18), **kwargs)


def test_full_history_schedule_has_nonoverlapping_targets():
    decisions = config().training_decisions()
    assert decisions == [date(2026, 6, 26), date(2026, 7, 10), date(2026, 7, 24),
                         date(2026, 8, 7), date(2026, 8, 21)]
    assert decisions[-1] < config().validation_decision < config().test_decision


def test_weekly_training_never_crosses_validation_target():
    settings = config(training_stride_days=7)
    decisions = settings.training_decisions()
    assert len(decisions) == 9
    assert all((right - left).days == 7 for left, right in pairwise(decisions))
    assert decisions[-1] == date(2026, 8, 21)
    assert all(day + pd.Timedelta(days=14) <= settings.validation_decision for day in decisions)


@pytest.mark.parametrize("stride", [0, 15, True, 1.5])
def test_rejects_invalid_training_stride(stride):
    with pytest.raises(ValueError, match="Шаг обучения"):
        config(training_stride_days=stride)


@pytest.mark.parametrize("kwargs", [{"max_views": True}, {"max_views": -1},
                                     {"window_days": 0}, {"iterations": ()}])
def test_config_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_missing_targets_never_disappear_from_dormant_denominator():
    positives = pd.DataFrame([["1", "absent"], ["1", "five"], ["1", "six"], ["2", "new"]],
                             columns=["sku", "normquery"])
    recent = pd.DataFrame([["1", "five", 5], ["1", "six", 6]], columns=["sku", "normquery", "views"])
    result = dormant_positives(positives, recent, 5, {"1"})
    assert list(result["normquery"]) == ["absent", "five"]
    assert list(dormant_positives(positives, recent, 0, {"1"})["normquery"]) == ["absent"]


def test_target_gap_and_old_context_gap_have_different_roles():
    days = pd.date_range("2026-01-09", "2026-10-03").difference(pd.to_datetime(["2026-05-19", "2026-07-21"]))
    assert missing_window_days(days, date(2026, 7, 10), 14) == ["2026-07-21"]
    assert missing_window_days(days, date(2026, 7, 24), 14) == ["2026-07-21"]
    assert missing_window_days(days, date(2026, 9, 18), 14) == []


def small_config():
    return DormantConfig(date(2025, 12, 31), date(2026, 1, 2), date(2026, 1, 4),
                         date(2026, 1, 6), window_days=2, per_source=5, iterations=(3, 6))


def test_real_training_and_future_isolation(tmp_path):
    facts, dictionary = write_dataset(tmp_path)
    before = run_full_history(facts, dictionary, None, tmp_path / "before", small_config())
    assert before["training"]["rows"] > 0
    assert before["test"]["catboost_pu"]["positive_pairs"] == 1
    assert before["source"]["fact_rows"] == len(pd.read_csv(facts))
    manifest = json.loads((tmp_path / "before" / "manifest.json").read_text())
    assert set(manifest["code"]) == {
        "cluster_data.py", "cluster_recommender.py", "dormant_learning.py",
        "full_history_recommender.py", "graph_learning.py",
    }
    for name, digest in manifest["files"].items():
        assert sha256(tmp_path / "before" / name) == digest
    frame = pd.read_csv(facts, dtype={"sku": str})
    frame.loc[frame["date"].ge("2026-01-07"), "order_count"] *= 10
    frame.to_csv(facts, index=False)
    after = run_full_history(facts, dictionary, None, tmp_path / "after", small_config())
    assert before["selected_method"] == after["selected_method"]
    assert before["validation"] == after["validation"]
    assert before["selected_iterations"] == after["selected_iterations"]
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "before" / "test_candidates.csv"),
                                  pd.read_csv(tmp_path / "after" / "test_candidates.csv"))


def test_incomplete_evaluation_and_existing_output_fail(tmp_path):
    facts, dictionary = write_dataset(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="перезапись"):
        run_full_history(facts, dictionary, None, output, small_config())
    frame = pd.read_csv(facts)
    frame.loc[frame["date"].ne("2026-01-08")].to_csv(facts, index=False)
    with pytest.raises(ValueError, match="Неполное окно test"):
        run_full_history(facts, dictionary, None, tmp_path / "rejected", small_config())
    assert not (tmp_path / "rejected").exists()


def interrupt_after_first_search(monkeypatch):
    original = experiment.generate_dormant
    calls = []

    def interrupted(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 2:
            raise RuntimeError("intentional interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(experiment, "generate_dormant", interrupted)
    return original


def test_interrupted_run_resumes_completed_search_with_hash_checks(tmp_path, monkeypatch):
    facts, dictionary = write_dataset(tmp_path)
    output = tmp_path / "resumed"
    original = interrupt_after_first_search(monkeypatch)
    with pytest.raises(RuntimeError, match="intentional interruption"):
        run_full_history(facts, dictionary, None, output, small_config())
    assert not output.exists()
    generated = []

    def track(*args, **kwargs):
        generated.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(experiment, "generate_dormant", track)
    result = run_full_history(facts, dictionary, None, output, small_config())
    assert result["training"]["rows"] > 0
    assert len(generated) == 4
    assert generated[0][1]["hidden"] is not None


def test_tampered_checkpoint_is_not_silently_reused(tmp_path, monkeypatch):
    facts, dictionary = write_dataset(tmp_path)
    output = tmp_path / "interrupted"
    original = interrupt_after_first_search(monkeypatch)
    with pytest.raises(RuntimeError, match="intentional interruption"):
        run_full_history(facts, dictionary, None, output, small_config())
    checkpoint = next((tmp_path / "interrupted_checkpoints").iterdir())
    with (checkpoint / "candidates.csv").open("a") as stream:
        stream.write("tampered\n")
    monkeypatch.setattr(experiment, "generate_dormant", original)
    with pytest.raises(ValueError, match="Контрольная сумма"):
        run_full_history(facts, dictionary, None, output, small_config())