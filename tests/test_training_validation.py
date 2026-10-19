import numpy as np
import pytest
import torch

from training_validation import TrainingValidation


@pytest.mark.parametrize("options", [
    {"validation_callback": 42},
    {"patience": 0}, {"patience": -1}, {"patience": True}, {"patience": 1.5},
    {"min_delta": -0.1}, {"min_delta": float("nan")}, {"min_delta": float("inf")},
    {"min_delta": True}, {"min_delta": "0.1"},
    {"selection_metric": ""}, {"selection_metric": "   "}, {"selection_metric": 42},
])
def test_invalid_validation_options_are_rejected(options):
    with pytest.raises(ValueError):
        TrainingValidation(**{"validation_callback": lambda model: 0.5, **options})


@pytest.mark.parametrize("options", [{"patience": 1}, {"min_delta": 0.1}, {"selection_metric": "Recall@5"}])
def test_selection_options_require_callback(options):
    with pytest.raises(ValueError, match="validation_callback"):
        TrainingValidation(**options)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf"), None,
                                  True, "0.5", [], np.array([0.5]), torch.tensor(0.5)])
def test_invalid_or_missing_validation_metric_is_never_silently_skipped(score):
    model = torch.nn.Linear(2, 2).train()
    validation = TrainingValidation(lambda current: score)
    with pytest.raises(ValueError, match="epoch 0.*finite float"):
        validation.evaluate(model, 0, lambda: model.weight.detach().clone())
    assert model.training and torch.is_grad_enabled()
    assert validation.scores == [] and validation.best_state is None


def test_callback_error_keeps_cause_and_restores_training_mode():
    model = torch.nn.Linear(2, 2).train()
    error = RuntimeError("no held-out targets")

    def validate(current):
        assert current is model and not current.training and not torch.is_grad_enabled()
        raise error

    validation = TrainingValidation(validate)
    with pytest.raises(ValueError, match="epoch 0.*no held-out targets") as caught:
        validation.evaluate(model, 0, lambda: model.weight.detach().clone())
    assert caught.value.__cause__ is error
    assert model.training and torch.is_grad_enabled()


def test_min_delta_resets_patience_but_never_discards_the_actual_maximum():
    scores = iter([0.0, 0.0625, 0.25, 0.3125, 0.375])
    validation = TrainingValidation(lambda model: next(scores), patience=2, min_delta=0.125)
    model = torch.nn.Linear(1, 1)
    stops = []
    for epoch in range(5):
        with torch.no_grad():
            model.weight.fill_(epoch)
        stops.append(validation.evaluate(model, epoch, lambda: model.weight.detach().clone()))
    assert stops == [False, False, False, False, True]
    assert validation.best_epoch == 4 and validation.best_score == 0.375
    torch.testing.assert_close(validation.best_state, torch.full_like(model.weight, 4))
    assert validation.audit(4, 10)["stopped_early"]
    assert not validation.audit(4, 4)["stopped_early"]


def test_patience_none_retains_earliest_tied_maximum_without_stopping():
    validation = TrainingValidation(lambda model: np.float32(-0.5))
    model = torch.nn.Linear(1, 1)
    initial = model.weight.detach().clone()
    for epoch in range(4):
        assert not validation.evaluate(model, epoch, lambda: model.weight.detach().clone())
        with torch.no_grad():
            model.weight.add_(1)
    with torch.no_grad():
        validation.restore(model.weight.copy_)
    torch.testing.assert_close(model.weight, initial, rtol=0, atol=0)
    assert validation.best_epoch == 0 and validation.best_score == -0.5
    assert validation.audit(3, 3)["selection_metric"] == "validation_score"


def test_disabled_validation_has_no_checkpoint_or_selection_claim():
    validation = TrainingValidation()

    def unexpected_call(*args):
        pytest.fail("Disabled validation must not snapshot or restore")

    assert not validation.evaluate(None, 0, unexpected_call)
    validation.restore(unexpected_call)
    assert validation.audit(3, 3) == {
        "validation_per_epoch": [], "best_epoch": None, "best_validation_score": None,
        "epochs_completed": 3, "stopped_early": False, "selection_metric": None,
        "patience": None, "min_delta": 0.0,
    }