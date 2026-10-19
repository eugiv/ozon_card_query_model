"""Общий выбор checkpoint по максимальной отложенной метрике, без train-loss fallback."""

from __future__ import annotations

import math
from numbers import Real


def nonnegative_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name}: ожидается конечное число >= 0")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}: ожидается конечное число >= 0")
    return value


class TrainingValidation:
    """Epoch 0 участвует в выборе; ties оставляют ранний checkpoint.

    Каждый строгий максимум сохраняется независимо от min_delta. Patience
    сбрасывается только при росте > min_delta от последнего значимого роста;
    patience=None отключает остановку, но не выбор лучших весов.
    Callback отвечает за отложенные цели/отсутствие leakage и не меняет модель.
    """

    def __init__(self, validation_callback=None, patience=None, min_delta=0.0, selection_metric=None):
        if validation_callback is not None and not callable(validation_callback):
            raise ValueError("validation_callback должен быть callable или None")
        if patience is not None and (isinstance(patience, bool) or not isinstance(patience, int) or patience < 1):
            raise ValueError("patience должен быть целым числом >= 1 или None")
        min_delta = nonnegative_finite(min_delta, "min_delta")
        if selection_metric is not None and (not isinstance(selection_metric, str) or not selection_metric.strip()):
            raise ValueError("selection_metric должен быть непустой строкой или None")
        if validation_callback is None and (patience is not None or min_delta != 0 or selection_metric is not None):
            raise ValueError("patience/min_delta/selection_metric требуют validation_callback")
        self.callback = validation_callback
        self.patience = patience
        self.min_delta = min_delta
        self.selection_metric = (selection_metric or "validation_score") if self.enabled else None
        self.scores = []
        self.best_epoch = None
        self.best_score = None
        self.best_state = None
        self._patience_score = None
        self._bad_epochs = 0
        self._stop = False

    @property
    def enabled(self):
        return self.callback is not None

    def evaluate(self, model, epoch, snapshot):
        """Оценить текущую модель в no_grad/eval; snapshot обязан возвращать копию."""
        if not self.enabled:
            return False
        import torch

        was_training = model.training if isinstance(model, torch.nn.Module) else None
        try:
            if was_training is not None:
                model.eval()
            with torch.no_grad():
                score = self.callback(model)
            if isinstance(score, bool) or not isinstance(score, Real):
                raise ValueError("validation metric должна быть finite float (есть ли отложенные цели?)")
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("validation metric должна быть finite float (есть ли отложенные цели?)")
        except Exception as error:
            raise ValueError(f"validation_callback failed at epoch {epoch}: {error}") from error
        finally:
            if was_training is not None:
                model.train(was_training)
        self.scores.append(score)
        if self.best_score is None or score > self.best_score:
            self.best_state = snapshot()
            self.best_score = score
            self.best_epoch = epoch
        if self._patience_score is None or score > self._patience_score + self.min_delta:
            self._patience_score = score
            self._bad_epochs = 0
        else:
            self._bad_epochs += 1
        self._stop = self.patience is not None and self._bad_epochs >= self.patience
        return self._stop

    def restore(self, restore_state):
        if self.best_state is not None:
            restore_state(self.best_state)

    def audit(self, epochs_completed, epochs_requested):
        return {
            "validation_per_epoch": list(self.scores),
            "best_epoch": self.best_epoch,
            "best_validation_score": self.best_score,
            "epochs_completed": epochs_completed,
            "stopped_early": self._stop and epochs_completed < epochs_requested,
            "selection_metric": self.selection_metric,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }