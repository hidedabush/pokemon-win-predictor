"""Metrics used everywhere.        Owner: Ruel

Every model/baseline is scored with the same function so the results table is consistent.
"""

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def evaluate(y_true, p_win) -> dict:
    """y_true: 0/1 labels; p_win: predicted P(p1 wins)."""
    p = np.clip(np.asarray(p_win, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "accuracy": accuracy_score(y_true, p >= 0.5),
        "log_loss": log_loss(y_true, p, labels=[0, 1]),
        "brier": brier_score_loss(y_true, p),
    }
