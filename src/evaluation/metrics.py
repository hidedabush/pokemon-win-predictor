"""Metrics used everywhere.

Every model/baseline is scored with the same function so the results table is consistent.
"""

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


EPS = 1e-6
METRICS = ("accuracy", "log_loss", "brier")


def evaluate(y_true, p_win) -> dict:
    """y_true: 0/1 labels; p_win: predicted P(p1 wins)."""
    p = np.clip(np.asarray(p_win, dtype=float), EPS, 1 - EPS)
    return {
        "accuracy": accuracy_score(y_true, p >= 0.5),
        "log_loss": log_loss(y_true, p, labels=[0, 1]),
        "brier": brier_score_loss(y_true, p),
    }


def per_battle(y_true, p_win) -> dict:
    """Each metric for every battle separately; their means are exactly evaluate()'s numbers."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_win, dtype=float), EPS, 1 - EPS)
    return {
        "accuracy": ((p >= 0.5) == (y == 1)).astype(float),
        "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)),
        "brier": (p - y) ** 2,
    }


def bootstrap_ci(y_true, predictions: dict, reference=None, n_boot=1000, level=0.95, seed=0,
                 chunk=100) -> dict:
    """Bootstrap confidence intervals, resampling battles with replacement.

    predictions: {name: p_win} for models scored on the SAME battles (same y_true).
    Every model is scored on the same resamples, so the differences against `reference`
    are paired: they cancel out the luck of which battles happened to be in the test set,
    which makes them much tighter than comparing two separate intervals.

    Returns {name: {metric: (lo, hi), "diff_<metric>": (estimate, lo, hi)}}, where diff is
    model minus reference (for log loss and Brier, negative = better than the reference).
    """
    names = list(predictions)
    per = {m: np.column_stack([per_battle(y_true, predictions[k])[m] for k in names])
           for m in METRICS}  # metric -> (n battles, n models)
    n = len(next(iter(per.values())))
    rng = np.random.default_rng(seed)
    boot = {m: [] for m in METRICS}  # metric -> (n_boot, n models) resampled means
    for start in range(0, n_boot, chunk):
        counts = rng.multinomial(n, np.full(n, 1 / n), size=min(chunk, n_boot - start))
        for m in METRICS:
            boot[m].append(counts @ per[m] / n)
    boot = {m: np.vstack(v) for m, v in boot.items()}

    q = [(1 - level) / 2 * 100, (1 + level) / 2 * 100]
    out = {}
    for j, name in enumerate(names):
        out[name] = {m: tuple(np.percentile(boot[m][:, j], q)) for m in METRICS}
        if reference is not None and name != reference:
            r = names.index(reference)
            for m in METRICS:
                diff = per[m][:, j].mean() - per[m][:, r].mean()
                out[name][f"diff_{m}"] = (diff, *np.percentile(boot[m][:, j] - boot[m][:, r], q))
    return out
