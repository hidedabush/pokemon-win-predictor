"""Report figures. Each one is saved to figures/ as PNG (300 dpi) and PDF.

    1. calibration_plot      predicted vs. actual win rate in 10 bins, one line per model
    2. feature_importance    top features (logistic weights or tree importance)
    3. hyperparam_curve      val log loss vs. C (logistic) or max_depth (trees)
    4. ablation_bar          how much worse the model gets when each feature group is removed

Same style as the Data figures (src/data/figures.py): large fonts, axis labels, no titles.
"""

from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve

from src.data.figures import INK_2, WIDTH, _save, plt  # also applies the shared style

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#d64f8a", "#52514e"]


def _save_to(fig, path):
    path = Path(path)
    _save(fig, path.stem, path.parent)  # writes <stem>.png and <stem>.pdf


def calibration_plot(results: dict, path: str, n_bins: int = 10):
    """results: {model_name: (y_true, p_win)}.

    Bins hold equal numbers of battles (quantiles), because almost every prediction is close
    to 50% and equal-width bins would be nearly empty at the ends.
    """
    fig, ax = plt.subplots(figsize=(WIDTH / 2, WIDTH / 2.2))
    lo, hi = 0.5, 0.5
    for (name, (y, p)), color in zip(results.items(), COLORS):
        actual, predicted = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(predicted, actual, "o-", markersize=3, linewidth=1, color=color, label=name)
        lo = min(lo, predicted.min(), actual.min())
        hi = max(hi, predicted.max(), actual.max())
    lo, hi = lo - 0.02, hi + 0.02
    ax.plot([lo, hi], [lo, hi], linestyle=(0, (3, 2)), linewidth=0.8, color=INK_2,
            label="perfectly calibrated")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="predicted P(player 1 wins)",
           ylabel="actual player 1 win rate")
    ax.grid(True)
    ax.legend(frameon=False)
    _save_to(fig, path)


def feature_importance(names, values, path: str, k: int = 15, xlabel: str = "importance"):
    """Horizontal bars for the k features with the largest |value|. Negative values (e.g.
    logistic weights that favour player 2) are drawn in a second color."""
    names, values = np.asarray(names), np.asarray(values, dtype=float)
    top = np.argsort(np.abs(values))[::-1][:k][::-1]  # largest at the top
    fig, ax = plt.subplots(figsize=(WIDTH / 2, 0.2 * len(top) + 0.6))
    ax.barh(np.arange(len(top)), values[top], height=0.7,
            color=[COLORS[0] if v >= 0 else COLORS[1] for v in values[top]])
    ax.set_yticks(np.arange(len(top)), names[top])
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True)
    _save_to(fig, path)


def hyperparam_curve(xs, val_losses, xlabel: str, path: str):
    """Val log loss for each grid value. val_losses: one list, or {line name: list} for
    several lines over the same xs. The best value on each line is circled.

    The x values are evenly spaced (good for a log-spaced C grid and for max_depth with None).
    """
    if not isinstance(val_losses, dict):
        val_losses = {None: val_losses}
    pos = np.arange(len(xs))
    fig, ax = plt.subplots(figsize=(WIDTH / 2, WIDTH / 2.8))
    for (name, losses), color in zip(val_losses.items(), COLORS):
        ax.plot(pos, losses, "o-", markersize=3, linewidth=1, color=color, label=name)
        best = int(np.argmin(losses))
        ax.plot(pos[best], losses[best], "o", markersize=8, markerfacecolor="none", color=color)
    ax.set_xticks(pos, ["none" if x is None or x != x else f"{x:g}" for x in xs])
    ax.set(xlabel=xlabel, ylabel="val log loss")
    ax.grid(True)
    if len(val_losses) > 1:
        ax.legend(frameon=False)
    _save_to(fig, path)


def ablation_bar(drops: dict, path: str):
    """drops: {metric label: {group: how much worse that metric got without the group}}.
    One panel per metric, groups in the same order on each."""
    fig, axes = plt.subplots(1, len(drops), figsize=(WIDTH, 2.0), sharey=True, squeeze=False)
    for ax, (label, by_group) in zip(axes[0], drops.items()):
        y = np.arange(len(by_group))[::-1]
        ax.barh(y, list(by_group.values()), height=0.6, color=COLORS[0])
        ax.set_yticks(y, list(by_group))
        ax.tick_params(axis="y", length=0)
        ax.axvline(0, color=INK_2, linewidth=0.8)
        ax.set_xlabel(label)
        ax.xaxis.grid(True)
    _save_to(fig, path)