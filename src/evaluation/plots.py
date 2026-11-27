"""Report figures. Save every figure to figures/ as PNG (300 dpi).

Figures for the report:
    1. calibration_plot      predicted vs. actual win rate in 10 bins, one line per model
    2. feature_importance    top features (logistic weights and/or tree importance)
    3. hyperparam_curve      val log loss vs. C (logistic) or max_depth (trees)
    4. ablation_bar          accuracy drop when each feature group is removed
    5. (optional) luck_hist  win-rate distribution from repeated simulated battles

Keep them readable at report size: large fonts, axis labels, no titles that repeat the caption.

TODO
- [ ] One function per figure above
"""

import matplotlib.pyplot as plt  # noqa: F401
from sklearn.calibration import calibration_curve  # noqa: F401


def calibration_plot(results: dict, path: str, n_bins: int = 10):
    """results: {model_name: (y_true, p_win)}"""
    raise NotImplementedError


def feature_importance(names, values, path: str, k: int = 15):
    raise NotImplementedError


def hyperparam_curve(xs, val_losses, xlabel: str, path: str):
    raise NotImplementedError


def ablation_bar(drops: dict, path: str):
    raise NotImplementedError
