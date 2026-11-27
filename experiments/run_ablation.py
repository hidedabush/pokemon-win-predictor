"""Ablation study.

Using the best model setting from run_main.py, retrain with one feature group removed at a
time (every group in FEATURE_GROUPS: stats, types, match, species, learned, rating) and
report the change in val/test accuracy and log loss.
Also: accuracy vs. training-set size (10k, 25k, 50k, 100k) as a learning curve, if time.

Output: results/ablation.csv, figures/ablation_bar.png
"""

from src.features.build_features import FEATURE_GROUPS  # noqa: F401
from src.utils import ROOT, load_config  # noqa: F401


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
