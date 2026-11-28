"""Ablation study, on val.

Tune gradient boosting once with every feature group (teams + ratings), then retrain those
same settings with one group removed at a time (every group in FEATURE_GROUPS: stats, types,
match, species, learned, rating) and report how much worse val accuracy and log loss get.
Also a learning curve: val scores when training on 10k, 25k, 50k and all 70k train battles.

Test is never touched here (only run_main.py scores test).

Output: results/ablation.csv, results/learning_curve.csv,
        figures/ablation_bar.png, figures/learning_curve.png
"""

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.evaluation.metrics import evaluate
from src.evaluation.plots import ablation_bar, hyperparam_curve
from src.features.build_features import FEATURE_GROUPS, load_features
from src.models.trees import Symmetric, train_gradient_boosting
from src.utils import ROOT, load_config

TRAIN_SIZES = [10_000, 25_000, 50_000, 70_000]  # battles (train has each one twice)


def fit_and_score(booster, X_train, y_train, X_val, y_val) -> dict:
    """Refit a fresh copy of booster (same settings) and score it on val."""
    model = Symmetric(clone(booster).fit(X_train, y_train))
    return evaluate(y_val, model.predict_proba(X_val)[:, 1])


def main():
    cfg = load_config()
    results = ROOT / cfg["paths"]["results_dir"]
    figures = ROOT / cfg["paths"]["figures_dir"]
    X_train, y_train = load_features("train")
    X_val, y_val = load_features("val")

    booster = train_gradient_boosting(X_train, y_train, X_val, y_val, cfg).model
    full = evaluate(y_val, Symmetric(booster).predict_proba(X_val)[:, 1])

    # 1. remove one group at a time
    rows = [{"removed": "nothing", **full}]
    for group in FEATURE_GROUPS:
        keep = [c for c in X_train.columns if not c.startswith(group + "_")]
        scores = fit_and_score(booster, X_train[keep], y_train, X_val[keep], y_val)
        rows.append({"removed": group, **scores})
        print(f"without {group}: {scores}")
    ablation = pd.DataFrame(rows)
    ablation["accuracy_drop"] = full["accuracy"] - ablation["accuracy"]
    ablation["log_loss_increase"] = ablation["log_loss"] - full["log_loss"]
    ablation.to_csv(results / "ablation.csv", index=False)
    drops = ablation[ablation["removed"] != "nothing"].set_index("removed")
    ablation_bar({"accuracy drop (percentage points)": (100 * drops["accuracy_drop"]).to_dict(),
                  "log loss increase": drops["log_loss_increase"].to_dict()},
                 figures / "ablation_bar.png")

    # 2. learning curve: sample whole battles so both copies of a battle stay together.
    # (The learned_* columns were fitted on all of train; we keep them as they are.)
    rng = np.random.default_rng(cfg["seed"])
    battles = X_train.index.unique()
    rows = []
    for n in TRAIN_SIZES:
        rows_in = X_train.index.isin(rng.choice(battles, size=min(n, len(battles)), replace=False))
        scores = fit_and_score(booster, X_train[rows_in], y_train[rows_in], X_val, y_val)
        rows.append({"train_battles": n, **scores})
        print(f"{n:,} train battles: {scores}")
    curve = pd.DataFrame(rows)
    curve.to_csv(results / "learning_curve.csv", index=False)
    hyperparam_curve(TRAIN_SIZES, curve["log_loss"].tolist(), "training battles",
                     figures / "learning_curve.png")

    print(ablation.round(4).to_string(index=False))


if __name__ == "__main__":
    main()