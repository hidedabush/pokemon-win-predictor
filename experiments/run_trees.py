"""Tune the tree models on val, for both feature settings, and make their figures.

Never touches test: run_main.py scores test once at the end.

Output:
    results/tree_tuning.csv            val scores for every model, setting and max_depth
    results/tree_group_importance.csv  permutation importance per feature group (teams only)
    figures/hyperparam_rf.png, figures/hyperparam_gb.png
    figures/feature_importance_rf.png  top 15 columns of the teams-only forest
    figures/calibration_val.png        teams-only trees on val (final version uses test)
"""

import pandas as pd

from src.evaluation.plots import calibration_plot, feature_importance, hyperparam_curve
from src.features.build_features import load_features
from src.models.trees import feature_importances, train_gradient_boosting, train_random_forest
from src.utils import ROOT, load_config

SETTINGS = {"teams only": ["rating"], "teams + ratings": []}
MODELS = {"random forest": train_random_forest, "gradient boosting": train_gradient_boosting}


def main():
    cfg = load_config()
    results = ROOT / cfg["paths"]["results_dir"]
    figures = ROOT / cfg["paths"]["figures_dir"]
    tuning, curves, calibration, importance = [], {m: {} for m in MODELS}, {}, []

    for setting, drop in SETTINGS.items():
        X_train, y_train = load_features("train", drop_groups=drop)
        X_val, y_val = load_features("val", drop_groups=drop)
        for name, train in MODELS.items():
            model = train(X_train, y_train, X_val, y_val, cfg)
            tuning.append(model.history_.assign(model=name, setting=setting))
            curves[name][setting] = model.history_["log_loss"].tolist()
            if setting == "teams only":
                calibration[name] = (y_val, model.predict_proba(X_val)[:, 1])
                importance.append(feature_importances(model, X_val, y_val, X_val.columns)
                                  .assign(model=name))
                if name == "random forest":
                    feature_importance(X_val.columns, model.model.feature_importances_,
                                       figures / "feature_importance_rf.png",
                                       xlabel="mean impurity decrease")
            del model  # a deep forest takes a lot of memory

    pd.concat(tuning).to_csv(results / "tree_tuning.csv", index=False)
    pd.concat(importance).to_csv(results / "tree_group_importance.csv", index=False)
    rf, gb = cfg["models"]["random_forest"], cfg["models"]["gradient_boosting"]
    hyperparam_curve(rf["max_depth_grid"], curves["random forest"], "max_depth",
                     figures / "hyperparam_rf.png")
    hyperparam_curve(gb["max_depth_grid"], curves["gradient boosting"], "max_depth",
                     figures / "hyperparam_gb.png")
    calibration_plot(calibration, figures / "calibration_val.png")
    print(pd.concat(tuning).round(4).to_string(index=False))


if __name__ == "__main__":
    main()