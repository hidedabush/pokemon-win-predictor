"""Main experiment: the results table for the report.

For each model (logistic, random forest, gradient boosting, optional MLP), and for each
feature setting:
    - teams only     (all groups except rating_*)
    - teams + ratings
tune on val, then score ONCE on test.

Each model is tuned on train/val first; only then are its test predictions made (and the model
freed, since a deep forest is large). Test scores are computed at the very end, so nothing
about test can influence a choice. A model whose code isn't written raises an error instead of
being left out silently. If results/main_results.csv already exists, this stops unless you
pass --force: the test set should only be scored once.

The baselines are scored separately: python -m experiments.run_baselines --final-test

Uncertainty: 15k test battles still leave a few tenths of a percent of luck in every number,
so each test metric gets a 95% bootstrap interval (resample the test battles with replacement,
evaluation.bootstrap_samples times). Each model also gets a PAIRED interval for its difference
from logistic regression (same resamples for both models). If that interval contains 0, the
two models can't be told apart on this test set.

Output:
    results/main_results.csv          model, setting, params, accuracy, log_loss, brier (test),
                                      <metric>_lo / <metric>_hi (95% interval),
                                      accuracy_vs_logistic, log_loss_vs_logistic (+ _lo / _hi;
                                      model minus logistic, empty on the logistic rows),
                                      val_accuracy, val_log_loss, val_brier (for reference)
    results/test_predictions.csv      battle_id, model, setting, p_win, y  (for the plots
                                      and experiments/analyze_errors.py)
    results/logistic_c_curve.csv      setting, C, val log loss / accuracy / brier
    results/logistic_top_coefficients.csv  setting, feature, coef
    figures/calibration_test.png      every model, teams only, on test
    figures/hyperparam_logistic.png   val log loss vs. C, both settings
    figures/feature_importance_logistic.png  largest logistic weights, teams only

Usage:
    python -m experiments.run_main                       # logistic, random forest, gradient boosting
    python -m experiments.run_main --models logistic mlp # only some models
    python -m experiments.run_main --force               # overwrite existing final results
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import METRICS, bootstrap_ci, evaluate
from src.evaluation.plots import calibration_plot, feature_importance, hyperparam_curve
from src.features.build_features import load_features
from src.models.logistic import top_coefficients, train_logistic
from src.utils import ROOT, load_config

# setting -> feature groups dropped. Same names as experiments/run_trees.py.
SETTINGS = {"teams only": ["rating"], "teams + ratings": []}
DEFAULT_MODELS = ["logistic", "random_forest", "gradient_boosting"]  # mlp is optional
REFERENCE = "logistic"  # main model: every other model gets a paired difference against it


def fit_logistic(X_train, y_train, X_val, y_val, cfg):
    model, curve = train_logistic(X_train, y_train, X_val, y_val,
                                  cfg["models"]["logistic"]["C_grid"])
    extras = {"c_curve": curve, "top_coefficients": top_coefficients(model, X_train.columns)}
    return model, f"C={model.named_steps['clf'].C:g}", extras


def fit_random_forest(X_train, y_train, X_val, y_val, cfg):
    from src.models.trees import train_random_forest
    return _unpack(train_random_forest(X_train, y_train, X_val, y_val, cfg))


def fit_gradient_boosting(X_train, y_train, X_val, y_val, cfg):
    from src.models.trees import train_gradient_boosting
    return _unpack(train_gradient_boosting(X_train, y_train, X_val, y_val, cfg))


def fit_mlp(X_train, y_train, X_val, y_val, cfg):
    from src.models.mlp import train_mlp
    return _unpack(train_mlp(X_train, y_train, X_val, y_val))


PARAMS_SHOWN = {"max_depth": "max_depth", "max_iter": "rounds", "n_estimators": "n_estimators",
                "learning_rate": "learning_rate", "hidden_layer_sizes": "hidden",
                "alpha": "alpha"}


def _unpack(out):
    """Other trainers may return model or (model, anything...); describe the model's params."""
    model = out[0] if isinstance(out, tuple) else out
    params = model.get_params()
    shown = [f"{label}={params[key]}" for key, label in PARAMS_SHOWN.items() if key in params]
    return model, " ".join(shown), {}


MODELS = {
    "logistic": fit_logistic,
    "random_forest": fit_random_forest,
    "gradient_boosting": fit_gradient_boosting,
    "mlp": fit_mlp,
}


def main(models=None, cfg=None, processed_dir=None, results_dir=None, figures_dir=None,
         force=False) -> pd.DataFrame:
    """Train every model x setting on train, tune on val, score test once. Return the table.

    processed_dir / results_dir / figures_dir default to the folders in config.yaml (tests
    pass temp folders).
    """
    cfg = cfg or load_config()
    results_dir = Path(results_dir or ROOT / cfg["paths"]["results_dir"])
    figures_dir = Path(figures_dir or ROOT / cfg["paths"]["figures_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if (results_dir / "main_results.csv").exists() and not force:
        raise SystemExit(f"{results_dir / 'main_results.csv'} already exists: test has been "
                         "scored. Pass --force only if the models really changed.")
    models = models or DEFAULT_MODELS

    runs = []  # dict per model x setting: name, setting, params, val, extras, p_test
    for setting, drop in SETTINGS.items():
        X_train, y_train = load_features("train", drop, processed_dir)
        X_val, y_val = load_features("val", drop, processed_dir)
        X_test, y_test = load_features("test", drop, processed_dir)
        for name in models:
            print(f"\n== {name} / {setting} ({X_train.shape[1]} features, {len(X_train):,} train rows)")
            try:
                model, params, extras = MODELS[name](X_train, y_train, X_val, y_val, cfg)
            except NotImplementedError:
                raise SystemExit(f"{name} isn't implemented yet; leave it out with --models.")
            val = evaluate(y_val, model.predict_proba(X_val)[:, 1])
            print(f"   picked {params}: val accuracy {val['accuracy']:.4f}, "
                  f"log loss {val['log_loss']:.4f}")
            # Tuning for this model is finished, so its test predictions can't affect any choice.
            # They are only scored after every model is done.
            runs.append({"model": name, "setting": setting, "params": params, "val": val,
                         "extras": extras, "p_test": model.predict_proba(X_test)[:, 1],
                         "test_ids": X_test.index, "y_test": y_test.to_numpy()})
            del model
            gc.collect()

    # Bootstrap intervals, per setting, all models on the same resampled battles.
    n_boot = cfg["evaluation"].get("bootstrap_samples", 1000)
    cis = {}
    for setting in SETTINGS:
        group = [r for r in runs if r["setting"] == setting]
        if not group:
            continue
        reference = REFERENCE if any(r["model"] == REFERENCE for r in group) else None
        ci = bootstrap_ci(group[0]["y_test"], {r["model"]: r["p_test"] for r in group},
                          reference=reference, n_boot=n_boot, seed=cfg["seed"])
        cis.update({(setting, name): v for name, v in ci.items()})

    rows, preds, curves, coefs, calibration = [], [], [], [], {}
    for r in runs:
        test = evaluate(r["y_test"], r["p_test"])
        ci = cis[(r["setting"], r["model"])]
        row = {"model": r["model"], "setting": r["setting"], "params": r["params"]}
        for m in METRICS:
            row.update({m: test[m], f"{m}_lo": ci[m][0], f"{m}_hi": ci[m][1]})
        for m in ("accuracy", "log_loss"):  # paired difference vs. logistic (empty for logistic)
            d = ci.get(f"diff_{m}", (np.nan, np.nan, np.nan))
            row.update({f"{m}_vs_{REFERENCE}": d[0], f"{m}_vs_{REFERENCE}_lo": d[1],
                        f"{m}_vs_{REFERENCE}_hi": d[2]})
        rows.append({**row, **{f"val_{k}": v for k, v in r["val"].items()}})
        preds.append(pd.DataFrame({"battle_id": r["test_ids"], "model": r["model"],
                                   "setting": r["setting"], "p_win": r["p_test"],
                                   "y": r["y_test"]}))
        if r["setting"] == "teams only":
            calibration[r["model"].replace("_", " ")] = (r["y_test"], r["p_test"])
        if "c_curve" in r["extras"]:
            curves.append(r["extras"]["c_curve"].assign(setting=r["setting"]))
        if "top_coefficients" in r["extras"]:
            coefs.append(r["extras"]["top_coefficients"].assign(setting=r["setting"]))

    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "main_results.csv", index=False)
    pd.concat(preds).to_csv(results_dir / "test_predictions.csv", index=False)
    calibration_plot(calibration, figures_dir / "calibration_test.png",
                     n_bins=cfg["evaluation"]["calibration_bins"])
    if curves:
        curve = pd.concat(curves)
        curve[["setting", "C", "val_log_loss", "val_accuracy", "val_brier", "best"]] \
            .to_csv(results_dir / "logistic_c_curve.csv", index=False)
        hyperparam_curve(cfg["models"]["logistic"]["C_grid"],
                         {s: g["val_log_loss"].tolist() for s, g in curve.groupby("setting", sort=False)},
                         "C (inverse L2 strength)", figures_dir / "hyperparam_logistic.png")
    if coefs:
        coef = pd.concat(coefs)
        coef[["setting", "feature", "coef"]] \
            .to_csv(results_dir / "logistic_top_coefficients.csv", index=False)
        only = coef[coef["setting"] == "teams only"]
        feature_importance(only["feature"], only["coef"],
                           figures_dir / "feature_importance_logistic.png",
                           xlabel="logistic weight (+ favours player 1)")

    print(f"\nTest results (scored once), 95% bootstrap intervals over {n_boot} resamples:")
    print(f"{'model':<19}{'setting':<17}{'accuracy':<23}{'log loss':<25}"
          f"{'log loss vs ' + REFERENCE + ' (paired)'}")
    for _, r in table.iterrows():
        vs = r[f"log_loss_vs_{REFERENCE}"]
        vs = "" if np.isnan(vs) else (
            f"{vs:+.4f} [{r[f'log_loss_vs_{REFERENCE}_lo']:+.4f}, {r[f'log_loss_vs_{REFERENCE}_hi']:+.4f}]"
            + ("  <- differs" if r[f"log_loss_vs_{REFERENCE}_lo"] > 0 or r[f"log_loss_vs_{REFERENCE}_hi"] < 0 else ""))
        print(f"{r['model']:<19}{r['setting']:<17}"
              f"{100 * r['accuracy']:.1f}% [{100 * r['accuracy_lo']:.1f}, {100 * r['accuracy_hi']:.1f}]{'':<5}"
              f"{r['log_loss']:.4f} [{r['log_loss_lo']:.4f}, {r['log_loss_hi']:.4f}]{'':<3}{vs}")
    print(f"\nwrote main_results.csv, test_predictions.csv to {results_dir}, figures to {figures_dir}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune on val, score test once.")
    parser.add_argument("--models", nargs="+", choices=list(MODELS),
                        help=f"default: {' '.join(DEFAULT_MODELS)}")
    parser.add_argument("--force", action="store_true", help="overwrite existing final results")
    args = parser.parse_args()
    main(args.models, force=args.force)
