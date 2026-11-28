"""Main experiment: the results table for the report.

For each model (logistic, random forest, gradient boosting, optional MLP), and for each
feature setting:
    - teams only     (all groups except rating_*)
    - teams + ratings
tune on val, then score ONCE on test.

Models whose code isn't written yet (raise NotImplementedError) are skipped with a note, so
this runs as soon as any one model exists. Rerun it when another model is added.

Output:
    results/main_results.csv          model, setting, params, accuracy, log_loss, brier (test),
                                      plus val_accuracy, val_log_loss, val_brier (for reference)
    results/test_predictions.csv      battle_id, model, setting, p_win, y  (for the plots
                                      and the "wrong predictions" examples)
    results/logistic_c_curve.csv      setting, C, val log loss / accuracy / brier (hyperparam figure)
    results/logistic_top_coefficients.csv  setting, feature, coef (feature-importance discussion)

Usage:
    python -m experiments.run_main                       # every model that exists
    python -m experiments.run_main --models logistic     # only some models
"""

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import evaluate
from src.features.build_features import load_features
from src.models.logistic import top_coefficients, train_logistic
from src.utils import ROOT, load_config

SETTINGS = {"teams_only": ["rating"], "teams_ratings": []}  # setting -> feature groups dropped


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


def _unpack(out):
    """Other trainers may return model or (model, anything...); describe the model's params."""
    model = out[0] if isinstance(out, tuple) else out
    params = {k: v for k, v in model.get_params().items()
              if k in ("max_depth", "n_estimators", "learning_rate", "hidden_layer_sizes", "alpha")}
    return model, " ".join(f"{k}={v}" for k, v in params.items()), {}


MODELS = {
    "logistic": fit_logistic,
    "random_forest": fit_random_forest,
    "gradient_boosting": fit_gradient_boosting,
    "mlp": fit_mlp,
}


def main(models=None, cfg=None, processed_dir=None, results_dir=None) -> pd.DataFrame:
    """Train every model x setting on train, tune on val, score test once. Return the table.

    processed_dir / results_dir default to the folders in config.yaml (tests pass temp folders).
    """
    cfg = cfg or load_config()
    results_dir = Path(results_dir or ROOT / cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    models = models or list(MODELS)

    # Fit and tune everything on train/val first ...
    fitted = []  # (model_name, setting, model, params, val_metrics, extras)
    for setting, drop in SETTINGS.items():
        X_train, y_train = load_features("train", drop, processed_dir)
        X_val, y_val = load_features("val", drop, processed_dir)
        for name in models:
            print(f"\n== {name} / {setting} ({X_train.shape[1]} features, {len(X_train):,} train rows)")
            try:
                model, params, extras = MODELS[name](X_train, y_train, X_val, y_val, cfg)
            except NotImplementedError:
                print(f"   skipped: {name} isn't implemented yet")
                continue
            val = evaluate(y_val, model.predict_proba(X_val)[:, 1])
            print(f"   picked {params}: val accuracy {val['accuracy']:.4f}, "
                  f"log loss {val['log_loss']:.4f}")
            fitted.append((name, setting, model, params, val, extras))
    if not fitted:
        raise SystemExit("No model could be trained.")

    # ... then touch test exactly once per final model.
    rows, preds, curves, coefs = [], [], [], []
    for setting, drop in SETTINGS.items():
        X_test, y_test = load_features("test", drop, processed_dir)
        for name, s, model, params, val, extras in fitted:
            if s != setting:
                continue
            p_win = model.predict_proba(X_test)[:, 1]
            test = evaluate(y_test, p_win)
            rows.append({"model": name, "setting": setting, "params": params, **test,
                         **{f"val_{k}": v for k, v in val.items()}})
            preds.append(pd.DataFrame({"battle_id": X_test.index, "model": name,
                                       "setting": setting, "p_win": p_win, "y": y_test.values}))
            if "c_curve" in extras:
                curves.append(extras["c_curve"].assign(setting=setting))
            if "top_coefficients" in extras:
                coefs.append(extras["top_coefficients"].assign(setting=setting))

    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "main_results.csv", index=False)
    pd.concat(preds).to_csv(results_dir / "test_predictions.csv", index=False)
    if curves:
        pd.concat(curves)[["setting", "C", "val_log_loss", "val_accuracy", "val_brier", "best"]] \
            .to_csv(results_dir / "logistic_c_curve.csv", index=False)
    if coefs:
        pd.concat(coefs)[["setting", "feature", "coef"]] \
            .to_csv(results_dir / "logistic_top_coefficients.csv", index=False)

    with pd.option_context("display.width", 140, "display.precision", 4):
        print("\nTest results (scored once):")
        print(table[["model", "setting", "params", "accuracy", "log_loss", "brier"]]
              .to_string(index=False))
    print(f"\nwrote main_results.csv, test_predictions.csv to {results_dir}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune on val, score test once.")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), help="default: all")
    main(parser.parse_args().models)
