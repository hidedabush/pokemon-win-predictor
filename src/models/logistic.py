"""L2-regularized logistic regression.

Course link (Methods section): sigmoid output = P(p1 wins); trained by minimizing
cross-entropy (= log loss) with gradient-based optimization (L-BFGS); the L2 penalty controls
overfitting on the many species_* columns; C (inverse penalty strength) is tuned on the
validation set (bias–variance trade-off: small C = simple model, large C = flexible model).

Details:
- Non-species columns are standardized (mean 0, std 1) with a scaler fitted on TRAIN only.
  species_* columns are already +1/0/−1 and stay as they are, so their weights read directly
  as "how much having this Pokémon (vs. the opponent having it) changes the log-odds".
- No intercept. Every feature is p1 − p2, so swapping the teams negates x, and without an
  intercept the model gives exactly P(swapped) = 1 − P(original). With swap augmentation the
  fitted intercept would be ~0 anyway; dropping it makes the symmetry exact.
- The model is fitted on train only (val is only used to pick C), so the learned_* features
  mean the same thing at fit time and at prediction time.

Usage:
    model, curve = train_logistic(X_train, y_train, X_val, y_val, cfg["models"]["logistic"]["C_grid"])
    p_win = model.predict_proba(X_test)[:, 1]
    top_coefficients(model, X_train.columns)
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import evaluate

MAX_ITER = 5000


def make_logistic(columns, C: float) -> Pipeline:
    """Unfitted pipeline: standardize non-species columns, then L2 logistic regression."""
    columns = list(columns)
    scaled = [c for c in columns if not c.startswith("species_")]
    unscaled = [c for c in columns if c.startswith("species_")]
    prep = ColumnTransformer(
        [("scale", StandardScaler(), scaled), ("keep", "passthrough", unscaled)],
        verbose_feature_names_out=False,
    )
    # l1_ratio=0 is a pure L2 penalty (the default; `penalty="l2"` is deprecated in sklearn 1.8+).
    clf = LogisticRegression(C=C, l1_ratio=0.0, fit_intercept=False, max_iter=MAX_ITER)
    return Pipeline([("prep", prep), ("clf", clf)])


def train_logistic(X_train, y_train, X_val, y_val, C_grid):
    """Fit one model per C on train, keep the one with the lowest val log loss.

    Returns (model, curve):
        model  fitted Pipeline with the best C; predict_proba() takes the raw feature frame
        curve  DataFrame with C, val_log_loss, val_accuracy, val_brier, best (for the
               hyperparameter figure: plots.hyperparam_curve(curve.C, curve.val_log_loss, ...))
    """
    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("train and val have different feature columns")
    rows, best, best_loss = [], None, np.inf
    for C in C_grid:
        model = make_logistic(X_train.columns, C).fit(X_train, y_train)
        m = evaluate(y_val, model.predict_proba(X_val)[:, 1])
        rows.append({"C": C, **{f"val_{k}": v for k, v in m.items()}})
        print(f"  C={C:<8g} val log loss {m['log_loss']:.4f}  accuracy {m['accuracy']:.4f}")
        if m["log_loss"] < best_loss:
            best, best_loss = model, m["log_loss"]
    curve = pd.DataFrame(rows)
    curve["best"] = curve["C"] == best.named_steps["clf"].C
    return best, curve


def top_coefficients(model, feature_names=None, k=20) -> pd.DataFrame:
    """The k largest positive and k most negative weights (positive = favours player 1).

    Weights of standardized columns are per 1 std of that feature; species_* weights are per
    "this side has it and the other doesn't". feature_names defaults to the names the model
    was fitted with. Returns DataFrame(feature, coef), sorted from most positive to most negative.
    """
    names = model.named_steps["prep"].get_feature_names_out()  # order after the transformer
    coef = pd.Series(model.named_steps["clf"].coef_[0], index=names)
    if feature_names is not None:
        coef = coef.reindex(list(feature_names))
    coef = coef.sort_values(ascending=False)
    top = pd.concat([coef.head(k), coef.tail(k)])
    top = top[~top.index.duplicated()]  # fewer than 2k features: don't list one twice
    return top.rename_axis("feature").reset_index(name="coef")
