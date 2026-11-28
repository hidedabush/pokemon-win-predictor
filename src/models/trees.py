"""Tree models: random forest and gradient boosting.

Course link: one deep tree = low bias / high variance; a forest averages many trees to lower
variance. Tune max_depth on the validation set.

Both train_* functions return a fitted model with
    model.predict_proba(X)   like any sklearn classifier (column 1 = P(p1 wins))
    model.history_           val scores for every grid value (for the hyperparameter figure)
    model.best_              the chosen grid value
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from src.evaluation.metrics import evaluate
from src.features.build_features import FEATURE_GROUPS


class Symmetric:
    """Wraps a fitted model so that swapping the two teams exactly flips the prediction.

    Every feature is p1 - p2, so the same battle seen from the other side is -X. Trees don't
    learn that symmetry exactly, so we average P(p1 wins | X) with 1 - P(p1 wins | -X).
    """

    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        p = 0.5 * (self.model.predict_proba(X)[:, 1] + 1 - self.model.predict_proba(-X)[:, 1])
        return np.column_stack([1 - p, p])

    def get_params(self, deep = True):
        return self.model.get_params(deep)


def train_random_forest(X_train, y_train, X_val, y_val, cfg):
    """Fit one forest per max_depth in the config grid; keep the best val log loss."""
    rf = cfg["models"]["random_forest"]
    best, rows = None, []
    for depth in rf["max_depth_grid"]:
        forest = RandomForestClassifier(n_estimators=rf["n_estimators"], max_depth=depth,
                                        n_jobs=-1, random_state=cfg["seed"])
        model = Symmetric(forest.fit(X_train, y_train))
        scores = evaluate(y_val, model.predict_proba(X_val)[:, 1])
        rows.append({"max_depth": depth, **scores})
        print(f"random forest  max_depth={depth}: {scores}")
        if best is None or scores["log_loss"] < best_loss:
            best, best_loss = model, scores["log_loss"]
            best.best_ = depth
    best.history_ = pd.DataFrame(rows)
    return best


def train_gradient_boosting(X_train, y_train, X_val, y_val, cfg):
    """Fit one booster per max_depth in the config grid.

    For each depth we also pick the number of boosting rounds on val (the round with the
    lowest val log loss), then keep the depth with the best val log loss overall.
    """
    gb = cfg["models"]["gradient_boosting"]
    best, rows = None, []
    for depth in gb["max_depth_grid"]:
        booster = HistGradientBoostingClassifier(
            learning_rate=gb["learning_rate"], max_depth=depth, max_iter=gb.get("max_iter", 300),
            early_stopping=False, random_state=cfg["seed"],
        ).fit(X_train, y_train)
        # val log loss after every round (same symmetric average as Symmetric)
        losses = [evaluate(y_val, 0.5 * (a[:, 1] + 1 - b[:, 1]))["log_loss"]
                  for a, b in zip(booster.staged_predict_proba(X_val),
                                  booster.staged_predict_proba(-X_val))]
        rounds = int(np.argmin(losses)) + 1
        model = Symmetric(booster.set_params(max_iter=rounds).fit(X_train, y_train))
        scores = evaluate(y_val, model.predict_proba(X_val)[:, 1])
        rows.append({"max_depth": depth, "rounds": rounds, **scores})
        print(f"gradient boosting  max_depth={depth}, rounds={rounds}: {scores}")
        if best is None or scores["log_loss"] < best_loss:
            best, best_loss = model, scores["log_loss"]
            best.best_ = depth
    best.history_ = pd.DataFrame(rows)
    return best


def feature_importances(model, X_val, y_val, feature_names, n_repeats=3, seed=0):
    """Permutation importance per feature group, on val.

    Shuffle all of one group's columns together (so the group's information is lost but
    nothing else changes) and measure how much worse val accuracy and log loss get.
    """
    rng = np.random.default_rng(seed)
    base = evaluate(y_val, model.predict_proba(X_val)[:, 1])
    rows = []
    for group in FEATURE_GROUPS:
        cols = [c for c in feature_names if c.startswith(group + "_")]
        if not cols:
            continue
        shuffled = []
        for _ in range(n_repeats):
            X = X_val.copy()
            X[cols] = X_val[cols].to_numpy()[rng.permutation(len(X_val))]
            shuffled.append(evaluate(y_val, model.predict_proba(X)[:, 1]))
        rows.append({
            "group": group,
            "columns": len(cols),
            "accuracy_drop": base["accuracy"] - np.mean([s["accuracy"] for s in shuffled]),
            "log_loss_increase": np.mean([s["log_loss"] for s in shuffled]) - base["log_loss"],
        })
    return pd.DataFrame(rows).sort_values("log_loss_increase", ascending=False)