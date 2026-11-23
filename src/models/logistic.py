"""L2-regularized logistic regression.        Owner: Nguyen Hang

Course link (Methods section): sigmoid output = P(p1 wins); trained by minimizing
cross-entropy (= log loss) with gradient-based optimization; L2 penalty controls overfitting
on the many species_* columns; C tuned on the validation set (bias–variance trade-off).

TODO
- [ ] Standardize non-species features (fit the scaler on train only)
- [ ] For each C in config models.logistic.C_grid: fit on train, log loss on val
- [ ] Keep the best C; return the fitted model
- [ ] Save val log loss vs. C (for the hyperparameter figure)
- [ ] top_coefficients(): largest +/− weights for the feature-importance discussion
"""

from sklearn.linear_model import LogisticRegression  # noqa: F401


def train_logistic(X_train, y_train, X_val, y_val, C_grid):
    raise NotImplementedError


def top_coefficients(model, feature_names, k=20):
    raise NotImplementedError
