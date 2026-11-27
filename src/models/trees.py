"""Tree models: random forest and gradient boosting.

Course link: one deep tree = low bias / high variance; a forest averages many trees to lower
variance. Tune max_depth on the validation set.

TODO
- [ ] train_random_forest(): grid over max_depth_grid, pick best val log loss
- [ ] train_gradient_boosting(): sklearn HistGradientBoostingClassifier, grid over max_depth
- [ ] feature_importances(): impurity or permutation importance, grouped by feature prefix
"""

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: F401


def train_random_forest(X_train, y_train, X_val, y_val, cfg):
    raise NotImplementedError


def train_gradient_boosting(X_train, y_train, X_val, y_val, cfg):
    raise NotImplementedError


def feature_importances(model, X_val, y_val, feature_names):
    raise NotImplementedError
