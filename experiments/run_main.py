"""Main experiment: the results table for the report.        Owner: Nguyen Hang

For each model (logistic, random forest, gradient boosting, optional MLP), and for each
feature setting:
    - teams only     (all groups except rating_*)
    - teams + ratings
tune on val, then score ONCE on test.

Output:
    results/main_results.csv     model, setting, accuracy, log_loss, brier (test)
    results/test_predictions.csv battle_id, model, setting, p_win, y  (for Ruel's plots
                                 and the "wrong predictions" examples)
"""

from src.evaluation.metrics import evaluate  # noqa: F401
from src.utils import ROOT, load_config  # noqa: F401


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
