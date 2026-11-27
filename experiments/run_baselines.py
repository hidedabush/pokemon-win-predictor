"""Score every baseline on val and test.        Owner: Andy

Output: results/baselines.csv  (model, split, accuracy, log_loss, brier)
"""

from src.evaluation.metrics import evaluate  # noqa: F401
from src.models.baselines import BASELINES  # noqa: F401
from src.utils import ROOT, load_config  # noqa: F401


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
