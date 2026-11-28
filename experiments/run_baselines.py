"""Score baselines on validation; include test only at the agreed final evaluation.

Output: results/baselines.csv (model, split, accuracy, log_loss, brier)
The final experiment runner can call main(include_test=True).
"""

import argparse

import pandas as pd

from src.evaluation.metrics import evaluate
from src.features.build_features import load_features
from src.models.baselines import BASELINES
from src.utils import ROOT, load_config


def score_split(split):
    X, y = load_features(split)
    rows = []
    for name, predict in BASELINES.items():
        probabilities = predict(X)
        scores = evaluate(y, probabilities)
        rows.append({"model": name, "split": split, **scores})
    return rows


def main(include_test=False):
    cfg = load_config()
    output = ROOT / cfg["paths"]["results_dir"] / "baselines.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = score_split("val")
    if include_test:
        rows += score_split("test")
    elif output.exists():
        # Do not erase previously recorded final test results.
        previous = pd.read_csv(output)
        if "test" in previous["split"].values:
            raise RuntimeError("Final baseline results already exist; not overwriting them.")
    results = pd.DataFrame(rows)
    results.to_csv(output, index=False)
    print(results.to_string(index=False))
    print(f"Saved {output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-test", action="store_true",
                        help="Include test only at the coordinated final evaluation.")
    args = parser.parse_args()
    main(include_test=args.final_test)
