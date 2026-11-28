"""Inspect already-saved final predictions; never trains or scores a new model.

Run only after the group has frozen models and completed final test evaluation.
Detailed examples are saved under data/interim (private, ignored by git).
"""

import argparse

import pandas as pd

from src.utils import ROOT, load_config


def analyze_predictions(predictions, battles, top=3):
    predictions = predictions.copy()
    battles = battles.copy()
    predictions["battle_id"] = predictions["battle_id"].astype(str)
    battles["battle_id"] = battles["battle_id"].astype(str)
    if not predictions["p_win"].between(0, 1).all():
        raise ValueError("Predictions must be probabilities in [0, 1].")
    if not predictions["y"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")
    if predictions.duplicated(["battle_id", "model", "setting"]).any():
        raise ValueError("Duplicate prediction for a battle/model/setting.")
    columns = ["battle_id", "p1_team", "p2_team", "p1_rating", "p2_rating", "p1_win"]
    joined = predictions.merge(battles[columns], on="battle_id", how="left",
                               validate="many_to_one", indicator=True)
    if not joined["_merge"].eq("both").all() or not joined["y"].eq(joined["p1_win"]).all():
        raise ValueError("Predictions do not match the supplied test battles and labels.")
    joined = joined.drop(columns=["_merge", "p1_win"])
    joined["predicted_winner"] = (joined["p_win"] >= 0.5).astype(int)
    joined["wrong"] = joined["predicted_winner"] != joined["y"]
    joined["confidence"] = (joined["p_win"] - 0.5).abs() + 0.5
    summary = []
    examples = []
    for (model, setting), group in joined.groupby(["model", "setting"]):
        wrong = group[group["wrong"]]
        summary.append({"model": model, "setting": setting, "n": len(group),
                        "wrong": len(wrong), "error_rate": len(wrong) / len(group)})
        examples.append(wrong.sort_values(["confidence", "battle_id"],
                                         ascending=[False, True]).head(top))
    return pd.DataFrame(summary), pd.concat(examples, ignore_index=True) if examples else joined.head(0)


def main(top=3):
    cfg = load_config()
    results = ROOT / cfg["paths"]["results_dir"]
    interim = ROOT / cfg["data"]["interim_dir"]
    predictions = pd.read_csv(results / "test_predictions.csv", dtype={"battle_id": str})
    battles = pd.read_parquet(interim / "test.parquet")
    summary, examples = analyze_predictions(predictions, battles, top)
    private = interim / "error_analysis"
    private.mkdir(parents=True, exist_ok=True)
    for col in ["p1_team", "p2_team"]:
        examples[col] = examples[col].map(lambda team: team if isinstance(team, str) else " | ".join(team))
    summary.to_csv(private / "error_summary.csv", index=False)
    examples.to_csv(private / "wrong_predictions.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Private examples saved to {private / 'wrong_predictions.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    main(args.top)
