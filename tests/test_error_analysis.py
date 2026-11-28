import pandas as pd
import pytest

from experiments.analyze_errors import analyze_predictions


def sample():
    battles = pd.DataFrame({"battle_id": ["a", "b", "c"],
                           "p1_team": [["pikachu"]] * 3, "p2_team": [["eevee"]] * 3,
                           "p1_rating": [1500] * 3, "p2_rating": [1490] * 3,
                           "p1_win": [0, 1, 1]})
    preds = pd.DataFrame({"battle_id": ["a", "b", "c"], "model": ["logistic"] * 3,
                         "setting": ["teams only"] * 3, "p_win": [0.9, 0.2, 0.5],
                         "y": [0, 1, 1]})
    return preds, battles


def test_errors():
    preds, battles = sample()
    summary, examples = analyze_predictions(preds, battles, top=1)
    assert summary.iloc[0]["wrong"] == 2
    assert examples.iloc[0]["battle_id"] == "a"
    assert examples.iloc[0]["confidence"] == pytest.approx(0.9)


def test_mismatched_labels():
    preds, battles = sample()
    preds.loc[0, "y"] = 1
    with pytest.raises(ValueError, match="do not match"):
        analyze_predictions(preds, battles)
