"""Run with: pytest"""

import pandas as pd

from src.evaluation.metrics import evaluate
from src.utils import ROOT, load_config, to_id


def test_to_id():
    assert to_id("Great Tusk") == "greattusk"
    assert to_id("Urshifu-Rapid-Strike") == "urshifurapidstrike"
    assert to_id("Flabébé") == "flabebe"  # accents dropped, not the whole letter
    assert to_id("Farfetch’d") == "farfetchd"
    assert to_id("Type: Null") == "typenull"


def test_config_loads():
    cfg = load_config()
    s = cfg["split"]
    assert abs(s["train"] + s["val"] + s["test"] - 1.0) < 1e-9


def test_sample_matches_contract():
    df = pd.read_csv(ROOT / "data/sample/battles_sample.csv")
    cols = ["battle_id", "upload_time", "p1_team", "p2_team", "p1_rating", "p2_rating", "p1_win"]
    assert list(df.columns) == cols
    assert all(len(t.split("|")) == 6 for t in df["p1_team"])


def test_evaluate_coin_flip():
    m = evaluate([0, 1, 0, 1], [0.5] * 4)
    assert abs(m["brier"] - 0.25) < 1e-9
