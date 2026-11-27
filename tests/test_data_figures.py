"""Tests for src/data/figures.py: the numbers behind the Data-section figures and table."""

import pandas as pd

from src.data.figures import filter_table, species_stats, split_timeline, top_species

A = ["greattusk", "kingambit", "gholdengo", "dragonite", "ironvaliant", "corviknight"]
B = ["greattusk", "toxapex", "heatran", "rotomwash", "alcremierubycream", "dragapult"]


def test_species_stats_counts_both_sides_and_merges_forms():
    train = pd.DataFrame({"p1_team": [A, B], "p2_team": [B, A], "p1_win": [1, 1]})
    stats = species_stats(train)
    # Great Tusk is on all 4 teams and wins 2 of them; Kingambit is on 2 and wins 1.
    assert stats.loc["greattusk", ["teams", "wins"]].tolist() == [4, 2]
    assert stats.loc["greattusk", "usage"] == 1.0
    assert stats.loc["kingambit", "win_rate"] == 0.5
    assert "alcremie" in stats.index and "alcremierubycream" not in stats.index
    assert stats.index[0] == "greattusk"  # most used first


def test_filter_table_has_every_step(tmp_path):
    stats = {"drop_counts": [{"step": "total rows", "battles": 1000, "dropped": 0},
                             {"step": "both >= 1300", "battles": 400, "dropped": 600}],
             "sample": {"seed": 42, "sampled": 100}}
    path = tmp_path / "results" / "table.tex"
    filter_table(stats, path)
    text = path.read_text()
    assert "1,000 & --" in text and "400 & 600" in text and "$\\geq$ 1300" in text
    assert "seed 42) & 100" in text
    assert text.count("\\\\") == 4  # header, two steps, sample


def test_figures_are_written(tmp_path):
    t0 = 1_750_000_000
    parts = {s: pd.DataFrame({"upload_time": [t0 + i * 86_400 * 20 + k for k in range(3)],
                              "p1_team": [A] * 3, "p2_team": [B] * 3, "p1_win": [1, 0, 1]})
             for i, s in enumerate(("train", "val", "test"))}
    split_timeline(parts, tmp_path)
    top_species(parts["train"], tmp_path, k=5)
    for name in ("data_split_timeline", "data_top_species"):
        for ext in ("pdf", "png"):
            assert (tmp_path / f"{name}.{ext}").stat().st_size > 0

