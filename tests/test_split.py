"""Tests for src/data/split.py on a small fake battles.parquet in a temp folder."""

import json

import pandas as pd
import pytest

from src.data.split import split
from tests.fake_data import tmp_config


def write_battles(cfg: dict, n: int, same_second: bool = False) -> pd.DataFrame:
    """n battles in shuffled order. With same_second, pairs of battles share an upload_time."""
    team = ["greattusk", "kingambit", "gholdengo", "dragonite", "ironvaliant", "corviknight"]
    df = pd.DataFrame({
        "battle_id": [f"gen9ou-{i:04d}" for i in range(n)],
        "upload_time": [1_750_000_000 + 60 * (i // 2 if same_second else i) for i in range(n)],
        "p1_team": [team] * n, "p2_team": [team] * n,
        "p1_rating": 1500.0, "p2_rating": 1500.0,
        "p1_win": [i % 2 for i in range(n)],
    })
    shuffled = df.sample(frac=1, random_state=0)
    shuffled.to_parquet(f"{cfg['data']['interim_dir']}/battles.parquet", index=False)
    return df


def test_sizes_order_and_files(tmp_path):
    cfg = tmp_config(tmp_path)
    df = write_battles(cfg, 100)
    parts = split(cfg)

    assert [len(parts[s]) for s in ("train", "val", "test")] == [70, 15, 15]
    # oldest -> train, newest -> test, every battle exactly once
    together = pd.concat(parts.values(), ignore_index=True)
    assert together["battle_id"].tolist() == df["battle_id"].tolist()
    for name, part in parts.items():
        saved = pd.read_parquet(tmp_path / "interim_dir" / f"{name}.parquet")
        assert saved["battle_id"].tolist() == part["battle_id"].tolist()

    stats = json.loads((tmp_path / "interim_dir/data_stats.json").read_text())
    assert [r["battles"] for r in stats["splits"]] == [70, 15, 15]
    assert stats["splits"][0]["p1_win_rate"] == round(parts["train"]["p1_win"].mean(), 4)


def test_same_second_ties_are_cut_by_battle_id(tmp_path):
    # 70% of 30 is 21: the cut falls between two battles uploaded in the same second.
    cfg = tmp_config(tmp_path)
    write_battles(cfg, 30, same_second=True)
    first = split(cfg)
    assert first["train"]["battle_id"].iloc[-1] == "gen9ou-0020"
    assert first["val"]["battle_id"].iloc[0] == "gen9ou-0021"
    assert first["train"]["upload_time"].max() == first["val"]["upload_time"].min()
    # Running it again on a differently shuffled file gives the same cut.
    write_battles(cfg, 30, same_second=True).sample(frac=1, random_state=7).to_parquet(
        tmp_path / "interim_dir/battles.parquet", index=False)
    again = split(cfg)
    assert all(again[s]["battle_id"].tolist() == first[s]["battle_id"].tolist() for s in first)


def test_fractions_must_sum_to_one(tmp_path):
    cfg = tmp_config(tmp_path)
    write_battles(cfg, 10)
    cfg["split"] = {"train": 0.7, "val": 0.2, "test": 0.2}
    with pytest.raises(ValueError):
        split(cfg)
