"""Checks on the real data files. Skipped on machines that haven't built them (data/ isn't in git).

After re-running any pipeline step, run `pytest tests/test_real_data.py` to make sure the
files still fit together.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import LABEL, feature_columns
from src.features.pokedex import load_aliases, load_pokedex
from src.utils import ROOT, load_config

CFG = load_config()
INTERIM = ROOT / CFG["data"]["interim_dir"]
PROCESSED = ROOT / CFG["data"]["processed_dir"]
SPLITS = ("train", "val", "test")
FILES = ([INTERIM / "battles.parquet", INTERIM / "data_stats.json"]
         + [INTERIM / f"{s}.parquet" for s in SPLITS]
         + [PROCESSED / f"{s}_features.parquet" for s in SPLITS])

pytestmark = pytest.mark.skipif(not all(p.exists() for p in FILES),
                                reason="real data not built on this machine")


@pytest.fixture(scope="module")
def battles():
    return pd.read_parquet(INTERIM / "battles.parquet")


@pytest.fixture(scope="module")
def splits():
    return {s: pd.read_parquet(INTERIM / f"{s}.parquet") for s in SPLITS}


@pytest.fixture(scope="module")
def feats():
    return {s: pd.read_parquet(PROCESSED / f"{s}_features.parquet") for s in SPLITS}


def test_battles_pass_every_filter(battles):
    d = CFG["data"]
    stats = json.loads((INTERIM / "data_stats.json").read_text())
    if stats["sample"]["limit"] is None:
        assert len(battles) == min(d["sample_size"], stats["sample"]["clean_battles"])
    assert battles["battle_id"].is_unique
    assert battles["upload_time"].is_monotonic_increasing
    assert battles["upload_time"].min() >= pd.Timestamp(d["min_upload_date"], tz="UTC").timestamp()
    ratings = battles[["p1_rating", "p2_rating"]]
    assert ratings.notna().all(axis=None) and (ratings >= d["min_rating"]).all(axis=None)
    assert ((battles["p1_rating"] - battles["p2_rating"]).abs() <= d["max_rating_gap"]).all()
    assert set(battles[LABEL].unique()) <= {0, 1}
    for col in ("p1_team", "p2_team"):
        assert battles[col].map(lambda t: len(set(t)) == 6).all()  # 6 different species


def test_every_species_is_in_the_pokedex(battles):
    dex = load_pokedex()
    seen = {s for col in ("p1_team", "p2_team") for t in battles[col] for s in t}
    assert seen <= set(dex)


def test_splits_partition_battles_by_time(battles, splits):
    together = pd.concat([splits[s] for s in SPLITS], ignore_index=True)
    assert together["battle_id"].tolist() == battles["battle_id"].tolist()
    for a, b in zip(SPLITS, SPLITS[1:]):
        assert splits[a]["upload_time"].max() <= splits[b]["upload_time"].min()
    sizes = [len(splits[s]) / len(battles) for s in SPLITS]
    assert np.allclose(sizes, [CFG["split"][s] for s in SPLITS], atol=0.001)


def test_feature_rows_line_up_with_splits(splits, feats):
    for s in ("val", "test"):
        assert feats[s]["battle_id"].tolist() == splits[s]["battle_id"].tolist()
        assert feats[s][LABEL].tolist() == splits[s][LABEL].tolist()
    train, n = feats["train"], len(splits["train"])
    assert len(train) == (2 * n if CFG["features"]["swap_augment"] else n)
    assert train["battle_id"].iloc[:n].tolist() == splits["train"]["battle_id"].tolist()
    assert train[LABEL].iloc[:n].tolist() == splits["train"][LABEL].tolist()


def test_swapped_half_is_negated(feats):
    if not CFG["features"]["swap_augment"]:
        pytest.skip("swap augmentation is off")
    train = feats["train"]
    n, cols = len(train) // 2, feature_columns(train.columns)
    assert np.array_equal(train[cols].iloc[:n].to_numpy(), -train[cols].iloc[n:].to_numpy())
    assert (train[LABEL].iloc[:n].to_numpy() == 1 - train[LABEL].iloc[n:].to_numpy()).all()


def test_feature_columns_are_sane(splits, feats):
    cols = feature_columns(feats["train"].columns)
    for df in feats.values():
        assert feature_columns(df.columns) == cols
        assert np.isfinite(df[cols].to_numpy(np.float32)).all()
    species = [c.removeprefix("species_") for c in cols if c.startswith("species_")]
    dex, aliases = load_pokedex(), load_aliases()
    assert all(s in dex and s not in aliases for s in species)
    val = splits["val"]
    assert np.allclose(feats["val"]["rating_diff"], val["p1_rating"] - val["p2_rating"])
