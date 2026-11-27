"""End to end on fake data: raw logs -> parse_logs -> split -> build_features -> load_features.

Everything is written to a temp folder, so this runs without the real dataset.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.parse_logs import build_battles
from src.data.split import split
from src.features.build_features import (LABEL, LEARNED_COLS, LearnedScores, SpeciesTable, build,
                                         feature_columns, load_features, out_of_fold_scores)
from src.features.pokedex import load_aliases, load_pokedex, load_type_chart
from tests.fake_data import random_battles, tmp_config, write_raw

N_OLD, N_NEW = 200, 10  # the N_NEW newest battles (all in test) have Flabébé on p1


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Run the whole pipeline once; return the config and every file it wrote."""
    tmp = tmp_path_factory.mktemp("pipeline")
    cfg = tmp_config(tmp)
    old = random_battles(N_OLD)
    new = random_battles(N_NEW, seed=1, start=old[-1]["upload_time"] + 60, first_id=5000,
                         always=["Flabébé"])
    write_raw(cfg, old + new)
    build_battles(cfg)
    split(cfg)
    build(cfg)
    interim, processed = tmp / "interim_dir", tmp / "processed_dir"
    return {
        "cfg": cfg,
        "splits": {s: pd.read_parquet(interim / f"{s}.parquet") for s in ("train", "val", "test")},
        "feats": {s: pd.read_parquet(processed / f"{s}_features.parquet")
                  for s in ("train", "val", "test")},
    }


@pytest.fixture(scope="module")
def table():
    dex = load_pokedex()
    return SpeciesTable(dex, load_type_chart(), ["hp", "atk", "def", "spa", "spd", "spe"],
                        load_aliases(dex))


def test_rows_line_up_with_split_files(run):
    splits, feats = run["splits"], run["feats"]
    assert sum(len(df) for df in splits.values()) == N_OLD + N_NEW
    for s in ("val", "test"):
        assert feats[s]["battle_id"].tolist() == splits[s]["battle_id"].tolist()
        assert feats[s][LABEL].tolist() == splits[s][LABEL].tolist()
        assert (feats[s]["swapped"] == 0).all()
    n = len(splits["train"])
    train = feats["train"]
    assert len(train) == 2 * n
    assert train["battle_id"].tolist() == splits["train"]["battle_id"].tolist() * 2


def test_train_second_half_is_the_swapped_copy(run):
    train = run["feats"]["train"]
    n = len(train) // 2
    cols = feature_columns(train.columns)
    first, second = train.iloc[:n], train.iloc[n:]
    assert np.array_equal(first[cols].to_numpy(), -second[cols].to_numpy())
    assert (first[LABEL].to_numpy() == 1 - second[LABEL].to_numpy()).all()
    assert (second["swapped"] == 1).all()


def test_same_columns_everywhere_and_no_nan(run):
    cols = feature_columns(run["feats"]["train"].columns)
    for df in run["feats"].values():
        assert feature_columns(df.columns) == cols
        assert np.isfinite(df[cols].to_numpy(float)).all()


def test_species_columns_come_from_train_only(run):
    cols = run["feats"]["train"].columns
    assert "species_flabebe" not in cols     # only in test battles
    assert "species_alcremie" in cols
    assert "species_alcremierubycream" not in cols  # cosmetic form, merged into alcremie
    assert "species_greattusk" in cols


def test_val_and_test_learned_scores_use_all_of_train(run, table):
    full = LearnedScores(run["cfg"]["features"]["learned_prior"]).fit(
        *table.teams(run["splits"]["train"]), run["splits"]["train"][LABEL].to_numpy())
    for s in ("val", "test"):
        expected = full.transform(*table.teams(run["splits"][s]))
        assert np.allclose(run["feats"][s][LEARNED_COLS].to_numpy(), expected, atol=1e-6)


def test_train_learned_scores_are_out_of_fold(run, table):
    fcfg, train = run["cfg"]["features"], run["splits"]["train"]
    expected = out_of_fold_scores(*table.teams(train), train[LABEL].to_numpy(),
                                  fcfg["learned_folds"], fcfg["learned_prior"], run["cfg"]["seed"])
    n = len(train)
    assert np.allclose(run["feats"]["train"][LEARNED_COLS].to_numpy()[:n], expected, atol=1e-6)


def test_val_and_test_labels_never_change_features(run, tmp_path):
    # Flip every val and test label and rebuild: no feature in any split may move.
    cfg = tmp_config(tmp_path)
    for name, df in run["splits"].items():
        df = df.copy()
        if name != "train":
            df[LABEL] = 1 - df[LABEL]
        df.to_parquet(tmp_path / "interim_dir" / f"{name}.parquet", index=False)
    build(cfg)
    for name, before in run["feats"].items():
        after = pd.read_parquet(tmp_path / "processed_dir" / f"{name}_features.parquet")
        cols = feature_columns(before.columns)
        assert np.array_equal(after[cols].to_numpy(), before[cols].to_numpy())


def test_load_features_on_pipeline_output(run):
    processed = run["cfg"]["data"]["processed_dir"]
    X, y = load_features("val", processed_dir=processed)
    assert list(X.index) == run["splits"]["val"]["battle_id"].tolist()
    assert X.index.name == "battle_id" and y.name == LABEL
    assert not {"swapped", LABEL, "battle_id"} & set(X.columns)
    X_teams, _ = load_features("val", drop_groups=["rating"], processed_dir=processed)
    assert list(X_teams.columns) == [c for c in X.columns if c != "rating_diff"]
