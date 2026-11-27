"""Tests for the Pokédex loader and the feature builder (uses poke-env's bundled data)."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (FEATURE_GROUPS, LABEL, SpeciesTable, as_team,
                                         build_features, feature_columns, fit_species_vocab,
                                         species_features, swap_augment)
from src.features.pokedex import (TYPES, effectiveness_vs, load_pokedex, load_type_chart,
                                  missing_species)
from src.utils import ROOT, load_config

STATS = ["hp", "atk", "def", "spa", "spd", "spe"]


@pytest.fixture(scope="module")
def dex():
    return load_pokedex()


@pytest.fixture(scope="module")
def chart():
    return load_type_chart()


@pytest.fixture(scope="module")
def table(dex, chart):
    return SpeciesTable(dex, chart, STATS)


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(ROOT / "data/sample/battles_sample.csv")


# --- pokedex.py -------------------------------------------------------------------------


def test_pokedex_entry(dex):
    assert dex["greattusk"] == {"types": ["ground", "fighting"], "hp": 115, "atk": 131,
                                "def": 131, "spa": 53, "spd": 53, "spe": 87}
    assert dex["ogerponwellspring"]["types"] == ["grass", "water"]


def test_type_chart_is_attacker_first(chart):
    # poke-env stores it defender-first; these all break if the flip is lost
    assert chart["ground"]["flying"] == 0       # Ground can't hit Flying
    assert chart["flying"]["ground"] == 1
    assert chart["electric"]["water"] == 2
    assert chart["water"]["electric"] == 1
    assert chart["fighting"]["ghost"] == 0
    assert chart["ghost"]["normal"] == 0
    assert chart["normal"]["ghost"] == 0
    assert set(chart) == set(TYPES) and all(set(row) == set(TYPES) for row in chart.values())


def test_effectiveness_vs_dual_types(chart):
    assert effectiveness_vs("ice", ["dragon", "ground"], chart) == 4      # Garchomp
    assert effectiveness_vs("ground", ["fire", "flying"], chart) == 0     # Charizard
    assert effectiveness_vs("fire", ["water", "dragon"], chart) == 0.25   # Kingdra


def test_sample_species_and_known_forms_are_in_pokedex(dex, sample):
    seen = {s for col in ("p1_team", "p2_team") for t in sample[col] for s in as_team(t)}
    forms = ["zamazenta", "urshifurapidstrike", "ogerponcornerstone", "landorustherian",
             "samurotthisui", "rotomwash", "tinglu", "kommoo", "porygonz", "basculegionf"]
    assert missing_species(seen | set(forms), dex) == []
    assert missing_species(["notamon"], dex) == ["notamon"]


# --- build_features.py ------------------------------------------------------------------

A = ["greattusk", "kingambit", "gholdengo", "dragonite", "ironvaliant", "corviknight"]
B = ["garganacl", "ragingbolt", "samurotthisui", "ironmoth", "landorustherian", "gholdengo"]


def battles(rows):
    return pd.DataFrame([{"battle_id": f"b{i}", "upload_time": i, "p1_team": a, "p2_team": b,
                          "p1_rating": ra, "p2_rating": rb, "p1_win": y}
                         for i, (a, b, ra, rb, y) in enumerate(rows)])


def test_stat_features_are_mean_difference(table, dex):
    f = build_features(battles([(A, B, 1500, 1450, 1)]), table, vocab=[]).iloc[0]
    for s in STATS:
        expected = np.mean([dex[x][s] for x in A]) - np.mean([dex[x][s] for x in B])
        assert f[f"stats_{s}"] == pytest.approx(expected, abs=1e-3)
    bst = lambda team: np.mean([sum(dex[x][s] for s in STATS) for x in team])  # noqa: E731
    assert f["stats_bst"] == pytest.approx(bst(A) - bst(B), abs=1e-3)
    assert f["rating_diff"] == 50


def test_type_counts(table):
    f = build_features(battles([(A, B, 1500, 1500, 1)]), table, vocab=[]).iloc[0]
    # A: Steel on Kingambit, Gholdengo, Corviknight; B: Steel on Gholdengo only
    assert f["types_steel"] == 3 - 1
    # A: Fairy on Iron Valiant; B: none
    assert f["types_fairy"] == 1


def test_matchup_by_hand(table):
    # Garchomp (Dragon/Ground) vs Heatran (Fire/Steel): Ground hits Heatran 4x; Fire and
    # Steel are both neutral on Garchomp. So A covers 1, B covers 0.
    f = build_features(battles([(["garchomp"], ["heatran"], 1500, 1500, 1)]),
                       table, vocab=[]).iloc[0]
    assert f["match_cover"] == 1 and f["match_pairs"] == 1
    # Ground can't hit Flying: Garchomp vs Corviknight (Flying/Steel); Steel is neutral on
    # Garchomp, so neither side hits.
    f = build_features(battles([(["garchomp"], ["corviknight"], 1500, 1500, 1)]),
                       table, vocab=[]).iloc[0]
    assert f["match_cover"] == 0 and f["match_pairs"] == 0


def test_species_vocab_from_train_only_and_min_count():
    train_a = [["greattusk", "kingambit"], ["greattusk", "dondozo"]]
    train_b = [["kingambit", "gliscor"], ["clodsire", "gliscor"]]
    assert fit_species_vocab(train_a, train_b) == sorted(
        ["greattusk", "kingambit", "dondozo", "gliscor", "clodsire"])
    # kingambit is on both sides of battle 1, which counts as one battle, not two
    assert fit_species_vocab(train_a, train_b, min_count=2) == ["gliscor", "greattusk"]


def test_species_onehot_signs_mirror_and_unknown():
    vocab = ["gholdengo", "greattusk", "kingambit"]
    m = species_features([A], [B], vocab)
    # gholdengo is on both teams -> 0; greattusk A -> +1; kingambit A -> +1
    assert m.tolist() == [[0, 1, 1]]
    m = species_features([B], [A], vocab)
    assert m.tolist() == [[0, -1, -1]]
    # val/test species not in the train vocab are simply ignored
    assert species_features([["pikachu"]], [["eevee"]], vocab).tolist() == [[0, 0, 0]]


def test_swapping_teams_negates_every_feature(table, sample):
    vocab = fit_species_vocab([as_team(t) for t in sample.p1_team],
                              [as_team(t) for t in sample.p2_team])
    swapped_input = sample.rename(columns={"p1_team": "p2_team", "p2_team": "p1_team",
                                           "p1_rating": "p2_rating", "p2_rating": "p1_rating"})
    f = build_features(sample, table, vocab)
    g = build_features(swapped_input, table, vocab)
    cols = feature_columns(f.columns)
    np.testing.assert_allclose(g[cols].to_numpy(float), -f[cols].to_numpy(float))


def test_swap_augment(table, sample):
    f = build_features(sample, table, vocab=[])
    aug = swap_augment(f)
    n = len(f)
    assert len(aug) == 2 * n
    assert aug["swapped"].tolist() == [0] * n + [1] * n
    assert (aug[LABEL].iloc[n:].to_numpy() == 1 - f[LABEL].to_numpy()).all()
    cols = feature_columns(f.columns)
    np.testing.assert_allclose(aug[cols].iloc[n:].to_numpy(float), -f[cols].to_numpy(float))
    assert aug[LABEL].mean() == 0.5


def test_output_columns_on_sample(table, sample):
    vocab = fit_species_vocab([as_team(t) for t in sample.p1_team],
                              [as_team(t) for t in sample.p2_team])
    f = build_features(sample, table, vocab)
    assert f.columns[:2].tolist() == ["battle_id", "swapped"] and f.columns[-1] == LABEL
    groups = {c.split("_", 1)[0] for c in feature_columns(f.columns)}
    assert groups == set(FEATURE_GROUPS)
    assert "rating_diff" not in feature_columns(f.columns, drop_groups=["rating"])
    assert not f[feature_columns(f.columns)].isna().any().any()


def test_config_has_feature_settings():
    fcfg = load_config()["features"]
    assert fcfg["stat_names"] == STATS
    assert fcfg["species_min_count"] >= 1
