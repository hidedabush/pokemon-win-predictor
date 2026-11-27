"""Tests for the Pokédex loader and the feature builder (uses poke-env's bundled data)."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (FEATURE_GROUPS, LABEL, LearnedScores, SpeciesTable,
                                         as_team, build_features, feature_columns,
                                         fit_species_vocab, out_of_fold_scores,
                                         species_features, swap_augment)
from src.features.pokedex import (TYPES, effectiveness_vs, load_pokedex, load_type_chart,
                                  missing_species, multiplier)
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
    assert dex["greattusk"] == {"types": ["ground", "fighting"], "abilities": ["protosynthesis"],
                                "hp": 115, "atk": 131, "def": 131, "spa": 53, "spd": 53,
                                "spe": 87}
    assert dex["ogerponwellspring"]["types"] == ["grass", "water"]


def test_ability_immunity_only_when_sure(dex, chart):
    assert multiplier("ground", dex["rotomwash"], chart) == 0       # Levitate is its only ability
    assert multiplier("electric", dex["rotomwash"], chart) == 1
    # Heatran may have Flash Fire OR Flame Body, so no sure Fire immunity: Fire stays neutral
    # (0.5x on Fire, 2x on Steel); Ground is 4x
    assert multiplier("fire", dex["heatran"], chart) == 1
    assert multiplier("ground", dex["heatran"], chart) == 4


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


def teams(df):
    return [as_team(t) for t in df.p1_team], [as_team(t) for t in df.p2_team]


def learned_on(df):
    return LearnedScores().fit(*teams(df), df[LABEL].to_numpy())


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


def test_stat_max_and_min(table, dex):
    f = build_features(battles([(A, B, 1500, 1500, 1)]), table, vocab=[]).iloc[0]
    for s in STATS:
        assert f[f"stats_max_{s}"] == max(dex[x][s] for x in A) - max(dex[x][s] for x in B)
        assert f[f"stats_min_{s}"] == min(dex[x][s] for x in A) - min(dex[x][s] for x in B)


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


def test_matchup_checks_effect_and_speed(table):
    # Garchomp (spe 102) vs Heatran (spe 77). Garchomp's best STAB on Heatran is Ground 4x
    # (log2 = 2); Heatran's best on Garchomp is 1x (log2 = 0). Neither resists all of the other's
    # types (Heatran is 2x weak to Ground; Garchomp takes Fire and Steel neutral).
    f = build_features(battles([(["garchomp"], ["heatran"], 1500, 1500, 1)]),
                       table, vocab=[]).iloc[0]
    assert f["match_effect"] == 2 and f["match_outspeed"] == 1 and f["match_checks"] == 0
    # Heatran (Fire/Steel) resists both Flying (0.5x) and Steel (0.25x), so it checks
    # Corviknight; Corviknight takes Fire 2x, so it doesn't check Heatran. Heatran's best STAB
    # is Fire 2x (log2 = 1); Corviknight's best is Flying 0.5x (log2 = -1). Speed 77 vs 67.
    f = build_features(battles([(["heatran"], ["corviknight"], 1500, 1500, 1)]),
                       table, vocab=[]).iloc[0]
    assert f["match_checks"] == 1 and f["match_effect"] == 2 and f["match_outspeed"] == 1
    # Levitate counts: Tusk's Ground does 0x to Rotom-Wash, so Tusk's best is Fighting 1x
    # (log2 = 0); Rotom's Water hits Tusk 2x (log2 = 1)
    f = build_features(battles([(["rotomwash"], ["greattusk"], 1500, 1500, 1)]),
                       table, vocab=[]).iloc[0]
    assert f["match_effect"] == 1


def test_type_holes_and_stacked(table):
    # Great Tusk (Ground/Fighting) resists Poison, Bug, Rock, Dark and is immune to Electric: 13
    f = build_features(battles([(["greattusk"], ["rotomwash"], 1500, 1500, 1)]),
                       table, vocab=[]).iloc[0]
    tusk = 18 - 5
    # Rotom-Wash resists Fire, Water, Ice, Steel, Flying and is immune to Ground (Levitate): 12
    rotom = 18 - 6
    assert f["types_holes"] == tusk - rotom
    # Tusk is weak to Water, Grass, Ice, Psychic, Flying, Fairy (1 each); Rotom to Grass only
    assert f["types_stacked"] == 1 - 1
    two_weak = build_features(battles([(["garchomp", "landorustherian"], ["greattusk"],
                                        1500, 1500, 1)]), table, vocab=[]).iloc[0]
    assert two_weak["types_stacked"] == 2 - 1      # both weak to Ice (4x)


def test_learned_scores_by_hand():
    # x beat y twice; with prior 2 the smoothed rate of x is (2 + 1) / (2 + 2) = 0.75
    s = LearnedScores(prior=2).fit([["x"], ["x"]], [["y"], ["y"]], np.array([1, 1]))
    sa, pairs, typical = s.transform([["x"]], [["y"]])[0]
    assert sa == pytest.approx(0.25 - (-0.25)) and pairs == pytest.approx(0.25)
    # reversed order gives the exact negative
    assert s.transform([["y"]], [["x"]])[0] == pytest.approx([-sa, -pairs, -typical])
    # unseen species sit at 50% -> 0
    assert s.transform([["new"]], [["other"]])[0].tolist() == [0, 0, 0]


def test_learned_typical_counts_teammates():
    s = LearnedScores().fit([["a", "b"], ["a", "b"]], [["c", "d"], ["c", "e"]], np.array([1, 0]))
    # (a, b) seen together twice, (c, e) once
    assert s.transform([["a", "b"]], [["c", "e"]])[0][2] == pytest.approx(np.log1p(2) - np.log1p(1))


def test_out_of_fold_never_uses_own_battle():
    # Every battle is a different, unique pair of species: out of fold, nothing was ever seen,
    # so every score is exactly 0. A plain fit on all battles would leak the label.
    teams_a = [[f"a{i}"] for i in range(10)]
    teams_b = [[f"b{i}"] for i in range(10)]
    y = np.array([1] * 10)
    assert np.all(out_of_fold_scores(teams_a, teams_b, y, folds=5, prior=1, seed=0) == 0)
    assert np.all(LearnedScores(1).fit(teams_a, teams_b, y).transform(teams_a, teams_b)[:, 0] > 0)


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
    learned = learned_on(sample)
    f = build_features(sample, table, vocab, learned.transform(*teams(sample)))
    g = build_features(swapped_input, table, vocab, learned.transform(*teams(swapped_input)))
    assert set(FEATURE_GROUPS) <= {c.split("_", 1)[0] for c in f.columns}
    cols = feature_columns(f.columns)
    np.testing.assert_allclose(g[cols].to_numpy(float), -f[cols].to_numpy(float))


def test_swap_augment(table, sample):
    learned = learned_on(sample).transform(*teams(sample))
    f = build_features(sample, table, vocab=[], learned=learned)
    aug = swap_augment(f)
    n = len(f)
    assert len(aug) == 2 * n
    assert aug["swapped"].tolist() == [0] * n + [1] * n
    assert (aug[LABEL].iloc[n:].to_numpy() == 1 - f[LABEL].to_numpy()).all()
    cols = feature_columns(f.columns)
    np.testing.assert_allclose(aug[cols].iloc[n:].to_numpy(float), -f[cols].to_numpy(float))
    assert aug[LABEL].mean() == 0.5


def test_output_columns_on_sample(table, sample):
    vocab = fit_species_vocab(*teams(sample))
    oof = out_of_fold_scores(*teams(sample), sample[LABEL].to_numpy(), folds=5, prior=20, seed=0)
    f = build_features(sample, table, vocab, oof)
    assert f.columns[:2].tolist() == ["battle_id", "swapped"] and f.columns[-1] == LABEL
    groups = {c.split("_", 1)[0] for c in feature_columns(f.columns)}
    assert groups == set(FEATURE_GROUPS)
    assert "rating_diff" not in feature_columns(f.columns, drop_groups=["rating"])
    assert not f[feature_columns(f.columns)].isna().any().any()


def test_config_has_feature_settings():
    fcfg = load_config()["features"]
    assert fcfg["stat_names"] == STATS
    assert fcfg["species_min_count"] >= 1
    assert fcfg["learned_folds"] >= 2 and fcfg["learned_prior"] > 0
