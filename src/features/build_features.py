"""Turn each battle into a feature row.

Input:  data/interim/{train,val,test}.parquet
Output: data/processed/{train,val,test}_features.parquet

Every output file has these columns:
    battle_id  replay id (train has each id twice when swap augmentation is on)
    swapped    1 if this is a swap-augmented copy (train only), else 0
    <features> see below; pick them with feature_columns(), never by position
    p1_win     the label

Team A is player 1 and team B is player 2. EVERY feature is written as A − B, so swapping the
two sides is exactly the same as negating every feature column (tests/test_features.py checks
this).

    stats_*    mean of each base stat over the team (stats_hp ... stats_spe) and the mean base
               stat total (stats_bst)
    types_*    number of Pokémon of each of the 18 types on the team
    match_*    STAB type matchups (a Pokémon "hits" another if one of its own types is super
               effective against it):
                 match_cover  how many of the other team's Pokémon at least one of ours hits
                 match_pairs  how many of the 36 (ours, theirs) pairs we hit
    species_*  one column per species: +1 on team A, −1 on team B, 0 otherwise (also 0 when both
               teams have it). Vocabulary comes from TRAIN only: species seen in fewer than
               features.species_min_count training battles get no column.
    rating_*   rating_diff = p1_rating − p2_rating (only for the "teams + ratings" model)

Swap augmentation (config features.swap_augment): TRAIN only. Every battle is added again with
the teams swapped (features negated) and the label flipped, so the model can't learn
"player 1 wins more".

Species missing from the Pokédex are left out of stats/types/match and listed when you run this.
"""

from collections import Counter
from itertools import product

import numpy as np
import pandas as pd

from src.features.pokedex import (STATS, TYPES, effectiveness_vs, load_pokedex,
                                  load_type_chart, missing_species)
from src.utils import ROOT, load_config

FEATURE_GROUPS = ["stats", "types", "match", "species", "rating"]
SPLITS = ["train", "val", "test"]
LABEL = "p1_win"


def feature_columns(columns, drop_groups=()) -> list[str]:
    """Feature column names, optionally without some groups.

    feature_columns(df.columns)                        all features
    feature_columns(df.columns, drop_groups=["rating"]) teams only
    """
    keep = [g for g in FEATURE_GROUPS if g not in drop_groups]
    return [c for c in columns if c.split("_", 1)[0] in keep]


def as_team(team) -> list[str]:
    """Team from parquet (list/array of IDs) or from the sample CSV ("a|b|c|d|e|f")."""
    return team.split("|") if isinstance(team, str) else list(team)


class SpeciesTable:
    """Per-species numbers looked up from the Pokédex once, then reused for every battle."""

    def __init__(self, dex: dict, chart: dict, stat_names: list[str]):
        self.dex, self.chart, self.stat_names = dex, chart, stat_names
        self._hits: dict[tuple[str, str], bool] = {}

    def stat_vector(self, s: str) -> np.ndarray:
        e = self.dex[s]
        return np.array([e[n] for n in self.stat_names] + [sum(e[n] for n in STATS)], float)

    def type_vector(self, s: str) -> np.ndarray:
        types = self.dex[s]["types"]
        return np.array([t in types for t in TYPES], float)

    def hits(self, attacker: str, defender: str) -> bool:
        """True if one of the attacker's own types is super effective against the defender."""
        key = (attacker, defender)
        if key not in self._hits:
            dfn = self.dex[defender]["types"]
            self._hits[key] = any(effectiveness_vs(t, dfn, self.chart) > 1
                                  for t in self.dex[attacker]["types"])
        return self._hits[key]


def team_stat_features(team, table: SpeciesTable) -> np.ndarray:
    known = [s for s in team if s in table.dex]
    if not known:
        return np.zeros(len(table.stat_names) + 1)
    return np.mean([table.stat_vector(s) for s in known], axis=0)


def team_type_features(team, table: SpeciesTable) -> np.ndarray:
    return sum((table.type_vector(s) for s in team if s in table.dex), np.zeros(len(TYPES)))


def matchup_features(team_a, team_b, table: SpeciesTable) -> np.ndarray:
    """[match_cover, match_pairs], each A − B."""
    a = [s for s in team_a if s in table.dex]
    b = [s for s in team_b if s in table.dex]

    def one_side(ours, theirs):
        hit = {(x, y): table.hits(x, y) for x, y in product(ours, theirs)}
        cover = sum(any(hit[x, y] for x in ours) for y in theirs)
        return np.array([cover, sum(hit.values())], float)

    return one_side(a, b) - one_side(b, a)


def fit_species_vocab(teams_a, teams_b, min_count: int = 1) -> list[str]:
    """Species seen in at least min_count battles (either side). Call on TRAIN only."""
    counts = Counter(s for a, b in zip(teams_a, teams_b) for s in set(a) | set(b))
    return sorted(s for s, n in counts.items() if n >= min_count)


def species_features(teams_a, teams_b, vocab: list[str]) -> np.ndarray:
    """int8 matrix: +1 if on team A, −1 if on team B (so 0 if on both or neither)."""
    index = {s: j for j, s in enumerate(vocab)}
    out = np.zeros((len(teams_a), len(vocab)), np.int8)
    for i, (a, b) in enumerate(zip(teams_a, teams_b)):
        for s in a:
            if s in index:
                out[i, index[s]] += 1
        for s in b:
            if s in index:
                out[i, index[s]] -= 1
    return out


def build_features(battles: pd.DataFrame, table: SpeciesTable, vocab: list[str]) -> pd.DataFrame:
    """One feature row per battle (no augmentation)."""
    teams_a = [as_team(t) for t in battles["p1_team"]]
    teams_b = [as_team(t) for t in battles["p2_team"]]

    stats = np.array([team_stat_features(a, table) - team_stat_features(b, table)
                      for a, b in zip(teams_a, teams_b)]).reshape(len(battles), -1)
    types = np.array([team_type_features(a, table) - team_type_features(b, table)
                      for a, b in zip(teams_a, teams_b)]).reshape(len(battles), -1)
    match = np.array([matchup_features(a, b, table)
                      for a, b in zip(teams_a, teams_b)]).reshape(len(battles), -1)
    rating = (battles["p1_rating"] - battles["p2_rating"]).fillna(0).to_numpy(float)

    stat_cols = [f"stats_{n}" for n in table.stat_names] + ["stats_bst"]
    parts = [
        pd.DataFrame({"battle_id": battles["battle_id"].to_numpy(), "swapped": np.int8(0)}),
        pd.DataFrame(stats.astype(np.float32), columns=stat_cols),
        pd.DataFrame(types.astype(np.float32), columns=[f"types_{t}" for t in TYPES]),
        pd.DataFrame(match.astype(np.float32), columns=["match_cover", "match_pairs"]),
        pd.DataFrame(species_features(teams_a, teams_b, vocab),
                     columns=[f"species_{s}" for s in vocab]),
        pd.DataFrame({"rating_diff": rating.astype(np.float32),
                      LABEL: battles[LABEL].to_numpy(np.int8)}),
    ]
    return pd.concat(parts, axis=1)


def swap_augment(feats: pd.DataFrame) -> pd.DataFrame:
    """Add every row again with the teams swapped: features negated, label flipped."""
    swapped = feats.copy()
    cols = feature_columns(feats.columns)
    swapped[cols] = -swapped[cols]
    swapped["swapped"] = np.int8(1)
    swapped[LABEL] = (1 - swapped[LABEL]).astype(np.int8)
    return pd.concat([feats, swapped], ignore_index=True)


def build(cfg: dict) -> None:
    fcfg = cfg["features"]
    interim, processed = ROOT / cfg["data"]["interim_dir"], ROOT / cfg["data"]["processed_dir"]
    processed.mkdir(parents=True, exist_ok=True)
    splits = {name: pd.read_parquet(interim / f"{name}.parquet") for name in SPLITS}

    dex = load_pokedex()
    table = SpeciesTable(dex, load_type_chart(), fcfg["stat_names"])
    seen = {s for df in splits.values() for col in ("p1_team", "p2_team")
            for team in df[col] for s in as_team(team)}
    missing = missing_species(seen, dex)
    print(f"{len(seen)} distinct species; {len(missing)} not in the Pokédex {missing or ''}")

    train = splits["train"]
    vocab = []
    if fcfg["use_species_onehot"]:
        vocab = fit_species_vocab([as_team(t) for t in train["p1_team"]],
                                  [as_team(t) for t in train["p2_team"]],
                                  fcfg.get("species_min_count", 1))

    for name, df in splits.items():
        if df[["p1_rating", "p2_rating"]].isna().any(axis=None):
            print(f"warning: {name} has missing ratings; rating_diff set to 0 for those rows")
        feats = build_features(df.reset_index(drop=True), table, vocab)
        if name == "train" and fcfg["swap_augment"]:
            feats = swap_augment(feats)
        out = processed / f"{name}_features.parquet"
        feats.to_parquet(out, index=False)
        print(f"{name}: {len(feats):,} rows -> {out.relative_to(ROOT)}")

    cols = feature_columns(feats.columns)
    counts = Counter(c.split("_", 1)[0] for c in cols)
    print("columns per group:", ", ".join(f"{g} {counts[g]}" for g in FEATURE_GROUPS),
          f"(total {len(cols)})")


if __name__ == "__main__":
    build(load_config())
