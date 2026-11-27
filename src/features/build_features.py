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

Type matchups use the type chart plus abilities that surely give an immunity (Levitate etc.,
see pokedex.multiplier). A Pokémon's STAB types are its own types.

    stats_*    stats_hp ... stats_spe   team mean of each base stat
               stats_bst                team mean base stat total
               stats_max_*, stats_min_* best and worst of each base stat on the team
                                        (stats_max_spe = the fastest Pokémon)
    types_*    types_normal ... types_fairy  number of team members of each type
               types_holes    attacking types that nothing on the team resists or is immune to
               types_stacked  most team members weak to one single attacking type
    match_*    cross-team matchups, each counted for A against B, minus B against A:
               match_cover    how many of their Pokémon at least one of ours hits super effectively
               match_pairs    how many of the 36 (ours, theirs) pairs we hit super effectively
               match_checks   how many of their Pokémon we have a resist-everything switch-in for
               match_effect   mean over the 36 pairs of log2(our best STAB multiplier), 0x -> -2
               match_outspeed how many of the 36 pairs we have the higher base Speed in
    species_*  one column per species: +1 on team A, −1 on team B, 0 otherwise (also 0 when both
               teams have it). Vocabulary comes from TRAIN only: species seen in fewer than
               features.species_min_count training battles get no column.
    learned_*  numbers fitted on the TRAIN battles (see LearnedScores):
               learned_species  mean win rate of the team's species, smoothed toward 50%
               learned_pairs    mean win rate of each (ours, theirs) species pair, smoothed
               learned_typical  how often the team's species are seen together (no label used)
               Train rows get out-of-fold values (fitted without their own fold), so a battle's
               own result never leaks into its features. Val/test use the fit on all of train.
    rating_*   rating_diff = p1_rating − p2_rating (only for the "teams + ratings" model)

Swap augmentation (config features.swap_augment): TRAIN only. Every battle is added again with
the teams swapped (features negated) and the label flipped, so the model can't learn
"player 1 wins more".

Species missing from the Pokédex are left out of the stats/types/match groups and listed when
you run this.
"""

from collections import Counter
from itertools import combinations, product

import numpy as np
import pandas as pd

from src.features.pokedex import (STATS, TYPES, load_pokedex, load_type_chart,
                                  missing_species, multiplier)
from src.utils import ROOT, load_config

FEATURE_GROUPS = ["stats", "types", "match", "species", "learned", "rating"]
SPLITS = ["train", "val", "test"]
LABEL = "p1_win"
MATCH_COLS = ["match_cover", "match_pairs", "match_checks", "match_effect", "match_outspeed"]
LEARNED_COLS = ["learned_species", "learned_pairs", "learned_typical"]


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


# --- Pokédex-based features ---------------------------------------------------------------


class SpeciesTable:
    """Per-species and per-pair numbers looked up from the Pokédex once, then reused."""

    def __init__(self, dex: dict, chart: dict, stat_names: list[str]):
        self.dex, self.chart, self.stat_names = dex, chart, stat_names
        self._defense: dict[str, np.ndarray] = {}
        self._best: dict[tuple[str, str], float] = {}

    def stat_vector(self, s: str) -> np.ndarray:
        """Base stats in stat_names order, then the base stat total."""
        e = self.dex[s]
        return np.array([e[n] for n in self.stat_names] + [sum(e[n] for n in STATS)], float)

    def type_vector(self, s: str) -> np.ndarray:
        types = self.dex[s]["types"]
        return np.array([t in types for t in TYPES], float)

    def defense(self, s: str) -> np.ndarray:
        """Multiplier of each of the 18 attacking types (TYPES order) against species s."""
        if s not in self._defense:
            self._defense[s] = np.array([multiplier(t, self.dex[s], self.chart) for t in TYPES])
        return self._defense[s]

    def best_stab(self, attacker: str, defender: str) -> float:
        """Best multiplier the attacker gets against the defender with its own types."""
        key = (attacker, defender)
        if key not in self._best:
            d = self.defense(defender)
            self._best[key] = max(d[TYPES.index(t)] for t in self.dex[attacker]["types"])
        return self._best[key]

    def hits(self, attacker: str, defender: str) -> bool:
        return self.best_stab(attacker, defender) > 1

    def walls(self, wall: str, attacker: str) -> bool:
        """True if `wall` resists (or is immune to) every one of the attacker's own types."""
        d = self.defense(wall)
        return all(d[TYPES.index(t)] <= 0.5 for t in self.dex[attacker]["types"])

    def speed(self, s: str) -> float:
        return self.dex[s]["spe"]


def team_stat_features(team, table: SpeciesTable) -> np.ndarray:
    """[mean of each stat and BST, max of each stat, min of each stat]."""
    known = [s for s in team if s in table.dex]
    k = len(table.stat_names)
    if not known:
        return np.zeros(3 * k + 1)
    m = np.array([table.stat_vector(s) for s in known])
    return np.concatenate([m.mean(axis=0), m[:, :k].max(axis=0), m[:, :k].min(axis=0)])


def team_type_features(team, table: SpeciesTable) -> np.ndarray:
    """[count of each type, holes, stacked]."""
    known = [s for s in team if s in table.dex]
    if not known:
        return np.zeros(len(TYPES) + 2)
    counts = np.sum([table.type_vector(s) for s in known], axis=0)
    d = np.array([table.defense(s) for s in known])        # members x attacking types
    holes = np.sum((d > 0.5).all(axis=0))
    stacked = (d >= 2).sum(axis=0).max()
    return np.concatenate([counts, [holes, stacked]])


def matchup_features(team_a, team_b, table: SpeciesTable) -> np.ndarray:
    """MATCH_COLS, each A − B."""
    a = [s for s in team_a if s in table.dex]
    b = [s for s in team_b if s in table.dex]

    def one_side(ours, theirs):
        pairs = list(product(ours, theirs))
        if not pairs:
            return np.zeros(len(MATCH_COLS))
        cover = sum(any(table.hits(x, y) for x in ours) for y in theirs)
        n_hit = sum(table.hits(x, y) for x, y in pairs)
        checks = sum(any(table.walls(x, y) for x in ours) for y in theirs)
        effect = np.mean([np.log2(max(table.best_stab(x, y), 0.25)) for x, y in pairs])
        outspeed = sum(table.speed(x) > table.speed(y) for x, y in pairs)
        return np.array([cover, n_hit, checks, effect, outspeed], float)

    return one_side(a, b) - one_side(b, a)


# --- species one-hot ------------------------------------------------------------------------


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


# --- learned scores -------------------------------------------------------------------------


class LearnedScores:
    """Win rates and co-occurrence counted on training battles.

    Every battle is counted from both sides, so a species' or pair's rate from B's point of view
    is exactly 1 − the rate from A's point of view, which keeps every column A − B. Rates are
    smoothed toward 50% with `prior` imaginary battles, so a pair seen twice stays near 50%.
    """

    def __init__(self, prior: float = 20):
        self.prior = prior

    def fit(self, teams_a, teams_b, y) -> "LearnedScores":
        self.sp_w, self.sp_n = Counter(), Counter()
        self.pair_w, self.pair_n = Counter(), Counter()
        self.together = Counter()
        for a, b, win in zip(teams_a, teams_b, y):
            for team, won in ((a, win), (b, 1 - win)):
                for s in team:
                    self.sp_w[s] += won
                    self.sp_n[s] += 1
                for pair in combinations(sorted(team), 2):
                    self.together[pair] += 1
            for x, z in product(a, b):
                self.pair_w[x, z] += win
                self.pair_n[x, z] += 1
                self.pair_w[z, x] += 1 - win
                self.pair_n[z, x] += 1
        return self

    def _rate(self, wins: float, n: int) -> float:
        return (wins + 0.5 * self.prior) / (n + self.prior) - 0.5

    def _one_team(self, team) -> tuple[float, float]:
        species = np.mean([self._rate(self.sp_w[s], self.sp_n[s]) for s in team])
        typical = np.mean([np.log1p(self.together[p]) for p in combinations(sorted(team), 2)]
                          or [0.0])
        return species, typical

    def transform(self, teams_a, teams_b) -> np.ndarray:
        """(n, 3) array in LEARNED_COLS order."""
        out = np.zeros((len(teams_a), len(LEARNED_COLS)))
        for i, (a, b) in enumerate(zip(teams_a, teams_b)):
            (sa, ta), (sb, tb) = self._one_team(a), self._one_team(b)
            pairs = np.mean([self._rate(self.pair_w[x, z], self.pair_n[x, z])
                             for x, z in product(a, b)])
            out[i] = [sa - sb, pairs, ta - tb]
        return out


def out_of_fold_scores(teams_a, teams_b, y, folds: int, prior: float, seed: int) -> np.ndarray:
    """LearnedScores for training rows, each fitted without the row's own fold."""
    n = len(teams_a)
    fold = np.random.default_rng(seed).permutation(n) % max(1, min(folds, n))
    out = np.zeros((n, len(LEARNED_COLS)))
    for k in np.unique(fold):
        fit_idx, pred_idx = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        scores = LearnedScores(prior).fit([teams_a[i] for i in fit_idx],
                                          [teams_b[i] for i in fit_idx], y[fit_idx])
        out[pred_idx] = scores.transform([teams_a[i] for i in pred_idx],
                                         [teams_b[i] for i in pred_idx])
    return out


# --- putting it together --------------------------------------------------------------------


def build_features(battles: pd.DataFrame, table: SpeciesTable, vocab: list[str],
                   learned: np.ndarray | None = None) -> pd.DataFrame:
    """One feature row per battle (no augmentation). `learned` is the (n, 3) array from
    LearnedScores.transform or out_of_fold_scores; without it the learned_* columns are left
    out."""
    n = len(battles)
    teams_a = [as_team(t) for t in battles["p1_team"]]
    teams_b = [as_team(t) for t in battles["p2_team"]]

    stats = np.array([team_stat_features(a, table) - team_stat_features(b, table)
                      for a, b in zip(teams_a, teams_b)]).reshape(n, -1)
    types = np.array([team_type_features(a, table) - team_type_features(b, table)
                      for a, b in zip(teams_a, teams_b)]).reshape(n, -1)
    match = np.array([matchup_features(a, b, table)
                      for a, b in zip(teams_a, teams_b)]).reshape(n, -1)
    rating = (battles["p1_rating"] - battles["p2_rating"]).fillna(0).to_numpy(float)

    names = table.stat_names
    stat_cols = ([f"stats_{s}" for s in names] + ["stats_bst"]
                 + [f"stats_max_{s}" for s in names] + [f"stats_min_{s}" for s in names])
    parts = [
        pd.DataFrame({"battle_id": battles["battle_id"].to_numpy(), "swapped": np.int8(0)}),
        pd.DataFrame(stats.astype(np.float32), columns=stat_cols),
        pd.DataFrame(types.astype(np.float32),
                     columns=[f"types_{t}" for t in TYPES] + ["types_holes", "types_stacked"]),
        pd.DataFrame(match.astype(np.float32), columns=MATCH_COLS),
        pd.DataFrame(species_features(teams_a, teams_b, vocab),
                     columns=[f"species_{s}" for s in vocab]),
    ]
    if learned is not None:
        parts.append(pd.DataFrame(np.asarray(learned, np.float32), columns=LEARNED_COLS))
    parts.append(pd.DataFrame({"rating_diff": rating.astype(np.float32),
                               LABEL: battles[LABEL].to_numpy(np.int8)}))
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
    splits = {name: pd.read_parquet(interim / f"{name}.parquet").reset_index(drop=True)
              for name in SPLITS}

    dex = load_pokedex()
    table = SpeciesTable(dex, load_type_chart(), fcfg["stat_names"])
    seen = {s for df in splits.values() for col in ("p1_team", "p2_team")
            for team in df[col] for s in as_team(team)}
    missing = missing_species(seen, dex)
    print(f"{len(seen)} distinct species; {len(missing)} not in the Pokédex {missing or ''}")

    train = splits["train"]
    train_a = [as_team(t) for t in train["p1_team"]]
    train_b = [as_team(t) for t in train["p2_team"]]
    train_y = train[LABEL].to_numpy()

    vocab = []
    if fcfg["use_species_onehot"]:
        vocab = fit_species_vocab(train_a, train_b, fcfg.get("species_min_count", 1))

    prior, folds = fcfg.get("learned_prior", 20), fcfg.get("learned_folds", 5)
    full_fit = LearnedScores(prior).fit(train_a, train_b, train_y)

    for name, df in splits.items():
        if df[["p1_rating", "p2_rating"]].isna().any(axis=None):
            print(f"warning: {name} has missing ratings; rating_diff set to 0 for those rows")
        if name == "train":
            learned = out_of_fold_scores(train_a, train_b, train_y, folds, prior, cfg["seed"])
        else:
            learned = full_fit.transform([as_team(t) for t in df["p1_team"]],
                                         [as_team(t) for t in df["p2_team"]])
        feats = build_features(df, table, vocab, learned)
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
