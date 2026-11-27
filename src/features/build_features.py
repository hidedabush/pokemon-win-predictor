"""Turn each battle into a feature row.

Input:  data/interim/{train,val,test}.parquet
Output: data/processed/{train,val,test}_features.parquet   (+ label column "p1_win")

Feature groups (keep the group prefix in column names so the ablation can drop a whole group):
    stats_*    team total/mean of each base stat, written as A − B
    types_*    count of each of the 18 types on each team, written as A − B
    match_*    type matchups: how many of B's Pokémon A can hit super-effectively
               (using A's STAB types), and the reverse, plus A − B
    species_*  one column per species: +1 if on team A, −1 if on team B, 0 otherwise
    rating_*   p1_rating − p2_rating   (only used in the "teams + ratings" model)

Swap augmentation (config features.swap_augment): for TRAIN only, add every battle again with
teams swapped and label flipped, so the model can't learn "p1 wins more".

TODO
- [ ] team_stat_features(), team_type_features(), matchup_features(), species_features()
- [ ] Fit the species vocabulary on TRAIN only (unknown species in val/test -> ignored)
- [ ] swap_augment() for the training set
- [ ] Save feature files; print number of columns per group
"""

from src.utils import ROOT, load_config  # noqa: F401

FEATURE_GROUPS = ["stats", "types", "match", "species", "rating"]


def build(cfg: dict) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    build(load_config())
