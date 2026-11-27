"""Simple baselines. Each returns P(p1 wins) for every row.

    coin_flip       always 0.5
    higher_bst      pick the team with higher total base stats    (sign of stats_bst)
    more_type_adv   pick the team with more type advantages      (sign of match_cover)
    higher_rating   pick the higher-rated player                 (sign of rating_diff)

All feature columns are p1 − p2, so > 0 favours p1. See src/features/build_features.py.

Hard 0/1 picks are fine for accuracy, but for log loss use e.g. 0.6/0.4 instead of 1/0
(a confident wrong 0 or 1 makes log loss infinite). Say which you used in the report.

TODO
- [ ] Implement the four functions on the feature DataFrame
- [ ] Ties -> 0.5
"""

import numpy as np
import pandas as pd


def coin_flip(X: pd.DataFrame) -> np.ndarray:
    return np.full(len(X), 0.5)


def higher_bst(X: pd.DataFrame) -> np.ndarray:
    raise NotImplementedError


def more_type_adv(X: pd.DataFrame) -> np.ndarray:
    raise NotImplementedError


def higher_rating(X: pd.DataFrame) -> np.ndarray:
    raise NotImplementedError


BASELINES = {
    "coin_flip": coin_flip,
    "higher_bst": higher_bst,
    "more_type_adv": more_type_adv,
    "higher_rating": higher_rating,
}
