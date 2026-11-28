"""Simple baselines returning P(player 1 wins) for each battle."""

import numpy as np
import pandas as pd


def coin_flip(X: pd.DataFrame) -> np.ndarray:
    return np.full(len(X), 0.5)


def _pick_by_sign(values: pd.Series) -> np.ndarray:
    # Start with ties at 0.5, then favor the side with the larger value.
    probabilities = np.full(len(values), 0.5)
    probabilities[values.to_numpy() > 0] = 0.6
    probabilities[values.to_numpy() < 0] = 0.4
    return probabilities


def higher_bst(X: pd.DataFrame) -> np.ndarray:
    return _pick_by_sign(X["stats_bst"])


def more_type_adv(X: pd.DataFrame) -> np.ndarray:
    return _pick_by_sign(X["match_cover"])


def higher_rating(X: pd.DataFrame) -> np.ndarray:
    return _pick_by_sign(X["rating_diff"])


BASELINES = {
    "coin_flip": coin_flip,
    "higher_bst": higher_bst,
    "more_type_adv": more_type_adv,
    "higher_rating": higher_rating,
}
