"""Time-based train / val / test split.        Owner: Alexander

Input:  data/interim/battles.parquet
Output: data/interim/{train,val,test}.parquet

Sort by upload_time: oldest 70% -> train, next 15% -> val, newest 15% -> test.
A time split avoids leakage from the same players/teams showing up on both sides of the split
and mirrors real use (predict future battles from past ones).

TODO
- [ ] Sort by upload_time, cut by the fractions in config
- [ ] Print size and date range of each split (goes in the Data section)
- [ ] Print the p1 win rate in each split (should be close to 50%)
"""

from src.utils import ROOT, load_config  # noqa: F401


def split(cfg: dict) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    split(load_config())
