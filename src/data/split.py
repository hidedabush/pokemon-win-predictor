"""Time-based train / val / test split.        Owner: Alexander

Input:  data/interim/battles.parquet
Output: data/interim/{train,val,test}.parquet
        split sizes, date ranges and win rates are added to data/interim/data_stats.json

Sort by upload_time: oldest 70% -> train, next 15% -> val, newest 15% -> test.
A time split avoids leakage from the same players/teams showing up on both sides of the split
and mirrors real use (predict future battles from past ones).

Usage:
    python -m src.data.split
"""

from datetime import datetime, timezone

import pandas as pd

from src.data.parse_logs import update_stats
from src.utils import ROOT, load_config

SPLITS = ("train", "val", "test")
WIN_RATE_WARN = 0.02  # only for the printout: flag a split whose p1 win rate is >2 points off 50%


def _date(t: int) -> str:
    return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%d")


def split(cfg: dict) -> dict[str, pd.DataFrame]:
    fracs = cfg["split"]
    if abs(sum(fracs[s] for s in SPLITS) - 1.0) > 1e-9:
        raise ValueError(f"split fractions in config.yaml must sum to 1, got {fracs}")

    interim = ROOT / cfg["data"]["interim_dir"]
    df = pd.read_parquet(interim / "battles.parquet")
    # battle_id breaks ties between battles uploaded in the same second, so the cut is deterministic.
    df = df.sort_values(["upload_time", "battle_id"]).reset_index(drop=True)

    n = len(df)
    n_train = round(n * fracs["train"])
    n_val = round(n * fracs["val"])
    parts = {
        "train": df.iloc[:n_train],
        "val": df.iloc[n_train:n_train + n_val],
        "test": df.iloc[n_train + n_val:],
    }

    rows = []
    print(f"\n{'split':<7}{'battles':>9}  {'first':<12}{'last':<12}{'p1 win rate':>12}")
    for name, part in parts.items():
        part.reset_index(drop=True).to_parquet(interim / f"{name}.parquet", index=False)
        win = part["p1_win"].mean()
        row = {
            "split": name, "battles": len(part),
            "first": _date(part["upload_time"].min()), "last": _date(part["upload_time"].max()),
            "p1_win_rate": round(float(win), 4),
        }
        rows.append(row)
        flag = "  <-- more than 2 points off 50%" if abs(win - 0.5) > WIN_RATE_WARN else ""
        print(f"{name:<7}{len(part):>9,}  {row['first']:<12}{row['last']:<12}{win:>11.1%}{flag}")

    # No battle in an earlier split may be newer than any battle in a later one.
    for a, b in zip(SPLITS, SPLITS[1:]):
        last, first = parts[a]["upload_time"].max(), parts[b]["upload_time"].min()
        assert last <= first, f"{a} and {b} overlap in time"
        same_second = "same second" if last == first else "no overlap"
        print(f"{a} ends {_date(last)}, {b} starts {_date(first)}: {same_second}")

    update_stats(cfg, splits=rows)
    print(f"wrote {', '.join(f'{s}.parquet' for s in SPLITS)} -> {interim}")
    return parts


if __name__ == "__main__":
    split(load_config())
