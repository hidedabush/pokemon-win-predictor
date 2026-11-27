"""Fake raw logs and a config that points at a temp folder, so pipeline tests never read or write
the real data/ folder (which isn't in git)."""

import copy

import numpy as np
import pandas as pd

from src.utils import load_config

# Real Gen 9 species names as they appear in logs (forms, accents, cosmetic forms included).
SPECIES = [
    "Great Tusk", "Kingambit", "Gholdengo", "Dragonite", "Iron Valiant", "Corviknight",
    "Garganacl", "Raging Bolt", "Samurott-Hisui", "Iron Moth", "Landorus-Therian", "Heatran",
    "Rotom-Wash", "Ogerpon-Wellspring", "Dragapult", "Zamazenta", "Cinderace", "Alcremie",
    "Alcremie-Ruby-Cream", "Toxapex",
]


def slim_log(p1, p2, team1, team2, winner, r1=1500, r2=1500) -> str:
    """A log the way download.py slims it: |player|, |poke| and |win| lines only."""
    lines = [f"|player|p1|{p1}|1|{r1}", f"|player|p2|{p2}|2|{r2}"]
    lines += [f"|poke|p1|{s}, M|" for s in team1] + [f"|poke|p2|{s}|" for s in team2]
    if winner is not None:
        lines.append(f"|win|{winner}")
    return "\n".join(lines)


def random_battles(n: int, seed: int = 0, start: int = 1_750_000_000, first_id: int = 1000,
                   species=SPECIES, always=()):
    """n clean battles with random teams, ratings and winners, one minute apart.
    Species in `always` are put on every p1 team."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        team1, team2 = (list(rng.choice(species, 6, replace=False)) for _ in range(2))
        team1 = list(always) + [s for s in team1 if s not in always][:6 - len(always)]
        r1 = int(rng.integers(1300, 1700))
        r2 = int(np.clip(r1 + rng.integers(-100, 101), 1300, None))
        winner = "alice" if rng.random() < 0.5 else "bob"
        rows.append({"battle_id": f"gen9ou-{first_id + i}", "upload_time": start + 60 * i,
                     "log_slim": slim_log("alice", "bob", team1, team2, winner, r1, r2)})
    return rows


def tmp_config(tmp_path, **data_overrides) -> dict:
    """config.yaml with every data folder moved into tmp_path."""
    cfg = copy.deepcopy(load_config())
    for key in ("raw_dir", "interim_dir", "processed_dir"):
        (tmp_path / key).mkdir()
        cfg["data"][key] = str(tmp_path / key)  # ROOT / "/abs/path" is just "/abs/path"
    cfg["data"].update(data_overrides)
    return cfg


def write_raw(cfg: dict, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df["players"], df["rating"] = None, None  # columns the real file has but parse_logs skips
    df.to_parquet(f"{cfg['data']['raw_dir']}/{cfg['data']['slim_file']}", index=False)
