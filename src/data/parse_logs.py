"""Turn raw Showdown logs into one clean row per battle.

Input:  data/raw/gen9ou_slim.parquet  (made by src/data/download.py)
        columns: battle_id, upload_time, players, rating, log_slim
        log_slim = only the |player|, |poke| and |win| lines of the log, joined by "\n"
Output: data/interim/battles.parquet  (format defined in data/README.md)
        data/interim/data_stats.json  (drop counts + species forms, for the report)

Lines in log_slim:
    |player|p1|<name>|<avatar>|<rating>
    |poke|p1|<Species>, L<level>, <gender>|      (team preview, 6 per side)
    |win|<name>
Real logs also end with empty "|player|p1|" / "|player|p2|" lines (players leaving),
so use the first |player| line for each side.

Ratings come from the |player| lines, not the dataset's `rating` column: that column is one
number per battle (usually ~25 below the players' average), so it can't give a rating gap.

Usage:
    python -m src.data.parse_logs              # all battles -> battles.parquet
    python -m src.data.parse_logs --limit 1000 # quick dev run on the first 1,000 rows
"""

import argparse
import json

import pandas as pd

from src.utils import ROOT, load_config, to_id

SIDES = ("p1", "p2")
TEAM_SIZE = 6  # team preview always shows six Pokémon per side in OU


def _rating(field: str) -> float | None:
    """Rating text from a |player| line -> float, or None if empty or not a number."""
    try:
        return float(field)
    except ValueError:
        return None


def read_log(log: str) -> dict | None:
    """Collect the raw facts from a log without judging them.

    Returns names, ratings, species names (as written, e.g. "Ogerpon-Wellspring") and the
    winner's name, or None if either side has no usable first |player| line.
    Only whole lines starting with |player|, |poke| or |win| count, so a chat message
    such as "|c|bob|gg |win|bob" can't fake a winner.
    """
    names: dict[str, str] = {}
    ratings: dict[str, float | None] = {}
    species: dict[str, list[str]] = {side: [] for side in SIDES}
    winner = None

    for line in log.splitlines():
        parts = line.split("|")  # "|player|p1|alice|2|1581" -> ["", "player", "p1", "alice", "2", "1581"]
        if len(parts) < 3 or parts[0] != "":
            continue
        kind, side = parts[1], parts[2]
        if kind == "player" and side in SIDES and side not in names:
            # First |player| line per side is the real one; later ones are leave/rejoin noise.
            names[side] = parts[3] if len(parts) > 3 else ""
            ratings[side] = _rating(parts[5]) if len(parts) > 5 else None
        elif kind == "poke" and side in SIDES and len(parts) > 3:
            species[side].append(parts[3].split(",")[0].strip())  # "Kingambit, M" -> "Kingambit"
        elif kind == "win" and winner is None:
            winner = line[len("|win|"):]

    if not all(names.get(side) for side in SIDES):
        return None
    return {
        "p1_name": names["p1"], "p2_name": names["p2"],
        "p1_rating": ratings["p1"], "p2_rating": ratings["p2"],
        "p1_species": species["p1"], "p2_species": species["p2"],
        "winner": winner,
    }


def has_full_teams(fields: dict) -> bool:
    return len(fields["p1_species"]) == TEAM_SIZE and len(fields["p2_species"]) == TEAM_SIZE


def p1_win(fields: dict) -> int | None:
    """1 if p1 won, 0 if p2 won, None if the winner can't be told apart (no |win| line,
    a name matching neither player, or both players with the same name)."""
    p1, p2, winner = fields["p1_name"], fields["p2_name"], fields["winner"]
    if winner is None or p1 == p2:
        return None
    if winner == p1:
        return 1
    if winner == p2:
        return 0
    return None


def parse_log(log: str) -> dict | None:
    """Parse one battle log. Return None if the battle can't be used.

    Returns p1_team, p2_team (6 species IDs each), p1_rating, p2_rating (float or None)
    and p1_win (0/1). Rating filters are applied later, in build_battles().
    """
    fields = read_log(log)
    if fields is None or not has_full_teams(fields):
        return None
    win = p1_win(fields)
    if win is None:
        return None
    return {
        "p1_team": [to_id(s) for s in fields["p1_species"]],
        "p2_team": [to_id(s) for s in fields["p2_species"]],
        "p1_rating": fields["p1_rating"],
        "p2_rating": fields["p2_rating"],
        "p1_win": win,
    }


def checks_passed(fields: dict | None, min_rating: float, max_gap: float) -> int:
    """How many of the filters below a battle passes, in order (stops at the first failure)."""
    if fields is None:
        return 0
    if not has_full_teams(fields):
        return 1
    if p1_win(fields) is None:
        return 2
    r1, r2 = fields["p1_rating"], fields["p2_rating"]
    if r1 is None or r2 is None:
        return 3
    if min(r1, r2) < min_rating:
        return 4
    if abs(r1 - r2) > max_gap:
        return 5
    return 6


def stats_path(cfg: dict):
    return ROOT / cfg["data"]["interim_dir"] / "data_stats.json"


def update_stats(cfg: dict, **sections) -> None:
    """Merge sections into data/interim/data_stats.json (numbers for the report)."""
    path = stats_path(cfg)
    stats = json.loads(path.read_text()) if path.exists() else {}
    stats.update(sections)
    path.write_text(json.dumps(stats, indent=2))


def build_battles(cfg: dict, limit: int | None = None) -> pd.DataFrame:
    """Parse every raw log, apply the filters, sample, and write battles.parquet."""
    d = cfg["data"]
    min_rating, max_gap = d["min_rating"], d["max_rating_gap"]
    raw = pd.read_parquet(ROOT / d["raw_dir"] / d["slim_file"],
                          columns=["battle_id", "upload_time", "log_slim"])
    if limit:
        raw = raw.head(limit)

    steps = [
        "total rows",
        "parsed OK",
        f"exactly {TEAM_SIZE} Pokémon per side",
        "has a winner",
        "both ratings present",
        f"both >= {min_rating}",
        f"|gap| <= {max_gap}",
    ]
    reached = [0] * len(steps)  # reached[k] = battles that passed the first k checks
    rows = []
    for battle_id, upload_time, log in zip(raw["battle_id"], raw["upload_time"], raw["log_slim"]):
        fields = read_log(log)
        passed = checks_passed(fields, min_rating, max_gap)
        for k in range(passed + 1):
            reached[k] += 1
        if passed == len(steps) - 1:
            rows.append({
                "battle_id": str(battle_id),
                "upload_time": int(upload_time),
                "p1_team": [to_id(s) for s in fields["p1_species"]],
                "p2_team": [to_id(s) for s in fields["p2_species"]],
                "p1_rating": float(fields["p1_rating"]),
                "p2_rating": float(fields["p2_rating"]),
                "p1_win": p1_win(fields),
                "_names": fields["p1_species"] + fields["p2_species"],  # for the forms list only
            })

    drops = [
        {"step": s, "battles": n, "dropped": (reached[k - 1] - n) if k else 0}
        for k, (s, n) in enumerate(zip(steps, reached))
    ]
    print(f"\n{'step':<30}{'battles':>12}{'dropped':>10}")
    for r in drops:
        print(f"{r['step']:<30}{r['battles']:>12,}{r['dropped']:>10,}")

    # Sort first: the raw file's row order can differ between downloads, and the sample below
    # must be the same for everyone who runs this with the same seed.
    df = pd.DataFrame(rows).sort_values(["upload_time", "battle_id"]).reset_index(drop=True)
    n = min(d["sample_size"], len(df))
    # Sample uniformly over the whole date range (not the newest n), so after the time split
    # val/test still cover the most recent months and train the oldest.
    df = df.sample(n=n, random_state=cfg["seed"]).sort_values(["upload_time", "battle_id"])
    df = df.reset_index(drop=True)
    print(f"sampled {n:,} of {len(rows):,} clean battles (seed {cfg['seed']})")

    # Species whose name had a form ("-"), so the Pokédex keys can be checked against them.
    forms: dict[str, set[str]] = {}
    for names in df["_names"]:
        for name in names:
            if "-" in name:
                forms.setdefault(to_id(name), set()).add(name)
    df = df.drop(columns="_names")

    out = ROOT / d["interim_dir"] / "battles.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df):,} battles -> {out}")

    update_stats(
        cfg,
        drop_counts=drops,
        sample={"clean_battles": len(rows), "sampled": n, "seed": cfg["seed"], "limit": limit},
        form_species={k: sorted(v) for k, v in sorted(forms.items())},
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, help="only parse the first N raw rows (dev runs)")
    build_battles(load_config(), limit=parser.parse_args().limit)
