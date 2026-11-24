"""Turn raw Showdown logs into one clean row per battle.        Owner: Alexander

Input:  data/raw/gen9ou_slim.parquet  (made by src/data/download.py)
        columns: battle_id, upload_time, players, rating, log_slim
        log_slim = only the |player|, |poke| and |win| lines of the log, joined by "\n"
Output: data/interim/battles.parquet  (format defined in data/README.md)

Lines in log_slim:
    |player|p1|<name>|<avatar>|<rating>
    |poke|p1|<Species>, L<level>, <gender>|      (team preview, 6 per side)
    |win|<name>
Real logs also end with empty "|player|p1|" / "|player|p2|" lines (players leaving),
so use the first |player| line for each side.

TODO
- [ ] parse_log(): return dict with p1_team, p2_team, p1_rating, p2_rating, p1_win (or None if unusable)
- [ ] Species: keep the part before the first comma, then src.utils.to_id()
- [ ] Map the |win| name back to p1/p2 using the |player| lines
- [ ] Drop: teams != 6 Pokémon, no winner, missing ratings
- [ ] Apply rating filter from config (min_rating, max_rating_gap)
- [ ] Keep battle_id and upload_time from the raw columns
- [ ] Print counts dropped at each step (useful for the Data section of the report)
"""

from src.utils import ROOT, load_config, to_id  # noqa: F401


def parse_log(log: str) -> dict | None:
    """Parse one battle log. Return None if the battle can't be used."""
    raise NotImplementedError


def build_battles(cfg: dict) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    build_battles(load_config())
