"""Tests for src/data/parse_logs.py. The logs are trimmed copies of real Gen 9 OU logs."""

import json

import pandas as pd

from src.data.parse_logs import build_battles, checks_passed, parse_log, read_log
from tests.fake_data import random_battles, slim_log, tmp_config, write_raw

P1_POKES = [
    "|poke|p1|Kilowattrel, M|",
    "|poke|p1|Glimmora, F|",
    "|poke|p1|Tinkaton, F|",
    "|poke|p1|Meowscarada, F|",
    "|poke|p1|Skeledirge, M|",
    "|poke|p1|Quaquaval, M|",
]
P2_POKES = [
    "|poke|p2|Samurott-Hisui, F|",
    "|poke|p2|Heatran, M|",
    "|poke|p2|Latios, M|",
    "|poke|p2|Iron Valiant|",
    "|poke|p2|Iron Treads|",
    "|poke|p2|Ogerpon-Wellspring, F|",
]
PLAYERS = ["|player|p1|Raijin Zerachu|volkner|1506", "|player|p2|Swankentora|n|1460"]


def log(*lines: str) -> str:
    return "\n".join(lines)


def test_clean_battle():
    battle = parse_log(log(*PLAYERS, *P1_POKES, *P2_POKES, "|win|Swankentora"))
    assert battle == {
        "p1_team": ["kilowattrel", "glimmora", "tinkaton", "meowscarada", "skeledirge", "quaquaval"],
        "p2_team": ["samurotthisui", "heatran", "latios", "ironvaliant", "irontreads", "ogerponwellspring"],
        "p1_rating": 1506.0,
        "p2_rating": 1460.0,
        "p1_win": 0,
    }


def test_trailing_empty_player_lines_are_ignored():
    # Players leaving add empty |player| lines; a rejoin can repeat the name without a rating.
    battle = parse_log(log(
        *PLAYERS, *P1_POKES, *P2_POKES, "|win|Raijin Zerachu",
        "|player|p1|", "|player|p2|Swankentora|n|", "|player|p2|",
    ))
    assert battle["p1_win"] == 1
    assert (battle["p1_rating"], battle["p2_rating"]) == (1506.0, 1460.0)


def test_chat_line_cannot_fake_a_winner():
    # Full (unslimmed) log: the chat message contains "|win|" but isn't a |win| line.
    battle = parse_log(log(
        "|j|☆Swankentora", *PLAYERS, "|gen|9", *P1_POKES, *P2_POKES,
        "|c|☆Swankentora|gg |win|Swankentora", "|start", "|win|Raijin Zerachu",
    ))
    assert battle["p1_win"] == 1


def test_five_pokemon_is_unusable():
    assert parse_log(log(*PLAYERS, *P1_POKES[:5], *P2_POKES, "|win|Swankentora")) is None


def test_missing_rating_is_none_and_fails_rating_check():
    # Tournament/unrated battles: the rating field after the avatar is empty.
    text = log("|player|p1|alz01|piers|", "|player|p2|CrispyFries2|101|",
               *P1_POKES, *P2_POKES, "|win|CrispyFries2")
    battle = parse_log(text)
    assert battle["p1_rating"] is None and battle["p2_rating"] is None
    assert checks_passed(read_log(text), min_rating=1300, max_gap=100) == 3  # stops at "ratings present"


def test_unknown_or_ambiguous_winner_is_unusable():
    assert parse_log(log(*PLAYERS, *P1_POKES, *P2_POKES)) is None  # no |win| (tie)
    assert parse_log(log(*PLAYERS, *P1_POKES, *P2_POKES, "|win|Guest 4951730")) is None
    same = ["|player|p1|alice|2|1500", "|player|p2|alice|2|1500"]
    assert parse_log(log(*same, *P1_POKES, *P2_POKES, "|win|alice")) is None


def passed(*lines: str) -> int:
    return checks_passed(read_log(log(*lines)), min_rating=1300, max_gap=100)


def test_checks_passed_stops_at_each_filter():
    teams = (*P1_POKES, *P2_POKES)
    assert passed(*P1_POKES, *P2_POKES) == 0                                     # no players
    assert passed(*PLAYERS, *P1_POKES[:5], *P2_POKES, "|win|Swankentora") == 1  # 5 Pokémon
    assert passed(*PLAYERS, *teams) == 2                                       # no winner
    assert passed("|player|p1|a|1|", "|player|p2|b|1|1500", *teams, "|win|a") == 3
    assert passed("|player|p1|a|1|1299", "|player|p2|b|1|1350", *teams, "|win|a") == 4
    assert passed("|player|p1|a|1|1300", "|player|p2|b|1|1401", *teams, "|win|a") == 5
    assert passed(*PLAYERS, *teams, "|win|Swankentora") == 6


def test_rating_and_gap_limits_are_inclusive():
    teams = (*P1_POKES, *P2_POKES)
    assert passed("|player|p1|a|1|1300", "|player|p2|b|1|1400", *teams, "|win|b") == 6
    assert passed("|player|p1|a|1|1400", "|player|p2|b|1|1300", *teams, "|win|b") == 6


def test_rating_that_is_not_a_number_is_none():
    fields = read_log(log("|player|p1|a|1|abc", "|player|p2|b|1|1500.5", *P1_POKES, *P2_POKES))
    assert fields["p1_rating"] is None and fields["p2_rating"] == 1500.5


def test_accented_and_apostrophe_names_match_pokedex_ids():
    teams = ["Flabébé-Blue", "Farfetch’d", "Type: Null", "Mr. Mime-Galar", "Porygon-Z", "Kommo-o"]
    battle = parse_log(slim_log("a", "b", teams, teams, "a"))
    assert battle["p1_team"] == ["flabebeblue", "farfetchd", "typenull", "mrmimegalar",
                                 "porygonz", "kommoo"]


def test_build_battles_filters_counts_and_samples(tmp_path):
    cfg = tmp_config(tmp_path, sample_size=30)
    good = random_battles(40)
    team = ["Great Tusk", "Kingambit", "Gholdengo", "Dragonite", "Iron Valiant", "Corviknight"]
    bad = [  # one battle failing each filter, in filter order
        "|player|p1||1|1500",
        slim_log("a", "b", team[:5], team, "a"),
        slim_log("a", "b", team, team, None),
        slim_log("a", "b", team, team, "a", r1=""),
        slim_log("a", "b", team, team, "a", r1=1200, r2=1250),
        slim_log("a", "b", team, team, "a", r1=1300, r2=1500),
    ]
    rows = good + [{"battle_id": f"gen9ou-{i}", "upload_time": 1_700_000_000 + i, "log_slim": text}
                   for i, text in enumerate(bad)]
    write_raw(cfg, rows)
    df = build_battles(cfg)

    assert len(df) == 30 and df["battle_id"].is_unique
    assert set(df["battle_id"]) <= {r["battle_id"] for r in good}
    assert df["upload_time"].is_monotonic_increasing
    assert list(df.columns) == ["battle_id", "upload_time", "p1_team", "p2_team",
                                "p1_rating", "p2_rating", "p1_win"]
    saved = pd.read_parquet(f"{cfg['data']['interim_dir']}/battles.parquet")
    for col in ("p1_team", "p2_team"):  # parquet gives the teams back as numpy arrays
        saved[col] = saved[col].map(list)
    assert saved.equals(df)

    stats = json.loads((tmp_path / "interim_dir/data_stats.json").read_text())
    assert [s["battles"] for s in stats["drop_counts"]] == [46, 45, 44, 43, 42, 41, 40]
    assert stats["sample"] == {"clean_battles": 40, "sampled": 30, "seed": cfg["seed"],
                               "limit": None}


def test_build_battles_sample_ignores_raw_row_order(tmp_path):
    # The raw file can come back in a different row order; the sample must not change.
    cfg = tmp_config(tmp_path, sample_size=20)
    rows = random_battles(50)
    write_raw(cfg, rows)
    first = build_battles(cfg)
    write_raw(cfg, rows[::-1])
    assert build_battles(cfg).equals(first)


def test_build_battles_lists_forms(tmp_path):
    cfg = tmp_config(tmp_path)
    write_raw(cfg, random_battles(30))
    build_battles(cfg)
    forms = json.loads((tmp_path / "interim_dir/data_stats.json").read_text())["form_species"]
    assert forms["samurotthisui"] == ["Samurott-Hisui"]
    assert "greattusk" not in forms  # no "-" in the name
