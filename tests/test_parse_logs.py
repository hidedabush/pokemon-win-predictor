"""Tests for src/data/parse_logs.py. The logs are trimmed copies of real Gen 9 OU logs."""

from src.data.parse_logs import checks_passed, parse_log, read_log

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
