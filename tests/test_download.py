"""Tests for src/data/download.py that need no network."""

import duckdb

from src.data.download import glob_escape, min_upload_time, slim_sql

FAKE_LOG = "\n".join(
    [
        "|j|☆alice",
        "|player|p1|alice|2|1581",
        "|player|p2|bob|lady-gen4|1622",
        "|teamsize|p1|6",
        "|gen|9",
        "|poke|p1|Great Tusk|",
        "|poke|p1|Ogerpon-Cornerstone, F|",
        "|poke|p2|Kingambit, M|",
        "|c|☆bob|gg |win|bob",  # chat text must not count as a |win| line
        "|start",
        "|switch|p1a: Great Tusk|Great Tusk|100/100",
        "|win|alice",
        "|player|p2|",
    ]
)


def slim(log: str) -> str:
    """Run the same SQL expression the download uses, on one string."""
    return duckdb.connect().execute(f"SELECT {slim_sql('$log')}", {"log": log}).fetchone()[0]


def test_slim_keeps_only_player_poke_win_lines():
    assert slim(FAKE_LOG).split("\n") == [
        "|player|p1|alice|2|1581",
        "|player|p2|bob|lady-gen4|1622",
        "|poke|p1|Great Tusk|",
        "|poke|p1|Ogerpon-Cornerstone, F|",
        "|poke|p2|Kingambit, M|",
        "|win|alice",
        "|player|p2|",
    ]


def test_slim_handles_windows_line_endings_and_empty_log():
    assert slim("|poke|p1|Gholdengo|\r\n|win|bob\r\n") == "|poke|p1|Gholdengo|\n|win|bob"
    assert slim("|start\n|turn|1") == ""


def test_glob_escape_makes_brackets_literal():
    assert glob_escape("[Gen 9] OU_part1.parquet") == "[[]Gen 9] OU_part1.parquet"


def test_min_upload_time_is_utc_midnight():
    assert min_upload_time({"data": {"min_upload_date": "2025-06-01"}}) == 1748736000
    assert min_upload_time({"data": {"min_upload_date": None}}) == 0
