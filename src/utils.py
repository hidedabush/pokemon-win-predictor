"""Shared helpers. Everyone imports from here so names and settings stay consistent."""

import re
import unicodedata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    """Load config.yaml as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def to_id(name: str) -> str:
    """Showdown-style ID: lowercase, letters and digits only.

    'Great Tusk' -> 'greattusk', 'Urshifu-Rapid-Strike' -> 'urshifurapidstrike'
    Accents are dropped first, so 'Flabébé' -> 'flabebe' (the Pokédex key), not 'flabb'.
    Use this for every species name so the battle data and the Pokédex match.
    """
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", name).lower())
