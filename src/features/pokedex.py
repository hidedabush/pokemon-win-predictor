"""Load Gen 9 Pokédex (types, base stats) and the type chart.

Source: Pokémon Showdown's data files, read through poke-env's bundled Gen 9 data (no network
needed). Cite both in report/references.bib.

    dex = load_pokedex()          {"greattusk": {"types": ["ground", "fighting"],
                                                 "abilities": ["protosynthesis"], "hp": 115, ...}}
    chart = load_type_chart()     chart[attacking_type][defending_type] -> 0, 0.5, 1 or 2
    effectiveness_vs("ground", ["fire", "flying"], chart) -> 0.0
    multiplier("ground", dex["rotomwash"], chart) -> 0.0   (types AND abilities, e.g. Levitate)

Watch out: poke-env stores its chart the other way round (defending type first, upper case),
so load_type_chart() flips it. tests/test_features.py checks the direction.

Run `python -m src.features.pokedex` to check that every species in
data/interim/battles.parquet is in the Pokédex.
"""

from functools import lru_cache

from src.utils import ROOT, load_config, to_id

TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground",
    "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy",
]
STATS = ["hp", "atk", "def", "spa", "spd", "spe"]

# Abilities that make a Pokémon immune to one attacking type. Team preview doesn't show which
# ability a Pokémon has, so multiplier() only uses one when EVERY ability the species can have
# gives the immunity (Rotom-Wash: Levitate only). That misses e.g. Heatran's usual Flash Fire,
# but never invents an immunity.
IMMUNITY_ABILITIES = {
    "levitate": "ground", "eartheater": "ground",
    "waterabsorb": "water", "stormdrain": "water", "dryskin": "water",
    "voltabsorb": "electric", "lightningrod": "electric", "motordrive": "electric",
    "flashfire": "fire", "wellbakedbody": "fire",
    "sapsipper": "grass",
}


@lru_cache(maxsize=1)
def _gen9():
    from poke_env.data import GenData  # imported here so the rest of the repo works without it

    return GenData.from_gen(9)


def load_pokedex() -> dict:
    """{species_id: {"types": [1-2 lowercase types], "abilities": [ability IDs],
                     "hp": .., "atk": .., ..., "spe": ..}}.

    Keys go through to_id() so they match the battle data (they already do in poke-env).
    """
    dex = {}
    for key, entry in _gen9().pokedex.items():
        stats = entry["baseStats"]
        dex[to_id(key)] = {"types": [t.lower() for t in entry["types"]],
                           "abilities": [to_id(a) for a in entry["abilities"].values()],
                           **{s: stats[s] for s in STATS}}
    return dex


def load_type_chart() -> dict:
    """chart[attacking_type][defending_type] = damage multiplier (0, 0.5, 1 or 2)."""
    by_defender = _gen9().type_chart  # by_defender["FLYING"]["GROUND"] == 0
    return {atk: {dfn: float(by_defender[dfn.upper()][atk.upper()]) for dfn in TYPES}
            for atk in TYPES}


def effectiveness_vs(attack_type: str, defender_types: list[str], chart: dict) -> float:
    """Multiplier of one attacking type against a (one- or two-type) Pokémon."""
    result = 1.0
    for t in defender_types:
        result *= chart[attack_type][t]
    return result


def ability_immunity(entry: dict) -> str | None:
    """The type this species is surely immune to through its ability, if any."""
    types = {IMMUNITY_ABILITIES.get(a) for a in entry["abilities"]}
    return types.pop() if len(types) == 1 and None not in types else None


def multiplier(attack_type: str, entry: dict, chart: dict) -> float:
    """Damage multiplier of one attacking type against a Pokédex entry (types + abilities)."""
    if attack_type == ability_immunity(entry):
        return 0.0
    return effectiveness_vs(attack_type, entry["types"], chart)


def missing_species(species_ids, dex: dict) -> list[str]:
    """Species IDs that aren't Pokédex keys, sorted."""
    return sorted(set(species_ids) - set(dex))


def main():
    import pandas as pd

    cfg = load_config()
    path = ROOT / cfg["data"]["interim_dir"] / "battles.parquet"
    df = pd.read_parquet(path, columns=["p1_team", "p2_team"])
    seen = {s for col in ("p1_team", "p2_team") for team in df[col] for s in team}
    missing = missing_species(seen, load_pokedex())
    print(f"{len(seen)} distinct species in {path.name}; {len(missing)} not in the Pokédex")
    for s in missing:
        print("  missing:", s)


if __name__ == "__main__":
    main()
