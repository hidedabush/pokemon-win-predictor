"""Load Gen 9 Pokédex (types, base stats) and the type chart.

Source: Pokémon Showdown data, easiest through poke-env's built-in Gen 9 data.
Cite both in report/references.bib.

TODO
- [ ] load_pokedex() -> {species_id: {"types": [...], "hp": .., "atk": .., ...}}
      Keys must use src.utils.to_id() so they match the battle data.
- [ ] load_type_chart() -> effectiveness(attacking_type, defending_type) -> 0, 0.5, 1, 2
- [ ] effectiveness_vs(attack_type, defender_types) multiplies over both defender types
- [ ] Check every species in the real battle data is found; print any that aren't
      (forms like Ogerpon-Wellspring, Urshifu-Rapid-Strike are the usual problem)
"""

TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground",
    "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy",
]


def load_pokedex() -> dict:
    raise NotImplementedError


def load_type_chart() -> dict:
    raise NotImplementedError


def effectiveness_vs(attack_type: str, defender_types: list[str], chart: dict) -> float:
    raise NotImplementedError
