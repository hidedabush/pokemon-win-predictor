# Data

Nothing in `raw/`, `interim/` or `processed/` is committed (see `.gitignore`).

## Source

- Replays: Hugging Face dataset `HolidayOugi/pokemon-showdown-replays`
  (https://huggingface.co/datasets/HolidayOugi/pokemon-showdown-replays)
  - Columns: `id, format, players, log, uploadtime, views, formatid, rating`
  - Filter: `formatid == "gen9ou"` (~2.5M replays; we sample ~100k)
  - No license stated: coursework use only, do not redistribute.
- Pokédex + type chart: Pokémon Showdown data (`data/pokedex.ts`, `data/typechart.ts`),
  most easily loaded through poke-env's built-in Gen 9 data.
- Backup if log parsing fails: `jakegrigsby/metamon-parsed-replays` (cc-by-nc-4.0).

## Shared battle format (the contract between data → features)

`data/interim/battles.parquet`, one row per battle:

| column | type | meaning |
|---|---|---|
| `battle_id` | str | replay id, e.g. `gen9ou-2101234567` |
| `upload_time` | int | unix seconds (used for the time-based split) |
| `p1_team` | list[str] | 6 species IDs, lowercase, no spaces/punctuation (`greattusk`, `kingambit`) |
| `p2_team` | list[str] | same, for player 2 |
| `p1_rating` | float or null | player 1's ladder rating before the battle |
| `p2_rating` | float or null | player 2's ladder rating before the battle |
| `p1_win` | int | 1 if player 1 won, 0 if player 2 won |

Rows are dropped if: either team isn't exactly 6 Pokémon, there's no `|win|` line (tie/forfeit
issues), or the ratings fail the filter in `config.yaml`.

Species IDs must match the keys of the Pokédex Andy loads (use the same `to_id()` helper in
`src/utils.py` everywhere). Watch out for forms at team preview, e.g. `Urshifu-*`, `Ogerpon-*`.

`data/sample/battles_sample.csv` is a tiny **fake** file in this format so features and
models can be built before the real data is ready. Replace it with real rows once parsing works.

## Things to check first (Alexander)

On 5–10 real Gen 9 OU logs, confirm:
1. `|poke|p1|...` and `|poke|p2|...` lines exist (team preview).
2. Each player's rating can be found (usually in `|player|p1|name|avatar|rating`).
3. Every battle has a `|win|<name>` line.

Write what you find here.
