# Data

Nothing in `raw/`, `interim/` or `processed/` is committed (see `.gitignore`).

## Source

- Replays: Hugging Face dataset `HolidayOugi/pokemon-showdown-replays`
  (https://huggingface.co/datasets/HolidayOugi/pokemon-showdown-replays)
  - Columns: `id, format, players, log, uploadtime, views, formatid, rating`
  - Filter: `formatid == "gen9ou"` (~2.5M replays; we sample ~100k)
  - No license stated: coursework use only, do not redistribute.
  - The dataset has one or more files per format. Gen 9 OU is only
    `[Gen 9] OU_part1.parquet` … `[Gen 9] OU_part13.parquet` (~6 GB of the ~73 GB), and
    `src/data/download.py` reads only those, streaming them with DuckDB (nothing big is saved).
  - Output: `data/raw/gen9ou_slim.parquet` with `battle_id, upload_time, players, rating, log_slim`,
    where `log_slim` keeps only the `|player|`, `|poke|` and `|win|` lines.
    Settings: `min_upload_date`, `max_rows` etc. in the `data:` section of `config.yaml`.

### How to run

Log in first, or Hugging Face rate-limits you (on the shared university network this happens
within minutes): run `hf auth login` and paste a free "Read" token from
https://huggingface.co/settings/tokens, or set `HF_TOKEN` (in Colab: add an `HF_TOKEN` secret).

```bash
python -m src.data.download --check     # cheap: list files, rows per formatid and date range
python -m src.data.download             # filter + slim -> data/raw/gen9ou_slim.parquet
python -m src.data.download --inspect   # print examples + the 3 checks below as percentages
```

If streaming keeps failing (network errors, rate limits), use
`python -m src.data.download --per-file`, which downloads one ~0.5 GB file at a time,
slims it, deletes it, and stops once `max_rows` is reached.
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
