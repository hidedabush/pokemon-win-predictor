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

Species IDs must match the keys of the Pokédex loaded in `src/features/pokedex.py` (use the
same `to_id()` helper in `src/utils.py` everywhere). Watch out for forms at team preview, e.g. `Urshifu-*`, `Ogerpon-*`.

`data/sample/battles_sample.csv` is a tiny **fake** file in this format, used by the tests and
for trying code without the real data. Keep it fake: real rows come from the unlicensed replay
dataset, so they must never be committed.

## Things to check first

Checked on all 1,190,747 logs in `data/raw/gen9ou_slim.parquet` (run 11/27). The same numbers
are written to `data/interim/data_stats.json` (not committed) for the report's Data section.

### What the real logs look like

```
|player|p1|Raijin Zerachu|volkner|1506      side | name | avatar | rating
|player|p2|Swankentora|n|1460
|poke|p1|Kilowattrel, M|                     species, gender (only the species is kept)
|poke|p2|Samurott-Hisui, F|                  -> samurotthisui
|win|Swankentora
|player|p1|                                  player left the room
```

- **`|player|` lines:** ~75% of battles have 2 per side (the real one + an empty one when the
  player leaves); some have dozens (reconnects). The **first** line per side holds the name and
  rating; 59 battles have a later line with a different spelling of the name, so later lines
  are ignored. 4 battles have no name on a side's first line (dropped).
- **Ratings:** taken from the `|player|` line, not the dataset's `rating` column. That column is
  one number per battle; it tracks the players' ratings (correlation 0.996) but is usually
  ~25 below their average (e.g. 1483 vs. 1506/1460, 1237 vs. 1278/1261), and one number can't
  give a rating gap. Tournament (`smogtours-`), `radicalred-`, `rom-` and unrated battles have
  an empty rating field.
- **Winner:** every `|win|` name matches one of the two players except 2 battles (a third name,
  e.g. a guest), and 247 battles have no `|win|` line (ties). No battle has two players with the
  same name.
- **Forms:** species are written with their form (`Ogerpon-Wellspring`, `Landorus-Therian`).
  Forms hidden at team preview are written with `-*`: `Zamazenta-*`, `Greninja-*`,
  `Dudunsparce-*` (and `Urshifu-*`, `Zacian-*`, `Silvally-*`, `Gourgeist-*`, which appear only
  in unrated battles and are all dropped). `to_id` turns `-*` into the base ID (`zamazenta`).
- **Species not legal in OU** (Megas, Arceus, Calyrex-Ice/Shadow, Kyurem-Black/White, `-Sevii`
  forms; 132 battles) come from unrated or `radicalred-` battles and none pass the rating filter.

### Drop counts (`python -m src.data.parse_logs`)

Applied in this order; `min_rating` 1300 and `max_rating_gap` 100 from `config.yaml`.

| step | battles | dropped |
|---|---:|---:|
| total rows | 1,190,747 | |
| parsed OK (both players named) | 1,190,743 | 4 |
| exactly 6 Pokémon per side | 1,179,001 | 11,742 |
| has a winner | 1,178,752 | 249 |
| both ratings present | 1,147,643 | 31,109 |
| both ≥ 1300 | 580,386 | 567,257 |
| \|gap\| ≤ 100 | 533,490 | 46,896 |
| random sample (seed 42) | **100,000** | |

The sample is drawn uniformly from the whole date range, so every month is represented and
val/test still cover the most recent months.

### Time-based split (`python -m src.data.split`)

| split | battles | first | last | p1 win rate |
|---|---:|---|---|---:|
| train | 70,000 | 2025-06-01 | 2026-04-08 | 49.8% |
| val | 15,000 | 2026-04-08 | 2026-07-01 | 49.9% |
| test | 15,000 | 2026-07-01 | 2026-10-05 | 49.7% |

Dates are UTC. The splits don't overlap in time (sorted by `upload_time`; a split can end on the
same day, or even second, the next one starts; `battle_id` breaks ties). All win rates are
within 1 point of 50%.

### Species IDs with a form in the final data (for the Pokédex keys)

Every distinct ID in `battles.parquet` whose original name contained `-` (105 IDs; the full
`id -> original name` map is `form_species` in `data_stats.json`). **All 105 are keys in
poke-env's Gen 9 Pokédex** (checked 11/27); `python -m src.features.pokedex` re-checks every
species in `battles.parquet`.

- **Hidden form, mapped to the base species:** `zamazenta` (Zamazenta-\*), `greninja`
  (Greninja-\*), `dudunsparce` (Dudunsparce-\*).
- **Battle forms / regional forms** (different types or stats; should be Pokédex keys):
  `arcaninehisui`, `articunogalar`, `avalugghisui`, `basculegionf`, `basculinbluestriped`,
  `basculinwhitestriped`, `braviaryhisui`, `decidueyehisui`, `deoxysdefense`, `deoxysspeed`,
  `dugtrioalola`, `electrodehisui`, `enamorustherian`, `exeggutoralola`, `geodudealola`,
  `golemalola`, `goodrahisui`, `graveleralola`, `hoopaunbound`, `indeedeef`, `keldeoresolute`,
  `kommoo`, `landorustherian`, `lilliganthisui`, `lycanrocdusk`, `lycanrocmidnight`,
  `mausholdfour`, `meowsticf`, `moltresgalar`, `mukalola`, `ninetalesalola`,
  `ogerponcornerstone`, `ogerponwellspring`, `oinkolognef`, `oricoriopau`, `oricoriopompom`,
  `oricoriosensu`, `persianalola`, `pikachualola`, `pikachukalos`, `pikachuoriginal`,
  `pikachusinnoh`, `pikachuworld`, `porygonz`,
  `qwilfishhisui`, `raichualola`, `rotomfan`, `rotomfrost`, `rotomheat`, `rotommow`,
  `rotomwash`, `samurotthisui`, `sandshrewalola`, `sandslashalola`, `sliggoohisui`,
  `slowbrogalar`, `slowkinggalar`, `sneaselhisui`, `squawkabillyblue`, `squawkabillywhite`,
  `taurospaldeaaqua`, `taurospaldeablaze`, `taurospaldeacombat`,
  `thundurustherian`, `tinglu`, `tornadustherian`, `toxtricitylowkey`, `typhlosionhisui`,
  `weezinggalar`, `wochien`, `zapdosgalar`, `zarudedada`, `zoroarkhisui`,
  `zoruahisui`
- **Cosmetic forms** (same types, base stats and abilities as the base species). They are kept
  as-is in `battles.parquet`; `build_features` merges them into the base species
  (`pokedex.load_aliases()`) so e.g. all Alcremie flavours share one species column:
  `alcremiecaramelswirl`, `alcremielemoncream`, `alcremiematchacream`, `alcremiemintcream`,
  `alcremierainbowswirl`, `alcremierubycream`, `alcremierubyswirl`, `alcremiesaltedcream`,
  `florgesblue`, `florgesorange`, `florgeswhite`, `florgesyellow`,
  `gastrodoneast`, `miniorblue`, `miniorgreen`, `miniorindigo`, `miniororange`,
  `miniorviolet`, `minioryellow`, `polteageistantique`, `sawsbuckautumn`, `sawsbucksummer`,
  `sawsbuckwinter`, `sinistchamasterpiece`, `tatsugiridroopy`, `tatsugiristretchy`,
  `vivillonfancy`, `vivillonpokeball`

### Features on the real data (`python -m src.features.build_features`, run 11/27)

599 distinct species, all in the Pokédex. 450 feature columns: stats 19, types 20, match 5,
species 402 (seen in ≥ 20 train battles), learned 3, rating 1. Train has 140,000 rows
(70,000 battles + their swapped copies); val and test 15,000 each. No missing values and no
constant columns.

How well each feature predicts the winner on its own (ROC AUC on val; 0.5 = no signal):
`learned_pairs` 0.547, `rating_diff` 0.532, `learned_species` 0.531, `match_effect` 0.525,
`match_checks` 0.522; every other feature is between 0.485 and 0.518. The learned scores
score no better on train (out-of-fold: 0.539, 0.522) than on val, so no label leaks in.

A quick, untuned logistic regression (standardised, C = 0.01) gets on val:

| features | accuracy | log loss | AUC |
|---|---:|---:|---:|
| teams only | 53.6% | 0.6905 | 0.553 |
| teams + rating | 54.1% | 0.6893 | 0.558 |
| teams only, without learned_* | 53.4% | 0.6905 | 0.547 |
| learned_* only | 53.3% | 0.6903 | 0.545 |

So team preview alone carries a small but real signal. Expect results in the mid-50s; much
higher on val/test would point to a leak. `rating_diff` is weak partly because the rating
filter keeps only battles with |gap| ≤ 100.
