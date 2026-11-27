# Predicting Who Wins a Pokémon Battle

COSC 4368 Final Project: Alexander Mayer, Andy Nguyen, Nguyen Hang, Ruel Cabading

Given both six-Pokémon teams at team preview (Gen 9 OU), predict the probability that each side wins.

New here? Read [project_overview.txt](project_overview.txt) first (motivation,
problem definition, methodology).

**Internal deadline:** Mon Nov 30 · **Real deadline:** Fri Dec 11, 11:59pm

## Pipeline

```
raw replays (Hugging Face, parquet)
   │  src/data/download.py
   ▼
data/raw/gen9ou_slim.parquet     (1.19M battles, only the log lines we need)
   │  src/data/parse_logs.py
   ▼
data/interim/battles.parquet      (shared format: see data/README.md)
   │  src/data/split.py
   ▼
data/interim/{train,val,test}.parquet
   │  src/features/build_features.py
   ▼
data/processed/{train,val,test}_features.parquet
   │  src/models/*  +  experiments/*
   ▼
results/*.csv  +  figures/*.png  →  report/
```

## Responsibilities

| Person | Code | Report sections |
|---|---|---|
| Alexander | `src/data/`, `src/features/` | Data |
| Andy | `src/models/baselines.py`, `experiments/run_baselines.py` | Background, Methods (features), Experiments (wrong predictions) |
| Nguyen Hang | `src/models/logistic.py`, `experiments/run_main.py` | Methods (models), Experiments (main table) |
| Ruel | `src/models/trees.py`, `src/evaluation/`, `experiments/run_ablation.py`, `src/simulation/` | Experiments (figures, ablation) |
| Everyone | | Abstract, Introduction, Conclusion |

Log finished work in `CONTRIBUTIONS.md`; it becomes the page-1 footnote.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Status

| Step | Script | Status |
|---|---|---|
| 1. Download | `src/data/download.py` | ✅ Done: 1,190,747 battles (Jun 2025 – Oct 2026) |
| 2. Parse logs | `src/data/parse_logs.py` | ✅ Done: 533,490 clean battles → 100,000 sampled (`data/interim/battles.parquet`) |
| 3. Split | `src/data/split.py` | ✅ Done: 70k / 15k / 15k by time (train to 2026-04-08, test from 2026-07-01) |
| 4. Features | `src/features/pokedex.py`, `build_features.py` | ✅ Done: 450 feature columns, 140k train rows (swap-augmented) / 15k / 15k (`data/processed/`) |
| 5. Baselines + logistic regression | `src/models/baselines.py`, `logistic.py`, `experiments/run_baselines.py`, `run_main.py` | ⬜ To do |
| 6. Trees, figures, ablation | `src/models/trees.py`, `src/evaluation/plots.py`, `experiments/run_ablation.py` | ⬜ To do |

Update this table when you finish a step.

## Running, step by step

Each step reads the previous step's output, so run them in order. All settings (date range,
rating filter, split sizes, seed) live in `config.yaml`. Don't hard-code them.

**1. Get the data (once per machine, ~15 min).** Needs a free Hugging Face login, or the
download gets rate-limited: `hf auth login` and paste a "Read" token from
https://huggingface.co/settings/tokens. Details in [data/README.md](data/README.md).

```bash
python -m src.data.download --check     # ~1 min sanity check
python -m src.data.download             # -> data/raw/gen9ou_slim.parquet (169 MB)
python -m src.data.download --inspect   # quick quality check
```

**2. Turn logs into clean battles and split them by time.**

```bash
python -m src.data.parse_logs           # -> data/interim/battles.parquet
python -m src.data.split                # -> data/interim/{train,val,test}.parquet
```

**3. Build features.**

```bash
python -m src.features.build_features   # -> data/processed/{train,val,test}_features.parquet
```

Rebuild after pulling a change listed in [CHANGELOG.md](CHANGELOG.md). `load_features()` stops
with an "out of date" error if your feature files don't match `config.yaml`.

**4. Train and evaluate.**

```bash
python -m experiments.run_baselines     # -> results/baselines.csv
python -m experiments.run_main          # tunes on val, scores test once -> results/main_results.csv
python -m experiments.run_ablation      # -> results/ablation.csv, figures/ablation_bar.png
pytest                                  # run the tests any time
```

### Getting started with training

Training needs the feature files from step 3, so until steps 2–3 exist you can't train on
real data. You don't have to wait, though:

1. **Write the model code against the shared formats now.** `src/models/logistic.py` gets
   `X_train, y_train, X_val, y_val`: a feature table plus the `p1_win` label. Column names
   start with their group (`stats_`, `types_`, `match_`, `species_`, `learned_`, `rating_`),
   and every feature is p1 − p2 (what each one means is at the top of `build_features.py`).
   Load a split with `load_features()` from `src/features/build_features.py`:
   ```python
   X_train, y_train = load_features("train")                       # all features
   X_val, y_val = load_features("val", drop_groups=["rating"])    # teams only
   ```
   X holds only feature columns and is indexed by `battle_id` (use `X.index` when saving
   test predictions). Train has every battle twice: the second copy has the teams swapped.
2. **Test it on a small fake feature table**, e.g. a few hundred rows of random numbers with
   those column prefixes and a random 0/1 label. Make sure the C grid loop runs and the
   function returns a model and val log losses. Score everything with
   `src.evaluation.metrics.evaluate()` so all models are compared the same way.
3. **Swap in the real files** once `data/processed/*_features.parquet` exists. Nothing else
   should need to change.

Rules for training: tune on **val** only, and score **test** once, at the very end, in
`experiments/run_main.py`.

## Rules

- **Never commit data.** The replay dataset has no license, so it's coursework only and we don't redistribute it. `data/` is git-ignored except the READMEs.
- Tune on **val** only. Touch **test** once, at the end, in `run_main.py`.
- Every tool, dataset and paper we use goes in `report/references.bib`.
- Work on a branch, open a PR, and have one other person look at it before merging.

## Schedule

| Day | Goal |
|---|---|
| Mon 11/23 | Repo + template set up; start parsing logs. Others build on the fake sample in `data/sample/`. |
| Tue 11/24 | 100k+ clean battles split; features working; baselines running |
| Wed 11/25 | Logistic regression tuned (teams-only vs. + ratings); tree model trained; first results table |
| Thu 11/26 | Thanksgiving: light day, rough draft of own report sections |
| Fri 11/27 | Ablation + all figures |
| Sat 11/28 | Abstract, Intro, Conclusion together; insert figures |
| Sun 11/29 | Trim to 4–6 pages, citations, footnote, supplementary ZIP |
| Mon 11/30 | Done |

Extras, only if ahead of schedule: MLP, luck simulations with poke-env, attention model.
