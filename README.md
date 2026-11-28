# Predicting Who Wins a Pokémon Battle

COSC 4368 Final Project: Alexander Mayer, Andy Nguyen, Nguyen Hang, Ruel Cabading

Given both six-Pokémon teams at team preview (Gen 9 OU), predict the probability that each side wins.

**Internal deadline:** Mon Nov 30 · **Real deadline:** Fri Dec 11, 11:59pm

New to the project? Read [project_overview.txt](project_overview.txt) for the motivation and
plan, then follow **Start here**.

## Start here

**1. Set up Python (once).**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Get the data and build the features (once per machine).** The data is not in git, so
everyone builds their own copy. The download takes about 15 minutes and needs a free Hugging
Face login, or it gets rate-limited: run `hf auth login` and paste a "Read" token from
https://huggingface.co/settings/tokens. The other three steps take a few minutes.

```bash
python -m src.data.download             # -> data/raw/gen9ou_slim.parquet (169 MB)
python -m src.data.parse_logs           # -> data/interim/battles.parquet
python -m src.data.split                # -> data/interim/{train,val,test}.parquet
python -m src.features.build_features   # -> data/processed/{train,val,test}_features.parquet
```

Prefer Colab? [notebooks/colab_run.ipynb](notebooks/colab_run.ipynb) runs the same steps and
keeps the data in Google Drive.

**3. Check it works.**

```bash
pytest
```

**4. After every `git pull`,** look at [CHANGELOG.md](CHANGELOG.md). If an entry says
**Rebuild**, rerun `python -m src.features.build_features`. You can't miss it:
`load_features()` stops with an "out of date" error when your feature files don't match
`config.yaml`.

## Who does what

Each code file below starts with a TODO list of what it needs.

| Person | Code | Report sections | Status |
|---|---|---|---|
| Alexander | `src/data/`, `src/features/` | Data | ✅ Data, features, Data section draft and figures done |
| Andy | `src/models/baselines.py`, `experiments/run_baselines.py` | Background, Methods (features), Experiments (wrong predictions) | ⬜ To do |
| Nguyen Hang | `src/models/logistic.py`, `experiments/run_main.py` | Methods (models), Experiments (main table) | 🟨 Logistic + `run_main` done (val: 53.8% / 0.6874 teams only, 54.7% / 0.6862 + ratings); run_main picks up trees once they exist. Report sections to do |
| Ruel | `src/models/trees.py`, `src/evaluation/plots.py`, `experiments/run_ablation.py` | Experiments (figures, ablation) | ⬜ To do |
| Everyone | | Abstract, Introduction, Conclusion | ⬜ To do |

Update your Status cell when you finish something, and add a line to
[CONTRIBUTIONS.md](CONTRIBUTIONS.md) (it becomes the page-1 footnote).

Optional extras, only if ahead of schedule: MLP (`src/models/mlp.py`), luck simulations with
poke-env (`src/simulation/luck_sim.py`).

## Writing model code

Load a split with `load_features()`:

```python
from src.features.build_features import load_features

X_train, y_train = load_features("train")                      # teams + ratings (450 columns)
X_val, y_val = load_features("val", drop_groups=["rating"])   # teams only (449 columns)
```

- `y` is `p1_win`: 1 if player 1 won. `X` holds only feature columns, both are indexed by
  `battle_id` (use `X.index` when saving test predictions).
- **Every feature is player 1 minus player 2**: positive favours player 1, 0 means even.
- **Train has every battle twice** (140k rows): the second copy has the teams swapped and the
  label flipped. Val and test have 15k rows each, once.
- Column names start with their group, which is what `drop_groups` uses:

  | Group | Columns | What it is |
  |---|---|---|
  | `stats_` | 19 | base stats: team mean, best and worst |
  | `types_` | 20 | type counts, shared weaknesses |
  | `match_` | 5 | type and speed matchups against the other team |
  | `species_` | 402 | +1 / −1 if only one side has that Pokémon |
  | `learned_` | 3 | win rates counted on training battles |
  | `rating_` | 1 | `rating_diff`, only in the teams + ratings setting |

  What each column means: the top of `src/features/build_features.py`, or the plain-language
  version in [report/notes/features_for_methods.md](report/notes/features_for_methods.md).
- Score every model with `src.evaluation.metrics.evaluate(y_true, p_win)` so the results
  table is consistent. It returns accuracy, log loss and Brier score.
- Hyperparameter grids live in `config.yaml` under `models:`. Read them from there, don't
  hard-code them.
- **Tune on val only. Score test once**, at the very end, in `experiments/run_main.py`.

What to expect on **val**: a coin flip gets 50% accuracy and log loss 0.693. A quick
standardized logistic regression on the teams-only features got about 54% and 0.688. Small
gains are normal here: the most-used Pokémon all win between 49% and 52% of their games.

## Pipeline

Each step reads the previous step's output. All settings (date range, rating filter, split
sizes, seed, feature options) live in `config.yaml`.

```
Hugging Face replays (34M, we read the 2.5M Gen 9 OU)
   │  src/data/download.py         data/raw/gen9ou_slim.parquet: 1.19M battles
   ▼
   │  src/data/parse_logs.py       data/interim/battles.parquet: 533k clean, 100k sampled
   ▼
   │  src/data/split.py            data/interim/{train,val,test}.parquet: 70k / 15k / 15k by time
   ▼
   │  src/features/build_features.py   data/processed/*_features.parquet: 450 columns
   ▼
   │  src/models/*  +  experiments/*
   ▼
results/*.csv  +  figures/*  →  report/
```

Train is the oldest 70% of battles (to 2026-04-08), val the next 15% (to 2026-07-01), and
test the newest 15% (to 2026-10-05). Download options and data checks are in
[data/README.md](data/README.md). Experiments, once written:

```bash
python -m experiments.run_baselines     # -> results/baselines.csv
python -m experiments.run_main          # tunes on val, scores test once -> results/main_results.csv
python -m experiments.run_ablation      # -> results/ablation.csv, figures/ablation_bar.png
python -m src.data.figures              # data figures + filter table for the Data section
```

## Where things live

```
config.yaml           every setting (data filters, split, features, model grids)
src/data/             download, parse, split, data figures
src/features/         Pokédex + type chart, feature builder, load_features()
src/models/           baselines, logistic, trees (+ optional mlp)
src/evaluation/       evaluate() metrics, report figures
experiments/          scripts that produce results/ and figures/
tests/                pytest; tests/fake_data.py makes small fake battles
data/                 not in git, except data/sample/battles_sample.csv (fake, for trying code)
results/, figures/    outputs used in the report
report/               sections/ (LaTeX drafts), notes/ (writing notes, slides), references.bib
CHANGELOG.md          changes that need a rebuild
CONTRIBUTIONS.md      who did what, for the footnote
```

## Rules

- **Never commit data.** The replay dataset has no license, so it's coursework only and we
  don't redistribute it. `data/` is git-ignored except the READMEs and the fake sample.
- Tune on **val** only. Touch **test** once, at the end, in `run_main.py`.
- Work on a branch, open a PR, and have one other person look at it before merging.
- Changing how the features are built? Bump `FEATURES_VERSION` in `build_features.py` and add
  a CHANGELOG entry, so everyone knows to rebuild.
- Every tool, dataset and paper we use goes in `report/references.bib`.

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
