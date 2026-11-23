# Predicting Who Wins a Pokémon Battle

COSC 4368 Final Project: Alexander Mayer, Andy Nguyen, Nguyen Hang, Ruel Cabading

Given both six-Pokémon teams at team preview (Gen 9 OU), predict the probability that each side wins.

**Internal deadline:** Mon Nov 30 · **Real deadline:** Fri Dec 11, 11:59pm

## Pipeline

```
raw replays (Hugging Face, parquet)
   │  src/data/download.py        ← Alexander
   ▼
data/raw/*.parquet
   │  src/data/parse_logs.py      ← Alexander
   ▼
data/interim/battles.parquet      (shared format: see data/README.md)
   │  src/data/split.py           ← Alexander
   ▼
data/interim/{train,val,test}.parquet
   │  src/features/build_features.py  ← Andy
   ▼
data/processed/{train,val,test}_features.parquet
   │  src/models/*  +  experiments/*  ← Nguyen Hang, Ruel
   ▼
results/*.csv  +  figures/*.png  →  report/
```

## Who owns what

| Person | Code | Report sections |
|---|---|---|
| Alexander | `src/data/` | Data |
| Andy | `src/features/` | Methods (features), Background |
| Nguyen Hang | `src/models/baselines.py`, `src/models/logistic.py`, `experiments/run_main.py` | Methods (models), Experiments (main table) |
| Ruel | `src/models/trees.py`, `src/evaluation/`, `experiments/run_ablation.py`, `src/simulation/` | Experiments (figures, ablation) |
| Everyone | | Abstract, Introduction, Conclusion |

Write what you did in `CONTRIBUTIONS.md` as you go. It becomes the page-1 footnote.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running (once the stubs are filled in)

```bash
python -m src.data.download          # pull Gen 9 OU replays
python -m src.data.parse_logs        # logs -> battles.parquet
python -m src.data.split             # time-based train/val/test
python -m src.features.build_features
python -m experiments.run_baselines
python -m experiments.run_main
python -m experiments.run_ablation
```

All settings (sample size, rating filter, split sizes, seed) live in `config.yaml`. Don't hard-code them.

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
