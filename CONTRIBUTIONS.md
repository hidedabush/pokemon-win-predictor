# Contribution log

Add a line whenever you finish something. This becomes the required page-1 footnote
(what each person did + a percentage).

| Date | Who | What |
|---|---|---|
| 11/23 | | Repo skeleton set up |
| 11/27 | Alexander | Log parsing + time-based split |
| 11/27 | Alexander | Pokédex/type chart loader + feature builder (stats, types, matchups, species, rating; swap augmentation) |
| 11/27 | Alexander | More features: stat max/min, type holes, ability immunities, checks/speed matchups, out-of-fold learned win rates; cosmetic-form merging; `load_features()` loader |
| 11/27 | Alexander | Data checks + tests: parse/split/features end to end on fake data, checks on the real files; `to_id` accent fix |
| 11/27 | Alexander | Data section draft (`report/sections/data.tex`), data figures and filter table (`src/data/figures.py`) |
| 11/27 | Alexander | Feature check on val: `learned_prior` 20 → 100, wider logistic C grid; stale-feature check in `load_features()`, `CHANGELOG.md` |
| 11/27 | Andy | Implemented four rule-based baselines, baseline evaluation, synthetic unit tests, and error-analysis functionality. Successfully evaluated baselines on 15,000 validation battles. Drafted Background and Methods (features) sections. |
| 11/27 | Nguyen Hang | L2 logistic regression with C tuned on val, top coefficients (`src/models/logistic.py`); main experiment script (`experiments/run_main.py`); tests (`tests/test_models.py`) |

## Draft footnote (fill in at the end)

> Author contributions: All authors took part in discussion and planning.
> A. Mayer: data collection, log parsing and feature engineering, Data section (25%).
> A. Nguyen: baselines, error analysis, Background and Methods (features) sections (25%).
> N. Hang: logistic regression, main experiments, Methods (models) section (25%).
> R. Cabading: tree models, evaluation figures and ablation study (25%).
