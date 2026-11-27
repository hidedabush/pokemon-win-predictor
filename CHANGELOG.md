# Changelog

Changes that affect the data or feature files everyone builds locally (`data/` is not in git).
Newest first. When an entry says **Rebuild**, run that command after pulling.

`load_features()` checks this for you: if your `data/processed/` files were built with other
settings than the current `config.yaml` (or an older `FEATURES_VERSION`), it stops with an
"out of date" error that names the command to run.

## 2026-11-27: features version 2

- `features.learned_prior` 20 → 100: learned win rates are smoothed harder toward 50%. On
  validation a quick logistic regression did slightly better with 100 than with 20 or 5.
- `models.logistic.C_grid` now also has 0.0001 and 0.0003; the best C in that check was
  below the old smallest value (0.001).
- `build_features` now writes `data/processed/features_meta.json` with the settings it used,
  and `load_features()` refuses files that don't match `config.yaml`.
- Unchanged: 450 columns, same names and groups; train/val/test splits are the same.

**Rebuild:** `python -m src.features.build_features` (about 1.5 minutes). Files built before
this change have no `features_meta.json`, so `load_features()` will ask for the rebuild.

## Version 1

The first feature build (450 columns, `learned_prior` 20).
