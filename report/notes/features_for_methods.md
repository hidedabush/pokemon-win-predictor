# Features: notes for the Methods (features) section

Written for whoever drafts Methods (features). This is the "what and why" in plain words. The
exact definitions are in the docstring at the top of `src/features/build_features.py`, which
is the source of truth if anything here disagrees with it.

## The setup in one paragraph

Each battle becomes one row of numbers. Team A is player 1, team B is player 2, and **every
feature is written as A − B**: positive means player 1 has more of it, zero means the teams are
even. The label is `p1_win` (1 if player 1 won). Load the data with `load_features("train")`;
pass `drop_groups=["rating"]` for the teams-only model.

## The six feature groups (450 columns)

| Group | Columns | What it measures |
|---|---|---|
| `stats_*` | 19 | Team mean of each base stat (HP, Atk, Def, SpA, SpD, Spe), mean base stat total, and the best and worst of each stat on the team (`stats_max_spe` = fastest Pokémon). |
| `types_*` | 20 | How many team members have each of the 18 types, plus two weakness measures: `types_holes` (attacking types nothing on the team resists or is immune to) and `types_stacked` (most members weak to one attacking type). |
| `match_*` | 5 | Cross-team matchups, A against B minus B against A: how many opposing Pokémon we hit super effectively, how many of the 36 (ours, theirs) pairs we hit super effectively, how many of theirs we have a safe switch-in for, average type effectiveness, and how many pairs we outspeed. |
| `species_*` | 402 | One column per species: +1 if only team A has it, −1 if only team B has it, 0 otherwise. Only species seen in at least 20 training battles get a column. |
| `learned_*` | 3 | Numbers counted on training battles: the average win rate of the team's species, the average win rate of each (ours, theirs) species pair, and how often the team's species are seen together. |
| `rating_diff` | 1 | Player 1's rating minus player 2's. Only in the teams + ratings model. |

Type matchups use the type chart and also abilities that always give an immunity (Levitate,
for example). Team preview doesn't show abilities, so an ability only counts when every
ability the species can have gives that immunity.

## Design choices worth explaining (and the class idea behind each)

- **A − B for everything, plus swap augmentation.** In training, every battle is added a second
  time with the teams swapped: all features negated and the label flipped. The model can't
  learn "player 1 usually wins"; it has to learn from the teams. For logistic regression this
  also means the intercept should come out near zero. *(Data augmentation; symmetry of the
  problem.)* Swap augmentation is used on train only, never on val or test.
- **Species vocabulary from train only.** Which species get a column is decided on the
  training set, so val and test never shape the features. *(No leakage from held-out data.)*
- **Learned win rates without leaking the label.** A battle's own result must not be in its
  own features, or the model would look much better than it is. Train rows get their
  `learned_*` values from a 5-fold fit that leaves their own fold out (out-of-fold, like
  cross-validation). Val and test use the fit on all of train. *(Target leakage;
  cross-validation.)*
- **Smoothing toward 50%.** Win rates are pulled toward 50% as if every species or pair had 100
  extra battles split evenly, so a pair seen twice doesn't get a 100% win rate.
  We tried 5, 20 and 100 with a quick logistic regression on validation; 100 was slightly
  best and 5 worst, though the differences are small. *(Regularization / a prior;
  bias–variance.)*
- **Cosmetic forms merged.** Forms that play the same (Alcremie's flavours, for example) share
  one species, so they don't split each other's data.
- **New species.** A species that first shows up after the training period has no column of its
  own and a learned win rate of 50%; the model only sees its types and stats. This affects
  7.7% of validation and 2.0% of test battles (see the Data section).

## Numbers you might want to quote

- 450 feature columns: 19 stats, 20 types, 5 match, 402 species, 3 learned, 1 rating.
- 140,000 training rows (70,000 battles, each twice), 15,000 val rows, 15,000 test rows.
- Settings live in `config.yaml` under `features:` (min count 20, prior 100, 5 folds).

## Checked by tests

`tests/test_features.py` and `tests/test_pipeline.py` check that swapping the teams negates
every column exactly, that val/test labels can't change any feature, and that the learned
scores for train rows are out-of-fold.
