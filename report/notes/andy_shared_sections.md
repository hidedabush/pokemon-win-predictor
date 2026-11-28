# Shared sections — discussion drafts, not final results

## Abstract (draft; fill results after the coordinated test evaluation)
We investigate how well a competitive Pokémon battle outcome can be predicted before the first turn using the two teams at preview. We convert 100,000 filtered Generation 9 OU battles into numerical features describing species, base stats, types, matchups and training-derived historical patterns, with ladder ratings included in a separate setting. We compare simple rule-based baselines with supervised classifiers using a chronological train/validation/test split and evaluate probability quality with accuracy, log loss and Brier score. [Insert the final comparative results and main conclusion after the group finishes evaluation.]

## Introduction (draft)
At team preview, competitive Pokémon players know the opposing species but not every detail of the strategy they will face. Some team matchups appear favorable before play begins, yet player skill and chance can change the outcome. Our project studies how much predictive information team composition provides and whether adding ladder ratings improves the prediction. We frame the task as binary classification, estimating the probability of player 1 winning rather than asserting a certain outcome. Comparing trained classifiers with simple baselines gives a measure of the added value of learned patterns.

## Conclusion (outline; results required)
- State the final teams-only performance and its improvement over the coin-flip baseline, if any.
- Compare teams-only with teams + ratings, using the final results.
- Discuss which feature groups mattered based on the actual ablation results.
- Explain limitations: hidden moves/items/abilities, player decisions, chance, rating-filtered sample, and changes in the metagame over time.
- Avoid claims about causation or model superiority that the results do not support.
