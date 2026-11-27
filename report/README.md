# Report

- Use the **provided course template** as-is (no font/margin changes). 4–6 pages, references
  don't count. Abstract ≤ 300 words.
- All four authors under the title; page-1 footnote with each person's role and % (see
  `CONTRIBUTIONS.md`).
- Figures come from `figures/`, tables from `results/`.
- Citations go in `references.bib`. Double-check every entry before submitting.

## Section owners

| Section | Owner |
|---|---|
| Abstract, Introduction, Conclusion | Everyone (Sat 11/28) |
| Background | Andy |
| Data | Alexander |
| Methods: features | Andy |
| Methods: models | Nguyen Hang |
| Experiments: main results table | Nguyen Hang |
| Experiments: figures, ablation | Ruel |
| Experiments: wrong predictions | Andy |

## Methods: tie every choice to class

Time-based train/val/test split · sigmoid output and cross-entropy (log loss) · gradient-based
training · L2 regularization and tuning on val · bias–variance (single tree vs. forest,
regularization strength) · evaluation beyond accuracy (log loss, Brier, calibration).

## Submission (due Fri Dec 11, 11:59pm; our target is Mon Nov 30)

- [ ] `report.pdf`
- [ ] `supplementary.zip`: code (this repo **without data**), extra results/figures
