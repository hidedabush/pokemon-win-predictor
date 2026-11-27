# Report

- Use the **provided course template** as-is (no font/margin changes). 4–6 pages, references
  don't count. Abstract ≤ 300 words.
- All four authors under the title; page-1 footnote with each person's role and % (see
  `CONTRIBUTIONS.md`).
- Figures come from `figures/`, tables from `results/`.
- Section drafts live in `report/sections/` (`data.tex` so far). Its figures and filter table
  are rebuilt with `python -m src.data.figures`.
- `report/notes/`: `features_for_methods.md` (what each feature is and why, for Methods) and
  `data_slides.md` (data slides and speaker notes for the presentation).
- Citations go in `references.bib`. Double-check every entry before submitting.

## Sections

Abstract · Introduction · Background · Data · Methods (features, models) · Experiments (main
results table, figures, ablation, wrong predictions) · Conclusion. Who writes which is in the
"Who does what" table in the main [README](../README.md).

## Methods: tie every choice to class

Time-based train/val/test split · sigmoid output and cross-entropy (log loss) · gradient-based
training · L2 regularization and tuning on val · bias–variance (single tree vs. forest,
regularization strength) · evaluation beyond accuracy (log loss, Brier, calibration).

## Submission (due Fri Dec 11, 11:59pm; our target is Mon Nov 30)

- [ ] `report.pdf`
- [ ] `supplementary.zip`: code (this repo **without data**), extra results/figures
