# Presentation: data slides (Alexander, about 2 minutes)

The group has 10 minutes plus 2 for questions, so the data part gets two slides. The figures
are in `figures/` (use the PNGs in slides). Copy the slide text into whatever tool the deck is
in; the speaker notes are what to say, not what to put on the slide.

---

## Slide 1: Where the battles come from

**On the slide**
- Public Pokémon Showdown replay dataset (Hugging Face): 34M replays, 2.5M Gen 9 OU
- We use Gen 9 OU from June 2025 to Oct 2026: **1.19M battles**
- From each replay we keep only: the 12 Pokémon at team preview, both ratings, the winner
- Filters: both players rated ≥ 1300, ratings within 100 points → **533k clean battles**
- Random sample of **100,000** battles; player 1 wins 49.8%
- Table: `results/data_filters.tex` (or just the three bold numbers above)

**Speaker notes**
> Our data is real ladder battles from Pokémon Showdown, from a public replay dataset. A replay
> is the full text log of a battle, but we only need what's visible before turn one: the two
> teams of six, and then who won. We also keep the players' ratings, which we use in one of our
> two models. We drop broken logs and, on purpose, the low ladder: both players must be rated
> at least 1300 and within 100 points of each other, so the teams decide more of the game than
> a skill gap does. That leaves about half a million battles; we work with a random 100,000.
> Player 1 wins almost exactly half, so there's no built-in side advantage.

---

## Slide 2: Splitting by time, and what the teams look like

**On the slide**
- Figure: `figures/data_split_timeline.png`
- Oldest 70% train · next 15% validation · newest 15% test
- Figure: `figures/data_top_species.png`
- Great Tusk is on 31% of teams, but every popular Pokémon wins 49–52%

**Speaker notes**
> We split by time, not at random. We train on the oldest 70%, tune on the next 15%, and test
> on the newest 15%, the same way you'd use this for real: learn from past battles, predict
> future ones. A random split would let the model peek at the test period, since popular teams
> change over time. On the right: usage is very uneven, Great Tusk is on almost a third of
> teams, but no popular Pokémon wins much more than half its games. So no single Pokémon
> decides a battle; any signal has to come from combinations and matchups, which is what our
> features look at. [Hand over to the features / methods part.]

---

## Likely questions

- **Why only 100k of 533k battles?** Enough to train and test on, and it keeps every step quick
  on a laptop. The code can use all of them (`sample_size` in `config.yaml`).
- **Why a rating cutoff at 1300?** Below that it's mostly new and casual accounts, where play
  quality is more random than the teams.
- **Why within 100 points?** With a big rating gap, the better player's skill could explain the
  result, and the model would learn the gap instead of the teams.
- **What about Pokémon that only appear later?** 18 new species show up in 7.7% of validation
  battles (13 in 2.0% of test); the model only knows their types and stats.
- **Do you look at moves or items?** No. Team preview only shows the species, and we predict
  from what both players can see before turn one.
