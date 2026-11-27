"""Figures and the filter table for the report's Data section.

Input:  data/interim/{train,val,test}.parquet and data/interim/data_stats.json
Output: figures/data_split_timeline.{pdf,png}   battles per week, colored by split
        figures/data_top_species.{pdf,png}      top train species: usage and win rate
        results/data_filters.tex                 LaTeX table of the filter steps

Run after src.data.split:
    python -m src.data.figures
"""

import json

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features.pokedex import _gen9, load_aliases, load_pokedex  # noqa: E402
from src.utils import ROOT, load_config  # noqa: E402

SPLITS = ("train", "val", "test")
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WIDTH = 6.5  # inches, full text width

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "pdf.fonttype": 42,  # embed TrueType so the PDF text stays selectable
})


def _save(fig, name: str, figures_dir) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.pdf/.png -> {figures_dir}")


def split_timeline(parts: dict[str, pd.DataFrame], figures_dir) -> None:
    """Battles per week (UTC, weeks start Monday), one color per split."""
    weeks = {}
    for s in SPLITS:
        t = pd.to_datetime(parts[s]["upload_time"], unit="s", utc=True).dt.tz_localize(None)
        weeks[s] = t.dt.to_period("W-SUN").dt.start_time.value_counts()
    counts = pd.DataFrame(weeks).fillna(0).sort_index()
    counts = counts.reindex(pd.date_range(counts.index.min(), counts.index.max(), freq="7D"),
                            fill_value=0)

    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    bottom = np.zeros(len(counts))
    for s in SPLITS:  # stacked, so the week a split ends in shows both colors
        n = len(parts[s])
        ax.bar(counts.index, counts[s], width=5.5, bottom=bottom, align="edge",
               color=SPLIT_COLORS[s], linewidth=0, label=f"{s} ({n / 1000:.0f}k battles)")
        bottom += counts[s].to_numpy()
    ax.set_ylabel("battles per week")
    ax.yaxis.grid(True)
    ax.margins(x=0.01)
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
    ax.legend(frameon=False, loc="upper left", ncols=3, bbox_to_anchor=(0, 1.15))
    _save(fig, "data_split_timeline", figures_dir)


def species_stats(train: pd.DataFrame) -> pd.DataFrame:
    """One row per species seen in train: teams it was on, share of teams, wins, win rate.
    Cosmetic forms count as their base species, like in the features."""
    aliases = load_aliases(load_pokedex())
    rows = [(aliases.get(s, s), w)
            for teams, won in ((train["p1_team"], train["p1_win"]),
                               (train["p2_team"], 1 - train["p1_win"]))
            for team, w in zip(teams, won) for s in team]
    df = pd.DataFrame(rows, columns=["species", "win"])
    out = df.groupby("species")["win"].agg(teams="size", wins="sum")
    out["usage"] = out["teams"] / (2 * len(train))
    out["win_rate"] = out["wins"] / out["teams"]
    return out.sort_values("teams", ascending=False)


def top_species(train: pd.DataFrame, figures_dir, k: int = 15) -> pd.DataFrame:
    """Two panels sharing the species axis: % of teams using it, and its win rate with a 95% CI."""
    stats = species_stats(train).head(k)
    names = [_gen9().pokedex[s]["name"] for s in stats.index]
    y = np.arange(k)[::-1]  # most used at the top
    se = np.sqrt(stats["win_rate"] * (1 - stats["win_rate"]) / stats["teams"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True,
                                      gridspec_kw={"wspace": 0.08})
    blue = SPLIT_COLORS["train"]
    left.barh(y, 100 * stats["usage"], height=0.7, color=blue, linewidth=0)
    left.set_yticks(y, names)
    left.tick_params(axis="y", length=0)
    left.set_xlabel("% of train teams")
    left.xaxis.grid(True)

    right.axvline(50, color=INK_2, linewidth=0.8, linestyle=(0, (3, 2)))
    right.errorbar(100 * stats["win_rate"], y, xerr=100 * 1.96 * se, fmt="o", markersize=4,
                   color=blue, ecolor=blue, elinewidth=1, capsize=0)
    right.set_xlim(45, 55)
    right.set_xlabel("win rate of teams with it (%, 95% CI)")
    right.xaxis.grid(True)
    right.tick_params(axis="y", length=0)
    _save(fig, "data_top_species", figures_dir)
    return stats


def filter_table(stats: dict, path) -> None:
    """LaTeX table of the filter steps, from the counts parse_logs saved."""
    labels = {  # the step names in data_stats.json, written for the report
        "total rows": "Gen 9 OU replays since 2025-06-01",
        "parsed OK": "log parses",
        "exactly 6 Pokémon per side": "exactly 6 Pok\\'emon per side",
        "has a winner": "has a winner (no tie)",
        "both ratings present": "both players rated",
        "both >= 1300": "both ratings $\\geq$ 1300",
        "|gap| <= 100": "rating gap $\\leq$ 100",
    }
    lines = [
        "% Written by `python -m src.data.figures` from data/interim/data_stats.json; don't edit.",
        "\\begin{tabular}{lrr}",
        "\\hline",
        "Step & Battles left & Dropped \\\\",
        "\\hline",
    ]
    for row in stats["drop_counts"]:
        dropped = f"{row['dropped']:,}" if row["dropped"] else "--"
        lines.append(f"{labels.get(row['step'], row['step'])} & {row['battles']:,} & {dropped} \\\\")
    s = stats["sample"]
    lines += [
        "\\hline",
        f"random sample (seed {s['seed']}) & {s['sampled']:,} & \\\\",
        "\\hline",
        "\\end{tabular}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path.name} -> {path.parent}")


def main(cfg: dict) -> None:
    interim = ROOT / cfg["data"]["interim_dir"]
    figures_dir = ROOT / cfg["paths"]["figures_dir"]
    parts = {s: pd.read_parquet(interim / f"{s}.parquet") for s in SPLITS}
    split_timeline(parts, figures_dir)
    top = top_species(parts["train"], figures_dir)
    print(top[["teams", "usage", "win_rate"]].round(3).to_string())
    stats = json.loads((interim / "data_stats.json").read_text())
    filter_table(stats, ROOT / cfg["paths"]["results_dir"] / "data_filters.tex")


if __name__ == "__main__":
    main(load_config())
