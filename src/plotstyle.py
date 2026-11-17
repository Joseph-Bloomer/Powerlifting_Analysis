"""Shared chart styling, so every figure in reports/figures reads as one set.

The palette was validated with the data-viz checker on the all-pairs list
against the light chart surface: lightness band, chroma floor, CVD separation
(worst 9.2 deutan), normal-vision floor (16.3) and contrast all pass. The one
warning is aqua at 2.74:1 against the surface, which obliges relief rather than
a swap: the kilo-milestone row it marks carries a legend label, a distinct
marker shape, and a saved table of the same numbers.

Figures commit to the light surface deliberately. These are report figures with
one rendering, not a page that follows a viewer's theme.

Colour is assigned by the job a mark does, never cycled:

- BAR        ordinary bins, the background mass
- FOUND      discovered milestone, a bar fill
- PRESPEC_LB pre-specified pound milestone, a caret under the axis
- PRESPEC_KG pre-specified kilo milestone, a circle under the axis

Pre-specified and discovered deliberately use different channels. Discovered is
a fill; pre-specified is a glyph below the baseline. A weight that is both shows
an orange bar standing directly on a marker, so the overlap reads at a glance
instead of needing a blended third colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
BAR = "#2a78d6"
FOUND = "#eb6834"
PRESPEC_LB = "#4a3aa7"
PRESPEC_KG = "#1baf7a"
SERIES = (BAR, FOUND, PRESPEC_LB, PRESPEC_KG)

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

DPI = 150

# The three region x loading-unit panels, in a fixed order everywhere.
PANELS = (
    ("us_lb", "US pound-loaded"),
    ("us_kg", "US kilo-loaded"),
    ("kilo", "Kilo countries"),
)


def new_figure(nrows: int, figsize: tuple[float, float]):
    fig, axes = plt.subplots(nrows, 1, figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.set_facecolor(SURFACE)
    return fig, axes


def style_axes(ax, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    """Recessive chrome: hairline y grid, no box, muted ticks."""
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def milestone_rows(
    ax, lb_bins: np.ndarray, kg_bins: np.ndarray, ymax: float
) -> None:
    """Pre-specified milestones as two glyph rows beneath the baseline.

    Two rows rather than one so a weight that is both a pound and a kilo
    milestone shows both marks instead of one hiding the other.
    """
    if len(lb_bins):
        ax.scatter(lb_bins, np.full(len(lb_bins), -ymax * 0.055), marker="^", s=26,
                   color=PRESPEC_LB, linewidths=0, zorder=5, clip_on=False)
    if len(kg_bins):
        ax.scatter(kg_bins, np.full(len(kg_bins), -ymax * 0.105), marker="o", s=18,
                   color=PRESPEC_KG, linewidths=0, zorder=5, clip_on=False)


def histogram_legend(fig, include_found: bool = True) -> None:
    """One legend for the milestone histograms, above the panels."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=BAR)]
    labels = ["third attempts"]
    if include_found:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=FOUND))
        labels.append("discovered milestone")
    handles += [
        plt.Line2D([], [], marker="^", color=PRESPEC_LB, linestyle="none", markersize=7),
        plt.Line2D([], [], marker="o", color=PRESPEC_KG, linestyle="none", markersize=6),
        plt.Line2D([], [], color=INK_2, linewidth=2.0),
    ]
    labels += ["pre-specified 50 lb", "pre-specified 10 kg", "counterfactual"]
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.995),
               frameon=False, fontsize=9, ncol=len(labels), labelcolor=INK_2)


def series_legend(fig, entries: list[tuple[str, str]], ncol: int | None = None) -> None:
    """Legend for line charts: (colour, label) pairs."""
    handles = [plt.Line2D([], [], color=c, linewidth=2.0, marker="o", markersize=5)
               for c, _ in entries]
    fig.legend(handles, [lab for _, lab in entries], loc="upper right",
               bbox_to_anchor=(0.995, 0.995), frameon=False, fontsize=9,
               ncol=ncol or len(entries), labelcolor=INK_2)


def suptitle(fig, text: str) -> None:
    fig.suptitle(text, x=0.008, ha="left", fontsize=13, color=INK)


def save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path
