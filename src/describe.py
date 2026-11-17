"""Describe stage: the descriptive figures, and the excess-mass table.

Reads data/processed/attempts_discovered.parquet and writes into
reports/figures plus reports/excess_mass.csv.

Figures, all on the shared styling in plotstyle.py:

1-3. describe_hist_{lift}.png - third-attempt weight histograms, one panel per
     group. Pound-loaded meets are drawn in pounds on their 5 lb grid and kilo
     meets in kilos on their 2.5 kg grid, because that is the grid the bar was
     actually loaded on. Pound and kilo milestones are marked on separate rows
     beneath the axis and discovered milestones colour the bar, so where the
     pre-specified grid and the discovered weights agree is visible at a glance.
4.   describe_success_by_jump.png - success rate against the size of the jump
     from the previous attempt, for attempts 2 and 3.
5.   describe_jump_vs_distance.png - mean jump against how far the previous
     attempt sat below the next pound milestone.
6.   describe_excess_mass.png - excess mass at every milestone, by group.

Everything here is descriptive. Nothing on these axes is a causal estimate, and
nothing is labelled as one; the regressions in econ_models.py are what can
support a claim.

Two scope notes. The histograms and the excess-mass estimates use attempt
selection only, never outcomes, so they run on the full sample. The success-rate
figure does use outcomes, but it is cut by jump size and attempt number, which
are pre-specified, not by any discovered milestone, so it is also not a test on
discovered weights and does not owe the test half. Any figure that crossed
on_discovered_milestone with success would (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

import plotstyle as ps
from constants import KG_PER_LB
from discover import DISCOVERY_BIN_WIDTH, GROUPS, assign_groups, scan_cell

ROOT = Path(__file__).resolve().parents[1]
IN_PARQUET = ROOT / "data" / "processed" / "attempts_discovered.parquet"
MILESTONES_PARQUET = ROOT / "data" / "processed" / "discovered_milestones.parquet"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_PATH = ROOT / "reports" / "excess_mass.csv"

LIFTS = ("squat", "bench", "deadlift")

# Uniform bins, so a bin index is arithmetic rather than a sorted label. The
# first version of this binned with cut() and keyed on the interval's string,
# which orders "[10, 12)" before "[2, 3)" and drew lines that doubled back.
JUMP_LO, JUMP_WIDTH, JUMP_BINS = 0.0, 1.0, 25     # percent of previous attempt
DIST_LO, DIST_WIDTH, DIST_BINS = 0.0, 2.5, 10     # kilos to the next milestone

MIN_BIN_N = 200  # a plotted point needs enough attempts to mean anything


def use_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def wilson(successes: np.ndarray, n: np.ndarray, z: float = 1.96):
    """Wilson score interval. Normal approximation collapses near 0 and 1,
    which is exactly where third-attempt success rates live."""
    n = np.maximum(n, 1)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    # Clip: a rate of exactly 0 or 1 can round a hair outside [0, 1].
    return np.clip(centre - half, 0.0, 1.0), np.clip(centre + half, 0.0, 1.0)


def uniform_bin(col: pl.Expr, lo: float, width: float, nbins: int) -> pl.Expr:
    """Index of a value's bin on a regular grid. Arithmetic, so it sorts."""
    return ((col - lo) / width).floor().cast(pl.Int32)


def bin_centres(lo: float, width: float, nbins: int) -> np.ndarray:
    return lo + (np.arange(nbins) + 0.5) * width


def group_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Attach the discovery group and the weight in that group's own unit."""
    df = assign_groups(df)
    expr = pl.lit(None, dtype=pl.Float64)
    for name, spec in GROUPS.items():
        native = (
            pl.col("weight_kg")
            if spec["unit"] == "kg"
            else pl.col("weight_kg") / KG_PER_LB
        )
        expr = pl.when(pl.col("group") == name).then(native).otherwise(expr)
    return df.with_columns(expr.alias("weight_native"))


def excess_mass_table(df: pl.DataFrame, discovered: pl.DataFrame) -> pl.DataFrame:
    """Excess mass at every bin, per group and lift, on the full sample.

    The counterfactual is the same leave-the-candidate-out polynomial the
    discovery stage uses, so the descriptive numbers and the discovered list are
    produced by one estimator rather than two that might disagree.
    """
    thirds = df.filter(pl.col("attempt_no") == 3)
    rows = []
    for group, spec in GROUPS.items():
        width = DISCOVERY_BIN_WIDTH[spec["unit"]]
        for lift in LIFTS:
            sub = thirds.filter(
                (pl.col("group") == group) & (pl.col("lift") == lift)
            )
            if sub.height == 0:
                continue
            cell = scan_cell(sub["weight_native"].to_numpy(), width)
            if cell.height == 0:
                continue
            rows.append(
                cell.with_columns(
                    pl.lit(group).alias("group"),
                    pl.lit(lift).alias("lift"),
                    pl.lit(spec["unit"]).alias("bin_unit"),
                    (
                        pl.col("bin") if spec["unit"] == "kg"
                        else pl.col("bin") * KG_PER_LB
                    ).alias("bin_kg"),
                )
            )
            print(f"  excess mass {group:6s} {lift:9s}: {sub.height:>8,} attempts")

    full = pl.concat(rows)

    # Carry the discovery-half verdict and the pre-specified label across.
    labels = discovered.select(
        "group", "lift", "bin", "prespecified_kind",
        pl.col("discovered").alias("discovered_in_half"),
    )
    return (
        full.with_columns(pl.col("bin").round(6))
        .join(labels.with_columns(pl.col("bin").round(6)),
              on=["group", "lift", "bin"], how="left")
        .with_columns(
            pl.col("prespecified_kind").fill_null("none"),
            pl.col("discovered_in_half").fill_null(False),
        )
    )


def plot_histograms(table: pl.DataFrame, lift: str) -> Path | None:
    """Third-attempt weights by group, each on its own loading grid."""
    cells = {
        g: table.filter(
            (pl.col("group") == g) & (pl.col("lift") == lift) & pl.col("scanned")
        ).sort("bin")
        for g, _ in ps.PANELS
    }
    if not any(c.height for c in cells.values()):
        return None

    lo_kg = min(c["bin_kg"].min() for c in cells.values() if c.height)
    hi_kg = max(c["bin_kg"].max() for c in cells.values() if c.height)
    pad = 0.02 * (hi_kg - lo_kg)
    lo_kg, hi_kg = lo_kg - pad, hi_kg + pad

    peak = max(
        float(c["count"].max()) / max(int(c["count"].sum()), 1)
        for c in cells.values() if c.height
    )
    ymax = peak * 1.24

    fig, axes = ps.new_figure(3, (13, 9.8))
    for ax, (group, label) in zip(axes, ps.PANELS):
        cell = cells[group]
        unit = GROUPS[group]["unit"]
        width = DISCOVERY_BIN_WIDTH[unit]
        if cell.height == 0:
            ax.text(0.5, 0.5, f"no {lift} data", ha="center", va="center",
                    transform=ax.transAxes, color=ps.MUTED, fontsize=9)
            ax.set_axis_off()
            continue

        total = max(int(cell["count"].sum()), 1)
        bins = cell["bin"].to_numpy()
        share = cell["count"].to_numpy() / total
        counter = cell["counterfactual"].to_numpy() / total
        found = cell["discovered_in_half"].to_numpy()
        kind = cell["prespecified_kind"].to_numpy()

        ax.bar(bins, share, width=width * 0.82, color=ps.BAR, linewidth=0)
        if found.any():
            ax.bar(bins[found], share[found], width=width * 0.82,
                   color=ps.FOUND, linewidth=0)
        ax.plot(bins, counter, color=ps.INK_2, linewidth=2.0, solid_capstyle="round")

        ax.set_ylim(-ymax * 0.15, ymax)
        ps.milestone_rows(
            ax,
            bins[np.isin(kind, ["lb50", "lb100"])],
            bins[np.isin(kind, ["kg10", "kg50"])],
            ymax,
        )
        for b, sh in zip(bins[found], share[found]):
            ax.annotate(f"{b:g}", (b, sh), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7.5, color=ps.INK)

        ax.set_xlim(*( (lo_kg / KG_PER_LB, hi_kg / KG_PER_LB) if unit == "lb"
                       else (lo_kg, hi_kg) ))
        ax.axhline(0, color=ps.AXIS, linewidth=1.0)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}" if v >= 0 else "")
        ps.style_axes(
            ax,
            xlabel=f"third-attempt weight ({unit})",
            ylabel="share of attempts",
            title=f"{label}  ·  {width:g} {unit} bins  ·  {total:,} third attempts",
        )

    ps.histogram_legend(fig)
    ps.suptitle(fig, f"{lift.capitalize()} third attempts: where lifters bunch")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return ps.save(fig, FIG_DIR / f"describe_hist_{lift}.png")


def plot_success_by_jump(df: pl.DataFrame) -> Path:
    """Success rate against the size of the jump from the previous attempt."""
    binned = df.filter(
        pl.col("attempt_no").is_in([2, 3]) & pl.col("jump_pct").is_not_null()
    ).with_columns(
        uniform_bin(pl.col("jump_pct"), JUMP_LO, JUMP_WIDTH, JUMP_BINS).alias("b")
    ).filter(pl.col("b").is_between(0, JUMP_BINS - 1))
    centres = bin_centres(JUMP_LO, JUMP_WIDTH, JUMP_BINS)

    fig, axes = ps.new_figure(3, (11, 9.2))
    for ax, lift in zip(axes, LIFTS):
        for colour, attempt in ((ps.BAR, 2), (ps.FOUND, 3)):
            cell = (
                binned.filter((pl.col("lift") == lift) & (pl.col("attempt_no") == attempt))
                .group_by("b")
                .agg(pl.len().alias("n"), pl.col("success").sum().alias("k"))
                .filter(pl.col("n") >= MIN_BIN_N)
                .sort("b")
            )
            if cell.height == 0:
                continue
            x = centres[cell["b"].to_numpy()]
            n = cell["n"].to_numpy().astype(float)
            k = cell["k"].to_numpy().astype(float)
            lo, hi = wilson(k, n)
            ax.fill_between(x, lo, hi, color=colour, alpha=0.18, linewidth=0)
            ax.plot(x, k / n, color=colour, linewidth=2.0, marker="o", markersize=4)

        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ps.style_axes(ax, xlabel="jump from previous attempt (%)",
                      ylabel="success rate", title=lift.capitalize())

    ps.series_legend(fig, [(ps.BAR, "attempt 2"), (ps.FOUND, "attempt 3")])
    ps.suptitle(fig, "Success rate by jump size (descriptive; shaded = 95% Wilson)")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return ps.save(fig, FIG_DIR / "describe_success_by_jump.png")


def plot_jump_vs_distance(df: pl.DataFrame) -> Path:
    """Mean jump against how far the previous attempt sat below a lb milestone."""
    binned = df.filter(
        pl.col("dist_to_next_lb_ms").is_not_null()
        & pl.col("jump_kg").is_not_null()
        & pl.col("group").is_not_null()
    ).with_columns(
        uniform_bin(pl.col("dist_to_next_lb_ms"), DIST_LO, DIST_WIDTH, DIST_BINS).alias("b")
    ).filter(pl.col("b").is_between(0, DIST_BINS - 1))
    centres = bin_centres(DIST_LO, DIST_WIDTH, DIST_BINS)
    colours = {"us_lb": ps.FOUND, "us_kg": ps.BAR, "kilo": ps.PRESPEC_KG}

    fig, axes = ps.new_figure(2, (11, 7.4))
    for ax, attempt in zip(axes, (2, 3)):
        for group, label in ps.PANELS:
            cell = (
                binned.filter((pl.col("group") == group) & (pl.col("attempt_no") == attempt))
                .group_by("b")
                .agg(pl.len().alias("n"),
                     pl.col("jump_kg").mean().alias("mean_jump"),
                     pl.col("jump_kg").std().alias("sd"))
                .filter(pl.col("n") >= MIN_BIN_N)
                .sort("b")
            )
            if cell.height == 0:
                continue
            x = centres[cell["b"].to_numpy()]
            n = cell["n"].to_numpy().astype(float)
            mean = cell["mean_jump"].to_numpy()
            se = cell["sd"].to_numpy() / np.sqrt(n)
            ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se,
                            color=colours[group], alpha=0.18, linewidth=0)
            ax.plot(x, mean, color=colours[group], linewidth=2.0, marker="o", markersize=5)

        ps.style_axes(
            ax,
            xlabel="kg from the previous attempt up to the next 50 lb milestone",
            ylabel="mean jump (kg)",
            title=f"Attempt {attempt}",
        )

    ps.series_legend(fig, [(colours[g], lab) for g, lab in ps.PANELS])
    ps.suptitle(fig, "Jump size against the gap to the next pound milestone (descriptive)")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return ps.save(fig, FIG_DIR / "describe_jump_vs_distance.png")


def plot_excess_mass(table: pl.DataFrame) -> Path:
    """Excess over the counterfactual at each pre-specified milestone."""
    marks = table.filter(
        pl.col("scanned")
        & ((pl.col("prespecified_kind") != "none") | pl.col("discovered_in_half"))
        & pl.col("excess_pct").is_finite()
    )
    lift_marker = {"squat": "o", "bench": "s", "deadlift": "^"}

    fig, axes = ps.new_figure(3, (12.5, 9.2))
    for ax, (group, label) in zip(axes, ps.PANELS):
        cell = marks.filter(pl.col("group") == group)
        unit = GROUPS[group]["unit"]
        ax.axhline(0, color=ps.AXIS, linewidth=1.2)
        for lift in LIFTS:
            pts = cell.filter(pl.col("lift") == lift)
            if pts.height == 0:
                continue
            x = pts["bin"].to_numpy()
            y = pts["excess_pct"].to_numpy()
            found = pts["discovered_in_half"].to_numpy()
            ax.vlines(x, 0, y, color=ps.GRID, linewidth=1.0, zorder=1)
            ax.scatter(x[~found], y[~found], marker=lift_marker[lift], s=34,
                       facecolor="none", edgecolor=ps.BAR, linewidths=1.4, zorder=3)
            ax.scatter(x[found], y[found], marker=lift_marker[lift], s=44,
                       color=ps.FOUND, linewidths=0, zorder=4)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
        ps.style_axes(ax, xlabel=f"milestone weight ({unit})",
                      ylabel="excess over counterfactual",
                      title=f"{label}  ·  {cell.height} milestone bins")

    handles = [
        ps.plt.Line2D([], [], marker=m, color=ps.BAR, linestyle="none",
                      markersize=7, markerfacecolor="none")
        for m in lift_marker.values()
    ] + [ps.plt.Line2D([], [], marker="o", color=ps.FOUND, linestyle="none", markersize=7)]
    fig.legend(handles, [*[l.capitalize() for l in LIFTS], "discovered"],
               loc="upper right", bbox_to_anchor=(0.995, 0.995), frameon=False,
               fontsize=9, ncol=4, labelcolor=ps.INK_2)
    ps.suptitle(fig, "Excess mass at each milestone, over a polynomial counterfactual")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return ps.save(fig, FIG_DIR / "describe_excess_mass.png")


def report(table: pl.DataFrame) -> None:
    marks = table.filter(pl.col("scanned") & (pl.col("prespecified_kind") != "none"))
    summary = (
        marks.group_by("group", "prespecified_kind")
        .agg(pl.len().alias("bins"),
             pl.col("excess_pct").median().round(3).alias("median_excess"),
             (pl.col("excess_pct") > 0).mean().round(3).alias("share_positive"))
        .sort("group", "prespecified_kind")
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=120):
        print("\nExcess mass at pre-specified milestones (full sample, descriptive):")
        print(summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=IN_PARQUET)
    parser.add_argument("--milestones", type=Path, default=MILESTONES_PARQUET)
    args = parser.parse_args()
    use_utf8_stdout()

    for path in (args.src, args.milestones):
        if not path.exists():
            print(f"{path} not found - run the discover stage first.", file=sys.stderr)
            return 1

    df = group_frame(pl.read_parquet(args.src))
    discovered = pl.read_parquet(args.milestones)
    print(f"loaded {df.height:,} attempts")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    table = excess_mass_table(df, discovered)
    out = table.select(
        "group", "lift", "bin", "bin_unit", "bin_kg", "prespecified_kind",
        "discovered_in_half", "count", "counterfactual", "excess", "excess_pct",
        "z", "dispersion", "scanned",
    ).sort("group", "lift", "bin")
    out.write_csv(TABLE_PATH)
    print(f"wrote {out.height:,} rows to {TABLE_PATH}")

    for lift in LIFTS:
        path = plot_histograms(table, lift)
        if path:
            print(f"wrote {path.name}")
    for fn in (plot_success_by_jump, plot_jump_vs_distance):
        print(f"wrote {fn(df).name}")
    print(f"wrote {plot_excess_mass(table).name}")

    report(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
