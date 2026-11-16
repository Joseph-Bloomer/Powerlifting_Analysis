"""Data-driven milestone detection (reference tool, NOT a pipeline stage).

Scans third-attempt weights (squat, bench, deadlift) and meet totals for
bunching: bins whose attempt counts stand out from their local neighbourhood.
The point is to let milestone weights reveal themselves from attempt selection
and failure rates, rather than defining them by hand. Output is descriptive
only; it feeds no pipeline features. The econ identification remains the
lb-milestone x lb-region design in CLAUDE.md.

Reads data/raw/openpowerlifting.parquet (will be re-pointed at the cleaned
parquet once the clean stage exists) and applies the CLAUDE.md sample filters
inline, logging rows removed by each. Writes:

- references/milestones_detected.csv: one row per (lift, region, unit, bin)
  with count, failure rate, local excess z-score and a flagged indicator.
- reports/figures/milestones_{lift}_{region}_{unit}.png: attempt-count
  histogram (flagged bins highlighted) over a failure-rate panel.

Weights are binned both in kilos (2.5 kg, the loadable increment) and in
pounds (5 lb), so milestones native to either unit system can surface in
either region. Lb milestones showing up only in lb-region kg-bins is exactly
the pattern the design predicts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW_PARQUET = ROOT / "data" / "raw" / "openpowerlifting.parquet"
OUT_CSV = ROOT / "references" / "milestones_detected.csv"
FIG_DIR = ROOT / "reports" / "figures"

KG_PER_LB = 0.45359237

# Ambiguous-unit countries excluded from the region comparison (CLAUDE.md).
AMBIGUOUS_COUNTRIES = [
    "Canada",
    "UK",
    "England",
    "Scotland",
    "Wales",
    "N.Ireland",
    "Northern Ireland",
]

LIFTS = {"squat": "Squat3Kg", "bench": "Bench3Kg", "deadlift": "Deadlift3Kg"}
ATTEMPT_COLS = [
    f"{lift}{n}Kg" for lift in ("Squat", "Bench", "Deadlift") for n in (1, 2, 3)
]

BIN_WIDTHS = {"kg": 2.5, "lb": 5.0}
NEIGHBOUR_K = 5  # bins each side used as the local baseline
MIN_COUNT = 50  # a flagged bin must hold at least this many attempts


def log_filter(name: str, before: int, after: int) -> None:
    print(f"filter {name}: removed {before - after:,} rows ({before:,} -> {after:,})")


def apply_sample_filters(df: pl.DataFrame) -> pl.DataFrame:
    """CLAUDE.md sample restrictions, each one logged."""
    n = df.height

    df = df.filter(pl.col("Event") == "SBD")
    log_filter("Event == SBD", n, df.height)
    n = df.height

    df = df.filter(pl.col("Equipment") == "Raw")
    log_filter("Equipment == Raw", n, df.height)
    n = df.height

    df = df.filter(pl.all_horizontal(pl.col(c).is_not_null() for c in ATTEMPT_COLS))
    log_filter("attempts 1-3 all present", n, df.height)
    n = df.height

    df = df.filter(pl.col("Place").cast(pl.Int32, strict=False).is_not_null())
    log_filter("numeric Place", n, df.height)
    n = df.height

    df = df.filter(~pl.col("MeetCountry").is_in(AMBIGUOUS_COUNTRIES))
    log_filter("drop ambiguous-unit countries", n, df.height)

    return df.with_columns(
        pl.when(pl.col("MeetCountry") == "USA")
        .then(pl.lit("lb"))
        .otherwise(pl.lit("kg"))
        .alias("region")
    )


def excess_z(counts: np.ndarray, k: int = NEIGHBOUR_K) -> np.ndarray:
    """Local excess z-score per bin on a regular grid of counts.

    Each bin is compared with the mean of the k bins on each side (the bin
    itself excluded). The denominator is the larger of the neighbourhood's
    standard deviation and the Poisson scale sqrt(mean), floored at 1, so a
    spike over a perfectly flat or empty neighbourhood still scores high
    instead of dividing by zero. Edge bins use the neighbours that exist.
    """
    n = len(counts)
    z = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        neigh = np.concatenate([counts[lo:i], counts[i + 1 : hi]])
        if len(neigh) < 2:
            continue
        mean = neigh.mean()
        denom = max(neigh.std(ddof=1), np.sqrt(mean), 1.0)
        z[i] = (counts[i] - mean) / denom
    return z


def bin_attempts(
    df: pl.DataFrame, weight_col: str, success_col: str | None, width: float
) -> pl.DataFrame:
    """Aggregate to a regular grid of bins: count, failure rate per bin.

    Missing bins inside the observed range are filled with count 0 so that the
    neighbourhood statistic sees a regular grid.
    """
    binned = (
        df.with_columns(((pl.col(weight_col) / width).round(0) * width).alias("bin"))
        .group_by("bin")
        .agg(
            pl.len().alias("count"),
            (
                (1 - pl.col(success_col).mean()).alias("fail_rate")
                if success_col
                else pl.lit(None, dtype=pl.Float64).alias("fail_rate")
            ),
        )
        .sort("bin")
    )
    if binned.height == 0:
        return binned
    lo, hi = binned["bin"].min(), binned["bin"].max()
    grid = pl.DataFrame(
        {"bin": np.round(np.arange(lo, hi + width / 2, width), 6)}
    ).with_columns(pl.col("bin").cast(binned["bin"].dtype))
    return grid.join(binned, on="bin", how="left").with_columns(
        pl.col("count").fill_null(0)
    )


def detect(
    df: pl.DataFrame, lift: str, region: str, unit: str, z_threshold: float
) -> pl.DataFrame:
    """Candidate milestone bins for one lift x region x unit cell."""
    sub = df.filter(pl.col("region") == region)
    if lift == "total":
        sub = sub.select(pl.col("TotalKg").alias("weight_kg")).drop_nulls()
        success = None
    else:
        col = LIFTS[lift]
        sub = sub.select(
            pl.col(col).abs().alias("weight_kg"),
            (pl.col(col) > 0).alias("success"),
        )
        success = "success"

    weight = (
        pl.col("weight_kg")
        if unit == "kg"
        else (pl.col("weight_kg") / KG_PER_LB).alias("weight_lb")
    )
    sub = sub.with_columns(weight.alias("weight"))

    binned = bin_attempts(sub, "weight", success, BIN_WIDTHS[unit])
    if binned.height == 0:
        return binned

    counts = binned["count"].to_numpy()
    z = excess_z(counts)
    return binned.with_columns(
        pl.lit(lift).alias("lift"),
        pl.lit(region).alias("region"),
        pl.lit(unit).alias("unit"),
        pl.Series("excess_z", np.round(z, 2)),
        pl.Series("flagged", (z >= z_threshold) & (counts >= MIN_COUNT)),
    ).select(
        "lift", "region", "unit", "bin", "count", "fail_rate", "excess_z", "flagged"
    )


def plot_cell(cell: pl.DataFrame, lift: str, region: str, unit: str) -> Path:
    """Two stacked panels sharing x: attempt counts, then failure rate."""
    # Trim extreme tails for readability; CSV keeps the full range.
    kept = cell.filter(pl.col("count") > 0)
    lo = kept["bin"].quantile(0.005)
    hi = kept["bin"].quantile(0.995)
    view = cell.filter(pl.col("bin").is_between(lo, hi))

    bins = view["bin"].to_numpy()
    counts = view["count"].to_numpy()
    flagged = view["flagged"].to_numpy()
    width = BIN_WIDTHS[unit] * 0.9

    base, accent, ink = "#5778a4", "#e49444", "#404040"
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 6), height_ratios=[2, 1]
    )
    ax1.bar(bins, counts, width=width, color=base, linewidth=0)
    ax1.bar(bins[flagged], counts[flagged], width=width, color=accent, linewidth=0)
    for b, c in zip(bins[flagged], counts[flagged]):
        ax1.annotate(
            f"{b:g}", (b, c), textcoords="offset points", xytext=(0, 3),
            ha="center", fontsize=7, color=ink,
        )
    ax1.set_ylabel("third attempts" if lift != "total" else "totals")
    ax1.set_title(
        f"{lift} - {region}-region meets, {BIN_WIDTHS[unit]:g} {unit} bins "
        f"(highlighted = local excess)"
    )

    if view["fail_rate"].null_count() < view.height:
        ax2.plot(bins, view["fail_rate"].to_numpy(), color=ink, linewidth=1.5)
        ax2.set_ylabel("failure rate")
        ax2.set_ylim(0, 1)
    else:
        ax2.text(0.5, 0.5, "failure rate not defined for totals",
                 ha="center", va="center", transform=ax2.transAxes, color=ink)
        ax2.set_yticks([])
    ax2.set_xlabel(f"weight ({unit})")

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out = FIG_DIR / f"milestones_{lift}_{region}_{unit}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--z-threshold", type=float, default=4.0,
        help="local excess z-score above which a bin is flagged (default 4)",
    )
    args = parser.parse_args()

    if not RAW_PARQUET.exists():
        print(f"{RAW_PARQUET} not found - run `make ingest` first.", file=sys.stderr)
        return 1

    cols = ["Event", "Equipment", "Place", "MeetCountry", "TotalKg", *ATTEMPT_COLS]
    df = pl.read_parquet(RAW_PARQUET, columns=cols)
    print(f"loaded {df.height:,} rows")
    df = apply_sample_filters(df)
    print(f"sample: {df.height:,} rows")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for lift in [*LIFTS, "total"]:
        for region in ("lb", "kg"):
            for unit in ("kg", "lb"):
                cell = detect(df, lift, region, unit, args.z_threshold)
                if cell.height == 0:
                    continue
                results.append(cell)
                fig_path = plot_cell(cell, lift, region, unit)
                n_flagged = cell["flagged"].sum()
                print(f"{lift:9s} {region}-region {unit}-bins: "
                      f"{n_flagged} flagged bins -> {fig_path.name}")

    out = pl.concat(results)
    out.write_csv(OUT_CSV)
    print(f"\nwrote {out.height:,} bins ({out['flagged'].sum()} flagged) to {OUT_CSV}")

    with pl.Config(tbl_rows=-1):
        print("\nFlagged bins (candidate milestones):")
        print(
            out.filter(pl.col("flagged"))
            .sort(["lift", "region", "unit", "bin"])
            .select("lift", "region", "unit", "bin", "count", "fail_rate", "excess_z")
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
