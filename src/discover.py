"""Discovery stage: find milestone weights from attempt selection alone.

Runs after features. Reads data/processed/attempts_features.parquet and writes

- data/processed/discovered_milestones.parquet: every bin scanned, with its
  counterfactual, excess mass, corrected p-value and whether it was discovered.
- data/processed/attempts_discovered.parquet: the features table plus `split`
  and `on_discovered_milestone`. Later stages read this one.

Three rules shape this stage.

**Outcomes are never read.** A milestone list chosen partly from success rates
cannot then be tested against success rates. DISCOVERY_COLUMNS is a whitelist
and the loader selects exactly those columns, so success, best3_kg, TotalKg and
Place are not merely unused here, they never come off disk. A test asserts that
inverting success changes nothing about the output.

**Discovery uses half the meets.** Every meet is assigned to the discovery or
the test half by a salted hash of its id, so the assignment is identical on
every run and every machine. Milestones are found in the discovery half; any
outcome test on them belongs in the test half.

**Bunching is excess over a smooth counterfactual.** For each candidate bin, a
Poisson polynomial is fitted to the surrounding bins with the candidate and its
immediate neighbours removed, and the bin is compared against that prediction.
Scanning hundreds of weights produces false positives by construction, so
p-values carry a Benjamini-Hochberg correction; see correct_for_multiplicity().

Weights are binned in the meet's own loading unit, so a pound-loaded meet is
scanned on the 5 lb grid its bar actually used and a kilo meet on the 2.5 kg
grid. Groups never mix units.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

from constants import (
    DISCOVERY_BIN_WIDTH,
    DONUT,
    FDR_Q,
    KG_PER_LB,
    MIN_COUNT,
    MIN_EXCESS_PCT,
    POLY_DEGREE,
    SCAN_HIGH_Q,
    SCAN_LOW_Q,
    SPLIT_NAMES,
    SPLIT_SALT,
    WINDOW,
)
from features import (
    KG_COARSE_STEP,
    KG_STEP,
    LB_COARSE_STEP,
    LB_STEP,
    on_kg_milestone,
    on_lb_milestone_kg_meet,
    on_lb_milestone_lb_meet,
)

ROOT = Path(__file__).resolve().parents[1]
IN_PARQUET = ROOT / "data" / "processed" / "attempts_features.parquet"
MILESTONES_PARQUET = ROOT / "data" / "processed" / "discovered_milestones.parquet"
OUT_PARQUET = ROOT / "data" / "processed" / "attempts_discovered.parquet"

LIFTS = ("squat", "bench", "deadlift")

# Each group is one region x loading unit cell, scanned in its own unit so that
# no histogram ever mixes a 2.5 kg grid with a 5 lb one.
GROUPS = {
    "us_lb": {"lb_region": 1, "loading_unit": "lb", "unit": "lb"},
    "us_kg": {"lb_region": 1, "loading_unit": "kg", "unit": "kg"},
    "kilo": {"lb_region": 0, "loading_unit": "kg", "unit": "kg"},
}

# The only columns discovery is allowed to see. Outcomes are absent by
# construction, not by discipline.
DISCOVERY_COLUMNS = [
    "meet_id",
    "lifter_id",
    "lift",
    "attempt_no",
    "weight_kg",
    "lb_region",
    "loading_unit",
]


def use_utf8_stdout() -> None:
    """Polars draws tables with box-drawing characters, which the default
    Windows console encoding (cp1252) cannot encode."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def assign_split(meet_ids: pl.Series) -> pl.DataFrame:
    """Deterministic discovery/test half per meet.

    blake2b over a fixed salt and the meet id. Python's hash() is salted per
    process, so it would reshuffle the halves on every run and make the test
    half meaningless.
    """
    unique = meet_ids.unique().sort()
    halves = [
        SPLIT_NAMES[
            hashlib.blake2b(
                f"{SPLIT_SALT}{mid}".encode(), digest_size=8
            ).digest()[0]
            % 2
        ]
        for mid in unique.to_list()
    ]
    return pl.DataFrame({"meet_id": unique, "split": pl.Series(halves)})


def assign_groups(df: pl.DataFrame) -> pl.DataFrame:
    """Label each attempt with its discovery group, or null if excluded."""
    expr = pl.lit(None, dtype=pl.Utf8)
    for name, spec in GROUPS.items():
        expr = (
            pl.when(
                (pl.col("lb_region") == spec["lb_region"])
                & (pl.col("loading_unit") == spec["loading_unit"])
            )
            .then(pl.lit(name))
            .otherwise(expr)
        )
    return df.with_columns(expr.alias("group"))


def to_bin_unit(weight_kg: pl.Expr, unit: str) -> pl.Expr:
    """Weight expressed in the group's own loading unit."""
    return weight_kg if unit == "kg" else weight_kg / KG_PER_LB


def bin_counts(weights: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Counts on a regular grid, gaps zero-filled.

    The counterfactual fit needs a regular grid: an absent weight is a count of
    zero, not a missing row, and dropping it would bend the polynomial.
    """
    if len(weights) == 0:
        return np.array([]), np.array([])
    snapped = np.round(weights / width) * width
    lo, hi = snapped.min(), snapped.max()
    edges = np.round(np.arange(lo, hi + width / 2, width), 6)
    index = np.round((snapped - lo) / width).astype(int)
    counts = np.bincount(index, minlength=len(edges)).astype(float)
    return edges, counts


def fit_counterfactual(
    counts: np.ndarray,
    i: int,
    window: int = WINDOW,
    donut: int = DONUT,
    degree: int = POLY_DEGREE,
) -> float:
    """Smooth counterfactual count for bin i, fitted from its neighbours.

    A Poisson GLM on a polynomial in bin index, fitted to the window around i
    with a donut around i removed. The log link is what keeps the prediction
    positive; a polynomial fitted to raw counts by least squares dips below
    zero in the sparse tails where most of the grid lives.

    Returns nan when the neighbourhood cannot support a fit.
    """
    n = len(counts)
    lo, hi = max(0, i - window), min(n, i + window + 1)
    idx = np.arange(lo, hi)
    keep = (idx < i - donut) | (idx > i + donut)
    x, y = idx[keep], counts[lo:hi][keep]

    if len(x) < degree + 2 or y.sum() <= 0:
        return float("nan")

    # Centre and scale so the Vandermonde matrix stays conditioned.
    xs = (x - i) / max(window, 1)
    design = np.vander(xs, degree + 1, increasing=True)
    try:
        model = sm.GLM(y, design, family=sm.families.Poisson()).fit()
    except Exception:
        return float("nan")

    predicted = float(model.predict(np.vander([0.0], degree + 1, increasing=True))[0])
    return predicted if np.isfinite(predicted) else float("nan")


def scan_cell(weights: np.ndarray, width: float) -> pl.DataFrame:
    """Counterfactual, excess and a dispersion-corrected z for every bin.

    The Poisson variance is too small here: a lifter contributes three
    attempts and meets repeat weights, so counts are over-dispersed relative to
    Poisson. A per-cell dispersion is estimated from the Pearson residuals away
    from the spikes and inflates the standard error, which stops mild
    clustering from reading as significant bunching.
    """
    edges, counts = bin_counts(weights, width)
    if len(edges) == 0:
        return pl.DataFrame()

    # Trim sparse tails: a handful of attempts at 500 kg should not anchor a fit.
    occupied = edges[counts > 0]
    lo_edge = np.quantile(np.repeat(edges, counts.astype(int)), SCAN_LOW_Q)
    hi_edge = np.quantile(np.repeat(edges, counts.astype(int)), SCAN_HIGH_Q)
    in_scan = (edges >= lo_edge) & (edges <= hi_edge) if len(occupied) else counts > 0

    counterfactual = np.full(len(edges), np.nan)
    for i in np.flatnonzero(in_scan):
        counterfactual[i] = fit_counterfactual(counts, i)

    with np.errstate(invalid="ignore", divide="ignore"):
        pearson = (counts - counterfactual) / np.sqrt(counterfactual)

    # Dispersion from the body of the distribution, excluding the spikes that
    # are the thing being detected; floored at 1 so it can only widen the test.
    usable = np.isfinite(pearson)
    if usable.sum() > 10:
        cut = np.quantile(np.abs(pearson[usable]), 0.95)
        core = usable & (np.abs(pearson) <= cut)
        dispersion = max(float(np.mean(pearson[core] ** 2)), 1.0) if core.sum() else 1.0
    else:
        dispersion = 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        excess = counts - counterfactual
        z = excess / np.sqrt(dispersion * counterfactual)

    return pl.DataFrame(
        {
            "bin": edges,
            "count": counts.astype(np.int64),
            "counterfactual": counterfactual,
            "excess": excess,
            "excess_pct": excess / counterfactual,
            "z": z,
            "dispersion": np.full(len(edges), dispersion),
            "scanned": in_scan,
        }
    )


def normal_sf(z: np.ndarray) -> np.ndarray:
    """One-sided upper-tail p. Bunching is an excess; a hole is not a milestone."""
    from scipy.special import erfc  # scipy ships with statsmodels

    return 0.5 * erfc(z / np.sqrt(2.0))


def correct_for_multiplicity(p: np.ndarray, q: float = FDR_Q) -> dict[str, np.ndarray]:
    """Benjamini-Hochberg FDR, with Bonferroni reported alongside.

    Why BH rather than Bonferroni. Around 1,400 bins are scanned in a run, and
    the output is a *list* of candidate milestones. What matters is that the
    list is mostly right, not that it is certain to contain no error at all.
    Bonferroni controls the latter and would need p < 4e-5 per bin, discarding
    real secondary milestones to buy a guarantee nobody is asking for. BH
    controls the expected share of false discoveries among what is reported,
    which is the quantity the list is actually judged on.

    BH requires independence or positive dependence. Neighbouring bins share
    overlapping counterfactual windows, so the tests are positively correlated,
    which is the regime BH is valid in. The Bonferroni-surviving subset is
    carried through as the conservative core.
    """
    finite = np.isfinite(p)
    m = int(finite.sum())
    p_bh = np.full(len(p), np.nan)
    p_bonf = np.full(len(p), np.nan)
    if m == 0:
        return {"p_bh": p_bh, "p_bonferroni": p_bonf, "n_tests": m}

    idx = np.flatnonzero(finite)
    order = idx[np.argsort(p[idx], kind="stable")]
    ranks = np.arange(1, m + 1)

    # Step-up: enforce monotonicity from the largest p downwards.
    adjusted = np.minimum.accumulate((p[order] * m / ranks)[::-1])[::-1]
    p_bh[order] = np.clip(adjusted, 0, 1)
    p_bonf[idx] = np.clip(p[idx] * m, 0, 1)
    return {"p_bh": p_bh, "p_bonferroni": p_bonf, "n_tests": m}


def prespecified_kind(bin_kg: pl.Expr, unit: str) -> pl.Expr:
    """Which hand-written milestone, if any, this bin coincides with.

    Reuses the flag helpers from the features stage so the discovered list is
    compared against exactly the definitions the regressions use.
    """
    on_lb = on_lb_milestone_lb_meet if unit == "lb" else on_lb_milestone_kg_meet
    return (
        pl.when(on_lb(bin_kg, LB_COARSE_STEP))
        .then(pl.lit("lb100"))
        .when(on_lb(bin_kg, LB_STEP))
        .then(pl.lit("lb50"))
        .when(on_kg_milestone(bin_kg, KG_COARSE_STEP))
        .then(pl.lit("kg50"))
        .when(on_kg_milestone(bin_kg, KG_STEP))
        .then(pl.lit("kg10"))
        .otherwise(pl.lit("none"))
    )


def discover(df: pl.DataFrame) -> pl.DataFrame:
    """Scan every lift x group cell in the discovery half.

    Third attempts only: that is the attempt where reaching a round number is
    worth stretching for, and the one the design is about.
    """
    thirds = df.filter((pl.col("attempt_no") == 3) & (pl.col("split") == "discovery"))
    cells = []

    for group, spec in GROUPS.items():
        unit = spec["unit"]
        width = DISCOVERY_BIN_WIDTH[unit]
        for lift in LIFTS:
            sub = thirds.filter(
                (pl.col("group") == group) & (pl.col("lift") == lift)
            ).select(to_bin_unit(pl.col("weight_kg"), unit).alias("w"))
            if sub.height == 0:
                continue

            cell = scan_cell(sub["w"].to_numpy(), width)
            if cell.height == 0:
                continue

            cells.append(
                cell.with_columns(
                    pl.lit(group).alias("group"),
                    pl.lit(lift).alias("lift"),
                    pl.lit(unit).alias("bin_unit"),
                    (
                        pl.col("bin")
                        if unit == "kg"
                        else pl.col("bin") * KG_PER_LB
                    ).alias("bin_kg"),
                )
            )
            print(
                f"scanned {group:6s} {lift:9s}: {sub.height:>8,} third attempts, "
                f"{int(cell['scanned'].sum()):>4} bins"
            )

    if not cells:
        return pl.DataFrame()

    out = pl.concat(cells)
    z = out["z"].to_numpy()
    p = np.where(np.isfinite(z) & out["scanned"].to_numpy(), normal_sf(z), np.nan)
    corrected = correct_for_multiplicity(p)

    out = out.with_columns(
        pl.Series("p", p),
        pl.Series("p_bh", corrected["p_bh"]),
        pl.Series("p_bonferroni", corrected["p_bonferroni"]),
    ).with_columns(
        (
            (pl.col("p_bh") <= FDR_Q)
            & (pl.col("count") >= MIN_COUNT)
            & (pl.col("excess_pct") >= MIN_EXCESS_PCT)
        )
        .fill_null(False)
        .alias("discovered"),
        (pl.col("p_bonferroni") <= FDR_Q).fill_null(False).alias("survives_bonferroni"),
    )

    kind = pl.lit("none")
    for unit in ("kg", "lb"):
        kind = (
            pl.when(pl.col("bin_unit") == unit)
            .then(prespecified_kind(pl.col("bin_kg"), unit))
            .otherwise(kind)
        )
    out = out.with_columns(kind.alias("prespecified_kind"))

    print(f"\n{corrected['n_tests']:,} bins tested, BH at q={FDR_Q}")
    return out.select(
        "group", "lift", "bin", "bin_unit", "bin_kg", "count", "counterfactual",
        "excess", "excess_pct", "z", "dispersion", "p", "p_bh", "p_bonferroni",
        "scanned", "discovered", "survives_bonferroni", "prespecified_kind",
    )


def add_discovered_flag(df: pl.DataFrame, milestones: pl.DataFrame) -> pl.DataFrame:
    """Flag attempts whose weight sits on a discovered milestone.

    Applies to every attempt number, not just thirds: the flag describes the
    weight, and discovery only used thirds to find which weights matter. Null
    outside the three groups, the same way lb_region is null where the region
    is ambiguous.
    """
    found = milestones.filter(pl.col("discovered")).select("group", "lift", "bin")
    if found.height == 0:
        return df.with_columns(pl.lit(None, dtype=pl.Boolean).alias("on_discovered_milestone"))

    binned = pl.lit(None, dtype=pl.Float64)
    for group, spec in GROUPS.items():
        width = DISCOVERY_BIN_WIDTH[spec["unit"]]
        snapped = (
            to_bin_unit(pl.col("weight_kg"), spec["unit"]) / width
        ).round() * width
        binned = (
            pl.when(pl.col("group") == group).then(snapped).otherwise(binned)
        )

    df = df.with_columns(binned.round(6).alias("_bin"))
    found = found.with_columns(pl.col("bin").round(6))

    return (
        df.join(
            found.with_columns(pl.lit(True).alias("_hit")),
            left_on=["group", "lift", "_bin"],
            right_on=["group", "lift", "bin"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("group").is_null())
            .then(pl.lit(None, dtype=pl.Boolean))
            .otherwise(pl.col("_hit").fill_null(False))
            .alias("on_discovered_milestone")
        )
        .drop("_bin", "_hit")
    )


def report(milestones: pl.DataFrame) -> None:
    """Per group: what was found, what was already pre-specified, what is new."""
    found = milestones.filter(pl.col("discovered"))
    print(f"\ndiscovered {found.height} milestone bins "
          f"({int(found['survives_bonferroni'].sum())} survive Bonferroni)")

    for group in GROUPS:
        sub = found.filter(pl.col("group") == group).sort("lift", "bin")
        unit = GROUPS[group]["unit"]
        known = sub.filter(pl.col("prespecified_kind") != "none")
        new = sub.filter(pl.col("prespecified_kind") == "none")
        print(f"\n--- {group} ({unit} bins) --- {sub.height} found, "
              f"{known.height} pre-specified, {new.height} new")
        if sub.height == 0:
            print("    nothing cleared the threshold")
            continue
        for r in sub.to_dicts():
            tag = r["prespecified_kind"]
            star = "*" if r["survives_bonferroni"] else " "
            label = f"{r['bin']:g} {unit}"
            if unit == "lb":
                label += f" ({r['bin_kg']:.1f} kg)"
            print(
                f"  {star} {r['lift']:9s} {label:>22s}  n={r['count']:>6,}  "
                f"excess={r['excess_pct']:>6.1%}  z={r['z']:>6.2f}  "
                f"{'[' + tag + ']' if tag != 'none' else '[new]'}"
            )
    print("\n  * = also survives Bonferroni")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=IN_PARQUET)
    parser.add_argument("--out", type=Path, default=OUT_PARQUET)
    parser.add_argument("--milestones", type=Path, default=MILESTONES_PARQUET)
    args = parser.parse_args()
    use_utf8_stdout()

    if not args.src.exists():
        print(f"{args.src} not found - run the features stage first.", file=sys.stderr)
        return 1

    # Outcomes are never loaded. See DISCOVERY_COLUMNS.
    scan_frame = pl.read_parquet(args.src, columns=DISCOVERY_COLUMNS)
    print(f"loaded {scan_frame.height:,} attempts ({len(DISCOVERY_COLUMNS)} columns)")

    splits = assign_split(scan_frame["meet_id"])
    scan_frame = assign_groups(scan_frame.join(splits, on="meet_id", how="left"))

    counts = splits.group_by("split").len().sort("split")
    print("meets per split: " + ", ".join(
        f"{r['split']}={r['len']:,}" for r in counts.to_dicts()))
    excluded = scan_frame.filter(pl.col("group").is_null()).height
    print(f"excluded from discovery: {excluded:,} attempts "
          f"({excluded / scan_frame.height:.1%}) - ambiguous or mixed-unit meets")

    milestones = discover(scan_frame)
    if milestones.height == 0:
        print("no bins scanned", file=sys.stderr)
        return 1

    args.milestones.parent.mkdir(parents=True, exist_ok=True)
    milestones.write_parquet(args.milestones)
    print(f"wrote {milestones.height:,} scanned bins to {args.milestones}")

    # Rebuild the full table: features + split + the discovered flag.
    full = assign_groups(pl.read_parquet(args.src).join(splits, on="meet_id", how="left"))
    full = add_discovered_flag(full, milestones).drop("group")

    full.write_parquet(args.out)
    print(f"wrote {full.height:,} attempts to {args.out}")

    report(milestones)
    return 0


if __name__ == "__main__":
    sys.exit(main())
