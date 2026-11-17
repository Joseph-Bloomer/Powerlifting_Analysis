"""Diagnostics on the three results that undercut the headline estimate.

Reads data/processed/attempts_discovered.parquet and writes
reports/diagnostics.txt, reports/diagnostics.csv and two figures.

The econ stage left three warnings. The lifter fixed effect takes the key
coefficient from -0.72pp to -0.22pp. The +12.5 kg placebo is significant with
the wrong sign. The asymmetry test moves more in the kilo region than the pound
one. Each has an innocent explanation and a damning one, and nothing in the
econ output separates them.

**Nothing here changes the headline specification and nothing is tuned to
recover significance.** This module imports econ_models and calls its
specification; it never edits it. Every coefficient is reported with a
confidence interval, nulls included and labelled as nulls rather than as
absence of evidence. Each section ends in one line of what the result implies,
scoped to that result.

Diagnostic 3 is exploratory and post hoc, and is labelled so wherever it
appears: in the table title, the figure title and the CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

import plotstyle as ps
from constants import KG_PER_LB, KG_STEP, LB_STEP, PLATE_STEP_KG, PLATE_STEP_LB
from econ_models import (
    ASYMMETRY_WINDOW_KG,
    BASE_TERMS,
    FE_BASE,
    FE_LIFTER,
    Fit,
    add_running_variable,
    build_sample,
    fit_model,
    key_term,
    on_fake_milestone,
    side_by_side,
    tidy,
)

ROOT = Path(__file__).resolve().parents[1]
IN_PARQUET = ROOT / "data" / "processed" / "attempts_discovered.parquet"
CSV_PATH = ROOT / "reports" / "diagnostics.csv"
TXT_PATH = ROOT / "reports" / "diagnostics.txt"
FIG_DIR = ROOT / "reports" / "figures"

# Placebo family, on each meet's own loadable grid. A kilo offset applied to a
# pound-loaded meet lands between pound weights, where no attempt can ever sit,
# which would leave the flag constant and unidentified.
PLACEBO_OFFSETS_KG = [2.5 * i for i in range(1, 9)]  # 2.5 .. 20.0 kg
PLACEBO_OFFSETS_LB = [5.0 * i for i in range(1, 9)]  # 5 .. 40 lb

PRIOR_MEET_BANDS = [
    ("0 (first meet)", 0, 0),
    ("1-2", 1, 2),
    ("3-5", 3, 5),
    ("6-10", 6, 10),
    ("11+", 11, 10_000),
]

DIST_BIN_KG = 2.5
PREV_BIN_KG = 2.5
MAX_DIST_KG = 25.0

GROUPS = (("us_lb", "US pound-loaded"), ("us_kg", "US kilo-loaded"), ("kilo", "Kilo countries"))


def use_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def interval(fit: Fit, term: str) -> str:
    """A coefficient with its interval, in percentage points."""
    row = fit.row(term)
    if row is None:
        return "not estimated"
    return (
        f"{row['pp']:+.3f}pp  95% CI [{row['ci_lo_pp']:+.3f}, {row['ci_hi_pp']:+.3f}]"
        f"  p={row['p']:.3g}"
    )


def spans_zero(fit: Fit, term: str) -> bool:
    row = fit.row(term)
    return row is not None and row["ci_lo_pp"] <= 0 <= row["ci_hi_pp"]


def assign_group(df: pl.DataFrame) -> pl.DataFrame:
    expr = pl.lit(None, dtype=pl.Utf8)
    for name, spec in (
        ("us_lb", (1, "lb")),
        ("us_kg", (1, "kg")),
        ("kilo", (0, "kg")),
    ):
        expr = (
            pl.when(
                (pl.col("lb_region") == spec[0]) & (pl.col("loading_unit") == spec[1])
            )
            .then(pl.lit(name))
            .otherwise(expr)
        )
    return df.with_columns(expr.alias("group"))


# --- 1. the fixed-effects identifying sample --------------------------------


def identifying_sample(sample: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Lifters who identify the fixed-effects estimate.

    Joe's definition: more than one meet, and milestone status that varies
    within the lifter. A lifter with one meet, or who is never on a milestone,
    contributes nothing to a within-lifter comparison.
    """
    per_lifter = sample.group_by("lifter_id").agg(
        pl.col("meet_id").n_unique().alias("meets"),
        pl.col("on_lb_milestone").n_unique().alias("ms_levels"),
        pl.col("lb_region").n_unique().alias("region_levels"),
    )
    keep = per_lifter.filter((pl.col("meets") > 1) & (pl.col("ms_levels") > 1))
    stats = {
        "lifters_total": per_lifter.height,
        "lifters_multi_meet": per_lifter.filter(pl.col("meets") > 1).height,
        "lifters_ms_varies": per_lifter.filter(pl.col("ms_levels") > 1).height,
        "lifters_identifying": keep.height,
        "lifters_multi_region": per_lifter.filter(pl.col("region_levels") > 1).height,
    }
    return sample.join(keep.select("lifter_id"), on="lifter_id", how="semi"), stats


def diagnostic_1(sample: pl.DataFrame) -> tuple[list[Fit], list[str]]:
    restricted, stats = identifying_sample(sample)
    fits = [
        fit_model(sample, "full cross-section", BASE_TERMS, FE_BASE),
        fit_model(restricted, "restricted cross-section", BASE_TERMS, FE_BASE,
                  note=">=2 meets and milestone status varies within lifter"),
        fit_model(sample, "lifter FE", BASE_TERMS, FE_LIFTER),
    ]
    fits = [f for f in fits if f]

    lines = [
        f"lifters in the headline sample        : {stats['lifters_total']:,}",
        f"  with more than one meet             : {stats['lifters_multi_meet']:,}",
        f"  with milestone status that varies   : {stats['lifters_ms_varies']:,}",
        f"  both, i.e. identifying the FE       : {stats['lifters_identifying']:,}",
        f"  competing in more than one region   : {stats['lifters_multi_region']:,}"
        "   <- all that identifies lb_region under lifter FE",
    ]
    full, restr, fe = (f.row("lb_ms_x_region") for f in fits)
    if restr and full and fe:
        closer = "the restricted sample" if abs(restr["pp"] - fe["pp"]) < abs(
            restr["pp"] - full["pp"]
        ) else "the full sample"
        lines.append(
            f"Implies: restricting the cross-section to the identifying lifters moves "
            f"the estimate to {restr['pp']:+.3f}pp, closer to {closer}."
        )
    return fits, lines


# --- 2. with and without jump_pct -------------------------------------------


def diagnostic_2(sample: pl.DataFrame) -> tuple[list[Fit], list[str]]:
    without = [t for t in BASE_TERMS if t != "jump_pct"]
    fits = [
        fit_model(sample, "with jump_pct", BASE_TERMS, FE_BASE),
        fit_model(sample, "without jump_pct", without, FE_BASE,
                  note="jump_pct is plausibly a mediator: a milestone changes the "
                       "jump a lifter picks"),
    ]
    fits = [f for f in fits if f]
    a, b = (f.row("lb_ms_x_region") for f in fits)
    lines = [
        f"Implies: dropping the jump control moves the coefficient from "
        f"{a['pp']:+.3f}pp to {b['pp']:+.3f}pp, a change of {b['pp'] - a['pp']:+.3f}pp."
    ]
    return fits, lines


# --- 3. exploratory: do not condition on the chosen weight ------------------


def binned(col: pl.Expr, width: float) -> pl.Expr:
    return (col / width).floor() * width


def diagnostic_3(sample: pl.DataFrame) -> tuple[list[Fit], list[str], pl.DataFrame]:
    """EXPLORATORY, POST HOC. Distance measured from the second attempt.

    on_lb_milestone describes the weight the lifter *chose*, so conditioning on
    it conditions on an outcome of the decision. The distance from the second
    attempt to the next milestone is fixed before the third attempt is picked.
    Second-attempt weight is controlled flexibly, as absorbed 2.5 kg bins,
    rather than with a functional form.
    """
    rows, fits = [], []
    for dist_col, grid in (("dist_to_next_lb_ms", "pound"), ("dist_to_next_kg_ms", "kilo")):
        for group, label in GROUPS:
            sub = sample.filter(
                (pl.col("group") == group)
                & pl.col(dist_col).is_not_null()
                & (pl.col(dist_col) <= MAX_DIST_KG)
            ).with_columns(
                binned(pl.col(dist_col), DIST_BIN_KG).alias("dist_bin"),
                binned(pl.col("prev_weight_kg"), PREV_BIN_KG).alias("prev_bin"),
            )
            if sub.height < 1000:
                continue

            for outcome in ("jump_kg", "success"):
                # The profile across distance bins, each against the far bin.
                levels = sorted(sub["dist_bin"].unique().to_list())
                terms = []
                work = sub
                for lv in levels[:-1]:
                    name = f"dist_{lv:g}"
                    work = work.with_columns(
                        (pl.col("dist_bin") == lv).cast(pl.Int8).alias(name)
                    )
                    terms.append(name)

                fit = fit_model(
                    work,
                    f"[exploratory] {label} / {grid} grid / {outcome}",
                    terms,
                    [*FE_BASE, "prev_bin"],
                    outcome=outcome,
                    note="EXPLORATORY, POST HOC; distance measured from the second "
                         "attempt; second-attempt weight absorbed in 2.5 kg bins; "
                         f"omitted category is the {levels[-1]:g} kg bin",
                )
                if fit is None:
                    continue
                fits.append(fit)
                for lv in levels[:-1]:
                    row = fit.row(f"dist_{lv:g}")
                    if row:
                        rows.append({
                            "grid": grid, "group": group, "outcome": outcome,
                            "dist_bin": lv, "coef": row["coef"],
                            "lo": row["coef"] - 1.96 * row["se_pp"] / 100,
                            "hi": row["coef"] + 1.96 * row["se_pp"] / 100,
                            "n": fit.n,
                        })

    profile = pl.DataFrame(rows) if rows else pl.DataFrame()
    lines = ["Implies: exploratory and post hoc; the profiles are in "
             "diagnostics_exploratory_profile.png and the CSV, not summarised here."]
    return fits, lines, profile


# --- 4. the placebo family --------------------------------------------------


def with_placebo(sample: pl.DataFrame, offset_kg: float, offset_lb: float) -> pl.DataFrame:
    """Fake milestone flag at one offset, on each meet's own loadable grid."""
    expr = pl.lit(False)
    for unit, shift in (("kg", offset_kg), ("lb", offset_lb * KG_PER_LB)):
        expr = (
            pl.when(pl.col("loading_unit") == unit)
            .then(on_fake_milestone(pl.col("weight_kg"), unit, LB_STEP, offset_kg=shift))
            .otherwise(expr)
        )
    return sample.with_columns(expr.cast(pl.Int8).alias("on_fake_milestone")).with_columns(
        (pl.col("on_fake_milestone") * pl.col("lb_region")).alias("fake_ms_x_region")
    )


def diagnostic_4(sample: pl.DataFrame) -> tuple[list[Fit], list[str], pl.DataFrame]:
    placebo_terms = [
        "jump_pct", "on_fake_milestone", "on_kg_milestone", "lb_region",
        "fake_ms_x_region", "kg_ms_x_region", "prior_meets", "would_be_pr",
        "no_history",
    ]
    rows, fits = [], []

    true_fit = fit_model(sample, "offset 0 (true milestones)", BASE_TERMS, FE_BASE)
    if true_fit:
        fits.append(true_fit)
        r = true_fit.row("lb_ms_x_region")
        rows.append({"offset_kg": 0.0, "offset_lb": 0.0, "pp": r["pp"],
                     "lo": r["ci_lo_pp"], "hi": r["ci_hi_pp"], "p": r["p"],
                     "is_true": True, "n": true_fit.n})

    for off_kg, off_lb in zip(PLACEBO_OFFSETS_KG, PLACEBO_OFFSETS_LB):
        shifted = with_placebo(sample, off_kg, off_lb)
        fit = fit_model(shifted, f"placebo +{off_kg:g} kg / +{off_lb:g} lb",
                        placebo_terms, FE_BASE)
        if fit is None:
            continue
        fits.append(fit)
        r = fit.row("fake_ms_x_region")
        if r:
            rows.append({"offset_kg": off_kg, "offset_lb": off_lb, "pp": r["pp"],
                         "lo": r["ci_lo_pp"], "hi": r["ci_hi_pp"], "p": r["p"],
                         "is_true": False, "n": fit.n})

    family = pl.DataFrame(rows) if rows else pl.DataFrame()
    lines = []
    if family.height:
        fake = family.filter(~pl.col("is_true"))
        n_sig = int((fake["p"] < 0.05).sum())
        lines.append(
            f"Implies: {n_sig} of {fake.height} placebo offsets are significant at 5% "
            f"(range {fake['pp'].min():+.3f}pp to {fake['pp'].max():+.3f}pp); "
            f"the true estimate is {family.filter(pl.col('is_true'))['pp'][0]:+.3f}pp."
        )
    return fits, lines, family


# --- 5. the asymmetry test, both ways ---------------------------------------


def add_kg_running_variable(df: pl.DataFrame) -> pl.DataFrame:
    """Signed kg from the previous attempt to the nearest *kilo* milestone.

    The pound version lives in econ_models.add_running_variable and is left
    alone; this is its mirror on the 10 kg grid.
    """
    nearest = (pl.col("prev_weight_kg") / KG_STEP).round() * KG_STEP
    return df.with_columns((pl.col("prev_weight_kg") - nearest).alias("running_kg"))


def milestone_grid(unit: str, lo_kg: float, hi_kg: float) -> list[float]:
    """The actual milestone weights in range, as the test would see them."""
    out = []
    if unit == "kg":
        step = KG_STEP
        v = np.ceil(lo_kg / step) * step
        while v <= hi_kg:
            out.append(round(float(v), 2))
            v += step
    else:
        v = np.ceil(lo_kg / KG_PER_LB / LB_STEP) * LB_STEP
        while v * KG_PER_LB <= hi_kg:
            exact = v * KG_PER_LB
            loadable = np.ceil(exact / PLATE_STEP_KG - 1e-9) * PLATE_STEP_KG
            out.append(round(float(loadable), 2))
            v += LB_STEP
    return out


def diagnostic_5(sample: pl.DataFrame) -> tuple[list[Fit], list[str]]:
    terms = ["running_kg", "above", "running_x_above"]
    fits, lines = [], []

    lo, hi = float(sample["prev_weight_kg"].quantile(0.01)), float(
        sample["prev_weight_kg"].quantile(0.99)
    )
    lines.append("Milestone weights the asymmetry test snaps to, 1st-99th percentile "
                 f"of second-attempt weight ({lo:.1f}-{hi:.1f} kg):")
    lines.append(f"  pound grid (50 lb), as loaded in kilo meets: "
                 f"{', '.join(f'{v:g}' for v in milestone_grid('lb', lo, hi))}")
    lines.append(f"  kilo grid (10 kg)                          : "
                 f"{', '.join(f'{v:g}' for v in milestone_grid('kg', lo, hi))}")
    lines.append("The econ stage used the POUND grid in both regions: "
                 "add_running_variable branches on the meet's loading unit, not "
                 "its region.")

    # The two grids are not disjoint, which the design reads as if they were.
    kg_grid = set(milestone_grid("kg", lo, hi))
    lb_grid = set(milestone_grid("lb", lo, hi))
    shared = sorted(kg_grid & lb_grid)
    lines.append("")
    lines.append(
        f"GRID OVERLAP: {len(shared)} of {len(lb_grid)} pound milestones "
        f"({len(shared) / max(len(lb_grid), 1):.0%}) are also kilo milestones: "
        f"{', '.join(f'{v:g}' for v in shared)} kg."
    )
    lines.append(
        "  The loadable pound grid is mostly spaced 22.5 kg against the kilo "
        "grid's 10 kg, so coincidences fall roughly every 90 kg. At those "
        "weights on_lb_milestone and on_kg_milestone are both 1 and no "
        "comparison can separate them."
    )

    for grid, adder in (("pound", add_running_variable), ("kilo", add_kg_running_variable)):
        framed = adder(sample)
        window = framed.filter(
            pl.col("running_kg").is_not_null()
            & (pl.col("running_kg").abs() <= ASYMMETRY_WINDOW_KG)
        ).with_columns(
            (pl.col("running_kg") > 0).cast(pl.Int8).alias("above")
        ).with_columns(
            (pl.col("running_kg") * pl.col("above")).alias("running_x_above")
        )
        for region, flag in (("lb region", 1), ("kg region", 0)):
            whose = "own" if (grid == "pound") == (flag == 1) else "other's"
            fit = fit_model(
                window.filter(pl.col("lb_region") == flag),
                f"{region} vs {grid} milestones ({whose})",
                terms, FE_BASE, outcome="jump_kg",
                note=f"outcome jump_kg in kg; |window| <= {ASYMMETRY_WINDOW_KG:g} kg",
            )
            if fit:
                fits.append(fit)

    for fit in fits:
        lines.append(f"  {fit.name:46s} slope difference "
                     f"{fit.row('running_x_above')['coef']:+.4f} kg/kg  "
                     f"[{fit.row('running_x_above')['ci_lo_pp'] / 100:+.4f}, "
                     f"{fit.row('running_x_above')['ci_hi_pp'] / 100:+.4f}]")
    lines.append("Implies: the econ asymmetry result compared both regions against "
                 "the pound grid; the own/other split above separates grid from region.")
    return fits, lines


# --- 6. heterogeneity by prior_meets ----------------------------------------


def diagnostic_6(sample: pl.DataFrame) -> tuple[list[Fit], list[str]]:
    fits = []
    for label, lo, hi in PRIOR_MEET_BANDS:
        band = sample.filter(pl.col("prior_meets").is_between(lo, hi))
        fit = fit_model(band, f"prior_meets {label}", BASE_TERMS, FE_BASE)
        if fit:
            fits.append(fit)

    lines = []
    for fit in fits:
        lines.append(f"  {fit.name:24s} n={fit.n:>9,}  lifters={fit.n_clusters:>8,}  "
                     f"{interval(fit, key_term(fit))}")
    spanning = sum(spans_zero(f, key_term(f)) for f in fits)
    lines.append(f"Implies: {len(fits) - spanning} of {len(fits)} experience bands have "
                 "an interval excluding zero.")
    return fits, lines


# --- figures ----------------------------------------------------------------


def plot_placebo_family(family: pl.DataFrame) -> Path | None:
    if family.height == 0:
        return None
    fam = family.sort("offset_kg")
    x = fam["offset_kg"].to_numpy()
    y = fam["pp"].to_numpy()
    lo, hi = fam["lo"].to_numpy(), fam["hi"].to_numpy()
    is_true = fam["is_true"].to_numpy()

    fig, axes = ps.new_figure(1, (11, 5.4))
    ax = axes[0]
    ax.axhline(0, color=ps.AXIS, linewidth=1.2)
    ax.vlines(x, lo, hi, color=ps.GRID, linewidth=2.0, zorder=1)
    ax.scatter(x[~is_true], y[~is_true], s=54, color=ps.BAR, zorder=3, label="placebo offset")
    ax.scatter(x[is_true], y[is_true], s=90, color=ps.FOUND, marker="D", zorder=4,
               label="true milestones (offset 0)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.1f}")
    ps.style_axes(
        ax,
        xlabel="milestone offset (kg in kilo meets; the equivalent 5 lb step in pound meets)",
        ylabel="coefficient (pp)",
        title="Headline coefficient against fake milestones at every offset",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_2)
    fig.tight_layout()
    return ps.save(fig, FIG_DIR / "diagnostics_placebo_family.png")


def plot_exploratory_profile(profile: pl.DataFrame) -> Path | None:
    if profile.height == 0:
        return None
    sub = profile.filter(pl.col("grid") == "pound")
    if sub.height == 0:
        return None

    colours = {"us_lb": ps.FOUND, "us_kg": ps.BAR, "kilo": ps.PRESPEC_KG}
    fig, axes = ps.new_figure(2, (11, 7.6))
    for ax, outcome, ylab in zip(axes, ("jump_kg", "success"),
                                 ("mean jump (kg)", "success (pp)")):
        cell_any = False
        for group, label in GROUPS:
            cell = sub.filter(
                (pl.col("group") == group) & (pl.col("outcome") == outcome)
            ).sort("dist_bin")
            if cell.height == 0:
                continue
            cell_any = True
            scale = 100.0 if outcome == "success" else 1.0
            x = cell["dist_bin"].to_numpy()
            ax.fill_between(x, cell["lo"].to_numpy() * scale, cell["hi"].to_numpy() * scale,
                            color=colours[group], alpha=0.16, linewidth=0)
            ax.plot(x, cell["coef"].to_numpy() * scale, color=colours[group],
                    linewidth=2.0, marker="o", markersize=4)
        ax.axhline(0, color=ps.AXIS, linewidth=1.0)
        ps.style_axes(
            ax,
            xlabel="kg from the second attempt to the next pound milestone",
            ylabel=ylab,
            title=f"outcome: {outcome}",
        )
        if not cell_any:
            ax.set_axis_off()

    ps.series_legend(fig, [(colours[g], lab) for g, lab in GROUPS])
    ps.suptitle(fig, "EXPLORATORY, POST HOC - not conditioning on the chosen weight")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return ps.save(fig, FIG_DIR / "diagnostics_exploratory_profile.png")


# --- driver -----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=IN_PARQUET)
    parser.add_argument("--only", default="", help="comma-separated section numbers")
    args = parser.parse_args()
    use_utf8_stdout()

    if not args.src.exists():
        print(f"{args.src} not found - run the discover stage first.", file=sys.stderr)
        return 1

    wanted = {int(x) for x in args.only.split(",") if x.strip()} or {1, 2, 3, 4, 5, 6}
    sample = assign_group(build_sample(pl.read_parquet(args.src)))
    print(f"\nsample: {sample.height:,} attempts, "
          f"{sample['lifter_id'].n_unique():,} lifters\n")

    sections, all_fits = [], []

    def section(num, title, fits, lines, terms):
        all_fits.extend(fits)
        body = side_by_side(fits, terms, f"{num}. {title}") if fits else "(no models)"
        sections.append(body + "\n\n" + "\n".join(lines))
        print(f"\n--- {num}. {title} ---")
        for line in lines:
            print(" ", line)

    profile = family = pl.DataFrame()

    if 1 in wanted:
        fits, lines = diagnostic_1(sample)
        section(1, "THE FIXED-EFFECTS IDENTIFYING SAMPLE", fits, lines, BASE_TERMS)
    if 2 in wanted:
        fits, lines = diagnostic_2(sample)
        section(2, "WITH AND WITHOUT THE JUMP CONTROL", fits, lines, BASE_TERMS)
    if 3 in wanted:
        fits, lines, profile = diagnostic_3(sample)
        section(3, "EXPLORATORY, POST HOC: NOT CONDITIONING ON THE CHOSEN WEIGHT",
                [], lines, [])
        all_fits.extend(fits)
    if 4 in wanted:
        fits, lines, family = diagnostic_4(sample)
        section(4, "THE PLACEBO FAMILY", fits,
                lines, ["on_fake_milestone", "fake_ms_x_region", "lb_ms_x_region"])
    if 5 in wanted:
        fits, lines = diagnostic_5(sample)
        section(5, "ASYMMETRY: OWN MILESTONES AND THE OTHER REGION'S", fits, lines,
                ["running_kg", "above", "running_x_above"])
    if 6 in wanted:
        fits, lines = diagnostic_6(sample)
        section(6, "HETEROGENEITY BY PRIOR MEETS", fits, lines, BASE_TERMS)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    for path in (plot_placebo_family(family), plot_exploratory_profile(profile)):
        if path:
            print(f"wrote {path.name}")

    if all_fits:
        out = tidy(all_fits)
        if profile.height:
            out = out.with_columns(pl.lit(None, dtype=pl.Utf8).alias("section"))
        out.write_csv(CSV_PATH)
        print(f"wrote {CSV_PATH}")
    if family.height:
        family.write_csv(CSV_PATH.with_name("diagnostics_placebo_family.csv"))
    if profile.height:
        profile.write_csv(CSV_PATH.with_name("diagnostics_exploratory_profile.csv"))

    TXT_PATH.write_text("\n\n\n".join(sections), encoding="utf-8")
    print(f"wrote {TXT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
