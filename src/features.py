"""Features stage: milestone distances, milestone flags, region and PR status.

Reads data/interim/attempts.parquet and writes
data/processed/attempts_features.parquet, one row per attempt as before, with
the variables the reference-dependence design needs.

Nothing here may use information from after the attempt it describes
(CLAUDE.md). Two consequences worth stating plainly:

- The milestone *distances* are measured from prev_weight_kg, not from the
  attempt itself. They describe the position the lifter was standing in when
  they chose this attempt, which is what the design is about. Attempt 1 has no
  predecessor, so its distances are null by construction.
- The milestone *flags* describe the attempted weight. That is the choice being
  studied, not an outcome, so it is allowed on the right-hand side.

success, best3_kg, TotalKg and Place are outcomes and are carried through for
use as dependent variables only; POST_ATTEMPT_COLUMNS names them and
FEATURE_COLUMNS deliberately excludes them.

Milestones are defined by the meet's loading unit, which the clean stage
decides once per meet. Never attempt by attempt: a kilo weight that happens to
fall near a round pound figure must not be read as a pound milestone.

- Kilo-loaded meets: the lifter cannot load the pound figure itself, so a pound
  milestone is the smallest loadable 2.5 kg weight at or above it. 500 lb
  (226.796 kg) becomes 227.5 kg. That offset is the identification: 227.5 kg is
  a round number to a lifter thinking in pounds and an unremarkable one to a
  lifter thinking in kilos.
- Pound-loaded meets: the bar is loaded in pounds, so the milestone is the
  pound figure exactly, 226.796 kg, with no kilo rounding.
- Mixed meets carry a null loading unit and get null milestone variables rather
  than a guess. They are a fraction of a percent of attempts.

Distances are returned in kilos whatever the loading unit, so the pooled
headline model can read one column; dist_to_next_lb_ms_lb repeats the pound
gap in pounds for plots and pound-meet work. loading_unit is for description
and one robustness split, not a second identification strategy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from constants import (
    AMBIGUOUS_COUNTRIES,
    KG_COARSE_STEP,
    KG_PER_LB,
    KG_STEP,
    LB_COARSE_STEP,
    LB_REGION_COUNTRY,
    LB_STEP,
    LB_TOL,
    PLATE_STEP_KG,
    PLATE_STEP_LB,
    TOL,
)

ROOT = Path(__file__).resolve().parents[1]
IN_PARQUET = ROOT / "data" / "interim" / "attempts.parquet"
OUT_PARQUET = ROOT / "data" / "processed" / "attempts_features.parquet"


SERIES = ["lifter_id", "meet_id", "lift"]

# Outcomes. Never valid on the right-hand side of a model.
POST_ATTEMPT_COLUMNS = ["success", "best3_kg", "TotalKg", "Place"]

# Everything known at the moment the attempt was chosen.
FEATURE_COLUMNS = [
    "lift",
    "attempt_no",
    "weight_kg",
    "weight_lb",
    "prev_weight_kg",
    "prev_success",
    "jump_kg",
    "jump_pct",
    "dist_to_next_lb_ms",
    "dist_to_next_lb_ms_lb",
    "dist_to_next_lb100_ms",
    "dist_to_next_kg_ms",
    "dist_to_next_kg50_ms",
    "on_lb_milestone",
    "on_lb100_milestone",
    "on_kg_milestone",
    "on_kg50_milestone",
    "lb_region",
    "loading_unit",
    "best_before_kg",
    "would_be_pr",
    "prior_meets",
    "prior_best_kg",
    "Sex",
    "Age",
    "BodyweightKg",
    "WeightClassKg",
    "Division",
    "Federation",
    "ParentFederation",
    "Tested",
    "MeetCountry",
    "MeetState",
    "Date",
]


def next_multiple_above(x: pl.Expr, step: float) -> pl.Expr:
    """Smallest multiple of step strictly greater than x.

    The tolerance keeps a weight that is already exactly on a multiple from
    being its own 'next' one through floating point drift.
    """
    return ((x / step + TOL).floor() + 1) * step


def loadable_at_or_above(kg: pl.Expr) -> pl.Expr:
    """Smallest weight on the 2.5 kg loadable grid at or above kg."""
    return (kg / PLATE_STEP_KG - TOL).ceil() * PLATE_STEP_KG


def loadable_at_or_above_lb(lb: pl.Expr) -> pl.Expr:
    """Smallest weight on the 5 lb loadable grid at or above lb."""
    return (lb / PLATE_STEP_LB - TOL).ceil() * PLATE_STEP_LB


def dist_to_next_lb_kg_meet(prev_kg: pl.Expr, step_lb: float) -> pl.Expr:
    """Kilo gap to the next pound milestone, in a kilo-loaded meet.

    The lifter cannot load the pound figure itself, so the reachable weight is
    the smallest loadable 2.5 kg weight at or above it: 500 lb becomes 227.5 kg.
    """
    target_lb = next_multiple_above(prev_kg / KG_PER_LB, step_lb)
    return loadable_at_or_above(target_lb * KG_PER_LB) - prev_kg


def dist_to_next_lb_lb_meet(prev_kg: pl.Expr, step_lb: float) -> pl.Expr:
    """Kilo gap to the next pound milestone, in a pound-loaded meet.

    Here the bar is loaded in pounds, so the milestone is the pound figure
    itself with no kilo rounding: 500 lb is 226.796 kg and is reachable exactly.
    """
    target_lb = next_multiple_above(prev_kg / KG_PER_LB, step_lb)
    return target_lb * KG_PER_LB - prev_kg


def dist_to_next_kg_milestone(prev_kg: pl.Expr, step_kg: float) -> pl.Expr:
    """Kilos from prev_kg up to the next kilo milestone.

    The loadable rounding is a no-op here (10 and 50 are both multiples of
    2.5 kg) and is applied anyway so the two unit systems are treated the same
    way rather than by two different rules.
    """
    return loadable_at_or_above(next_multiple_above(prev_kg, step_kg)) - prev_kg


def on_lb_milestone_kg_meet(kg: pl.Expr, step_lb: float) -> pl.Expr:
    """In a kilo-loaded meet: is this the loadable weight for a lb milestone?"""
    nearest_lb = (kg / KG_PER_LB / step_lb).round() * step_lb
    target = loadable_at_or_above(nearest_lb * KG_PER_LB)
    return ((kg - target).abs() < TOL) & (nearest_lb > 0)


def on_lb_milestone_lb_meet(kg: pl.Expr, step_lb: float) -> pl.Expr:
    """In a pound-loaded meet: is this exactly on a lb milestone?

    Compared in pounds with the storage tolerance, because the kilo value is a
    rounded conversion and will not match to the last decimal.
    """
    lb = kg / KG_PER_LB
    return ((lb - (lb / step_lb).round() * step_lb).abs() < LB_TOL) & (lb > step_lb / 2)


def on_kg_milestone(kg: pl.Expr, step_kg: float) -> pl.Expr:
    """Is this weight an exact multiple of the kilo milestone step?"""
    return ((kg / step_kg) - (kg / step_kg).round()).abs() < TOL


def by_loading_unit(kg_expr: pl.Expr, lb_expr: pl.Expr) -> pl.Expr:
    """Pick the kilo-meet or pound-meet definition from the meet's unit.

    The unit is a property of the meet, decided once in the clean stage, so a
    single attempt can never flip which definition applies to it. Meets whose
    loading is mixed carry a null unit and get null milestone variables rather
    than a guess.
    """
    unit = pl.col("loading_unit")
    return (
        pl.when(unit == "kg").then(kg_expr).when(unit == "lb").then(lb_expr).otherwise(None)
    )


def add_weight_and_milestones(long: pl.DataFrame) -> pl.DataFrame:
    """Pound weight, milestone distances and milestone flags.

    Distances are returned in kilos regardless of the meet's loading unit so
    that one pooled model can read the column. dist_to_next_lb_ms_lb carries
    the same gap in pounds for plots and for pound-meet work.
    """
    prev = pl.col("prev_weight_kg")
    kg = pl.col("weight_kg")

    dist_lb = by_loading_unit(
        dist_to_next_lb_kg_meet(prev, LB_STEP), dist_to_next_lb_lb_meet(prev, LB_STEP)
    )
    dist_lb100 = by_loading_unit(
        dist_to_next_lb_kg_meet(prev, LB_COARSE_STEP),
        dist_to_next_lb_lb_meet(prev, LB_COARSE_STEP),
    )
    return long.with_columns(
        (kg / KG_PER_LB).alias("weight_lb"),
        dist_lb.alias("dist_to_next_lb_ms"),
        dist_lb100.alias("dist_to_next_lb100_ms"),
        (dist_lb / KG_PER_LB).alias("dist_to_next_lb_ms_lb"),
        by_loading_unit(
            dist_to_next_kg_milestone(prev, KG_STEP),
            dist_to_next_kg_milestone(prev, KG_STEP),
        ).alias("dist_to_next_kg_ms"),
        by_loading_unit(
            dist_to_next_kg_milestone(prev, KG_COARSE_STEP),
            dist_to_next_kg_milestone(prev, KG_COARSE_STEP),
        ).alias("dist_to_next_kg50_ms"),
        by_loading_unit(
            on_lb_milestone_kg_meet(kg, LB_STEP), on_lb_milestone_lb_meet(kg, LB_STEP)
        ).alias("on_lb_milestone"),
        by_loading_unit(
            on_lb_milestone_kg_meet(kg, LB_COARSE_STEP),
            on_lb_milestone_lb_meet(kg, LB_COARSE_STEP),
        ).alias("on_lb100_milestone"),
        by_loading_unit(
            on_kg_milestone(kg, KG_STEP), on_kg_milestone(kg, KG_STEP)
        ).alias("on_kg_milestone"),
        by_loading_unit(
            on_kg_milestone(kg, KG_COARSE_STEP), on_kg_milestone(kg, KG_COARSE_STEP)
        ).alias("on_kg50_milestone"),
    )


def add_region(long: pl.DataFrame) -> pl.DataFrame:
    """lb_region: 1 for US meets, 0 for kilo countries, null where ambiguous.

    Null rather than dropped: the ambiguous meets are still usable for
    descriptive work, and the region comparison filters them itself.
    """
    country = pl.col("MeetCountry")
    return long.with_columns(
        pl.when(country.is_null() | country.is_in(AMBIGUOUS_COUNTRIES))
        .then(None)
        .when(country == LB_REGION_COUNTRY)
        .then(1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("lb_region")
    )


def add_pr_flag(long: pl.DataFrame) -> pl.DataFrame:
    """Would this attempt, if made, be a lifetime best for the lift?

    The comparison point combines two strictly-earlier sources: prior_best_kg,
    the best successful lift at earlier *meets*, and the best successful
    attempt earlier in this same meet. Using only the first would call a third
    attempt a PR when the lifter had already beaten it on their second an hour
    before.

    Failures are carried as 0 through the running maximum rather than as null,
    because a cumulative maximum over nulls leaves holes at the failed attempts
    instead of carrying the running best past them.

    Null where the lifter has no observed history: unknown, not false.
    """
    scored = pl.when(pl.col("success")).then(pl.col("weight_kg")).otherwise(0.0)
    earlier_today = scored.cum_max().shift(1).over(SERIES)

    long = long.sort([*SERIES, "attempt_no"]).with_columns(
        pl.when(earlier_today > 0).then(earlier_today).alias("_today")
    )

    best_before = pl.max_horizontal("prior_best_kg", "_today")
    return long.with_columns(best_before.alias("best_before_kg")).with_columns(
        pl.when(pl.col("best_before_kg").is_null())
        .then(pl.lit(None, dtype=pl.Boolean))
        .otherwise(pl.col("weight_kg") > pl.col("best_before_kg"))
        .alias("would_be_pr")
    ).drop("_today")


def check_no_lookahead(long: pl.DataFrame) -> None:
    """Guard the one rule this stage can break silently."""
    overlap = set(FEATURE_COLUMNS) & set(POST_ATTEMPT_COLUMNS)
    if overlap:
        raise AssertionError(f"outcome columns listed as features: {sorted(overlap)}")

    missing = [c for c in FEATURE_COLUMNS if c not in long.columns]
    if missing:
        raise AssertionError(f"declared features not in the frame: {missing}")

    # Distances are defined off the previous attempt, so they exist exactly
    # where a previous attempt exists. A non-null distance on attempt 1 would
    # mean something had been measured off the attempt itself.
    first = long.filter(pl.col("attempt_no") == 1)
    for col in ("dist_to_next_lb_ms", "dist_to_next_kg_ms"):
        leaked = first.filter(pl.col(col).is_not_null()).height
        if leaked:
            raise AssertionError(f"{leaked:,} attempt-1 rows have a {col}")

    # A PR flag must never be set from the attempt's own result.
    bad = long.filter(
        pl.col("would_be_pr") & (pl.col("weight_kg") <= pl.col("best_before_kg"))
    ).height
    if bad:
        raise AssertionError(f"{bad:,} would_be_pr rows do not beat the prior best")

    print(f"no-lookahead checks ok: {len(FEATURE_COLUMNS)} features declared")


def use_utf8_stdout() -> None:
    """Polars draws tables with box-drawing characters, which the default
    Windows console encoding (cp1252) cannot encode."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def summary(long: pl.DataFrame) -> None:
    """Descriptive summary only. No interpretation."""
    n = long.height
    amb = long.filter(pl.col("lb_region").is_null()).height
    print(f"\nattempts:           {n:,}")
    print(f"ambiguous region:   {amb:,} ({amb / n:.1%}) - Canada and the UK")
    print(f"mixed loading unit: {long.filter(pl.col('loading_unit').is_null()).height:,}")
    print(f"no milestone dist:  {long.filter(pl.col('dist_to_next_lb_ms').is_null()).height:,}")
    print(f"no PR baseline:     {long.filter(pl.col('would_be_pr').is_null()).height:,}")

    region = (
        pl.when(pl.col("lb_region") == 1)
        .then(pl.lit("lb region"))
        .when(pl.col("lb_region") == 0)
        .then(pl.lit("kg region"))
        .otherwise(pl.lit("ambiguous"))
    )
    unit = pl.col("loading_unit").fill_null("mixed")

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=140):
        print("\nRegion by meet loading unit (attempts / meets):")
        print(
            long.group_by(region.alias("region"), unit.alias("loading_unit"))
            .agg(pl.len().alias("attempts"), pl.col("meet_id").n_unique().alias("meets"))
            .sort("region", "loading_unit")
        )

        print("\nShare of attempts on a 50 lb milestone, by group and attempt number:")
        group = (
            pl.when(pl.col("loading_unit") == "lb")
            .then(pl.lit("lb-loaded"))
            .when((pl.col("loading_unit") == "kg") & (pl.col("lb_region") == 1))
            .then(pl.lit("kg-loaded, lb region"))
            .when((pl.col("loading_unit") == "kg") & (pl.col("lb_region") == 0))
            .then(pl.lit("kg-loaded, kg region"))
            .otherwise(pl.lit("other"))
        )
        print(
            long.filter(pl.col("loading_unit").is_not_null())
            .group_by(group.alias("group"), "attempt_no")
            .agg(
                pl.len().alias("attempts"),
                pl.col("on_lb_milestone").mean().round(4).alias("on_lb_ms"),
                pl.col("on_kg_milestone").mean().round(4).alias("on_kg_ms"),
            )
            .filter(pl.col("group") != "other")
            .sort("group", "attempt_no")
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=IN_PARQUET)
    parser.add_argument("--out", type=Path, default=OUT_PARQUET)
    args = parser.parse_args()
    use_utf8_stdout()

    if not args.src.exists():
        print(f"{args.src} not found - run the clean stage first.", file=sys.stderr)
        return 1

    long = pl.read_parquet(args.src)
    print(f"loaded {long.height:,} attempts")

    long = add_weight_and_milestones(long)
    long = add_region(long)
    long = add_pr_flag(long)
    check_no_lookahead(long)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    long.write_parquet(args.out)
    print(f"\nwrote {long.height:,} attempts to {args.out}")

    summary(long)
    return 0


if __name__ == "__main__":
    sys.exit(main())
