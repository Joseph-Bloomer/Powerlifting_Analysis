"""Clean stage: restrict the sample and reshape attempts to long format.

Reads data/raw/openpowerlifting.parquet and writes one row per
lifter x meet x lift x attempt number to data/interim/attempts.parquet.

The study is about attempt *selection*, so the unit of analysis is the attempt
and each row carries what the lifter had just done: prev_weight_kg,
prev_success, jump_kg, jump_pct. The wide OPL schema cannot express that.

Sample (CLAUDE.md): Event == SBD, Equipment == Raw, numeric Place, and
attempts 1-3 all present. Completeness is judged *per lift*: a lifter whose
squat series is complete but whose bench is missing an attempt contributes the
squat series and not the bench one.

The OPL sign convention (negative value = failed attempt) is decoded here once:
weight_kg is the absolute value and success is a separate boolean. Nothing
downstream should have to know about the sign again.

Fourth attempts do not count towards the total and are ignored entirely.
Every filter logs how many rows it removes. No attempt value is ever imputed.

Not every meet loads in kilos. American meets that load in pounds are stored as
the kilo conversion of a round pound number, which never lands on the kilo
grid, so an attempt is valid if it is on the 0.5 kg grid *or* within rounding
tolerance of a whole pound. Each meet then gets a loading_unit of kg or lb from
the share of its attempts that are pound-native, null where that share is
mixed. Treating the pound meets as transcription errors discarded every
literally-loaded pound milestone, 500 lb included.

Milestone flags, region and any other treatment variable belong to the
features stage, not here. MeetCountry is passed through unfiltered so features
can apply the region rules itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from constants import (
    ATTEMPT_NUMBERS,
    GRID_KG,
    KG_PER_LB,
    LB_TOL,
    LIFTS,
    LOADING_KG_THRESHOLD,
    LOADING_LB_THRESHOLD,
    MIN_ATTEMPTS_FOR_LOADING_UNIT,
    TOL,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_PARQUET = ROOT / "data" / "raw" / "openpowerlifting.parquet"
OUT_PARQUET = ROOT / "data" / "interim" / "attempts.parquet"

MAX_JUMP_PCT = 25.0

ID_COLUMNS = ["lifter_id", "meet_id"]

# Meet-level columns carried through to the long frame. Event and Equipment
# are dropped: they are constant once the sample filters have run.
PASSTHROUGH = [
    "Name",
    "Sex",
    "Age",
    "BodyweightKg",
    "WeightClassKg",
    "Division",
    "Federation",
    "ParentFederation",
    "Tested",
    "Date",
    "MeetCountry",
    "MeetState",
    "MeetName",
    "Place",
    "TotalKg",
]

# Read only what we need; the raw parquet holds every federation and event.
READ_COLUMNS = [
    "Event",
    "Equipment",
    *PASSTHROUGH,
    *[f"{lift}{n}Kg" for lift in LIFTS for n in ATTEMPT_NUMBERS],
    *[f"Best3{lift}Kg" for lift in LIFTS],
]


def log_filter(name: str, before: int, after: int) -> None:
    """One line per filter, in the same format the other stages use."""
    print(f"filter {name}: removed {before - after:,} rows ({before:,} -> {after:,})")


def apply_row_filters(lf: pl.LazyFrame) -> pl.DataFrame:
    """CLAUDE.md sample restrictions that apply to a whole lifter-meet row.

    Attempt completeness is not here: it is judged per lift inside to_long().
    """
    df = lf.collect()
    n = df.height
    print(f"loaded {n:,} rows")

    df = df.filter(pl.col("Event") == "SBD")
    log_filter("Event == SBD", n, df.height)
    n = df.height

    df = df.filter(pl.col("Equipment") == "Raw")
    log_filter("Equipment == Raw", n, df.height)
    n = df.height

    # Place holds DQ, DD, NS and G as well as numbers; non-numeric means the
    # result did not stand, so the attempts are not a normal competition series.
    df = df.filter(pl.col("Place").cast(pl.Int32, strict=False).is_not_null())
    log_filter("numeric Place", n, df.height)

    return df.with_columns(
        pl.col("Place").cast(pl.Int32, strict=False),
        pl.col("Date").str.to_date("%Y-%m-%d", strict=False),
    ).drop("Event", "Equipment")


def add_ids(df: pl.DataFrame) -> pl.DataFrame:
    """Stable lifter and meet keys.

    OPL already disambiguates lifters who share a name with a '#N' suffix
    ("John Doe #1"), so Name is the lifter key and the suffix is kept. Stripping
    it would merge distinct people. The bulk CSV carries no meet key column, so
    the meet key is the usual composite.
    """
    return df.with_columns(
        pl.col("Name").str.strip_chars().alias("lifter_id"),
        pl.concat_str(
            [
                pl.col("Federation"),
                pl.col("Date").cast(pl.Utf8),
                pl.col("MeetName"),
                pl.col("MeetCountry"),
            ],
            separator="|",
            ignore_nulls=False,
        ).alias("meet_id"),
    )


def to_long(df: pl.DataFrame) -> pl.DataFrame:
    """Reshape to one row per lifter x meet x lift x attempt number.

    A lift contributes its series only if all three of its attempts are
    reported; partial series are dropped rather than imputed. The sign
    convention is decoded here: weight_kg is the absolute value, success is
    the sign.
    """
    frames = []
    keep = [*ID_COLUMNS, *PASSTHROUGH]

    for prefix, lift in LIFTS.items():
        cols = [f"{prefix}{n}Kg" for n in ATTEMPT_NUMBERS]
        n = df.height
        sub = df.filter(pl.all_horizontal(pl.col(c).is_not_null() for c in cols))
        log_filter(f"{lift} attempts 1-3 all present", n, sub.height)

        frames.append(
            sub.select(
                *keep,
                pl.col(f"Best3{prefix}Kg").alias("best3_kg"),
                *cols,
            )
            .unpivot(
                index=[*keep, "best3_kg"],
                on=cols,
                variable_name="attempt_col",
                value_name="raw_kg",
            )
            .with_columns(
                pl.lit(lift).alias("lift"),
                pl.col("attempt_col")
                .str.extract(r"(\d)Kg$")
                .cast(pl.Int8)
                .alias("attempt_no"),
                pl.col("raw_kg").abs().alias("weight_kg"),
                (pl.col("raw_kg") > 0).alias("success"),
            )
            .drop("attempt_col", "raw_kg")
        )

    long = pl.concat(frames)

    # A literal 0 sits on the 0.5 kg grid but is not a real attempt, so it has
    # to go before the grid check rather than after it. Dropped at series grain
    # like the other implausibility checks, so no surviving attempt is left
    # pointing at a gap.
    n = long.height
    long = long.filter(
        ~(pl.col("weight_kg") <= 0).any().over([*ID_COLUMNS, "lift"])
    )
    log_filter("non-positive attempt weight", n, long.height)

    return long


def dedupe_entries(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse to one entry per lifter x meet.

    A lifter entered in more than one division at the same meet (Open and
    M40-44, say) gets one OPL row per division, each recording the *same*
    physical attempts. Keeping both would count one performance twice, and
    would make the series window in add_prev_and_jumps() span two copies, so
    attempt 1 of the second entry would inherit attempt 3 of the first.

    Done here on the wide frame rather than after the reshape: the duplication
    is a property of the raw row, and there are far fewer rows to sort.

    Exact duplicates collapse to one row. Where two entries disagree about the
    attempts there is no basis for choosing between them, so the lifter-meet is
    dropped as ambiguous rather than resolved arbitrarily.
    """
    entry = ID_COLUMNS
    attempts = [f"{lift}{n}Kg" for lift in LIFTS for n in ATTEMPT_NUMBERS]

    n = df.height
    disagreeing = (
        df.group_by(entry)
        .agg(pl.struct(attempts).n_unique().alias("variants"))
        .filter(pl.col("variants") > 1)
        .select(entry)
    )
    df = df.join(disagreeing, on=entry, how="anti")
    log_filter("duplicate entries disagree", n, df.height)

    # Sorting before unique() makes the surviving row's division and place
    # deterministic rather than whatever order the scan happened to produce.
    n = df.height
    df = df.sort([*entry, "Division", "Place"]).unique(
        subset=entry, keep="first", maintain_order=True
    )
    log_filter("duplicate division entries", n, df.height)

    return df


def on_kg_grid(kg: pl.Expr) -> pl.Expr:
    """Does this weight sit on the 0.5 kg loadable grid?"""
    return ((kg / GRID_KG) - (kg / GRID_KG).round()).abs() <= TOL


def is_whole_pound(kg: pl.Expr) -> pl.Expr:
    """Is this weight a whole number of pounds, within storage rounding?

    Meets that load in pounds are stored by OpenPowerlifting as the kilo
    conversion of a round pound number, which is never a round kilo figure.
    """
    lb = kg / KG_PER_LB
    return (lb - lb.round()).abs() <= LB_TOL


def add_loading_unit(long: pl.DataFrame) -> pl.DataFrame:
    """Classify each meet as kilo-loaded or pound-loaded.

    A meet's loading unit is a property of its bar and its plates, so it is
    decided once per meet from the share of its attempts that are pound-native
    (a whole number of pounds and off the kilo grid) and then applied to every
    attempt. Deciding attempt by attempt would let a kilo weight that happens
    to be near a round pound figure masquerade as a pound meet.

    The share is strongly bimodal, so meets in the middle are genuinely mixed
    rather than borderline; they are marked null rather than forced either way.
    Meets too small for the share to mean anything are also null.
    """
    pound_native = is_whole_pound(pl.col("weight_kg")) & ~on_kg_grid(pl.col("weight_kg"))

    per_meet = long.group_by("meet_id").agg(
        pl.len().alias("meet_attempts"),
        pound_native.mean().alias("lb_share"),
    )

    classified = per_meet.with_columns(
        pl.when(pl.col("meet_attempts") < MIN_ATTEMPTS_FOR_LOADING_UNIT)
        .then(None)
        .when(pl.col("lb_share") >= LOADING_LB_THRESHOLD)
        .then(pl.lit("lb"))
        .when(pl.col("lb_share") <= LOADING_KG_THRESHOLD)
        .then(pl.lit("kg"))
        .otherwise(None)
        .alias("loading_unit")
    )

    counts = classified.group_by("loading_unit").agg(
        pl.len().alias("meets"), pl.col("meet_attempts").sum().alias("attempts")
    )
    for row in counts.sort("meets", descending=True).to_dicts():
        unit = row["loading_unit"] or "mixed/small (null)"
        print(
            f"loading unit {unit:>18s}: {row['meets']:>7,} meets, "
            f"{row['attempts']:>10,} attempts"
        )

    return long.join(
        classified.select("meet_id", "loading_unit", "lb_share"), on="meet_id", how="left"
    )


def add_prev_and_jumps(long: pl.DataFrame) -> pl.DataFrame:
    """Attach the previous attempt in the same series and the jump to it.

    The window is the series (lifter x meet x lift), so bench attempt 1 never
    inherits squat attempt 3 and one meet never leaks into the next. Attempt 1
    has no predecessor and gets nulls.
    """
    series = [*ID_COLUMNS, "lift"]
    return long.sort([*series, "attempt_no"]).with_columns(
        pl.col("weight_kg").shift(1).over(series).alias("prev_weight_kg"),
        pl.col("success").shift(1).over(series).alias("prev_success"),
    ).with_columns(
        (pl.col("weight_kg") - pl.col("prev_weight_kg")).alias("jump_kg"),
    ).with_columns(
        (100 * pl.col("jump_kg") / pl.col("prev_weight_kg")).alias("jump_pct"),
    )


def drop_implausible(long: pl.DataFrame) -> pl.DataFrame:
    """Drop series containing a record that cannot describe a real attempt.

    One bad value means the series is miscoded, so all three attempts go. That
    also keeps prev_weight_kg and prev_success consistent: no surviving row
    points at an attempt that is no longer in the data.

    Applied one filter at a time so the logged counts are disjoint and sum to
    the total removed.
    """
    series = [*ID_COLUMNS, "lift"]
    checks = [
        (
            "weight on neither the kg nor the lb grid",
            ~on_kg_grid(pl.col("weight_kg")) & ~is_whole_pound(pl.col("weight_kg")),
        ),
        (
            f"jump over {MAX_JUMP_PCT:g}%",
            pl.col("jump_pct").abs() > MAX_JUMP_PCT,
        ),
        (
            "negative jump after a success",
            (pl.col("jump_kg") < 0) & pl.col("prev_success"),
        ),
    ]

    for name, bad in checks:
        n = long.height
        long = long.filter(~bad.fill_null(False).any().over(series))
        log_filter(name, n, long.height)

    return long


def add_history(long: pl.DataFrame) -> pl.DataFrame:
    """Lifter history from strictly earlier meet dates only.

    prior_meets counts the lifter's earlier meets for that lift and
    prior_best_kg is their best *successful* weight at those meets. Both are
    computed within the cleaned sample, so they mean "prior as observed here",
    not prior across all of powerlifting.

    Dates, not rows, are the unit: a lifter who competes twice on one date gets
    the same prior values at both meets, excluding both. A cumulative shift over
    meets would leak one of them into the other.
    """
    key = ["lifter_id", "lift", "Date"]

    per_date = (
        long.group_by(key)
        .agg(
            pl.col("meet_id").n_unique().alias("meets_on_date"),
            pl.col("weight_kg").filter(pl.col("success")).max().alias("best_on_date"),
        )
        .sort(key)
    )

    prior = per_date.with_columns(
        pl.col("meets_on_date")
        .cum_sum()
        .shift(1, fill_value=0)
        .over(["lifter_id", "lift"])
        .cast(pl.UInt32)
        .alias("prior_meets"),
        pl.col("best_on_date")
        .cum_max()
        .shift(1)
        .over(["lifter_id", "lift"])
        .alias("prior_best_kg"),
    ).select(*key, "prior_meets", "prior_best_kg")

    return long.join(prior, on=key, how="left")


def check_invariants(long: pl.DataFrame) -> None:
    """Fail loudly rather than write a table a later stage would misread."""
    series = [*ID_COLUMNS, "lift"]

    n_series = long.select(pl.struct(series).n_unique()).item()
    first = long.filter(pl.col("attempt_no") == 1)
    if first.height != n_series:
        raise AssertionError(
            f"{first.height:,} attempt-1 rows for {n_series:,} series: "
            "the series key does not identify one attempt series"
        )

    # Attempt 1 has no predecessor; every later attempt has one. A leak here
    # means the series window is spanning something it should not.
    leaked = first.filter(pl.col("prev_weight_kg").is_not_null()).height
    if leaked:
        raise AssertionError(f"{leaked:,} attempt-1 rows have a predecessor")

    later = long.filter(pl.col("attempt_no") > 1)
    missing = later.filter(pl.col("prev_weight_kg").is_null()).height
    if missing:
        raise AssertionError(f"{missing:,} attempts past the first lack a predecessor")

    off_both = long.filter(
        ~on_kg_grid(pl.col("weight_kg")) & ~is_whole_pound(pl.col("weight_kg"))
    ).height
    if off_both:
        raise AssertionError(f"{off_both:,} weights are on neither loadable grid")

    # A pound-loaded meet whose attempts sit on the kilo grid, or the reverse,
    # would mean the classification is not describing the bar.
    mislabelled = long.filter(
        (pl.col("loading_unit") == "lb")
        & on_kg_grid(pl.col("weight_kg"))
        & ~is_whole_pound(pl.col("weight_kg"))
    ).height
    if mislabelled > 0.5 * long.filter(pl.col("loading_unit") == "lb").height:
        raise AssertionError("most attempts in lb-loaded meets are kilo-native")

    print(f"invariants ok: {n_series:,} series, one row per lift and attempt")


def use_utf8_stdout() -> None:
    """Polars draws tables with box-drawing characters, which the default
    Windows console encoding (cp1252) cannot encode."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def summary(long: pl.DataFrame) -> None:
    """Descriptive summary of the written table. No interpretation."""
    n_series = long.select(pl.struct(ID_COLUMNS + ["lift"]).n_unique()).item()
    repeats = long.filter((pl.col("jump_kg") == 0) & pl.col("prev_success")).height

    print(f"\nattempts:      {long.height:,}")
    print(f"series:        {n_series:,}")
    print(f"lifters:       {long['lifter_id'].n_unique():,}")
    print(f"meets:         {long['meet_id'].n_unique():,}")
    print(f"dates:         {long['Date'].min()} to {long['Date'].max()}")
    print(f"with history:  {long.filter(pl.col('prior_meets') > 0).height:,} attempts")
    print(f"repeats kept:  {repeats:,} (jump of 0 kg after a success)")
    for unit in ("kg", "lb", None):
        sub = long.filter(
            pl.col("loading_unit").is_null()
            if unit is None
            else pl.col("loading_unit") == unit
        )
        label = unit or "mixed"
        print(f"  {label:>5s}-loaded: {sub.height:>9,} attempts, "
              f"{sub['meet_id'].n_unique():>6,} meets")

    table = (
        long.group_by("lift", "attempt_no")
        .agg(
            pl.len().alias("attempts"),
            pl.col("success").mean().round(3).alias("success_rate"),
            pl.col("weight_kg").median().alias("median_kg"),
            pl.col("jump_pct").median().round(2).alias("median_jump_pct"),
        )
        .sort("lift", "attempt_no")
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=120):
        print("\nBy lift and attempt number:")
        print(table)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=RAW_PARQUET,
        help=f"input parquet from the ingest stage (default {RAW_PARQUET})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_PARQUET,
        help=f"output parquet (default {OUT_PARQUET})",
    )
    args = parser.parse_args()
    use_utf8_stdout()

    if not args.raw.exists():
        print(f"{args.raw} not found - run `make ingest` first.", file=sys.stderr)
        return 1

    df = apply_row_filters(pl.scan_parquet(args.raw).select(READ_COLUMNS))
    df = add_ids(df)

    df = dedupe_entries(df)

    long = to_long(df)
    long = add_loading_unit(long)
    long = add_prev_and_jumps(long)
    long = drop_implausible(long)
    long = add_history(long)
    check_invariants(long)

    long = long.select(
        *ID_COLUMNS,
        "lift",
        "attempt_no",
        "weight_kg",
        "success",
        "prev_weight_kg",
        "prev_success",
        "jump_kg",
        "jump_pct",
        "prior_meets",
        "prior_best_kg",
        "best3_kg",
        "loading_unit",
        "lb_share",
        *PASSTHROUGH,
    ).sort([*ID_COLUMNS, "lift", "attempt_no"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    long.write_parquet(args.out)
    print(f"\nwrote {long.height:,} attempts to {args.out}")

    summary(long)
    return 0


if __name__ == "__main__":
    sys.exit(main())
