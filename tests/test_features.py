import datetime as dt

import polars as pl
import pytest

from features import (
    FEATURE_COLUMNS,
    KG_PER_LB,
    POST_ATTEMPT_COLUMNS,
    add_pr_flag,
    add_region,
    add_weight_and_milestones,
    check_no_lookahead,
    dist_to_next_kg_milestone,
    dist_to_next_lb_kg_meet,
    dist_to_next_lb_lb_meet,
    loadable_at_or_above,
    next_multiple_above,
    on_kg_milestone,
    on_lb_milestone_kg_meet,
    on_lb_milestone_lb_meet,
)


def evaluate(expr_fn, values: list[float]) -> list:
    """Run a bare expression helper over a column of values."""
    return (
        pl.DataFrame({"x": values})
        .select(expr_fn(pl.col("x")).alias("out"))["out"]
        .to_list()
    )


def series(
    weights: list[float],
    successes: list[bool] | None = None,
    *,
    lift: str = "deadlift",
    meet: str = "m1",
    lifter: str = "l1",
    date: str = "2020-01-01",
    country: str = "USA",
    loading_unit: str | None = "kg",
    prior_best: float | None = None,
) -> pl.DataFrame:
    """One attempt series in the shape clean.py emits."""
    successes = [True] * len(weights) if successes is None else successes
    n = len(weights)
    prev_w = [None, *weights[:-1]]
    prev_s = [None, *successes[:-1]]
    return pl.DataFrame(
        {
            "lifter_id": [lifter] * n,
            "meet_id": [meet] * n,
            "lift": [lift] * n,
            "attempt_no": list(range(1, n + 1)),
            "weight_kg": weights,
            "success": successes,
            "prev_weight_kg": prev_w,
            "prev_success": prev_s,
            "jump_kg": [None if p is None else w - p for w, p in zip(weights, prev_w)],
            "jump_pct": [
                None if p is None else 100 * (w - p) / p for w, p in zip(weights, prev_w)
            ],
            "prior_meets": [0 if prior_best is None else 1] * n,
            "prior_best_kg": [prior_best] * n,
            "best3_kg": [max(weights)] * n,
            "Name": [lifter] * n,
            "Sex": ["M"] * n,
            "Age": [25.0] * n,
            "BodyweightKg": [90.0] * n,
            "WeightClassKg": ["90"] * n,
            "Division": ["Open"] * n,
            "Federation": ["USAPL"] * n,
            "ParentFederation": ["IPF"] * n,
            "Tested": ["Yes"] * n,
            "Date": [dt.date.fromisoformat(date)] * n,
            "loading_unit": [loading_unit] * n,
            "lb_share": [0.0 if loading_unit == "kg" else 1.0] * n,
            "MeetCountry": [country] * n,
            "MeetState": ["CA"] * n,
            "MeetName": ["Meet"] * n,
            "Place": [1] * n,
            "TotalKg": [500.0] * n,
        },
        schema_overrides={
            "attempt_no": pl.Int8,
            "prev_weight_kg": pl.Float64,
            "prev_success": pl.Boolean,
            "prior_best_kg": pl.Float64,
            "prior_meets": pl.UInt32,
        },
    )


def featured(df: pl.DataFrame) -> pl.DataFrame:
    return add_pr_flag(add_region(add_weight_and_milestones(df)))


# --- grid helpers -----------------------------------------------------------


def test_next_multiple_is_strictly_above():
    assert evaluate(lambda c: next_multiple_above(c, 50.0), [0.0, 49.9, 50.0, 50.1]) == [
        50.0,
        50.0,
        100.0,
        100.0,
    ]


def test_loadable_rounds_up_to_the_plate_grid():
    out = evaluate(loadable_at_or_above, [226.796, 227.5, 227.6, 100.0])
    assert out == [227.5, 227.5, 230.0, 100.0]


@pytest.mark.parametrize(
    "lb, expected_kg",
    [(400, 182.5), (450, 205.0), (500, 227.5), (550, 250.0), (600, 272.5), (1000, 455.0)],
)
def test_pound_milestones_land_on_the_documented_kilo_weights(lb, expected_kg):
    # CLAUDE.md: 500 lb is about 226.8 kg, so 227.5 kg is the weight that
    # should matter in US meets. The whole identification rests on this.
    got = evaluate(loadable_at_or_above, [lb * KG_PER_LB])[0]
    assert got == pytest.approx(expected_kg)


# --- milestone distances ----------------------------------------------------


def test_lb_distance_measured_from_the_previous_attempt():
    # 220 kg = 485 lb; the next 50 lb multiple is 500 lb -> 227.5 kg.
    got = evaluate(lambda c: dist_to_next_lb_kg_meet(c, 50.0), [220.0])[0]
    assert got == pytest.approx(7.5)


def test_lb_distance_skips_past_a_weight_already_on_a_milestone():
    # Standing exactly on 500 lb, the next one is 550 lb -> 250 kg.
    got = evaluate(lambda c: dist_to_next_lb_kg_meet(c, 50.0), [227.5])[0]
    assert got == pytest.approx(22.5)


def test_coarse_lb_grade_targets_the_rounder_number():
    # From 220 kg: next 50 lb is 500 lb, next 100 lb is also 500 lb.
    # From 230 kg (507 lb): next 50 lb is 550, next 100 lb is 600.
    fine = evaluate(lambda c: dist_to_next_lb_kg_meet(c, 50.0), [230.0])[0]
    coarse = evaluate(lambda c: dist_to_next_lb_kg_meet(c, 100.0), [230.0])[0]
    assert fine == pytest.approx(20.0)  # 550 lb -> 250.0
    assert coarse == pytest.approx(42.5)  # 600 lb -> 272.5


def test_kg_distance_to_the_next_ten():
    assert evaluate(lambda c: dist_to_next_kg_milestone(c, 10.0), [192.5]) == [
        pytest.approx(7.5)
    ]
    assert evaluate(lambda c: dist_to_next_kg_milestone(c, 10.0), [200.0]) == [
        pytest.approx(10.0)
    ]


def test_kg_distance_to_the_next_fifty():
    assert evaluate(lambda c: dist_to_next_kg_milestone(c, 50.0), [227.5]) == [
        pytest.approx(22.5)
    ]


def test_distances_are_null_on_the_first_attempt():
    out = featured(series([180.0, 190.0, 200.0]))
    first = out.filter(pl.col("attempt_no") == 1).to_dicts()[0]
    for col in (
        "dist_to_next_lb_ms",
        "dist_to_next_lb100_ms",
        "dist_to_next_kg_ms",
        "dist_to_next_kg50_ms",
    ):
        assert first[col] is None, col


def test_distance_uses_prev_weight_not_the_attempt():
    # Attempt 2 is at 227.5, but the distance must be measured from 220.
    out = featured(series([200.0, 220.0, 227.5]))
    third = out.filter(pl.col("attempt_no") == 3).to_dicts()[0]
    assert third["prev_weight_kg"] == 220.0
    assert third["dist_to_next_lb_ms"] == pytest.approx(7.5)


# --- milestone flags --------------------------------------------------------


def test_on_lb_milestone_true_at_the_loadable_five_hundred():
    assert evaluate(lambda c: on_lb_milestone_kg_meet(c, 50.0), [227.5]) == [True]


@pytest.mark.parametrize("kg", [225.0, 226.0, 230.0, 100.0])
def test_on_lb_milestone_false_off_the_loadable_weight(kg):
    assert evaluate(lambda c: on_lb_milestone_kg_meet(c, 50.0), [kg]) == [False]


def test_on_lb100_is_a_subset_of_on_lb50():
    # 227.5 is 500 lb (both grades); 250.0 is 550 lb (50 lb only).
    fine = evaluate(lambda c: on_lb_milestone_kg_meet(c, 50.0), [227.5, 250.0])
    coarse = evaluate(lambda c: on_lb_milestone_kg_meet(c, 100.0), [227.5, 250.0])
    assert fine == [True, True]
    assert coarse == [True, False]


def test_on_kg_milestone_matches_exact_tens():
    assert evaluate(lambda c: on_kg_milestone(c, 10.0), [200.0, 227.5, 205.0]) == [
        True,
        False,
        False,
    ]


def test_on_kg50_matches_exact_fifties():
    assert evaluate(lambda c: on_kg_milestone(c, 50.0), [200.0, 210.0]) == [True, False]


def test_a_kilo_milestone_is_not_a_pound_milestone():
    # The design needs these to be different weights, not the same ones.
    kg = [100.0, 150.0, 200.0, 250.0]
    assert evaluate(lambda c: on_kg_milestone(c, 50.0), kg) == [True] * 4
    assert evaluate(lambda c: on_lb_milestone_kg_meet(c, 50.0), kg) == [
        False,
        False,
        False,
        True,  # 250 kg is also 550 lb
    ]


def test_weight_lb_conversion():
    out = featured(series([227.5]))
    assert out["weight_lb"][0] == pytest.approx(501.6, abs=0.1)


# --- region -----------------------------------------------------------------


def test_usa_is_lb_region():
    assert featured(series([200.0], country="USA"))["lb_region"][0] == 1


@pytest.mark.parametrize("country", ["Sweden", "Russia", "Australia", "Ireland"])
def test_kilo_countries_are_kg_region(country):
    # Ireland is the Republic, not the UK, so it is a kilo country.
    assert featured(series([200.0], country=country))["lb_region"][0] == 0


@pytest.mark.parametrize(
    "country", ["Canada", "UK", "England", "Scotland", "Wales", "N.Ireland"]
)
def test_ambiguous_countries_are_null_not_zero(country):
    assert featured(series([200.0], country=country))["lb_region"][0] is None


def test_ambiguous_rows_are_kept_not_dropped():
    out = featured(series([180.0, 190.0, 200.0], country="Canada"))
    assert out.height == 3


# --- lifetime PR ------------------------------------------------------------


def test_pr_against_an_earlier_meet():
    out = featured(series([180.0, 190.0, 200.0], prior_best=195.0))
    flags = out.sort("attempt_no")["would_be_pr"].to_list()
    assert flags == [False, False, True]


def test_pr_accounts_for_earlier_attempts_at_the_same_meet():
    # Prior meets say 185, but attempt 2 already hit 190 today, so a 188 kg
    # third attempt is not a lifetime PR.
    out = featured(series([180.0, 190.0, 188.0], prior_best=185.0))
    third = out.filter(pl.col("attempt_no") == 3).to_dicts()[0]
    assert third["best_before_kg"] == 190.0
    assert third["would_be_pr"] is False


def test_failed_attempts_do_not_set_the_baseline():
    out = featured(series([180.0, 200.0, 190.0], [True, False, True], prior_best=None))
    third = out.filter(pl.col("attempt_no") == 3).to_dicts()[0]
    assert third["best_before_kg"] == 180.0  # the failed 200 does not count
    assert third["would_be_pr"] is True


def test_running_best_carries_past_a_failure():
    out = featured(series([180.0, 200.0, 170.0], [True, False, True], prior_best=None))
    third = out.filter(pl.col("attempt_no") == 3).to_dicts()[0]
    assert third["best_before_kg"] == 180.0
    assert third["would_be_pr"] is False


def test_pr_is_null_when_there_is_no_history():
    out = featured(series([180.0, 190.0, 200.0], prior_best=None))
    first = out.filter(pl.col("attempt_no") == 1).to_dicts()[0]
    assert first["best_before_kg"] is None
    assert first["would_be_pr"] is None


def test_pr_baseline_excludes_the_attempt_itself():
    out = featured(series([180.0, 190.0, 200.0], prior_best=None))
    for row in out.sort("attempt_no").to_dicts():
        if row["best_before_kg"] is not None:
            assert row["best_before_kg"] < row["weight_kg"]


def test_pr_does_not_cross_lifts_or_lifters():
    df = pl.concat(
        [
            series([180.0, 190.0, 200.0], lift="squat", prior_best=None),
            series([100.0, 110.0, 120.0], lift="bench", prior_best=None),
        ]
    )
    out = featured(df)
    bench_first = out.filter(
        (pl.col("lift") == "bench") & (pl.col("attempt_no") == 1)
    ).to_dicts()[0]
    assert bench_first["best_before_kg"] is None


# --- no lookahead -----------------------------------------------------------


def test_outcomes_are_not_declared_as_features():
    assert not set(FEATURE_COLUMNS) & set(POST_ATTEMPT_COLUMNS)


def test_known_outcomes_are_all_listed_as_post_attempt():
    for col in ("success", "best3_kg", "TotalKg", "Place"):
        assert col in POST_ATTEMPT_COLUMNS


def test_check_no_lookahead_passes_on_a_clean_frame():
    check_no_lookahead(featured(series([180.0, 190.0, 200.0], prior_best=170.0)))


def test_check_no_lookahead_catches_a_distance_on_attempt_one():
    bad = featured(series([180.0, 190.0, 200.0])).with_columns(
        pl.col("dist_to_next_lb_ms").fill_null(1.0)
    )
    with pytest.raises(AssertionError, match="attempt-1 rows have a"):
        check_no_lookahead(bad)


def test_check_no_lookahead_catches_a_bogus_pr_flag():
    bad = featured(series([180.0, 190.0, 200.0], prior_best=250.0)).with_columns(
        pl.lit(True).alias("would_be_pr")
    )
    with pytest.raises(AssertionError, match="do not beat the prior best"):
        check_no_lookahead(bad)


# --- loading unit drives the milestone definition ---------------------------

LB500 = 500 * KG_PER_LB  # 226.796 kg, the weight a pound-loaded meet reaches


def test_pound_meet_flags_the_exact_pound_figure():
    out = featured(series([LB500], loading_unit="lb"))
    assert out["on_lb_milestone"][0] is True


def test_pound_meet_does_not_flag_the_kilo_rounded_weight():
    # 227.5 is the kilo-meet stand-in for 500 lb. In a pound-loaded meet the
    # lifter could load 500 lb exactly, so 227.5 is not the milestone.
    out = featured(series([227.5], loading_unit="lb"))
    assert out["on_lb_milestone"][0] is False


def test_kilo_meet_does_not_flag_the_exact_pound_figure():
    # The mirror image: 226.796 is not loadable on a kilo bar.
    out = featured(series([LB500], loading_unit="kg"))
    assert out["on_lb_milestone"][0] is False


def test_kilo_meet_flags_the_rounded_weight():
    assert featured(series([227.5], loading_unit="kg"))["on_lb_milestone"][0] is True


def test_same_weight_different_meet_unit_gives_different_flag():
    # The identification depends on this: one weight, two meanings, decided by
    # the meet rather than by the attempt.
    kg_meet = featured(series([227.5], loading_unit="kg"))["on_lb_milestone"][0]
    lb_meet = featured(series([227.5], loading_unit="lb"))["on_lb_milestone"][0]
    assert kg_meet is True and lb_meet is False


def test_mixed_meet_gets_null_milestones():
    out = featured(series([180.0, 227.5, 250.0], loading_unit=None))
    for col in ("on_lb_milestone", "on_kg_milestone", "dist_to_next_lb_ms"):
        assert out[col].null_count() == out.height, col


def test_pound_meet_distance_targets_the_pound_figure():
    # From 220 kg (485 lb) the next 50 lb multiple is 500 lb = 226.796 kg,
    # not the 227.5 a kilo meet would have to round up to.
    out = featured(series([200.0, 220.0, LB500], loading_unit="lb"))
    third = out.filter(pl.col("attempt_no") == 3).to_dicts()[0]
    assert third["dist_to_next_lb_ms"] == pytest.approx(LB500 - 220.0)
    assert third["dist_to_next_lb_ms"] < 7.5  # the kilo-meet answer


def test_distance_in_pounds_matches_the_kilo_column():
    out = featured(series([200.0, 220.0, 227.5], loading_unit="kg"))
    row = out.filter(pl.col("attempt_no") == 3).to_dicts()[0]
    assert row["dist_to_next_lb_ms_lb"] == pytest.approx(
        row["dist_to_next_lb_ms"] / KG_PER_LB
    )


def test_pound_flag_tolerates_opl_storage_rounding():
    # OPL stores kilos to two decimals, so 500 lb comes back as 226.80.
    assert featured(series([226.80], loading_unit="lb"))["on_lb_milestone"][0] is True


def test_loading_unit_is_declared_a_feature():
    assert "loading_unit" in FEATURE_COLUMNS
    assert "dist_to_next_lb_ms_lb" in FEATURE_COLUMNS
