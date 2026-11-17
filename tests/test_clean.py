import datetime as dt

import polars as pl
import pytest

from clean import (
    add_history,
    add_ids,
    add_prev_and_jumps,
    add_loading_unit,
    apply_row_filters,
    dedupe_entries,
    drop_implausible,
    to_long,
)


def make_row(**overrides) -> dict:
    """One lifter-meet in raw wide form, with a clean attempt series per lift."""
    row = {
        "Event": "SBD",
        "Equipment": "Raw",
        "Name": "Test Lifter",
        "Sex": "M",
        "Age": 25.0,
        "BodyweightKg": 90.0,
        "WeightClassKg": "90",
        "Division": "Open",
        "Federation": "USAPL",
        "ParentFederation": "IPF",
        "Tested": "Yes",
        "Date": "2020-01-01",
        "MeetCountry": "USA",
        "MeetState": "CA",
        "MeetName": "Nationals",
        "Place": "1",
        "TotalKg": 500.0,
        "Squat1Kg": 180.0,
        "Squat2Kg": 190.0,
        "Squat3Kg": 200.0,
        "Bench1Kg": 120.0,
        "Bench2Kg": 130.0,
        "Bench3Kg": 140.0,
        "Deadlift1Kg": 200.0,
        "Deadlift2Kg": 215.0,
        "Deadlift3Kg": 227.5,
        "Best3SquatKg": 200.0,
        "Best3BenchKg": 140.0,
        "Best3DeadliftKg": 227.5,
    }
    row.update(overrides)
    return row


def raw_frame(*rows: dict) -> pl.DataFrame:
    rows = list(rows) or [make_row()]
    # Pin the numeric columns: a lone None in a test row would otherwise infer
    # as Null dtype and the reshape would see a different schema than ingest
    # produces.
    numeric = [
        k for k in rows[0] if k.endswith("Kg") and k != "WeightClassKg"
    ] + ["Age"]
    return pl.DataFrame(rows, schema_overrides={k: pl.Float64 for k in numeric})


def prepared(*rows: dict) -> pl.DataFrame:
    """Raw rows through the row filters and ID step, ready for to_long()."""
    return add_ids(apply_row_filters(raw_frame(*rows).lazy()))


def long_frame(*rows: dict) -> pl.DataFrame:
    return to_long(prepared(*rows))


def attempt(long: pl.DataFrame, lift: str, n: int) -> dict:
    return long.filter(
        (pl.col("lift") == lift) & (pl.col("attempt_no") == n)
    ).to_dicts()[0]


# --- row filters ------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, label",
    [
        ({"Event": "B"}, "Event == SBD"),
        ({"Equipment": "Wraps"}, "Equipment == Raw"),
        ({"Place": "DQ"}, "numeric Place"),
        ({"Place": "DD"}, "numeric Place"),
        ({"Place": "NS"}, "numeric Place"),
        ({"Place": "G"}, "numeric Place"),
    ],
)
def test_row_filters_remove_and_log(overrides, label, capsys):
    df = apply_row_filters(raw_frame(make_row(), make_row(**overrides)).lazy())
    assert df.height == 1
    assert f"filter {label}: removed 1 rows" in capsys.readouterr().out


def test_row_filters_cast_place_and_date():
    df = apply_row_filters(raw_frame().lazy())
    assert df["Place"].to_list() == [1]
    assert df["Date"].to_list() == [dt.date(2020, 1, 1)]


# --- reshape ----------------------------------------------------------------


def test_to_long_grain():
    long = long_frame()
    assert long.height == 9
    assert sorted(long["lift"].unique().to_list()) == ["bench", "deadlift", "squat"]
    assert sorted(long["attempt_no"].unique().to_list()) == [1, 2, 3]
    # One row per lift x attempt, no duplication across the join of the three.
    assert long.select(pl.struct("lift", "attempt_no").n_unique()).item() == 9


def test_to_long_maps_columns_to_the_right_lift_and_number():
    long = long_frame()
    assert attempt(long, "squat", 1)["weight_kg"] == 180.0
    assert attempt(long, "squat", 3)["weight_kg"] == 200.0
    assert attempt(long, "bench", 2)["weight_kg"] == 130.0
    assert attempt(long, "deadlift", 3)["weight_kg"] == 227.5


def test_to_long_ignores_fourth_attempts():
    # A 4th attempt in the raw row must not reach the long frame at all.
    long = long_frame(make_row(Deadlift4Kg=250.0))
    assert long.filter(pl.col("lift") == "deadlift").height == 3
    assert 250.0 not in long["weight_kg"].to_list()


def test_to_long_carries_best3_per_lift():
    long = long_frame()
    assert attempt(long, "squat", 1)["best3_kg"] == 200.0
    assert attempt(long, "bench", 1)["best3_kg"] == 140.0
    assert attempt(long, "deadlift", 1)["best3_kg"] == 227.5


def test_per_lift_completeness_keeps_other_lifts(capsys):
    long = long_frame(make_row(Bench2Kg=None))
    assert long.height == 6
    assert sorted(long["lift"].unique().to_list()) == ["deadlift", "squat"]
    out = capsys.readouterr().out
    assert "filter bench attempts 1-3 all present: removed 1 rows" in out
    assert "filter squat attempts 1-3 all present: removed 0 rows" in out


def test_partial_series_never_imputed():
    long = long_frame(make_row(Squat3Kg=None))
    assert long.filter(pl.col("lift") == "squat").height == 0
    assert long.height == 6


# --- sign handling ----------------------------------------------------------


def test_negative_value_is_a_failed_attempt_at_the_absolute_weight():
    long = long_frame(make_row(Deadlift3Kg=-227.5))
    row = attempt(long, "deadlift", 3)
    assert row["weight_kg"] == 227.5
    assert row["success"] is False


def test_positive_value_is_a_successful_attempt():
    row = attempt(long_frame(), "deadlift", 3)
    assert row["weight_kg"] == 227.5
    assert row["success"] is True


def test_sign_is_decoded_per_attempt_not_per_series():
    long = long_frame(make_row(Squat1Kg=180.0, Squat2Kg=-190.0, Squat3Kg=190.0))
    squat = long.filter(pl.col("lift") == "squat").sort("attempt_no")
    assert squat["weight_kg"].to_list() == [180.0, 190.0, 190.0]
    assert squat["success"].to_list() == [True, False, True]


def test_zero_weight_drops_the_whole_series(capsys):
    long = long_frame(make_row(Bench1Kg=0.0))
    assert long.filter(pl.col("lift") == "bench").height == 0
    assert long.height == 6  # squat and deadlift survive
    out = capsys.readouterr().out
    assert "filter non-positive attempt weight: removed 3 rows" in out


# --- duplicate division entries ---------------------------------------------


def deduped(*rows: dict) -> pl.DataFrame:
    """Raw rows through the row filters, IDs and dedup, as main() orders them."""
    return dedupe_entries(prepared(*rows))


def test_multi_division_entry_collapses_to_one_entry(capsys):
    # One performance, recorded once per division the lifter entered.
    df = deduped(make_row(Division="Open"), make_row(Division="M40-44"))
    assert df.height == 1
    assert to_long(df).height == 9
    assert "filter duplicate division entries: removed 1 rows" in capsys.readouterr().out


def test_multi_division_entry_does_not_leak_into_the_next_series():
    # Regression: without dedup the series window spans both copies, so
    # attempt 1 of the second entry inherits attempt 3 of the first.
    long = add_prev_and_jumps(
        to_long(deduped(make_row(Division="Open"), make_row(Division="M40-44")))
    )
    firsts = long.filter(pl.col("attempt_no") == 1)
    assert firsts.height == 3
    assert firsts["prev_weight_kg"].null_count() == 3


def test_disagreeing_entries_drop_the_lifter_meet(capsys):
    # Same lifter, same meet, but the two division rows do not agree on the
    # third squat. Nothing says which is right, so the entry goes.
    df = deduped(
        make_row(Division="Open", Squat3Kg=200.0),
        make_row(Division="M40-44", Squat3Kg=202.5),
    )
    assert df.height == 0
    assert "filter duplicate entries disagree: removed 2 rows" in capsys.readouterr().out


def test_disagreement_on_success_also_drops():
    df = deduped(
        make_row(Division="Open", Squat3Kg=200.0),
        make_row(Division="M40-44", Squat3Kg=-200.0),
    )
    assert df.height == 0


def test_dedupe_keeps_genuinely_separate_meets():
    df = deduped(make_row(MeetName="First"), make_row(MeetName="Second"))
    assert df.height == 2
    assert to_long(df).height == 18


def test_dedupe_keeps_different_lifters_at_one_meet():
    df = deduped(make_row(), make_row(Name="Someone Else"))
    assert df.height == 2


def test_dedupe_is_deterministic():
    a = deduped(make_row(Division="Open"), make_row(Division="M40-44"))
    b = deduped(make_row(Division="M40-44"), make_row(Division="Open"))
    assert a["Division"].to_list() == b["Division"].to_list()


# --- previous attempt and jumps ---------------------------------------------


def test_first_attempt_has_no_predecessor():
    long = add_prev_and_jumps(long_frame())
    row = attempt(long, "squat", 1)
    assert row["prev_weight_kg"] is None
    assert row["prev_success"] is None
    assert row["jump_kg"] is None
    assert row["jump_pct"] is None


def test_jumps_measured_against_the_previous_attempt():
    long = add_prev_and_jumps(long_frame())
    row = attempt(long, "squat", 2)
    assert row["prev_weight_kg"] == 180.0
    assert row["prev_success"] is True
    assert row["jump_kg"] == pytest.approx(10.0)
    assert row["jump_pct"] == pytest.approx(100 * 10.0 / 180.0)


def test_prev_success_follows_the_sign_of_the_previous_attempt():
    long = add_prev_and_jumps(long_frame(make_row(Squat2Kg=-190.0)))
    assert attempt(long, "squat", 3)["prev_success"] is False
    assert attempt(long, "squat", 3)["prev_weight_kg"] == 190.0


def test_no_bleed_across_lifts():
    # Bench attempt 1 must not inherit squat attempt 3.
    long = add_prev_and_jumps(long_frame())
    assert attempt(long, "bench", 1)["prev_weight_kg"] is None
    assert attempt(long, "deadlift", 1)["prev_weight_kg"] is None


def test_no_bleed_across_meets():
    long = add_prev_and_jumps(
        long_frame(
            make_row(),
            make_row(MeetName="Regionals", Date="2021-01-01"),
        )
    )
    firsts = long.filter(pl.col("attempt_no") == 1)
    assert firsts.height == 6
    assert firsts["prev_weight_kg"].null_count() == 6


# --- implausible records ----------------------------------------------------


def pipeline_to_drops(*rows: dict) -> pl.DataFrame:
    return drop_implausible(add_prev_and_jumps(long_frame(*rows)))


def test_weight_on_neither_grid_drops_the_whole_series(capsys):
    # 200.3 kg is neither a 0.5 kg step nor a whole pound (441.58 lb).
    long = pipeline_to_drops(make_row(Squat3Kg=200.3))
    assert long.filter(pl.col("lift") == "squat").height == 0
    assert long.height == 6  # bench and deadlift survive
    out = capsys.readouterr().out
    assert "filter weight on neither the kg nor the lb grid: removed 3 rows" in out


def test_whole_pound_weights_are_kept():
    # 226.80 kg is exactly 500 lb as OPL stores it: off the kilo grid, but a
    # real attempt in a pound-loaded meet. Dropping it was the old bug.
    long = pipeline_to_drops(
        make_row(Squat1Kg=181.44, Squat2Kg=204.12, Squat3Kg=226.80)
    )
    squat = long.filter(pl.col("lift") == "squat")
    assert squat.height == 3
    assert 226.80 in squat["weight_kg"].to_list()


def test_half_kilo_weights_are_on_the_grid():
    long = pipeline_to_drops(make_row(Squat1Kg=182.5, Squat2Kg=190.0, Squat3Kg=197.5))
    assert long.filter(pl.col("lift") == "squat").height == 3


def test_oversized_jump_drops_the_whole_series(capsys):
    long = pipeline_to_drops(make_row(Bench3Kg=200.0))  # 130 -> 200 is +54%
    assert long.filter(pl.col("lift") == "bench").height == 0
    assert "filter jump over 25%: removed 3 rows" in capsys.readouterr().out


def test_oversized_downward_jump_also_drops():
    # 200 failed, then 100: a 50% drop is miscoded, not a real attempt choice.
    long = pipeline_to_drops(make_row(Squat2Kg=-200.0, Squat3Kg=100.0))
    assert long.filter(pl.col("lift") == "squat").height == 0


def test_negative_jump_after_a_success_drops(capsys):
    long = pipeline_to_drops(make_row(Squat2Kg=190.0, Squat3Kg=185.0))
    assert long.filter(pl.col("lift") == "squat").height == 0
    assert "filter negative jump after a success: removed 3 rows" in capsys.readouterr().out


def test_negative_jump_after_a_failure_is_kept():
    # Same weights, but attempt 2 failed, so going lighter is legitimate.
    long = pipeline_to_drops(make_row(Squat2Kg=-190.0, Squat3Kg=185.0))
    squat = long.filter(pl.col("lift") == "squat").sort("attempt_no")
    assert squat.height == 3
    assert squat["success"].to_list() == [True, False, True]


def test_repeat_after_a_success_is_kept():
    long = pipeline_to_drops(make_row(Squat2Kg=190.0, Squat3Kg=190.0))
    assert long.filter(pl.col("lift") == "squat").height == 3


def test_clean_series_survives_every_filter():
    long = pipeline_to_drops()
    assert long.height == 9


# --- lifter history ---------------------------------------------------------


def history(*rows: dict) -> pl.DataFrame:
    return add_history(drop_implausible(add_prev_and_jumps(long_frame(*rows))))


def meet_history(long: pl.DataFrame, meet: str, lift: str = "deadlift") -> dict:
    return long.filter(
        (pl.col("MeetName") == meet) & (pl.col("lift") == lift) & (pl.col("attempt_no") == 1)
    ).to_dicts()[0]


def test_first_meet_has_no_history():
    row = meet_history(history(), "Nationals")
    assert row["prior_meets"] == 0
    assert row["prior_best_kg"] is None


def test_later_meet_sees_the_earlier_one():
    long = history(
        make_row(MeetName="First", Date="2019-01-01"),
        make_row(MeetName="Second", Date="2020-01-01"),
    )
    second = meet_history(long, "Second")
    assert second["prior_meets"] == 1
    assert second["prior_best_kg"] == 227.5


def test_history_is_strictly_earlier_not_same_date():
    long = history(
        make_row(MeetName="Morning", Date="2020-01-01"),
        make_row(MeetName="Evening", Date="2020-01-01", Deadlift3Kg=240.0),
    )
    for meet in ("Morning", "Evening"):
        row = meet_history(long, meet)
        assert row["prior_meets"] == 0, meet
        assert row["prior_best_kg"] is None, meet


def test_history_ignores_failed_attempts():
    long = history(
        make_row(MeetName="First", Date="2019-01-01", Deadlift3Kg=-227.5),
        make_row(MeetName="Second", Date="2020-01-01"),
    )
    # 227.5 failed, so the prior best is the heaviest *successful* attempt.
    assert meet_history(long, "Second")["prior_best_kg"] == 215.0


def test_history_is_per_lift():
    long = history(
        make_row(MeetName="First", Date="2019-01-01"),
        make_row(MeetName="Second", Date="2020-01-01"),
    )
    assert meet_history(long, "Second", "squat")["prior_best_kg"] == 200.0
    assert meet_history(long, "Second", "bench")["prior_best_kg"] == 140.0


def test_history_does_not_cross_lifters():
    long = history(
        make_row(Name="Someone Else", MeetName="First", Date="2019-01-01"),
        make_row(MeetName="Second", Date="2020-01-01"),
    )
    assert meet_history(long, "Second")["prior_meets"] == 0


def test_name_suffix_distinguishes_lifters():
    # OPL disambiguates same-named lifters with a '#N' suffix; they are not
    # the same person and must not share a history.
    long = history(
        make_row(Name="Test Lifter #1", MeetName="First", Date="2019-01-01"),
        make_row(MeetName="Second", Date="2020-01-01"),
    )
    assert long["lifter_id"].n_unique() == 2
    assert meet_history(long, "Second")["prior_meets"] == 0


def test_prior_best_accumulates_as_a_running_max():
    long = history(
        make_row(MeetName="First", Date="2018-01-01", Deadlift3Kg=250.0),
        make_row(MeetName="Second", Date="2019-01-01", Deadlift3Kg=230.0),
        make_row(MeetName="Third", Date="2020-01-01"),
    )
    third = meet_history(long, "Third")
    assert third["prior_meets"] == 2
    assert third["prior_best_kg"] == 250.0  # not 230, the most recent


# --- meet loading unit ------------------------------------------------------


def lb_row(**over):
    """A lifter-meet whose attempts are whole pounds (400/450/500 lb etc.)."""
    base = dict(
        Squat1Kg=181.44, Squat2Kg=204.12, Squat3Kg=226.80,   # 400/450/500 lb
        Bench1Kg=102.06, Bench2Kg=113.40, Bench3Kg=124.74,   # 225/250/275 lb
        Deadlift1Kg=204.12, Deadlift2Kg=226.80, Deadlift3Kg=249.48,  # 450/500/550
        Best3SquatKg=226.80, Best3BenchKg=124.74, Best3DeadliftKg=249.48,
    )
    base.update(over)
    return make_row(**base)


def with_units(*rows):
    return add_loading_unit(long_frame(*rows))


def test_kilo_meet_is_classified_kg():
    out = with_units(*[make_row(Name=f"L{i}") for i in range(4)])
    assert out["loading_unit"].unique().to_list() == ["kg"]
    assert out["lb_share"][0] == 0.0


def test_pound_meet_is_classified_lb():
    out = with_units(*[lb_row(Name=f"L{i}") for i in range(4)])
    assert out["loading_unit"].unique().to_list() == ["lb"]
    assert out["lb_share"][0] == 1.0


def test_mixed_meet_is_null():
    rows = [make_row(Name=f"K{i}") for i in range(2)] + [lb_row(Name=f"P{i}") for i in range(2)]
    out = with_units(*rows)
    assert out["loading_unit"].unique().to_list() == [None]


def test_loading_unit_is_per_meet_not_per_attempt():
    # Every attempt of a pound meet carries 'lb', including the ones that
    # happen to sit on the kilo grid.
    out = with_units(*[lb_row(Name=f"L{i}") for i in range(4)])
    assert out.filter(pl.col("loading_unit") != "lb").height == 0


def test_small_meets_get_no_loading_unit():
    # One lifter is 9 attempts; two attempts is far too few to classify.
    out = add_loading_unit(long_frame(make_row(Bench1Kg=None, Deadlift1Kg=None)))
    assert out.height == 3
    assert out["loading_unit"].unique().to_list() == [None]
