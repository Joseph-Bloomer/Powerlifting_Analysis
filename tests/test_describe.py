import numpy as np
import polars as pl
import pytest

import describe as d
import plotstyle as ps


def attempts_frame(n: int = 4000, seed: int = 0) -> pl.DataFrame:
    """A small full-shape frame in the layout describe.py reads."""
    rng = np.random.default_rng(seed)
    weights = np.round(rng.normal(200, 30, n) / 2.5) * 2.5
    weights[: n // 10] = 227.5  # a planted spike so a milestone has mass
    prev = weights - rng.uniform(2.5, 15, n)
    return pl.DataFrame(
        {
            "meet_id": [f"m{i % 50}" for i in range(n)],
            "lifter_id": [f"l{i}" for i in range(n)],
            "lift": rng.choice(["squat", "bench", "deadlift"], n),
            "attempt_no": rng.choice([1, 2, 3], n).astype(np.int8),
            "weight_kg": weights,
            "success": rng.random(n) < 0.6,
            "jump_kg": weights - prev,
            "jump_pct": 100 * (weights - prev) / prev,
            "dist_to_next_lb_ms": rng.uniform(0, 22, n),
            "lb_region": np.full(n, 1, dtype=np.int8),
            "loading_unit": ["kg"] * n,
            "split": rng.choice(["discovery", "test"], n),
        }
    )


def milestones_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "group": ["us_kg"] * 3,
            "lift": ["squat", "bench", "deadlift"],
            "bin": [227.5, 227.5, 227.5],
            "prespecified_kind": ["lb100"] * 3,
            "discovered": [True, False, True],
        }
    )


# --- binning ----------------------------------------------------------------


def test_uniform_bin_indexes_are_ordered_by_value():
    """Regression: the first version keyed bins on the interval's string, so
    "[10, 12)" sorted before "[2, 3)" and the plotted lines doubled back."""
    values = [0.5, 1.5, 2.5, 9.5, 10.5, 12.5, 20.5]
    out = (
        pl.DataFrame({"x": values})
        .select(d.uniform_bin(pl.col("x"), 0.0, 1.0, 25).alias("b"))["b"]
        .to_list()
    )
    assert out == [0, 1, 2, 9, 10, 12, 20]
    assert out == sorted(out)  # value order and bin order agree


def test_uniform_bin_respects_the_width():
    out = (
        pl.DataFrame({"x": [0.0, 2.4, 2.6, 7.6]})
        .select(d.uniform_bin(pl.col("x"), 0.0, 2.5, 10).alias("b"))["b"]
        .to_list()
    )
    assert out == [0, 0, 1, 3]


def test_bin_centres_sit_in_the_middle():
    assert d.bin_centres(0.0, 2.5, 3).tolist() == [1.25, 3.75, 6.25]


def test_bin_centres_index_alignment():
    # A bin index must address its own centre.
    centres = d.bin_centres(0.0, 1.0, 25)
    assert centres[10] == pytest.approx(10.5)


# --- Wilson interval --------------------------------------------------------


def test_wilson_brackets_the_point_estimate():
    lo, hi = d.wilson(np.array([50.0]), np.array([100.0]))
    assert lo[0] < 0.5 < hi[0]


def test_wilson_narrows_as_n_grows():
    narrow = np.subtract(*reversed(d.wilson(np.array([5000.0]), np.array([10000.0]))))
    wide = np.subtract(*reversed(d.wilson(np.array([5.0]), np.array([10.0]))))
    assert narrow[0] < wide[0]


def test_wilson_stays_inside_zero_one_at_the_extremes():
    lo, hi = d.wilson(np.array([0.0, 20.0]), np.array([20.0, 20.0]))
    assert lo[0] >= 0 and hi[1] <= 1


# --- frame preparation ------------------------------------------------------


def test_group_frame_labels_and_converts():
    out = d.group_frame(attempts_frame(200))
    assert out["group"].unique().to_list() == ["us_kg"]
    # A kilo-loaded group keeps kilos as its native unit.
    assert out["weight_native"].to_list() == out["weight_kg"].to_list()


def test_group_frame_converts_pound_meets_to_pounds():
    df = attempts_frame(200).with_columns(pl.lit("lb").alias("loading_unit"))
    out = d.group_frame(df)
    assert out["group"].unique().to_list() == ["us_lb"]
    assert out["weight_native"][0] == pytest.approx(out["weight_kg"][0] / 0.45359237)


def test_group_frame_leaves_excluded_rows_null():
    df = attempts_frame(200).with_columns(pl.lit(None, dtype=pl.Utf8).alias("loading_unit"))
    out = d.group_frame(df)
    assert out["group"].null_count() == out.height
    assert out["weight_native"].null_count() == out.height


# --- excess-mass table ------------------------------------------------------


def test_excess_mass_table_carries_labels_and_verdicts():
    table = d.excess_mass_table(d.group_frame(attempts_frame()), milestones_frame())
    assert {"excess_pct", "prespecified_kind", "discovered_in_half"} <= set(table.columns)
    spike = table.filter((pl.col("bin") == 227.5) & (pl.col("lift") == "deadlift"))
    assert spike["prespecified_kind"][0] == "lb100"
    assert bool(spike["discovered_in_half"][0]) is True


def test_excess_mass_table_defaults_unjoined_bins():
    table = d.excess_mass_table(d.group_frame(attempts_frame()), milestones_frame())
    other = table.filter(pl.col("bin") != 227.5)
    assert other["prespecified_kind"].null_count() == 0
    assert not other["discovered_in_half"].any()


def test_excess_mass_finds_the_planted_spike():
    table = d.excess_mass_table(d.group_frame(attempts_frame()), milestones_frame())
    spike = table.filter((pl.col("bin") == 227.5) & pl.col("scanned"))
    assert spike.height > 0
    assert (spike["excess_pct"] > 0).all()


# --- figures render ---------------------------------------------------------


def test_every_figure_writes_a_file(tmp_path, monkeypatch):
    """Smoke test. The plotting bugs this file has caught were silent: the
    figure was written, it was just wrong."""
    monkeypatch.setattr(d, "FIG_DIR", tmp_path)
    monkeypatch.setattr(d, "MIN_BIN_N", 1)
    df = d.group_frame(attempts_frame())
    table = d.excess_mass_table(df, milestones_frame())

    assert d.plot_histograms(table, "deadlift").exists()
    assert d.plot_success_by_jump(df).exists()
    assert d.plot_jump_vs_distance(df).exists()
    assert d.plot_excess_mass(table).exists()


def test_histogram_returns_none_for_a_lift_with_no_data(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "FIG_DIR", tmp_path)
    table = d.excess_mass_table(d.group_frame(attempts_frame()), milestones_frame())
    assert d.plot_histograms(table.filter(pl.col("lift") == "nothing"), "nothing") is None


# --- shared styling ---------------------------------------------------------


def test_palette_slots_are_distinct():
    assert len(set(ps.SERIES)) == len(ps.SERIES)


def test_panels_are_the_three_discovery_groups():
    from discover import GROUPS

    assert [g for g, _ in ps.PANELS] == list(GROUPS)


def test_describe_never_reads_outcomes_for_the_histograms():
    """The histograms and excess mass are attempt selection only, so dropping
    the success column entirely must not change them."""
    df = d.group_frame(attempts_frame())
    without = df.drop("success")
    a = d.excess_mass_table(df, milestones_frame())
    b = d.excess_mass_table(without, milestones_frame())
    assert a.equals(b)
