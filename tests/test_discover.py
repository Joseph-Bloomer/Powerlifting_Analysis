import numpy as np
import polars as pl
import pytest

from constants import KG_PER_LB
from discover import (
    DISCOVERY_COLUMNS,
    GROUPS,
    add_discovered_flag,
    assign_groups,
    assign_split,
    bin_counts,
    correct_for_multiplicity,
    discover,
    fit_counterfactual,
    normal_sf,
    prespecified_kind,
    scan_cell,
)
from features import POST_ATTEMPT_COLUMNS


def attempts(
    weights: list[float],
    *,
    lift: str = "deadlift",
    group: str = "us_kg",
    meets: list[str] | None = None,
) -> pl.DataFrame:
    """Third attempts in the shape discover() expects from the features table."""
    spec = GROUPS[group]
    n = len(weights)
    meets = meets or [f"m{i % 40}" for i in range(n)]
    return pl.DataFrame(
        {
            "meet_id": meets,
            "lifter_id": [f"l{i}" for i in range(n)],
            "lift": [lift] * n,
            "attempt_no": np.full(n, 3, dtype=np.int8),
            "weight_kg": weights,
            "lb_region": np.full(n, spec["lb_region"], dtype=np.int8),
            "loading_unit": [spec["loading_unit"]] * n,
        }
    )


def planted(
    spike_kg: float = 227.5,
    n_flat: int = 9000,
    n_spike: int = 900,
    seed: int = 0,
    group: str = "us_kg",
) -> pl.DataFrame:
    """A smooth background with one weight deliberately over-represented."""
    rng = np.random.default_rng(seed)
    flat = np.round(rng.normal(200, 35, n_flat) / 2.5) * 2.5
    flat = flat[(flat > 80) & (flat < 330)]
    weights = np.concatenate([flat, np.full(n_spike, spike_kg)])
    return attempts(list(weights), group=group)


def with_split(df: pl.DataFrame, half: str = "discovery") -> pl.DataFrame:
    return assign_groups(df.with_columns(pl.lit(half).alias("split")))


# --- split ------------------------------------------------------------------


def test_split_is_deterministic_across_calls():
    ids = pl.Series("meet_id", [f"meet-{i}" for i in range(500)])
    assert assign_split(ids).equals(assign_split(ids))


def test_split_is_stable_under_input_order():
    ids = [f"meet-{i}" for i in range(200)]
    a = assign_split(pl.Series("meet_id", ids))
    b = assign_split(pl.Series("meet_id", list(reversed(ids))))
    assert a.sort("meet_id").equals(b.sort("meet_id"))


def test_split_is_roughly_even():
    out = assign_split(pl.Series("meet_id", [f"meet-{i}" for i in range(4000)]))
    share = (out["split"] == "discovery").mean()
    assert 0.45 < share < 0.55


def test_split_uses_both_names():
    out = assign_split(pl.Series("meet_id", [f"m{i}" for i in range(100)]))
    assert sorted(out["split"].unique().to_list()) == ["discovery", "test"]


def test_every_meet_gets_exactly_one_split():
    ids = pl.Series("meet_id", ["a", "b", "a", "c", "b"])
    out = assign_split(ids)
    assert out.height == 3
    assert out["meet_id"].n_unique() == 3


# --- no outcome data --------------------------------------------------------


def test_discovery_columns_exclude_every_outcome():
    assert not set(DISCOVERY_COLUMNS) & set(POST_ATTEMPT_COLUMNS)


def test_success_is_not_a_discovery_column():
    assert "success" not in DISCOVERY_COLUMNS


def test_detection_ignores_outcome_columns_entirely():
    """The behavioural guard: outcomes may change arbitrarily, output may not.

    A column whitelist can be widened by accident in a later edit. This fails
    on behaviour instead, so it keeps biting after someone 'just adds one more
    column' to the loader.
    """
    base = with_split(planted())
    rng = np.random.default_rng(7)
    n = base.height

    polluted = base.with_columns(
        pl.Series("success", rng.random(n) < 0.5),
        pl.Series("best3_kg", rng.uniform(100, 300, n)),
        pl.Series("TotalKg", rng.uniform(300, 800, n)),
        pl.Series("Place", rng.integers(1, 20, n)),
    )
    inverted = polluted.with_columns(
        ~pl.col("success"),
        pl.col("best3_kg") * -1,
        pl.col("TotalKg") * 0,
        pl.col("Place") * 0,
    )
    assert discover(polluted).equals(discover(inverted))
    assert discover(polluted).equals(discover(base))


def test_detection_is_unchanged_when_outcomes_are_absent():
    base = with_split(planted())
    withcols = base.with_columns(pl.lit(True).alias("success"))
    assert discover(base).equals(discover(withcols))


# --- binning and counterfactual ---------------------------------------------


def test_bin_counts_fills_gaps_with_zero():
    edges, counts = bin_counts(np.array([100.0, 100.0, 110.0]), 2.5)
    assert len(edges) == 5  # 100, 102.5, 105, 107.5, 110
    assert counts.tolist() == [2, 0, 0, 0, 1]


def test_bin_counts_snaps_to_the_grid():
    edges, counts = bin_counts(np.array([101.0, 99.4]), 2.5)
    assert counts.sum() == 2


def test_counterfactual_recovers_a_smooth_trend():
    # No spike: the fit should land near the actual count.
    counts = np.array([100 + 2 * i for i in range(60)], dtype=float)
    assert fit_counterfactual(counts, 30) == pytest.approx(counts[30], rel=0.05)


def test_counterfactual_ignores_the_candidate_bin():
    # The spike must not pull up its own baseline.
    counts = np.full(60, 100.0)
    counts[30] = 5000.0
    assert fit_counterfactual(counts, 30) == pytest.approx(100.0, rel=0.15)


def test_counterfactual_ignores_spillover_next_to_the_candidate():
    counts = np.full(60, 100.0)
    counts[29:32] = [400.0, 5000.0, 400.0]
    assert fit_counterfactual(counts, 30) == pytest.approx(100.0, rel=0.25)


def test_counterfactual_is_never_negative_in_sparse_tails():
    counts = np.concatenate([np.full(30, 200.0), np.linspace(200, 0, 30)])
    values = [fit_counterfactual(counts, i) for i in range(len(counts))]
    assert all(v > 0 or np.isnan(v) for v in values)


def test_counterfactual_returns_nan_when_it_cannot_fit():
    assert np.isnan(fit_counterfactual(np.zeros(60), 30))
    assert np.isnan(fit_counterfactual(np.array([1.0, 2.0]), 1))


def test_scan_cell_marks_a_planted_spike():
    rng = np.random.default_rng(1)
    w = np.concatenate([
        np.round(rng.normal(200, 30, 20000) / 2.5) * 2.5,
        np.full(2000, 227.5),
    ])
    cell = scan_cell(w, 2.5)
    row = cell.filter(pl.col("bin") == 227.5).to_dicts()[0]
    assert row["excess"] > 0
    assert row["z"] > 5
    flat = cell.filter(pl.col("scanned") & (pl.col("bin") == 180.0)).to_dicts()[0]
    assert abs(flat["z"]) < 5


def test_dispersion_is_at_least_one():
    rng = np.random.default_rng(2)
    cell = scan_cell(np.round(rng.normal(200, 30, 5000) / 2.5) * 2.5, 2.5)
    assert cell["dispersion"][0] >= 1.0


# --- multiplicity -----------------------------------------------------------


def test_bh_is_monotone_and_at_least_the_raw_p():
    p = np.array([0.001, 0.01, 0.02, 0.2, 0.9])
    out = correct_for_multiplicity(p)
    assert np.all(out["p_bh"] >= p - 1e-12)
    assert np.all(np.diff(out["p_bh"][np.argsort(p)]) >= -1e-12)


def test_bh_is_less_conservative_than_bonferroni():
    p = np.array([0.001, 0.01, 0.02, 0.2, 0.9])
    out = correct_for_multiplicity(p)
    assert np.all(out["p_bh"] <= out["p_bonferroni"] + 1e-12)


def test_bh_handles_nan_without_counting_it_as_a_test():
    p = np.array([0.01, np.nan, 0.02])
    out = correct_for_multiplicity(p)
    assert out["n_tests"] == 2
    assert np.isnan(out["p_bh"][1])


def test_bonferroni_multiplies_by_the_number_of_tests():
    p = np.array([0.01, 0.02])
    assert correct_for_multiplicity(p)["p_bonferroni"].tolist() == [0.02, 0.04]


def test_all_null_p_values_are_handled():
    out = correct_for_multiplicity(np.array([np.nan, np.nan]))
    assert out["n_tests"] == 0


def test_one_sided_p_ignores_deficits():
    # A hole is not a milestone: a large negative z must not be significant.
    assert normal_sf(np.array([-6.0]))[0] > 0.99
    assert normal_sf(np.array([6.0]))[0] < 1e-8


# --- pre-specified coincidence ----------------------------------------------


def evaluate_kind(values, unit):
    return (
        pl.DataFrame({"b": values})
        .select(prespecified_kind(pl.col("b"), unit).alias("k"))["k"]
        .to_list()
    )


def test_kind_labels_kilo_meet_pound_milestones():
    # 227.5 kg is the loadable stand-in for 500 lb, a 100 lb multiple, so the
    # coarser grade wins. 250.0 is 550 lb: a 50 lb multiple only.
    assert evaluate_kind([227.5, 250.0], "kg") == ["lb100", "lb50"]


def test_kind_prefers_the_coarser_grade():
    # 500 lb is both a 50 and a 100 lb multiple; report the rounder one.
    assert evaluate_kind([500 * KG_PER_LB], "lb") == ["lb100"]
    assert evaluate_kind([550 * KG_PER_LB], "lb") == ["lb50"]


def test_kind_labels_kilo_milestones():
    assert evaluate_kind([200.0, 210.0], "kg") == ["kg50", "kg10"]


def test_kind_reports_none_for_unremarkable_weights():
    assert evaluate_kind([212.5], "kg") == ["none"]


# --- end to end -------------------------------------------------------------


def test_discover_finds_the_planted_weight():
    out = discover(with_split(planted()))
    found = out.filter(pl.col("discovered"))
    assert 227.5 in found["bin"].to_list()
    assert found.filter(pl.col("bin") == 227.5)["prespecified_kind"][0] == "lb100"


def test_discover_finds_nothing_in_smooth_data():
    rng = np.random.default_rng(3)
    w = np.round(rng.normal(200, 35, 20000) / 2.5) * 2.5
    out = discover(with_split(attempts(list(w))))
    assert out.filter(pl.col("discovered")).height == 0


def test_discover_uses_the_discovery_half_only():
    df = with_split(planted(), half="test")
    assert discover(df).height == 0


def test_discover_uses_third_attempts_only():
    df = with_split(planted()).with_columns(pl.lit(2, dtype=pl.Int8).alias("attempt_no"))
    assert discover(df).height == 0


def test_discover_keeps_groups_in_their_own_unit():
    out = discover(with_split(planted(group="us_lb", n_flat=9000, n_spike=900)))
    assert out["bin_unit"].unique().to_list() == ["lb"]


def test_discovered_flag_marks_matching_attempts():
    df = with_split(planted())
    milestones = discover(df)
    flagged = add_discovered_flag(df, milestones)
    on = flagged.filter(pl.col("weight_kg") == 227.5)
    assert on.height > 0
    assert on["on_discovered_milestone"].all()
    off = flagged.filter(pl.col("weight_kg") == 180.0)
    assert not off["on_discovered_milestone"].any()


def test_discovered_flag_is_null_outside_the_groups():
    df = with_split(planted()).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("loading_unit")
    )
    df = assign_groups(df.drop("group"))
    flagged = add_discovered_flag(df, discover(with_split(planted())))
    assert flagged["on_discovered_milestone"].null_count() == flagged.height


def test_discovered_flag_covers_all_attempt_numbers():
    # Discovery looks at thirds; the flag describes the weight, so a first
    # attempt at a milestone weight is still flagged.
    df = with_split(planted())
    milestones = discover(df)
    firsts = df.with_columns(pl.lit(1, dtype=pl.Int8).alias("attempt_no"))
    flagged = add_discovered_flag(firsts, milestones)
    assert flagged.filter(pl.col("weight_kg") == 227.5)["on_discovered_milestone"].all()


def test_discovered_bins_clear_both_floors():
    out = discover(with_split(planted()))
    found = out.filter(pl.col("discovered"))
    assert (found["count"] >= 50).all()
    assert (found["excess_pct"] >= 0.15).all()
