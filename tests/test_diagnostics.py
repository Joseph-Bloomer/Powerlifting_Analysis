import numpy as np
import polars as pl
import pytest

import diagnostics as dg
import econ_models as em
from constants import KG_PER_LB
from test_econ_models import attempts


def sample(n: int = 4000, seed: int = 3) -> pl.DataFrame:
    """A headline sample with group labels, as diagnostics expects it."""
    rng = np.random.default_rng(seed)
    df = attempts(n=n, seed=seed).with_columns(
        pl.Series("lifter_id", [f"l{i % 250}" for i in range(n)]),
        pl.Series("meet_id", [f"m{i % 90}" for i in range(n)]),
        pl.Series("loading_unit", rng.choice(["kg", "lb"], n, p=[0.9, 0.1])),
        pl.Series("dist_to_next_lb_ms", rng.uniform(0, 22, n)),
        pl.Series("dist_to_next_kg_ms", rng.uniform(0, 10, n)),
    )
    return dg.assign_group(em.build_sample(df))


# --- the identifying sample -------------------------------------------------


def test_identifying_sample_needs_both_conditions():
    base = sample()
    restricted, fe_matching, stats = dg.identifying_sample(base)
    per = base.group_by("lifter_id").agg(
        pl.col("meet_id").n_unique().alias("meets"),
        pl.col("on_lb_milestone").n_unique().alias("lv"),
    )
    expected = per.filter((pl.col("meets") > 1) & (pl.col("lv") > 1)).height
    assert stats["lifters_identifying"] == expected
    assert restricted["lifter_id"].n_unique() == expected


def test_identifying_sample_is_a_subset():
    base = sample()
    restricted, _, _ = dg.identifying_sample(base)
    assert restricted.height <= base.height
    assert set(restricted["lifter_id"].unique()) <= set(base["lifter_id"].unique())


def test_identifying_sample_drops_single_meet_lifters():
    base = sample()
    restricted, _, _ = dg.identifying_sample(base)
    meets = restricted.group_by("lifter_id").agg(pl.col("meet_id").n_unique().alias("m"))
    assert (meets["m"] > 1).all()


def test_identifying_sample_drops_lifters_never_on_a_milestone():
    base = sample()
    restricted, _, _ = dg.identifying_sample(base)
    lv = restricted.group_by("lifter_id").agg(
        pl.col("on_lb_milestone").n_unique().alias("lv")
    )
    assert (lv["lv"] > 1).all()


def test_stats_report_the_multi_region_count():
    _, _, stats = dg.identifying_sample(sample())
    assert "lifters_multi_region" in stats
    assert stats["lifters_multi_region"] <= stats["lifters_total"]


# --- the headline is not altered -------------------------------------------


def test_diagnostics_does_not_change_the_headline_terms():
    """The whole point of this module is to leave the specification alone."""
    assert em.BASE_TERMS == [
        "jump_pct", "on_lb_milestone", "on_kg_milestone", "lb_region",
        "lb_ms_x_region", "kg_ms_x_region", "prior_meets", "would_be_pr",
        "no_history",
    ]


def test_dropping_jump_pct_changes_only_that_term():
    base = sample()
    with_jump = em.fit_model(base, "a", em.BASE_TERMS, em.FE_BASE)
    without = em.fit_model(
        base, "b", [t for t in em.BASE_TERMS if t != "jump_pct"], em.FE_BASE
    )
    assert "jump_pct" in with_jump.terms
    assert "jump_pct" not in without.terms
    assert set(without.terms) < set(with_jump.terms)


# --- the placebo family -----------------------------------------------------


def test_placebo_offsets_cover_the_documented_range():
    assert dg.PLACEBO_OFFSETS_KG[0] == 2.5 and dg.PLACEBO_OFFSETS_KG[-1] == 20.0
    assert dg.PLACEBO_OFFSETS_LB[0] == 5.0 and dg.PLACEBO_OFFSETS_LB[-1] == 40.0
    assert len(dg.PLACEBO_OFFSETS_KG) == len(dg.PLACEBO_OFFSETS_LB) == 8


def test_default_offset_reproduces_the_econ_placebo():
    """Adding the parameter must not have moved the headline placebo."""
    weights = pl.DataFrame({"w": [227.5 + em.PLACEBO_OFFSET_KG, 240.0, 227.5]})
    default = weights.select(
        em.on_fake_milestone(pl.col("w"), "kg", em.LB_STEP).alias("f")
    )["f"].to_list()
    explicit = weights.select(
        em.on_fake_milestone(
            pl.col("w"), "kg", em.LB_STEP, offset_kg=em.PLACEBO_OFFSET_KG
        ).alias("f")
    )["f"].to_list()
    assert default == explicit


def test_each_placebo_offset_flags_a_different_set():
    base = sample()
    seen = []
    for off_kg, off_lb in zip(dg.PLACEBO_OFFSETS_KG[:4], dg.PLACEBO_OFFSETS_LB[:4]):
        flagged = dg.with_placebo(base, off_kg, off_lb)["on_fake_milestone"].to_list()
        seen.append(tuple(flagged))
    assert len(set(seen)) == len(seen)


def test_placebo_flag_is_never_the_real_milestone():
    """Checked on the loadable grid, not on a fixture: the question is about
    the milestone rules, not about any particular sample."""
    from features import on_lb_milestone_kg_meet

    grid = pl.DataFrame({"w": np.arange(60.0, 330.0, 2.5)})
    real = grid.select(
        on_lb_milestone_kg_meet(pl.col("w"), em.LB_STEP).alias("r")
    )["r"]
    for off_kg in dg.PLACEBO_OFFSETS_KG:
        fake = grid.select(
            em.on_fake_milestone(
                pl.col("w"), "kg", em.LB_STEP, offset_kg=off_kg
            ).alias("f")
        )["f"]
        assert fake.any(), f"offset {off_kg} flags nothing"
        assert not (real & fake).any(), f"offset {off_kg} collides with a real one"


def test_pound_offsets_are_identified_but_kilo_offsets_are_not():
    """In a pound-loaded meet the bar carries whole pounds. A kilo offset lands
    between pound weights where no attempt can ever sit, so the flag would be
    constant and unidentified; the equivalent pound offsets avoid that."""
    pounds = np.arange(200.0, 700.0, 5.0) * KG_PER_LB
    grid = pl.DataFrame({"w": pounds})

    by_pound = grid.select(
        em.on_fake_milestone(
            pl.col("w"), "lb", em.LB_STEP, offset_kg=25.0 * KG_PER_LB
        ).alias("f")
    )["f"]
    by_kilo = grid.select(
        em.on_fake_milestone(pl.col("w"), "lb", em.LB_STEP, offset_kg=12.5).alias("f")
    )["f"]
    assert by_pound.sum() > 0
    assert by_kilo.sum() == 0


# --- the asymmetry grids ----------------------------------------------------


def test_kg_running_variable_is_signed_around_a_ten():
    df = pl.DataFrame({"prev_weight_kg": [198.0, 202.0, 200.0]})
    out = dg.add_kg_running_variable(df)["running_kg"].to_list()
    assert out[0] < 0 < out[1]
    assert out[2] == pytest.approx(0.0)


def test_kg_grid_lists_ten_kilo_multiples():
    grid = dg.milestone_grid("kg", 195.0, 225.0)
    assert grid == [200.0, 210.0, 220.0]


def test_pound_grid_lists_loadable_kilo_weights():
    # 450 lb -> 205.0, 500 lb -> 227.5, 550 lb -> 250.0
    grid = dg.milestone_grid("lb", 200.0, 255.0)
    assert 227.5 in grid
    assert all(abs(v / 2.5 - round(v / 2.5)) < 1e-9 for v in grid)


def test_the_two_grids_overlap_every_ninety_kilos():
    """They are not disjoint, which the design treats them as being.

    The loadable pound grid is mostly spaced 22.5 kg against the kilo grid's
    10 kg, so coincidences fall roughly every 90 kg -- not exactly, because
    rounding each pound figure up to the 2.5 kg grid drifts (700 lb is
    317.51 kg and rounds to 320.0, not 317.5). A third of pound milestones in
    the competitive range are also kilo milestones, and at those weights
    nothing can separate the two treatments. Diagnostic 5 reports this.
    """
    kg = set(dg.milestone_grid("kg", 60.0, 320.0))
    lb = set(dg.milestone_grid("lb", 60.0, 320.0))
    shared = sorted(kg & lb)
    assert shared == [70.0, 160.0, 250.0, 320.0]
    assert len(shared) / len(lb) == pytest.approx(1 / 3, abs=0.01)
    gaps = [b - a for a, b in zip(shared, shared[1:])]
    assert gaps == [90.0, 90.0, 70.0]


def test_pound_grid_spacing_is_not_uniform():
    """The rounding drift that breaks the 90 kg periodicity."""
    grid = dg.milestone_grid("lb", 60.0, 320.0)
    gaps = {round(b - a, 2) for a, b in zip(grid, grid[1:])}
    assert gaps == {22.5, 25.0}


def test_pound_and_kilo_running_variables_disagree():
    df = pl.DataFrame({"prev_weight_kg": [225.0], "loading_unit": ["kg"]})
    lb = em.add_running_variable(df)["running_kg"][0]
    kg = dg.add_kg_running_variable(df)["running_kg"][0]
    assert abs(lb - kg) > 1e-6


# --- bands ------------------------------------------------------------------


def test_prior_meet_bands_are_contiguous_and_cover_zero():
    lows = [lo for _, lo, _ in dg.PRIOR_MEET_BANDS]
    highs = [hi for _, _, hi in dg.PRIOR_MEET_BANDS]
    assert lows[0] == 0
    for prev_hi, lo in zip(highs, lows[1:]):
        assert lo == prev_hi + 1


def test_bands_partition_the_sample():
    base = sample()
    total = sum(
        base.filter(pl.col("prior_meets").is_between(lo, hi)).height
        for _, lo, hi in dg.PRIOR_MEET_BANDS
    )
    assert total == base.height


# --- reporting --------------------------------------------------------------


def test_interval_reports_a_confidence_interval():
    fit = em.fit_model(sample(), "x", em.BASE_TERMS, em.FE_BASE)
    text = dg.interval(fit, "on_lb_milestone")
    assert "95% CI" in text and "pp" in text


def test_interval_handles_a_term_that_was_dropped():
    fit = em.fit_model(sample(), "x", em.BASE_TERMS, em.FE_BASE)
    assert dg.interval(fit, "not_a_term") == "not estimated"


def test_spans_zero_detects_a_null():
    fit = em.Fit("x", ["a"], np.array([0.0]), np.array([1.0]), 10, 5, 0.5)
    assert dg.spans_zero(fit, "a")
    certain = em.Fit("y", ["a"], np.array([1.0]), np.array([0.001]), 10, 5, 0.5)
    assert not dg.spans_zero(certain, "a")


def test_binned_floors_to_the_grid():
    out = (
        pl.DataFrame({"x": [0.0, 2.4, 2.6, 7.6]})
        .select(dg.binned(pl.col("x"), 2.5).alias("b"))["b"]
        .to_list()
    )
    assert out == [0.0, 0.0, 2.5, 7.5]


def test_assign_group_matches_the_discover_definition():
    from discover import GROUPS as DISCOVER_GROUPS

    assert [g for g, _ in dg.GROUPS] == list(DISCOVER_GROUPS)


def test_fe_matching_restriction_is_the_looser_one():
    """FE_LIFTER absorbs lifter_id, not lifter x meet, so a single-meet lifter
    whose milestone status varies across lifts still identifies the estimate."""
    base = sample()
    as_specified, fe_matching, stats = dg.identifying_sample(base)
    assert fe_matching.height >= as_specified.height
    assert stats["lifters_ms_varies"] >= stats["lifters_identifying"]
    assert set(as_specified["lifter_id"].unique()) <= set(
        fe_matching["lifter_id"].unique()
    )
