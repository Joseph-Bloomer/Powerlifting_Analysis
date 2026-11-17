import datetime as dt

import numpy as np
import polars as pl
import pytest

import econ_models as em
from constants import KG_PER_LB


def toy(n: int = 4000, seed: int = 0) -> pl.DataFrame:
    """A small frame in the shape build_sample() produces."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 40, n)  # lifter
    f = rng.integers(0, 6, n)  # another fixed effect
    x1 = rng.normal(size=n) + 0.3 * g
    x2 = rng.normal(size=n)
    # A lifter-level shock, so clustering has something real to correct for.
    shock = rng.normal(scale=0.8, size=40)[g]
    y = 0.4 * x1 - 0.25 * x2 + g * 0.05 + f * 0.1 + shock + rng.normal(scale=0.5, size=n)
    return pl.DataFrame(
        {
            "lifter_id": [f"l{i}" for i in g],
            "fe2": [f"f{i}" for i in f],
            "x1": x1,
            "x2": x2,
            "y": y,
        }
    )


def attempts(n: int = 3000, seed: int = 1) -> pl.DataFrame:
    """A frame in the shape the discover stage emits."""
    rng = np.random.default_rng(seed)
    weight = np.round(rng.normal(200, 30, n) / 2.5) * 2.5
    prev = weight - rng.uniform(2.5, 15, n)
    return pl.DataFrame(
        {
            "lifter_id": [f"l{i % 300}" for i in range(n)],
            "meet_id": [f"m{i % 60}" for i in range(n)],
            "lift": rng.choice(["squat", "bench", "deadlift"], n),
            "attempt_no": np.full(n, 3, dtype=np.int8),
            "weight_kg": weight,
            "prev_weight_kg": prev,
            "prev_success": np.ones(n, dtype=bool),
            "success": rng.random(n) < 0.6,
            "jump_kg": weight - prev,
            "jump_pct": 100 * (weight - prev) / prev,
            "on_lb_milestone": rng.random(n) < 0.13,
            "on_kg_milestone": rng.random(n) < 0.3,
            "on_discovered_milestone": rng.random(n) < 0.1,
            "lb_region": rng.integers(0, 2, n).astype(np.int8),
            "loading_unit": ["kg"] * n,
            "split": rng.choice(["discovery", "test"], n),
            "prior_meets": rng.integers(0, 10, n),
            "would_be_pr": rng.random(n) < 0.5,
            "Sex": rng.choice(["M", "F"], n),
            "Age": rng.uniform(18, 60, n),
            "WeightClassKg": rng.choice(["83", "93", "105"], n),
            "Tested": rng.choice(["Yes", None], n),
            "Date": [dt.date(2015 + i % 8, 1, 1) for i in range(n)],
        }
    )


# --- absorption -------------------------------------------------------------


def test_absorb_one_group_equals_demeaning():
    df = toy()
    codes, nlev = em.codes_of(df, "lifter_id")
    values = df.select("x1", "x2").to_numpy()
    out = em.absorb(values, [(codes, nlev)])
    expected = values - np.array(
        [np.bincount(codes, weights=values[:, j], minlength=nlev)
         / np.bincount(codes, minlength=nlev) for j in range(2)]
    ).T[codes]
    assert np.allclose(out, expected)


def test_absorb_leaves_group_means_at_zero():
    df = toy()
    groups = [em.codes_of(df, c) for c in ("lifter_id", "fe2")]
    out = em.absorb(df.select("x1", "x2").to_numpy(), groups)
    for codes, nlev in groups:
        means = np.bincount(codes, weights=out[:, 0], minlength=nlev) / np.bincount(
            codes, minlength=nlev
        )
        assert np.abs(means).max() < 1e-6


def test_absorb_with_no_groups_just_centres():
    values = np.array([[1.0, 10.0], [3.0, 20.0]])
    assert np.allclose(em.absorb(values, []), values - values.mean(axis=0))


def test_codes_of_gives_nulls_their_own_level():
    df = pl.DataFrame({"g": ["a", None, "a", "b"]})
    codes, nlev = em.codes_of(df, "g")
    assert nlev == 3
    assert codes[0] == codes[2] and codes[1] not in (codes[0], codes[3])


# --- the estimator, against statsmodels -------------------------------------


def test_absorbed_coefficients_match_dummy_regression():
    """Frisch-Waugh-Lovell: absorbing a fixed effect must give the same slopes
    as putting it in as dummies."""
    import statsmodels.api as sm

    df = toy()
    codes, nlev = em.codes_of(df, "lifter_id")
    stacked = em.absorb(df.select("y", "x1", "x2").to_numpy(), [(codes, nlev)])
    beta, _ = em.ols_cluster(stacked[:, 0], stacked[:, 1:], codes, nlev, nlev)

    dummies = np.eye(nlev)[codes][:, 1:]
    design = sm.add_constant(np.hstack([df.select("x1", "x2").to_numpy(), dummies]))
    reference = sm.OLS(df["y"].to_numpy(), design).fit()

    assert np.allclose(beta, reference.params[1:3], atol=1e-8)


def test_clustered_standard_errors_match_statsmodels():
    """The sandwich and its finite-sample factor, checked against a library."""
    import statsmodels.api as sm

    df = toy()
    codes, nlev = em.codes_of(df, "lifter_id")
    stacked = em.absorb(df.select("y", "x1", "x2").to_numpy(), [(codes, nlev)])
    _, se = em.ols_cluster(stacked[:, 0], stacked[:, 1:], codes, nlev, nlev)

    dummies = np.eye(nlev)[codes][:, 1:]
    design = sm.add_constant(np.hstack([df.select("x1", "x2").to_numpy(), dummies]))
    reference = sm.OLS(df["y"].to_numpy(), design).fit(
        cov_type="cluster", cov_kwds={"groups": codes}
    )
    assert np.allclose(se, reference.bse[1:3], rtol=1e-6)


def test_clustered_errors_exceed_plain_ones_under_within_cluster_correlation():
    import statsmodels.api as sm

    df = toy()
    codes, nlev = em.codes_of(df, "lifter_id")
    X = sm.add_constant(df.select("x1").to_numpy())
    y = df["y"].to_numpy()
    plain = sm.OLS(y, X).fit().bse[1]
    clustered = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": codes}).bse[1]
    assert clustered > plain


# --- sample construction ----------------------------------------------------


def test_build_sample_keeps_only_thirds_after_a_successful_second():
    df = attempts()
    mixed = pl.concat([
        df,
        df.head(100).with_columns(pl.lit(2, dtype=pl.Int8).alias("attempt_no")),
        df.head(100).with_columns(pl.lit(False).alias("prev_success")),
    ])
    out = em.build_sample(mixed)
    assert (out["attempt_no"] == 3).all()
    assert out["prev_success"].all()


def test_build_sample_drops_rows_without_a_region():
    df = attempts().with_columns(
        pl.when(pl.int_range(pl.len()) < 50)
        .then(None)
        .otherwise(pl.col("lb_region"))
        .alias("lb_region")
    )
    assert em.build_sample(df).height == df.height - 50


def test_build_sample_keeps_lifters_with_unknown_age():
    df = attempts().with_columns(pl.lit(None, dtype=pl.Float64).alias("Age"))
    out = em.build_sample(df)
    assert out.height == df.height
    assert out["age_class"].unique().to_list() == ["unknown"]


def test_no_history_indicator_tracks_the_null_pr_flag():
    df = attempts().with_columns(
        pl.when(pl.int_range(pl.len()) < 30)
        .then(None)
        .otherwise(pl.col("would_be_pr"))
        .alias("would_be_pr")
    )
    out = em.build_sample(df)
    assert out["no_history"].sum() == 30
    assert out["would_be_pr"].null_count() == 0


def test_interactions_are_the_product_of_their_parts():
    out = em.build_sample(attempts())
    assert (
        out["lb_ms_x_region"] == out["on_lb_milestone"] * out["lb_region"]
    ).all()


# --- placebo ----------------------------------------------------------------


def test_fake_milestone_sits_offset_from_a_real_one():
    # 227.5 kg is the loadable 500 lb; the placebo is 12.5 kg above it.
    values = [227.5, 227.5 + em.PLACEBO_OFFSET_KG]
    out = (
        pl.DataFrame({"w": values})
        .select(em.on_fake_milestone(pl.col("w"), "kg", 50.0).alias("f"))["f"]
        .to_list()
    )
    assert out == [False, True]


def test_fake_and_real_milestones_never_coincide():
    """The placebo only works if no fake milestone is also a real one.

    Checked over the whole loadable grid rather than on a fixture, because the
    question is about the milestone rules, not about any particular sample.
    """
    from features import on_lb_milestone_kg_meet

    grid = pl.DataFrame({"w": np.arange(40.0, 400.0, 2.5)})
    out = grid.select(
        on_lb_milestone_kg_meet(pl.col("w"), em.LB_STEP).alias("real"),
        em.on_fake_milestone(pl.col("w"), "kg", em.LB_STEP).alias("fake"),
    )
    assert out["real"].any() and out["fake"].any()
    assert not (out["real"] & out["fake"]).any()


# --- asymmetry running variable ---------------------------------------------


def test_running_variable_is_negative_below_a_milestone():
    df = pl.DataFrame({
        "prev_weight_kg": [225.0, 230.0],   # 227.5 is the loadable 500 lb
        "loading_unit": ["kg", "kg"],
    })
    out = em.add_running_variable(df)["running_kg"].to_list()
    assert out[0] < 0 < out[1]


def test_running_variable_uses_the_pound_grid_in_pound_meets():
    exact = 500 * KG_PER_LB
    df = pl.DataFrame({"prev_weight_kg": [exact], "loading_unit": ["lb"]})
    assert em.add_running_variable(df)["running_kg"][0] == pytest.approx(0.0, abs=1e-6)


# --- fitting ----------------------------------------------------------------


def test_fit_drops_terms_with_no_variance():
    sample = em.build_sample(attempts()).with_columns(
        pl.lit(1, dtype=pl.Int8).alias("lb_region")
    ).with_columns(
        (pl.col("on_lb_milestone") * pl.col("lb_region")).alias("lb_ms_x_region")
    )
    fit = em.fit_model(sample, "us only", em.BASE_TERMS, em.FE_BASE)
    assert "lb_region" in fit.dropped
    assert "on_lb_milestone" in fit.terms


def test_fit_reports_percentage_points():
    fit = em.fit_model(em.build_sample(attempts()), "t", em.BASE_TERMS, em.FE_BASE)
    row = fit.row("on_lb_milestone")
    assert row["pp"] == pytest.approx(row["coef"] * 100)
    assert row["ci_lo_pp"] < row["pp"] < row["ci_hi_pp"]


def test_fit_returns_none_on_an_empty_sample():
    empty = em.build_sample(attempts()).head(0)
    assert em.fit_model(empty, "none", em.BASE_TERMS, em.FE_BASE) is None


def test_lifter_fixed_effects_change_the_estimate_but_still_fit():
    sample = em.build_sample(attempts())
    base = em.fit_model(sample, "a", em.BASE_TERMS, em.FE_BASE)
    with_fe = em.fit_model(sample, "b", em.BASE_TERMS, em.FE_LIFTER)
    assert with_fe is not None
    assert np.isfinite(with_fe.beta).all()
    assert with_fe.n == base.n


def test_tidy_export_has_one_row_per_term():
    sample = em.build_sample(attempts())
    fits = [em.fit_model(sample, "a", em.BASE_TERMS, em.FE_BASE)]
    out = em.tidy(fits)
    assert out.height == len(fits[0].terms)
    assert {"pp", "se_pp", "ci_lo_pp", "ci_hi_pp"} <= set(out.columns)


def test_side_by_side_renders_every_model():
    sample = em.build_sample(attempts())
    fits = [
        em.fit_model(sample, "model one", em.BASE_TERMS, em.FE_BASE),
        em.fit_model(sample, "model two", em.BASE_TERMS, em.FE_LIFTER),
    ]
    text = em.side_by_side(fits, em.BASE_TERMS, "T")
    assert "model one" in text and "model two" in text
    assert "clustered by lifter" in text


# --- collinearity -----------------------------------------------------------


def test_independent_columns_drops_an_exact_duplicate():
    x = np.arange(50.0)
    X = np.column_stack([x, x * 1.0, np.sin(x)])
    assert em.independent_columns(X) == [0, 2]


def test_independent_columns_drops_an_exact_linear_combination():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=60), rng.normal(size=60)
    X = np.column_stack([a, b, 2 * a - 3 * b])
    assert em.independent_columns(X) == [0, 1]


def test_independent_columns_keeps_a_full_rank_design():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 4))
    assert em.independent_columns(X) == [0, 1, 2, 3]


def test_us_only_sample_drops_the_redundant_interaction():
    """lb_region is 1 throughout a US-only split, so lb_ms_x_region is just a
    copy of on_lb_milestone. Keeping both halves the estimate."""
    sample = em.build_sample(attempts()).with_columns(
        pl.lit(1, dtype=pl.Int8).alias("lb_region")
    ).with_columns(
        (pl.col("on_lb_milestone") * pl.col("lb_region")).alias("lb_ms_x_region")
    )
    fit = em.fit_model(sample, "us only", em.BASE_TERMS, em.FE_BASE)
    assert "on_lb_milestone" in fit.terms
    assert "lb_ms_x_region" not in fit.terms
    assert em.key_term(fit) == "on_lb_milestone"


def test_duplicate_regressor_does_not_halve_the_coefficient():
    sample = em.build_sample(attempts())
    base = em.fit_model(sample, "a", ["jump_pct", "on_lb_milestone"], em.FE_BASE)
    doubled = sample.with_columns(pl.col("on_lb_milestone").alias("copy_ms"))
    with_dup = em.fit_model(
        doubled, "b", ["jump_pct", "on_lb_milestone", "copy_ms"], em.FE_BASE
    )
    assert "copy_ms" in with_dup.dropped
    assert with_dup.row("on_lb_milestone")["pp"] == pytest.approx(
        base.row("on_lb_milestone")["pp"], rel=1e-6
    )


def test_key_term_picks_the_discovered_interaction():
    fit = em.Fit("x", ["jump_pct", "disc_ms_x_region"], np.zeros(2), np.ones(2), 1, 1, 0.5)
    assert em.key_term(fit) == "disc_ms_x_region"


def test_barely_varying_flag_is_dropped():
    """A flag that is on for a handful of rows is not identified, even though
    it technically varies. This produced a 52pp coefficient with a 37pp
    standard error in the pound-loaded column."""
    sample = em.build_sample(attempts()).with_columns(
        (pl.int_range(pl.len()) < 3).cast(pl.Int8).alias("on_kg_milestone")
    )
    fit = em.fit_model(sample, "sparse flag", em.BASE_TERMS, em.FE_BASE)
    assert "on_kg_milestone" in fit.dropped
    assert "jump_pct" in fit.terms


def test_well_supported_flag_is_kept():
    sample = em.build_sample(attempts())
    fit = em.fit_model(sample, "ok", em.BASE_TERMS, em.FE_BASE)
    assert "on_kg_milestone" in fit.terms


def test_logit_subsample_is_reproducible():
    """polars unique() does not promise an order, so sampling off it gave a
    different 25% each run and the logit AME changed sign between runs."""
    sample = em.build_sample(attempts(n=6000))
    a = em.run_logit(sample, seed=0)
    b = em.run_logit(sample, seed=0)
    assert a.n == b.n
    assert np.allclose(a.beta, b.beta)
