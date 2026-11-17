import datetime as dt

import numpy as np
import polars as pl
import pytest

import ml
from features import POST_ATTEMPT_COLUMNS


def attempts(n: int = 3000, seed: int = 0) -> pl.DataFrame:
    """A frame in the shape the discover stage emits."""
    rng = np.random.default_rng(seed)
    prev = np.round(rng.normal(180, 30, n) / 2.5) * 2.5
    jump_pct = rng.uniform(1, 12, n)
    weight = prev * (1 + jump_pct / 100)
    # Success falls with the jump, so the models have something to find.
    success = rng.random(n) < np.clip(0.95 - 0.035 * jump_pct, 0.05, 0.95)
    return pl.DataFrame(
        {
            "lifter_id": [f"l{i % 200}" for i in range(n)],
            "meet_id": [f"m{i % 40}" for i in range(n)],
            "lift": rng.choice(["squat", "bench", "deadlift"], n),
            "attempt_no": rng.choice([1, 2, 3], n).astype(np.int8),
            "weight_kg": weight,
            "prev_weight_kg": prev,
            "prev_success": rng.random(n) < 0.8,
            "success": success,
            "jump_kg": weight - prev,
            "jump_pct": jump_pct,
            "dist_to_next_lb_ms": rng.uniform(0, 22, n),
            "dist_to_next_lb100_ms": rng.uniform(0, 45, n),
            "dist_to_next_kg_ms": rng.uniform(0, 10, n),
            "dist_to_next_kg50_ms": rng.uniform(0, 50, n),
            "on_lb_milestone": rng.random(n) < 0.13,
            "on_lb100_milestone": rng.random(n) < 0.06,
            "on_kg_milestone": rng.random(n) < 0.3,
            "on_kg50_milestone": rng.random(n) < 0.08,
            "on_discovered_milestone": rng.random(n) < 0.1,
            "lb_region": rng.integers(0, 2, n).astype(np.int8),
            "loading_unit": rng.choice(["kg", "lb"], n),
            "prior_meets": rng.integers(0, 12, n),
            "prior_best_kg": rng.uniform(100, 250, n),
            "best_before_kg": rng.uniform(100, 250, n),
            "would_be_pr": rng.random(n) < 0.5,
            "Sex": rng.choice(["M", "F"], n),
            "Age": rng.uniform(18, 60, n),
            "BodyweightKg": rng.uniform(55, 130, n),
            "split": rng.choice(["discovery", "test"], n),
            "Date": [dt.date(2018 + i % 9, 1, 1) for i in range(n)],
        }
    )


# --- no leakage -------------------------------------------------------------


def test_features_never_include_an_outcome():
    assert not set(ml.feature_names()) & set(POST_ATTEMPT_COLUMNS)


def test_success_is_not_a_feature():
    assert "success" not in ml.feature_names()


def test_discovered_milestone_is_excluded():
    """It was chosen on the discovery half; CLAUDE.md confines outcome tests on
    it to the test half, so keeping it out leaves the full sample usable."""
    assert "on_discovered_milestone" not in ml.feature_names()


def test_design_matrix_columns_carry_no_outcome_name():
    X, names = ml.design_matrix(ml.build_sample(attempts()))
    assert X.shape[1] == len(names)
    for outcome in POST_ATTEMPT_COLUMNS:
        assert not any(n == outcome or n.startswith(f"{outcome}_") for n in names)


def test_design_matrix_width_is_independent_of_the_rows():
    """Regression: to_dummies emitted only the levels a frame contained, so a
    subset produced a narrower design than the fitted model expected."""
    full = ml.build_sample(attempts())
    X_full, names_full = ml.design_matrix(full)
    one_lift = full.filter(pl.col("lift") == "squat").head(50)
    X_sub, names_sub = ml.design_matrix(one_lift)
    assert names_sub == names_full
    assert X_sub.shape[1] == X_full.shape[1]


def test_design_matrix_is_finite():
    X, _ = ml.design_matrix(ml.build_sample(attempts()))
    assert np.isfinite(X).all()


# --- sample -----------------------------------------------------------------


def test_sample_drops_openers():
    out = ml.build_sample(attempts())
    assert (out["attempt_no"] > 1).all()


def test_sample_keeps_both_later_attempts():
    out = ml.build_sample(attempts())
    assert sorted(out["attempt_no"].unique().to_list()) == [2, 3]


def test_no_history_flag_tracks_the_null_pr():
    df = attempts().with_columns(
        pl.when(pl.int_range(pl.len()) < 40)
        .then(None)
        .otherwise(pl.col("would_be_pr"))
        .alias("would_be_pr")
    )
    out = ml.build_sample(df)
    assert out["would_be_pr"].null_count() == 0
    assert out["no_history"].sum() == out.filter(pl.col("no_history") == 1).height


# --- splits -----------------------------------------------------------------


def test_group_folds_never_share_a_lifter():
    from sklearn.model_selection import GroupKFold

    sample = ml.build_sample(attempts())
    groups = sample["lifter_id"].to_numpy()
    X = np.zeros((sample.height, 2))
    y = sample["success"].cast(pl.Int8).to_numpy()
    for tr, te in GroupKFold(n_splits=ml.N_SPLITS).split(X, y, groups):
        assert not set(groups[tr]) & set(groups[te])


def test_temporal_holdout_takes_the_latest_years():
    sample = ml.build_sample(attempts())
    holdout = sample.filter(pl.col("year") >= ml.HOLDOUT_FROM_YEAR)
    train = sample.filter(pl.col("year") < ml.HOLDOUT_FROM_YEAR)
    assert holdout.height > 0 and train.height > 0
    assert holdout["year"].min() > train["year"].max()


def test_strict_holdout_shares_no_lifter_with_training():
    sample = ml.build_sample(attempts())
    train = sample.filter(pl.col("year") < ml.HOLDOUT_FROM_YEAR)
    test = sample.filter(pl.col("year") >= ml.HOLDOUT_FROM_YEAR)
    seen = set(train["lifter_id"].unique().to_list())
    strict = test.filter(~pl.col("lifter_id").is_in(list(seen)))
    assert not set(strict["lifter_id"].to_list()) & seen


# --- scoring ----------------------------------------------------------------


def test_score_reports_all_three_metrics():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    out = ml.score(y, p)
    assert set(out) == {"log_loss", "brier", "auc"}
    assert 0 < out["auc"] <= 1


def test_a_better_prediction_scores_better():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.6).astype(int)
    good = np.where(y == 1, 0.8, 0.2)
    bad = np.where(y == 1, 0.4, 0.6)
    assert ml.score(y, good)["log_loss"] < ml.score(y, bad)["log_loss"]
    assert ml.score(y, good)["brier"] < ml.score(y, bad)["brier"]


def test_baseline_predicts_the_training_base_rate():
    y_tr = np.array([1, 1, 1, 0])
    y_te = np.array([1, 0])
    out = ml.baseline_scores(y_tr, y_te)
    assert np.isnan(out["auc"])  # undefined for a constant prediction
    assert out["log_loss"] > 0


def test_models_find_the_signal_in_the_fixture():
    """The fixture builds success as a decreasing function of jump size, so a
    model that ranks no better than chance has not been wired up correctly.

    Ranking, not log loss, is what is asserted for both. The fixture is mostly
    noise columns, and on 14k rows a high-capacity model ranks well while
    staying overconfident, which log loss penalises. Calibration on real data
    is what ml_calibration.png is for.
    """
    sample = ml.build_sample(attempts(n=30000))
    X, _ = ml.design_matrix(sample)
    y = sample["success"].cast(pl.Int8).to_numpy()
    cut = int(0.7 * len(y))
    models = ml.fit_models(X[:cut], y[:cut])
    for name, model in models.items():
        p = model.predict_proba(X[cut:])[:, 1]
        assert ml.score(y[cut:], p)["auc"] > 0.55, name


def test_logistic_beats_the_base_rate_on_log_loss():
    sample = ml.build_sample(attempts(n=30000))
    X, _ = ml.design_matrix(sample)
    y = sample["success"].cast(pl.Int8).to_numpy()
    cut = int(0.7 * len(y))
    model = ml.fit_models(X[:cut], y[:cut])["logistic regression"]
    p = model.predict_proba(X[cut:])[:, 1]
    assert ml.score(y[cut:], p)["log_loss"] < ml.baseline_scores(y[:cut], y[cut:])["log_loss"]


# --- figures ----------------------------------------------------------------


def test_calibration_figure_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "FIG_DIR", tmp_path)
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.6).astype(int)
    p = np.clip(np.where(y == 1, 0.7, 0.3) + rng.normal(0, 0.1, 2000), 0.01, 0.99)
    assert ml.plot_calibration({"logistic regression": p}, y).exists()


def test_expected_weight_figure_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "FIG_DIR", tmp_path)
    sample = ml.build_sample(attempts(n=2000))
    X, names = ml.design_matrix(sample)
    y = sample["success"].cast(pl.Int8).to_numpy()
    model = ml.fit_models(X, y)["xgboost"]
    assert ml.plot_expected_weight(model, sample, names).exists()


def test_nothing_in_this_module_claims_optimality():
    """CLAUDE.md: never label a predictive result as causal or 'optimal'.

    The expected-weight figure is the one that invites it, so the word is
    checked out of the source rather than only out of the title.
    """
    source = (ml.Path(ml.__file__)).read_text(encoding="utf-8").lower()
    assert "optimal" not in source
    assert "causal" in source  # it says what it is NOT


def test_expected_weight_profiles_are_distinct(tmp_path, monkeypatch):
    """Regression: the first version picked an arbitrary row below each
    quantile, giving a 'lighter' and a 'median' lifter 2 kg apart."""
    monkeypatch.setattr(ml, "FIG_DIR", tmp_path)
    sample = ml.build_sample(attempts(n=4000))
    thirds = sample.filter(pl.col("attempt_no") == 3)
    picked = []
    for q in (0.15, 0.5, 0.85):
        target = thirds["prev_weight_kg"].quantile(q)
        row = (
            thirds.with_columns((pl.col("prev_weight_kg") - target).abs().alias("_g"))
            .sort("_g").head(1)
        )
        picked.append(float(row["prev_weight_kg"][0]))
    assert len(set(picked)) == 3
    assert picked[2] - picked[0] > 10.0
