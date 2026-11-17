"""ML stage: predicting attempt success from pre-attempt information only.

Reads data/processed/attempts_discovered.parquet and writes
reports/ml_metrics.csv plus four figures.

Sample: attempts 2 and 3. Attempt 1 has no predecessor, so jump_pct,
prev_success and every milestone distance are null there by construction;
including it would mean imputing a third of the rows and letting the two models
see different information.

Features are the pre-attempt set declared in features.FEATURE_COLUMNS, minus
on_discovered_milestone. That flag was chosen on the discovery half, and
CLAUDE.md requires any outcome test on discovered milestones to use the test
half only; leaving it out keeps the full sample available and the rule intact.
The pre-specified milestone features stay in. A test asserts the feature list
never intersects features.POST_ATTEMPT_COLUMNS.

Evaluated three ways, because they answer different questions:

- GroupKFold by lifter on pre-2025 data: generalising to an unseen lifter.
- Temporal holdout, 2025-2026: generalising forward.
- Strict holdout, 2025+ *and* the lifter unseen in training. The temporal
  holdout splits by date, so a lifter sits on both sides of it, which is what
  CLAUDE.md's by-lifter rule exists to prevent. Reported separately rather than
  folded in.

Everything here is predictive. A model that says success falls with jump size
is describing what lifters who took big jumps did, not what would happen if a
given lifter took a bigger one. Nothing in this file is labelled causal, and
no jump size is presented as a recommendation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

import plotstyle as ps
from features import POST_ATTEMPT_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
IN_PARQUET = ROOT / "data" / "processed" / "attempts_discovered.parquet"
METRICS_PATH = ROOT / "reports" / "ml_metrics.csv"
FIG_DIR = ROOT / "reports" / "figures"

HOLDOUT_FROM_YEAR = 2025  # the latest two years in the data
N_SPLITS = 5
SHAP_SAMPLE = 50_000
SEED = 0

NUMERIC_FEATURES = [
    "weight_kg",
    "prev_weight_kg",
    "jump_kg",
    "jump_pct",
    "dist_to_next_lb_ms",
    "dist_to_next_lb100_ms",
    "dist_to_next_kg_ms",
    "dist_to_next_kg50_ms",
    "prior_meets",
    "prior_best_kg",
    "best_before_kg",
    "Age",
    "BodyweightKg",
]
BOOLEAN_FEATURES = [
    "prev_success",
    "on_lb_milestone",
    "on_lb100_milestone",
    "on_kg_milestone",
    "on_kg50_milestone",
    "would_be_pr",
    "no_history",
    "lb_region",
]
# Levels are pinned rather than read off the frame. to_dummies emits a column
# per level it *observes*, so a subset missing a level produced a narrower
# design than the model was fitted on, and the expected-weight plot died with
# "Feature shape mismatch, expected: 26, got 21".
CATEGORICAL_LEVELS = {
    "lift": ["bench", "deadlift", "squat"],
    "Sex": ["F", "M", "Mx"],
    "loading_unit": ["kg", "lb"],
    "attempt_no": ["2", "3"],
}
CATEGORICAL_FEATURES = list(CATEGORICAL_LEVELS)

# Milestone features SHAP is asked about by name.
SHAP_FEATURES = [
    "jump_pct",
    "on_lb_milestone",
    "on_kg_milestone",
    "dist_to_next_lb_ms",
]


def use_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def feature_names() -> list[str]:
    return [*NUMERIC_FEATURES, *BOOLEAN_FEATURES, *CATEGORICAL_FEATURES]


def build_sample(df: pl.DataFrame) -> pl.DataFrame:
    """Attempts 2 and 3, with every model variable present."""
    n0 = df.height
    sample = df.filter(pl.col("attempt_no") > 1)
    print(f"filter attempts 2 and 3: {n0:,} -> {sample.height:,}")

    n = sample.height
    sample = sample.filter(pl.col("lb_region").is_not_null()
                           & pl.col("on_lb_milestone").is_not_null())
    print(f"filter region and milestone defined: {n:,} -> {sample.height:,}")

    return sample.with_columns(
        pl.col("Date").dt.year().alias("year"),
        pl.col("would_be_pr").is_null().cast(pl.Int8).alias("no_history"),
        pl.col("would_be_pr").fill_null(False).alias("would_be_pr"),
        pl.col("best_before_kg").fill_null(0.0).alias("best_before_kg"),
        pl.col("prior_best_kg").fill_null(0.0).alias("prior_best_kg"),
    )


def design_matrix(df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Numeric design, one-hot on the few categoricals.

    Kept deliberately narrow: the high-cardinality meet columns would make a
    dense matrix the logistic regression cannot hold, and lb_region plus
    loading_unit already carry what MeetCountry would say here.
    """
    frames = [df.select([pl.col(c).cast(pl.Float64) for c in NUMERIC_FEATURES])]
    names = list(NUMERIC_FEATURES)

    frames.append(df.select([pl.col(c).cast(pl.Int8).cast(pl.Float64)
                             for c in BOOLEAN_FEATURES]))
    names += BOOLEAN_FEATURES

    # One column per level after the first, built against the pinned level list
    # so the width never depends on what this particular frame contains.
    dummy_exprs = []
    for column, levels in CATEGORICAL_LEVELS.items():
        for level in levels[1:]:
            name = f"{column}_{level}"
            dummy_exprs.append(
                (pl.col(column).cast(pl.Utf8) == level).cast(pl.Float64).alias(name)
            )
            names.append(name)
    frames.append(df.select(dummy_exprs))

    matrix = pl.concat(frames, how="horizontal").to_numpy().astype(np.float32)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), names


def fit_models(X: np.ndarray, y: np.ndarray, seed: int = SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    ).fit(X, y)

    boost = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        # Without these a deep forest memorises noise columns rather than
        # abstaining: on a small, mostly-noise sample it still ranks well
        # (AUC 0.63) but its probabilities go overconfident and log loss
        # falls behind predicting the base rate.
        min_child_weight=20,
        reg_lambda=2.0,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=seed,
    ).fit(X, y)
    return {"logistic regression": logit, "xgboost": boost}


def score(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    return {
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
    }


def baseline_scores(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Predict the training base rate for everyone. Anything that cannot beat
    this on log loss has learnt nothing."""
    p = np.full(len(y_test), float(y_train.mean()))
    from sklearn.metrics import brier_score_loss, log_loss

    return {
        "log_loss": float(log_loss(y_test, np.clip(p, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_test, p)),
        "auc": float("nan"),  # undefined for a constant prediction
    }


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> list[dict]:
    """GroupKFold on lifter: every fold scores lifters it never trained on."""
    from sklearn.model_selection import GroupKFold

    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=N_SPLITS).split(X, y, groups)):
        assert not set(groups[tr]) & set(groups[te]), "fold shares a lifter"
        for name, model in fit_models(X[tr], y[tr]).items():
            p = model.predict_proba(X[te])[:, 1]
            rows.append({"evaluation": "GroupKFold by lifter (pre-2025)",
                         "model": name, "fold": fold, "n": len(te), **score(y[te], p)})
        rows.append({"evaluation": "GroupKFold by lifter (pre-2025)",
                     "model": "baseline (base rate)", "fold": fold, "n": len(te),
                     **baseline_scores(y[tr], y[te])})
        print(f"  fold {fold}: {len(tr):,} train / {len(te):,} test")
    return rows


# --- figures ----------------------------------------------------------------


def plot_calibration(preds: dict[str, np.ndarray], y: np.ndarray) -> Path:
    """Does a predicted 70% actually come true 70% of the time?"""
    from sklearn.calibration import calibration_curve

    colours = {"logistic regression": ps.BAR, "xgboost": ps.FOUND}
    fig, axes = ps.new_figure(1, (8, 6.4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=ps.MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    for name, p in preds.items():
        true, pred = calibration_curve(y, p, n_bins=20, strategy="quantile")
        ax.plot(pred, true, color=colours.get(name, ps.PRESPEC_LB), linewidth=2.0,
                marker="o", markersize=5, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ps.style_axes(ax, xlabel="predicted probability of success",
                  ylabel="observed share successful",
                  title="Calibration on the 2025-2026 holdout (dashed = perfect)")
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_2)
    fig.tight_layout()
    return ps.save(fig, FIG_DIR / "ml_calibration.png")


def plot_shap(model, X: np.ndarray, names: list[str], seed: int = SEED) -> list[Path]:
    """SHAP on a sample: the full 4.9M rows is not tractable."""
    import matplotlib.pyplot as plt
    import shap

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(SHAP_SAMPLE, len(X)), replace=False)
    sub = X[idx]
    values = shap.TreeExplainer(model).shap_values(sub)

    plt.figure(figsize=(9, 7))
    shap.summary_plot(values, sub, feature_names=names, show=False, max_display=18)
    fig = plt.gcf()
    fig.patch.set_facecolor(ps.SURFACE)
    fig.suptitle("SHAP: what the model leans on (descriptive, not causal)",
                 x=0.01, ha="left", fontsize=12, color=ps.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    paths = [ps.save(fig, FIG_DIR / "ml_shap_summary.png")]

    available = [f for f in SHAP_FEATURES if f in names]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, feature in zip(axes.ravel(), available):
        j = names.index(feature)
        ax.set_facecolor(ps.SURFACE)
        ax.scatter(sub[:, j], values[:, j], s=3, alpha=0.18, color=ps.BAR, linewidths=0)
        ax.axhline(0, color=ps.AXIS, linewidth=1.0)
        ps.style_axes(ax, xlabel=feature, ylabel="SHAP value", title=feature)
    for ax in axes.ravel()[len(available):]:
        ax.set_axis_off()
    fig.suptitle("SHAP dependence (descriptive, not causal)",
                 x=0.01, ha="left", fontsize=12, color=ps.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    paths.append(ps.save(fig, FIG_DIR / "ml_shap_dependence.png"))
    return paths


def plot_expected_weight(model, sample: pl.DataFrame, names: list[str]) -> Path:
    """Predicted P(success) x attempted weight, across jump sizes.

    This is a conditional prediction, not a response curve. The model is being
    shown hypothetical rows that differ only in the jump, but lifters who take
    different jumps differ in ways the features do not capture, so the shape
    describes the data rather than what a given lifter would face. The peak of
    the curve is not a recommendation and is not presented as one.
    """
    thirds = sample.filter(pl.col("attempt_no") == 3)
    profiles = []
    for q, label in ((0.15, "lighter lifter"), (0.5, "median lifter"),
                     (0.85, "heavier lifter")):
        # The row nearest the quantile, not an arbitrary row below it: the
        # first version took .tail(1) of everything under the cut and produced
        # a "lighter" and a "median" profile 2 kg apart.
        target = thirds["prev_weight_kg"].quantile(q)
        row = (
            thirds.with_columns(
                (pl.col("prev_weight_kg") - target).abs().alias("_gap")
            )
            .sort("_gap")
            .head(1)
            .drop("_gap")
        )
        if row.height:
            profiles.append((label, row))

    jumps = np.arange(0.0, 16.1, 0.5)
    colours = [ps.BAR, ps.FOUND, ps.PRESPEC_KG]

    fig, axes = ps.new_figure(1, (10, 6))
    ax = axes[0]
    for (label, row), colour in zip(profiles, colours):
        prev = float(row["prev_weight_kg"][0])
        grid = pl.concat([
            row.with_columns(
                pl.lit(prev * (1 + j / 100)).alias("weight_kg"),
                pl.lit(prev * j / 100).alias("jump_kg"),
                pl.lit(j).alias("jump_pct"),
            )
            for j in jumps
        ])
        X, _ = design_matrix(grid)
        p = model.predict_proba(X)[:, 1]
        weights = prev * (1 + jumps / 100)
        ax.plot(jumps, p * weights, color=colour, linewidth=2.0,
                label=f"{label} (second attempt {prev:.0f} kg)")

    ps.style_axes(
        ax,
        xlabel="jump from the second attempt (%)",
        ylabel="predicted P(success) x attempted weight (kg)",
        title="Descriptive: model prediction across jump sizes, not a causal response",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_2)
    fig.tight_layout()
    return ps.save(fig, FIG_DIR / "ml_expected_weight.png")


# --- driver -----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=IN_PARQUET)
    parser.add_argument("--skip-cv", action="store_true",
                        help="skip the GroupKFold sweep, which is the slow part")
    args = parser.parse_args()
    use_utf8_stdout()

    if not args.src.exists():
        print(f"{args.src} not found - run the discover stage first.", file=sys.stderr)
        return 1

    sample = build_sample(pl.read_parquet(args.src))
    print(f"\nsample: {sample.height:,} attempts, "
          f"{sample['lifter_id'].n_unique():,} lifters, "
          f"mean success {sample['success'].mean():.4f}")

    is_holdout = (sample["year"] >= HOLDOUT_FROM_YEAR).to_numpy()
    train, test = sample.filter(~is_holdout), sample.filter(is_holdout)
    print(f"train (pre-{HOLDOUT_FROM_YEAR}): {train.height:,}   "
          f"holdout ({HOLDOUT_FROM_YEAR}+): {test.height:,}")

    X_tr, names = design_matrix(train)
    X_te, _ = design_matrix(test)
    y_tr = train["success"].cast(pl.Int8).to_numpy()
    y_te = test["success"].cast(pl.Int8).to_numpy()
    print(f"design: {X_tr.shape[1]} columns")

    rows = []
    if not args.skip_cv:
        print("\nGroupKFold by lifter")
        rows += cross_validate(X_tr, y_tr, train["lifter_id"].to_numpy())

    print("\ntemporal holdout")
    models = fit_models(X_tr, y_tr)
    preds = {name: m.predict_proba(X_te)[:, 1] for name, m in models.items()}

    seen = set(train["lifter_id"].unique().to_list())
    unseen = ~np.fromiter((lid in seen for lid in test["lifter_id"].to_list()),
                          dtype=bool, count=test.height)
    print(f"  strict holdout (unseen lifters): {int(unseen.sum()):,} of {test.height:,}")

    for name, p in preds.items():
        rows.append({"evaluation": f"temporal holdout ({HOLDOUT_FROM_YEAR}+)",
                     "model": name, "fold": -1, "n": len(y_te), **score(y_te, p)})
        if unseen.any():
            rows.append({"evaluation": "strict holdout (later AND unseen lifter)",
                         "model": name, "fold": -1, "n": int(unseen.sum()),
                         **score(y_te[unseen], p[unseen])})
    rows.append({"evaluation": f"temporal holdout ({HOLDOUT_FROM_YEAR}+)",
                 "model": "baseline (base rate)", "fold": -1, "n": len(y_te),
                 **baseline_scores(y_tr, y_te)})

    metrics = pl.DataFrame(rows)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_csv(METRICS_PATH)
    print(f"\nwrote {METRICS_PATH}")

    summary = (
        metrics.group_by("evaluation", "model")
        .agg(pl.col("log_loss").mean().round(5),
             pl.col("brier").mean().round(5),
             pl.col("auc").mean().round(4),
             pl.col("n").sum())
        .sort("evaluation", "model")
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=140):
        print(summary)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nwrote {plot_calibration(preds, y_te).name}")
    for path in plot_shap(models["xgboost"], X_te, names):
        print(f"wrote {path.name}")
    print(f"wrote {plot_expected_weight(models['xgboost'], test, names).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
