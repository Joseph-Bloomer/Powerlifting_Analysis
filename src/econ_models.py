"""Econ models: linear probability models of third-attempt success.

Reads data/processed/attempts_discovered.parquet and writes
reports/econ_models.csv (tidy) and reports/econ_models.txt (side by side).

Sample: third attempts that followed a *successful* second. A lifter whose
second attempt failed is choosing a third under a different problem, and
mixing the two would blur what the milestone coefficient means.

The headline estimate is on_lb_milestone x lb_region. A pound milestone is a
round number only to someone thinking in pounds, so if reference dependence is
what drives attempt choice the pound milestone should bite in US meets and not
in kilo countries. The same weight, two meanings, is the identification.

Effect sizes are reported in percentage points of success probability, because
the outcome is a 0/1 and an LPM coefficient times 100 is already pp. A
significant coefficient worth 0.3pp and one worth 4pp are different findings
and the table has to show which is which (CLAUDE.md).

Fixed effects are absorbed by alternating projections rather than entered as
dummies: 296k lifters would be a design matrix no machine here will hold.
Standard errors cluster on lifter throughout. The clustered sandwich and the
absorption are both checked against statsmodels in the tests.

Nothing here is described as loss aversion. The asymmetry test is reported as
what it is, and whether it supports that reading is a judgement for the write-up.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from constants import KG_PER_LB
from features import (
    LB_COARSE_STEP,
    LB_STEP,
    on_lb_milestone_kg_meet,
    on_lb_milestone_lb_meet,
)

ROOT = Path(__file__).resolve().parents[1]
IN_PARQUET = ROOT / "data" / "processed" / "attempts_discovered.parquet"
CSV_PATH = ROOT / "reports" / "econ_models.csv"
TXT_PATH = ROOT / "reports" / "econ_models.txt"

# CLAUDE.md: the placebo milestone sits 12.5 kg off the real one.
PLACEBO_OFFSET_KG = 12.5

# Asymmetry test: half-width of the window around a milestone, in kg.
ASYMMETRY_WINDOW_KG = 6.0

# A binary regressor needs at least this many observations in its smaller cell
# before its coefficient means anything.
MIN_BINARY_SUPPORT = 50

FE_BASE = ["Sex", "age_class", "weight_class", "year"]
FE_LIFTER = ["lifter_id", *FE_BASE]  # largest first, see absorb()

# Regressors in reporting order. Terms with no variance in a subsample are
# dropped automatically and the drop is logged.
BASE_TERMS = [
    "jump_pct",
    "on_lb_milestone",
    "on_kg_milestone",
    "lb_region",
    "lb_ms_x_region",
    "kg_ms_x_region",
    "prior_meets",
    "would_be_pr",
    "no_history",
]


def use_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


# --- estimation -------------------------------------------------------------


def codes_of(frame: pl.DataFrame, column: str) -> tuple[np.ndarray, int]:
    """Integer codes for a grouping column, nulls given their own level."""
    filled = frame[column].cast(pl.Utf8).fill_null("__null__")
    cat = filled.cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
    return cat, int(cat.max()) + 1 if len(cat) else 0


def absorb(
    values: np.ndarray,
    group_codes: list[tuple[np.ndarray, int]],
    tol: float = 1e-8,
    max_iter: int = 100,
) -> np.ndarray:
    """Within-transform every column on all fixed-effect groups at once.

    Alternating projections: demean on each group set in turn and repeat until
    the means stop moving. With one group set this converges immediately; with
    several it is the standard way to absorb many fixed effects without ever
    building their dummy matrix.

    The working array is Fortran-ordered so that each column is contiguous.
    Row-major, every bincount strides across the row and the 296k-lifter pass
    spends its time missing cache; this was the difference between minutes and
    tens of minutes on the real sample.

    group_codes should be ordered with the largest group set first: it is the
    one convergence is judged on, and it is exactly the one the final pass of
    each sweep does not leave at zero.
    """
    if not group_codes:
        return values - values.mean(axis=0)

    out = np.asfortranarray(values, dtype=float)
    scale = max(float(np.abs(out).max()), 1.0)
    prepared = [
        (codes, n, np.maximum(np.bincount(codes, minlength=n), 1).astype(float))
        for codes, n in group_codes
    ]

    for _ in range(max_iter):
        for codes, n_levels, counts in prepared:
            for j in range(out.shape[1]):
                col = out[:, j]
                col -= (np.bincount(codes, weights=col, minlength=n_levels) / counts)[codes]

        codes, n_levels, counts = prepared[0]
        worst = max(
            float(
                np.abs(
                    np.bincount(codes, weights=out[:, j], minlength=n_levels) / counts
                ).max()
            )
            for j in range(out.shape[1])
        )
        if worst < tol * scale:
            break
    return out


@dataclass
class Fit:
    """One estimated specification."""

    name: str
    terms: list[str]
    beta: np.ndarray
    se: np.ndarray
    n: int
    n_clusters: int
    outcome_mean: float
    note: str = ""
    dropped: list[str] = field(default_factory=list)

    def t(self) -> np.ndarray:
        return self.beta / self.se

    def p(self) -> np.ndarray:
        from scipy.special import erfc

        return erfc(np.abs(self.t()) / np.sqrt(2.0))

    def row(self, term: str) -> dict | None:
        if term not in self.terms:
            return None
        i = self.terms.index(term)
        return {
            "term": term,
            "coef": self.beta[i],
            "pp": self.beta[i] * 100,
            "se_pp": self.se[i] * 100,
            "t": self.t()[i],
            "p": self.p()[i],
            "ci_lo_pp": (self.beta[i] - 1.96 * self.se[i]) * 100,
            "ci_hi_pp": (self.beta[i] + 1.96 * self.se[i]) * 100,
        }


def ols_cluster(
    y: np.ndarray,
    X: np.ndarray,
    cluster_codes: np.ndarray,
    n_clusters: int,
    n_absorbed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """OLS with cluster-robust standard errors.

    The meat of the sandwich sums X'e within each lifter, so repeated attempts
    by the same lifter are allowed to be correlated however they like. The
    finite-sample factor is the usual G/(G-1) x (N-1)/(N-K), with K counting the
    absorbed fixed effects as estimated parameters.
    """
    xtx = X.T @ X
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta

    k = X.shape[1]
    u = np.empty((n_clusters, k))
    for j in range(k):
        u[:, j] = np.bincount(
            cluster_codes, weights=X[:, j] * resid, minlength=n_clusters
        )

    n = len(y)
    dof = n - k - n_absorbed
    correction = (n_clusters / max(n_clusters - 1, 1)) * ((n - 1) / max(dof, 1))
    cov = xtx_inv @ (u.T @ u) @ xtx_inv * correction
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None))


def independent_columns(X: np.ndarray, tol: float = 1e-9) -> list[int]:
    """Indices of columns that add rank, keeping the earliest of a dependent set.

    The US-only subsamples need this. With lb_region constant at 1,
    lb_ms_x_region is *identical* to on_lb_milestone, and a pseudo-inverse
    happily splits one effect across the two columns and halves it. Terms are
    ordered so the plain milestone flag survives and the now-redundant
    interaction is the one dropped.

    Worked on the Gram matrix, which is only as big as the regressor count.
    """
    gram = X.T @ X
    d = np.sqrt(np.diag(gram))
    d[d == 0] = 1.0
    normalised = gram / np.outer(d, d)

    keep: list[int] = []
    for j in range(X.shape[1]):
        trial = [*keep, j]
        block = normalised[np.ix_(trial, trial)]
        if np.linalg.matrix_rank(block, tol=tol) == len(trial):
            keep.append(j)
    return keep


def key_term(fit: "Fit") -> str:
    """The coefficient a specification is really about."""
    for term in (
        "lb_ms_x_region", "disc_ms_x_region", "fake_ms_x_region",
        "on_lb_milestone", "on_discovered_milestone", "on_fake_milestone",
    ):
        if term in fit.terms:
            return term
    return fit.terms[0]


def fit_model(
    df: pl.DataFrame,
    name: str,
    terms: list[str],
    fe_cols: list[str],
    outcome: str = "success",
    note: str = "",
) -> Fit | None:
    """Absorb the fixed effects, then OLS with lifter-clustered errors."""
    if df.height == 0:
        return None

    # A term with no variance in this subsample is not identified. The US-only
    # splits hit this by design: lb_region is 1 everywhere, so its interactions
    # vanish and the milestone main effect carries the estimate instead.
    # A flag that is almost never on is not identified either, even though it
    # technically varies. Pound-loaded meets contain a couple of dozen weights
    # that happen to be an exact 10 kg multiple, and estimating on_kg_milestone
    # off those produced a 52pp coefficient with a 37pp standard error.
    usable, dropped = [], []
    for t in terms:
        col = df[t].cast(pl.Float64)
        if col.n_unique() <= 1:
            dropped.append(t)
            continue
        if set(col.unique().drop_nulls().to_list()) <= {0.0, 1.0}:
            on = int(col.sum())
            if min(on, df.height - on) < MIN_BINARY_SUPPORT:
                dropped.append(t)
                continue
        usable.append(t)
    if not usable:
        return None

    y = df[outcome].cast(pl.Float64).to_numpy().reshape(-1, 1)
    X = df.select([pl.col(t).cast(pl.Float64) for t in usable]).to_numpy()

    groups = [codes_of(df, fe) for fe in fe_cols]
    n_absorbed = sum(n for _, n in groups) - (len(groups) - 1 if groups else 0)

    stacked = absorb(np.hstack([y, X]), groups)
    yd, Xd = stacked[:, 0], stacked[:, 1:]

    # Collinearity is judged after absorption: a term can be fine in the raw
    # data and redundant once the fixed effects are taken out.
    keep = independent_columns(Xd)
    if len(keep) < len(usable):
        dropped += [usable[j] for j in range(len(usable)) if j not in keep]
        usable = [usable[j] for j in keep]
        Xd = Xd[:, keep]

    cluster_codes, n_clusters = codes_of(df, "lifter_id")
    beta, se = ols_cluster(yd, Xd, cluster_codes, n_clusters, n_absorbed)

    return Fit(
        name=name,
        terms=usable,
        beta=beta,
        se=se,
        n=df.height,
        n_clusters=n_clusters,
        outcome_mean=float(df[outcome].cast(pl.Float64).mean()),
        note=note,
        dropped=dropped,
    )


# --- sample and variables ---------------------------------------------------


def on_fake_milestone(
    weight_kg: pl.Expr, unit: str, step: float, offset_kg: float | None = None
) -> pl.Expr:
    """A placebo milestone sits `offset_kg` above a real one.

    Shifting the weight back by the offset and asking whether *that* is a real
    milestone defines the fake grid without duplicating the milestone rules.

    The offset defaults to PLACEBO_OFFSET_KG, the single placebo CLAUDE.md
    names. diagnostics.py sweeps it to build a family of placebos; passing it
    does not change the headline placebo.
    """
    shift = PLACEBO_OFFSET_KG if offset_kg is None else offset_kg
    helper = on_lb_milestone_lb_meet if unit == "lb" else on_lb_milestone_kg_meet
    return helper(weight_kg - shift, step)


def build_sample(df: pl.DataFrame) -> pl.DataFrame:
    """Third attempts after a successful second, with the model variables."""
    n0 = df.height
    sample = df.filter((pl.col("attempt_no") == 3) & pl.col("prev_success"))
    print(f"filter third attempt after a successful second: {n0:,} -> {sample.height:,}")

    n = sample.height
    sample = sample.filter(
        pl.col("lb_region").is_not_null() & pl.col("on_lb_milestone").is_not_null()
    )
    print(f"filter region and milestone defined: {n:,} -> {sample.height:,}")

    placebo = pl.lit(False)
    for unit in ("kg", "lb"):
        placebo = (
            pl.when(pl.col("loading_unit") == unit)
            .then(on_fake_milestone(pl.col("weight_kg"), unit, LB_STEP))
            .otherwise(placebo)
        )

    return sample.with_columns(
        pl.col("Date").dt.year().alias("year"),
        # Explicit unknown levels: a tenth of lifters have no recorded age and
        # dropping them would bias the sample towards federations that record it.
        pl.when(pl.col("Age").is_null())
        .then(pl.lit("unknown"))
        .otherwise((pl.col("Age") / 5).floor().cast(pl.Int32).cast(pl.Utf8))
        .alias("age_class"),
        pl.col("WeightClassKg").fill_null("unknown").alias("weight_class"),
        pl.col("would_be_pr").is_null().cast(pl.Int8).alias("no_history"),
        pl.col("would_be_pr").fill_null(False).cast(pl.Int8).alias("would_be_pr"),
        pl.col("on_lb_milestone").cast(pl.Int8).alias("on_lb_milestone"),
        pl.col("on_kg_milestone").cast(pl.Int8).alias("on_kg_milestone"),
        placebo.cast(pl.Int8).alias("on_fake_milestone"),
    ).with_columns(
        (pl.col("on_lb_milestone") * pl.col("lb_region")).alias("lb_ms_x_region"),
        (pl.col("on_kg_milestone") * pl.col("lb_region")).alias("kg_ms_x_region"),
        (pl.col("on_fake_milestone") * pl.col("lb_region")).alias("fake_ms_x_region"),
        pl.col("on_discovered_milestone")
        .cast(pl.Int8)
        .alias("on_discovered_milestone"),
    ).with_columns(
        (pl.col("on_discovered_milestone") * pl.col("lb_region")).alias(
            "disc_ms_x_region"
        )
    )


# --- asymmetry --------------------------------------------------------------


def add_running_variable(df: pl.DataFrame) -> pl.DataFrame:
    """Signed kg from the previous attempt to the nearest pound milestone.

    Negative means the lifter sat below the milestone and reaching it is in
    front of them; positive means they had already passed it.
    """
    expr = pl.lit(None, dtype=pl.Float64)
    for unit in ("kg", "lb"):
        if unit == "lb":
            prev_lb = pl.col("prev_weight_kg") / KG_PER_LB
            nearest = (prev_lb / LB_STEP).round() * LB_STEP * KG_PER_LB
        else:
            prev_lb = pl.col("prev_weight_kg") / KG_PER_LB
            nearest_lb = (prev_lb / LB_STEP).round() * LB_STEP
            # In a kilo meet the reachable weight is the loadable one.
            nearest = (nearest_lb * KG_PER_LB / 2.5).ceil() * 2.5
        expr = (
            pl.when(pl.col("loading_unit") == unit)
            .then(pl.col("prev_weight_kg") - nearest)
            .otherwise(expr)
        )
    return df.with_columns(expr.alias("running_kg"))


def asymmetry_test(df: pl.DataFrame) -> list[Fit]:
    """Is the jump pulled upward below a milestone but not just past one?

    Within a window either side of a pound milestone, the jump is regressed on
    the signed distance with its own slope on each side and a step at zero. If
    the pull is about reaching a round number, the slope should differ below and
    above, and the difference should be a US phenomenon.
    """
    window = add_running_variable(df).filter(
        pl.col("running_kg").is_not_null()
        & (pl.col("running_kg").abs() <= ASYMMETRY_WINDOW_KG)
    )
    window = window.with_columns(
        (pl.col("running_kg") > 0).cast(pl.Int8).alias("above"),
    ).with_columns(
        (pl.col("running_kg") * pl.col("above")).alias("running_x_above"),
    )

    terms = ["running_kg", "above", "running_x_above"]
    fits = []
    for label, sub in (
        ("lb region", window.filter(pl.col("lb_region") == 1)),
        ("kg region", window.filter(pl.col("lb_region") == 0)),
    ):
        fit = fit_model(
            sub,
            f"asymmetry: {label}",
            terms,
            FE_BASE,
            outcome="jump_kg",
            note=f"jump_kg on signed kg to the nearest 50 lb milestone, "
                 f"|window| <= {ASYMMETRY_WINDOW_KG:g} kg",
        )
        if fit:
            fits.append(fit)
    return fits


# --- specifications ---------------------------------------------------------


def run_specifications(sample: pl.DataFrame) -> list[Fit]:
    fits: list[Fit] = []

    def add(fit: Fit | None) -> None:
        if fit is None:
            return
        fits.append(fit)
        key = key_term(fit)
        row = fit.row(key)
        print(f"  {fit.name:44s} n={fit.n:>9,}  {key} = "
              f"{row['pp']:+6.2f}pp (se {row['se_pp']:.2f})")

    print("\nheadline")
    add(fit_model(sample, "headline (base FE)", BASE_TERMS, FE_BASE))
    add(fit_model(sample, "headline + lifter FE", BASE_TERMS, FE_LIFTER))

    print("\nsecondary 1: US meets by loading unit")
    for group, label in (("lb", "US pound-loaded"), ("kg", "US kilo-loaded")):
        sub = sample.filter(
            (pl.col("lb_region") == 1) & (pl.col("loading_unit") == group)
        )
        add(fit_model(sub, f"{label} only", BASE_TERMS, FE_BASE,
                      note="lb_region is constant here, so its interactions drop "
                           "and the milestone main effect carries the estimate"))

    print("\nsecondary 2: discovered milestones, test half only")
    disc_terms = [
        "jump_pct", "on_discovered_milestone", "lb_region", "disc_ms_x_region",
        "prior_meets", "would_be_pr", "no_history",
    ]
    test_half = sample.filter(
        (pl.col("split") == "test") & pl.col("on_discovered_milestone").is_not_null()
    )
    add(fit_model(test_half, "discovered milestones (test half)", disc_terms, FE_BASE,
                  note="milestones found in the discovery half; this is the only "
                       "outcome test on them, run on meets discovery never saw"))

    print("\nrobustness")
    for lift in ("squat", "bench", "deadlift"):
        add(fit_model(sample.filter(pl.col("lift") == lift),
                      f"lift = {lift}", BASE_TERMS, FE_BASE))
    add(fit_model(sample.filter(pl.col("Tested") == "Yes"),
                  "tested only", BASE_TERMS, FE_BASE))

    placebo_terms = [
        "jump_pct", "on_fake_milestone", "on_kg_milestone", "lb_region",
        "fake_ms_x_region", "kg_ms_x_region", "prior_meets", "would_be_pr",
        "no_history",
    ]
    add(fit_model(sample, "placebo (+12.5 kg fake milestones)",
                  placebo_terms, FE_BASE,
                  note="should show no effect if the design is sound"))

    return fits


def run_logit(sample: pl.DataFrame, seed: int = 0) -> Fit | None:
    """Logit robustness check, as an average marginal effect in pp.

    On a random sample of lifters: logit cannot absorb fixed effects, so the
    design matrix has to carry them as dummies and the full sample will not fit
    in memory. Clustered by lifter as everywhere else.
    """
    import statsmodels.api as sm

    # sort() before sample(): polars unique() does not promise an order, so
    # without it the "random 25%" differs between runs and this check moved
    # from -1.11pp to +0.06pp on identical inputs.
    keep = sample["lifter_id"].unique().sort().sample(fraction=0.25, seed=seed)
    sub = sample.join(keep.to_frame(), on="lifter_id", how="semi")
    if sub.height == 0:
        return None

    terms = [t for t in BASE_TERMS if sub[t].cast(pl.Float64).n_unique() > 1]
    X = sub.select([pl.col(t).cast(pl.Float64) for t in terms]).to_numpy()
    # drop_first, or the level dummies are collinear with the intercept.
    dummies = (
        sub.select(pl.col("Sex").cast(pl.Utf8), pl.col("year").cast(pl.Utf8))
        .to_dummies(drop_first=True)
        .to_numpy()
        .astype(float)
    )
    design = np.hstack([np.ones((sub.height, 1)), X, dummies])
    y = sub["success"].cast(pl.Float64).to_numpy()
    codes, _ = codes_of(sub, "lifter_id")

    # Discrete models take cov_type in fit(); they have no
    # get_robustcov_results. lbfgs rather than the default Newton step, which
    # does not converge here with the year dummies in the design.
    res = sm.Logit(y, design).fit(
        disp=0,
        method="lbfgs",
        maxiter=500,
        cov_type="cluster",
        cov_kwds={"groups": codes},
    )
    converged = bool(res.mle_retvals.get("converged", True))

    # Average marginal effect: the logit slope scaled by the mean density.
    link = design @ res.params
    density = float(np.mean(np.exp(-link) / (1 + np.exp(-link)) ** 2))
    idx = slice(1, 1 + len(terms))
    return Fit(
        name="logit (25% lifter sample, AME)",
        terms=terms,
        beta=res.params[idx] * density,
        se=np.sqrt(np.diag(res.cov_params()))[idx] * density,
        n=sub.height,
        n_clusters=int(codes.max()) + 1,
        outcome_mean=float(y.mean()),
        note="average marginal effects; Sex and year dummies only (logit cannot "
             "absorb the lifter or weight-class effects)"
             + ("" if converged else "; WARNING: the likelihood did not converge"),
    )


# --- output -----------------------------------------------------------------


def tidy(fits: list[Fit]) -> pl.DataFrame:
    rows = []
    for fit in fits:
        for term in fit.terms:
            row = fit.row(term)
            row.update(model=fit.name, n=fit.n, n_lifters=fit.n_clusters,
                       outcome_mean=fit.outcome_mean, note=fit.note)
            rows.append(row)
    return pl.DataFrame(rows).select(
        "model", "term", "coef", "pp", "se_pp", "t", "p", "ci_lo_pp", "ci_hi_pp",
        "n", "n_lifters", "outcome_mean", "note",
    )


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def side_by_side(fits: list[Fit], terms: list[str], title: str) -> str:
    """A regression table in percentage points, coefficients over std errors."""
    width = 22
    lines = [title, "=" * (26 + width * len(fits))]
    lines.append("effect sizes in percentage points of success probability")
    lines.append("")
    header = f"{'':26s}" + "".join(f"{f.name[:width-2]:>{width}s}" for f in fits)
    lines += [header, "-" * len(header)]

    for term in terms:
        if not any(term in f.terms for f in fits):
            continue
        coefs, ses = f"{term:26s}", f"{'':26s}"
        for fit in fits:
            row = fit.row(term)
            if row is None:
                coefs += f"{'':>{width}s}"
                ses += f"{'':>{width}s}"
            else:
                coefs += f"{row['pp']:>{width - 4}.3f}{stars(row['p']):<4s}"
                ses += f"{'(' + format(row['se_pp'], '.3f') + ')':>{width}s}"
        lines += [coefs, ses]

    lines.append("-" * len(header))
    for label, get in (
        ("observations", lambda f: f"{f.n:,}"),
        ("lifters", lambda f: f"{f.n_clusters:,}"),
        ("mean outcome", lambda f: f"{f.outcome_mean:.3f}"),
    ):
        lines.append(f"{label:26s}" + "".join(f"{get(f):>{width}s}" for f in fits))
    lines.append("")
    lines.append("std errors clustered by lifter in parentheses; "
                 "* p<0.1, ** p<0.05, *** p<0.01")
    for fit in fits:
        if fit.note:
            lines.append(f"  [{fit.name}] {fit.note}")
        if fit.dropped:
            lines.append(f"  [{fit.name}] dropped, no variance: {', '.join(fit.dropped)}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=IN_PARQUET)
    parser.add_argument("--skip-logit", action="store_true")
    args = parser.parse_args()
    use_utf8_stdout()

    if not args.src.exists():
        print(f"{args.src} not found - run the discover stage first.", file=sys.stderr)
        return 1

    sample = build_sample(pl.read_parquet(args.src))
    print(f"\nsample: {sample.height:,} attempts, "
          f"{sample['lifter_id'].n_unique():,} lifters, "
          f"mean success {sample['success'].mean():.4f}")

    fits = run_specifications(sample)
    print("\nasymmetry")
    asym = asymmetry_test(sample)
    for fit in asym:
        row = fit.row("running_x_above")
        print(f"  {fit.name:44s} n={fit.n:>9,}  slope difference = "
              f"{row['coef']:+.4f} kg per kg (se {row['se_pp']/100:.4f})")

    if not args.skip_logit:
        print("\nlogit")
        try:
            logit = run_logit(sample)
        except Exception as exc:  # noqa: BLE001
            # A robustness check must not discard ten models that already
            # converged. The first version let a logit crash kill the run
            # before anything had been written.
            logit = None
            print(f"  logit failed, continuing without it: {exc}", file=sys.stderr)
        if logit:
            fits.append(logit)
            row = logit.row(key_term(logit))
            print(f"  {logit.name:44s} n={logit.n:>9,}  {key_term(logit)} = "
                  f"{row['pp']:+6.2f}pp (se {row['se_pp']:.2f})")

    all_fits = fits + asym
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tidy(all_fits).write_csv(CSV_PATH)
    print(f"\nwrote {CSV_PATH}")

    by_name = {f.name: f for f in fits}
    headline = [
        by_name[n] for n in (
            "headline (base FE)", "headline + lifter FE",
            "US pound-loaded only", "US kilo-loaded only",
            "discovered milestones (test half)",
        ) if n in by_name
    ]
    sections = [
        side_by_side(headline, [*BASE_TERMS, "on_discovered_milestone",
                                "disc_ms_x_region"],
                     "HEADLINE AND SECONDARY ANALYSES"),
        side_by_side([f for f in fits if f not in headline],
                     [*BASE_TERMS, "on_fake_milestone", "fake_ms_x_region"],
                     "ROBUSTNESS"),
        side_by_side(asym, ["running_kg", "above", "running_x_above"],
                     "ASYMMETRY AROUND A POUND MILESTONE (outcome: jump_kg, in kg)"),
    ]
    text = "\n\n\n".join(sections)
    TXT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {TXT_PATH}\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
