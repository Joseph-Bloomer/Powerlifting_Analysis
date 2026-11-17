# Notes

Running list of things parked during the build, for Joe to go through. Each
item says whether it needs a decision or is just a caveat to carry.

---

## Resolved

- **Item 0 — pound-denominated attempts.** Done, option 2 plus a light option 3.
  clean.py accepts an attempt on the 0.5 kg grid *or* within rounding tolerance
  of a whole pound; the "neither grid" filter now removes 17,190 rows instead of
  93,813. Each meet carries a `loading_unit` (kg / lb / null) set from the share
  of its attempts that are pound-native. features.py defines milestones by that
  unit. CLAUDE.md identification section updated.
- **Item 1 — detect_milestones.py.** Re-pointed at data/interim/attempts.parquet;
  inline sample filters deleted, only the region split remains.
- **Item 2 — lifter_id from Name.** Accepted as a caveat, no change. Still needs
  a sentence in the limitations: a Name-based id merges same-named lifters OPL
  never disambiguated and splits anyone who changed names, which matters because
  SEs cluster by lifter and splits are by lifter.
- **Item 3 — CLAUDE.md data facts.** Multi-division and loading-unit bullets added.
- **Item 4 — duplicated constants.** src/constants.py created; clean.py,
  features.py and detect_milestones.py all import from it.

## Open

- **ml.py fills best_before_kg and prior_best_kg with 0.0 when there is no
  history.** Visible as a detached cluster at the right of the SHAP summary:
  the tree splits those rows off cleanly, and the `no_history` indicator is in
  the design beside them, so nothing is being hidden. But 0 kg is not a
  plausible weight, and the logistic regression sees it after scaling as an
  extreme value rather than a category. A median fill plus the indicator, or a
  separate model for first-timers, would be cleaner if the predictive stage
  ever carries weight.
- **The placebo family is the open question for the design, not a detail.**
  Seven of eight fake offsets are significant and five exceed the true estimate
  in absolute size. Until that is explained the headline coefficient cannot be
  read as a milestone effect. Candidate explanations worth testing: the two
  milestone grids coincide every ~90 kg (diagnostic 5); attempt weights cluster
  on the 2.5 kg loadable grid, so any offset grid picks up loading structure;
  and the kilo comb is strong enough that a 10 kg-aligned offset is itself a
  milestone grid.

- **Runtime.** econ_models.py takes roughly 10-15 minutes on this machine. The
  cost is absorbing 296k lifter fixed effects by alternating projections across
  five fixed-effect sets; the inner loop is already Fortran-ordered, which cut
  it by about half. If it becomes annoying, the next step is a sparse incidence
  matmul instead of one bincount per column.
- **Two estimator bugs worth knowing about**, both caught and fixed, both
  invisible in the output rather than loud:
  1. In the US-only splits `lb_region` is constant, so `lb_ms_x_region` is an
     exact copy of `on_lb_milestone`. The pseudo-inverse split the effect
     across the two identical columns and reported half of it. Now any column
     that does not add rank after absorption is dropped and logged.
  2. The describe stage binned with `cut()` keyed on the interval's string
     label, so "[10, 12)" sorted before "[2, 3)" and the plotted lines doubled
     back on themselves. Now binned arithmetically.

- **The dispersion correction suppresses pervasive bunching.** Visible in
  describe_excess_mass.png, bottom panel. In kilo countries essentially *every*
  10 kg multiple carries excess mass: median +47.6%, positive in 100% of bins.
  But discovery flags only 200 kg and 300 kg. The mechanism is that the
  per-cell dispersion is estimated from Pearson residuals across the whole
  distribution, and when bunching is regular and everywhere it inflates that
  dispersion, which then widens the standard error for the very comb that
  caused it. So the discovery list under-reports kilo milestones, while the
  excess-mass table reports them correctly.

  This does not touch the headline design: the lb-vs-kg contrast is unaffected,
  and under-detection is conservative. But the discovered list should not be
  read as "these are the only milestones". Fixing it properly means estimating
  dispersion from a comb-free reference (e.g. residuals at non-multiple bins
  only), which changes the estimator, so I have not done it.

- **Totals scan dropped.** The old detect_milestones.py also scanned meet
  totals and found 455.0 kg (1000 lb) and 907.5 kg (2000 lb) bunching in
  lb-region. discover.py scans the three lifts only, since totals are not a
  lift and your spec asked for per-lift x per-group. Easy to add back as a
  fourth cell if you want those totals on the record.
- **Plate-count milestones.** Discovery turned up 225, 315 and 405 lb in
  pound-loaded meets, which are not round numbers: they are 2, 3 and 4 plates
  per side on a 45 lb bar. If that is a real reference point it is a different
  mechanism from round-number reference dependence, and the pre-specified
  50 lb / 10 kg grid does not capture it. Worth deciding whether the design
  should treat plate counts as a separate milestone family.

- **Loading-unit thresholds.** A meet is lb-loaded at a pound-native share >= 0.8
  and kg-loaded at <= 0.2. The observed distribution is sharply bimodal
  (19,589 meets below 0.05, 365 above 0.95, one single meet between 0.3 and 0.7),
  so the cutoffs sit in empty space, but they are still hard-coded in
  constants.py if you want them moved.
- **Distance units.** You asked for pound-meet distances in pounds.
  `dist_to_next_lb_ms` is in **kilos for every meet** so the pooled headline
  model can read one column without mixing units, and `dist_to_next_lb_ms_lb`
  carries the same gap in pounds for plots and pound-meet work. Say the word if
  you want the primary column switched to pounds for lb meets.
- **Placebo milestones.** CLAUDE.md names fake milestones offset by 12.5 kg as
  the placebo. Not built yet; no prompt has asked for them.

