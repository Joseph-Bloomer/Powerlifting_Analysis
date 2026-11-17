# Powerlifting reference points

## Research question
Do lifters stretch their third attempts to hit round-number milestones, and
does that lower their success rate? This is a reference-dependence study.
Only describe results as loss aversion if the asymmetry test supports it.

## Identification (do not change without asking me)
Attempt weight is chosen, so raw comparisons are confounded. The design rests
on pound milestones and kilo milestones being different weights: 500 lb is
about 226.8 kg, so 227.5 kg should matter in US meets and not in kilo
countries. The key estimate is on_lb_milestone x lb_region. Fake milestones
offset by 12.5 kg are the placebo and should show no effect.

Milestones are defined by the meet's loading unit, which clean.py decides once
per meet and never attempt by attempt. A kilo weight that happens to fall near
a round pound figure is not a pound milestone.
- Kilo-loaded meets: the lifter cannot load the pound figure itself, so a pound
  milestone is the smallest loadable 2.5 kg weight at or above it. 500 lb
  becomes 227.5 kg. This is the bulk of the sample and carries the estimate.
- Pound-loaded meets: the bar is loaded in pounds, so the milestone is the
  pound figure exactly, 226.8 kg, and distances are measured in pounds.
- Mixed meets get a null loading unit and null milestone variables.

The headline model stays pooled over both loading units. loading_unit is for
plots and one robustness split, not a second identification strategy.

## Data facts
- Source: OpenPowerlifting bulk CSV. Column docs:
  https://openpowerlifting.gitlab.io/opl-csv/bulk-csv-docs.html
- Negative attempt values mean a failed attempt; the weight is the absolute value.
- Many federations do not report attempts. Never impute them.
- 4th attempts do not count towards the total. Exclude them.
- Place can be non-numeric (DQ, DD, NS, G). Exclude those rows.
- Sample: Event == SBD, Equipment == Raw, attempts 1-3 all present.
- A lifter entered in several divisions at one meet appears once per division,
  with the same attempts. Deduplicate to one entry per lifter per meet.
- Not every meet loads in kilos. Pound-loaded meets are stored as the kilo
  conversion of a round pound number, which never lands on the kilo grid. An
  attempt is valid if it is on the 0.5 kg grid or within rounding tolerance of
  a whole pound; treating the latter as errors discards every literally loaded
  pound milestone, 500 lb included.
- Canada and the UK are excluded from the region comparison as ambiguous.

## Pipeline
ingest -> clean -> features -> discover -> describe -> econ_models -> ml
Each stage is a script in src/, reads the previous stage's parquet and writes
its own. Stages never reach back into raw data. Run via the Makefile.

## Rules
- Polars and DuckDB, not pandas, except where statsmodels needs it.
- No feature may use information from after the attempt it describes.
  Experience and prior-best variables use strictly earlier meet dates.
- Train/test splits are by lifter, never by row. The discovery/test split is
  the exception and is by meet: bunching is a meet-level phenomenon.
- Milestone discovery never uses outcome data: not the success column, nor
  anything derived from it.
- Any outcome test on discovered milestones uses the test half only.
- Cluster standard errors by lifter.
- Every filter logs how many rows it removes.
- Report effect sizes in percentage points, not just p-values.
- Plots are descriptive unless a regression backs them. Never label a
  predictive result as causal or "optimal".
- Do not write interpretation or conclusions in the README. I write those.
- If the data contradicts an assumption here, stop and tell me rather than
  working around it.

## Environment
Python via uv. Tests with pytest. Data folders are gitignored.
