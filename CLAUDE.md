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

## Data facts
- Source: OpenPowerlifting bulk CSV. Column docs:
  https://openpowerlifting.gitlab.io/opl-csv/bulk-csv-docs.html
- Negative attempt values mean a failed attempt; the weight is the absolute value.
- Many federations do not report attempts. Never impute them.
- 4th attempts do not count towards the total. Exclude them.
- Place can be non-numeric (DQ, DD, NS, G). Exclude those rows.
- Sample: Event == SBD, Equipment == Raw, attempts 1-3 all present.
- Canada and the UK are excluded from the region comparison as ambiguous.

## Pipeline
ingest -> clean -> features -> describe -> econ_models -> ml
Each stage is a script in src/, reads the previous stage's parquet and writes
its own. Stages never reach back into raw data. Run via the Makefile.

## Rules
- Polars and DuckDB, not pandas, except where statsmodels needs it.
- No feature may use information from after the attempt it describes.
  Experience and prior-best variables use strictly earlier meet dates.
- Train/test splits are by lifter, never by row.
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
