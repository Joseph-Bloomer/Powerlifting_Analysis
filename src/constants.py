"""Constants shared by the pipeline stages and the milestone reference tool.

Kept in one place so the regression sample and the milestone detector can never
drift apart about what counts as a pound milestone or which meets are
ambiguous. No logic here, no imports from other project modules.
"""

from __future__ import annotations

KG_PER_LB = 0.45359237

# Smallest loadable increment on a symmetric barbell, in each unit system.
PLATE_STEP_KG = 2.5
PLATE_STEP_LB = 5.0

GRID_KG = 0.5  # attempts in kilo-loaded meets sit on this grid
TOL = 1e-6  # float comparison slack on the kilo grid

# Pound slack. OpenPowerlifting stores kilos to two decimals, so a whole-pound
# weight survives the round trip to within about 0.011 lb. 0.03 is comfortably
# wider than that and still far tighter than the 5 lb loadable step.
LB_TOL = 0.03

# Milestone grades. The coarse grade is a subset of the fine one, kept separate
# because the rounder number is the stronger reference point.
LB_STEP = 50.0
LB_COARSE_STEP = 100.0
KG_STEP = 10.0
KG_COARSE_STEP = 50.0

# Region. Canada and the UK are excluded as ambiguous (CLAUDE.md). Ireland is
# the Republic, not part of the UK, so it stays in as a kilo country.
LB_REGION_COUNTRY = "USA"
AMBIGUOUS_COUNTRIES = [
    "Canada",
    "UK",
    "England",
    "Scotland",
    "Wales",
    "N.Ireland",
    "Northern Ireland",
]

# Meet loading unit. A meet is classified by the share of its attempts that are
# pound-native (a whole number of pounds, off the kilo grid). The observed
# distribution of that share is sharply bimodal: of 20,050 meets with at least
# 30 attempts, 19,589 sit below 0.05 and 365 above 0.95, with a single meet
# anywhere between 0.3 and 0.7. These cutoffs sit in empty space.
LOADING_LB_THRESHOLD = 0.8
LOADING_KG_THRESHOLD = 0.2
MIN_ATTEMPTS_FOR_LOADING_UNIT = 9  # below this the share is too noisy to trust

# Raw column prefixes and the attempt numbers that count towards the total.
LIFTS = {"Squat": "squat", "Bench": "bench", "Deadlift": "deadlift"}
ATTEMPT_NUMBERS = (1, 2, 3)

# --- Discovery stage --------------------------------------------------------

# Split salt. Fixed so the discovery/test assignment is identical on every run
# and every machine; hashlib, never Python's hash(), which is salted per process.
SPLIT_SALT = "powerlifting-reference-points-v1"
SPLIT_NAMES = ("discovery", "test")

# Bin width per loading unit, the loadable increment in that unit.
DISCOVERY_BIN_WIDTH = {"kg": PLATE_STEP_KG, "lb": PLATE_STEP_LB}

# Local counterfactual: a degree-POLY_DEGREE polynomial fitted to the WINDOW
# bins each side of the candidate, with DONUT bins each side removed so the
# candidate and its spillover cannot lift its own baseline.
POLY_DEGREE = 5
WINDOW = 20
DONUT = 1

# A discovered bin must be statistically and substantively excessive.
FDR_Q = 0.05
MIN_COUNT = 50
MIN_EXCESS_PCT = 0.15

# Trim the sparse tails before scanning; the full range stays in the output.
SCAN_LOW_Q = 0.005
SCAN_HIGH_Q = 0.995
