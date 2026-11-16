PY = uv run python

.PHONY: ingest clean features describe model ml milestones all test

ingest:
	$(PY) src/ingest.py

clean:
	$(PY) src/clean.py

features:
	$(PY) src/features.py

describe:
	$(PY) src/describe.py

model:
	$(PY) src/econ_models.py

ml:
	$(PY) src/ml.py

# Reference tool, not a pipeline stage: writes references/milestones_detected.csv
milestones:
	$(PY) src/detect_milestones.py

all: ingest clean features describe model ml

test:
	uv run pytest
