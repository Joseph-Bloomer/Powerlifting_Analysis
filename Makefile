PY = uv run python

.PHONY: ingest clean features discover describe model ml all test

ingest:
	$(PY) src/ingest.py

clean:
	$(PY) src/clean.py

features:
	$(PY) src/features.py

discover:
	$(PY) src/discover.py

describe:
	$(PY) src/describe.py

model:
	$(PY) src/econ_models.py

ml:
	$(PY) src/ml.py

all: ingest clean features discover describe model ml

test:
	uv run pytest
