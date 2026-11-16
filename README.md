# powerlifting-reference-points

Reference-dependence study of third-attempt selection in powerlifting, using
the [OpenPowerlifting bulk CSV](https://openpowerlifting.gitlab.io/opl-csv/bulk-csv-docs.html).
See CLAUDE.md for the research design and project rules.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and `make`.

```bash
uv sync
```

## Running

The pipeline is `ingest -> clean -> features -> describe -> model -> ml`; each
stage reads the previous stage's parquet from `data/` and writes its own.

```bash
make ingest      # download the bulk CSV (~GBs), convert to parquet, print coverage report
make all         # run every stage in order
make milestones  # reference tool: detect candidate milestone bins from the data
make test        # run the test suite
```

`make ingest` skips the download if the parquet already exists; pass
`--force` to `src/ingest.py` to rebuild. Data folders are gitignored.
