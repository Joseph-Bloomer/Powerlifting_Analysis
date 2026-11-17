"""Ingest stage: download the OpenPowerlifting bulk CSV and store it as parquet.

Finds the current zip link on the bulk-csv page, downloads and unzips it,
converts the CSV to data/raw/openpowerlifting.parquet, then prints a report:
row count, column list, and the share of rows with non-null Squat1Kg, Bench1Kg
and Deadlift1Kg broken down by Federation and by year.

No rows are filtered at this stage.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

import polars as pl
import requests

BULK_CSV_PAGE = "https://openpowerlifting.gitlab.io/opl-csv/bulk-csv.html"
FALLBACK_ZIP_URL = (
    "https://openpowerlifting.gitlab.io/opl-csv/files/openpowerlifting-latest.zip"
)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PARQUET_PATH = RAW_DIR / "openpowerlifting.parquet"

ATTEMPT_COLUMNS = [
    f"{lift}{n}Kg" for lift in ("Squat", "Bench", "Deadlift") for n in (1, 2, 3, 4)
]
FIRST_ATTEMPTS = ["Squat1Kg", "Bench1Kg", "Deadlift1Kg"]

# Columns whose types Polars tends to mis-infer from the first rows.
SCHEMA_OVERRIDES: dict[str, pl.DataType] = {
    **{c: pl.Float64 for c in ATTEMPT_COLUMNS},
    **{
        c: pl.Float64
        for c in (
            "Best3SquatKg",
            "Best3BenchKg",
            "Best3DeadliftKg",
            "TotalKg",
            "BodyweightKg",
            "Age",
            "Dots",
            "Wilks",
            "Glossbrenner",
            "Goodlift",
        )
    },
    "Place": pl.Utf8,  # holds DQ, DD, NS, G as well as numbers
    "WeightClassKg": pl.Utf8,  # holds values like "90+"
    "Date": pl.Utf8,
}


def find_zip_url(page_html: str) -> str:
    """Extract the bulk CSV zip link from the download page HTML.

    Falls back to the stable 'latest' URL if no link is found.
    """
    matches = re.findall(r'href="([^"]+\.zip)"', page_html)
    for href in matches:
        if "openpowerlifting" in href:
            if href.startswith("http"):
                return href
            # Resolve relative to the page's directory.
            base = BULK_CSV_PAGE.rsplit("/", 1)[0]
            return f"{base}/{href.lstrip('./')}"
    return FALLBACK_ZIP_URL


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(
                        f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB",
                        end="",
                        flush=True,
                    )
    print()


def extract_csv(zip_path: Path, out_dir: Path) -> Path:
    """Unzip the archive and return the path of the main CSV inside it."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found inside {zip_path}")
        # The bulk archive holds one large CSV; take the biggest if several.
        main = max(csv_names, key=lambda n: zf.getinfo(n).file_size)
        print(f"Extracting {main}")
        zf.extract(main, out_dir)
    return out_dir / main


def csv_to_parquet(csv_path: Path, parquet_path: Path) -> None:
    print(f"Converting {csv_path} -> {parquet_path}")
    (
        pl.scan_csv(
            csv_path,
            schema_overrides=SCHEMA_OVERRIDES,
            infer_schema_length=50_000,
        ).sink_parquet(parquet_path)
    )


def non_null_share_report(lf: pl.LazyFrame, by: pl.Expr, label: str) -> pl.DataFrame:
    """Share of rows with non-null first attempts for each group."""
    return (
        lf.group_by(by.alias(label))
        .agg(
            pl.len().alias("rows"),
            *[
                (pl.col(c).is_not_null().mean()).round(3).alias(f"share_{c}")
                for c in FIRST_ATTEMPTS
            ],
        )
        .sort("rows", descending=True)
        .collect()
    )


def print_report(parquet_path: Path) -> None:
    # Polars draws tables with box-drawing characters, which the default
    # Windows console encoding (cp1252) cannot encode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    lf = pl.scan_parquet(parquet_path)
    schema = lf.collect_schema()
    n_rows = lf.select(pl.len()).collect().item()

    print(f"\nRows: {n_rows:,}")
    print(f"\nColumns ({len(schema.names())}):")
    for name in schema.names():
        print(f"  {name}: {schema[name]}")

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=120):
        print("\nShare of non-null first attempts by Federation:")
        print(non_null_share_report(lf, pl.col("Federation"), "Federation"))

        print("\nShare of non-null first attempts by year:")
        year = pl.col("Date").str.slice(0, 4).cast(pl.Int32, strict=False)
        print(non_null_share_report(lf, year, "year").sort("year"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download and rebuild the parquet even if it already exists",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if PARQUET_PATH.exists() and not args.force:
        print(f"{PARQUET_PATH} already exists; skipping download (use --force).")
        print_report(PARQUET_PATH)
        return 0

    try:
        page = requests.get(BULK_CSV_PAGE, timeout=60)
        page.raise_for_status()
        zip_url = find_zip_url(page.text)
    except requests.RequestException as exc:
        print(f"Could not read {BULK_CSV_PAGE} ({exc}); using fallback URL.")
        zip_url = FALLBACK_ZIP_URL

    zip_path = RAW_DIR / "openpowerlifting.zip"
    extract_dir = RAW_DIR / "_extracted"
    try:
        download(zip_url, zip_path)
        csv_path = extract_csv(zip_path, extract_dir)
        csv_to_parquet(csv_path, PARQUET_PATH)
    finally:
        zip_path.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)

    print_report(PARQUET_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
