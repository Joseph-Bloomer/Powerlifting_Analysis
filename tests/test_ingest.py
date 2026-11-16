import polars as pl

from ingest import FALLBACK_ZIP_URL, find_zip_url, non_null_share_report


def test_find_zip_url_absolute_link():
    html = '<a href="https://openpowerlifting.gitlab.io/opl-csv/files/openpowerlifting-latest.zip">Download</a>'
    assert find_zip_url(html).endswith("openpowerlifting-latest.zip")


def test_find_zip_url_relative_link():
    html = '<a href="files/openpowerlifting-latest.zip">Download</a>'
    url = find_zip_url(html)
    assert url.startswith("https://openpowerlifting.gitlab.io/opl-csv/")
    assert url.endswith("files/openpowerlifting-latest.zip")


def test_find_zip_url_falls_back_when_no_link():
    assert find_zip_url("<html><body>nothing here</body></html>") == FALLBACK_ZIP_URL


def test_find_zip_url_ignores_unrelated_zips():
    html = '<a href="files/something-else.zip">other</a>'
    assert find_zip_url(html) == FALLBACK_ZIP_URL


def test_non_null_share_report():
    lf = pl.LazyFrame(
        {
            "Federation": ["A", "A", "B", "B"],
            "Squat1Kg": [100.0, None, 100.0, 100.0],
            "Bench1Kg": [60.0, 60.0, None, None],
            "Deadlift1Kg": [None, None, 120.0, 120.0],
        }
    )
    out = non_null_share_report(lf, pl.col("Federation"), "Federation").sort(
        "Federation"
    )
    assert out["rows"].to_list() == [2, 2]
    assert out["share_Squat1Kg"].to_list() == [0.5, 1.0]
    assert out["share_Bench1Kg"].to_list() == [1.0, 0.0]
    assert out["share_Deadlift1Kg"].to_list() == [0.0, 1.0]
