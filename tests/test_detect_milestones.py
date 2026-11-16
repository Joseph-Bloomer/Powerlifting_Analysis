import numpy as np
import polars as pl

from detect_milestones import apply_sample_filters, bin_attempts, detect, excess_z


def make_sample(n_flat: int = 2000, n_spike: int = 400, seed: int = 0) -> pl.DataFrame:
    """Synthetic SBD/Raw sample: flat deadlift thirds plus a spike at 227.5 kg."""
    rng = np.random.default_rng(seed)
    flat = rng.uniform(150, 300, n_flat)
    flat = np.round(flat / 2.5) * 2.5
    spike = np.full(n_spike, 227.5)
    weights = np.concatenate([flat, spike])
    # Roughly half the attempts fail; sign encodes success.
    sign = np.where(rng.random(len(weights)) < 0.5, 1.0, -1.0)
    n = len(weights)
    return pl.DataFrame(
        {
            "Event": ["SBD"] * n,
            "Equipment": ["Raw"] * n,
            "Place": ["1"] * n,
            "MeetCountry": ["USA"] * n,
            "TotalKg": rng.uniform(300, 700, n),
            "Squat1Kg": [100.0] * n,
            "Squat2Kg": [105.0] * n,
            "Squat3Kg": [110.0] * n,
            "Bench1Kg": [60.0] * n,
            "Bench2Kg": [65.0] * n,
            "Bench3Kg": [70.0] * n,
            "Deadlift1Kg": [140.0] * n,
            "Deadlift2Kg": [150.0] * n,
            "Deadlift3Kg": weights * sign,
        }
    )


def test_excess_z_flags_planted_spike():
    counts = np.full(41, 100.0)
    counts[20] = 400.0
    z = excess_z(counts)
    assert z[20] > 4
    assert abs(z[10]) < 1


def test_excess_z_handles_flat_and_short_arrays():
    assert np.all(excess_z(np.full(10, 5.0)) == 0)
    assert len(excess_z(np.array([1.0]))) == 1


def test_apply_sample_filters_logs_and_filters(capsys):
    df = make_sample(n_flat=50, n_spike=0)
    # Pollute with rows that each filter should remove.
    bad = df[:4].with_columns(
        pl.Series("Event", ["B", "SBD", "SBD", "SBD"]),
        pl.Series("Equipment", ["Raw", "Wraps", "Raw", "Raw"]),
        pl.Series("Place", ["1", "1", "DQ", "1"]),
        pl.Series("MeetCountry", ["USA", "USA", "USA", "Canada"]),
    )
    out = apply_sample_filters(pl.concat([df, bad]))
    assert out.height == df.height
    assert out["region"].unique().to_list() == ["lb"]
    captured = capsys.readouterr().out
    assert "Event == SBD: removed 1" in captured
    assert "Equipment == Raw: removed 1" in captured
    assert "numeric Place: removed 1" in captured
    assert "drop ambiguous-unit countries: removed 1" in captured


def test_detect_flags_planted_milestone_only():
    df = apply_sample_filters(make_sample())
    cell = detect(df, "deadlift", "lb", "kg", z_threshold=4.0)
    flagged = cell.filter(pl.col("flagged"))["bin"].to_list()
    assert flagged == [227.5]
    fail = cell.filter(pl.col("bin") == 227.5)["fail_rate"][0]
    assert 0.3 < fail < 0.7


def test_detect_no_flags_on_flat_data():
    df = apply_sample_filters(make_sample(n_spike=0, n_flat=5000))
    cell = detect(df, "deadlift", "lb", "kg", z_threshold=4.0)
    assert cell.filter(pl.col("flagged")).height == 0


def test_bin_attempts_fills_gaps_with_zero():
    df = pl.DataFrame({"weight": [100.0, 100.0, 110.0], "ok": [True, False, True]})
    out = bin_attempts(df, "weight", "ok", 2.5)
    assert out.height == 5  # 100, 102.5, 105, 107.5, 110
    assert out.filter(pl.col("bin") == 105.0)["count"][0] == 0
    assert out.filter(pl.col("bin") == 100.0)["fail_rate"][0] == 0.5
