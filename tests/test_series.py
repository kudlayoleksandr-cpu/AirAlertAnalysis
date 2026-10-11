import numpy as np
import pandas as pd
import pytest

from src import load, series


def _utc(*stamps):
    return pd.to_datetime(list(stamps), utc=True)


def _kyiv(stamp):
    return pd.Timestamp(stamp, tz="Europe/Kyiv")


def _daily(intervals, first_day, last_day):
    starts = [a for a, _ in intervals]
    ends = [b for _, b in intervals]
    return series.daily_hours(pd.DatetimeIndex(starts), pd.DatetimeIndex(ends), first_day, last_day)


def _brute_force_daily(intervals, first_day, last_day):
    """Minute-by-minute reference implementation (intervals on whole minutes)."""
    out = {}
    for day in pd.date_range(first_day, last_day, freq="D"):
        lo = day.tz_localize("Europe/Kyiv").tz_convert("UTC")
        hi = (day + pd.Timedelta(days=1)).tz_localize("Europe/Kyiv").tz_convert("UTC")
        minutes = pd.date_range(lo, hi, freq="min", inclusive="left")
        covered = np.zeros(len(minutes), dtype=bool)
        for a, b in intervals:
            covered |= (minutes >= a) & (minutes < b)
        out[day] = covered.sum() / 60
    return pd.Series(out)


# ---- merge_intervals ----

def test_merge_overlapping_nested_and_touching():
    merged = series.merge_intervals(
        _utc("2025-01-01 10:00", "2025-01-01 10:30", "2025-01-01 10:10", "2025-01-01 12:00", "2025-01-01 13:00"),
        _utc("2025-01-01 11:00", "2025-01-01 11:30", "2025-01-01 10:20", "2025-01-01 13:00", "2025-01-01 13:30"),
    )
    assert merged["started_at"].tolist() == list(_utc("2025-01-01 10:00", "2025-01-01 12:00"))
    assert merged["finished_at"].tolist() == list(_utc("2025-01-01 11:30", "2025-01-01 13:30"))


def test_merge_is_order_independent_and_handles_empty():
    starts = _utc("2025-01-02 00:00", "2025-01-01 00:00")
    ends = _utc("2025-01-02 01:00", "2025-01-01 01:00")
    merged = series.merge_intervals(starts, ends)
    assert merged["started_at"].is_monotonic_increasing
    assert len(merged) == 2
    assert len(series.merge_intervals(starts[:0], ends[:0])) == 0


# ---- daily series ----

def test_alert_inside_one_day():
    daily = _daily([(_kyiv("2025-06-10 13:00"), _kyiv("2025-06-10 14:30"))], "2025-06-09", "2025-06-11")
    assert daily.tolist() == pytest.approx([0, 1.5, 0])


def test_alert_crossing_midnight_is_split():
    daily = _daily([(_kyiv("2025-06-10 23:00"), _kyiv("2025-06-11 02:30"))], "2025-06-10", "2025-06-11")
    assert daily.tolist() == pytest.approx([1.0, 2.5])


def test_days_are_kyiv_local_not_utc():
    # 22:30 UTC in winter (UTC+2) is 00:30 the next day in Kyiv
    daily = _daily(
        [(pd.Timestamp("2025-01-10 22:30", tz="UTC"), pd.Timestamp("2025-01-10 23:30", tz="UTC"))],
        "2025-01-10", "2025-01-11",
    )
    assert daily.tolist() == pytest.approx([0, 1.0])


def test_overlapping_alerts_are_not_double_counted():
    daily = _daily(
        [
            (_kyiv("2025-06-10 10:00"), _kyiv("2025-06-10 12:00")),
            (_kyiv("2025-06-10 11:00"), _kyiv("2025-06-10 13:00")),  # partial overlap
            (_kyiv("2025-06-10 10:30"), _kyiv("2025-06-10 11:30")),  # nested
            (_kyiv("2025-06-10 10:00"), _kyiv("2025-06-10 12:00")),  # identical
        ],
        "2025-06-10", "2025-06-10",
    )
    assert daily.tolist() == pytest.approx([3.0])


def test_overlap_across_midnight():
    daily = _daily(
        [
            (_kyiv("2025-06-10 22:00"), _kyiv("2025-06-11 01:00")),
            (_kyiv("2025-06-10 23:00"), _kyiv("2025-06-11 03:00")),
        ],
        "2025-06-10", "2025-06-11",
    )
    assert daily.tolist() == pytest.approx([2.0, 3.0])


def test_alert_spanning_several_days():
    daily = _daily([(_kyiv("2025-06-10 18:00"), _kyiv("2025-06-13 06:00"))], "2025-06-09", "2025-06-14")
    assert daily.tolist() == pytest.approx([0, 6, 24, 24, 6, 0])


def test_spring_clock_change_day_has_23_hours():
    # 2025-03-30: clocks go forward 03:00 -> 04:00 in Kyiv
    daily = _daily([(_kyiv("2025-03-29 12:00"), _kyiv("2025-03-31 12:00"))], "2025-03-29", "2025-03-31")
    assert daily.tolist() == pytest.approx([12, 23, 12])


def test_autumn_clock_change_day_has_25_hours():
    # 2025-10-26: clocks go back 04:00 -> 03:00 in Kyiv
    daily = _daily([(_kyiv("2025-10-25 12:00"), _kyiv("2025-10-27 12:00"))], "2025-10-25", "2025-10-27")
    assert daily.tolist() == pytest.approx([12, 25, 12])


def test_alert_across_the_spring_gap_counts_elapsed_time():
    # 02:30 EET to 04:30 EEST is one elapsed hour, because 03:00-04:00 does not exist
    daily = _daily([(_kyiv("2025-03-30 02:30"), _kyiv("2025-03-30 04:30"))], "2025-03-30", "2025-03-30")
    assert daily.tolist() == pytest.approx([1.0])


def test_alerts_outside_the_requested_range_are_clipped():
    daily = _daily([(_kyiv("2025-06-01 00:00"), _kyiv("2025-06-30 00:00"))], "2025-06-10", "2025-06-11")
    assert daily.tolist() == pytest.approx([24, 24])


def test_no_alerts_gives_zeros():
    empty = pd.DatetimeIndex([], tz="UTC")
    daily = series.daily_hours(empty, empty, "2025-06-10", "2025-06-12")
    assert daily.tolist() == [0, 0, 0]


def test_daily_matches_brute_force_on_random_intervals():
    rng = np.random.default_rng(0)
    first, last = "2025-03-25", "2025-04-03"  # includes the spring clock change
    base = pd.Timestamp("2025-03-24 12:00", tz="UTC")
    intervals = []
    for _ in range(60):
        start = base + pd.Timedelta(minutes=int(rng.integers(0, 12 * 24 * 60)))
        intervals.append((start, start + pd.Timedelta(minutes=int(rng.integers(1, 30 * 60)))))
    expected = _brute_force_daily(intervals, first, last)
    result = _daily(intervals, first, last)
    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy(), atol=1e-9)


# ---- hourly series ----

def test_hourly_splits_within_and_across_hours():
    hourly = series.hourly_hours(
        pd.DatetimeIndex([_kyiv("2025-06-10 13:15")]), pd.DatetimeIndex([_kyiv("2025-06-10 15:30")]),
        "2025-06-10", "2025-06-10",
    )
    assert len(hourly) == 24
    assert hourly.index[0] == _kyiv("2025-06-10 00:00")
    assert hourly[hourly > 0].tolist() == pytest.approx([0.75, 1.0, 0.5])
    assert hourly[hourly > 0].index.hour.tolist() == [13, 14, 15]


@pytest.mark.parametrize("day, n_hours", [("2025-03-30", 23), ("2025-10-26", 25), ("2025-06-10", 24)])
def test_hourly_length_on_clock_change_days(day, n_hours):
    hourly = series.hourly_hours(
        pd.DatetimeIndex([_kyiv(day)]), pd.DatetimeIndex([_kyiv(day) + pd.Timedelta(hours=30)]), day, day,
    )
    assert len(hourly) == n_hours
    assert hourly.sum() == pytest.approx(n_hours)


def test_hourly_sums_to_daily():
    rng = np.random.default_rng(1)
    base = pd.Timestamp("2025-10-20", tz="UTC")
    starts = pd.DatetimeIndex([base + pd.Timedelta(minutes=int(m)) for m in rng.integers(0, 10 * 24 * 60, 80)])
    ends = starts + pd.to_timedelta(rng.integers(1, 600, 80), unit="min")
    first, last = "2025-10-22", "2025-10-28"  # includes the autumn clock change
    daily = series.daily_hours(starts, ends, first, last)
    hourly = series.hourly_hours(starts, ends, first, last)
    by_day = hourly.groupby(hourly.index.tz_localize(None).normalize()).sum()
    np.testing.assert_allclose(by_day.to_numpy(), daily.to_numpy(), atol=1e-9)


# ---- region selection ----

def _alerts(rows):
    frame = pd.DataFrame(rows, columns=["oblast", "level", "started_at", "finished_at", "standing_alert"])
    frame["started_at"] = pd.to_datetime(frame["started_at"], utc=True)
    frame["finished_at"] = pd.to_datetime(frame["finished_at"], utc=True)
    return frame


def test_region_series_is_union_across_levels_and_ignores_other_oblasts():
    alerts = _alerts([
        ("A oblast", "oblast", "2025-06-10 09:00", "2025-06-10 10:00", False),
        ("A oblast", "raion", "2025-06-10 09:30", "2025-06-10 11:00", False),
        ("A oblast", "hromada", "2025-06-10 12:00", "2025-06-10 12:30", False),
        ("B oblast", "oblast", "2025-06-10 00:00", "2025-06-10 20:00", False),
    ])
    daily = series.region_daily(alerts, "A oblast", "2025-06-10", "2025-06-10")
    assert daily.tolist() == pytest.approx([2.5])


def test_standing_alerts_are_excluded_unless_requested():
    alerts = _alerts([
        ("A oblast", "raion", "2025-06-10 09:00", "2025-06-10 10:00", False),
        ("A oblast", "hromada", "2025-01-01 00:00", "2025-12-31 00:00", True),
    ])
    assert series.region_daily(alerts, "A oblast", "2025-06-10", "2025-06-10").tolist() == pytest.approx([1.0])
    with_standing = series.region_daily(alerts, "A oblast", "2025-06-10", "2025-06-10", include_standing=True)
    assert with_standing.tolist() == pytest.approx([24.0])


def test_unknown_region_raises():
    alerts = _alerts([("A oblast", "oblast", "2025-06-10 09:00", "2025-06-10 10:00", False)])
    with pytest.raises(ValueError):
        series.region_daily(alerts, "Kyiv", "2025-06-10", "2025-06-10")


# ---- checks against the real pinned file ----

needs_data = pytest.mark.skipif(not load.RAW_PATH.exists(), reason="run scripts/fetch_data.py first")


@pytest.fixture(scope="module")
def alerts():
    return load.load()


@needs_data
def test_kyiv_city_series_shape_and_bounds(alerts):
    daily = series.region_daily(alerts, "Kyiv City")
    assert daily.index[0] == pd.Timestamp("2022-03-16")
    assert daily.index[-1] == pd.Timestamp("2026-08-28")
    assert not daily.isna().any()
    assert daily.min() >= 0
    assert daily.max() <= 25


@needs_data
def test_kyiv_city_monthly_sanity_totals(alerts):
    daily = series.region_daily(alerts, "Kyiv City")
    assert daily["2025-02"].sum() == pytest.approx(121.78, abs=0.01)
    # the user's ~80 h for Aug 2026 included 4.5 h on the discarded 2026-08-29
    assert daily["2026-08"].sum() == pytest.approx(74.96, abs=0.01)


@needs_data
def test_kyiv_city_total_equals_sum_of_merged_intervals(alerts):
    rows = series.region_rows(alerts, "Kyiv City")
    first = pd.Timestamp(series.FIRST_COMPLETE_DAY, tz="Europe/Kyiv")
    merged = series.merge_intervals(rows["started_at"].clip(lower=first), rows["finished_at"].clip(lower=first))
    total = (merged["finished_at"] - merged["started_at"]).dt.total_seconds().sum() / 3600
    assert series.region_daily(alerts, "Kyiv City").sum() == pytest.approx(total, abs=1e-6)


@needs_data
def test_kharkivska_standing_alert_rule(alerts):
    kept = series.region_daily(alerts, "Kharkivska oblast")
    everything = series.region_daily(alerts, "Kharkivska oblast", include_standing=True)
    # with the two standing hromada alerts the oblast is "under alert" all day
    assert everything["2024-06":"2025-07"].min() >= 23
    assert kept["2024-06":"2025-07"].mean() < 20
    # outside their span the rule changes nothing
    pd.testing.assert_series_equal(kept[:"2024-05-11"], everything[:"2024-05-11"])
    pd.testing.assert_series_equal(kept["2026-01-07":], everything["2026-01-07":])
    # any threshold between 51 h and 10,000 h selects the same two rows
    durations = series.region_rows(alerts, "Kharkivska oblast", include_standing=True)["duration_h"]
    assert ((durations > 51) & (durations < 10_000)).sum() == 0
    assert (durations > load.STANDING_ALERT_HOURS).sum() == 2
