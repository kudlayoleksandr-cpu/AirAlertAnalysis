import pandas as pd
import pytest

from src import load


def _raw(rows):
    """rows: (oblast, raion, hromada, level, started_at, finished_at)"""
    frame = pd.DataFrame(rows, columns=load.COLUMNS[:-1], dtype=str)
    frame["source"] = "official"
    return frame


def _row(start, finish, oblast="Kyiv City"):
    return (oblast, "", "", "oblast", f"{start}+00:00", f"{finish}+00:00")


def test_exact_duplicates_are_dropped():
    raw = _raw([_row("2025-01-01 10:00:00", "2025-01-01 11:00:00")] * 3)
    assert len(load.clean(raw)) == 1


def test_rows_differing_in_any_field_are_not_duplicates():
    raw = _raw([
        _row("2025-01-01 10:00:00", "2025-01-01 11:00:00"),
        _row("2025-01-01 10:00:00", "2025-01-01 11:00:01"),
        _row("2025-01-01 10:00:00", "2025-01-01 11:00:00", oblast="Lvivska oblast"),
    ])
    assert len(load.clean(raw)) == 3


def test_nonpositive_durations_are_dropped():
    raw = _raw([
        _row("2025-01-01 10:00:00", "2025-01-01 10:00:00"),
        _row("2025-01-01 10:00:00", "2025-01-01 09:00:00"),
        _row("2025-01-01 10:00:00", "2025-01-01 10:00:01"),
    ])
    cleaned = load.clean(raw)
    assert len(cleaned) == 1
    assert cleaned["duration_h"].iloc[0] == pytest.approx(1 / 3600)


def test_cutoff_is_kyiv_midnight_after_last_day():
    # Kyiv is UTC+3 in August, UTC+2 in January
    assert load.cutoff_instant("2026-08-28") == pd.Timestamp("2026-08-28 21:00", tz="UTC")
    assert load.cutoff_instant("2026-01-15") == pd.Timestamp("2026-01-15 22:00", tz="UTC")


def test_rows_after_cutoff_dropped_and_straddlers_clipped():
    raw = _raw([
        _row("2026-08-28 20:00:00", "2026-08-28 20:30:00"),  # entirely before
        _row("2026-08-28 20:50:00", "2026-09-02 13:00:00"),  # straddles the cutoff
        _row("2026-08-28 21:00:00", "2026-08-28 22:00:00"),  # starts at the cutoff
        _row("2026-09-03 10:00:00", "2026-09-03 11:00:00"),  # after the feed resumed
    ])
    cleaned = load.clean(raw, last_day="2026-08-28")
    cutoff = pd.Timestamp("2026-08-28 21:00", tz="UTC")
    assert len(cleaned) == 2
    assert (cleaned["finished_at"] <= cutoff).all()
    assert cleaned["clipped_at_cutoff"].tolist() == [False, True]
    assert cleaned["finished_at"].iloc[1] == cutoff


def test_long_alerts_are_flagged_not_dropped():
    raw = _raw([
        _row("2025-01-01 00:00:00", "2025-01-01 23:00:00"),
        _row("2025-01-02 00:00:00", "2025-01-04 00:00:00"),
        _row("2025-02-01 00:00:00", "2025-03-01 00:00:00"),
    ])
    cleaned = load.clean(raw)
    assert len(cleaned) == 3
    assert cleaned["long_alert"].tolist() == [False, True, True]
    assert cleaned["standing_alert"].tolist() == [False, False, True]


def test_output_is_sorted_and_utc():
    raw = _raw([
        _row("2025-01-02 10:00:00", "2025-01-02 11:00:00"),
        _row("2025-01-01 10:00:00", "2025-01-01 11:00:00"),
    ])
    cleaned = load.clean(raw)
    assert cleaned["started_at"].is_monotonic_increasing
    assert str(cleaned["started_at"].dt.tz) == "UTC"


def test_non_utc_timestamp_is_rejected():
    raw = _raw([("Kyiv City", "", "", "oblast", "2025-01-01 10:00:00+02:00", "2025-01-01 11:00:00+00:00")])
    with pytest.raises(ValueError):
        load.clean(raw)


def test_unknown_level_is_rejected():
    raw = _raw([("Kyiv City", "", "", "city", "2025-01-01 10:00:00+00:00", "2025-01-01 11:00:00+00:00")])
    with pytest.raises(ValueError):
        load.clean(raw)


def test_find_feed_break_reports_day_before_the_partial_day():
    starts = pd.to_datetime(
        ["2026-08-28 12:00", "2026-08-28 23:30", "2026-08-29 04:55", "2026-09-02 13:48", "2026-09-02 14:00"],
        utc=True,
    ).to_series()
    brk = load.find_feed_break(starts)
    assert brk["last_start_before"] == pd.Timestamp("2026-08-29 04:55", tz="UTC")
    assert brk["first_start_after"] == pd.Timestamp("2026-09-02 13:48", tz="UTC")
    assert brk["last_complete_day"] == "2026-08-28"


# ---- checks against the real pinned file (skipped if it has not been fetched) ----

needs_data = pytest.mark.skipif(not load.RAW_PATH.exists(), reason="run scripts/fetch_data.py first")


@pytest.fixture(scope="module")
def report():
    return load.validate_raw(load.read_raw())


@needs_data
def test_raw_file_matches_profiled_facts(report):
    assert report["rows"] == 299_275
    assert report["first_start"].strftime("%Y-%m-%d") == "2022-03-15"
    assert report["last_start"].strftime("%Y-%m-%d") == "2026-09-07"
    assert report["sorted_by_start"] is False
    assert report["sources"] == ["official"]
    assert report["duplicate_rows"] == 113_845
    assert report["max_copies_of_a_row"] == 2
    assert report["nonpositive_duration_raw"] == 4
    assert report["over_24h_raw"] == 1_318
    assert report["max_duration_h"] == pytest.approx(14_497.3, abs=0.1)
    assert report["level_fields_consistent"]
    assert report["kyiv_city_levels"] == ["oblast"]


@needs_data
def test_feed_break_and_last_complete_day(report):
    assert set(report["rows_on_gap_days_utc"].values()) == {0}
    assert report["rows_on_2026_08_29_utc"] == 88
    brk = report["feed_break"]
    assert brk["gap"] > pd.Timedelta(days=4)
    # nothing else in four and a half years comes close, so this is the break
    assert brk["second_longest_gap"] < pd.Timedelta(hours=24)
    assert brk["last_complete_day"] == load.LAST_COMPLETE_DAY


@needs_data
def test_long_alerts_do_not_touch_kyiv_city_or_lvivska(report):
    longest = report["max_duration_h_by_region"]
    assert longest["Kyiv City"] < 12
    assert longest["Lvivska oblast"] < 12
    assert longest["Kharkivska oblast"] > 10_000


@needs_data
def test_cleaned_data_invariants():
    cleaned = load.load()
    assert not cleaned.duplicated(subset=load.COLUMNS).any()
    assert (cleaned["finished_at"] > cleaned["started_at"]).all()
    assert cleaned["finished_at"].max() <= load.cutoff_instant()
    assert cleaned["started_at"].is_monotonic_increasing
