import numpy as np
import pandas as pd
import pytest

from src import blocks

B = list(blocks.BLOCK_STARTS)


def _kyiv(stamp):
    return pd.Timestamp(stamp, tz="Europe/Kyiv")


def _alerts(rows):
    """rows: (local start, local end) of Kyiv City alerts"""
    frame = pd.DataFrame(rows, columns=["started_at", "finished_at"])
    for column in ("started_at", "finished_at"):
        frame[column] = pd.DatetimeIndex([_kyiv(v) for v in frame[column]]).tz_convert("UTC")
    frame["oblast"], frame["level"] = "Kyiv City", "oblast"
    return frame


def _outcomes(values, first_day="2025-01-01"):
    """values: one row of four 0/1 block outcomes per day"""
    days = pd.date_range(first_day, periods=len(values), freq="D")
    table = pd.DataFrame(np.asarray(values, dtype=float), index=days, columns=B)
    table["on_at_midnight"] = False
    return table


@pytest.fixture(scope="module")
def random_outcomes():
    rng = np.random.default_rng(0)
    return _outcomes(rng.random((500, 4)) < [0.5, 0.2, 0.1, 0.3], first_day="2024-06-01")


# ---- outcomes from alerts ----

def test_block_outcomes_basic_and_boundaries():
    alerts = _alerts([
        ("2026-01-10 05:50", "2026-01-10 06:10"),   # crosses a block boundary: blocks 0 and 6
        ("2026-01-11 18:00", "2026-01-11 18:05"),   # starts exactly at a block start
        ("2026-01-12 11:00", "2026-01-12 12:00"),   # ends exactly at a block end: block 6 only
        ("2026-01-13 23:30", "2026-01-14 00:30"),   # crosses midnight
    ])
    table = blocks.block_outcomes(alerts, "2026-01-16", first_day="2026-01-09")
    assert table.index[0] == pd.Timestamp("2026-01-09") and table.index[-1] == pd.Timestamp("2026-01-15")
    assert table.loc["2026-01-09", B].tolist() == [0, 0, 0, 0]
    assert table.loc["2026-01-10", B].tolist() == [1, 1, 0, 0]
    assert table.loc["2026-01-11", B].tolist() == [0, 0, 0, 1]
    assert table.loc["2026-01-12", B].tolist() == [0, 1, 0, 0]
    assert table.loc["2026-01-13", B].tolist() == [0, 0, 0, 1]
    assert table.loc["2026-01-14", B].tolist() == [1, 0, 0, 0]
    assert table["on_at_midnight"].tolist() == [False, False, False, False, False, True, False]


def test_alert_spanning_several_blocks_and_days():
    alerts = _alerts([("2026-01-10 17:00", "2026-01-11 07:00")])
    table = blocks.block_outcomes(alerts, "2026-01-13", first_day="2026-01-10")
    assert table.loc["2026-01-10", B].tolist() == [0, 0, 1, 1]
    assert table.loc["2026-01-11", B].tolist() == [1, 1, 0, 0]
    assert table.loc["2026-01-12", B].tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize("day, inside, outside", [
    ("2025-03-30", "2025-03-30 05:30", "2025-03-30 06:30"),   # 23-hour day: 00-06 lasts 5 hours
    ("2025-10-26", "2025-10-26 05:30", "2025-10-26 06:30"),   # 25-hour day: 00-06 lasts 7 hours
])
def test_blocks_follow_the_wall_clock_on_clock_change_days(day, inside, outside):
    early = blocks.block_outcomes(_alerts([(inside, inside[:-2] + "45")]), pd.Timestamp(day) + pd.Timedelta(days=1), first_day=day)
    late = blocks.block_outcomes(_alerts([(outside, outside[:-2] + "45")]), pd.Timestamp(day) + pd.Timedelta(days=1), first_day=day)
    assert early.loc[day, B].tolist() == [1, 0, 0, 0]
    assert late.loc[day, B].tolist() == [0, 1, 0, 0]


def test_outcomes_use_only_what_is_known_at_midnight():
    alerts = _alerts([
        ("2026-01-10 20:00", "2026-01-10 21:00"),
        ("2026-01-11 23:50", "2026-01-12 03:00"),   # still open at the midnight of the 12th
        ("2026-01-12 10:00", "2026-01-12 11:00"),   # not started yet
    ])
    table = blocks.block_outcomes(alerts, "2026-01-12", first_day="2026-01-10")
    altered = alerts.copy()
    altered.loc[1, "finished_at"] = _kyiv("2026-01-14 09:00").tz_convert("UTC")
    altered = pd.concat([altered.iloc[:2], _alerts([("2026-01-12 00:30", "2026-01-12 23:00")])], ignore_index=True)
    pd.testing.assert_frame_equal(blocks.block_outcomes(altered, "2026-01-12", first_day="2026-01-10"), table)
    assert table.index[-1] == pd.Timestamp("2026-01-11")
    assert table.loc["2026-01-11", B].tolist() == [0, 0, 0, 1]
    # the state at the issue instant itself is known, and is not changed by the alert's later end
    assert blocks.on_at_midnight(alerts, "2026-01-12") is True
    assert blocks.on_at_midnight(altered, "2026-01-12") is True
    assert blocks.on_at_midnight(alerts, "2026-01-11") is False


# ---- forecasts ----

def test_all_history_baseline_is_the_plain_block_rate():
    outcomes = _outcomes([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
    p = blocks.forecast_blocks(outcomes, "2025-01-05", blocks.BASELINE)
    assert p.tolist() == pytest.approx([0.75, 0.25, 0.0, 0.25])


def test_trailing_rate_is_shrunk_toward_the_all_history_rate():
    # 20 quiet days, then 7 days with an alert in block 0 only
    outcomes = _outcomes([[0, 0, 0, 0]] * 20 + [[1, 0, 0, 0]] * 7)
    p = blocks.forecast_blocks(outcomes, "2025-01-28", {"window": 7, "last_year": False})
    history = 7 / 27
    assert p[0] == pytest.approx((7 + blocks.PRIOR_DAYS * history) / (7 + blocks.PRIOR_DAYS))
    assert 0 < p[0] < 1                      # never exactly 0 or 1 when history is mixed
    assert p[6] == 0                          # a block that has never had an alert stays at 0
    wide = blocks.forecast_blocks(outcomes, "2025-01-28", {"window": 14, "last_year": False})
    assert wide[0] == pytest.approx((7 + blocks.PRIOR_DAYS * history) / (14 + blocks.PRIOR_DAYS))


def test_last_year_term_is_a_fixed_blend():
    days = 400
    values = np.zeros((days, 4))
    values[:, 1] = 1                                  # block 6: an alert every day
    outcomes = _outcomes(values, first_day="2025-01-01")
    issue = pd.Timestamp("2025-01-01") + pd.Timedelta(days=days)   # 2026-02-05
    centre = issue - pd.Timedelta(days=365)
    lo, hi = centre - pd.Timedelta(days=15), centre + pd.Timedelta(days=14)
    outcomes.loc[lo:hi, 0] = 1                        # block 0: alerts only in last year's 30-day window
    history = outcomes[0].mean()
    assert outcomes.loc[lo:hi].shape[0] == 30
    trailing = (0 + blocks.PRIOR_DAYS * history) / (14 + blocks.PRIOR_DAYS)
    last_year = (30 + blocks.PRIOR_DAYS * history) / (30 + blocks.PRIOR_DAYS)
    p = blocks.forecast_blocks(outcomes, issue, {"window": 14, "last_year": True})
    assert p[0] == pytest.approx(0.5 * trailing + 0.5 * last_year)
    assert p[6] == pytest.approx(1.0)
    without = blocks.forecast_blocks(outcomes, issue, {"window": 14, "last_year": False})
    assert without[0] == pytest.approx(trailing)


def test_last_year_term_with_no_data_a_year_ago_falls_back_to_history():
    outcomes = _outcomes([[1, 0, 0, 0], [0, 0, 0, 0]] * 30)
    p = blocks.forecast_blocks(outcomes, "2025-03-02", {"window": 14, "last_year": True})
    plain = blocks.forecast_blocks(outcomes, "2025-03-02", {"window": 14, "last_year": False})
    assert p[0] == pytest.approx(0.5 * plain[0] + 0.5 * 0.5)


ALL_SPECS = [blocks.BASELINE, blocks.REFERENCE] + blocks.CANDIDATES


def test_candidates_are_the_eight_specified():
    assert len(blocks.CANDIDATES) == 8
    assert sorted({c["window"] for c in blocks.CANDIDATES}) == [7, 14, 21, 30]
    assert sum(c["last_year"] for c in blocks.CANDIDATES) == 4
    assert len({blocks.label(spec) for spec in ALL_SPECS}) == 10


# ---- leakage ----

@pytest.mark.parametrize("spec", ALL_SPECS, ids=blocks.label)
def test_forecast_ignores_the_issue_day_and_everything_after(random_outcomes, spec):
    issue = random_outcomes.index[450]
    original = blocks.forecast_blocks(random_outcomes, issue, spec)
    altered = random_outcomes.copy()
    altered.loc[issue:, B] = 1 - altered.loc[issue:, B]
    pd.testing.assert_series_equal(blocks.forecast_blocks(altered, issue, spec), original)
    pd.testing.assert_series_equal(blocks.forecast_blocks(random_outcomes.loc[:issue - pd.Timedelta(days=1)], issue, spec), original)
    assert original.between(0, 1).all()


@pytest.mark.parametrize("spec", ALL_SPECS, ids=blocks.label)
def test_forecast_does_depend_on_the_day_before(random_outcomes, spec):
    issue = random_outcomes.index[450]
    changed = random_outcomes.copy()
    changed.loc[issue - pd.Timedelta(days=1), B] = 1 - changed.loc[issue - pd.Timedelta(days=1), B]
    assert not np.allclose(blocks.forecast_blocks(changed, issue, spec), blocks.forecast_blocks(random_outcomes, issue, spec))


def test_rolling_forecast_for_day_d_ignores_day_d_and_later(random_outcomes):
    first, last = random_outcomes.index[440], random_outcomes.index[460]
    target = random_outcomes.index[450]
    altered = random_outcomes.copy()
    altered.loc[target:, B] = 1 - altered.loc[target:, B]
    for spec in ALL_SPECS:
        original = blocks.rolling_forecasts(random_outcomes, first, last, spec)
        changed = blocks.rolling_forecasts(altered, first, last, spec)
        pd.testing.assert_frame_equal(original.loc[:target], changed.loc[:target])
        assert not original.loc[target + pd.Timedelta(days=1):].equals(changed.loc[target + pd.Timedelta(days=1):])


def test_selection_ignores_the_cutoff_day_and_everything_after(random_outcomes):
    cutoff = random_outcomes.index[400]
    chosen, table = blocks.select(random_outcomes, cutoff, days=60)
    altered = random_outcomes.copy()
    altered.loc[cutoff:, B] = 1 - altered.loc[cutoff:, B]
    chosen2, table2 = blocks.select(altered, cutoff, days=60)
    assert chosen == chosen2
    pd.testing.assert_frame_equal(table, table2)
    assert len(table) == 10
    candidate_rows = table.iloc[2:]
    assert blocks.label(chosen) == candidate_rows.loc[candidate_rows["val_brier"].idxmin(), "method"]


def test_gaps_and_missing_history_are_rejected(random_outcomes):
    issue = random_outcomes.index[450]
    with pytest.raises(ValueError):
        blocks.forecast_blocks(random_outcomes.drop(random_outcomes.index[430]), issue, blocks.BASELINE)
    with pytest.raises(ValueError):
        blocks.forecast_blocks(random_outcomes, random_outcomes.index[0], blocks.BASELINE)
    with pytest.raises(ValueError):   # the issue day is too far past the last known day
        blocks.forecast_blocks(random_outcomes, random_outcomes.index[-1] + pd.Timedelta(days=3), blocks.BASELINE)


# ---- scoring ----

def test_evaluation_frame_and_scores(random_outcomes):
    first, last = random_outcomes.index[440], random_outcomes.index[469]
    specs = {blocks.label(spec): spec for spec in ALL_SPECS}
    frame = blocks.evaluation_frame(random_outcomes, first, last, specs)
    assert len(frame) == 30 * 4
    assert frame["block"].tolist()[:4] == B
    day = random_outcomes.index[445]
    row = frame[(frame["day"] == day) & (frame["block"] == 6)].iloc[0]
    assert row["y"] == random_outcomes.loc[day, 6]
    assert row["trailing 14d"] == pytest.approx(blocks.forecast_blocks(random_outcomes, day, {"window": 14, "last_year": False})[6])

    scores = blocks.score_table(frame, list(specs))
    assert scores.loc["all history", "skill_vs_baseline"] == 0
    assert scores.loc["trailing 90d", "skill_vs_reference"] == 0
    expected = np.mean((frame["trailing 7d"] - frame["y"]) ** 2)
    assert scores.loc["trailing 7d", "brier"] == pytest.approx(expected)

    value, low, high = blocks.skill_interval(frame, "trailing 30d", "all history")
    assert low <= value <= high
    assert value == pytest.approx(scores.loc["trailing 30d", "skill_vs_baseline"])
