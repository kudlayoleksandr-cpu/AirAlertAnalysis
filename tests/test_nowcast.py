import numpy as np
import pandas as pd
import pytest

from src import nowcast

MODEL = {"oblasts": "ring", "window_days": 60}
BASELINE_C = {"oblasts": "none", "window_days": None}
P_COLUMNS = [f"p_{lead}" for lead in nowcast.LEADS]


def _kyiv(stamp):
    return pd.Timestamp(stamp, tz="Europe/Kyiv")


def _alerts(rows):
    """rows: (oblast, level, local start, local end)"""
    frame = pd.DataFrame(rows, columns=["oblast", "level", "started_at", "finished_at"])
    for column in ("started_at", "finished_at"):
        frame[column] = pd.DatetimeIndex([_kyiv(v) for v in frame[column]]).tz_convert("UTC")
    return frame


def _random_rows(rng, first_day, last_day):
    """Nights with a Kyivska raion alert that is usually followed by a Kyiv City alert, plus noise."""
    rows = []
    for day in pd.date_range(first_day, last_day, freq="D"):
        evening = day.tz_localize("Europe/Kyiv") + pd.Timedelta(hours=19)
        if rng.random() < 0.5:
            event = evening + pd.Timedelta(minutes=int(rng.integers(0, 9 * 60)))
            rows.append(("Kyivska oblast", "raion", event - pd.Timedelta(minutes=40), event + pd.Timedelta(minutes=80)))
            if rng.random() < 0.7:
                rows.append(("Kyiv City", "oblast", event, event + pd.Timedelta(minutes=int(rng.integers(30, 90)))))
        if rng.random() < 0.15:
            start = evening + pd.Timedelta(minutes=int(rng.integers(0, 9 * 60)))
            rows.append(("Kyiv City", "oblast", start, start + pd.Timedelta(minutes=int(rng.integers(10, 60)))))
        for oblast, level in (("Sumska oblast", "raion"), ("Lvivska oblast", "raion"), ("Kyivska oblast", "hromada")):
            if rng.random() < 0.6:
                start = day.tz_localize("Europe/Kyiv") + pd.Timedelta(minutes=int(rng.integers(0, 24 * 60)))
                rows.append((oblast, level, start, start + pd.Timedelta(minutes=int(rng.integers(20, 400)))))
    frame = pd.DataFrame(rows, columns=["oblast", "level", "started_at", "finished_at"])
    for column in ("started_at", "finished_at"):
        frame[column] = pd.DatetimeIndex(frame[column]).tz_convert("UTC")
    return frame.sort_values("started_at").reset_index(drop=True)


@pytest.fixture(scope="module")
def alerts():
    return _random_rows(np.random.default_rng(3), "2025-05-01", "2026-02-20")


def _corrupt_after(alerts, cutoff, seed=11):
    """Change everything that cannot be known at ``cutoff``.

    Alerts starting after the cutoff are replaced by unrelated ones, and alerts
    open at the cutoff get a different end time.
    """
    cutoff = pd.Timestamp(cutoff).tz_convert("UTC")
    rng = np.random.default_rng(seed)
    past = alerts[alerts["started_at"] <= cutoff].copy()
    still_open = past["finished_at"] > cutoff
    past.loc[still_open, "finished_at"] = [
        cutoff + pd.Timedelta(minutes=int(m)) for m in rng.integers(1, 3000, still_open.sum())]
    start_day = cutoff.tz_convert("Europe/Kyiv").tz_localize(None).normalize() + pd.Timedelta(days=1)
    future = _random_rows(rng, start_day, start_day + pd.Timedelta(days=20))
    future = future[future["started_at"] > cutoff]
    assert len(future) > 0
    return pd.concat([past, future], ignore_index=True)


# ---- what is known at the cutoff ----

def test_known_at_drops_future_alerts_and_hides_open_end_times():
    alerts = _alerts([
        ("Kyiv City", "oblast", "2026-01-10 18:00", "2026-01-10 18:30"),  # finished
        ("Kyiv City", "oblast", "2026-01-10 19:50", "2026-01-10 23:00"),  # open at the cutoff
        ("Kyiv City", "oblast", "2026-01-10 20:30", "2026-01-10 21:00"),  # not started yet
    ])
    cutoff = _kyiv("2026-01-10 20:00")
    visible = nowcast.known_at(alerts, cutoff)
    assert len(visible) == 2
    assert visible["open_at_cutoff"].tolist() == [False, True]
    assert visible["finished_at"].iloc[0] == _kyiv("2026-01-10 18:30")
    assert visible["finished_at"].iloc[1] == cutoff + nowcast.OPEN_HORIZON

    longer = alerts.copy()
    longer.loc[1, "finished_at"] = _kyiv("2026-01-12 05:00").tz_convert("UTC")
    pd.testing.assert_frame_equal(nowcast.known_at(longer, cutoff), visible)


def test_naive_times_are_rejected(alerts):
    with pytest.raises(ValueError):
        nowcast.known_at(alerts, "2026-01-10 20:00")


# ---- target, own state and features ----

@pytest.fixture(scope="module")
def small_table():
    alerts = _alerts([
        ("Kyiv City", "oblast", "2026-01-10 19:20", "2026-01-10 19:50"),
        ("Kyiv City", "oblast", "2026-01-10 21:00", "2026-01-10 21:10"),
        ("Kyivska oblast", "raion", "2026-01-10 18:10", "2026-01-10 19:10"),
        ("Kyivska oblast", "raion", "2026-01-10 19:00", "2026-01-10 20:00"),   # overlaps: one spell
        ("Kyivska oblast", "oblast", "2026-01-10 22:00", "2026-01-10 22:20"),
        ("Kyivska oblast", "hromada", "2026-01-10 23:00", "2026-01-10 23:59"),  # ignored
    ])
    return nowcast.build_table(alerts, _kyiv("2026-01-11 06:00"), start="2026-01-10")


def _at(table, stamp):
    return table.loc[_kyiv(stamp).tz_convert("UTC")]


def test_target_is_any_alert_in_the_next_l_minutes(small_table):
    row = _at(small_table, "2026-01-10 19:00")
    assert not row["kyiv_on"]
    assert [row[f"y_{lead}"] for lead in nowcast.LEADS] == [0, 1, 1, 1, 1]  # alert starts at 19:20
    assert _at(small_table, "2026-01-10 19:15")["y_15"] == 1
    assert _at(small_table, "2026-01-10 18:00")["y_60"] == 0
    assert _at(small_table, "2026-01-10 18:00")["y_120"] == 1
    assert _at(small_table, "2026-01-10 22:00")[[f"y_{lead}" for lead in nowcast.LEADS]].sum() == 0


def test_target_window_is_half_open(small_table):
    # an alert that starts exactly at t + L is outside [t, t + L)
    assert _at(small_table, "2026-01-10 20:45")["y_15"] == 0
    assert _at(small_table, "2026-01-10 20:45")["y_30"] == 1
    # an alert that ended before t does not count
    assert _at(small_table, "2026-01-10 20:00")["y_15"] == 0


def test_under_alert_at_issue_time_means_target_is_one(small_table):
    for stamp in ("2026-01-10 19:30", "2026-01-10 19:45", "2026-01-10 21:00"):
        row = _at(small_table, stamp)
        assert row["kyiv_on"]
        assert all(row[f"y_{lead}"] == 1 for lead in nowcast.LEADS)
        assert all(row[f"rate_b_{lead}"] == 1 for lead in nowcast.LEADS)
    assert not _at(small_table, "2026-01-10 19:15")["kyiv_on"]
    assert not _at(small_table, "2026-01-10 20:00")["kyiv_on"]  # ended at 19:50


def test_minutes_since_current_alert_started(small_table):
    column = nowcast.feature_column("Kyivska oblast")
    assert _at(small_table, "2026-01-10 18:00")[column] == 0
    assert _at(small_table, "2026-01-10 18:15")[column] == 5
    assert _at(small_table, "2026-01-10 19:30")[column] == 80   # unbroken spell since 18:10
    assert _at(small_table, "2026-01-10 20:00")[column] == 0    # ended exactly now
    assert _at(small_table, "2026-01-10 22:15")[column] == 15   # oblast-level row counts
    assert _at(small_table, "2026-01-10 23:30")[column] == 0    # hromada-level row does not
    assert _at(small_table, "2026-01-10 19:30")[nowcast.feature_column("Sumska oblast")] == 0


def test_outcomes_not_yet_known_at_the_cutoff_are_missing():
    alerts = _alerts([("Kyiv City", "oblast", "2026-01-10 19:20", "2026-01-10 19:50")])
    cutoff = _kyiv("2026-01-10 20:00")
    table = nowcast.build_table(alerts, cutoff, start="2026-01-10")
    assert table.index[-1] == cutoff
    assert _at(table, "2026-01-10 19:45")["y_15"] == 1          # window ends exactly at the cutoff
    assert np.isnan(_at(table, "2026-01-10 19:45")["y_30"])
    assert np.isnan(_at(table, "2026-01-10 20:00")["y_15"])
    assert _at(table, "2026-01-10 17:00")["y_180"] == 1
    assert np.isnan(_at(table, "2026-01-10 17:15")["y_180"])


@pytest.mark.parametrize("evening, expected", [("2026-01-10", 48), ("2026-03-28", 44), ("2025-10-25", 52)])
def test_issue_times_per_night_including_clock_changes(evening, expected):
    alerts = _alerts([("Kyiv City", "oblast", "2025-10-20 19:20", "2025-10-20 19:50")])
    day = pd.Timestamp(evening)
    table = nowcast.build_table(alerts, _kyiv(str(day + pd.Timedelta(days=2))), start=str(day - pd.Timedelta(days=1)))
    night = table[table["is_night"] & (table["night"] == day)]
    assert len(night) == expected
    local = night.index.tz_convert("Europe/Kyiv")
    assert local[0] == _kyiv(f"{evening} 18:00")
    assert (local[-1].hour, local[-1].minute) == (5, 45)
    # consecutive issue times are always 15 elapsed minutes apart, whatever the clocks do
    assert (np.diff(night.index) == np.timedelta64(15, "m")).all()


def test_trailing_hour_of_day_rates():
    # the same alert every evening, 19:55-20:25, for 120 days
    days = pd.date_range("2025-09-01", periods=120, freq="D")
    alerts = _alerts([("Kyiv City", "oblast", f"{d:%Y-%m-%d} 19:55", f"{d:%Y-%m-%d} 20:25") for d in days])
    table = nowcast.build_table(alerts, _kyiv("2025-12-20 12:00"), start="2025-09-01")
    # hour 19: of 19:00, 19:15, 19:30, 19:45 only the last has an alert within 15 minutes
    row = _at(table, "2025-12-15 19:00")
    assert row["rate_a_15"] == pytest.approx(0.25)
    assert row["rate_b_15"] == pytest.approx(0.25)
    # hour 20: 20:00 and 20:15 are under alert, 20:30 and 20:45 are quiet
    assert _at(table, "2025-12-15 20:00")["rate_a_15"] == pytest.approx(0.5)
    # no alert ever starts within 15 minutes of a quiet issue time in hour 20
    assert _at(table, "2025-12-15 20:30")["rate_b_15"] == pytest.approx(nowcast.RATE_BOUNDS[0])
    # by 20:30 tonight's 20:00 and 20:15 outcomes are known and have replaced the two oldest
    assert _at(table, "2025-12-15 20:30")["rate_a_15"] == pytest.approx(180 / 358)
    # too little history: no rate
    assert np.isnan(_at(table, "2025-09-03 19:00")["rate_a_15"])


def test_trailing_rate_excludes_outcomes_not_yet_known():
    # one alert only, tonight at 19:05: at 19:15 the 19:00 outcome for L=180 is not yet known
    days = pd.date_range("2025-09-01", periods=100, freq="D")
    quiet = [("Lvivska oblast", "raion", f"{d:%Y-%m-%d} 10:00", f"{d:%Y-%m-%d} 10:30") for d in days]
    alerts = _alerts(quiet + [("Kyiv City", "oblast", "2025-12-01 19:05", "2025-12-01 19:10")])
    table = nowcast.build_table(alerts, _kyiv("2025-12-03 12:00"), start="2025-09-01")
    low = nowcast.RATE_BOUNDS[0]
    assert _at(table, "2025-12-01 19:00")["y_180"] == 1
    assert _at(table, "2025-12-01 19:15")["rate_a_180"] == low   # 19:00 + 180 min is still in the future
    assert _at(table, "2025-12-01 19:15")["rate_a_15"] > low     # 19:00 + 15 min is known
    assert _at(table, "2025-12-02 19:00")["rate_a_180"] > low    # a day later it is known


# ---- leakage: tables ----

def _assert_same_up_to(full, part, cutoff):
    """``part`` was built at ``cutoff``; ``full`` at a later time or from altered future data."""
    head = full.loc[:cutoff]
    assert head.index.equals(part.index)
    pd.testing.assert_frame_equal(head[_feature_columns(part)], part[_feature_columns(part)])
    for lead in nowcast.LEADS:
        known = part[f"y_{lead}"].notna()
        pd.testing.assert_series_equal(head.loc[known, f"y_{lead}"], part.loc[known, f"y_{lead}"])
        assert (part.index[~known] + pd.Timedelta(minutes=lead) > cutoff).all()
        pd.testing.assert_series_equal(head[f"rate_a_{lead}"], part[f"rate_a_{lead}"])
        pd.testing.assert_series_equal(head[f"rate_b_{lead}"], part[f"rate_b_{lead}"])


def _feature_columns(table):
    return ["hour", "night", "is_night", "kyiv_on"] + [c for c in table.columns if c.startswith("mins_")]


@pytest.mark.parametrize("cutoff", ["2026-01-20 21:15", "2026-01-25 02:30", "2026-02-01 18:00"])
def test_table_is_unchanged_by_anything_after_the_cutoff(alerts, cutoff):
    cutoff = _kyiv(cutoff).tz_convert("UTC")
    at_cutoff = nowcast.build_table(alerts, cutoff)
    pd.testing.assert_frame_equal(nowcast.build_table(_corrupt_after(alerts, cutoff), cutoff), at_cutoff)
    # and a table built later agrees with it on everything that was known then
    _assert_same_up_to(nowcast.build_table(alerts, cutoff + pd.Timedelta(days=9)), at_cutoff, cutoff)
    _assert_same_up_to(
        nowcast.build_table(_corrupt_after(alerts, cutoff), cutoff + pd.Timedelta(days=9)), at_cutoff, cutoff)


def test_corruption_helper_really_changes_the_future(alerts):
    cutoff = _kyiv("2026-01-25 02:30").tz_convert("UTC")
    later = cutoff + pd.Timedelta(days=9)
    original = nowcast.build_table(alerts, later).loc[cutoff + pd.Timedelta(hours=1):]
    altered = nowcast.build_table(_corrupt_after(alerts, cutoff), later).loc[cutoff + pd.Timedelta(hours=1):]
    assert not original["kyiv_on"].equals(altered["kyiv_on"])


# ---- leakage: models and forecasts ----

@pytest.fixture(scope="module")
def table(alerts):
    return nowcast.build_table(alerts, _kyiv("2026-02-20 12:00"))


@pytest.mark.parametrize("spec", [MODEL, BASELINE_C])
def test_training_rows_respect_the_fitting_cutoff(table, spec):
    fit_cutoff = nowcast.evening_start("2026-01-25")
    for lead in nowcast.LEADS:
        rows = nowcast.training_rows(table, fit_cutoff, lead, spec)
        assert len(rows) > 100
        assert (rows.index + pd.Timedelta(minutes=lead) <= fit_cutoff).all()
        assert (rows.index >= nowcast.SWITCH_DONE).all()
        assert rows["is_night"].all() and not rows["kyiv_on"].any()
        if spec["window_days"] is not None:
            assert (rows.index >= fit_cutoff - pd.Timedelta(days=spec["window_days"])).all()


@pytest.mark.parametrize("spec", [MODEL, BASELINE_C])
@pytest.mark.parametrize("issue", ["2026-01-25 18:00", "2026-01-25 21:15", "2026-01-26 03:45"])
def test_forecast_ignores_everything_after_the_issue_time(alerts, table, spec, issue):
    issue = _kyiv(issue)
    original = nowcast.forecast_at(alerts, issue, spec)
    assert original[P_COLUMNS].between(0, 1).all()
    # replacing everything after the issue time changes nothing
    pd.testing.assert_series_equal(nowcast.forecast_at(_corrupt_after(alerts, issue), issue, spec), original)
    # nor does removing it
    pd.testing.assert_series_equal(nowcast.forecast_at(alerts[alerts["started_at"] <= issue.tz_convert("UTC")], issue, spec), original)
    # and the rolling path, working from a table built later, gives the same numbers
    rolled = nowcast.night_forecasts(table, "2026-01-25", spec).loc[issue.tz_convert("UTC")]
    pd.testing.assert_series_equal(rolled, original, check_names=False)


def test_forecast_does_respond_to_what_is_known_at_the_issue_time(alerts):
    spec = MODEL
    issue = _kyiv("2026-01-27 21:15")
    quiet = alerts[~((alerts["started_at"] > issue.tz_convert("UTC") - pd.Timedelta(hours=6))
                     & (alerts["started_at"] <= issue.tz_convert("UTC")))]
    neighbour = _alerts([("Kyivska oblast", "raion", "2026-01-27 20:40", "2026-01-27 23:00")])
    with_neighbour = pd.concat([quiet, neighbour], ignore_index=True)
    before = nowcast.forecast_at(quiet, issue, spec)
    after = nowcast.forecast_at(with_neighbour, issue, spec)
    # in the synthetic data a Kyivska alert is usually followed by a Kyiv City one about 40 minutes later
    assert after["p_30"] > before["p_30"] + 0.1


def test_baseline_c_responds_to_kyiv_citys_own_recent_alert_but_not_to_neighbours(alerts):
    issue = _kyiv("2026-01-27 21:15")
    utc = issue.tz_convert("UTC")
    quiet = alerts[~((alerts["started_at"] > utc - pd.Timedelta(hours=12)) & (alerts["started_at"] <= utc))]
    before = nowcast.forecast_at(quiet, issue, BASELINE_C)
    neighbour = _alerts([("Kyivska oblast", "raion", "2026-01-27 20:40", "2026-01-27 23:00")])
    pd.testing.assert_series_equal(
        nowcast.forecast_at(pd.concat([quiet, neighbour], ignore_index=True), issue, BASELINE_C), before)
    own = _alerts([("Kyiv City", "oblast", "2026-01-27 20:30", "2026-01-27 21:00")])
    after = nowcast.forecast_at(pd.concat([quiet, own], ignore_index=True), issue, BASELINE_C)
    assert not np.allclose(after[P_COLUMNS].to_numpy(float), before[P_COLUMNS].to_numpy(float))


def test_under_alert_gives_probability_one(alerts):
    issue = _kyiv("2026-01-27 21:15")
    own = _alerts([("Kyiv City", "oblast", "2026-01-27 21:00", "2026-01-27 22:00")])
    result = nowcast.forecast_at(pd.concat([alerts, own], ignore_index=True), issue, MODEL)
    assert (result[P_COLUMNS] == 1).all()


def test_forecast_at_rejects_daytime_and_off_grid_times(alerts):
    with pytest.raises(ValueError):
        nowcast.forecast_at(alerts, _kyiv("2026-01-27 12:00"), MODEL)
    with pytest.raises(ValueError):
        nowcast.forecast_at(alerts, _kyiv("2026-01-27 21:07"), MODEL)


@pytest.mark.parametrize("spec", [MODEL, BASELINE_C])
def test_rolling_forecasts_up_to_a_time_ignore_later_data(alerts, spec):
    cutoff = _kyiv("2026-01-27 23:30").tz_convert("UTC")
    later = _kyiv("2026-02-05 12:00")
    first, last = "2026-01-25", "2026-01-30"
    original = nowcast.rolling_forecasts(nowcast.build_table(alerts, later), first, last, spec)
    altered = nowcast.rolling_forecasts(nowcast.build_table(_corrupt_after(alerts, cutoff), later), first, last, spec)
    pd.testing.assert_frame_equal(original.loc[:cutoff], altered.loc[:cutoff])
    assert not original.loc[cutoff + pd.Timedelta(days=1):].equals(altered.loc[cutoff + pd.Timedelta(days=1):])


def test_selection_ignores_everything_after_the_cutoff(alerts):
    cutoff = nowcast.evening_start("2026-02-01")
    kwargs = {
        "model_candidates": [{"oblasts": "kyivska", "window_days": 60}, {"oblasts": "ring", "window_days": None}],
        "baseline_c_candidates": [{"oblasts": "none", "window_days": 60}, {"oblasts": "none", "window_days": None}],
        "nights": 6,
    }
    chosen = nowcast.select(alerts, cutoff, **kwargs)
    again = nowcast.select(_corrupt_after(alerts, cutoff), cutoff, **kwargs)
    assert chosen["model"] == again["model"] and chosen["baseline_c"] == again["baseline_c"]
    pd.testing.assert_frame_equal(chosen["model_table"], again["model_table"])
    pd.testing.assert_frame_equal(chosen["baseline_c_table"], again["baseline_c_table"])
    pd.testing.assert_frame_equal(chosen["validation_frame"], again["validation_frame"])
    assert chosen["validation_last_evening"] == pd.Timestamp("2026-01-31")
    assert chosen["validation_first_evening"] == pd.Timestamp("2026-01-26")
    # every validation outcome was known at the cutoff
    frame = chosen["validation_frame"]
    assert frame[[f"y_{lead}" for lead in nowcast.LEADS]].notna().all().all()
    assert frame.index.max() + pd.Timedelta(minutes=180) <= cutoff
    # the winner is the candidate with the lowest validation score
    best = chosen["model_table"]["val_brier"].idxmin()
    assert chosen["model"] == kwargs["model_candidates"][best]


@pytest.fixture(scope="module")
def scored(table):
    return nowcast.scored_frame(table, {
        "c": nowcast.rolling_forecasts(table, "2026-01-20", "2026-02-10", BASELINE_C),
        "model": nowcast.rolling_forecasts(table, "2026-01-20", "2026-02-10", MODEL),
    })


def test_model_learns_the_planted_neighbour_signal(scored):
    # sanity check of the whole pipeline: the synthetic neighbour signal should give real skill
    scores = nowcast.brier_table(scored, ["a", "b", "c", "model"], starts_only=True)
    assert scores.loc[60, "skill_model_vs_b"] > 0.05
    assert scores.loc[60, "skill_model_vs_c"] > 0.05
    assert {"skill_b_vs_a", "skill_c_vs_a", "skill_c_vs_b", "skill_model_vs_a"} <= set(scores.columns)


def test_scores_on_all_issue_times_include_those_under_alert(scored):
    starts = nowcast.brier_table(scored, ["a", "b", "c", "model"], starts_only=True)
    everything = nowcast.brier_table(scored, ["a", "b", "c", "model"], starts_only=False)
    assert (everything["n"] > starts["n"]).all()
    assert (everything["n"] - starts["n"] == scored["kyiv_on"].sum()).all()


# ---- own gap ----

def test_minutes_since_kyiv_citys_last_alert_ended(small_table):
    assert _at(small_table, "2026-01-10 19:00")["own_gap"] == nowcast.OWN_GAP_CAP   # none before
    assert _at(small_table, "2026-01-10 19:30")["own_gap"] == 0                     # under alert
    assert _at(small_table, "2026-01-10 20:00")["own_gap"] == 10                    # ended 19:50
    assert _at(small_table, "2026-01-10 20:45")["own_gap"] == 55
    assert _at(small_table, "2026-01-10 21:15")["own_gap"] == 5                     # second alert ended 21:10


# ---- lead-lag ----

def test_lead_lag_finds_the_planted_lead(alerts):
    cutoff = _kyiv("2026-02-20 12:00")
    ccf = nowcast.lead_lag(alerts, cutoff, oblasts=("Kyivska oblast", "Lvivska oblast"), max_lag_minutes=180)
    assert ccf.index[0] == -180 and ccf.index[-1] == 180 and len(ccf) == 25
    # in the synthetic data Kyivska alerts start 40 minutes before Kyiv City's
    assert 15 <= ccf["Kyivska oblast"].idxmax() <= 60
    assert ccf["Kyivska oblast"].max() > 0.3
    assert ccf["Lvivska oblast"].abs().max() < 0.1                  # unrelated noise
    # and nothing after the cutoff is used
    pd.testing.assert_frame_equal(
        nowcast.lead_lag(_corrupt_after(alerts, cutoff), cutoff, oblasts=("Kyivska oblast", "Lvivska oblast"),
                         max_lag_minutes=180), ccf)


def test_lead_lag_removes_a_shared_daily_cycle():
    # two series that are both "on" at the same hours every night, and otherwise unrelated
    rng = np.random.default_rng(5)
    rows = []
    for day in pd.date_range("2025-11-10", periods=240, freq="D"):
        stamp = f"{day:%Y-%m-%d}"
        rows.append(("Kyiv City", "oblast", f"{stamp} 22:00", f"{stamp} 23:00"))
        rows.append(("Kyivska oblast", "raion", f"{stamp} 22:00", f"{stamp} 23:00"))
        for oblast, level in (("Kyiv City", "oblast"), ("Kyivska oblast", "raion")):
            start = day + pd.Timedelta(minutes=int(rng.integers(0, 20 * 60)))
            rows.append((oblast, level, str(start), str(start + pd.Timedelta(minutes=45))))
    alerts = _alerts(rows)
    ccf = nowcast.lead_lag(alerts, _kyiv("2026-07-08 12:00"), oblasts=("Kyivska oblast",), max_lag_minutes=120)
    assert ccf["Kyivska oblast"].abs().max() < 0.1
    # without removing the cycle the shared 22:00-23:00 hour would give a strong correlation at lag 0
    utc = alerts.copy()
    table = nowcast.build_table(utc, _kyiv("2026-07-08 12:00"), start="2025-11-10")
    raw = np.corrcoef(table["kyiv_on"], table[nowcast.feature_column("Kyivska oblast")] > 0)[0, 1]
    assert raw > 0.4


def test_lead_lag_of_constant_series_is_zero():
    alerts = _alerts([("Kyiv City", "oblast", "2025-11-10 22:00", "2025-11-10 23:00")])
    ccf = nowcast.lead_lag(alerts, _kyiv("2025-11-20 12:00"), oblasts=("Kyivska oblast",), max_lag_minutes=60)
    assert (ccf["Kyivska oblast"] == 0).all()


# ---- scoring ----

def test_brier_and_skill():
    y = np.array([1, 0, 0, 1])
    assert nowcast.brier(y, np.array([1, 0, 0, 1])) == 0
    assert nowcast.brier(y, np.full(4, 0.5)) == pytest.approx(0.25)
    assert nowcast.skill(y, np.array([0.9, 0.1, 0.1, 0.9]), np.full(4, 0.5)) == pytest.approx(1 - 0.01 / 0.25)
    assert nowcast.skill(y, np.full(4, 0.5), np.full(4, 0.5)) == 0


def test_skill_interval_contains_the_point_estimate(scored):
    value, low, high = nowcast.skill_interval(scored, 60, "model", "c")
    assert low <= value <= high
    assert value == pytest.approx(nowcast.brier_table(scored, ["a", "b", "c", "model"], True).loc[60, "skill_model_vs_c"])
