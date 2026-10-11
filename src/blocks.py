"""Forecast v3: day-ahead probability of an alert in each 6-hour block, Kyiv City.

At midnight (Europe/Kyiv) starting day D, give the probability that Kyiv City
is under alert at some instant in each of 00-06, 06-12, 12-18 and 18-24 of D.
Every candidate is a rate computed from past days; nothing is fitted.

Leakage rule: ``block_outcomes`` builds its table from
``nowcast.known_at(alerts, midnight)``, and every forecasting or selecting
function takes the outcomes table and an issue day and uses only rows for
days before it.
"""
import numpy as np
import pandas as pd

from src.load import KYIV_TZ
from src.nowcast import known_at
from src.series import FIRST_COMPLETE_DAY, covered_hours_before, merge_intervals

TARGET = "Kyiv City"
BLOCK_STARTS = (0, 6, 12, 18)  # wall-clock hours, Europe/Kyiv
BLOCK_HOURS = 6
PRIOR_DAYS = 7  # shrinkage: as if this many extra days at the all-history rate had been seen
LAST_YEAR_HALF_WIDTH = 15  # "same period last year" = 30 days centred on the date a year earlier
BLEND = 0.5  # weight of the trailing rate when the last-year term is added
VALIDATION_DAYS = 90
FIRST_TEST_DAY = "2026-05-31"
LAST_TEST_DAY = "2026-08-28"

BASELINE = {"window": None, "last_year": False}   # all-history block rate
REFERENCE = {"window": 90, "last_year": False}    # trailing 90 days, for interpretation only
CANDIDATES = [
    {"window": window, "last_year": last_year}
    for last_year in (False, True)
    for window in (7, 14, 21, 30)
]


def label(spec: dict) -> str:
    if spec["window"] is None:
        return "all history"
    return f"trailing {spec['window']}d" + (" + last year" if spec["last_year"] else "")


def _midnight(day) -> pd.Timestamp:
    return pd.Timestamp(day).normalize().tz_localize(None).tz_localize(KYIV_TZ)


def block_outcomes(alerts: pd.DataFrame, issue_day, first_day=FIRST_COMPLETE_DAY) -> pd.DataFrame:
    """Outcomes known at midnight starting ``issue_day``.

    One row per day from ``first_day`` to the day before ``issue_day``, with a
    0/1 column per block (named by its starting hour) and ``on_at_midnight``:
    whether Kyiv City was under alert at 00:00 of that day.
    """
    cutoff = _midnight(issue_day)
    visible = known_at(alerts, cutoff)
    own = visible[visible["oblast"] == TARGET]
    kyiv = merge_intervals(own["started_at"], own["finished_at"])
    days = pd.date_range(first_day, pd.Timestamp(issue_day).normalize() - pd.Timedelta(days=1), freq="D")
    table = pd.DataFrame(index=days)
    table.index.name = "day"
    for hour in BLOCK_STARTS:
        # wall-clock block edges: 00, 06, 12, 18 and 24 always exist, whatever the clocks do
        start = (days + pd.Timedelta(hours=hour)).tz_localize(KYIV_TZ)
        end = (days + pd.Timedelta(hours=hour + BLOCK_HOURS)).tz_localize(KYIV_TZ)
        covered = covered_hours_before(kyiv, end) - covered_hours_before(kyiv, start)
        table[hour] = (covered > 1e-9).astype(float)
    table["on_at_midnight"] = _active_at(kyiv, days.tz_localize(KYIV_TZ))
    return table


def _active_at(merged: pd.DataFrame, instants: pd.DatetimeIndex) -> np.ndarray:
    start = pd.DatetimeIndex(merged["started_at"]).as_unit("ns").asi8
    finish = pd.DatetimeIndex(merged["finished_at"]).as_unit("ns").asi8
    at = instants.tz_convert("UTC").as_unit("ns").asi8
    if len(start) == 0:
        return np.zeros(len(at), dtype=bool)
    k = np.searchsorted(start, at, side="right") - 1
    return (k >= 0) & (finish[np.maximum(k, 0)] > at)


def on_at_midnight(alerts: pd.DataFrame, day) -> bool:
    """Is Kyiv City under alert at 00:00 of ``day``? Uses only data known at that instant."""
    cutoff = _midnight(day)
    visible = known_at(alerts, cutoff)
    own = visible[visible["oblast"] == TARGET]
    kyiv = merge_intervals(own["started_at"], own["finished_at"])
    return bool(_active_at(kyiv, pd.DatetimeIndex([cutoff]))[0])


def past_outcomes(outcomes: pd.DataFrame, issue_day) -> pd.DataFrame:
    """Block columns for the days before ``issue_day``, checked to be gap-free."""
    issue_day = pd.Timestamp(issue_day).normalize()
    past = outcomes.loc[outcomes.index < issue_day, list(BLOCK_STARTS)]
    if past.empty or past.isna().any().any():
        raise ValueError("no complete outcomes before the issue day")
    expected = pd.date_range(past.index[0], issue_day - pd.Timedelta(days=1), freq="D")
    if not past.index.equals(expected):
        raise ValueError("outcomes must have one row per day up to the day before the issue day")
    return past


def _shrunk_rate(rows: pd.DataFrame, history_rate: pd.Series) -> pd.Series:
    """Block rate over ``rows``, pulled toward the all-history rate by PRIOR_DAYS pseudo-days."""
    return (rows.sum() + PRIOR_DAYS * history_rate) / (len(rows) + PRIOR_DAYS)


def forecast_blocks(outcomes: pd.DataFrame, issue_day, spec: dict) -> pd.Series:
    """Probability of an alert in each block of ``issue_day``, from days before it only."""
    issue_day = pd.Timestamp(issue_day).normalize()
    past = past_outcomes(outcomes, issue_day)
    history_rate = past.mean()
    if spec["window"] is None:
        return history_rate
    probability = _shrunk_rate(past.iloc[-spec["window"]:], history_rate)
    if spec["last_year"]:
        centre = issue_day - pd.Timedelta(days=365)
        around = past.loc[centre - pd.Timedelta(days=LAST_YEAR_HALF_WIDTH):
                          centre + pd.Timedelta(days=LAST_YEAR_HALF_WIDTH - 1)]
        probability = BLEND * probability + (1 - BLEND) * _shrunk_rate(around, history_rate)
    return probability


def rolling_forecasts(outcomes: pd.DataFrame, first_day, last_day, spec: dict) -> pd.DataFrame:
    """Forecasts for each day in first_day..last_day; the row for day D uses days before D only."""
    days = pd.date_range(first_day, last_day, freq="D")
    result = pd.DataFrame([forecast_blocks(outcomes, day, spec) for day in days], index=days)
    result.index.name = "day"
    return result


def brier(actual: pd.DataFrame, predicted: pd.DataFrame) -> float:
    return float(((predicted - actual) ** 2).to_numpy().mean())


def select(outcomes: pd.DataFrame, cutoff_day, candidates=None, days=VALIDATION_DAYS):
    """Pick the candidate with the lowest Brier score on the ``days`` days before ``cutoff_day``.

    Only rows for days before ``cutoff_day`` are used. Returns (spec, table).
    """
    candidates = CANDIDATES if candidates is None else candidates
    cutoff_day = pd.Timestamp(cutoff_day).normalize()
    visible = outcomes.loc[outcomes.index < cutoff_day]
    first, last = cutoff_day - pd.Timedelta(days=days), cutoff_day - pd.Timedelta(days=1)
    actual = visible.loc[first:last, list(BLOCK_STARTS)]
    rows = []
    for spec in [BASELINE, REFERENCE] + list(candidates):
        rows.append({"method": label(spec), **spec,
                     "val_brier": brier(actual, rolling_forecasts(visible, first, last, spec))})
    table = pd.DataFrame(rows)
    scores = table["val_brier"].iloc[2:].to_numpy()
    return candidates[int(np.argmin(scores))], table


def evaluation_frame(outcomes: pd.DataFrame, first_day, last_day, specs: dict) -> pd.DataFrame:
    """Long table (day, block) with the outcome, the midnight state and each method's probability."""
    actual = outcomes.loc[first_day:last_day]
    frame = actual[list(BLOCK_STARTS)].stack().rename("y").to_frame()
    frame.index.names = ["day", "block"]
    frame["on_at_midnight"] = actual["on_at_midnight"].reindex(frame.index.get_level_values("day")).to_numpy()
    for name, spec in specs.items():
        frame[name] = rolling_forecasts(outcomes, first_day, last_day, spec).stack().to_numpy()
    return frame.reset_index()


def score_table(frame: pd.DataFrame, methods, baseline="all history", reference="trailing 90d") -> pd.DataFrame:
    """Brier score and skill against the baseline and the reference, for the rows of ``frame``."""
    y = frame["y"].to_numpy()
    out = []
    for m in methods:
        score = float(np.mean((frame[m].to_numpy() - y) ** 2))
        out.append({
            "method": m, "brier": score,
            "skill_vs_baseline": 1 - score / float(np.mean((frame[baseline].to_numpy() - y) ** 2)),
            "skill_vs_reference": 1 - score / float(np.mean((frame[reference].to_numpy() - y) ** 2)),
        })
    return pd.DataFrame(out).set_index("method")


def skill_interval(frame: pd.DataFrame, method: str, reference: str, draws=5000, seed=0):
    """Brier skill of ``method`` against ``reference`` with a 95% interval from resampling whole days."""
    y = frame["y"].to_numpy()
    codes, _ = pd.factorize(frame["day"])
    n_days = codes.max() + 1
    err_m = np.bincount(codes, weights=(frame[method].to_numpy() - y) ** 2, minlength=n_days)
    err_r = np.bincount(codes, weights=(frame[reference].to_numpy() - y) ** 2, minlength=n_days)
    picks = np.random.default_rng(seed).integers(0, n_days, size=(draws, n_days))
    samples = 1 - err_m[picks].sum(axis=1) / err_r[picks].sum(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return 1 - err_m.sum() / err_r.sum(), float(low), float(high)
