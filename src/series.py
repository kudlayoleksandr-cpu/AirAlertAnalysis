"""Turn alert intervals into daily and hourly "time under alert" series.

All series are built from the *union* of the intervals, so overlapping alerts
are never counted twice, and are binned on Europe/Kyiv local boundaries.
"""
import numpy as np
import pandas as pd

from src.load import KYIV_TZ, LAST_COMPLETE_DAY

# The file starts at 2022-03-15 16:10 UTC, so 2022-03-15 is a partial day.
FIRST_COMPLETE_DAY = "2022-03-16"

_NS_PER_HOUR = 3_600_000_000_000


def _ns(values) -> np.ndarray:
    """UTC instants as int64 nanoseconds."""
    return pd.DatetimeIndex(values).tz_convert("UTC").as_unit("ns").asi8


def merge_intervals(started_at, finished_at) -> pd.DataFrame:
    """Union of [start, finish) intervals as sorted, disjoint intervals.

    Touching intervals (one ends exactly when the next starts) are merged.
    """
    start, finish = _ns(started_at), _ns(finished_at)
    keep = finish > start
    order = np.argsort(start[keep], kind="stable")
    start, finish = start[keep][order], finish[keep][order]
    if len(start) == 0:
        empty = pd.DatetimeIndex([], tz="UTC")
        return pd.DataFrame({"started_at": empty, "finished_at": empty})
    reach = np.maximum.accumulate(finish)
    # a new merged interval begins where a start lies beyond everything before it
    new_group = np.r_[True, start[1:] > reach[:-1]]
    group_start = start[new_group]
    group_end = reach[np.r_[new_group[1:], True]]
    return pd.DataFrame({
        "started_at": pd.to_datetime(group_start, utc=True),
        "finished_at": pd.to_datetime(group_end, utc=True),
    })


def _hours_between_edges(merged: pd.DataFrame, edges: pd.DatetimeIndex) -> np.ndarray:
    """Hours of ``merged`` (disjoint, sorted) falling between consecutive edges."""
    start, finish, edge = _ns(merged["started_at"]), _ns(merged["finished_at"]), _ns(edges)
    if len(start) == 0:
        return np.zeros(len(edge) - 1)
    full_before = np.r_[0, np.cumsum(finish - start)]  # alert time in the first k intervals
    k = np.searchsorted(start, edge, side="right")  # intervals starting at or before the edge
    last = np.maximum(k - 1, 0)
    # the k-th interval may still be running at the edge: remove the part after it
    overshoot = np.where(k > 0, np.maximum(finish[last] - edge, 0), 0)
    covered = full_before[k] - overshoot
    return np.diff(covered) / _NS_PER_HOUR


def daily_hours(started_at, finished_at, first_day, last_day, tz=KYIV_TZ) -> pd.Series:
    """Hours under alert per local calendar day, for first_day..last_day inclusive.

    The index is a naive daily DatetimeIndex of local dates. Days on which the
    clocks change are 23 or 25 hours long and are not rescaled.
    """
    days = pd.date_range(first_day, last_day, freq="D")
    edges = pd.date_range(days[0], days[-1] + pd.Timedelta(days=1), freq="D").tz_localize(tz)
    merged = merge_intervals(started_at, finished_at)
    return pd.Series(_hours_between_edges(merged, edges), index=days, name="alert_hours")


def hourly_hours(started_at, finished_at, first_day, last_day, tz=KYIV_TZ) -> pd.Series:
    """Hours under alert (0..1) per elapsed hour, indexed by tz-aware local hour start.

    A 23-hour day has 23 entries and a 25-hour day has 25 (the repeated local
    hour appears twice, with different UTC offsets).
    """
    first = pd.Timestamp(first_day).tz_localize(tz)
    end = (pd.Timestamp(last_day) + pd.Timedelta(days=1)).tz_localize(tz)
    edges = pd.date_range(first.tz_convert("UTC"), end.tz_convert("UTC"), freq="h")
    merged = merge_intervals(started_at, finished_at)
    index = edges[:-1].tz_convert(tz)
    return pd.Series(_hours_between_edges(merged, edges), index=index, name="alert_hours")


def region_rows(alerts: pd.DataFrame, oblast: str, include_standing: bool = False) -> pd.DataFrame:
    """Rows of every level (oblast, raion, hromada) belonging to ``oblast``.

    Standing alerts (> 7 days, see load.STANDING_ALERT_HOURS) are excluded by
    default: a single permanently alerted frontline hromada would otherwise
    make "an alert is active anywhere in the oblast" true around the clock.
    """
    rows = alerts[alerts["oblast"] == oblast]
    if rows.empty:
        raise ValueError(f"no rows for oblast {oblast!r}")
    if not include_standing:
        rows = rows[~rows["standing_alert"]]
    return rows


def region_daily(alerts, oblast, first_day=FIRST_COMPLETE_DAY, last_day=LAST_COMPLETE_DAY,
                 include_standing=False) -> pd.Series:
    """Daily hours during which an alert was active anywhere in ``oblast``."""
    rows = region_rows(alerts, oblast, include_standing)
    return daily_hours(rows["started_at"], rows["finished_at"], first_day, last_day)


def region_hourly(alerts, oblast, first_day=FIRST_COMPLETE_DAY, last_day=LAST_COMPLETE_DAY,
                  include_standing=False) -> pd.Series:
    rows = region_rows(alerts, oblast, include_standing)
    return hourly_hours(rows["started_at"], rows["finished_at"], first_day, last_day)
