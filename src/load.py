"""Load, validate, dedupe and clean the official air raid alerts file.

Run ``python -m src.load`` from the project root to print the validation
report for the raw file.
"""
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "official_data_en.csv"

COLUMNS = ["oblast", "raion", "hromada", "level", "started_at", "finished_at", "source"]
LEVELS = {"oblast", "raion", "hromada"}
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S%z"
KYIV_TZ = "Europe/Kyiv"

# Last complete calendar day (Europe/Kyiv) before the feed broke on 2026-08-29.
# validate_raw() re-derives this from the data; see README "Cleaning rules".
LAST_COMPLETE_DAY = "2026-08-28"

LONG_ALERT_HOURS = 24
# Alerts longer than this are "standing" alerts of frontline hromadas. They are
# kept in the cleaned data and flagged; see README for how each series uses them.
STANDING_ALERT_HOURS = 7 * 24


def read_raw(path=RAW_PATH) -> pd.DataFrame:
    """Read the CSV as strings, exactly as stored (empty fields stay empty)."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(raw.columns) != COLUMNS:
        raise ValueError(f"unexpected columns: {list(raw.columns)}")
    return raw


def _parse_utc(strings: pd.Series) -> pd.Series:
    if not strings.str.endswith("+00:00").all():
        raise ValueError("found a timestamp that is not UTC (+00:00)")
    return pd.to_datetime(strings, format=TIMESTAMP_FORMAT, utc=True)


def cutoff_instant(last_day=LAST_COMPLETE_DAY) -> pd.Timestamp:
    """UTC instant at which the Kyiv-local day after ``last_day`` begins."""
    next_midnight = pd.Timestamp(last_day).tz_localize(KYIV_TZ) + pd.DateOffset(days=1)
    return next_midnight.tz_convert("UTC")


def find_feed_break(started_at: pd.Series) -> dict:
    """Locate the longest nationwide silence between consecutive alert starts."""
    starts = started_at.sort_values().reset_index(drop=True)
    gaps = starts.diff()
    i = gaps.idxmax()
    last_before = starts[i - 1]
    last_local_day = last_before.tz_convert(KYIV_TZ).normalize()
    return {
        "gap": gaps[i],
        "last_start_before": last_before,
        "first_start_after": starts[i],
        "second_longest_gap": gaps.drop(i).max(),
        # the local day in which the feed stopped is partial, so the last
        # complete day is the one before it
        "last_complete_day": (last_local_day - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    }


def clean(raw: pd.DataFrame, last_day=LAST_COMPLETE_DAY) -> pd.DataFrame:
    """Return deduplicated, validated intervals ending at the analysis cutoff.

    Rules, in order:
      1. drop exact duplicate rows;
      2. drop rows with finished_at <= started_at;
      3. drop rows starting at or after the cutoff (start of the Kyiv-local day
         after ``last_day``) and clip finished_at of the remaining rows to it;
      4. flag, but keep, long (> 24 h) and standing (> 7 d) alerts.
    """
    df = raw.drop_duplicates().copy()
    bad_levels = set(df["level"]) - LEVELS
    if bad_levels:
        raise ValueError(f"unexpected level values: {bad_levels}")
    df["started_at"] = _parse_utc(df["started_at"])
    df["finished_at"] = _parse_utc(df["finished_at"])

    df = df[df["finished_at"] > df["started_at"]]

    cutoff = cutoff_instant(last_day)
    df = df[df["started_at"] < cutoff].copy()
    # durations and flags describe the alert as recorded, before clipping
    df["duration_h"] = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 3600
    df["long_alert"] = df["duration_h"] > LONG_ALERT_HOURS
    df["standing_alert"] = df["duration_h"] > STANDING_ALERT_HOURS
    df["clipped_at_cutoff"] = df["finished_at"] > cutoff
    df["finished_at"] = df["finished_at"].clip(upper=cutoff)

    return df.sort_values(["started_at", "oblast", "raion", "hromada"]).reset_index(drop=True)


def load(path=RAW_PATH, last_day=LAST_COMPLETE_DAY) -> pd.DataFrame:
    return clean(read_raw(path), last_day)


def validate_raw(raw: pd.DataFrame) -> dict:
    """Compute the facts the analysis relies on, from the raw file."""
    start = _parse_utc(raw["started_at"])
    finish = _parse_utc(raw["finished_at"])
    hours = (finish - start).dt.total_seconds() / 3600
    dup = raw.duplicated()
    dedup = raw[~dup]
    d_start, d_hours = start[~dup], hours[~dup]

    has_raion, has_hromada = dedup["raion"] != "", dedup["hromada"] != ""
    level_fields_ok = (
        ((dedup["level"] == "oblast") & ~has_raion & ~has_hromada)
        | ((dedup["level"] == "raion") & has_raion & ~has_hromada)
        | ((dedup["level"] == "hromada") & has_raion & has_hromada)
    )

    rows_per_utc_day = d_start.dt.strftime("%Y-%m-%d").value_counts()
    prior_days = pd.date_range("2026-08-01", "2026-08-28").strftime("%Y-%m-%d")
    prior = rows_per_utc_day.reindex(prior_days, fill_value=0)
    gap_days = ["2026-08-30", "2026-08-31", "2026-09-01"]

    month = d_start.dt.strftime("%Y-%m")
    oblast_share = (dedup["level"] == "oblast").groupby(month).mean()
    # Kyiv City is only ever oblast-level, so measure the switch without it
    not_kyiv = dedup["oblast"] != "Kyiv City"
    last_oblast_level_row = d_start[(dedup["level"] == "oblast") & not_kyiv].max()

    brk = find_feed_break(d_start)
    before_cutoff = d_start < cutoff_instant(brk["last_complete_day"])

    return {
        "rows": len(raw),
        "first_start": start.min(),
        "last_start": start.max(),
        "sorted_by_start": bool(start.is_monotonic_increasing),
        "sources": sorted(raw["source"].unique()),
        "levels": sorted(raw["level"].unique()),
        "duplicate_rows": int(dup.sum()),
        "duplicate_share": float(dup.mean()),
        "max_copies_of_a_row": int(raw.value_counts().max()),
        "last_start_with_duplicate": start[raw.duplicated(keep=False)].max(),
        "rows_after_dedupe": len(dedup),
        "level_fields_consistent": bool(level_fields_ok.all()),
        "nonpositive_duration_raw": int((finish <= start).sum()),
        "nonpositive_duration_dedup": int((finish <= start)[~dup].sum()),
        "over_24h_raw": int((hours > LONG_ALERT_HOURS).sum()),
        "over_24h_dedup": int((d_hours > LONG_ALERT_HOURS).sum()),
        "over_24h_dedup_before_cutoff": int((d_hours[before_cutoff] > LONG_ALERT_HOURS).sum()),
        "over_7d_dedup_before_cutoff": int((d_hours[before_cutoff] > STANDING_ALERT_HOURS).sum()),
        "max_duration_h": float(hours.max()),
        "max_duration_h_by_region": {
            name: float(d_hours[dedup["oblast"] == name].max())
            for name in ["Kyiv City", "Kharkivska oblast", "Lvivska oblast"]
        },
        "rows_on_gap_days_utc": {d: int(rows_per_utc_day.get(d, 0)) for d in gap_days},
        "rows_on_2026_08_29_utc": int(rows_per_utc_day.get("2026-08-29", 0)),
        "rows_per_day_2026_08_01_to_28": (int(prior.min()), int(prior.median()), int(prior.max())),
        "feed_break": brk,
        "kyiv_city_levels": sorted(dedup.loc[~not_kyiv, "level"].unique()),
        "oblast_level_row_share_by_month": oblast_share.loc["2025-06":"2026-02"].round(3).to_dict(),
        "last_oblast_level_row_outside_kyiv_city": last_oblast_level_row,
    }


def main() -> None:
    report = validate_raw(read_raw())
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"{key}: {value}")
    cleaned = load()
    print(f"\ncleaned rows: {len(cleaned)}")
    print(f"cleaned span: {cleaned['started_at'].min()} .. {cleaned['finished_at'].max()}")
    print(f"rows clipped at cutoff: {int(cleaned['clipped_at_cutoff'].sum())}")
    print(f"rows per level: {cleaned['level'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
