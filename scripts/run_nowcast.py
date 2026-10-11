"""Forecast v2: choose settings on pre-test data, then score the test period once.

Usage:
    python scripts/run_nowcast.py select   # uses data known at 18:00 on the first test evening
    python scripts/run_nowcast.py test     # runs the test period; refuses to run twice
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import load, nowcast  # noqa: E402

RESULTS = ROOT / "results"
SELECTION = RESULTS / "nowcast_selection.json"
VALIDATION = RESULTS / "nowcast_validation_forecasts.csv"
LEAD_LAG = RESULTS / "nowcast_lead_lag.csv"
TEST = RESULTS / "nowcast_test_forecasts.csv"


def select() -> None:
    alerts = load.load()
    cutoff = nowcast.evening_start(nowcast.FIRST_TEST_EVENING)
    choice = nowcast.select(alerts, cutoff)
    RESULTS.mkdir(exist_ok=True)
    SELECTION.write_text(json.dumps({
        "data_known_until_utc": str(choice["cutoff"]),
        "validation_first_evening": f"{choice['validation_first_evening']:%Y-%m-%d}",
        "validation_last_evening": f"{choice['validation_last_evening']:%Y-%m-%d}",
        "baseline_c": choice["baseline_c"],
        "model": choice["model"],
        "baseline_c_candidates": json.loads(choice["baseline_c_table"].to_json(orient="records")),
        "model_candidates": json.loads(choice["model_table"].to_json(orient="records")),
    }, indent=2) + "\n")
    choice["validation_frame"].to_csv(VALIDATION)
    # the lead-lag analysis is descriptive, but it is also computed from pre-test data only
    nowcast.lead_lag(alerts, cutoff).to_csv(LEAD_LAG)
    print(choice["baseline_c_table"].to_string())
    print(choice["model_table"].to_string())
    print("chosen:", choice["baseline_c"], choice["model"])


def test() -> None:
    if TEST.exists():
        sys.exit(f"{TEST} already exists: the test period has been run. Not running it again.")
    choice = json.loads(SELECTION.read_text())
    table = nowcast.build_table(load.load(), load.cutoff_instant())
    first, last = nowcast.FIRST_TEST_EVENING, nowcast.LAST_TEST_EVENING
    frame = nowcast.scored_frame(table, {
        "c": nowcast.rolling_forecasts(table, first, last, choice["baseline_c"]),
        "model": nowcast.rolling_forecasts(table, first, last, choice["model"]),
    })
    frame.to_csv(TEST)
    print(f"wrote {TEST}: {len(frame)} issue times, {frame['night'].nunique()} nights, "
          f"run at {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    {"select": select, "test": test}[sys.argv[1]]()
