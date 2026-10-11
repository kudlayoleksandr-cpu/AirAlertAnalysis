"""Forecast v3: choose the window on pre-test data, then score the test period once.

Usage:
    python scripts/run_blocks.py select   # uses outcomes known at midnight starting the first test day
    python scripts/run_blocks.py test     # runs the test period; refuses to run twice
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import blocks, load  # noqa: E402

RESULTS = ROOT / "results"
SELECTION = RESULTS / "blocks_selection.json"
TEST = RESULTS / "blocks_test_forecasts.csv"


def select() -> None:
    outcomes = blocks.block_outcomes(load.load(), blocks.FIRST_TEST_DAY)
    chosen, table = blocks.select(outcomes, blocks.FIRST_TEST_DAY)
    RESULTS.mkdir(exist_ok=True)
    SELECTION.write_text(json.dumps({
        "outcomes_known_until": f"midnight starting {blocks.FIRST_TEST_DAY} (Europe/Kyiv)",
        "validation_days": blocks.VALIDATION_DAYS,
        "chosen": chosen,
        "chosen_label": blocks.label(chosen),
        "candidates": json.loads(table.to_json(orient="records")),
    }, indent=2) + "\n")
    print(table.to_string())
    print("chosen:", blocks.label(chosen))


def test() -> None:
    if TEST.exists():
        sys.exit(f"{TEST} already exists: the test period has been run. Not running it again.")
    json.loads(SELECTION.read_text())  # the selection must exist before the test is run
    last = pd.Timestamp(blocks.LAST_TEST_DAY)
    outcomes = blocks.block_outcomes(load.load(), last + pd.Timedelta(days=1))
    specs = {blocks.label(spec): spec for spec in [blocks.BASELINE, blocks.REFERENCE] + blocks.CANDIDATES}
    frame = blocks.evaluation_frame(outcomes, blocks.FIRST_TEST_DAY, last, specs)
    frame.to_csv(TEST, index=False)
    print(f"wrote {TEST}: {len(frame)} forecasts over {frame['day'].nunique()} days, "
          f"run at {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    {"select": select, "test": test}[sys.argv[1]]()
