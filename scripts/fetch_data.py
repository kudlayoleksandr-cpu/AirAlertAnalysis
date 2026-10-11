"""Download the official alerts file at the pinned commit into data/raw/.

Usage:  python scripts/fetch_data.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

COMMIT = "bf0c323c78e3e4508721458797dcd38c95df13f3"
URL = (
    "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/"
    f"{COMMIT}/datasets/official_data_en.csv"
)
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "official_data_en.csv"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {URL}")
    with urllib.request.urlopen(URL) as response:
        payload = response.read()
    DEST.write_bytes(payload)
    print(f"wrote {DEST} ({len(payload):,} bytes)")
    print(f"sha256 {hashlib.sha256(payload).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
