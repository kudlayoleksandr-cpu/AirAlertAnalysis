import sys
from pathlib import Path

# make ``import src...`` work when pytest is run from the project root or elsewhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
