"""Module defines common test data."""

from pathlib import Path

_THIS_DIR = Path(__file__).parent

_DATA_DIR = _THIS_DIR / "data"

# Paths to data files
MLB_TEAMS_2012_CSV = _DATA_DIR / "mlb_teams_2012.csv"
MLB_TEAMS_2012_SQL = _DATA_DIR / "mlb_teams_2012.sql"
