"""Project configuration: paths and expected raw files.

All paths are derived from this file's location so the project remains
portable across Windows/Linux without hard-coded absolute paths.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

EXPECTED_RAW_FILES = {
    "players": "players.csv",
    "player_valuations": "player_valuations.csv",
    "appearances": "appearances.csv",
    "games": "games.csv",
    "clubs": "clubs.csv",
    "competitions": "competitions.csv",
}

OPTIONAL_RAW_FILES = {
    "transfers": "transfers.csv",
    "club_games": "club_games.csv",
    "game_events": "game_events.csv",
    "game_lineups": "game_lineups.csv",
    "countries": "countries.csv",
    "national_teams": "national_teams.csv",
}

RAW_FILE_EXTENSIONS = (".csv", ".csv.gz")
