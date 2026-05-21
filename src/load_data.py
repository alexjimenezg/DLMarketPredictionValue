"""Sprint 1 — Raw data loading, validation, and summarisation.

Public API:
    validate_raw_files(raw_dir, expected) -> None
    load_csv_file(path)                   -> pd.DataFrame
    load_raw_datasets(raw_dir)            -> dict[str, pd.DataFrame]
    summarize_dataframe(df)               -> dict
    build_data_summary(datasets)          -> dict
    save_data_summary(summary, output)    -> None
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import (
    EXPECTED_RAW_FILES,
    OPTIONAL_RAW_FILES,
    RAW_DATA_DIR,
    RAW_FILE_EXTENSIONS,
)

PLAYER_ID_CANDIDATES = ("player_id", "playerID", "playerId")
CLUB_ID_CANDIDATES = ("club_id", "clubID", "clubId")
COMPETITION_ID_CANDIDATES = ("competition_id", "competitionID", "competitionId")
DATE_CANDIDATES = ("date", "datetime", "valuation_date")


def _first_present(df: pd.DataFrame, candidates) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _resolve_raw_path(raw_dir: Path, base_filename: str) -> Path | None:
    """Return the first existing path matching `<basename>` with .csv or .csv.gz."""
    base = base_filename
    if base.endswith(".csv"):
        base = base[:-4]
    for ext in RAW_FILE_EXTENSIONS:
        candidate = raw_dir / f"{base}{ext}"
        if candidate.is_file():
            return candidate
    return None


def validate_raw_files(
    raw_dir: Path = RAW_DATA_DIR,
    expected: Dict[str, str] = EXPECTED_RAW_FILES,
) -> None:
    """Raise FileNotFoundError listing every missing CSV at once.

    Accepts either uncompressed `.csv` or gzipped `.csv.gz` for each
    expected file.
    """
    raw_dir = Path(raw_dir)
    missing = [
        filename
        for filename in expected.values()
        if _resolve_raw_path(raw_dir, filename) is None
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing raw CSV files in {raw_dir.resolve()}: "
            f"{', '.join(missing)} "
            f"(accepted extensions: {', '.join(RAW_FILE_EXTENSIONS)}). "
            "Place the Transfermarkt CSVs there and retry."
        )


def load_csv_file(path: Path) -> pd.DataFrame:
    """Read a single CSV with UTF-8, falling back to latin-1 if needed.

    Pandas auto-detects gzip from the `.gz` suffix; no extra flags needed.
    """
    path = Path(path)
    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        warnings.warn(
            f"UTF-8 decode failed for {path.name}; retrying with latin-1.",
            stacklevel=2,
        )
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    print(f"  loaded {path.name}: shape={df.shape}")
    return df


def load_raw_datasets(
    raw_dir: Path = RAW_DATA_DIR,
    include_optional: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Validate and load all expected raw CSVs into a dict keyed by logical name.

    If `include_optional=True`, also loads any present file from
    `OPTIONAL_RAW_FILES` (transfers, club_games, game_events, etc.) —
    silently skipping ones that aren't there.
    """
    raw_dir = Path(raw_dir)
    validate_raw_files(raw_dir)
    print(f"Loading raw datasets from {raw_dir.resolve()}")
    datasets: Dict[str, pd.DataFrame] = {}
    for name, filename in EXPECTED_RAW_FILES.items():
        datasets[name] = load_csv_file(_resolve_raw_path(raw_dir, filename))
    if include_optional:
        for name, filename in OPTIONAL_RAW_FILES.items():
            path = _resolve_raw_path(raw_dir, filename)
            if path is not None:
                datasets[name] = load_csv_file(path)
            else:
                print(f"  optional {filename}: not found, skipping")
    return datasets


def summarize_dataframe(df: pd.DataFrame) -> dict:
    """Return rows / columns / column_names / null_percentage for a dataframe."""
    rows, cols = df.shape
    if rows == 0:
        null_pct = {col: 0.0 for col in df.columns}
    else:
        null_pct = {
            col: round(float(pct), 2)
            for col, pct in (df.isna().mean() * 100).items()
        }
    return {
        "rows": int(rows),
        "columns": int(cols),
        "column_names": list(df.columns),
        "null_percentage": null_pct,
    }


def _augment_player_valuations(df: pd.DataFrame) -> dict:
    extras: dict = {"total_valuations": int(len(df))}
    date_col = _first_present(df, DATE_CANDIDATES)
    if date_col is not None:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        if parsed.notna().any():
            extras["min_date"] = parsed.min().date().isoformat()
            extras["max_date"] = parsed.max().date().isoformat()
        else:
            warnings.warn(
                f"player_valuations.{date_col} could not be parsed as dates."
            )
    else:
        warnings.warn(
            "player_valuations: no date column found "
            f"(looked for {DATE_CANDIDATES})."
        )
    pid_col = _first_present(df, PLAYER_ID_CANDIDATES)
    if pid_col is not None:
        extras["unique_players"] = int(df[pid_col].nunique())
    else:
        warnings.warn(
            "player_valuations: no player id column found "
            f"(looked for {PLAYER_ID_CANDIDATES})."
        )
    return extras


def _augment_unique_count(
    df: pd.DataFrame, candidates, key: str, table: str
) -> dict:
    col = _first_present(df, candidates)
    if col is None:
        warnings.warn(
            f"{table}: no id column found (looked for {candidates})."
        )
        return {}
    return {key: int(df[col].nunique())}


def build_data_summary(
    datasets: Dict[str, pd.DataFrame],
) -> dict:
    """Combine per-table summaries plus table-specific entity counts."""
    summary: dict = {}
    for name, df in datasets.items():
        entry = summarize_dataframe(df)
        if name == "player_valuations":
            entry.update(_augment_player_valuations(df))
        elif name == "players":
            entry.update(
                _augment_unique_count(
                    df, PLAYER_ID_CANDIDATES, "unique_players", name
                )
            )
        elif name == "clubs":
            entry.update(
                _augment_unique_count(
                    df, CLUB_ID_CANDIDATES, "unique_clubs", name
                )
            )
        elif name == "competitions":
            entry.update(
                _augment_unique_count(
                    df, COMPETITION_ID_CANDIDATES, "unique_competitions", name
                )
            )
        summary[name] = entry
    return summary


def save_data_summary(summary: dict, output_path: Path) -> None:
    """Persist the summary dict as JSON, creating parents if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Summary written to {output_path.resolve()}")
