"""Sprint 2 — Build the supervised target dataset.

Produces a `(player_id, valuation_date)` table where each row pairs a
player's current valuation with their next historical valuation. No
performance features yet, no joins beyond `player_valuations`.

Public API:
    standardize_valuation_columns(df)        -> pd.DataFrame
    prepare_valuations(df)                   -> pd.DataFrame
    create_next_value_target(valuations)     -> pd.DataFrame
    build_target_dataset(datasets)           -> pd.DataFrame
    validate_target_dataset(df)              -> None
    save_target_dataset(df, output_path)     -> None
    build_target_summary(df)                 -> dict
    save_target_summary(summary, output)     -> None
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.load_data import (
    DATE_CANDIDATES,
    PLAYER_ID_CANDIDATES,
    _first_present as first_present,
)

MARKET_VALUE_CANDIDATES = ("market_value_in_eur", "market_value", "value_eur")

REQUIRED_COLUMNS = ("player_id", "date", "market_value_in_eur")
OPTIONAL_PASSTHROUGH = (
    "current_club_id",
    "player_club_domestic_competition_id",
    "domestic_competition_id",
)
TARGET_REQUIRED = (
    "player_id",
    "valuation_date",
    "market_value_in_eur",
    "next_valuation_date",
    "next_market_value_in_eur",
    "log_market_value",
    "log_next_market_value",
    "days_to_next_valuation",
    "value_change_abs",
    "value_change_pct",
    "log_value_change",
)

_CANDIDATE_MAP = {
    "player_id": PLAYER_ID_CANDIDATES,
    "date": DATE_CANDIDATES,
    "market_value_in_eur": MARKET_VALUE_CANDIDATES,
}


def standardize_valuation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename source columns to canonical names; raise if any are missing."""
    out = df.copy()
    rename_map: dict[str, str] = {}
    for canonical, candidates in _CANDIDATE_MAP.items():
        if canonical in out.columns:
            continue
        found = first_present(out, candidates)
        if found is None:
            raise ValueError(
                f"Missing required column for target build: expected one of "
                f"{list(candidates)} for '{canonical}'."
            )
        rename_map[found] = canonical
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def prepare_valuations(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes, drop invalid rows, deduplicate, and sort."""
    out = standardize_valuation_columns(df)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["market_value_in_eur"] = pd.to_numeric(
        out["market_value_in_eur"], errors="coerce"
    )
    out = out.dropna(subset=["player_id", "date", "market_value_in_eur"])
    out = out[out["market_value_in_eur"] > 0]
    out = out.sort_values(
        ["player_id", "date", "market_value_in_eur"],
        ascending=[True, True, False],
    )
    out = out.drop_duplicates(subset=["player_id", "date"], keep="first")
    out = out.sort_values(["player_id", "date"]).reset_index(drop=True)
    return out


def create_next_value_target(valuations: pd.DataFrame) -> pd.DataFrame:
    """Build (current, next) valuation pairs per player and derived columns."""
    out = valuations.copy()
    grouped = out.groupby("player_id", sort=False)
    out["next_valuation_date"] = grouped["date"].shift(-1)
    out["next_market_value_in_eur"] = grouped["market_value_in_eur"].shift(-1)

    out = out.rename(columns={"date": "valuation_date"})

    out["log_market_value"] = np.log1p(out["market_value_in_eur"])
    out["log_next_market_value"] = np.log1p(out["next_market_value_in_eur"])
    out["days_to_next_valuation"] = (
        out["next_valuation_date"] - out["valuation_date"]
    ).dt.days
    out["value_change_abs"] = (
        out["next_market_value_in_eur"] - out["market_value_in_eur"]
    )
    out["value_change_pct"] = (
        out["value_change_abs"] / out["market_value_in_eur"]
    )
    out["log_value_change"] = (
        out["log_next_market_value"] - out["log_market_value"]
    )

    out = out.dropna(
        subset=["next_valuation_date", "next_market_value_in_eur"]
    )
    out["days_to_next_valuation"] = out["days_to_next_valuation"].astype(int)

    leading = list(TARGET_REQUIRED)
    passthrough = [c for c in OPTIONAL_PASSTHROUGH if c in out.columns]
    rest = [c for c in out.columns if c not in leading and c not in passthrough]
    out = out[leading + passthrough + rest].reset_index(drop=True)
    return out


def build_target_dataset(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Top-level entry point: take raw datasets, return validated target df."""
    if "player_valuations" not in datasets:
        raise KeyError("'player_valuations' missing from datasets dict")
    valuations = prepare_valuations(datasets["player_valuations"])
    target = create_next_value_target(valuations)
    validate_target_dataset(target)
    return target


def validate_target_dataset(df: pd.DataFrame) -> None:
    """Raise ValueError if any structural invariant is broken."""
    missing = [c for c in TARGET_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Target dataset missing columns: {missing}")

    if df["valuation_date"].isna().any():
        raise ValueError("Found null valuation_date values.")
    if df["next_valuation_date"].isna().any():
        raise ValueError("Found null next_valuation_date values.")
    if df["next_market_value_in_eur"].isna().any():
        raise ValueError("Found null next_market_value_in_eur values.")

    n_neg_curr = int((df["market_value_in_eur"] <= 0).sum())
    if n_neg_curr:
        raise ValueError(
            f"Found {n_neg_curr} rows where market_value_in_eur <= 0."
        )
    n_neg_next = int((df["next_market_value_in_eur"] <= 0).sum())
    if n_neg_next:
        raise ValueError(
            f"Found {n_neg_next} rows where next_market_value_in_eur <= 0."
        )

    n_bad_order = int(
        (df["next_valuation_date"] <= df["valuation_date"]).sum()
    )
    if n_bad_order:
        raise ValueError(
            f"Found {n_bad_order} rows where next_valuation_date "
            f"<= valuation_date."
        )

    n_bad_days = int((df["days_to_next_valuation"] <= 0).sum())
    if n_bad_days:
        raise ValueError(
            f"Found {n_bad_days} rows where days_to_next_valuation <= 0."
        )

    dup_keys = ["player_id", "valuation_date", "next_valuation_date"]
    n_dups = int(df.duplicated(subset=dup_keys).sum())
    if n_dups:
        raise ValueError(
            f"Found {n_dups} duplicate rows on {dup_keys}."
        )


def save_target_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Write the target dataset to parquet plus a CSV mirror."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"Target dataset written to {output_path.resolve()}")
    print(f"CSV mirror written to {csv_path.resolve()}")


def build_target_summary(df: pd.DataFrame) -> dict:
    """Aggregate descriptive stats of the target dataset for JSON storage."""
    return {
        "rows": int(len(df)),
        "unique_players": int(df["player_id"].nunique()),
        "min_valuation_date": df["valuation_date"].min().date().isoformat(),
        "max_valuation_date": df["valuation_date"].max().date().isoformat(),
        "min_next_valuation_date": df["next_valuation_date"]
        .min()
        .date()
        .isoformat(),
        "max_next_valuation_date": df["next_valuation_date"]
        .max()
        .date()
        .isoformat(),
        "min_days_to_next_valuation": int(df["days_to_next_valuation"].min()),
        "median_days_to_next_valuation": float(
            df["days_to_next_valuation"].median()
        ),
        "max_days_to_next_valuation": int(df["days_to_next_valuation"].max()),
        "target_min": float(df["next_market_value_in_eur"].min()),
        "target_median": float(df["next_market_value_in_eur"].median()),
        "target_mean": float(df["next_market_value_in_eur"].mean()),
        "target_max": float(df["next_market_value_in_eur"].max()),
        "log_target_min": float(df["log_next_market_value"].min()),
        "log_target_median": float(df["log_next_market_value"].median()),
        "log_target_mean": float(df["log_next_market_value"].mean()),
        "log_target_max": float(df["log_next_market_value"].max()),
    }


def save_target_summary(summary: dict, output_path: Path) -> None:
    """Persist the summary dict as JSON, creating parents if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Target summary written to {output_path.resolve()}")
