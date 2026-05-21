"""Sprint 4 - Build player performance features.

Run from the project root:

    python -m src.make_performance_features

Reads:
    data/interim/player_profile_features.parquet
    data/raw/appearances.csv  or  appearances.csv.gz
    data/raw/games.csv        or  games.csv.gz

Writes:
    data/interim/player_performance_features.parquet
    reports/metrics/player_performance_summary.json
"""
from __future__ import annotations

import pandas as pd

from src.config import INTERIM_DATA_DIR, METRICS_DIR, RAW_DATA_DIR
from src.features import (
    PERFORMANCE_WINDOWS,
    aggregate_player_performance_windows,
    build_player_performance_summary,
    prepare_appearances_with_games,
    save_player_performance_dataset,
    save_player_performance_summary,
    validate_player_performance_dataset,
)
from src.load_data import _resolve_raw_path, load_csv_file


def _load_required_raw_csv(filename: str) -> pd.DataFrame:
    path = _resolve_raw_path(RAW_DATA_DIR, filename)
    if path is None:
        raise FileNotFoundError(
            f"{filename}(.gz) not found in {RAW_DATA_DIR.resolve()}"
        )
    print(f"Loading {path.name}")
    return load_csv_file(path)


def _performance_feature_count(df: pd.DataFrame) -> int:
    return int(
        sum(
            any(col.endswith(f"_{window}") for window in PERFORMANCE_WINDOWS)
            for col in df.columns
        )
    )


def main() -> None:
    base_path = INTERIM_DATA_DIR / "player_profile_features.parquet"
    print(f"Loading base player profile features from {base_path}")
    base = pd.read_parquet(base_path)
    print(f"base shape: {base.shape}")

    appearances_raw = _load_required_raw_csv("appearances.csv")
    print(f"appearances shape: {appearances_raw.shape}")

    games_raw = _load_required_raw_csv("games.csv")
    print(f"games shape: {games_raw.shape}")

    appearances_games = prepare_appearances_with_games(appearances_raw, games_raw)
    print(f"appearances_games shape: {appearances_games.shape}")

    enriched = aggregate_player_performance_windows(
        base, appearances_games, windows=PERFORMANCE_WINDOWS
    )
    validate_player_performance_dataset(enriched, expected_rows=len(base))

    out_parquet = INTERIM_DATA_DIR / "player_performance_features.parquet"
    save_player_performance_dataset(enriched, out_parquet)

    summary = build_player_performance_summary(enriched)
    save_player_performance_summary(
        summary, METRICS_DIR / "player_performance_summary.json"
    )

    feature_count = _performance_feature_count(enriched)
    print(f"\nfinal shape: {enriched.shape}")
    print(f"performance features created: {feature_count}")
    for window in PERFORMANCE_WINDOWS:
        print(
            f"played_any_rate_{window}: "
            f"{summary[f'played_any_rate_{window}']:.6f}"
        )
        print(
            f"median_minutes_{window}: "
            f"{summary[f'median_minutes_{window}']:.6f}"
        )

    preview_cols = [
        c
        for c in (
            "player_id",
            "player_name",
            "valuation_date",
            "minutes_365",
            "goals_365",
            "assists_365",
            "appearances_365",
            "goals_per_90_365",
            "market_value_in_eur",
            "next_market_value_in_eur",
        )
        if c in enriched.columns
    ]
    print("\nfirst 10 rows:")
    print(enriched[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
