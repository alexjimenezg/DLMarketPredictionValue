"""Sprint 3 — Build player profile features.

Run from the project root:

    python -m src.make_player_features

Reads:
    data/interim/valuation_targets.parquet
    data/raw/players.csv  or  players.csv.gz

Writes:
    data/interim/player_profile_features.parquet
    reports/metrics/player_profile_summary.json
"""
from __future__ import annotations

from src.config import (
    INTERIM_DATA_DIR,
    METRICS_DIR,
    RAW_DATA_DIR,
)
from src.features import (
    add_player_profile_features,
    build_player_profile_summary,
    prepare_players,
    save_player_profile_dataset,
    save_player_profile_summary,
    validate_player_profile_dataset,
)
from src.load_data import _resolve_raw_path, load_csv_file

import pandas as pd


def main() -> None:
    targets_path = INTERIM_DATA_DIR / "valuation_targets.parquet"
    print(f"Loading targets from {targets_path}")
    targets = pd.read_parquet(targets_path)
    print(f"  targets shape: {targets.shape}")

    players_path = _resolve_raw_path(RAW_DATA_DIR, "players.csv")
    if players_path is None:
        raise FileNotFoundError(
            f"players.csv(.gz) not found in {RAW_DATA_DIR.resolve()}"
        )
    print(f"Loading players from {players_path.name}")
    players_raw = load_csv_file(players_path)
    print(f"  players shape: {players_raw.shape}")

    players = prepare_players(players_raw)
    print(f"  players after prepare: {players.shape}")

    enriched = add_player_profile_features(targets, players)
    validate_player_profile_dataset(enriched, expected_rows=len(targets))

    out_parquet = INTERIM_DATA_DIR / "player_profile_features.parquet"
    save_player_profile_dataset(enriched, out_parquet)

    summary = build_player_profile_summary(enriched)
    save_player_profile_summary(
        summary, METRICS_DIR / "player_profile_summary.json"
    )

    print(f"\nfinal shape: {enriched.shape}")
    print(f"unique players: {summary['unique_players']}")
    if "missing_age_at_valuation_pct" in summary:
        print(f"missing age: {summary['missing_age_at_valuation_pct']}%")
    if "position_counts" in summary:
        print("position distribution:")
        for k, v in summary["position_counts"].items():
            print(f"  {k}: {v}")

    preview_cols = [
        c
        for c in (
            "player_id",
            "player_name",
            "valuation_date",
            "age_at_valuation",
            "age_bucket",
            "position",
            "sub_position",
            "foot",
            "market_value_in_eur",
            "next_market_value_in_eur",
        )
        if c in enriched.columns
    ]
    print("\nfirst 10 rows:")
    print(enriched[preview_cols].head(10))


if __name__ == "__main__":
    main()
