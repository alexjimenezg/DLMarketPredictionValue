"""Sprint 5 — Build the final modelling dataset with club & competition context.

Run from the project root:

    python -m src.make_final_dataset

Reads:
    data/interim/player_performance_features.parquet
    data/raw/clubs.csv         or  clubs.csv.gz
    data/raw/competitions.csv  or  competitions.csv.gz

Writes:
    data/processed/player_market_value_dataset.parquet
    reports/metrics/final_dataset_summary.json
"""
from __future__ import annotations

import pandas as pd

from src.config import (
    INTERIM_DATA_DIR,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.features import (
    add_club_competition_context,
    build_final_dataset_summary,
    prepare_clubs,
    prepare_competitions,
    save_final_dataset,
    save_final_dataset_summary,
    validate_final_dataset,
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


def main() -> None:
    base_path = INTERIM_DATA_DIR / "player_performance_features.parquet"
    print(f"Loading base from {base_path}")
    base = pd.read_parquet(base_path)
    print(f"base shape: {base.shape}")

    clubs_raw = _load_required_raw_csv("clubs.csv")
    print(f"clubs shape: {clubs_raw.shape}")

    competitions_raw = _load_required_raw_csv("competitions.csv")
    print(f"competitions shape: {competitions_raw.shape}")

    clubs = prepare_clubs(clubs_raw)
    competitions = prepare_competitions(competitions_raw)
    print(f"clubs prepared: {clubs.shape} | competitions prepared: {competitions.shape}")

    enriched = add_club_competition_context(base, clubs, competitions)
    validate_final_dataset(enriched, expected_rows=len(base))

    out_parquet = PROCESSED_DATA_DIR / "player_market_value_dataset.parquet"
    save_final_dataset(enriched, out_parquet)

    summary = build_final_dataset_summary(enriched)
    save_final_dataset_summary(
        summary, METRICS_DIR / "final_dataset_summary.json"
    )

    print(f"\nfinal shape: {enriched.shape}")
    if "unique_clubs" in summary:
        print(f"unique clubs: {summary['unique_clubs']}")
    if "unique_competitions" in summary:
        print(f"unique competitions: {summary['unique_competitions']}")
    if "missing_club_name_pct" in summary:
        print(f"missing club_name: {summary['missing_club_name_pct']}%")
    if "competition_name_missing_pct" in summary:
        print(
            f"missing competition_name: "
            f"{summary['competition_name_missing_pct']}%"
        )
    if "top_20_competitions" in summary:
        print("top 10 competitions:")
        for name, count in list(summary["top_20_competitions"].items())[:10]:
            print(f"  {name}: {count}")

    preview_cols = [
        c
        for c in (
            "player_id",
            "player_name",
            "valuation_date",
            "current_club_id",
            "club_name",
            "final_competition_id",
            "competition_name",
            "market_value_in_eur",
            "next_market_value_in_eur",
        )
        if c in enriched.columns
    ]
    print("\nfirst 10 rows:")
    print(enriched[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
