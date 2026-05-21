"""Sprint 2 — Build the supervised target dataset.

Run from the project root:

    python -m src.make_targets

Loads `data/raw/*.csv`, builds the (current, next) valuation pairs from
`player_valuations`, validates invariants, persists parquet/csv/json, and
prints a console summary.
"""
from __future__ import annotations

from src.build_dataset import (
    build_target_dataset,
    build_target_summary,
    save_target_dataset,
    save_target_summary,
)
from src.config import INTERIM_DATA_DIR, METRICS_DIR
from src.load_data import load_raw_datasets


def main() -> None:
    datasets = load_raw_datasets()
    target = build_target_dataset(datasets)

    parquet_path = INTERIM_DATA_DIR / "valuation_targets.parquet"
    save_target_dataset(target, parquet_path)

    summary = build_target_summary(target)
    save_target_summary(summary, METRICS_DIR / "target_summary.json")

    print(f"\nshape: {target.shape}")
    print(
        f"date range: {summary['min_valuation_date']} -> "
        f"{summary['max_next_valuation_date']}"
    )
    print(f"unique players: {summary['unique_players']}")

    print("\ndays_to_next_valuation:")
    print(target["days_to_next_valuation"].describe())

    print("\nnext_market_value_in_eur:")
    print(target["next_market_value_in_eur"].describe())

    print("\nfirst 10 rows:")
    print(
        target[
            [
                "player_id",
                "valuation_date",
                "market_value_in_eur",
                "next_valuation_date",
                "next_market_value_in_eur",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()
