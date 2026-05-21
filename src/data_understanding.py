"""Sprint 1 — Data Understanding entry point.

Run from the project root:

    python -m src.data_understanding

Loads the raw Transfermarkt CSVs from ``data/raw/``, prints a console
summary per table, and writes ``reports/metrics/data_summary.json``.
"""
from __future__ import annotations

import json

from src.config import METRICS_DIR, RAW_DATA_DIR
from src.load_data import (
    build_data_summary,
    load_raw_datasets,
    save_data_summary,
)


def _print_table_overview(name: str, df) -> None:
    print(f"\n=== {name} ===")
    print(f"shape  : {df.shape}")
    print(f"columns: {list(df.columns)}")
    print("head   :")
    print(df.head())
    if len(df) > 0:
        nulls = (df.isna().mean() * 100).round(2)
        nulls = nulls[nulls > 0].sort_values(ascending=False)
        if not nulls.empty:
            print("null %% (>0):")
            print(nulls.to_string())


def main() -> None:
    datasets = load_raw_datasets(RAW_DATA_DIR)

    for name, df in datasets.items():
        _print_table_overview(name, df)

    summary = build_data_summary(datasets)

    print("\n=== Aggregated summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    output = METRICS_DIR / "data_summary.json"
    save_data_summary(summary, output)


if __name__ == "__main__":
    main()
