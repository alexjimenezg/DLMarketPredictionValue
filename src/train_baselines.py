"""Sprint 6 — Baseline regression models for log_next_market_value.

Models trained:
    - mean / median / previous_value (no fitting needed)
    - Ridge (numeric + curated categoricals via OneHot)
    - HistGradientBoostingRegressor (numeric only)

Run from project root:

    python -m src.train_baselines
"""
from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    ROOT_DIR,
)
from src.evaluate import (
    plot_model_comparison,
    plot_real_vs_predicted,
    plot_residuals,
    regression_metrics,
    save_metrics,
)

DATASET_PATH = PROCESSED_DATA_DIR / "player_market_value_dataset.parquet"
MODELS_DIR = ROOT_DIR / "models"

TARGET_COL = "log_next_market_value"
TARGET_COL_EUR = "next_market_value_in_eur"
PREVIOUS_VALUE_COL = "log_market_value"

TRAIN_END = pd.Timestamp("2022-01-01")
VAL_END = pd.Timestamp("2024-01-01")

# Hard-excluded columns (target leakage, identifiers, free text, dates)
EXCLUDED_FEATURES = (
    "next_market_value_in_eur",
    "log_next_market_value",
    "next_valuation_date",
    "value_change_abs",
    "value_change_pct",
    "log_value_change",
    "player_name",
    "name",
    "first_name",
    "last_name",
    "url",
    "filename",
    "image_url",
    "agent_name",
    "date_of_birth",
    "contract_expiration_date",
    "valuation_date",
    "player_id",
)

# Categoricals chosen for low/medium cardinality (Ridge OHE-friendly)
CATEGORICAL_ALLOWLIST = (
    "position",
    "sub_position",
    "foot",
    "age_bucket",
    "country_of_birth",
    "country_of_citizenship",
    "current_club_domestic_competition_id",
    "player_club_domestic_competition_id",
    "club_domestic_competition_id",
    "final_competition_id",
    "competition_type",
    "competition_sub_type",
    "competition_country_name",
    "competition_confederation",
    "current_club_id",
)

# Categoricals excluded explicitly (high-cardinality identifiers/strings)
HIGH_CARDINALITY_EXCLUDE = (
    "club_name",
    "club_code",
    "club_stadium_name",
    "club_net_transfer_record",
    "competition_code",
    "competition_country_id",
    "competition_domestic_league_code",
    "city_of_birth",
    "player_code",
    "current_national_team_id",
    "current_club_name",
    "current_club_domestic_competition_id_x",
)


def _sanitize_extension_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast pandas extension dtypes (string/boolean/Int*) to numpy-compatible.

    Mutates the input DataFrame in-place to avoid a large up-front copy
    (the sanitised frame is downstream of a fresh parquet read, so
    mutating it is safe and saves ~200 MB on this dataset).

    sklearn imputers/encoders cannot evaluate `X != X` on `pd.NA`, so any
    nullable extension dtype must be flattened: strings/booleans become
    object with None for missing, nullable ints become float64 with NaN.
    """
    for col in df.columns:
        s = df[col]
        if not pd.api.types.is_extension_array_dtype(s.dtype):
            continue
        if isinstance(s.dtype, (pd.StringDtype, pd.BooleanDtype)):
            df[col] = s.astype(object).where(s.notna(), None)
        else:
            df[col] = pd.to_numeric(s, errors="coerce").astype("float64")
    return df


def load_final_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    print(f"Loading {path}")
    df = pd.read_parquet(path)
    print(f"  shape: {df.shape}")
    df = _sanitize_extension_dtypes(df)
    return df


def get_feature_columns(df: pd.DataFrame) -> dict:
    """Split columns into numeric/categorical/excluded for the pipelines."""
    excluded = set(EXCLUDED_FEATURES) | set(HIGH_CARDINALITY_EXCLUDE)
    excluded |= {c for c in df.columns if c.startswith("next_")}

    candidate_cols = [c for c in df.columns if c not in excluded]

    numeric_features: list[str] = []
    categorical_features: list[str] = []
    for col in candidate_cols:
        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            numeric_features.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            numeric_features.append(col)
        elif col in CATEGORICAL_ALLOWLIST:
            categorical_features.append(col)
        else:
            excluded.add(col)

    return {
        "target": [TARGET_COL],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_features": sorted(excluded),
    }


def temporal_train_val_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "valuation_date" not in df.columns:
        raise ValueError("Dataset is missing 'valuation_date'.")
    dates = pd.to_datetime(df["valuation_date"])
    train_df = df[dates < TRAIN_END].copy()
    val_df = df[(dates >= TRAIN_END) & (dates < VAL_END)].copy()
    test_df = df[dates >= VAL_END].copy()
    for name, split in (("train", train_df), ("val", val_df), ("test", test_df)):
        if len(split) == 0:
            warnings.warn(
                f"Temporal split produced 0 rows for '{name}' — continuing."
            )
    return train_df, val_df, test_df


def train_constant_baselines(
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
    splits: dict[str, pd.DataFrame],
) -> dict:
    """Compute mean / median / previous_value baseline metrics."""
    results: dict[str, dict] = {"validation": {}, "test": {}}
    val_n = len(val_y)
    test_n = len(test_y)

    if len(train_y) == 0:
        return results

    mean_pred = float(np.mean(train_y))
    median_pred = float(np.median(train_y))

    if val_n:
        results["validation"]["mean"] = regression_metrics(
            val_y, np.full(val_n, mean_pred)
        )
        results["validation"]["median"] = regression_metrics(
            val_y, np.full(val_n, median_pred)
        )
        if PREVIOUS_VALUE_COL in splits["val"].columns:
            prev = splits["val"][PREVIOUS_VALUE_COL].to_numpy(dtype="float64")
            results["validation"]["previous_value"] = regression_metrics(
                val_y, prev
            )

    if test_n:
        results["test"]["mean"] = regression_metrics(
            test_y, np.full(test_n, mean_pred)
        )
        results["test"]["median"] = regression_metrics(
            test_y, np.full(test_n, median_pred)
        )
        if PREVIOUS_VALUE_COL in splits["test"].columns:
            prev = splits["test"][PREVIOUS_VALUE_COL].to_numpy(dtype="float64")
            results["test"]["previous_value"] = regression_metrics(
                test_y, prev
            )

    return results


def _build_onehot() -> OneHotEncoder:
    params = {"handle_unknown": "ignore"}
    if "min_frequency" in inspect.signature(OneHotEncoder).parameters:
        params["min_frequency"] = 50
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        params["sparse_output"] = True
    else:
        params["sparse"] = True
    return OneHotEncoder(**params)


def build_ridge_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _build_onehot()),
        ]
    )
    transformers = [("num", numeric_pipe, numeric_features)]
    if categorical_features:
        transformers.append(("cat", categorical_pipe, categorical_features))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        steps=[("preprocess", pre), ("model", Ridge(alpha=1.0))]
    )


def build_histgb_pipeline(numeric_features: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_features,
            )
        ],
        remainder="drop",
    )
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=42,
    )
    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def _evaluate_split(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
) -> tuple[dict, np.ndarray]:
    pred = pipe.predict(X)
    return regression_metrics(y, pred), pred


def _print_metrics_table(label: str, metrics: dict) -> None:
    print(f"\n=== {label} ===")
    if not metrics:
        print("  (empty)")
        return
    keys = ["mae_log", "rmse_log", "r2_log", "mae_eur", "median_absolute_percentage_error"]
    header = f"{'model':<18}" + "".join(f"{k:>14}" for k in keys)
    print(header)
    for model, m in metrics.items():
        row = f"{model:<18}"
        for k in keys:
            v = m.get(k, float("nan"))
            row += f"{v:>14.4f}"
        print(row)


def train_and_evaluate_baselines() -> dict:
    df = load_final_dataset(DATASET_PATH)
    feature_info = get_feature_columns(df)
    numeric_features = feature_info["numeric_features"]
    categorical_features = feature_info["categorical_features"]
    print(f"  numeric features: {len(numeric_features)}")
    print(f"  categorical features: {len(categorical_features)}")

    train_df, val_df, test_df = temporal_train_val_test_split(df)
    print(
        f"split sizes — train: {len(train_df):,} | "
        f"val: {len(val_df):,} | test: {len(test_df):,}"
    )

    target = TARGET_COL
    train_df = train_df.dropna(subset=[target])
    val_df = val_df.dropna(subset=[target])
    test_df = test_df.dropna(subset=[target])

    train_y = train_df[target].to_numpy(dtype="float64")
    val_y = val_df[target].to_numpy(dtype="float64")
    test_y = test_df[target].to_numpy(dtype="float64")

    splits = {"train": train_df, "val": val_df, "test": test_df}

    feature_cols = numeric_features + categorical_features
    train_X = train_df[feature_cols]
    val_X = val_df[feature_cols]
    test_X = test_df[feature_cols]

    constant_results = train_constant_baselines(
        train_y, val_y, test_y, splits
    )
    validation_metrics = constant_results["validation"]
    test_metrics = constant_results["test"]

    # Ridge
    print("\nFitting Ridge…")
    ridge_pipe = build_ridge_pipeline(numeric_features, categorical_features)
    ridge_pipe.fit(train_X, train_y)
    if len(val_y):
        validation_metrics["ridge"], _ = _evaluate_split(
            ridge_pipe, val_X, val_y
        )
    ridge_pred_test = None
    if len(test_y):
        test_metrics["ridge"], ridge_pred_test = _evaluate_split(
            ridge_pipe, test_X, test_y
        )

    # HistGB
    print("Fitting HistGradientBoostingRegressor…")
    histgb_pipe = build_histgb_pipeline(numeric_features)
    histgb_pipe.fit(train_X[numeric_features], train_y)
    if len(val_y):
        validation_metrics["histgb"], _ = _evaluate_split(
            histgb_pipe, val_X[numeric_features], val_y
        )
    histgb_pred_test = None
    if len(test_y):
        test_metrics["histgb"], histgb_pred_test = _evaluate_split(
            histgb_pipe, test_X[numeric_features], test_y
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge_pipe, MODELS_DIR / "ridge_baseline.joblib")
    print(f"  saved {MODELS_DIR / 'ridge_baseline.joblib'}")
    joblib.dump(histgb_pipe, MODELS_DIR / "histgb_baseline.joblib")
    print(f"  saved {MODELS_DIR / 'histgb_baseline.joblib'}")

    metrics_payload = {
        "split_info": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_date_max": (
                train_df["valuation_date"].max().date().isoformat()
                if len(train_df)
                else None
            ),
            "val_date_min": (
                val_df["valuation_date"].min().date().isoformat()
                if len(val_df)
                else None
            ),
            "val_date_max": (
                val_df["valuation_date"].max().date().isoformat()
                if len(val_df)
                else None
            ),
            "test_date_min": (
                test_df["valuation_date"].min().date().isoformat()
                if len(test_df)
                else None
            ),
        },
        "feature_info": {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "excluded_features": feature_info["excluded_features"],
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }

    save_metrics(metrics_payload, METRICS_DIR / "baseline_metrics.json")

    plot_model_comparison(
        metrics_payload, FIGURES_DIR / "baseline_model_comparison.png"
    )

    # Pick HistGB on test for the diagnostic plots if available, else Ridge
    diag_pred = histgb_pred_test if histgb_pred_test is not None else ridge_pred_test
    if diag_pred is not None and len(test_y):
        plot_real_vs_predicted(
            test_y, diag_pred, FIGURES_DIR / "baseline_real_vs_predicted.png"
        )
        plot_residuals(
            test_y, diag_pred, FIGURES_DIR / "baseline_residuals.png"
        )

    _print_metrics_table("VALIDATION", validation_metrics)
    _print_metrics_table("TEST", test_metrics)

    return metrics_payload


def main() -> None:
    train_and_evaluate_baselines()


if __name__ == "__main__":
    main()
