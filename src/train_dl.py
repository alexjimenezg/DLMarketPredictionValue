"""Sprint 7 — Deep Learning MLP for log_next_market_value.

Two models:
    mlp_numeric  → numeric features only, MLP 256→128→64→1, Huber loss.
    mlp_tabular  → numeric + curated categoricals (OneHot, min_frequency=50).

Run from project root:

    python -m src.train_dl
"""
from __future__ import annotations

import inspect
import json
import os
import random
import warnings
from pathlib import Path

# Quiet TF logs before importing it
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
from keras import callbacks, layers, losses, models, optimizers
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    ROOT_DIR,
)
from src.evaluate import (
    plot_real_vs_predicted,
    plot_residuals,
    regression_metrics,
)
from src.train_baselines import _sanitize_extension_dtypes

DATASET_PATH = PROCESSED_DATA_DIR / "player_market_value_dataset.parquet"
MODELS_DIR = ROOT_DIR / "models"

TARGET_COL = "log_next_market_value"
TRAIN_END = pd.Timestamp("2022-01-01")
VAL_END = pd.Timestamp("2024-01-01")

EXCLUDED_FEATURES = (
    "next_market_value_in_eur",
    "log_next_market_value",
    "next_valuation_date",
    "value_change_abs",
    "value_change_pct",
    "log_value_change",
    "days_to_next_valuation",  # not knowable at prediction time
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

NUMERIC_ID_EXCLUDE = (
    "current_club_id",
    "club_id",
    "country_id",
    "competition_country_id",
    "current_national_team_id",
)

CATEGORICAL_ALLOWLIST = (
    "position",
    "sub_position",
    "foot",
    "country_of_citizenship",
    "age_bucket",
    "final_competition_id",
    "competition_name",
    "competition_country_name",
    "competition_type",
    "competition_sub_type",
    "competition_confederation",
    "current_club_domestic_competition_id",
    "player_club_domestic_competition_id",
    "club_domestic_competition_id",
)

BASELINE_REFERENCE = {
    "previous_value": {
        "test_mae_log": 0.216,
        "test_r2_log": 0.956,
        "test_mae_eur": 900_000,
        "test_median_ape": 0.167,
    },
    "histgb": {
        "test_mae_log": 0.209,
        "test_r2_log": 0.966,
        "test_mae_eur": 1_040_000,
        "test_median_ape": 0.143,
    },
}


def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        import keras
        keras.utils.set_random_seed(seed)
    except Exception:
        pass


def load_final_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    print(f"Loading {path}")
    df = pd.read_parquet(path)
    print(f"  shape: {df.shape}")
    return _sanitize_extension_dtypes(df)


def get_dl_feature_columns(df: pd.DataFrame) -> dict:
    excluded = set(EXCLUDED_FEATURES) | set(NUMERIC_ID_EXCLUDE)
    excluded |= {c for c in df.columns if c.startswith("next_")}

    numeric_features: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_bool_dtype(df[col].dtype):
            numeric_features.append(col)
        elif pd.api.types.is_numeric_dtype(df[col].dtype):
            numeric_features.append(col)

    categorical_features = [
        c for c in CATEGORICAL_ALLOWLIST if c in df.columns and c not in excluded
    ]

    used = set(numeric_features) | set(categorical_features)
    excluded |= {c for c in df.columns if c not in used}

    return {
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
    train_df = df.loc[dates < TRAIN_END]
    val_df = df.loc[(dates >= TRAIN_END) & (dates < VAL_END)]
    test_df = df.loc[dates >= VAL_END]
    return train_df, val_df, test_df


def build_numeric_preprocessor(
    numeric_features: list[str],
) -> ColumnTransformer:
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", pipe, numeric_features)],
        remainder="drop",
    )


def _build_onehot() -> OneHotEncoder:
    params = {"handle_unknown": "ignore"}
    if "min_frequency" in inspect.signature(OneHotEncoder).parameters:
        params["min_frequency"] = 50
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        params["sparse_output"] = True
    else:
        params["sparse"] = True
    return OneHotEncoder(**params)


def build_tabular_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
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
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def build_mlp(input_dim: int, learning_rate: float = 1e-3) -> models.Model:
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.10)(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.Huber(delta=1.0),
        metrics=["mae"],
    )
    return model


def _to_dense_float32(X) -> np.ndarray:
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype="float32")


def _fit_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    epochs: int = 1000,
    batch_size: int = 1024,
) -> tuple[models.Model, dict]:
    model = build_mlp(input_dim=input_dim)
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=4,
            factor=0.5,
            min_lr=1e-6,
        ),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=2,
    )
    return model, history.history


def _predict(model: models.Model, X: np.ndarray) -> np.ndarray:
    pred = model.predict(X, batch_size=4096, verbose=0).reshape(-1)
    return pred.astype("float64")


def train_mlp_numeric(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
) -> dict:
    """Fit numeric-only MLP. Returns dict with metrics, preds, history, model."""
    pre = build_numeric_preprocessor(numeric_features)
    X_train = _to_dense_float32(pre.fit_transform(train_df[numeric_features]))
    X_val = _to_dense_float32(pre.transform(val_df[numeric_features]))
    X_test = _to_dense_float32(pre.transform(test_df[numeric_features]))

    y_train = train_df[TARGET_COL].to_numpy(dtype="float32")
    y_val = val_df[TARGET_COL].to_numpy(dtype="float32")
    y_test = test_df[TARGET_COL].to_numpy(dtype="float32")

    model, history = _fit_mlp(
        X_train, y_train, X_val, y_val, input_dim=X_train.shape[1]
    )

    val_pred = _predict(model, X_val)
    test_pred = _predict(model, X_test)

    return {
        "preprocessor": pre,
        "model": model,
        "history": history,
        "val_metrics": regression_metrics(y_val.astype("float64"), val_pred),
        "test_metrics": regression_metrics(y_test.astype("float64"), test_pred),
        "test_pred": test_pred,
        "test_y": y_test.astype("float64"),
    }


def train_mlp_tabular(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    max_input_dim: int = 5000,
) -> dict | None:
    """Fit numeric+categorical MLP. Returns None and warns if too wide / OOM."""
    try:
        pre = build_tabular_preprocessor(numeric_features, categorical_features)
        X_train_raw = pre.fit_transform(train_df[numeric_features + categorical_features])
        if X_train_raw.shape[1] > max_input_dim:
            warnings.warn(
                f"mlp_tabular: input dim {X_train_raw.shape[1]} exceeds "
                f"max_input_dim={max_input_dim}; skipping."
            )
            return None
        X_train = _to_dense_float32(X_train_raw)
        X_val = _to_dense_float32(
            pre.transform(val_df[numeric_features + categorical_features])
        )
        X_test = _to_dense_float32(
            pre.transform(test_df[numeric_features + categorical_features])
        )
    except (MemoryError, ValueError) as exc:
        warnings.warn(f"mlp_tabular skipped: {type(exc).__name__}: {exc}")
        return None

    y_train = train_df[TARGET_COL].to_numpy(dtype="float32")
    y_val = val_df[TARGET_COL].to_numpy(dtype="float32")
    y_test = test_df[TARGET_COL].to_numpy(dtype="float32")

    print(f"  tabular input dim: {X_train.shape[1]}")
    model, history = _fit_mlp(
        X_train, y_train, X_val, y_val, input_dim=X_train.shape[1]
    )

    val_pred = _predict(model, X_val)
    test_pred = _predict(model, X_test)

    return {
        "preprocessor": pre,
        "model": model,
        "history": history,
        "val_metrics": regression_metrics(y_val.astype("float64"), val_pred),
        "test_metrics": regression_metrics(y_test.astype("float64"), test_pred),
        "test_pred": test_pred,
        "test_y": y_test.astype("float64"),
    }


def evaluate_dl_model(y_true_log, y_pred_log) -> dict:
    return regression_metrics(y_true_log, y_pred_log)


def save_dl_metrics(metrics: dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False, default=str)
    print(f"DL metrics written to {output_path.resolve()}")


def plot_training_curves(
    histories: dict[str, dict], output_path: Path
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(histories)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    for ax, (name, hist) in zip(axes[0], histories.items()):
        if "loss" in hist:
            ax.plot(hist["loss"], label="train")
        if "val_loss" in hist:
            ax.plot(hist["val_loss"], label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("Huber loss")
        ax.set_title(f"{name} training curves")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Training curves figure written to {output_path.resolve()}")


def _print_metrics_table(label: str, metrics: dict) -> None:
    print(f"\n=== {label} ===")
    if not metrics:
        print("  (empty)")
        return
    keys = ["mae_log", "rmse_log", "r2_log", "mae_eur", "median_absolute_percentage_error"]
    header = f"{'model':<18}" + "".join(f"{k:>14}" for k in keys)
    print(header)
    for model_name, m in metrics.items():
        row = f"{model_name:<18}"
        for k in keys:
            v = m.get(k, float("nan"))
            row += f"{v:>14.4f}"
        print(row)


def main() -> None:
    set_global_seed(42)

    df = load_final_dataset()
    feature_info = get_dl_feature_columns(df)
    numeric_features = feature_info["numeric_features"]
    categorical_features = feature_info["categorical_features"]
    print(f"  numeric features: {len(numeric_features)}")
    print(f"  categorical features: {len(categorical_features)}")

    keep_cols = (
        [TARGET_COL, "valuation_date"]
        + numeric_features
        + categorical_features
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    train_df, val_df, test_df = temporal_train_val_test_split(df)
    del df
    train_df = train_df.dropna(subset=[TARGET_COL])
    val_df = val_df.dropna(subset=[TARGET_COL])
    test_df = test_df.dropna(subset=[TARGET_COL])
    print(
        f"split sizes — train: {len(train_df):,} | "
        f"val: {len(val_df):,} | test: {len(test_df):,}"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Training mlp_numeric ---")
    res_num = train_mlp_numeric(
        train_df, val_df, test_df, numeric_features
    )
    res_num["model"].save(MODELS_DIR / "mlp_numeric.keras")
    joblib.dump(
        res_num["preprocessor"],
        MODELS_DIR / "dl_numeric_preprocessor.joblib",
    )
    print(f"  saved {MODELS_DIR / 'mlp_numeric.keras'}")
    print(f"  saved {MODELS_DIR / 'dl_numeric_preprocessor.joblib'}")

    print("\n--- Training mlp_tabular ---")
    res_tab = train_mlp_tabular(
        train_df, val_df, test_df, numeric_features, categorical_features
    )
    if res_tab is not None:
        res_tab["model"].save(MODELS_DIR / "mlp_tabular.keras")
        joblib.dump(
            res_tab["preprocessor"],
            MODELS_DIR / "dl_tabular_preprocessor.joblib",
        )
        print(f"  saved {MODELS_DIR / 'mlp_tabular.keras'}")
        print(f"  saved {MODELS_DIR / 'dl_tabular_preprocessor.joblib'}")

    validation_metrics = {"mlp_numeric": res_num["val_metrics"]}
    test_metrics = {"mlp_numeric": res_num["test_metrics"]}
    histories = {"mlp_numeric": res_num["history"]}
    if res_tab is not None:
        validation_metrics["mlp_tabular"] = res_tab["val_metrics"]
        test_metrics["mlp_tabular"] = res_tab["test_metrics"]
        histories["mlp_tabular"] = res_tab["history"]

    payload = {
        "split_info": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_date_max": train_df["valuation_date"].max().date().isoformat(),
            "val_date_min": val_df["valuation_date"].min().date().isoformat(),
            "val_date_max": val_df["valuation_date"].max().date().isoformat(),
            "test_date_min": test_df["valuation_date"].min().date().isoformat(),
        },
        "feature_info": feature_info,
        "validation": validation_metrics,
        "test": test_metrics,
        "baseline_reference": BASELINE_REFERENCE,
    }
    save_dl_metrics(payload, METRICS_DIR / "deep_learning_metrics.json")

    plot_training_curves(histories, FIGURES_DIR / "dl_training_curves.png")

    # Diagnostic plots: prefer tabular if trained, else numeric
    diag = res_tab if res_tab is not None else res_num
    plot_real_vs_predicted(
        diag["test_y"],
        diag["test_pred"],
        FIGURES_DIR / "dl_real_vs_predicted.png",
    )
    plot_residuals(
        diag["test_y"],
        diag["test_pred"],
        FIGURES_DIR / "dl_residuals.png",
    )

    _print_metrics_table("VALIDATION", validation_metrics)
    _print_metrics_table("TEST", test_metrics)
    print("\nBaseline reference (test):")
    for name, vals in BASELINE_REFERENCE.items():
        print(f"  {name}: {vals}")


if __name__ == "__main__":
    main()
