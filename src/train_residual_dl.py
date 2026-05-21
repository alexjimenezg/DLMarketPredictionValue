"""Sprint 8 — Residual Deep Learning.

Predicts the residual `log_next_market_value - log_market_value` (i.e. the
delta vs. the no-change baseline) and reconstructs the final prediction as
`log_market_value + pred_residual`. Compared against `previous_value`,
`histgb` (Sprint 6) and `mlp_numeric` direct (Sprint 7).

Run from project root:

    python -m src.train_residual_dl
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import callbacks, layers, losses, models, optimizers, regularizers
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
CURRENT_COL = "log_market_value"

TRAIN_END = pd.Timestamp("2022-01-01")
VAL_END = pd.Timestamp("2024-01-01")

EXCLUDED_FEATURES = (
    "next_market_value_in_eur",
    "log_next_market_value",
    "next_valuation_date",
    "value_change_abs",
    "value_change_pct",
    "log_value_change",
    "days_to_next_valuation",
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
    "mlp_numeric_direct": {
        "test_mae_log": 0.296,
        "test_r2_log": 0.937,
        "test_mae_eur": 612_000,
        "test_median_ape": 0.227,
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


def get_residual_dl_feature_columns(df: pd.DataFrame) -> dict:
    """Numeric-only features (no categoricals to avoid OOM)."""
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

    used = set(numeric_features)
    excluded |= {c for c in df.columns if c not in used}

    return {
        "numeric_features": numeric_features,
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


def build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
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


def build_residual_mlp(
    input_dim: int, learning_rate: float = 5e-4
) -> models.Model:
    reg = regularizers.l2(1e-4)
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.10)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.Huber(delta=0.5),
        metrics=["mae"],
    )
    return model


def _to_dense_float32(X) -> np.ndarray:
    return np.asarray(X, dtype="float32")


def train_residual_mlp(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
) -> dict:
    """Fit residual MLP and reconstruct absolute log predictions."""
    pre = build_preprocessor(numeric_features)
    X_train = _to_dense_float32(pre.fit_transform(train_df[numeric_features]))
    X_val = _to_dense_float32(pre.transform(val_df[numeric_features]))
    X_test = _to_dense_float32(pre.transform(test_df[numeric_features]))

    log_curr_train = train_df[CURRENT_COL].to_numpy(dtype="float64")
    log_curr_val = val_df[CURRENT_COL].to_numpy(dtype="float64")
    log_curr_test = test_df[CURRENT_COL].to_numpy(dtype="float64")

    log_next_train = train_df[TARGET_COL].to_numpy(dtype="float64")
    log_next_val = val_df[TARGET_COL].to_numpy(dtype="float64")
    log_next_test = test_df[TARGET_COL].to_numpy(dtype="float64")

    y_res_train = (log_next_train - log_curr_train).astype("float32")
    y_res_val = (log_next_val - log_curr_val).astype("float32")

    model = build_residual_mlp(input_dim=X_train.shape[1])
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6
        ),
    ]
    history = model.fit(
        X_train,
        y_res_train,
        validation_data=(X_val, y_res_val),
        epochs=100,
        batch_size=1024,
        callbacks=cb,
        verbose=2,
    ).history

    val_res_pred = model.predict(X_val, batch_size=4096, verbose=0).reshape(-1)
    test_res_pred = model.predict(X_test, batch_size=4096, verbose=0).reshape(-1)

    val_pred_log = log_curr_val + val_res_pred.astype("float64")
    test_pred_log = log_curr_test + test_res_pred.astype("float64")

    return {
        "preprocessor": pre,
        "model": model,
        "history": history,
        "y_res_train": y_res_train.astype("float64"),
        "val_metrics": regression_metrics(log_next_val, val_pred_log),
        "test_metrics": regression_metrics(log_next_test, test_pred_log),
        "test_y_log": log_next_test,
        "test_pred_log": test_pred_log,
    }


def plot_residual_target_distribution(
    y_residual_train: np.ndarray, output_path: Path
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    clipped = np.clip(y_residual_train, -2.5, 2.5)
    ax.hist(clipped, bins=80, color="steelblue")
    ax.axvline(0, color="red", ls="--", lw=1, label="no-change baseline")
    ax.set_xlabel("residual = log_next - log_curr  (clipped to ±2.5)")
    ax.set_ylabel("count")
    ax.set_title(
        f"Train residual target distribution (n={len(y_residual_train):,})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Residual target distribution figure written to {output_path.resolve()}")


def plot_training_curves(history: dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if "loss" in history:
        ax.plot(history["loss"], label="train")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Huber(delta=0.5) loss on residual")
    ax.set_title("mlp_residual training curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Training curves figure written to {output_path.resolve()}")


def _print_metrics_table(label: str, metrics: dict) -> None:
    print(f"\n=== {label} ===")
    keys = ["mae_log", "rmse_log", "r2_log", "mae_eur", "median_absolute_percentage_error"]
    header = f"{'model':<22}" + "".join(f"{k:>14}" for k in keys)
    print(header)
    for name, m in metrics.items():
        row = f"{name:<22}"
        for k in keys:
            v = m.get(k, float("nan"))
            row += f"{v:>14.4f}"
        print(row)


def main() -> None:
    set_global_seed(42)

    df = load_final_dataset()
    feature_info = get_residual_dl_feature_columns(df)
    numeric_features = feature_info["numeric_features"]
    print(f"  numeric features: {len(numeric_features)}")

    keep_cols = [TARGET_COL, CURRENT_COL, "valuation_date"] + numeric_features
    keep_cols = list(dict.fromkeys(c for c in keep_cols if c in df.columns))
    df = df[keep_cols]

    train_df, val_df, test_df = temporal_train_val_test_split(df)
    del df
    train_df = train_df.dropna(subset=[TARGET_COL, CURRENT_COL])
    val_df = val_df.dropna(subset=[TARGET_COL, CURRENT_COL])
    test_df = test_df.dropna(subset=[TARGET_COL, CURRENT_COL])
    print(
        f"split sizes — train: {len(train_df):,} | "
        f"val: {len(val_df):,} | test: {len(test_df):,}"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Training mlp_residual ---")
    res = train_residual_mlp(train_df, val_df, test_df, numeric_features)
    res["model"].save(MODELS_DIR / "mlp_residual.keras")
    joblib.dump(
        res["preprocessor"],
        MODELS_DIR / "residual_dl_preprocessor.joblib",
    )
    print(f"  saved {MODELS_DIR / 'mlp_residual.keras'}")
    print(f"  saved {MODELS_DIR / 'residual_dl_preprocessor.joblib'}")

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
        "target_info": {
            "target": "log_next_market_value - log_market_value",
            "train_residual_mean": float(np.mean(res["y_res_train"])),
            "train_residual_median": float(np.median(res["y_res_train"])),
            "train_residual_std": float(np.std(res["y_res_train"])),
        },
        "validation": {"mlp_residual": res["val_metrics"]},
        "test": {"mlp_residual": res["test_metrics"]},
        "baseline_reference": BASELINE_REFERENCE,
    }
    out_json = METRICS_DIR / "residual_dl_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Metrics written to {out_json.resolve()}")

    plot_residual_target_distribution(
        res["y_res_train"], FIGURES_DIR / "residual_target_distribution.png"
    )
    plot_training_curves(
        res["history"], FIGURES_DIR / "residual_dl_training_curves.png"
    )
    plot_real_vs_predicted(
        res["test_y_log"],
        res["test_pred_log"],
        FIGURES_DIR / "residual_dl_real_vs_predicted.png",
    )
    plot_residuals(
        res["test_y_log"],
        res["test_pred_log"],
        FIGURES_DIR / "residual_dl_residuals.png",
    )

    _print_metrics_table("VALIDATION", payload["validation"])
    _print_metrics_table("TEST", payload["test"])
    print("\nBaseline reference (test):")
    for name, vals in BASELINE_REFERENCE.items():
        print(f"  {name}: {vals}")
    print("\nTarget info:")
    for k, v in payload["target_info"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
