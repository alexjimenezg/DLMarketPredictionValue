"""Sprint 6 — Regression metrics and diagnostic plots.

All metrics are computed in two scales:
- log scale (mae_log, rmse_log, r2_log) using y as `log_next_market_value`.
- euro scale via `np.expm1(y)`, with predictions clipped at 0 first to
  avoid negative euro values.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def regression_metrics(
    y_true_log: np.ndarray, y_pred_log: np.ndarray
) -> dict:
    """Return a dict of regression metrics on log and euro scale."""
    y_true_log = np.asarray(y_true_log, dtype="float64")
    y_pred_log = np.asarray(y_pred_log, dtype="float64")

    mae_log = float(mean_absolute_error(y_true_log, y_pred_log))
    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    r2_log = float(r2_score(y_true_log, y_pred_log))

    y_pred_clipped = np.clip(y_pred_log, 0, None)
    y_true_eur = np.expm1(y_true_log)
    y_pred_eur = np.expm1(y_pred_clipped)

    mae_eur = float(mean_absolute_error(y_true_eur, y_pred_eur))
    rmse_eur = float(np.sqrt(mean_squared_error(y_true_eur, y_pred_eur)))
    median_ae_eur = float(median_absolute_error(y_true_eur, y_pred_eur))

    safe = y_true_eur > 0
    if safe.any():
        ape = np.abs(y_true_eur[safe] - y_pred_eur[safe]) / y_true_eur[safe]
        median_ape = float(np.median(ape))
    else:
        median_ape = float("nan")

    return {
        "mae_log": mae_log,
        "rmse_log": rmse_log,
        "r2_log": r2_log,
        "mae_eur": mae_eur,
        "rmse_eur": rmse_eur,
        "median_absolute_error_eur": median_ae_eur,
        "median_absolute_percentage_error": median_ape,
    }


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Metrics written to {output_path.resolve()}")


def plot_model_comparison(metrics: dict, output_path: Path) -> None:
    """Bar chart comparing MAE_log and R2_log across models on test split."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    test_metrics = metrics.get("test", {})
    if not test_metrics:
        return
    models = list(test_metrics.keys())
    mae_log = [test_metrics[m]["mae_log"] for m in models]
    r2_log = [test_metrics[m]["r2_log"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(models, mae_log, color="steelblue")
    axes[0].set_title("Test MAE (log scale)")
    axes[0].set_ylabel("MAE_log")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(models, r2_log, color="seagreen")
    axes[1].set_title("Test R2 (log scale)")
    axes[1].set_ylabel("R2_log")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Comparison figure written to {output_path.resolve()}")


def _sample_indices(n: int, sample_size: int, seed: int = 42) -> np.ndarray:
    if sample_size >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=sample_size, replace=False)


def plot_real_vs_predicted(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    output_path: Path,
    sample_size: int = 5000,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_true_log = np.asarray(y_true_log, dtype="float64")
    y_pred_log = np.asarray(y_pred_log, dtype="float64")
    idx = _sample_indices(len(y_true_log), sample_size)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true_log[idx], y_pred_log[idx], alpha=0.25, s=10)
    lo = min(y_true_log[idx].min(), y_pred_log[idx].min())
    hi = max(y_true_log[idx].max(), y_pred_log[idx].max())
    ax.plot([lo, hi], [lo, hi], color="red", lw=1, ls="--", label="y=x")
    ax.set_xlabel("y_true (log)")
    ax.set_ylabel("y_pred (log)")
    ax.set_title(f"Real vs Predicted (log scale, n={len(idx):,})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Real-vs-predicted figure written to {output_path.resolve()}")


def plot_residuals(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    output_path: Path,
    sample_size: int = 5000,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_true_log = np.asarray(y_true_log, dtype="float64")
    y_pred_log = np.asarray(y_pred_log, dtype="float64")
    residuals = y_pred_log - y_true_log
    idx = _sample_indices(len(y_true_log), sample_size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred_log[idx], residuals[idx], alpha=0.25, s=10)
    axes[0].axhline(0, color="red", lw=1, ls="--")
    axes[0].set_xlabel("y_pred (log)")
    axes[0].set_ylabel("residual = pred - true (log)")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=80, color="slategray")
    axes[1].axvline(0, color="red", lw=1, ls="--")
    axes[1].set_xlabel("residual (log)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Residual distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Residuals figure written to {output_path.resolve()}")
