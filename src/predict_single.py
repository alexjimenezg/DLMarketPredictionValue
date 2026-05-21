"""Single-player inference helpers for the local Streamlit demo.

The deployed Sprint 8 model predicts:

    residual = log_next_market_value - log_market_value

Final value is reconstructed as:

    pred_log_next = log_market_value + predicted_residual
    pred_next_value = expm1(pred_log_next)
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.config import PROCESSED_DATA_DIR, ROOT_DIR

MODEL_PATH = ROOT_DIR / "models" / "mlp_residual.keras"
PREPROCESSOR_PATH = ROOT_DIR / "models" / "residual_dl_preprocessor.joblib"
DATASET_PATH = PROCESSED_DATA_DIR / "player_market_value_dataset.parquet"

TARGET_COLUMNS = {
    "next_market_value_in_eur",
    "log_next_market_value",
    "value_change_abs",
    "value_change_pct",
    "log_value_change",
    "next_valuation_date",
    "days_to_next_valuation",
    "player_id",
}

FALLBACK_NUMERIC_FEATURES = [
    "market_value_in_eur",
    "log_market_value",
    "last_season",
    "height_in_cm",
    "international_caps",
    "international_goals",
    "age_at_valuation",
    "contract_days_remaining",
    "appearances_90",
    "minutes_90",
    "goals_90",
    "assists_90",
    "yellow_cards_90",
    "red_cards_90",
    "goal_contributions_90",
    "goals_per_90_90",
    "assists_per_90_90",
    "goal_contributions_per_90_90",
    "minutes_per_appearance_90",
    "played_any_90",
    "appearances_180",
    "minutes_180",
    "goals_180",
    "assists_180",
    "yellow_cards_180",
    "red_cards_180",
    "goal_contributions_180",
    "goals_per_90_180",
    "assists_per_90_180",
    "goal_contributions_per_90_180",
    "minutes_per_appearance_180",
    "played_any_180",
    "appearances_365",
    "minutes_365",
    "goals_365",
    "assists_365",
    "yellow_cards_365",
    "red_cards_365",
    "goal_contributions_365",
    "goals_per_90_365",
    "assists_per_90_365",
    "goal_contributions_per_90_365",
    "minutes_per_appearance_365",
    "played_any_365",
    "club_squad_size",
    "club_average_age",
    "club_foreigners_number",
    "club_foreigners_percentage",
    "club_national_team_players",
    "club_stadium_seats",
    "club_last_season",
    "club_foreigners_ratio",
    "club_national_team_players_ratio",
    "total_clubs",
]


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the residual MLP and fitted sklearn preprocessor."""
    missing = [path for path in (MODEL_PATH, PREPROCESSOR_PATH) if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing model artifact(s). Expected files: "
            f"{MODEL_PATH} and {PREPROCESSOR_PATH}. Missing: {missing_text}"
        )
    model = load_model(MODEL_PATH, compile=False)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def load_training_schema() -> pd.DataFrame:
    """Load the final training dataset as schema reference."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {DATASET_PATH}. "
            "Run python -m src.make_final_dataset first."
        )
    return pd.read_parquet(DATASET_PATH)


def _features_from_transformers(preprocessor: Any) -> list[str]:
    features: list[str] = []
    for transformer in getattr(preprocessor, "transformers_", []) or []:
        if len(transformer) < 3:
            continue
        cols = transformer[2]
        if cols is None or cols == "drop":
            continue
        if isinstance(cols, slice):
            continue
        if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
            features.extend([str(col) for col in cols])
    return features


def get_expected_model_features(preprocessor) -> list[str]:
    """Best-effort extraction of fitted preprocessor input columns."""
    if hasattr(preprocessor, "feature_names_in_"):
        return [str(col) for col in preprocessor.feature_names_in_]

    features = _features_from_transformers(preprocessor)
    if features:
        return features

    if hasattr(preprocessor, "named_steps"):
        for step in preprocessor.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return [str(col) for col in step.feature_names_in_]
            features = _features_from_transformers(step)
            if features:
                return features

    return list(FALLBACK_NUMERIC_FEATURES)


def infer_age_bucket(age: float) -> str:
    if age < 18:
        return "U18"
    if age <= 21:
        return "18-21"
    if age <= 25:
        return "22-25"
    if age <= 29:
        return "26-29"
    if age <= 33:
        return "30-33"
    return "34+"


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rate_per_90(count: float, minutes: float) -> float:
    return (count / minutes * 90.0) if minutes > 0 else 0.0


def _add_window_features(row: dict[str, Any], payload: dict[str, Any], window: int) -> None:
    appearances = max(_safe_float(payload.get(f"appearances_{window}")), 0.0)
    minutes = max(_safe_float(payload.get(f"minutes_{window}")), 0.0)
    goals = max(_safe_float(payload.get(f"goals_{window}")), 0.0)
    assists = max(_safe_float(payload.get(f"assists_{window}")), 0.0)
    yellow_cards = max(_safe_float(payload.get(f"yellow_cards_{window}")), 0.0)
    red_cards = max(_safe_float(payload.get(f"red_cards_{window}")), 0.0)
    goal_contributions = goals + assists

    row[f"appearances_{window}"] = appearances
    row[f"minutes_{window}"] = minutes
    row[f"goals_{window}"] = goals
    row[f"assists_{window}"] = assists
    row[f"yellow_cards_{window}"] = yellow_cards
    row[f"red_cards_{window}"] = red_cards
    row[f"goal_contributions_{window}"] = goal_contributions
    row[f"goals_per_90_{window}"] = _rate_per_90(goals, minutes)
    row[f"assists_per_90_{window}"] = _rate_per_90(assists, minutes)
    row[f"goal_contributions_per_90_{window}"] = _rate_per_90(
        goal_contributions, minutes
    )
    row[f"minutes_per_appearance_{window}"] = (
        minutes / appearances if appearances > 0 else 0.0
    )
    row[f"played_any_{window}"] = 1 if appearances > 0 else 0


def build_player_input(
    payload: dict[str, Any], expected_features: list[str]
) -> pd.DataFrame:
    """Build a one-row dataframe with exactly the columns the model expects."""
    expected = [col for col in expected_features if col not in TARGET_COLUMNS]

    current_value = _safe_float(payload.get("current_market_value_eur"))
    if current_value <= 0:
        raise ValueError("current_market_value_eur must be > 0.")

    age = _safe_float(payload.get("age_at_valuation"), default=np.nan)
    row: dict[str, Any] = {
        "market_value_in_eur": current_value,
        "log_market_value": np.log1p(current_value),
        "age_at_valuation": age,
        "age_bucket": payload.get("age_bucket") or infer_age_bucket(age),
        "height_in_cm": _safe_float(payload.get("height_in_cm"), default=np.nan),
        "position": payload.get("position"),
        "sub_position": payload.get("sub_position"),
        "foot": payload.get("foot"),
        "country_of_citizenship": payload.get("country_of_citizenship"),
        "competition_name": payload.get("competition_name"),
        "competition_country_name": payload.get("competition_country_name"),
        "competition_type": payload.get("competition_type"),
        "competition_sub_type": payload.get("competition_sub_type"),
        "competition_confederation": payload.get("competition_confederation"),
        "club_squad_size": _safe_float(payload.get("club_squad_size"), default=np.nan),
        "club_average_age": _safe_float(payload.get("club_average_age"), default=np.nan),
        "club_foreigners_ratio": _safe_float(
            payload.get("club_foreigners_ratio"), default=np.nan
        ),
        "club_national_team_players_ratio": _safe_float(
            payload.get("club_national_team_players_ratio"), default=np.nan
        ),
        "club_stadium_seats": _safe_float(
            payload.get("club_stadium_seats"), default=np.nan
        ),
        "last_season": _safe_float(payload.get("last_season"), default=np.nan),
        "club_last_season": _safe_float(
            payload.get("club_last_season"), default=np.nan
        ),
        "international_caps": _safe_float(
            payload.get("international_caps"), default=np.nan
        ),
        "international_goals": _safe_float(
            payload.get("international_goals"), default=np.nan
        ),
        "contract_days_remaining": _safe_float(
            payload.get("contract_days_remaining"), default=np.nan
        ),
        "club_foreigners_number": _safe_float(
            payload.get("club_foreigners_number"), default=np.nan
        ),
        "club_foreigners_percentage": _safe_float(
            payload.get("club_foreigners_percentage"), default=np.nan
        ),
        "club_national_team_players": _safe_float(
            payload.get("club_national_team_players"), default=np.nan
        ),
        "total_clubs": _safe_float(payload.get("total_clubs"), default=np.nan),
    }

    for window in (90, 180, 365):
        _add_window_features(row, payload, window)

    final_row = {}
    for col in expected:
        if col in row:
            final_row[col] = row[col]
        elif col in FALLBACK_NUMERIC_FEATURES:
            final_row[col] = np.nan
        else:
            final_row[col] = payload.get(col, np.nan)

    return pd.DataFrame([final_row], columns=expected)


def _trend_label(change_pct: float) -> str:
    if change_pct >= 0.15:
        return "Strong increase"
    if change_pct >= 0.05:
        return "Moderate increase"
    if change_pct > -0.05:
        return "Stable"
    if change_pct > -0.15:
        return "Moderate decrease"
    return "Strong decrease"


def predict_next_value(payload: dict[str, Any]) -> dict[str, Any]:
    """Predict the next market value for a single fictional player."""
    model, preprocessor = load_artifacts()
    expected_features = get_expected_model_features(preprocessor)
    input_df = build_player_input(payload, expected_features)

    X = preprocessor.transform(input_df)
    X = np.asarray(X, dtype="float32")
    pred_residual = float(model.predict(X, verbose=0).reshape(-1)[0])

    current_value = _safe_float(payload.get("current_market_value_eur"))
    previous_value = _safe_float(payload.get("previous_market_value_eur"))
    log_market_value = float(input_df["log_market_value"].iloc[0])
    pred_log_next = log_market_value + pred_residual
    pred_next_value = max(float(np.expm1(pred_log_next)), 0.0)
    change_abs = pred_next_value - current_value
    change_pct = change_abs / current_value if current_value > 0 else np.nan

    return {
        "player_name": payload.get("player_name", "Fictional player"),
        "previous_market_value_eur": previous_value,
        "current_market_value_eur": current_value,
        "predicted_next_market_value_eur": pred_next_value,
        "expected_change_abs": change_abs,
        "expected_change_pct": change_pct,
        "predicted_residual_log": pred_residual,
        "trend_label": _trend_label(change_pct),
        "model_input_columns": list(input_df.columns),
    }


def format_eur(value: float) -> str:
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000:
        return f"{sign}€{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{sign}€{value / 1_000:.0f}K"
    return f"{sign}€{value:.0f}"
