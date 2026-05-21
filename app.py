from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from src.predict_single import format_eur, predict_next_value


st.set_page_config(
    page_title="MarketScout DL",
    page_icon="MS",
    layout="wide",
)


FICTIONAL_PROFILES = {
    "Young breakout winger": {
        "player_name": "Leo Marin",
        "age_at_valuation": 20,
        "position": "Attack",
        "sub_position": "Left Winger",
        "foot": "right",
        "height_in_cm": 178,
        "country_of_citizenship": "Spain",
        "previous_market_value_eur": 7_000_000,
        "current_market_value_eur": 12_000_000,
        "minutes_365": 2200,
        "appearances_365": 34,
        "goals_365": 12,
        "assists_365": 9,
        "competition_name": "premier-league",
        "competition_country_name": "England",
    },
    "Prime midfielder": {
        "player_name": "Mateo Alvarez",
        "age_at_valuation": 26,
        "position": "Midfield",
        "sub_position": "Central Midfield",
        "foot": "both",
        "height_in_cm": 183,
        "country_of_citizenship": "Argentina",
        "previous_market_value_eur": 32_000_000,
        "current_market_value_eur": 35_000_000,
        "minutes_365": 2800,
        "appearances_365": 40,
        "goals_365": 6,
        "assists_365": 11,
        "competition_name": "laliga",
        "competition_country_name": "Spain",
    },
    "Aging striker": {
        "player_name": "Marco Bellini",
        "age_at_valuation": 33,
        "position": "Attack",
        "sub_position": "Centre-Forward",
        "foot": "right",
        "height_in_cm": 186,
        "country_of_citizenship": "Italy",
        "previous_market_value_eur": 14_000_000,
        "current_market_value_eur": 9_000_000,
        "minutes_365": 1600,
        "appearances_365": 28,
        "goals_365": 10,
        "assists_365": 2,
        "competition_name": "serie-a",
        "competition_country_name": "Italy",
    },
    "Young goalkeeper": {
        "player_name": "Jonas Weber",
        "age_at_valuation": 22,
        "position": "Goalkeeper",
        "sub_position": "Goalkeeper",
        "foot": "left",
        "height_in_cm": 193,
        "country_of_citizenship": "Germany",
        "previous_market_value_eur": 5_000_000,
        "current_market_value_eur": 8_000_000,
        "minutes_365": 2700,
        "appearances_365": 30,
        "goals_365": 0,
        "assists_365": 0,
        "competition_name": "bundesliga",
        "competition_country_name": "Germany",
    },
}

DEFAULTS = {
    "player_name": "Leo Marin",
    "age_at_valuation": 20,
    "position": "Attack",
    "sub_position": "Left Winger",
    "foot": "right",
    "height_in_cm": 178,
    "country_of_citizenship": "Spain",
    "previous_market_value_eur": 7_000_000,
    "current_market_value_eur": 12_000_000,
    "minutes_365": 2200,
    "appearances_365": 34,
    "goals_365": 12,
    "assists_365": 9,
    "yellow_cards_365": 3,
    "red_cards_365": 0,
    "minutes_180": 1200,
    "appearances_180": 18,
    "goals_180": 7,
    "assists_180": 5,
    "yellow_cards_180": 1,
    "red_cards_180": 0,
    "minutes_90": 650,
    "appearances_90": 9,
    "goals_90": 4,
    "assists_90": 3,
    "yellow_cards_90": 0,
    "red_cards_90": 0,
    "competition_name": "premier-league",
    "competition_country_name": "England",
    "competition_type": "domestic_league",
    "competition_sub_type": "first_tier",
    "competition_confederation": "uefa",
    "club_squad_size": 25,
    "club_average_age": 26.2,
    "club_foreigners_ratio": 0.55,
    "club_national_team_players_ratio": 0.28,
    "club_stadium_seats": 52000,
}

POSITIONS = ["Attack", "Midfield", "Defender", "Goalkeeper"]
SUB_POSITIONS = [
    "Centre-Forward",
    "Left Winger",
    "Right Winger",
    "Attacking Midfield",
    "Central Midfield",
    "Defensive Midfield",
    "Centre-Back",
    "Left-Back",
    "Right-Back",
    "Goalkeeper",
]
COMPETITIONS = [
    "premier-league",
    "laliga",
    "serie-a",
    "bundesliga",
    "ligue-1",
    "eredivisie",
    "liga-portugal",
    "super-lig",
    "liga-mx",
]


def init_state() -> None:
    if "profile" not in st.session_state:
        st.session_state.profile = DEFAULTS.copy()


def load_profile(name: str) -> None:
    profile = DEFAULTS.copy()
    profile.update(FICTIONAL_PROFILES[name])
    minutes_365 = profile["minutes_365"]
    apps_365 = profile["appearances_365"]
    goals_365 = profile["goals_365"]
    assists_365 = profile["assists_365"]
    profile.update(
        {
            "minutes_180": int(minutes_365 * 0.55),
            "appearances_180": max(1, int(apps_365 * 0.55)),
            "goals_180": int(math.ceil(goals_365 * 0.55)),
            "assists_180": int(math.ceil(assists_365 * 0.55)),
            "minutes_90": int(minutes_365 * 0.30),
            "appearances_90": max(1, int(apps_365 * 0.30)),
            "goals_90": int(math.ceil(goals_365 * 0.30)),
            "assists_90": int(math.ceil(assists_365 * 0.30)),
        }
    )
    st.session_state.profile = profile


def number_input(label: str, key: str, **kwargs):
    value = st.session_state.profile.get(key, DEFAULTS.get(key, 0))
    return st.number_input(label, value=value, key=f"input_{key}", **kwargs)


def text_input(label: str, key: str):
    value = st.session_state.profile.get(key, DEFAULTS.get(key, ""))
    return st.text_input(label, value=value, key=f"input_{key}")


def select_input(label: str, key: str, options: list[str]):
    value = st.session_state.profile.get(key, DEFAULTS.get(key, options[0]))
    index = options.index(value) if value in options else 0
    return st.selectbox(label, options, index=index, key=f"input_{key}")


def validate_payload(payload: dict) -> list[str]:
    errors: list[str] = []
    if payload["current_market_value_eur"] <= 0:
        errors.append("Current market value must be greater than 0.")
    if payload["previous_market_value_eur"] < 0:
        errors.append("Previous market value must be >= 0.")
    if not 15 <= payload["age_at_valuation"] <= 45:
        errors.append("Age must be between 15 and 45.")
    for key in (
        "appearances_90",
        "goals_90",
        "assists_90",
        "appearances_180",
        "goals_180",
        "assists_180",
        "appearances_365",
        "goals_365",
        "assists_365",
    ):
        if payload[key] < 0:
            errors.append(f"{key} must be non-negative.")
    for key in ("club_foreigners_ratio", "club_national_team_players_ratio"):
        if not 0 <= payload[key] <= 1:
            errors.append(f"{key} must be between 0 and 1.")
    return errors


def trend_progress(change_pct: float) -> float:
    clipped = max(min(change_pct, 0.30), -0.30)
    return (clipped + 0.30) / 0.60


init_state()

st.markdown(
    """
    <style>
    .main-title {font-size: 2.4rem; font-weight: 800; margin-bottom: 0;}
    .subtitle {color: #5c6670; font-size: 1.05rem; margin-top: 0.2rem;}
    .result-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        background: #ffffff;
    }
    .trend-pill {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: #0f172a;
        color: white;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">MarketScout DL</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Predicting football player market value with residual deep learning</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model")
    st.write(
        "Residual MLP trained on historical Transfermarkt valuations. "
        "It predicts the log-value change from current value, then reconstructs "
        "the next estimated market value."
    )
    profile_name = st.selectbox("Fictional profile", list(FICTIONAL_PROFILES))
    if st.button("Load fictional profile", use_container_width=True):
        load_profile(profile_name)
        st.rerun()
    st.caption("All example players are fictional and editable.")

payload = {}

tab_profile, tab_market, tab_perf, tab_context = st.tabs(
    ["Player profile", "Market values", "Performance", "Club & competition"]
)

with tab_profile:
    c1, c2, c3 = st.columns(3)
    with c1:
        payload["player_name"] = text_input("Player name", "player_name")
        payload["age_at_valuation"] = number_input(
            "Age", "age_at_valuation", min_value=15, max_value=45, step=1
        )
        payload["position"] = select_input("Position", "position", POSITIONS)
    with c2:
        payload["sub_position"] = select_input(
            "Sub-position", "sub_position", SUB_POSITIONS
        )
        payload["foot"] = select_input("Foot", "foot", ["right", "left", "both"])
        payload["height_in_cm"] = number_input(
            "Height (cm)", "height_in_cm", min_value=140, max_value=220, step=1
        )
    with c3:
        payload["country_of_citizenship"] = text_input(
            "Country of citizenship", "country_of_citizenship"
        )

with tab_market:
    c1, c2, c3 = st.columns(3)
    with c1:
        payload["previous_market_value_eur"] = number_input(
            "Previous market value EUR",
            "previous_market_value_eur",
            min_value=0,
            step=250_000,
        )
    with c2:
        payload["current_market_value_eur"] = number_input(
            "Current market value EUR",
            "current_market_value_eur",
            min_value=1,
            step=250_000,
        )
    with c3:
        prev = payload["previous_market_value_eur"]
        curr = payload["current_market_value_eur"]
        prev_change = (curr - prev) / prev if prev > 0 else np.nan
        st.metric(
            "Previous to current change",
            "n/a" if np.isnan(prev_change) else f"{prev_change:.1%}",
        )
        st.metric("Current log value", f"{np.log1p(curr):.3f}")

with tab_perf:
    st.subheader("Performance last 365 days")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        payload["minutes_365"] = number_input(
            "Minutes 365", "minutes_365", min_value=0, step=90
        )
    with c2:
        payload["appearances_365"] = number_input(
            "Appearances 365", "appearances_365", min_value=0, step=1
        )
    with c3:
        payload["goals_365"] = number_input(
            "Goals 365", "goals_365", min_value=0, step=1
        )
    with c4:
        payload["assists_365"] = number_input(
            "Assists 365", "assists_365", min_value=0, step=1
        )
    with c5:
        payload["yellow_cards_365"] = number_input(
            "Yellow cards 365", "yellow_cards_365", min_value=0, step=1
        )
    with c6:
        payload["red_cards_365"] = number_input(
            "Red cards 365", "red_cards_365", min_value=0, step=1
        )

    st.subheader("Performance last 180 days")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        payload["minutes_180"] = number_input(
            "Minutes 180", "minutes_180", min_value=0, step=90
        )
    with c2:
        payload["appearances_180"] = number_input(
            "Appearances 180", "appearances_180", min_value=0, step=1
        )
    with c3:
        payload["goals_180"] = number_input(
            "Goals 180", "goals_180", min_value=0, step=1
        )
    with c4:
        payload["assists_180"] = number_input(
            "Assists 180", "assists_180", min_value=0, step=1
        )

    st.subheader("Performance last 90 days")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        payload["minutes_90"] = number_input(
            "Minutes 90", "minutes_90", min_value=0, step=90
        )
    with c2:
        payload["appearances_90"] = number_input(
            "Appearances 90", "appearances_90", min_value=0, step=1
        )
    with c3:
        payload["goals_90"] = number_input("Goals 90", "goals_90", min_value=0, step=1)
    with c4:
        payload["assists_90"] = number_input(
            "Assists 90", "assists_90", min_value=0, step=1
        )

    if not (
        payload["minutes_365"] >= payload["minutes_180"] >= payload["minutes_90"]
    ):
        st.warning("Minutes are not ordered as 365 >= 180 >= 90. Prediction is still allowed.")

with tab_context:
    c1, c2, c3 = st.columns(3)
    with c1:
        payload["competition_name"] = select_input(
            "Competition", "competition_name", COMPETITIONS
        )
        payload["competition_country_name"] = text_input(
            "Competition country", "competition_country_name"
        )
        payload["competition_type"] = text_input(
            "Competition type", "competition_type"
        )
    with c2:
        payload["competition_sub_type"] = text_input(
            "Competition sub-type", "competition_sub_type"
        )
        payload["competition_confederation"] = text_input(
            "Competition confederation", "competition_confederation"
        )
        payload["club_squad_size"] = number_input(
            "Club squad size", "club_squad_size", min_value=1, max_value=80, step=1
        )
    with c3:
        payload["club_average_age"] = number_input(
            "Club average age", "club_average_age", min_value=15.0, max_value=40.0, step=0.1
        )
        payload["club_foreigners_ratio"] = number_input(
            "Club foreigners ratio",
            "club_foreigners_ratio",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        )
        payload["club_national_team_players_ratio"] = number_input(
            "National team players ratio",
            "club_national_team_players_ratio",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        )
        payload["club_stadium_seats"] = number_input(
            "Club stadium seats", "club_stadium_seats", min_value=0, step=1000
        )

errors = validate_payload(payload)
for error in errors:
    st.error(error)

predict = st.button(
    "Predict next market value",
    type="primary",
    use_container_width=True,
    disabled=bool(errors),
)

if predict:
    try:
        result = predict_next_value(payload)
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
    else:
        st.divider()
        st.markdown(f"### Prediction for {result['player_name']}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Current value", format_eur(result["current_market_value_eur"]))
        c2.metric(
            "Predicted next value",
            format_eur(result["predicted_next_market_value_eur"]),
        )
        c3.metric("Expected change", format_eur(result["expected_change_abs"]))
        c4.metric("Expected change %", f"{result['expected_change_pct']:.1%}")
        c5.markdown(
            f"<div class='result-card'><span class='trend-pill'>{result['trend_label']}</span></div>",
            unsafe_allow_html=True,
        )

        st.progress(
            trend_progress(result["expected_change_pct"]),
            text="Change scale: -30% to +30%",
        )

        summary = pd.DataFrame(
            [
                {
                    "player_name": payload["player_name"],
                    "age": payload["age_at_valuation"],
                    "position": payload["position"],
                    "current_value": format_eur(payload["current_market_value_eur"]),
                    "minutes_365": payload["minutes_365"],
                    "goals_365": payload["goals_365"],
                    "assists_365": payload["assists_365"],
                    "competition": payload["competition_name"],
                    "predicted_residual_log": round(result["predicted_residual_log"], 4),
                }
            ]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.warning(
            "This is a demo using a residual DL model trained on Transfermarkt "
            "historical valuations. It estimates perceived market value, not actual transfer price."
        )
