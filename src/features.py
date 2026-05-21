"""Sprint 3 — Player profile features.

Joins `valuation_targets.parquet` with `players.csv(.gz)` to attach
profile features (age, position, foot, country, contract status...) to
each (player_id, valuation_date) row. No performance features yet.

Public API:
    standardize_players_columns(players)        -> pd.DataFrame
    prepare_players(players)                    -> pd.DataFrame
    add_player_profile_features(targets, players) -> pd.DataFrame
    validate_player_profile_dataset(df, expected_rows) -> None
    build_player_profile_summary(df)            -> dict
    save_player_profile_dataset(df, output)     -> None
    save_player_profile_summary(summary, output) -> None
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.load_data import (
    COMPETITION_ID_CANDIDATES,
    DATE_CANDIDATES,
    PLAYER_ID_CANDIDATES,
    _first_present as first_present,
)

PLAYERS_DESIRABLE = (
    "name",
    "first_name",
    "last_name",
    "date_of_birth",
    "position",
    "sub_position",
    "foot",
    "height_in_cm",
    "country_of_birth",
    "country_of_citizenship",
    "current_club_id",
    "current_club_name",
    "contract_expiration_date",
    "agent_name",
)

PLAYERS_FORBIDDEN = ("market_value_in_eur", "highest_market_value_in_eur")

AGE_BUCKETS = ["U18", "18-21", "22-25", "26-29", "30-33", "34+"]
AGE_BINS = [-np.inf, 18, 22, 26, 30, 34, np.inf]

GAME_ID_CANDIDATES = ("game_id", "gameID", "gameId", "match_id", "matchID")
MINUTES_CANDIDATES = ("minutes_played", "minutes", "playing_time")
GOALS_CANDIDATES = ("goals", "goal")
ASSISTS_CANDIDATES = ("assists", "assist")
YELLOW_CARDS_CANDIDATES = ("yellow_cards", "yellow_card", "yellowcards")
RED_CARDS_CANDIDATES = ("red_cards", "red_card", "redcards")
PLAYER_CLUB_ID_CANDIDATES = ("player_club_id", "club_id", "playerClubId")

PERFORMANCE_WINDOWS = (90, 180, 365)
PERFORMANCE_BASE_STATS = (
    "appearances",
    "minutes",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
    "goal_contributions",
    "goals_per_90",
    "assists_per_90",
    "goal_contributions_per_90",
    "minutes_per_appearance",
    "played_any",
)
PERFORMANCE_SUM_FEATURES = (
    "appearances",
    "minutes",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
    "goal_contributions",
)


def standardize_players_columns(players: pd.DataFrame) -> pd.DataFrame:
    """Rename `player_id` variants to canonical name; raise if absent."""
    out = players.copy()
    if "player_id" not in out.columns:
        found = first_present(out, PLAYER_ID_CANDIDATES)
        if found is None:
            raise ValueError(
                "players: missing required column 'player_id' "
                f"(looked for {list(PLAYER_ID_CANDIDATES)})."
            )
        out = out.rename(columns={found: "player_id"})
    return out


def prepare_players(players: pd.DataFrame) -> pd.DataFrame:
    """Standardize, dedupe, type-coerce, and strip leakage-prone columns."""
    out = standardize_players_columns(players)

    drop_cols = [c for c in PLAYERS_FORBIDDEN if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    if "date_of_birth" in out.columns:
        out["date_of_birth"] = pd.to_datetime(
            out["date_of_birth"], errors="coerce"
        )
    if "contract_expiration_date" in out.columns:
        out["contract_expiration_date"] = pd.to_datetime(
            out["contract_expiration_date"], errors="coerce"
        )
    if "height_in_cm" in out.columns:
        out["height_in_cm"] = pd.to_numeric(
            out["height_in_cm"], errors="coerce"
        )

    string_cols = [
        c
        for c in (
            "name",
            "first_name",
            "last_name",
            "position",
            "sub_position",
            "foot",
            "country_of_birth",
            "country_of_citizenship",
            "current_club_name",
            "agent_name",
        )
        if c in out.columns
    ]
    for col in string_cols:
        out[col] = out[col].astype("string").str.strip()
        out[col] = out[col].mask(out[col] == "", pd.NA)

    out = out.dropna(subset=["player_id"])
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce")
    out = out.dropna(subset=["player_id"])
    out["player_id"] = out["player_id"].astype("int64")

    sort_cols = ["player_id"]
    if "date_of_birth" in out.columns:
        sort_cols.append("date_of_birth")
    out = out.sort_values(sort_cols, na_position="last")
    out = out.drop_duplicates(subset=["player_id"], keep="first")
    out = out.reset_index(drop=True)
    return out


def _build_player_name(df: pd.DataFrame) -> pd.Series:
    if "name" in df.columns:
        primary = df["name"].astype("string").str.strip()
    else:
        primary = pd.Series(pd.NA, index=df.index, dtype="string")
    has_first = "first_name" in df.columns
    has_last = "last_name" in df.columns
    if has_first or has_last:
        first = (
            df["first_name"].astype("string").str.strip()
            if has_first
            else pd.Series("", index=df.index, dtype="string")
        )
        last = (
            df["last_name"].astype("string").str.strip()
            if has_last
            else pd.Series("", index=df.index, dtype="string")
        )
        combined = (first.fillna("") + " " + last.fillna("")).str.strip()
        combined = combined.mask(combined == "", pd.NA)
        primary = primary.where(primary.notna() & (primary != ""), combined)
    primary = primary.mask(primary == "", pd.NA)
    return primary


def _bucket_age(age: pd.Series) -> pd.Series:
    bucketed = pd.cut(age, bins=AGE_BINS, labels=AGE_BUCKETS, right=False)
    return bucketed.astype("string")


def add_player_profile_features(
    targets: pd.DataFrame, players: pd.DataFrame
) -> pd.DataFrame:
    """Left-join targets <- prepared players, then derive profile features."""
    if "player_id" not in targets.columns:
        raise ValueError("targets: missing 'player_id'")
    n_in = len(targets)

    overlap = (set(players.columns) & set(targets.columns)) - {"player_id"}
    players_safe = players.drop(columns=sorted(overlap), errors="ignore")

    out = targets.merge(players_safe, on="player_id", how="left", validate="many_to_one")
    if len(out) != n_in:
        raise RuntimeError(
            f"Join changed row count: {n_in} -> {len(out)}. "
            "Players had duplicate player_id values?"
        )

    out["player_name"] = _build_player_name(out)

    if "date_of_birth" in out.columns:
        delta_days = (out["valuation_date"] - out["date_of_birth"]).dt.days
        age = delta_days / 365.25
        implausible = ((age < 10) | (age > 60)).fillna(False)
        n_bad = int(implausible.sum())
        if n_bad:
            print(
                f"  WARNING: {n_bad} rows have implausible age "
                "(<10 or >60); likely bad date_of_birth in source. "
                "Setting age_at_valuation to NaN for those rows."
            )
        age = age.mask(implausible)
        out["age_at_valuation"] = age
        out["age_bucket"] = _bucket_age(out["age_at_valuation"])

    if "contract_expiration_date" in out.columns:
        out["contract_days_remaining"] = (
            out["contract_expiration_date"] - out["valuation_date"]
        ).dt.days
        out["is_contract_expired_at_valuation"] = (
            out["contract_expiration_date"] < out["valuation_date"]
        )
        out["is_contract_expired_at_valuation"] = out[
            "is_contract_expired_at_valuation"
        ].mask(out["contract_expiration_date"].isna(), pd.NA)

    return out


def validate_player_profile_dataset(
    df: pd.DataFrame, expected_rows: int
) -> None:
    """Raise ValueError on row-count drift, missing keys, dups, leakage."""
    if len(df) != expected_rows:
        raise ValueError(
            f"Row count mismatch: expected {expected_rows}, got {len(df)}."
        )
    required = ("player_id", "valuation_date", "next_valuation_date",
                "next_market_value_in_eur")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    dup_keys = ["player_id", "valuation_date", "next_valuation_date"]
    n_dups = int(df.duplicated(subset=dup_keys).sum())
    if n_dups:
        raise ValueError(f"Found {n_dups} duplicate rows on {dup_keys}.")

    if "age_at_valuation" in df.columns:
        ages = df["age_at_valuation"].dropna()
        n_bad = int(((ages < 10) | (ages > 60)).sum())
        if n_bad:
            raise ValueError(
                f"Found {n_bad} rows with implausible age_at_valuation "
                "(<10 or >60)."
            )

    forbidden = (
        "highest_market_value_in_eur",
        "players_market_value_in_eur",
        "market_value_in_eur_y",
    )
    leaked = [c for c in forbidden if c in df.columns]
    if leaked:
        raise ValueError(f"Forbidden leakage columns present: {leaked}")


def _value_counts_dict(series: pd.Series, top: int | None = None) -> dict:
    vc = series.dropna().value_counts()
    if top is not None:
        vc = vc.head(top)
    return {str(k): int(v) for k, v in vc.items()}


def build_player_profile_summary(df: pd.DataFrame) -> dict:
    """Aggregate descriptive stats for the joined dataset."""
    n = len(df)
    summary: dict = {
        "rows": int(n),
        "unique_players": int(df["player_id"].nunique()),
        "min_valuation_date": df["valuation_date"].min().date().isoformat(),
        "max_valuation_date": df["valuation_date"].max().date().isoformat(),
        "missing_player_name_pct": (
            round(float(df["player_name"].isna().mean() * 100), 2)
            if "player_name" in df.columns
            else None
        ),
    }

    if "age_at_valuation" in df.columns:
        ages = df["age_at_valuation"]
        summary["missing_age_at_valuation_pct"] = round(
            float(ages.isna().mean() * 100), 2
        )
        non_null = ages.dropna()
        if not non_null.empty:
            summary["min_age_at_valuation"] = round(float(non_null.min()), 2)
            summary["median_age_at_valuation"] = round(
                float(non_null.median()), 2
            )
            summary["max_age_at_valuation"] = round(float(non_null.max()), 2)
        summary["age_bucket_counts"] = _value_counts_dict(df.get("age_bucket"))

    if "position" in df.columns:
        summary["position_counts"] = _value_counts_dict(df["position"])
    if "sub_position" in df.columns:
        summary["sub_position_top_20"] = _value_counts_dict(
            df["sub_position"], top=20
        )
    if "foot" in df.columns:
        summary["foot_counts"] = _value_counts_dict(df["foot"])
    if "country_of_citizenship" in df.columns:
        summary["country_top_20"] = _value_counts_dict(
            df["country_of_citizenship"], top=20
        )

    if "height_in_cm" in df.columns:
        summary["missing_height_pct"] = round(
            float(df["height_in_cm"].isna().mean() * 100), 2
        )
        median_h = df["height_in_cm"].median()
        summary["median_height_in_cm"] = (
            float(median_h) if pd.notna(median_h) else None
        )

    if "contract_expiration_date" in df.columns:
        summary["contract_available_pct"] = round(
            float(df["contract_expiration_date"].notna().mean() * 100), 2
        )
    if "is_contract_expired_at_valuation" in df.columns:
        flag = df["is_contract_expired_at_valuation"]
        non_null = flag.dropna()
        if not non_null.empty:
            summary["contract_expired_pct"] = round(
                float(non_null.astype(bool).mean() * 100), 2
            )
        else:
            summary["contract_expired_pct"] = None

    return summary


def save_player_profile_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Persist as parquet (no CSV mirror — file is large)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Player profile dataset written to {output_path.resolve()}")


def save_player_profile_summary(summary: dict, output_path: Path) -> None:
    """Persist the summary dict as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Player profile summary written to {output_path.resolve()}")


def _rename_first_present(
    df: pd.DataFrame, candidates: tuple[str, ...], canonical: str
) -> pd.DataFrame:
    """Rename the first candidate column to `canonical` when needed."""
    if canonical in df.columns:
        return df
    found = first_present(df, candidates)
    if found is None:
        return df
    return df.rename(columns={found: canonical})


def _require_columns(df: pd.DataFrame, table: str, columns: tuple[str, ...]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{table}: missing required columns: {missing}.")


def _performance_feature_columns(windows=PERFORMANCE_WINDOWS) -> list[str]:
    return [
        f"{stat}_{window}"
        for window in windows
        for stat in PERFORMANCE_BASE_STATS
    ]


def standardize_appearances_columns(appearances: pd.DataFrame) -> pd.DataFrame:
    """Normalize key/stat columns in appearances; require player_id and game_id."""
    out = appearances.copy()
    rename_specs = (
        (PLAYER_ID_CANDIDATES, "player_id"),
        (GAME_ID_CANDIDATES, "game_id"),
        (MINUTES_CANDIDATES, "minutes_played"),
        (GOALS_CANDIDATES, "goals"),
        (ASSISTS_CANDIDATES, "assists"),
        (YELLOW_CARDS_CANDIDATES, "yellow_cards"),
        (RED_CARDS_CANDIDATES, "red_cards"),
        (PLAYER_CLUB_ID_CANDIDATES, "player_club_id"),
        (COMPETITION_ID_CANDIDATES, "competition_id"),
        (DATE_CANDIDATES, "date"),
    )
    for candidates, canonical in rename_specs:
        out = _rename_first_present(out, candidates, canonical)

    _require_columns(out, "appearances", ("player_id", "game_id"))
    return out


def standardize_games_columns(games: pd.DataFrame) -> pd.DataFrame:
    """Normalize game columns; require game_id and date."""
    out = games.copy()
    rename_specs = (
        (GAME_ID_CANDIDATES, "game_id"),
        (DATE_CANDIDATES, "date"),
        (COMPETITION_ID_CANDIDATES, "competition_id"),
    )
    for candidates, canonical in rename_specs:
        out = _rename_first_present(out, candidates, canonical)

    _require_columns(out, "games", ("game_id", "date"))
    return out


def prepare_appearances_with_games(
    appearances: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    """Return one clean row per player/game with historical performance stats."""
    apps = standardize_appearances_columns(appearances)
    games_std = standardize_games_columns(games)

    app_cols = [
        c
        for c in (
            "player_id",
            "game_id",
            "date",
            "minutes_played",
            "goals",
            "assists",
            "yellow_cards",
            "red_cards",
            "player_club_id",
            "competition_id",
        )
        if c in apps.columns
    ]
    apps = apps[app_cols].copy()

    game_cols = [
        c
        for c in (
            "game_id",
            "date",
            "competition_id",
            "season",
            "home_club_id",
            "away_club_id",
            "home_club_goals",
            "away_club_goals",
        )
        if c in games_std.columns
    ]
    games_std = games_std[game_cols].copy()
    games_std = games_std.rename(columns={"date": "game_date_from_games"})
    if "competition_id" in games_std.columns:
        games_std = games_std.rename(
            columns={"competition_id": "competition_id_from_games"}
        )

    if "date" in apps.columns:
        apps["game_date_from_appearances"] = pd.to_datetime(
            apps["date"], errors="coerce"
        )
        apps = apps.drop(columns=["date"])
    else:
        apps["game_date_from_appearances"] = pd.NaT

    games_std["game_date_from_games"] = pd.to_datetime(
        games_std["game_date_from_games"], errors="coerce"
    )
    games_std = games_std.drop_duplicates(subset=["game_id"], keep="first")

    out = apps.merge(games_std, on="game_id", how="left", validate="many_to_one")
    out["game_date"] = out["game_date_from_games"].combine_first(
        out["game_date_from_appearances"]
    )
    out = out.drop(
        columns=["game_date_from_games", "game_date_from_appearances"],
        errors="ignore",
    )

    if "competition_id" not in out.columns and "competition_id_from_games" in out.columns:
        out = out.rename(columns={"competition_id_from_games": "competition_id"})
    elif "competition_id_from_games" in out.columns:
        out["competition_id"] = out["competition_id"].combine_first(
            out["competition_id_from_games"]
        )
        out = out.drop(columns=["competition_id_from_games"])

    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce")
    out["game_id"] = pd.to_numeric(out["game_id"], errors="coerce")

    numeric_stats = (
        "minutes_played",
        "goals",
        "assists",
        "yellow_cards",
        "red_cards",
    )
    for col in numeric_stats:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out = out.dropna(subset=["player_id", "game_id", "game_date"])
    out["player_id"] = out["player_id"].astype("int64")
    out["game_id"] = out["game_id"].astype("int64")
    out["goal_contributions"] = out["goals"] + out["assists"]

    group_cols = ["player_id", "game_id", "game_date"]
    optional_first_cols = [
        c
        for c in (
            "player_club_id",
            "competition_id",
            "season",
            "home_club_id",
            "away_club_id",
            "home_club_goals",
            "away_club_goals",
        )
        if c in out.columns
    ]
    aggregations = {col: "sum" for col in (*numeric_stats, "goal_contributions")}
    aggregations.update({col: "first" for col in optional_first_cols})

    out = (
        out.groupby(group_cols, as_index=False, sort=False)
        .agg(aggregations)
        .sort_values(["player_id", "game_date", "game_id"])
        .reset_index(drop=True)
    )
    return out


def aggregate_player_performance_windows(
    base_df: pd.DataFrame,
    appearances_games: pd.DataFrame,
    windows=(90, 180, 365),
) -> pd.DataFrame:
    """Add strictly historical rolling-window performance features."""
    required_base = ("player_id", "valuation_date")
    required_apps = (
        "player_id",
        "game_date",
        "minutes_played",
        "goals",
        "assists",
        "yellow_cards",
        "red_cards",
        "goal_contributions",
    )
    _require_columns(base_df, "base_df", required_base)
    _require_columns(appearances_games, "appearances_games", required_apps)

    out = base_df.copy()
    out["valuation_date"] = pd.to_datetime(out["valuation_date"], errors="coerce")
    if out["valuation_date"].isna().any():
        raise ValueError("base_df: valuation_date contains null/unparseable values.")

    feature_cols = _performance_feature_columns(windows)
    feature_values = {
        col: np.zeros(len(out), dtype="int8" if col.startswith("played_any_") else "float64")
        for col in feature_cols
    }

    apps = appearances_games.copy()
    apps["game_date"] = pd.to_datetime(apps["game_date"], errors="coerce")
    apps = apps.dropna(subset=["player_id", "game_date"])
    apps["player_id"] = pd.to_numeric(apps["player_id"], errors="coerce")
    apps = apps.dropna(subset=["player_id"])
    apps["player_id"] = apps["player_id"].astype(out["player_id"].dtype)
    stat_cols = [
        "minutes_played",
        "goals",
        "assists",
        "yellow_cards",
        "red_cards",
        "goal_contributions",
    ]
    for col in stat_cols:
        apps[col] = pd.to_numeric(apps[col], errors="coerce").fillna(0)
    apps = apps.sort_values(["player_id", "game_date", "game_id"])

    base_ordered = out[["player_id", "valuation_date"]].copy()
    base_ordered["_row_id"] = np.arange(len(out))
    base_groups = base_ordered.sort_values(
        ["player_id", "valuation_date", "_row_id"]
    ).groupby("player_id", sort=False)
    app_groups = {pid: grp for pid, grp in apps.groupby("player_id", sort=False)}

    total_players = len(base_groups)
    print(f"Aggregating performance windows for {total_players} players...")

    for i, (player_id, player_rows) in enumerate(base_groups, start=1):
        player_apps = app_groups.get(player_id)
        if player_apps is None or player_apps.empty:
            continue

        row_ids = player_rows["_row_id"].to_numpy()
        valuation_dates = player_rows["valuation_date"].to_numpy(dtype="datetime64[ns]")
        game_dates = player_apps["game_date"].to_numpy(dtype="datetime64[ns]")

        count_prefix = np.concatenate(
            ([0.0], np.ones(len(player_apps), dtype="float64").cumsum())
        )
        prefixes = {
            "minutes": np.concatenate(
                ([0.0], player_apps["minutes_played"].to_numpy(dtype="float64").cumsum())
            ),
            "goals": np.concatenate(
                ([0.0], player_apps["goals"].to_numpy(dtype="float64").cumsum())
            ),
            "assists": np.concatenate(
                ([0.0], player_apps["assists"].to_numpy(dtype="float64").cumsum())
            ),
            "yellow_cards": np.concatenate(
                ([0.0], player_apps["yellow_cards"].to_numpy(dtype="float64").cumsum())
            ),
            "red_cards": np.concatenate(
                ([0.0], player_apps["red_cards"].to_numpy(dtype="float64").cumsum())
            ),
            "goal_contributions": np.concatenate(
                (
                    [0.0],
                    player_apps["goal_contributions"]
                    .to_numpy(dtype="float64")
                    .cumsum(),
                )
            ),
        }

        for window in windows:
            starts = valuation_dates - np.timedelta64(int(window), "D")
            start_idx = np.searchsorted(game_dates, starts, side="left")
            end_idx = np.searchsorted(game_dates, valuation_dates, side="left")

            used = end_idx > start_idx
            if used.any():
                last_used = game_dates[end_idx[used] - 1]
                if np.any(last_used >= valuation_dates[used]):
                    raise RuntimeError(
                        "Temporal leakage detected during aggregation: "
                        "a selected game_date is >= valuation_date."
                    )

            appearances_sum = count_prefix[end_idx] - count_prefix[start_idx]
            minutes_sum = prefixes["minutes"][end_idx] - prefixes["minutes"][start_idx]
            goals_sum = prefixes["goals"][end_idx] - prefixes["goals"][start_idx]
            assists_sum = (
                prefixes["assists"][end_idx] - prefixes["assists"][start_idx]
            )
            yellow_sum = (
                prefixes["yellow_cards"][end_idx]
                - prefixes["yellow_cards"][start_idx]
            )
            red_sum = (
                prefixes["red_cards"][end_idx] - prefixes["red_cards"][start_idx]
            )
            gc_sum = (
                prefixes["goal_contributions"][end_idx]
                - prefixes["goal_contributions"][start_idx]
            )

            feature_values[f"appearances_{window}"][row_ids] = appearances_sum
            feature_values[f"minutes_{window}"][row_ids] = minutes_sum
            feature_values[f"goals_{window}"][row_ids] = goals_sum
            feature_values[f"assists_{window}"][row_ids] = assists_sum
            feature_values[f"yellow_cards_{window}"][row_ids] = yellow_sum
            feature_values[f"red_cards_{window}"][row_ids] = red_sum
            feature_values[f"goal_contributions_{window}"][row_ids] = gc_sum
            feature_values[f"goals_per_90_{window}"][row_ids] = np.divide(
                goals_sum * 90.0,
                minutes_sum,
                out=np.zeros_like(goals_sum, dtype="float64"),
                where=minutes_sum > 0,
            )
            feature_values[f"assists_per_90_{window}"][row_ids] = np.divide(
                assists_sum * 90.0,
                minutes_sum,
                out=np.zeros_like(assists_sum, dtype="float64"),
                where=minutes_sum > 0,
            )
            feature_values[f"goal_contributions_per_90_{window}"][row_ids] = np.divide(
                gc_sum * 90.0,
                minutes_sum,
                out=np.zeros_like(gc_sum, dtype="float64"),
                where=minutes_sum > 0,
            )
            feature_values[f"minutes_per_appearance_{window}"][row_ids] = np.divide(
                minutes_sum,
                appearances_sum,
                out=np.zeros_like(minutes_sum, dtype="float64"),
                where=appearances_sum > 0,
            )
            feature_values[f"played_any_{window}"][row_ids] = (
                appearances_sum > 0
            ).astype("int8")

        if i % 5000 == 0:
            print(f"  processed {i:,}/{total_players:,} players")

    flag_cols = [f"played_any_{window}" for window in windows]
    for col in feature_cols:
        out[col] = feature_values[col]
    out[flag_cols] = out[flag_cols].astype("int8")
    return out


def validate_player_performance_dataset(
    df: pd.DataFrame, expected_rows: int
) -> None:
    """Validate row preservation, target preservation, and feature sanity."""
    if len(df) != expected_rows:
        raise ValueError(
            f"Row count mismatch: expected {expected_rows}, got {len(df)}."
        )

    required = (
        "player_id",
        "valuation_date",
        "next_valuation_date",
        "next_market_value_in_eur",
        "log_next_market_value",
    )
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    dup_keys = ["player_id", "valuation_date", "next_valuation_date"]
    n_dups = int(df.duplicated(subset=dup_keys).sum())
    if n_dups:
        raise ValueError(f"Found {n_dups} duplicate rows on {dup_keys}.")

    feature_cols = [
        col
        for col in df.columns
        if any(col.endswith(f"_{window}") for window in PERFORMANCE_WINDOWS)
        and col.rsplit("_", 1)[0] in PERFORMANCE_BASE_STATS
    ]
    if not feature_cols:
        raise ValueError("No performance feature columns found.")

    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns
    missing_numeric = df[numeric_features].isna().sum()
    bad_missing = missing_numeric[missing_numeric > 0]
    if not bad_missing.empty:
        raise ValueError(
            "Performance numeric features contain nulls: "
            f"{bad_missing.to_dict()}"
        )

    sum_cols = [
        col
        for col in feature_cols
        if col.rsplit("_", 1)[0] in PERFORMANCE_SUM_FEATURES
    ]
    bad_negative = [col for col in sum_cols if (df[col] < 0).any()]
    if bad_negative:
        raise ValueError(f"Negative count/sum performance features: {bad_negative}")

    per_90_cols = [col for col in feature_cols if "per_90" in col]
    if per_90_cols:
        values = df[per_90_cols].to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError("per_90 features contain NaN or infinite values.")

    for window in PERFORMANCE_WINDOWS:
        col = f"played_any_{window}"
        if col in df.columns:
            values = set(df[col].dropna().astype(int).unique())
            if not values.issubset({0, 1}):
                raise ValueError(f"{col} contains values outside 0/1: {values}")


def build_player_performance_summary(df: pd.DataFrame) -> dict:
    """Build descriptive metrics for Sprint 4 performance features."""
    feature_cols = [
        col
        for col in df.columns
        if any(col.endswith(f"_{window}") for window in PERFORMANCE_WINDOWS)
        and col.rsplit("_", 1)[0] in PERFORMANCE_BASE_STATS
    ]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns
    n_numeric_values = len(df) * len(numeric_features)
    missing_pct = (
        0.0
        if n_numeric_values == 0
        else float(df[numeric_features].isna().sum().sum() / n_numeric_values * 100)
    )

    summary: dict = {
        "rows": int(len(df)),
        "unique_players": int(df["player_id"].nunique()),
        "min_valuation_date": df["valuation_date"].min().date().isoformat(),
        "max_valuation_date": df["valuation_date"].max().date().isoformat(),
        "performance_feature_count": int(len(feature_cols)),
        "missing_numeric_feature_pct": round(missing_pct, 4),
    }

    for window in PERFORMANCE_WINDOWS:
        summary[f"played_any_rate_{window}"] = round(
            float(df[f"played_any_{window}"].mean()), 6
        )
        for stat in ("minutes", "goals", "assists"):
            col = f"{stat}_{window}"
            summary[f"median_{stat}_{window}"] = round(float(df[col].median()), 6)
            summary[f"mean_{stat}_{window}"] = round(float(df[col].mean()), 6)
            summary[f"max_{stat}_{window}"] = round(float(df[col].max()), 6)

    return summary


def save_player_performance_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Persist the performance feature dataset as parquet only."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Player performance dataset written to {output_path.resolve()}")


def save_player_performance_summary(summary: dict, output_path: Path) -> None:
    """Persist the performance summary dict as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Player performance summary written to {output_path.resolve()}")


# ----------------------------------------------------------------------
# Sprint 5 — Club & Competition context features
# ----------------------------------------------------------------------

CLUB_ID_CANDIDATES = ("club_id", "clubID", "clubId")

CLUBS_DROP_COLUMNS = (
    "total_market_value",
    "coach_name",
    "url",
    "filename",
    "image_url",
)

CLUBS_NUMERIC = (
    "squad_size",
    "average_age",
    "foreigners_number",
    "foreigners_percentage",
    "national_team_players",
    "stadium_seats",
    "last_season",
)

CLUBS_STRING = (
    "name",
    "club_code",
    "stadium_name",
    "net_transfer_record",
)

COMPETITIONS_STRING = (
    "name",
    "competition_code",
    "type",
    "sub_type",
    "country_name",
    "domestic_league_code",
    "confederation",
)

# Final dataset rename map for competitions columns
COMPETITION_RENAME = {
    "name": "competition_name",
    "type": "competition_type",
    "sub_type": "competition_sub_type",
    "country_name": "competition_country_name",
    "confederation": "competition_confederation",
    "competition_code": "competition_code",
    "country_id": "competition_country_id",
    "domestic_league_code": "competition_domestic_league_code",
}

FINAL_FORBIDDEN_COLUMNS = (
    "total_market_value",
    "highest_market_value_in_eur",
    "market_value_in_eur_y",
    "players_market_value_in_eur",
)


def standardize_clubs_columns(clubs: pd.DataFrame) -> pd.DataFrame:
    """Rename `club_id` variants and drop leakage/noise columns."""
    out = clubs.copy()
    if "club_id" not in out.columns:
        found = first_present(out, CLUB_ID_CANDIDATES)
        if found is None:
            raise ValueError(
                "clubs: missing required column 'club_id' "
                f"(looked for {list(CLUB_ID_CANDIDATES)})."
            )
        out = out.rename(columns={found: "club_id"})

    drop_now = [c for c in CLUBS_DROP_COLUMNS if c in out.columns]
    if drop_now:
        out = out.drop(columns=drop_now)
    return out


def prepare_clubs(clubs: pd.DataFrame) -> pd.DataFrame:
    """Standardize, dedupe, type-coerce and derive ratios."""
    out = standardize_clubs_columns(clubs)

    for col in CLUBS_NUMERIC:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in CLUBS_STRING:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
            out[col] = out[col].mask(out[col] == "", pd.NA)

    if {"foreigners_number", "squad_size"}.issubset(out.columns):
        squad = out["squad_size"]
        ratio = out["foreigners_number"] / squad.where(squad > 0)
        out["foreigners_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)
    if {"national_team_players", "squad_size"}.issubset(out.columns):
        squad = out["squad_size"]
        ratio = out["national_team_players"] / squad.where(squad > 0)
        out["national_team_players_ratio"] = ratio.replace(
            [np.inf, -np.inf], np.nan
        )

    out = out.dropna(subset=["club_id"])
    out["club_id"] = pd.to_numeric(out["club_id"], errors="coerce")
    out = out.dropna(subset=["club_id"])
    out["club_id"] = out["club_id"].astype("int64")
    out = out.drop_duplicates(subset=["club_id"], keep="first").reset_index(
        drop=True
    )
    return out


def standardize_competitions_columns(
    competitions: pd.DataFrame,
) -> pd.DataFrame:
    """Rename `competition_id` variants; require it."""
    out = competitions.copy()
    if "competition_id" not in out.columns:
        found = first_present(out, COMPETITION_ID_CANDIDATES)
        if found is None:
            raise ValueError(
                "competitions: missing required column 'competition_id' "
                f"(looked for {list(COMPETITION_ID_CANDIDATES)})."
            )
        out = out.rename(columns={found: "competition_id"})
    if "url" in out.columns:
        out = out.drop(columns=["url"])
    return out


def prepare_competitions(competitions: pd.DataFrame) -> pd.DataFrame:
    """Standardize, dedupe and rename for join-friendly column names."""
    out = standardize_competitions_columns(competitions)

    for col in COMPETITIONS_STRING:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
            out[col] = out[col].mask(out[col] == "", pd.NA)

    out = out.dropna(subset=["competition_id"])
    out["competition_id"] = out["competition_id"].astype("string").str.strip()
    out = out.drop_duplicates(subset=["competition_id"], keep="first")

    rename = {
        src_col: dst_col
        for src_col, dst_col in COMPETITION_RENAME.items()
        if src_col in out.columns
    }
    out = out.rename(columns=rename)

    keep_cols = ["competition_id"] + [
        c for c in COMPETITION_RENAME.values() if c in out.columns
    ]
    extra = [c for c in out.columns if c not in keep_cols]
    out = out[keep_cols + extra].reset_index(drop=True)
    return out


def add_club_competition_context(
    base_df: pd.DataFrame,
    clubs: pd.DataFrame,
    competitions: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join base <- clubs <- competitions; build final_competition_id."""
    n_in = len(base_df)
    out = base_df.copy()

    join_key = (
        "current_club_id"
        if "current_club_id" in out.columns
        else "club_id" if "club_id" in out.columns else None
    )

    if join_key is not None:
        clubs_renamed = clubs.copy()
        clubs_renamed["club_id"] = pd.to_numeric(
            clubs_renamed["club_id"], errors="coerce"
        ).astype("Int64")

        rename_map: dict[str, str] = {}
        for col in clubs_renamed.columns:
            if col == "club_id":
                continue
            target = f"club_{col}" if not col.startswith("club_") else col
            rename_map[col] = target
        clubs_renamed = clubs_renamed.rename(columns=rename_map)

        # Avoid clashes with columns already present in base
        clash = (set(clubs_renamed.columns) & set(out.columns)) - {"club_id"}
        if clash:
            clubs_renamed = clubs_renamed.drop(columns=sorted(clash))

        base_key = pd.to_numeric(out[join_key], errors="coerce").astype("Int64")
        out["_clubs_join_key"] = base_key
        clubs_renamed = clubs_renamed.rename(columns={"club_id": "_clubs_join_key"})

        out = out.merge(
            clubs_renamed, on="_clubs_join_key", how="left",
            validate="many_to_one",
        )
        out = out.drop(columns=["_clubs_join_key"])
        if len(out) != n_in:
            raise RuntimeError(
                f"Clubs join changed row count: {n_in} -> {len(out)}."
            )

    # Build final_competition_id with priority coalesce
    coalesce_priority = [
        "current_club_domestic_competition_id",
        "player_club_domestic_competition_id",
        "club_domestic_competition_id",
    ]
    available = [c for c in coalesce_priority if c in out.columns]
    if available:
        final = pd.Series(pd.NA, index=out.index, dtype="string")
        for col in available:
            cand = out[col].astype("string").str.strip()
            cand = cand.mask(cand == "", pd.NA)
            final = final.where(final.notna(), cand)
        out["final_competition_id"] = final

    if (
        "final_competition_id" in out.columns
        and "competition_id" in competitions.columns
    ):
        comp = competitions.copy()
        comp["competition_id"] = (
            comp["competition_id"].astype("string").str.strip()
        )

        clash = (set(comp.columns) & set(out.columns)) - {"competition_id"}
        if clash:
            comp = comp.drop(columns=sorted(clash))

        comp = comp.rename(
            columns={"competition_id": "final_competition_id"}
        )
        out = out.merge(
            comp, on="final_competition_id", how="left",
            validate="many_to_one",
        )
        if len(out) != n_in:
            raise RuntimeError(
                f"Competitions join changed row count: {n_in} -> {len(out)}."
            )

    return out


def validate_final_dataset(
    df: pd.DataFrame, expected_rows: int
) -> None:
    """Raise ValueError on row drift, missing target, dups, leakage, infs."""
    if len(df) != expected_rows:
        raise ValueError(
            f"Row count mismatch: expected {expected_rows}, got {len(df)}."
        )
    required = (
        "player_id",
        "valuation_date",
        "next_valuation_date",
        "next_market_value_in_eur",
        "log_next_market_value",
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    dup_keys = ["player_id", "valuation_date", "next_valuation_date"]
    n_dups = int(df.duplicated(subset=dup_keys).sum())
    if n_dups:
        raise ValueError(f"Found {n_dups} duplicate rows on {dup_keys}.")

    leaked = [c for c in FINAL_FORBIDDEN_COLUMNS if c in df.columns]
    if leaked:
        raise ValueError(f"Forbidden leakage columns present: {leaked}")

    ratio_cols = [
        c
        for c in ("club_foreigners_ratio", "club_national_team_players_ratio")
        if c in df.columns
    ]
    if ratio_cols:
        values = df[ratio_cols].to_numpy(dtype="float64")
        if np.isinf(values).any():
            raise ValueError(f"Inf values found in ratio columns: {ratio_cols}")


def _safe_pct(series: pd.Series) -> float:
    return round(float(series.isna().mean() * 100), 2)


def build_final_dataset_summary(df: pd.DataFrame) -> dict:
    """Aggregate descriptive stats for the final modelling dataset."""
    summary: dict = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "unique_players": int(df["player_id"].nunique()),
        "min_valuation_date": df["valuation_date"].min().date().isoformat(),
        "max_valuation_date": df["valuation_date"].max().date().isoformat(),
        "target_median": float(df["next_market_value_in_eur"].median()),
        "target_mean": float(df["next_market_value_in_eur"].mean()),
        "target_max": float(df["next_market_value_in_eur"].max()),
    }

    if "current_club_id" in df.columns:
        summary["unique_clubs"] = int(df["current_club_id"].nunique())
    if "final_competition_id" in df.columns:
        summary["unique_competitions"] = int(
            df["final_competition_id"].nunique()
        )
        summary["missing_final_competition_id_pct"] = _safe_pct(
            df["final_competition_id"]
        )
    if "club_name" in df.columns:
        summary["missing_club_name_pct"] = _safe_pct(df["club_name"])
    for col_name, key in (
        ("club_squad_size", "club_squad_size_missing_pct"),
        ("club_average_age", "club_average_age_missing_pct"),
        ("club_stadium_seats", "club_stadium_seats_missing_pct"),
        ("competition_name", "competition_name_missing_pct"),
    ):
        if col_name in df.columns:
            summary[key] = _safe_pct(df[col_name])

    if "competition_name" in df.columns:
        summary["top_20_competitions"] = _value_counts_dict(
            df["competition_name"], top=20
        )
    if "club_name" in df.columns:
        summary["top_20_clubs"] = _value_counts_dict(df["club_name"], top=20)

    numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
    summary["numeric_feature_count"] = int(numeric_cols)
    summary["categorical_feature_count"] = int(df.shape[1] - numeric_cols)

    return summary


def save_final_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Persist the final modelling dataset as parquet only."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Final dataset written to {output_path.resolve()}")


def save_final_dataset_summary(summary: dict, output_path: Path) -> None:
    """Persist the final summary dict as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Final summary written to {output_path.resolve()}")
