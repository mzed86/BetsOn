"""Consolidate Understat CSVs into Parquet files (matches, teams, players)."""

import logging
from pathlib import Path

import pandas as pd

from src.data.team_mapping import normalize_series

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/understat")
OUTPUT_DIR = Path("data/processed")

# Column renames for matches: dotted source names → clean names
MATCHES_RENAME = {
    "h.id": "home_id",
    "h.title": "home_team",
    "h.short_title": "home_short",
    "a.id": "away_id",
    "a.title": "away_team",
    "a.short_title": "away_short",
    "goals.h": "goals_h",
    "goals.a": "goals_a",
    "xG.h": "xg_h",
    "xG.a": "xg_a",
    "forecast.w": "forecast_w",
    "forecast.d": "forecast_d",
    "forecast.l": "forecast_l",
}


def _extract_season_year(filename: str) -> int:
    """Extract season start year from filename like '2023_matches.csv'."""
    return int(filename.split("_")[0])


def process_matches(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Process all Understat match CSVs into a single DataFrame."""
    frames = []

    for league_dir in sorted(raw_dir.iterdir()):
        if not league_dir.is_dir():
            continue
        league = league_dir.name

        for csv_path in sorted(league_dir.glob("*_matches.csv")):
            season = _extract_season_year(csv_path.name)
            logger.info("Processing understat matches: %s/%d", league, season)

            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            # Parse datetime and extract date-only column
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["Date"] = df["datetime"].dt.normalize()

            # Rename dotted columns
            df = df.rename(columns=MATCHES_RENAME)

            # Normalize team names
            df["home_team"] = normalize_series(df["home_team"], source="understat")
            df["away_team"] = normalize_series(df["away_team"], source="understat")

            # Cast numeric columns
            for col in ("xg_h", "xg_a", "forecast_w", "forecast_d", "forecast_l"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            for col in ("goals_h", "goals_a"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(pd.Int64Dtype())

            df["league"] = league
            df["season"] = season
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No understat match CSVs found in {raw_dir}")

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["Date", "league"]).reset_index(drop=True)
    return result


def process_teams(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Process all Understat team CSVs into a single DataFrame."""
    frames = []

    for league_dir in sorted(raw_dir.iterdir()):
        if not league_dir.is_dir():
            continue
        league = league_dir.name

        for csv_path in sorted(league_dir.glob("*_teams.csv")):
            season = _extract_season_year(csv_path.name)
            logger.info("Processing understat teams: %s/%d", league, season)

            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            # Normalize team names
            df["title"] = normalize_series(df["title"], source="understat")

            # Cast numeric columns
            numeric_cols = [
                "total_xG", "total_xGA", "total_scored", "total_missed",
                "total_pts", "total_wins", "total_draws", "total_loses",
                "matches_played",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["league"] = league
            df["season"] = season
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No understat team CSVs found in {raw_dir}")

    result = pd.concat(frames, ignore_index=True)
    return result


def process_players(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Process all Understat player CSVs into a single DataFrame."""
    frames = []

    for league_dir in sorted(raw_dir.iterdir()):
        if not league_dir.is_dir():
            continue
        league = league_dir.name

        for csv_path in sorted(league_dir.glob("*_players.csv")):
            season = _extract_season_year(csv_path.name)
            logger.info("Processing understat players: %s/%d", league, season)

            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            # Normalize team names
            if "team_title" in df.columns:
                df["team_title"] = normalize_series(df["team_title"], source="understat")

            # Cast numeric columns
            numeric_cols = [
                "games", "time", "goals", "xG", "assists", "xA",
                "shots", "key_passes", "yellow_cards", "red_cards",
                "npg", "npxG", "xGChain", "xGBuildup",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["league"] = league
            df["season"] = season
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No understat player CSVs found in {raw_dir}")

    result = pd.concat(frames, ignore_index=True)
    return result


def process(raw_dir: Path = RAW_DIR, output_dir: Path = OUTPUT_DIR) -> None:
    """Process all Understat data and write three Parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    matches = process_matches(raw_dir)
    matches_path = output_dir / "understat_matches.parquet"
    matches.to_parquet(matches_path, index=False)
    logger.info("Wrote %s: %d rows", matches_path, len(matches))

    teams = process_teams(raw_dir)
    teams_path = output_dir / "understat_teams.parquet"
    teams.to_parquet(teams_path, index=False)
    logger.info("Wrote %s: %d rows", teams_path, len(teams))

    players = process_players(raw_dir)
    players_path = output_dir / "understat_players.parquet"
    players.to_parquet(players_path, index=False)
    logger.info("Wrote %s: %d rows", players_path, len(players))
