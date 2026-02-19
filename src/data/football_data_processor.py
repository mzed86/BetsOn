"""Consolidate football-data.co.uk CSVs into a single Parquet file."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.team_mapping import normalize_series

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/football_data")
OUTPUT_PATH = Path("data/processed/football_data_matches.parquet")

# Core stat columns that should be integers
INT_COLUMNS = [
    "FTHG", "FTAG", "HTHG", "HTAG",
    "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR",
]


def _season_code_to_start_year(code: str) -> int:
    """Convert season code like '2324' to start year 2023."""
    start = int(code[:2])
    return start + 2000 if start < 90 else start + 1900


def _process_single_csv(path: Path, league: str, season_code: str) -> pd.DataFrame | None:
    """Read and process a single football-data CSV."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except (UnicodeDecodeError, Exception):
            continue
    else:
        logger.warning("Failed to read %s with any encoding", path)
        return None

    # Drop fully empty trailing rows and columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    if df.empty or "HomeTeam" not in df.columns:
        logger.warning("Skipping %s: missing HomeTeam column or empty", path)
        return None

    # Parse dates — dayfirst=True handles both DD/MM/YY and DD/MM/YYYY
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    # Add metadata
    df["league"] = league
    df["season"] = _season_code_to_start_year(season_code)

    # Normalize team names
    df["HomeTeam"] = normalize_series(df["HomeTeam"], source="football_data")
    df["AwayTeam"] = normalize_series(df["AwayTeam"], source="football_data")

    # Rename to consistent snake_case for key columns
    df = df.rename(columns={"HomeTeam": "home_team", "AwayTeam": "away_team"})

    # Cast stat columns to nullable integer
    for col in INT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(pd.Int64Dtype())

    # Drop the Div column (redundant with league)
    df = df.drop(columns=["Div"], errors="ignore")

    return df


def process(raw_dir: Path = RAW_DIR, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Process all football-data CSVs and write consolidated Parquet.

    Args:
        raw_dir: Directory containing league subdirectories with season CSVs.
        output_path: Path for the output Parquet file.

    Returns:
        The consolidated DataFrame.
    """
    frames = []

    for league_dir in sorted(raw_dir.iterdir()):
        if not league_dir.is_dir():
            continue
        league = league_dir.name

        for csv_path in sorted(league_dir.glob("*.csv")):
            season_code = csv_path.stem  # e.g., "2324"
            logger.info("Processing %s/%s", league, season_code)

            df = _process_single_csv(csv_path, league, season_code)
            if df is not None:
                frames.append(df)

    if not frames:
        raise RuntimeError(f"No valid CSVs found in {raw_dir}")

    # Outer join preserves columns that only exist in some eras
    result = pd.concat(frames, join="outer", ignore_index=True)

    # Sort by date for clean output
    result = result.sort_values(["Date", "league"]).reset_index(drop=True)

    # Coerce mixed-type object columns to numeric where possible.
    # After concat, columns like BbAH/BbOU end up as object due to
    # str vs numeric inconsistencies across CSVs from different eras.
    non_text_skip = {"home_team", "away_team", "league", "FTR", "HTR", "Referee", "Time"}
    for col in result.columns:
        if result[col].dtype == object and col not in non_text_skip:
            converted = pd.to_numeric(result[col], errors="coerce")
            # Only convert if at least some values parsed successfully
            if converted.notna().any():
                result[col] = converted

    # --- Derived target columns ---
    # Use .to_numpy(dtype=float, na_value=np.nan) to avoid nullable Int64 issues
    # with np.where when NA values are present.

    # Over/Under 2.5 goals result
    if "FTHG" in result.columns and "FTAG" in result.columns:
        fthg = result["FTHG"].to_numpy(dtype=float, na_value=np.nan)
        ftag = result["FTAG"].to_numpy(dtype=float, na_value=np.nan)
        total_goals = fthg + ftag
        missing = np.isnan(total_goals)
        result["ou25_result"] = np.where(missing, pd.NA, np.where(total_goals > 2.5, "Over", "Under"))

    # Both Teams To Score result
    if "FTHG" in result.columns and "FTAG" in result.columns:
        fthg = result["FTHG"].to_numpy(dtype=float, na_value=np.nan)
        ftag = result["FTAG"].to_numpy(dtype=float, na_value=np.nan)
        missing = np.isnan(fthg) | np.isnan(ftag)
        btts = (fthg > 0) & (ftag > 0)
        result["btts_result"] = np.where(missing, pd.NA, np.where(btts, "Yes", "No"))

    # Corner match result (H/D/A based on corner counts)
    if "HC" in result.columns and "AC" in result.columns:
        hc = result["HC"].to_numpy(dtype=float, na_value=np.nan)
        ac = result["AC"].to_numpy(dtype=float, na_value=np.nan)
        missing = np.isnan(hc) | np.isnan(ac)
        result["corner_ftr"] = np.where(
            missing, pd.NA,
            np.where(hc > ac, "H", np.where(hc == ac, "D", "A")),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    logger.info(
        "Wrote %s: %d rows, %d columns",
        output_path, len(result), len(result.columns),
    )

    return result
