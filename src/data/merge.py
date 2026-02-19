"""Merge football-data and Understat match data."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

# Leagues that exist in both data sources
OVERLAPPING_LEAGUES = {"EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"}

# Columns to bring in from Understat
UNDERSTAT_MERGE_COLS = ["xg_h", "xg_a", "forecast_w", "forecast_d", "forecast_l"]


def validate_merge(combined: pd.DataFrame) -> None:
    """Log merge rates per league and warn if below threshold."""
    for league in sorted(combined["league"].unique()):
        league_df = combined[combined["league"] == league]
        total = len(league_df)

        if league in OVERLAPPING_LEAGUES:
            matched = league_df["xg_h"].notna().sum()
            rate = matched / total if total > 0 else 0
            level = logging.WARNING if rate < 0.95 else logging.INFO
            logger.log(
                level,
                "Merge rate for %s: %d/%d (%.1f%%)",
                league, matched, total, rate * 100,
            )
        else:
            logger.info(
                "League %s: %d rows (no Understat data expected)", league, total,
            )


def merge(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Left join football-data with Understat xG data.

    Args:
        processed_dir: Directory containing the processed Parquet files.

    Returns:
        Combined DataFrame with football-data as the base and Understat xG
        columns added for overlapping leagues.
    """
    fd_path = processed_dir / "football_data_matches.parquet"
    us_path = processed_dir / "understat_matches.parquet"

    fd = pd.read_parquet(fd_path)
    us = pd.read_parquet(us_path)

    # Prepare Understat side: keep only needed columns for the join
    us_for_merge = us[["Date", "league", "home_team", "away_team"] + UNDERSTAT_MERGE_COLS].copy()

    # Left join on match identity
    combined = fd.merge(
        us_for_merge,
        on=["Date", "league", "home_team", "away_team"],
        how="left",
        suffixes=("", "_us"),
    )

    validate_merge(combined)

    output_path = processed_dir / "matches_combined.parquet"
    combined.to_parquet(output_path, index=False)
    logger.info("Wrote %s: %d rows, %d columns", output_path, len(combined), len(combined.columns))

    return combined
