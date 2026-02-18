"""Match context features: rest days, match number, early season flag."""

import logging

import pandas as pd

from src.features.config import MatchContextConfig

logger = logging.getLogger(__name__)


def compute_match_context_features(
    matches: pd.DataFrame,
    config: MatchContextConfig,
) -> pd.DataFrame:
    """Compute contextual features for each match.

    Args:
        matches: Match-level DataFrame with Date, home_team, away_team,
                 league, season columns.
        config: Match context configuration.

    Returns:
        DataFrame with same index as matches, containing rest days,
        match number, and early season flag columns.
    """
    # Build team-perspective view (2 rows per match)
    home = pd.DataFrame({
        "match_idx": matches.index,
        "Date": matches["Date"],
        "team": matches["home_team"],
        "venue": "home",
        "league": matches["league"],
        "season": matches["season"],
    })
    away = pd.DataFrame({
        "match_idx": matches.index,
        "Date": matches["Date"],
        "team": matches["away_team"],
        "venue": "away",
        "league": matches["league"],
        "season": matches["season"],
    })
    team_df = pd.concat([home, away], ignore_index=True)
    team_df = team_df.sort_values(["team", "Date"]).reset_index(drop=True)

    # Rest days: days since team's previous match (any venue)
    team_df["prev_date"] = team_df.groupby("team")["Date"].shift(1)
    team_df["days_rest"] = (team_df["Date"] - team_df["prev_date"]).dt.days
    team_df["days_rest"] = team_df["days_rest"].clip(upper=config.summer_break_threshold)

    # Match number within season (per team)
    team_df["match_num"] = team_df.groupby(["team", "season"]).cumcount() + 1
    team_df["is_early_season"] = (team_df["match_num"] <= config.early_season_threshold).astype(int)

    # Split back to home/away and join to match index
    home_rows = team_df[team_df["venue"] == "home"].set_index("match_idx")
    away_rows = team_df[team_df["venue"] == "away"].set_index("match_idx")

    result = pd.DataFrame(index=matches.index)
    result["h_days_rest"] = home_rows["days_rest"]
    result["a_days_rest"] = away_rows["days_rest"]
    result["rest_diff"] = result["h_days_rest"] - result["a_days_rest"]
    result["h_match_num"] = home_rows["match_num"]
    result["a_match_num"] = away_rows["match_num"]
    result["h_is_early_season"] = home_rows["is_early_season"]
    result["a_is_early_season"] = away_rows["is_early_season"]

    logger.info("Match context features: %d columns", len(result.columns))
    return result
