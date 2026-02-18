"""Rolling team statistics computed from a team-perspective view of match data.

Algorithm: melt matches to team-perspective rows, shift(1) to prevent leakage,
compute rolling means, then join back to match-level DataFrame.
"""

import logging

import pandas as pd

from src.features.config import RollingConfig

logger = logging.getLogger(__name__)

# Stats computed from the team-perspective melt
_CORE_STATS = [
    "ppg",
    "goals_for",
    "goals_against",
    "goal_diff",
    "win_rate",
    "draw_rate",
    "loss_rate",
    "clean_sheet_rate",
    "shots_on_target",
    "shots_conceded",
]

_XG_STATS = [
    "xg_for",
    "xg_against",
    "xg_overperf",
]


def _melt_to_team_perspective(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert match-level data to team-perspective rows (2 rows per match).

    Each match produces one row for the home team and one for the away team,
    recording stats from that team's perspective.

    Args:
        matches: Match-level DataFrame with standard columns.

    Returns:
        DataFrame with columns: match_idx, Date, team, opponent, venue, league,
        season, plus per-team stats.
    """
    home = pd.DataFrame({
        "match_idx": matches.index,
        "Date": matches["Date"],
        "team": matches["home_team"],
        "opponent": matches["away_team"],
        "venue": "home",
        "league": matches["league"],
        "season": matches["season"],
        "goals_for": matches["FTHG"],
        "goals_against": matches["FTAG"],
        "shots_on_target": matches["HST"],
        "shots_conceded": matches["AST"],
    })

    away = pd.DataFrame({
        "match_idx": matches.index,
        "Date": matches["Date"],
        "team": matches["away_team"],
        "opponent": matches["home_team"],
        "venue": "away",
        "league": matches["league"],
        "season": matches["season"],
        "goals_for": matches["FTAG"],
        "goals_against": matches["FTHG"],
        "shots_on_target": matches["AST"],
        "shots_conceded": matches["HST"],
    })

    # Add xG if available
    if "xg_h" in matches.columns:
        home["xg_for"] = matches["xg_h"]
        home["xg_against"] = matches["xg_a"]
        away["xg_for"] = matches["xg_a"]
        away["xg_against"] = matches["xg_h"]

    team_df = pd.concat([home, away], ignore_index=True)

    # Derived stats
    team_df["goal_diff"] = team_df["goals_for"] - team_df["goals_against"]
    team_df["win_rate"] = (team_df["goals_for"] > team_df["goals_against"]).astype(float)
    team_df["draw_rate"] = (team_df["goals_for"] == team_df["goals_against"]).astype(float)
    team_df["loss_rate"] = (team_df["goals_for"] < team_df["goals_against"]).astype(float)
    team_df["clean_sheet_rate"] = (team_df["goals_against"] == 0).astype(float)

    # Points: 3 for win, 1 for draw, 0 for loss
    team_df["ppg"] = team_df["win_rate"] * 3 + team_df["draw_rate"] * 1

    # xG overperformance (goals_for - xg_for)
    if "xg_for" in team_df.columns:
        team_df["xg_overperf"] = team_df["goals_for"] - team_df["xg_for"]

    # Sort by team then date for correct rolling order
    team_df = team_df.sort_values(["team", "Date"]).reset_index(drop=True)

    return team_df


def _compute_rolling_stats(
    team_df: pd.DataFrame,
    stat_columns: list[str],
    windows: list[int],
    min_periods: int,
) -> pd.DataFrame:
    """Compute shifted rolling means for each team across all matches.

    Uses shift(1) before rolling to prevent leakage — the rolling window
    for match N only includes data from matches before N.

    Args:
        team_df: Team-perspective DataFrame sorted by (team, Date).
        stat_columns: Columns to compute rolling stats for.
        windows: List of rolling window sizes.
        min_periods: Minimum observations for a valid rolling value.

    Returns:
        DataFrame indexed like team_df with rolling columns named
        rolling{window}_{stat}.
    """
    result = pd.DataFrame(index=team_df.index)
    grouped = team_df.groupby("team")

    for stat in stat_columns:
        if stat not in team_df.columns:
            continue
        shifted = grouped[stat].shift(1)
        for w in windows:
            col_name = f"rolling{w}_{stat}"
            result[col_name] = shifted.groupby(team_df["team"]).rolling(
                w, min_periods=min_periods
            ).mean().droplevel(0)

    return result


def _compute_venue_rolling_stats(
    team_df: pd.DataFrame,
    stat_columns: list[str],
    windows: list[int],
    min_periods: int,
) -> pd.DataFrame:
    """Compute rolling stats using only home or away matches for each team.

    Args:
        team_df: Team-perspective DataFrame sorted by (team, Date).
        stat_columns: Columns to compute rolling stats for.
        windows: List of rolling window sizes.
        min_periods: Minimum observations for a valid rolling value.

    Returns:
        DataFrame indexed like team_df with rolling columns named
        venue_rolling{window}_{stat}. NaN for rows where the venue
        doesn't match (filled later during join).
    """
    result = pd.DataFrame(index=team_df.index)

    for venue in ["home", "away"]:
        mask = team_df["venue"] == venue
        venue_df = team_df[mask]
        grouped = venue_df.groupby("team")

        for stat in stat_columns:
            if stat not in venue_df.columns:
                continue
            shifted = grouped[stat].shift(1)
            for w in windows:
                col_name = f"venue_rolling{w}_{stat}"
                rolled = shifted.groupby(venue_df["team"]).rolling(
                    w, min_periods=min_periods
                ).mean().droplevel(0)
                # Write back to full-index result (NaN for other venue)
                if col_name not in result.columns:
                    result[col_name] = pd.NA
                result.loc[rolled.index, col_name] = rolled

    return result


def compute_rolling_features(matches: pd.DataFrame, config: RollingConfig) -> pd.DataFrame:
    """Compute all rolling features and join back to match-level DataFrame.

    Args:
        matches: Match-level DataFrame (must have a unique index).
        config: Rolling configuration with windows and min_periods.

    Returns:
        DataFrame with same index as matches, containing rolling feature
        columns prefixed with h_ (home team) and a_ (away team).
    """
    logger.info("Melting matches to team perspective...")
    team_df = _melt_to_team_perspective(matches)

    # Determine which stats to compute
    stats = [s for s in _CORE_STATS if s in team_df.columns]
    xg_stats = [s for s in _XG_STATS if s in team_df.columns]
    all_stats = stats + xg_stats

    logger.info(
        "Computing overall rolling stats (windows=%s, stats=%d)...",
        config.windows, len(all_stats),
    )
    overall_rolling = _compute_rolling_stats(
        team_df, all_stats, config.windows, config.min_periods
    )

    logger.info(
        "Computing venue-specific rolling stats (windows=%s)...",
        config.venue_windows,
    )
    venue_rolling = _compute_venue_rolling_stats(
        team_df, stats, config.venue_windows, config.min_periods
    )

    # Combine rolling results with team_df for joining
    team_df = pd.concat([team_df, overall_rolling, venue_rolling], axis=1)

    # Split back into home and away, join to match index
    rolling_cols = [c for c in team_df.columns if "rolling" in c]
    home_rows = team_df[team_df["venue"] == "home"].set_index("match_idx")
    away_rows = team_df[team_df["venue"] == "away"].set_index("match_idx")

    result = pd.DataFrame(index=matches.index)
    for col in rolling_cols:
        result[f"h_{col}"] = home_rows[col]
        result[f"a_{col}"] = away_rows[col]

    logger.info("Rolling features: %d columns", len(result.columns))
    return result
