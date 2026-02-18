"""Elo rating system for football match outcome prediction.

Iterates chronologically through matches, recording pre-match Elo ratings
as features (no leakage) and updating after each result.
"""

import logging

import numpy as np
import pandas as pd

from src.features.config import EloConfig

logger = logging.getLogger(__name__)


def compute_elo_features(matches: pd.DataFrame, config: EloConfig) -> pd.DataFrame:
    """Compute Elo ratings for all matches chronologically.

    Pre-match ratings are recorded as features before updating, preventing
    any leakage of the current match result.

    Args:
        matches: Match-level DataFrame sorted by Date, with columns
                 home_team, away_team, FTHG, FTAG.
        config: Elo configuration (k_factor, initial_rating, home_advantage).

    Returns:
        DataFrame with same index as matches, containing columns:
        h_elo, a_elo, elo_diff, h_elo_expected.
    """
    n = len(matches)
    h_elo = np.empty(n, dtype=np.float64)
    a_elo = np.empty(n, dtype=np.float64)
    h_expected = np.empty(n, dtype=np.float64)

    ratings: dict[str, float] = {}
    k = config.k_factor
    init = config.initial_rating
    home_adv = config.home_advantage

    home_teams = matches["home_team"].values
    away_teams = matches["away_team"].values
    home_goals = matches["FTHG"].values
    away_goals = matches["FTAG"].values

    for i in range(n):
        ht = home_teams[i]
        at = away_teams[i]

        # Pre-match ratings (feature values — no leakage)
        r_home = ratings.get(ht, init)
        r_away = ratings.get(at, init)
        h_elo[i] = r_home
        a_elo[i] = r_away

        # Expected score with home advantage
        exp_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home - home_adv) / 400.0))
        h_expected[i] = exp_home

        # Actual score (1 = win, 0.5 = draw, 0 = loss)
        hg = home_goals[i]
        ag = away_goals[i]
        if pd.isna(hg) or pd.isna(ag):
            continue  # Skip null results (don't update ratings)

        if hg > ag:
            s_home = 1.0
        elif hg == ag:
            s_home = 0.5
        else:
            s_home = 0.0

        # Update ratings
        ratings[ht] = r_home + k * (s_home - exp_home)
        ratings[at] = r_away + k * ((1.0 - s_home) - (1.0 - exp_home))

    result = pd.DataFrame(
        {
            "h_elo": h_elo,
            "a_elo": a_elo,
            "elo_diff": h_elo - a_elo,
            "h_elo_expected": h_expected,
        },
        index=matches.index,
    )

    logger.info(
        "Elo features: %d teams rated, final rating range [%.0f, %.0f]",
        len(ratings),
        min(ratings.values()),
        max(ratings.values()),
    )
    return result
