"""Odds-derived features: implied probabilities, overround, favourite."""

import logging

import pandas as pd

from src.features.config import OddsConfig

logger = logging.getLogger(__name__)


def _implied_probs(
    matches: pd.DataFrame,
    cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Compute normalized implied probabilities from a H/D/A odds triplet.

    Args:
        matches: Match-level DataFrame.
        cols: Three column names [home_odds, draw_odds, away_odds].
        prefix: Output column prefix (e.g., "b365_prob").

    Returns:
        DataFrame with columns {prefix}_h, {prefix}_d, {prefix}_a,
        and {prefix[:-5]}_overround (only for primary).
    """
    h_col, d_col, a_col = cols
    result = pd.DataFrame(index=matches.index)

    # Safely handle odds <= 0 or NaN
    h_odds = matches[h_col].where(matches[h_col] > 0)
    d_odds = matches[d_col].where(matches[d_col] > 0)
    a_odds = matches[a_col].where(matches[a_col] > 0)

    raw_h = 1.0 / h_odds
    raw_d = 1.0 / d_odds
    raw_a = 1.0 / a_odds
    total = raw_h + raw_d + raw_a

    result[f"{prefix}_h"] = raw_h / total
    result[f"{prefix}_d"] = raw_d / total
    result[f"{prefix}_a"] = raw_a / total
    result[f"{prefix}_overround"] = total

    return result


def compute_odds_features(matches: pd.DataFrame, config: OddsConfig) -> pd.DataFrame:
    """Compute all odds-derived features.

    Args:
        matches: Match-level DataFrame with betting odds columns.
        config: Odds configuration with column triplets.

    Returns:
        DataFrame with implied probability columns, overround, and favourite.
    """
    parts = []

    # B365 implied probabilities
    if all(c in matches.columns for c in config.primary):
        parts.append(_implied_probs(matches, config.primary, "b365_prob"))

    # Pinnacle opening
    if all(c in matches.columns for c in config.pinnacle_opening):
        parts.append(_implied_probs(matches, config.pinnacle_opening, "ps_prob"))

    # Pinnacle closing
    if all(c in matches.columns for c in config.pinnacle_closing):
        parts.append(_implied_probs(matches, config.pinnacle_closing, "psc_prob"))

    result = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=matches.index)

    # Favourite based on B365 probabilities
    if "b365_prob_h" in result.columns:
        prob_cols = ["b365_prob_h", "b365_prob_d", "b365_prob_a"]
        labels = ["H", "D", "A"]
        probs = result[prob_cols]
        valid = probs.notna().all(axis=1)
        result["favourite"] = pd.NA
        result["favourite_prob"] = pd.NA
        if valid.any():
            col_to_label = dict(zip(prob_cols, labels))
            result.loc[valid, "favourite"] = probs.loc[valid].idxmax(axis=1).map(col_to_label)
            result.loc[valid, "favourite_prob"] = probs.loc[valid].max(axis=1)

    logger.info("Odds features: %d columns", len(result.columns))
    return result
