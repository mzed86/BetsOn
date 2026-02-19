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


def _implied_probs_2way(
    matches: pd.DataFrame,
    cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Compute normalized implied probabilities from an Over/Under odds pair.

    Args:
        matches: Match-level DataFrame.
        cols: Two column names [over_odds, under_odds].
        prefix: Output column prefix (e.g., "b365_ou25").

    Returns:
        DataFrame with columns {prefix}_over, {prefix}_under,
        and {prefix}_overround.
    """
    over_col, under_col = cols
    result = pd.DataFrame(index=matches.index)

    over_odds = matches[over_col].where(matches[over_col] > 0)
    under_odds = matches[under_col].where(matches[under_col] > 0)

    raw_over = 1.0 / over_odds
    raw_under = 1.0 / under_odds
    total = raw_over + raw_under

    result[f"{prefix}_over"] = raw_over / total
    result[f"{prefix}_under"] = raw_under / total
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

    # Additional bookmaker implied probabilities
    additional_books = getattr(config, "additional_books", {})
    book_prob_dfs = {}  # book_name -> DataFrame with {book}_prob_h/d/a
    for book_name, cols in additional_books.items():
        if all(c in matches.columns for c in cols):
            book_probs = _implied_probs(matches, cols, f"{book_name}_prob")
            parts.append(book_probs)
            result = pd.concat([result, book_probs], axis=1)
            book_prob_dfs[book_name] = book_probs

    # Cross-book aggregation: max odds, min prob, prob spread per outcome
    if book_prob_dfs:
        # Collect all book prob columns (including B365 and Pinnacle opening)
        all_book_probs = {}  # suffix -> list of (book_name, Series)
        for suffix in ["h", "d", "a"]:
            all_book_probs[suffix] = []
            if f"b365_prob_{suffix}" in result.columns:
                all_book_probs[suffix].append(("b365", result[f"b365_prob_{suffix}"]))
            for book_name, bdf in book_prob_dfs.items():
                col = f"{book_name}_prob_{suffix}"
                if col in result.columns:
                    all_book_probs[suffix].append((book_name, result[col]))

        # Build odds from all available books for max_odds computation
        all_book_odds = {}  # suffix -> list of Series
        odds_suffix_map = {"h": 0, "d": 1, "a": 2}
        b365_cols = config.primary
        for suffix, idx in odds_suffix_map.items():
            all_book_odds[suffix] = []
            if b365_cols[idx] in matches.columns:
                all_book_odds[suffix].append(matches[b365_cols[idx]])
            for book_name, cols in additional_books.items():
                if cols[idx] in matches.columns:
                    all_book_odds[suffix].append(matches[cols[idx]])

        for suffix in ["h", "d", "a"]:
            if len(all_book_probs[suffix]) >= 2:
                prob_stack = pd.concat(
                    [s for _, s in all_book_probs[suffix]], axis=1,
                )
                result[f"min_book_prob_{suffix}"] = prob_stack.min(axis=1)
                result[f"book_prob_spread_{suffix}"] = (
                    prob_stack.max(axis=1) - prob_stack.min(axis=1)
                )
            if len(all_book_odds[suffix]) >= 2:
                odds_stack = pd.concat(
                    [s.rename(f"o{i}") for i, s in enumerate(all_book_odds[suffix])],
                    axis=1,
                )
                result[f"max_odds_{suffix}"] = odds_stack.max(axis=1)

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

    # Over/Under 2.5 implied probabilities
    ou25_books = getattr(config, "ou25_books", {})
    for book_name, cols in ou25_books.items():
        if len(cols) == 2 and all(c in matches.columns for c in cols):
            ou_probs = _implied_probs_2way(matches, cols, book_name)
            result = pd.concat([result, ou_probs], axis=1)

    # Corner match result implied probabilities (3-way, reuse _implied_probs)
    corner_books = getattr(config, "corner_books", {})
    for book_name, cols in corner_books.items():
        if len(cols) == 3 and all(c in matches.columns for c in cols):
            corner_probs = _implied_probs(matches, cols, book_name)
            result = pd.concat([result, corner_probs], axis=1)

    logger.info("Odds features: %d columns", len(result.columns))
    return result
