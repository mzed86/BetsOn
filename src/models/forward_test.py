"""Forward test infrastructure: generate Kelly-sized bets on upcoming fixtures.

Takes a CSV of upcoming fixtures with current odds, applies saved disagreement
selectors with league filters, and outputs recommended bets with Kelly sizing.

Only disagreement-based strategies work forward (they use pre-match opening odds).
CLV classifier and Meta model require closing odds and are not forward-compatible.

CLI: python -m src.models.forward_test --fixtures path/to/fixtures.csv --bankroll 1000
     python -m src.models.forward_test --config configs/forward_test_config.yaml --fixtures path/to/fixtures.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.value_metrics import (
    adjust_odds_for_costs,
    compute_kelly_fraction,
)
from src.models.train import load_config
from src.models.value_betting import DisagreementSelector, GenericDisagreementSelector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "forward_test_config.yaml"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

# Column mappings
B365_1X2_COLS = {"H": "B365H", "D": "B365D", "A": "B365A"}
PS_1X2_COLS = {"H": "PSH", "D": "PSD", "A": "PSA"}
B365_OU_COLS = {"Over": "B365>2.5", "Under": "B365<2.5"}
PS_OU_COLS = {"Over": "P>2.5", "Under": "P<2.5"}
B365_CORNER_COLS = {"H": "B365CH", "D": "B365CD", "A": "B365CA"}
AVG_CORNER_COLS = {"H": "AvgCH", "D": "AvgCD", "A": "AvgCA"}


def _implied_probs_3way(
    df: pd.DataFrame,
    cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Compute normalized 3-way implied probabilities.

    Args:
        df: DataFrame with odds columns.
        cols: [home_odds, draw_odds, away_odds] column names.
        prefix: Output column prefix.

    Returns:
        DataFrame with {prefix}_h, {prefix}_d, {prefix}_a columns.
    """
    h_col, d_col, a_col = cols
    result = pd.DataFrame(index=df.index)

    h_odds = df[h_col].where(df[h_col] > 0)
    d_odds = df[d_col].where(df[d_col] > 0)
    a_odds = df[a_col].where(df[a_col] > 0)

    raw_h = 1.0 / h_odds
    raw_d = 1.0 / d_odds
    raw_a = 1.0 / a_odds
    total = raw_h + raw_d + raw_a

    result[f"{prefix}_h"] = raw_h / total
    result[f"{prefix}_d"] = raw_d / total
    result[f"{prefix}_a"] = raw_a / total
    return result


def _implied_probs_2way(
    df: pd.DataFrame,
    cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Compute normalized 2-way implied probabilities.

    Args:
        df: DataFrame with odds columns.
        cols: [over_odds, under_odds] column names.
        prefix: Output column prefix.

    Returns:
        DataFrame with {prefix}_over, {prefix}_under columns.
    """
    over_col, under_col = cols
    result = pd.DataFrame(index=df.index)

    over_odds = df[over_col].where(df[over_col] > 0)
    under_odds = df[under_col].where(df[under_col] > 0)

    raw_over = 1.0 / over_odds
    raw_under = 1.0 / under_odds
    total = raw_over + raw_under

    result[f"{prefix}_over"] = raw_over / total
    result[f"{prefix}_under"] = raw_under / total
    return result


def compute_forward_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute implied probs and disagreement from raw odds columns.

    Args:
        df: DataFrame with raw odds columns from the fixtures CSV.

    Returns:
        DataFrame with added implied probability and disagreement columns.
    """
    result = df.copy()

    # --- 1X2 ---
    has_b365_1x2 = all(c in df.columns for c in B365_1X2_COLS.values())
    has_ps_1x2 = all(c in df.columns for c in PS_1X2_COLS.values())

    if has_b365_1x2:
        b365_probs = _implied_probs_3way(
            df, list(B365_1X2_COLS.values()), "b365_prob",
        )
        for col in b365_probs.columns:
            result[col] = b365_probs[col]

    if has_ps_1x2:
        ps_probs = _implied_probs_3way(
            df, list(PS_1X2_COLS.values()), "ps_prob",
        )
        for col in ps_probs.columns:
            result[col] = ps_probs[col]

    if has_b365_1x2 and has_ps_1x2:
        for suffix in ["h", "d", "a"]:
            result[f"odds_disagree_{suffix}"] = result[f"b365_prob_{suffix}"] - result[f"ps_prob_{suffix}"]

    # --- O/U 2.5 ---
    has_b365_ou = all(c in df.columns for c in B365_OU_COLS.values())
    has_ps_ou = all(c in df.columns for c in PS_OU_COLS.values())

    if has_b365_ou:
        b365_ou = _implied_probs_2way(
            df, list(B365_OU_COLS.values()), "b365_ou25",
        )
        for col in b365_ou.columns:
            result[col] = b365_ou[col]

    if has_ps_ou:
        ps_ou = _implied_probs_2way(
            df, list(PS_OU_COLS.values()), "ps_ou25",
        )
        for col in ps_ou.columns:
            result[col] = ps_ou[col]

    if has_b365_ou and has_ps_ou:
        for suffix in ["over", "under"]:
            result[f"ou25_disagree_{suffix}"] = result[f"b365_ou25_{suffix}"] - result[f"ps_ou25_{suffix}"]

    # --- Corners ---
    has_b365_corner = all(c in df.columns for c in B365_CORNER_COLS.values())
    has_avg_corner = all(c in df.columns for c in AVG_CORNER_COLS.values())

    if has_b365_corner:
        b365_corner = _implied_probs_3way(
            df, list(B365_CORNER_COLS.values()), "b365_corner",
        )
        for col in b365_corner.columns:
            result[col] = b365_corner[col]

    if has_avg_corner:
        avg_corner = _implied_probs_3way(
            df, list(AVG_CORNER_COLS.values()), "avg_corner",
        )
        for col in avg_corner.columns:
            result[col] = avg_corner[col]

    if has_b365_corner and has_avg_corner:
        for suffix in ["h", "d", "a"]:
            result[f"corner_disagree_{suffix}"] = result[f"b365_corner_{suffix}"] - result[f"avg_corner_{suffix}"]

    return result


def generate_forward_bets(
    df: pd.DataFrame,
    config: dict,
) -> list[dict]:
    """Apply strategies to fixtures and generate recommended bets.

    Args:
        df: DataFrame with computed forward features.
        config: Forward test config dict.

    Returns:
        List of bet dicts with sizing info.
    """
    strategies_cfg = config.get("strategies", {})
    commission_pct = config.get("commission_pct", 3.8)
    kelly_cfg = config.get("kelly", {})
    kelly_fraction = kelly_cfg.get("fraction", 0.25)
    max_bet_fraction = kelly_cfg.get("max_bet_fraction", 0.05)
    bankroll = kelly_cfg.get("bankroll", 1000.0)

    bets = []

    # --- 1X2 Disagree ---
    s1x2_cfg = strategies_cfg.get("1x2_disagree", {})
    if s1x2_cfg.get("enabled", False):
        has_disagree = all(f"odds_disagree_{s}" in df.columns for s in ["h", "d", "a"])
        if not has_disagree:
            print("  WARNING: Skipping 1X2 strategy (missing PSH/PSD/PSA)")
        else:
            selector_path = MODELS_DIR / "disagreement_selector.pkl"
            if selector_path.exists():
                selector = DisagreementSelector.load(selector_path)
                pct = s1x2_cfg.get("percentile", 7)
                if s1x2_cfg.get("use_league_filter", True) and selector.profitable_leagues:
                    masks = selector.select_with_league_filter(
                        df, df["league"], percentile=pct,
                        allowed_leagues=selector.profitable_leagues,
                    )
                else:
                    masks = selector.select(df, percentile=pct)

                for outcome in ["H", "D", "A"]:
                    suffix = {"H": "h", "D": "d", "A": "a"}[outcome]
                    odds_col = B365_1X2_COLS[outcome]
                    prob_col = f"ps_prob_{suffix}"
                    indices = np.where(masks[outcome])[0]
                    for idx in indices:
                        odds_val = df.iloc[idx][odds_col]
                        prob_val = df.iloc[idx].get(prob_col, np.nan)
                        if not np.isfinite(odds_val) or odds_val <= 1.0 or not np.isfinite(prob_val):
                            continue
                        adj_odds = float(adjust_odds_for_costs(np.array([odds_val]), commission_pct)[0])
                        if adj_odds <= 1.0:
                            continue
                        raw_f = compute_kelly_fraction(prob_val, adj_odds)
                        if raw_f <= 0:
                            continue
                        f = min(raw_f * kelly_fraction, max_bet_fraction)
                        stake = f * bankroll
                        bets.append({
                            "date": df.iloc[idx]["Date"],
                            "home_team": df.iloc[idx]["home_team"],
                            "away_team": df.iloc[idx]["away_team"],
                            "market": "1X2",
                            "outcome": outcome,
                            "odds": float(odds_val),
                            "kelly_pct": round(f * 100, 2),
                            "stake": round(stake, 2),
                            "league": df.iloc[idx]["league"],
                        })
            else:
                print(f"  WARNING: 1X2 selector not found at {selector_path}")

    # --- O/U Disagree ---
    sou_cfg = strategies_cfg.get("ou_disagree", {})
    if sou_cfg.get("enabled", False):
        has_disagree = all(f"ou25_disagree_{s}" in df.columns for s in ["over", "under"])
        if not has_disagree:
            print("  WARNING: Skipping O/U strategy (missing P>2.5/P<2.5)")
        else:
            selector_path = MODELS_DIR / "ou25_disagreement_selector.pkl"
            if selector_path.exists():
                selector = GenericDisagreementSelector.load(selector_path)
                pct = sou_cfg.get("percentile", 3)
                if sou_cfg.get("use_league_filter", True) and selector.profitable_leagues:
                    masks = selector.select_with_league_filter(
                        df, df["league"], percentile=pct,
                        allowed_leagues=selector.profitable_leagues,
                    )
                else:
                    masks = selector.select(df, percentile=pct)

                for outcome in ["Over", "Under"]:
                    suffix = {"Over": "over", "Under": "under"}[outcome]
                    odds_col = B365_OU_COLS[outcome]
                    prob_col = f"ps_ou25_{suffix}"
                    if odds_col not in df.columns:
                        continue
                    indices = np.where(masks[outcome])[0]
                    for idx in indices:
                        odds_val = df.iloc[idx][odds_col]
                        prob_val = df.iloc[idx].get(prob_col, np.nan)
                        if not np.isfinite(odds_val) or odds_val <= 1.0 or not np.isfinite(prob_val):
                            continue
                        adj_odds = float(adjust_odds_for_costs(np.array([odds_val]), commission_pct)[0])
                        if adj_odds <= 1.0:
                            continue
                        raw_f = compute_kelly_fraction(prob_val, adj_odds)
                        if raw_f <= 0:
                            continue
                        f = min(raw_f * kelly_fraction, max_bet_fraction)
                        stake = f * bankroll
                        bets.append({
                            "date": df.iloc[idx]["Date"],
                            "home_team": df.iloc[idx]["home_team"],
                            "away_team": df.iloc[idx]["away_team"],
                            "market": "O/U",
                            "outcome": outcome,
                            "odds": float(odds_val),
                            "kelly_pct": round(f * 100, 2),
                            "stake": round(stake, 2),
                            "league": df.iloc[idx]["league"],
                        })
            else:
                print(f"  WARNING: O/U selector not found at {selector_path}")

    # --- Corner Disagree ---
    scorner_cfg = strategies_cfg.get("corner_disagree", {})
    if scorner_cfg.get("enabled", False):
        has_disagree = all(f"corner_disagree_{s}" in df.columns for s in ["h", "d", "a"])
        if not has_disagree:
            print("  WARNING: Skipping Corners strategy (missing AvgCH/AvgCD/AvgCA)")
        else:
            selector_path = MODELS_DIR / "corner_disagreement_selector.pkl"
            if selector_path.exists():
                selector = GenericDisagreementSelector.load(selector_path)
                pct = scorner_cfg.get("percentile", 15)
                if scorner_cfg.get("use_league_filter", True) and selector.profitable_leagues:
                    masks = selector.select_with_league_filter(
                        df, df["league"], percentile=pct,
                        allowed_leagues=selector.profitable_leagues,
                    )
                else:
                    masks = selector.select(df, percentile=pct)

                for outcome in ["H", "D", "A"]:
                    suffix = {"H": "h", "D": "d", "A": "a"}[outcome]
                    odds_col = B365_CORNER_COLS[outcome]
                    prob_col = f"avg_corner_{suffix}"
                    if odds_col not in df.columns:
                        continue
                    indices = np.where(masks[outcome])[0]
                    for idx in indices:
                        odds_val = df.iloc[idx][odds_col]
                        prob_val = df.iloc[idx].get(prob_col, np.nan)
                        if not np.isfinite(odds_val) or odds_val <= 1.0 or not np.isfinite(prob_val):
                            continue
                        adj_odds = float(adjust_odds_for_costs(np.array([odds_val]), commission_pct)[0])
                        if adj_odds <= 1.0:
                            continue
                        raw_f = compute_kelly_fraction(prob_val, adj_odds)
                        if raw_f <= 0:
                            continue
                        f = min(raw_f * kelly_fraction, max_bet_fraction)
                        stake = f * bankroll
                        bets.append({
                            "date": df.iloc[idx]["Date"],
                            "home_team": df.iloc[idx]["home_team"],
                            "away_team": df.iloc[idx]["away_team"],
                            "market": "Corners",
                            "outcome": outcome,
                            "odds": float(odds_val),
                            "kelly_pct": round(f * 100, 2),
                            "stake": round(stake, 2),
                            "league": df.iloc[idx]["league"],
                        })
            else:
                print(f"  WARNING: Corner selector not found at {selector_path}")

    return bets


def print_forward_bets(bets: list[dict], bankroll: float) -> None:
    """Print formatted table of recommended bets."""
    if not bets:
        print("\n  No bets recommended for these fixtures.")
        return

    print(f"\n{'=' * 95}")
    print("=== RECOMMENDED BETS ===")
    print(f"{'=' * 95}")
    print(f"  {'Date':<12} {'Match':<30} {'Market':<10} {'Bet':<8} {'Odds':>6} "
          f"{'Kelly%':>7} {'Stake':>8} {'League':<10}")
    print(f"  {'-' * 92}")

    total_stake = 0.0
    for b in sorted(bets, key=lambda x: (x["date"], x["home_team"])):
        match_str = f"{b['home_team']} vs {b['away_team']}"
        if len(match_str) > 28:
            match_str = match_str[:28]
        date_str = str(b["date"])[:10]
        print(f"  {date_str:<12} {match_str:<30} {b['market']:<10} {b['outcome']:<8} "
              f"{b['odds']:>6.2f} {b['kelly_pct']:>6.2f}% {b['stake']:>8.2f} {b['league']:<10}")
        total_stake += b["stake"]

    pct_of_bankroll = total_stake / bankroll * 100 if bankroll > 0 else 0
    print(f"\n  Total: {len(bets)} bets | Stake: {total_stake:.2f} ({pct_of_bankroll:.1f}% of bankroll)")


def main(
    fixtures_path: Path | None = None,
    config_path: Path | None = None,
    bankroll_override: float | None = None,
) -> None:
    """Run forward test on upcoming fixtures."""
    config_path = config_path or DEFAULT_CONFIG
    config = load_config(config_path)

    if bankroll_override is not None:
        config.setdefault("kelly", {})["bankroll"] = bankroll_override

    if fixtures_path is None:
        print("ERROR: --fixtures path is required.")
        return

    print(f"Loading fixtures from {fixtures_path}")
    df = pd.read_csv(fixtures_path)
    print(f"  Fixtures loaded: {len(df)} matches")

    # Ensure Date column is parsed
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    # Ensure required columns
    if "home_team" not in df.columns or "away_team" not in df.columns:
        print("ERROR: Fixtures CSV must have 'home_team' and 'away_team' columns.")
        return
    if "league" not in df.columns:
        print("ERROR: Fixtures CSV must have a 'league' column.")
        return

    # Compute features
    df = compute_forward_features(df)

    # Generate bets
    bets = generate_forward_bets(df, config)

    # Print results
    bankroll = config.get("kelly", {}).get("bankroll", 1000.0)
    print_forward_bets(bets, bankroll)

    # Save output
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    if bets:
        bets_df = pd.DataFrame(bets)
        out_path = PREDICTIONS_DIR / "forward_bets.csv"
        bets_df.to_csv(out_path, index=False)
        print(f"\n  Saved {out_path}")
    else:
        print("\n  No bets to save.")


def cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Forward test: generate bets on upcoming fixtures")
    parser.add_argument(
        "--fixtures",
        type=Path,
        required=True,
        help="Path to fixtures CSV with upcoming match odds",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to forward test config YAML",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=None,
        help="Override bankroll from config",
    )
    args = parser.parse_args()
    main(
        fixtures_path=args.fixtures,
        config_path=args.config,
        bankroll_override=args.bankroll,
    )


if __name__ == "__main__":
    cli()
