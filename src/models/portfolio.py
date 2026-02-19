"""Portfolio revenue model: combine all profitable strategies into a single Kelly-sized bankroll simulation.

Loads features.parquet, computes disagreement columns per market, applies saved
selectors at configured percentiles with league filters, then runs a chronological
Kelly simulation across all markets.

CLI: python -m src.models.portfolio --config configs/portfolio_config.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.value_metrics import (
    adjust_odds_for_costs,
    compute_kelly_fraction,
)
from src.models.train import load_config
from src.models.train_clv import compute_targets
from src.models.train_value_corners import (
    B365_CORNER_COLS,
    CORNER_OUTCOMES,
    CORNER_SUFFIXES,
    _compute_corner_disagreement,
)
from src.models.train_value_ou import (
    B365_OU_COLS,
    OU_OUTCOMES,
    OU_SUFFIXES,
    _compute_ou_disagreement,
)
from src.models.value_betting import DisagreementSelector, GenericDisagreementSelector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "portfolio_config.yaml"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "portfolio"

# 1X2 column mappings
B365_1X2_COLS = {"H": "B365H", "D": "B365D", "A": "B365A"}
OUTCOMES_1X2 = ["H", "D", "A"]
SUFFIXES_1X2 = {"H": "h", "D": "d", "A": "a"}


def _build_bet_stream_1x2(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    strategy_cfg: dict,
) -> list[dict]:
    """Build flat bet list from 1X2 disagreement masks."""
    bets = []
    for outcome in OUTCOMES_1X2:
        suffix = SUFFIXES_1X2[outcome]
        odds_col = B365_1X2_COLS[outcome]
        prob_col = f"ps_prob_{suffix}"
        mask = masks[outcome]
        indices = np.where(mask)[0]
        for idx in indices:
            odds_val = df.iloc[idx][odds_col]
            prob_val = df.iloc[idx].get(prob_col, np.nan)
            if not np.isfinite(odds_val) or odds_val <= 1.0:
                continue
            if not np.isfinite(prob_val):
                continue
            bets.append({
                "date": df.iloc[idx]["Date"],
                "match_idx": idx,
                "home_team": df.iloc[idx]["home_team"],
                "away_team": df.iloc[idx]["away_team"],
                "market": "1X2",
                "outcome": outcome,
                "odds": float(odds_val),
                "result": df.iloc[idx]["FTR"],
                "estimated_prob": float(prob_val),
                "season": int(df.iloc[idx]["season"]),
                "league": df.iloc[idx]["league"],
            })
    return bets


def _build_bet_stream_ou(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    strategy_cfg: dict,
) -> list[dict]:
    """Build flat bet list from O/U disagreement masks."""
    bets = []
    for outcome in OU_OUTCOMES:
        suffix = OU_SUFFIXES[outcome]
        odds_col = B365_OU_COLS[outcome]
        prob_col = f"ps_ou25_{suffix}"
        mask = masks[outcome]
        indices = np.where(mask)[0]
        for idx in indices:
            if odds_col not in df.columns:
                continue
            odds_val = df.iloc[idx][odds_col]
            prob_val = df.iloc[idx].get(prob_col, np.nan)
            if not np.isfinite(odds_val) or odds_val <= 1.0:
                continue
            if not np.isfinite(prob_val):
                continue
            bets.append({
                "date": df.iloc[idx]["Date"],
                "match_idx": idx,
                "home_team": df.iloc[idx]["home_team"],
                "away_team": df.iloc[idx]["away_team"],
                "market": "O/U",
                "outcome": outcome,
                "odds": float(odds_val),
                "result": df.iloc[idx]["ou25_result"],
                "estimated_prob": float(prob_val),
                "season": int(df.iloc[idx]["season"]),
                "league": df.iloc[idx]["league"],
            })
    return bets


def _build_bet_stream_corners(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    strategy_cfg: dict,
) -> list[dict]:
    """Build flat bet list from corner disagreement masks."""
    bets = []
    for outcome in CORNER_OUTCOMES:
        suffix = CORNER_SUFFIXES[outcome]
        odds_col = B365_CORNER_COLS[outcome]
        prob_col = f"avg_corner_{suffix}"
        mask = masks[outcome]
        indices = np.where(mask)[0]
        for idx in indices:
            if odds_col not in df.columns:
                continue
            odds_val = df.iloc[idx][odds_col]
            prob_val = df.iloc[idx].get(prob_col, np.nan)
            if not np.isfinite(odds_val) or odds_val <= 1.0:
                continue
            if not np.isfinite(prob_val):
                continue
            bets.append({
                "date": df.iloc[idx]["Date"],
                "match_idx": idx,
                "home_team": df.iloc[idx]["home_team"],
                "away_team": df.iloc[idx]["away_team"],
                "market": "Corners",
                "outcome": outcome,
                "odds": float(odds_val),
                "result": df.iloc[idx]["corner_ftr"],
                "estimated_prob": float(prob_val),
                "season": int(df.iloc[idx]["season"]),
                "league": df.iloc[idx]["league"],
            })
    return bets


def simulate_portfolio_kelly(
    bets: list[dict],
    commission_pct: float = 3.8,
    kelly_fraction: float = 0.25,
    max_bet_fraction: float = 0.05,
    bankroll: float = 1000.0,
) -> dict:
    """Chronological Kelly simulation across all markets.

    All bets on the same date use the same pre-date bankroll. Returns a dict
    with summary stats, per-bet log, bankroll curve, and per-market breakdown.

    Args:
        bets: List of bet dicts with date, odds, estimated_prob, result, outcome, market.
        commission_pct: Commission applied to profit portion of odds.
        kelly_fraction: Fractional Kelly multiplier.
        max_bet_fraction: Maximum fraction of bankroll per bet.
        bankroll: Starting bankroll.

    Returns:
        Dict with keys: summary, bet_log, bankroll_curve, per_market.
    """
    if not bets:
        return {
            "summary": {
                "n_bets": 0,
                "starting_bankroll": bankroll,
                "ending_bankroll": bankroll,
                "profit": 0.0,
                "roi_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "total_staked": 0.0,
                "months": 0,
                "bets_per_month": 0.0,
            },
            "bet_log": [],
            "bankroll_curve": [],
            "per_market": {},
        }

    # Sort by date, then by match_idx for deterministic ordering
    bets_sorted = sorted(bets, key=lambda b: (b["date"], b["match_idx"]))

    starting_bankroll = bankroll
    current_bankroll = bankroll
    peak_bankroll = bankroll
    max_drawdown_pct = 0.0
    total_staked = 0.0
    n_placed = 0

    bet_log = []
    bankroll_curve = [{"date": bets_sorted[0]["date"], "bankroll": bankroll}]
    per_market = {}

    i = 0
    while i < len(bets_sorted):
        date = bets_sorted[i]["date"]
        pre_date_bankroll = current_bankroll

        # Gather all bets for this date
        date_bets = []
        while i < len(bets_sorted) and bets_sorted[i]["date"] == date:
            date_bets.append(bets_sorted[i])
            i += 1

        for bet in date_bets:
            adj_odds = float(adjust_odds_for_costs(
                np.array([bet["odds"]]), commission_pct,
            )[0])
            if adj_odds <= 1.0:
                continue

            raw_f = compute_kelly_fraction(bet["estimated_prob"], adj_odds)
            if raw_f <= 0:
                continue

            f = min(raw_f * kelly_fraction, max_bet_fraction)
            stake = f * pre_date_bankroll
            if stake <= 0:
                continue

            won = bet["result"] == bet["outcome"]
            if won:
                pnl = stake * (adj_odds - 1.0)
            else:
                pnl = -stake

            current_bankroll += pnl
            total_staked += stake
            n_placed += 1

            # Per-market tracking
            mkt = bet["market"]
            if mkt not in per_market:
                per_market[mkt] = {"n_bets": 0, "staked": 0.0, "pnl": 0.0, "wins": 0}
            per_market[mkt]["n_bets"] += 1
            per_market[mkt]["staked"] += stake
            per_market[mkt]["pnl"] += pnl
            if won:
                per_market[mkt]["wins"] += 1

            bet_log.append({
                "date": bet["date"],
                "home_team": bet["home_team"],
                "away_team": bet["away_team"],
                "market": mkt,
                "outcome": bet["outcome"],
                "odds": bet["odds"],
                "adj_odds": round(adj_odds, 4),
                "estimated_prob": bet["estimated_prob"],
                "kelly_f": round(f, 6),
                "stake": round(stake, 2),
                "won": won,
                "pnl": round(pnl, 2),
                "bankroll_after": round(current_bankroll, 2),
                "season": bet["season"],
                "league": bet["league"],
            })

        # Update drawdown after processing the date
        if current_bankroll > peak_bankroll:
            peak_bankroll = current_bankroll
        dd = (peak_bankroll - current_bankroll) / peak_bankroll * 100 if peak_bankroll > 0 else 0.0
        if dd > max_drawdown_pct:
            max_drawdown_pct = dd

        bankroll_curve.append({"date": date, "bankroll": round(current_bankroll, 2)})

    # Compute summary
    profit = current_bankroll - starting_bankroll
    roi_pct = (profit / total_staked * 100) if total_staked > 0 else 0.0

    dates = [b["date"] for b in bets_sorted]
    if dates:
        date_range = pd.Timestamp(max(dates)) - pd.Timestamp(min(dates))
        months = max(date_range.days / 30.44, 1)
    else:
        months = 1
    bets_per_month = n_placed / months

    # Per-market ROI
    for mkt in per_market:
        pm = per_market[mkt]
        pm["roi_pct"] = round(pm["pnl"] / pm["staked"] * 100, 2) if pm["staked"] > 0 else 0.0
        pm["win_rate"] = round(pm["wins"] / pm["n_bets"] * 100, 1) if pm["n_bets"] > 0 else 0.0
        pm["staked"] = round(pm["staked"], 2)
        pm["pnl"] = round(pm["pnl"], 2)

    return {
        "summary": {
            "n_bets": n_placed,
            "starting_bankroll": starting_bankroll,
            "ending_bankroll": round(current_bankroll, 2),
            "profit": round(profit, 2),
            "roi_pct": round(roi_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "total_staked": round(total_staked, 2),
            "months": round(months, 1),
            "bets_per_month": round(bets_per_month, 1),
        },
        "bet_log": bet_log,
        "bankroll_curve": bankroll_curve,
        "per_market": per_market,
    }


def compute_revenue_projections(
    summary: dict,
    bankroll_levels: list[float],
) -> pd.DataFrame:
    """Scale simulated results to different bankroll levels.

    Args:
        summary: Summary dict from simulate_portfolio_kelly.
        bankroll_levels: List of starting bankroll amounts to project.

    Returns:
        DataFrame with columns: bankroll, ending, profit, monthly, yearly.
    """
    if summary["starting_bankroll"] == 0 or summary["months"] == 0:
        return pd.DataFrame(columns=["bankroll", "ending", "profit", "monthly", "yearly"])

    # Scale factor: how much did the bankroll grow?
    growth_factor = summary["ending_bankroll"] / summary["starting_bankroll"]
    months = summary["months"]

    rows = []
    for bk in bankroll_levels:
        ending = bk * growth_factor
        profit = ending - bk
        monthly = profit / months if months > 0 else 0
        yearly = monthly * 12
        rows.append({
            "bankroll": bk,
            "ending": round(ending, 2),
            "profit": round(profit, 2),
            "monthly": round(monthly, 2),
            "yearly": round(yearly, 2),
        })
    return pd.DataFrame(rows)


def print_portfolio_summary(result: dict, test_seasons: list[int]) -> None:
    """Print formatted portfolio summary tables."""
    s = result["summary"]
    pm = result["per_market"]

    n_years = len(test_seasons) if test_seasons else 1

    print(f"\n{'=' * 70}")
    print("=== PORTFOLIO SIMULATION SUMMARY ===")
    print(f"{'=' * 70}")
    print(f"  Period:            {test_seasons[0]}-{test_seasons[-1]} ({n_years} seasons)")
    print(f"  Starting bankroll: {s['starting_bankroll']:,.2f}")
    print(f"  Ending bankroll:   {s['ending_bankroll']:,.2f}")
    print(f"  Profit:            {s['profit']:+,.2f}")
    print(f"  ROI (on staked):   {s['roi_pct']:+.2f}%")
    print(f"  Total staked:      {s['total_staked']:,.2f}")
    print(f"  Total bets:        {s['n_bets']}")
    print(f"  Bets/month:        {s['bets_per_month']:.1f}")
    print(f"  Max drawdown:      {s['max_drawdown_pct']:.2f}%")

    print(f"\n  --- Per-Market Breakdown ---")
    print(f"  {'Market':<12} {'Bets':>7} {'Staked':>10} {'PnL':>10} {'ROI%':>8} {'Win%':>7}")
    print(f"  {'-' * 57}")
    for mkt in sorted(pm.keys()):
        m = pm[mkt]
        print(f"  {mkt:<12} {m['n_bets']:>7} {m['staked']:>10,.2f} {m['pnl']:>+10,.2f} "
              f"{m['roi_pct']:>+7.2f}% {m['win_rate']:>6.1f}%")


def print_revenue_projections(proj_df: pd.DataFrame) -> None:
    """Print formatted revenue projection table."""
    print(f"\n  --- Revenue Projections ---")
    print(f"  {'Bankroll':>10} {'Ending':>12} {'Profit':>12} {'Monthly':>10} {'Yearly':>12}")
    print(f"  {'-' * 60}")
    for _, row in proj_df.iterrows():
        print(f"  {row['bankroll']:>10,.0f} {row['ending']:>12,.2f} {row['profit']:>+12,.2f} "
              f"{row['monthly']:>+10,.2f} {row['yearly']:>+12,.2f}")


def main(config_path: Path | None = None) -> None:
    """Run the portfolio revenue simulation."""
    config_path = config_path or DEFAULT_CONFIG
    config = load_config(config_path)

    # --- Load data ---
    print(f"Loading features from {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"  Total matches: {len(df)}")

    # Drop rows with missing FTR
    df = df.dropna(subset=["FTR"])

    # Compute O/U result if missing
    if "ou25_result" not in df.columns:
        if "FTHG" in df.columns and "FTAG" in df.columns:
            total = df["FTHG"] + df["FTAG"]
            df["ou25_result"] = np.where(total > 2.5, "Over", "Under")
            missing = df["FTHG"].isna() | df["FTAG"].isna()
            df.loc[missing, "ou25_result"] = pd.NA

    # Compute corner result if missing
    if "corner_ftr" not in df.columns:
        if "HC" in df.columns and "AC" in df.columns:
            df["corner_ftr"] = np.where(
                df["HC"] > df["AC"], "H",
                np.where(df["HC"] == df["AC"], "D", "A"),
            )
            missing = df["HC"].isna() | df["AC"].isna()
            df.loc[missing, "corner_ftr"] = pd.NA

    # Compute disagreement columns
    df = compute_targets(df)
    df = _compute_ou_disagreement(df)
    df = _compute_corner_disagreement(df)

    # Filter to test seasons
    test_seasons = config["split"]["test_seasons"]
    test_df = df[df["season"].isin(test_seasons)].copy().reset_index(drop=True)
    print(f"  Test set: {len(test_df)} matches (seasons {test_seasons})")

    strategies_cfg = config.get("strategies", {})
    commission_pct = config.get("commission_pct", 3.8)
    kelly_cfg = config.get("kelly", {})

    all_bets = []

    # --- 1X2 Disagree ---
    s1x2_cfg = strategies_cfg.get("1x2_disagree", {})
    if s1x2_cfg.get("enabled", False):
        selector_path = MODELS_DIR / "disagreement_selector.pkl"
        if selector_path.exists():
            selector_1x2 = DisagreementSelector.load(selector_path)
            pct = s1x2_cfg.get("percentile", 7)
            if s1x2_cfg.get("use_league_filter", True) and selector_1x2.profitable_leagues:
                masks_1x2 = selector_1x2.select_with_league_filter(
                    test_df, test_df["league"], percentile=pct,
                    allowed_leagues=selector_1x2.profitable_leagues,
                )
            else:
                masks_1x2 = selector_1x2.select(test_df, percentile=pct)
            bets_1x2 = _build_bet_stream_1x2(test_df, masks_1x2, s1x2_cfg)
            print(f"  1X2 Disagree (p={pct}): {len(bets_1x2)} bets")
            all_bets.extend(bets_1x2)
        else:
            print(f"  WARNING: 1X2 selector not found at {selector_path}")

    # --- O/U Disagree ---
    sou_cfg = strategies_cfg.get("ou_disagree", {})
    if sou_cfg.get("enabled", False):
        selector_path = MODELS_DIR / "ou25_disagreement_selector.pkl"
        if selector_path.exists():
            selector_ou = GenericDisagreementSelector.load(selector_path)
            pct = sou_cfg.get("percentile", 3)
            if sou_cfg.get("use_league_filter", True) and selector_ou.profitable_leagues:
                masks_ou = selector_ou.select_with_league_filter(
                    test_df, test_df["league"], percentile=pct,
                    allowed_leagues=selector_ou.profitable_leagues,
                )
            else:
                masks_ou = selector_ou.select(test_df, percentile=pct)
            bets_ou = _build_bet_stream_ou(test_df, masks_ou, sou_cfg)
            print(f"  O/U Disagree (p={pct}): {len(bets_ou)} bets")
            all_bets.extend(bets_ou)
        else:
            print(f"  WARNING: O/U selector not found at {selector_path}")

    # --- Corner Disagree ---
    scorner_cfg = strategies_cfg.get("corner_disagree", {})
    if scorner_cfg.get("enabled", False):
        selector_path = MODELS_DIR / "corner_disagreement_selector.pkl"
        if selector_path.exists():
            selector_corner = GenericDisagreementSelector.load(selector_path)
            pct = scorner_cfg.get("percentile", 15)
            if scorner_cfg.get("use_league_filter", True) and selector_corner.profitable_leagues:
                masks_corner = selector_corner.select_with_league_filter(
                    test_df, test_df["league"], percentile=pct,
                    allowed_leagues=selector_corner.profitable_leagues,
                )
            else:
                masks_corner = selector_corner.select(test_df, percentile=pct)
            bets_corner = _build_bet_stream_corners(test_df, masks_corner, scorner_cfg)
            print(f"  Corner Disagree (p={pct}): {len(bets_corner)} bets")
            all_bets.extend(bets_corner)
        else:
            print(f"  WARNING: Corner selector not found at {selector_path}")

    if not all_bets:
        print("\n  No bets generated. Check that selector .pkl files exist in outputs/models/.")
        return

    print(f"\n  Total portfolio bets: {len(all_bets)}")

    # --- Run simulation ---
    result = simulate_portfolio_kelly(
        all_bets,
        commission_pct=commission_pct,
        kelly_fraction=kelly_cfg.get("fraction", 0.25),
        max_bet_fraction=kelly_cfg.get("max_bet_fraction", 0.05),
        bankroll=kelly_cfg.get("bankroll", 1000.0),
    )

    # --- Print results ---
    print_portfolio_summary(result, test_seasons)

    # --- Revenue projections ---
    proj_levels = config.get("revenue_projections", {}).get("bankroll_levels", [1000, 5000, 10000, 25000])
    proj_df = compute_revenue_projections(result["summary"], proj_levels)
    print_revenue_projections(proj_df)

    # --- Save outputs ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    summary_path = OUTPUT_DIR / "portfolio_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"summary": result["summary"], "per_market": result["per_market"]}, f, indent=2, default=str)
    print(f"\n  Saved {summary_path}")

    # Bankroll curve
    curve_df = pd.DataFrame(result["bankroll_curve"])
    curve_path = OUTPUT_DIR / "bankroll_curve.parquet"
    curve_df.to_parquet(curve_path, index=False)
    print(f"  Saved {curve_path}")

    # Per-bet log
    log_df = pd.DataFrame(result["bet_log"])
    log_path = OUTPUT_DIR / "per_bet_log.parquet"
    log_df.to_parquet(log_path, index=False)
    print(f"  Saved {log_path}")

    # Revenue projections
    proj_path = OUTPUT_DIR / "revenue_projections.parquet"
    proj_df.to_parquet(proj_path, index=False)
    print(f"  Saved {proj_path}")


def cli():
    """CLI entry point with --config argument."""
    parser = argparse.ArgumentParser(description="Portfolio revenue simulation")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to portfolio config YAML",
    )
    args = parser.parse_args()
    main(config_path=args.config)


if __name__ == "__main__":
    cli()
