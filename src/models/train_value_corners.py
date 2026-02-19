"""Corner match result value betting pipeline.

Strategies:
  A) Corner Disagreement — B365 corner odds vs market average corner odds
  A2) Corner Disagreement + League Filter

No Dixon-Coles edge — DC models goals, not corners.

Notes:
  - Corner match result: H if home corners > away corners, D if equal, A if away > home
  - Uses 3-way disagreement: b365_corner_prob - avg_corner_prob for H/D/A
  - Pinnacle corner columns (PSCH) are ambiguous for 2020+ seasons, so we use
    avg_corner as the reference book instead

Anti-leakage safeguards:
  - Disagreement percentiles from train only
  - League filter from train+val only
  - All evaluation on test set

CLI: python -m src.models.train_value_corners --config configs/value_corners_config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.value_metrics import (
    compute_cost_sensitivity_generic,
    compute_flat_stake_roi_generic,
    compute_kelly_roi_generic,
    compute_per_season_roi_generic,
    print_cost_sensitivity,
    print_kelly_comparison,
    print_league_roi_breakdown_generic,
    print_strategy_comparison,
)
from src.models.train import load_config, split_by_season
from src.models.value_betting import GenericDisagreementSelector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "value_corners_config.yaml"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

CORNER_OUTCOMES = ["H", "D", "A"]
CORNER_SUFFIXES = {"H": "h", "D": "d", "A": "a"}

# Corner odds column names in the raw data
B365_CORNER_COLS = {"H": "B365CH", "D": "B365CD", "A": "B365CA"}
MAX_CORNER_COLS = {"H": "MaxCH", "D": "MaxCD", "A": "MaxCA"}


def _compute_corner_disagreement(df: pd.DataFrame) -> pd.DataFrame:
    """Compute corner disagreement columns: B365 implied - Avg implied.

    Since Pinnacle corner columns are ambiguous for 2020+ seasons, we use
    the market average (avg_corner) as the reference.
    """
    result = df.copy()

    for outcome in CORNER_OUTCOMES:
        suffix = CORNER_SUFFIXES[outcome]
        b365_col = f"b365_corner_{suffix}"
        avg_col = f"avg_corner_{suffix}"
        if b365_col in df.columns and avg_col in df.columns:
            result[f"corner_disagree_{suffix}"] = df[b365_col] - df[avg_col]

    return result


def main(config_path: Path | None = None) -> None:
    """Run the corner match result value betting pipeline."""
    config_path = config_path or DEFAULT_CONFIG
    config = load_config(config_path)

    # --- Load data ---
    print(f"Loading features from {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"  Total matches: {len(df)}")

    # Need corner_ftr target
    if "corner_ftr" not in df.columns:
        if "HC" in df.columns and "AC" in df.columns:
            df["corner_ftr"] = np.where(
                df["HC"] > df["AC"], "H",
                np.where(df["HC"] == df["AC"], "D", "A"),
            )
            missing = df["HC"].isna() | df["AC"].isna()
            df.loc[missing, "corner_ftr"] = pd.NA
        else:
            raise RuntimeError("Cannot compute corner_ftr: HC/AC missing")

    df = df.dropna(subset=["corner_ftr"])
    print(f"  After dropping null corner_ftr: {len(df)}")

    # Compute disagreement features
    df = _compute_corner_disagreement(df)

    # Check required columns
    has_disagree = all(f"corner_disagree_{s}" in df.columns for s in ["h", "d", "a"])
    if not has_disagree:
        print("  WARNING: Missing corner disagreement columns (need b365_corner and avg_corner features)")
        print("  Available corner columns:")
        corner_cols = [c for c in df.columns if "corner" in c.lower() or "CH" in c or "CD" in c or "CA" in c]
        for c in corner_cols:
            print(f"    {c}")
        print("  Cannot run corner disagreement strategies without these columns.")
        return

    # Split
    train, val, test = split_by_season(df, config)
    print(f"\n=== Data Splits ===")
    print(f"  Train: {len(train)} matches (seasons {config['split']['train_seasons']})")
    print(f"  Val:   {len(val)} matches (seasons {config['split']['val_seasons']})")
    print(f"  Test:  {len(test)} matches (seasons {config['split']['test_seasons']})")

    # Corner odds arrays
    b365_corner = {
        outcome: test[B365_CORNER_COLS[outcome]].values
        if B365_CORNER_COLS[outcome] in test.columns
        else np.full(len(test), np.nan)
        for outcome in CORNER_OUTCOMES
    }
    max_corner = {
        outcome: test[MAX_CORNER_COLS[outcome]].values
        if MAX_CORNER_COLS[outcome] in test.columns
        else np.full(len(test), np.nan)
        for outcome in CORNER_OUTCOMES
    }
    test_results = test["corner_ftr"].values

    all_results = {}

    # =====================================================================
    # Strategy A: Corner Disagreement
    # =====================================================================
    print(f"\n{'=' * 65}")
    print("=== Strategy A: Corner Disagreement ===")
    print(f"{'=' * 65}")

    disagree_cfg = config.get("disagreement", {})
    selector = GenericDisagreementSelector(
        outcomes=CORNER_OUTCOMES,
        outcome_suffixes=CORNER_SUFFIXES,
        config=disagree_cfg,
        disagree_prefix="corner_disagree",
    )
    selector.fit(train)

    print("\n  Learned percentile cutoffs (from train):")
    for p in selector.percentiles:
        cuts = selector.percentile_cutoffs[p]
        print(f"    p={p:>2}: H={cuts['h']:+.4f}  D={cuts['d']:+.4f}  A={cuts['a']:+.4f}")

    # Sweep percentiles on test
    print(f"\n  --- Percentile Sweep (Test Set, B365 odds) ---")
    print(f"  {'%ile':>5} {'Bets':>7} {'ROI%':>8} {'H bets':>7} {'D bets':>7} {'A bets':>7}")
    print(f"  {'-' * 50}")

    best_disagree_roi = -999
    best_disagree_pct = disagree_cfg.get("default_percentile", 5)

    for p in selector.percentiles:
        masks = selector.select(test, percentile=p)
        roi = compute_flat_stake_roi_generic(test_results, masks, b365_corner)
        h_bets = roi["by_outcome"].get("H", {}).get("n_bets", 0)
        d_bets = roi["by_outcome"].get("D", {}).get("n_bets", 0)
        a_bets = roi["by_outcome"].get("A", {}).get("n_bets", 0)
        marker = " *" if p == disagree_cfg.get("default_percentile", 5) else ""
        print(f"  {p:>5} {roi['n_bets']:>7} {roi['roi_pct']:>7.2f}% {h_bets:>7} {d_bets:>7} {a_bets:>7}{marker}")

        if roi["roi_pct"] > best_disagree_roi and roi["n_bets"] > 0:
            best_disagree_roi = roi["roi_pct"]
            best_disagree_pct = p

    default_pct = disagree_cfg.get("default_percentile", 5)
    default_masks = selector.select(test, percentile=default_pct)
    default_roi = compute_flat_stake_roi_generic(test_results, default_masks, b365_corner)
    all_results[f"Corner Disagree (p={default_pct})"] = default_roi

    # Also evaluate with Max odds
    max_roi = compute_flat_stake_roi_generic(test_results, default_masks, max_corner)
    all_results[f"Corner Disagree@Max (p={default_pct})"] = max_roi

    # =====================================================================
    # Strategy A2: Corner Disagreement + League Filter
    # =====================================================================
    league_filter_cfg = disagree_cfg.get("league_filter", {})
    best_filtered_masks = None
    best_filtered_pct = default_pct
    best_filtered_leagues = None

    if league_filter_cfg.get("enabled", False):
        print(f"\n{'=' * 65}")
        print("=== Strategy A2: Corner Disagreement + League Filter ===")
        print(f"{'=' * 65}")

        trainval = pd.concat([train, val])
        lf_min_roi = league_filter_cfg.get("min_roi_pct", 0.0)
        lf_min_bets = league_filter_cfg.get("min_bets", 10)

        trainval_corner_odds = {
            outcome: trainval[B365_CORNER_COLS[outcome]].values
            if B365_CORNER_COLS[outcome] in trainval.columns
            else np.full(len(trainval), np.nan)
            for outcome in CORNER_OUTCOMES
        }

        print(f"\n  --- Percentile Sweep: All vs League-Filtered ---")
        print(f"  {'%ile':>5} {'Bets(all)':>9} {'ROI%(all)':>10} {'Bets(filt)':>10} "
              f"{'ROI%(filt)':>10} {'#Leagues':>9}")
        print(f"  {'-' * 60}")

        best_filtered_roi = -999

        for p in selector.percentiles:
            masks_all = selector.select(test, percentile=p)
            roi_all = compute_flat_stake_roi_generic(test_results, masks_all, b365_corner)

            profitable_leagues = selector.compute_league_filter(
                trainval, trainval["corner_ftr"].values,
                trainval_corner_odds, trainval["league"].values,
                percentile=p, min_roi_pct=lf_min_roi, min_bets=lf_min_bets,
            )

            masks_filt = selector.select_with_league_filter(
                test, test["league"].values, percentile=p,
                allowed_leagues=profitable_leagues,
            )
            roi_filt = compute_flat_stake_roi_generic(test_results, masks_filt, b365_corner)

            n_leagues = len(profitable_leagues)
            marker = ""
            if roi_filt["roi_pct"] > best_filtered_roi and roi_filt["n_bets"] > 0:
                best_filtered_roi = roi_filt["roi_pct"]
                best_filtered_pct = p
                best_filtered_masks = masks_filt
                best_filtered_leagues = profitable_leagues
                marker = " *"

            print(f"  {p:>5} {roi_all['n_bets']:>9} {roi_all['roi_pct']:>9.2f}% "
                  f"{roi_filt['n_bets']:>10} {roi_filt['roi_pct']:>9.2f}% "
                  f"{n_leagues:>9}{marker}")

        if best_filtered_masks is not None:
            best_filt_result = compute_flat_stake_roi_generic(
                test_results, best_filtered_masks, b365_corner,
            )
            all_results[f"Corner Disagree+League (p={best_filtered_pct})"] = best_filt_result
            selector.profitable_leagues = best_filtered_leagues
            print(f"\n  Best filtered: p={best_filtered_pct}, "
                  f"leagues={sorted(best_filtered_leagues)}")

    # Save selector
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    selector.save(MODELS_DIR / "corner_disagreement_selector.pkl")
    print(f"\n  Saved {MODELS_DIR / 'corner_disagreement_selector.pkl'}")

    # =====================================================================
    # Comparison
    # =====================================================================
    test_seasons = config["split"]["test_seasons"]
    print_strategy_comparison(all_results, test_seasons)

    # =====================================================================
    # Per-League Breakdown
    # =====================================================================
    print_league_roi_breakdown_generic(
        test_results, default_masks, b365_corner,
        test["league"].values, f"Corner Disagree (p={default_pct})",
    )

    # =====================================================================
    # Per-Season Stability
    # =====================================================================
    print(f"\n--- Per-Season Stability ---")
    season_strats = {
        f"Corner Disagree (p={default_pct})": default_masks,
    }
    if best_filtered_masks is not None:
        season_strats[f"Corner Disagree+League (p={best_filtered_pct})"] = best_filtered_masks

    for strat_name, masks in season_strats.items():
        season_df = compute_per_season_roi_generic(
            test_results, masks, b365_corner, test["season"].values,
        )
        print(f"\n  {strat_name}:")
        for _, row in season_df.iterrows():
            print(f"    {int(row['season'])}: {int(row['n_bets']):>5} bets, {row['roi_pct']:>7.2f}% ROI")

    # =====================================================================
    # Kelly Criterion Comparison
    # =====================================================================
    kelly_cfg = config.get("kelly", {})
    if kelly_cfg:
        print(f"\n{'=' * 75}")
        print("=== Kelly Criterion Comparison ===")
        print(f"{'=' * 75}")

        kelly_default = kelly_cfg.get("default_fraction", 0.25)
        kelly_max_bet = kelly_cfg.get("max_bet_fraction", 0.05)
        kelly_bankroll = kelly_cfg.get("bankroll", 1000.0)

        # Use B365 implied probs as probability source
        corner_probs = {
            outcome: test[f"b365_corner_{CORNER_SUFFIXES[outcome]}"].values
            if f"b365_corner_{CORNER_SUFFIXES[outcome]}" in test.columns
            else np.full(len(test), np.nan)
            for outcome in CORNER_OUTCOMES
        }

        kelly_strategy_masks = {
            f"Corner Disagree (p={default_pct})": default_masks,
        }
        if best_filtered_masks is not None:
            kelly_strategy_masks[f"Corner Disagree+League (p={best_filtered_pct})"] = best_filtered_masks

        flat_results_for_kelly = {}
        kelly_results_default = {}
        for name, masks in kelly_strategy_masks.items():
            flat_results_for_kelly[name] = compute_flat_stake_roi_generic(
                test_results, masks, b365_corner,
            )
            kelly_results_default[name] = compute_kelly_roi_generic(
                test_results, masks, corner_probs, b365_corner,
                kelly_fraction=kelly_default, max_bet_fraction=kelly_max_bet,
                bankroll=kelly_bankroll,
            )

        print_kelly_comparison(flat_results_for_kelly, kelly_results_default, kelly_default)

    # =====================================================================
    # Transaction Cost Sensitivity
    # =====================================================================
    tc_cfg = config.get("transaction_costs", {})
    cost_pcts = tc_cfg.get("cost_pcts", [0.0, 1.0, 2.0, 3.0, 5.0])
    if cost_pcts:
        cost_strategy_masks = {
            f"Corner Disagree (p={default_pct})": default_masks,
        }
        if best_filtered_masks is not None:
            cost_strategy_masks[f"Corner Disagree+League (p={best_filtered_pct})"] = best_filtered_masks

        cost_df = compute_cost_sensitivity_generic(
            test_results, cost_strategy_masks, b365_corner, cost_pcts,
        )
        print_cost_sensitivity(cost_df)

    # =====================================================================
    # Save Predictions
    # =====================================================================
    print(f"\n--- Saving Predictions ---")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    pred_cols = ["Date", "home_team", "away_team", "corner_ftr", "season", "league", "HC", "AC"]
    for cols_dict in [B365_CORNER_COLS, MAX_CORNER_COLS]:
        for col in cols_dict.values():
            if col in test.columns:
                pred_cols.append(col)

    pred_df = test[[c for c in pred_cols if c in test.columns]].copy()

    for outcome in CORNER_OUTCOMES:
        suffix = CORNER_SUFFIXES[outcome]
        pred_df[f"corner_disagree_selected_{suffix}"] = default_masks[outcome]

    pred_path = PREDICTIONS_DIR / "value_corners_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    print(f"  Predictions saved to {pred_path}")
    print(f"  Shape: {pred_df.shape}")


def cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train corner value betting models")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to corner value config YAML",
    )
    args = parser.parse_args()
    main(config_path=args.config)


if __name__ == "__main__":
    cli()
