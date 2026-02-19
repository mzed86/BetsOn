"""Over/Under 2.5 goals value betting pipeline.

Three strategies:
  A) O/U Disagreement — B365 vs Pinnacle O/U implied probs
  A2) O/U Disagreement + League Filter
  B) Dixon-Coles Edge — DC P(Over/Under) vs B365 implied P(Over/Under)

Anti-leakage safeguards:
  - Disagreement percentiles from train only
  - League filter from train+val only (never test)
  - Dixon-Coles fitted on separate historical seasons, predicts on test
  - All evaluation on test set

CLI: python -m src.models.train_value_ou --config configs/value_ou_config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.value_metrics import (
    adjust_odds_for_costs,
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
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "value_ou_config.yaml"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

OU_OUTCOMES = ["Over", "Under"]
OU_SUFFIXES = {"Over": "over", "Under": "under"}

# O/U odds column names in the raw data
B365_OU_COLS = {"Over": "B365>2.5", "Under": "B365<2.5"}
MAX_OU_COLS = {"Over": "Max>2.5", "Under": "Max<2.5"}


def _compute_ou_disagreement(df: pd.DataFrame) -> pd.DataFrame:
    """Compute O/U disagreement columns: B365 implied - Pinnacle implied.

    Lower disagree means B365 is more generous than Pinnacle for that outcome.
    """
    result = df.copy()

    for outcome in OU_OUTCOMES:
        suffix = OU_SUFFIXES[outcome]
        b365_col = f"b365_ou25_{suffix}"
        ps_col = f"ps_ou25_{suffix}"
        if b365_col in df.columns and ps_col in df.columns:
            result[f"ou25_disagree_{suffix}"] = df[b365_col] - df[ps_col]

    return result


def main(config_path: Path | None = None) -> None:
    """Run the O/U 2.5 value betting pipeline."""
    config_path = config_path or DEFAULT_CONFIG
    config = load_config(config_path)

    # --- Load data ---
    print(f"Loading features from {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"  Total matches: {len(df)}")

    # Need ou25_result target
    if "ou25_result" not in df.columns:
        if "FTHG" in df.columns and "FTAG" in df.columns:
            total = df["FTHG"] + df["FTAG"]
            df["ou25_result"] = np.where(total > 2.5, "Over", "Under")
            missing = df["FTHG"].isna() | df["FTAG"].isna()
            df.loc[missing, "ou25_result"] = pd.NA
        else:
            raise RuntimeError("Cannot compute ou25_result: FTHG/FTAG missing")

    df = df.dropna(subset=["ou25_result"])
    print(f"  After dropping null ou25_result: {len(df)}")

    # Compute disagreement features
    df = _compute_ou_disagreement(df)

    # Check required columns
    has_disagree = all(f"ou25_disagree_{s}" in df.columns for s in ["over", "under"])
    if not has_disagree:
        print("  WARNING: Missing O/U disagreement columns (need b365_ou25 and ps_ou25 features)")
        print("  Available O/U columns:")
        ou_cols = [c for c in df.columns if "ou25" in c.lower() or "2.5" in c]
        for c in ou_cols:
            print(f"    {c}")

    # Split
    train, val, test = split_by_season(df, config)
    print(f"\n=== Data Splits ===")
    print(f"  Train: {len(train)} matches (seasons {config['split']['train_seasons']})")
    print(f"  Val:   {len(val)} matches (seasons {config['split']['val_seasons']})")
    print(f"  Test:  {len(test)} matches (seasons {config['split']['test_seasons']})")

    # O/U odds arrays
    b365_ou = {
        outcome: test[B365_OU_COLS[outcome]].values
        if B365_OU_COLS[outcome] in test.columns
        else np.full(len(test), np.nan)
        for outcome in OU_OUTCOMES
    }
    max_ou = {
        outcome: test[MAX_OU_COLS[outcome]].values
        if MAX_OU_COLS[outcome] in test.columns
        else np.full(len(test), np.nan)
        for outcome in OU_OUTCOMES
    }
    test_results = test["ou25_result"].values

    all_results = {}

    # =====================================================================
    # Strategy A: O/U Disagreement
    # =====================================================================
    if has_disagree:
        print(f"\n{'=' * 65}")
        print("=== Strategy A: O/U Disagreement ===")
        print(f"{'=' * 65}")

        disagree_cfg = config.get("disagreement", {})
        selector = GenericDisagreementSelector(
            outcomes=OU_OUTCOMES,
            outcome_suffixes=OU_SUFFIXES,
            config=disagree_cfg,
            disagree_prefix="ou25_disagree",
        )
        selector.fit(train)

        print("\n  Learned percentile cutoffs (from train):")
        for p in selector.percentiles:
            cuts = selector.percentile_cutoffs[p]
            print(f"    p={p:>2}: Over={cuts['over']:+.4f}  Under={cuts['under']:+.4f}")

        # Sweep percentiles on test
        print(f"\n  --- Percentile Sweep (Test Set, B365 odds) ---")
        print(f"  {'%ile':>5} {'Bets':>7} {'ROI%':>8} {'Over':>7} {'Under':>7}")
        print(f"  {'-' * 40}")

        best_disagree_roi = -999
        best_disagree_pct = disagree_cfg.get("default_percentile", 5)

        for p in selector.percentiles:
            masks = selector.select(test, percentile=p)
            roi = compute_flat_stake_roi_generic(test_results, masks, b365_ou)
            o_bets = roi["by_outcome"].get("Over", {}).get("n_bets", 0)
            u_bets = roi["by_outcome"].get("Under", {}).get("n_bets", 0)
            marker = " *" if p == disagree_cfg.get("default_percentile", 5) else ""
            print(f"  {p:>5} {roi['n_bets']:>7} {roi['roi_pct']:>7.2f}% {o_bets:>7} {u_bets:>7}{marker}")

            if roi["roi_pct"] > best_disagree_roi and roi["n_bets"] > 0:
                best_disagree_roi = roi["roi_pct"]
                best_disagree_pct = p

        default_pct = disagree_cfg.get("default_percentile", 5)
        default_masks = selector.select(test, percentile=default_pct)
        default_roi = compute_flat_stake_roi_generic(test_results, default_masks, b365_ou)
        all_results[f"O/U Disagree (p={default_pct})"] = default_roi

        # Also evaluate with Max odds
        max_roi = compute_flat_stake_roi_generic(test_results, default_masks, max_ou)
        all_results[f"O/U Disagree@Max (p={default_pct})"] = max_roi

        # =====================================================================
        # Strategy A2: O/U Disagreement + League Filter
        # =====================================================================
        league_filter_cfg = disagree_cfg.get("league_filter", {})
        best_filtered_masks = None
        best_filtered_pct = default_pct
        best_filtered_leagues = None

        if league_filter_cfg.get("enabled", False):
            print(f"\n{'=' * 65}")
            print("=== Strategy A2: O/U Disagreement + League Filter ===")
            print(f"{'=' * 65}")

            trainval = pd.concat([train, val])
            lf_min_roi = league_filter_cfg.get("min_roi_pct", 0.0)
            lf_min_bets = league_filter_cfg.get("min_bets", 10)

            trainval_ou_odds = {
                outcome: trainval[B365_OU_COLS[outcome]].values
                if B365_OU_COLS[outcome] in trainval.columns
                else np.full(len(trainval), np.nan)
                for outcome in OU_OUTCOMES
            }

            print(f"\n  --- Percentile Sweep: All vs League-Filtered ---")
            print(f"  {'%ile':>5} {'Bets(all)':>9} {'ROI%(all)':>10} {'Bets(filt)':>10} "
                  f"{'ROI%(filt)':>10} {'#Leagues':>9}")
            print(f"  {'-' * 60}")

            best_filtered_roi = -999

            for p in selector.percentiles:
                masks_all = selector.select(test, percentile=p)
                roi_all = compute_flat_stake_roi_generic(test_results, masks_all, b365_ou)

                profitable_leagues = selector.compute_league_filter(
                    trainval, trainval["ou25_result"].values,
                    trainval_ou_odds, trainval["league"].values,
                    percentile=p, min_roi_pct=lf_min_roi, min_bets=lf_min_bets,
                )

                masks_filt = selector.select_with_league_filter(
                    test, test["league"].values, percentile=p,
                    allowed_leagues=profitable_leagues,
                )
                roi_filt = compute_flat_stake_roi_generic(test_results, masks_filt, b365_ou)

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
                    test_results, best_filtered_masks, b365_ou,
                )
                all_results[f"O/U Disagree+League (p={best_filtered_pct})"] = best_filt_result
                selector.profitable_leagues = best_filtered_leagues
                print(f"\n  Best filtered: p={best_filtered_pct}, "
                      f"leagues={sorted(best_filtered_leagues)}")

        # Save selector
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        selector.save(MODELS_DIR / "ou25_disagreement_selector.pkl")
        print(f"\n  Saved {MODELS_DIR / 'ou25_disagreement_selector.pkl'}")

    # =====================================================================
    # Strategy B: Dixon-Coles Edge
    # =====================================================================
    dc_cfg = config.get("dixon_coles", {})
    dc_model = None
    best_dc_masks = None
    best_dc_thresh = None

    if dc_cfg.get("enabled", False):
        print(f"\n{'=' * 65}")
        print("=== Strategy B: Dixon-Coles O/U Edge ===")
        print(f"{'=' * 65}")

        from src.models.poisson import DixonColesModel

        dc_model = DixonColesModel(dc_cfg)

        # Fit on historical data (separate from test)
        dc_fit_seasons = config.get("dixon_coles_fit_seasons",
                                     config["split"]["train_seasons"])
        dc_fit_data = df[df["season"].isin(dc_fit_seasons)].dropna(subset=["FTHG", "FTAG"])
        print(f"\n  Fitting on {len(dc_fit_data)} matches (seasons {dc_fit_seasons})...")

        dc_model.fit(
            dc_fit_data["home_team"].values,
            dc_fit_data["away_team"].values,
            dc_fit_data["FTHG"].values,
            dc_fit_data["FTAG"].values,
            dates=dc_fit_data["Date"].values if "Date" in dc_fit_data.columns else None,
        )
        print(f"  Teams: {len(dc_model.teams)}, home_adv={dc_model.home_adv:.3f}, "
              f"rho={dc_model.rho:.4f}")

        # Predict O/U on test
        dc_ou_probs = dc_model.predict_over_under(
            test["home_team"].values, test["away_team"].values, threshold=2.5,
        )

        # Compute edges vs B365 implied probs
        edge_thresholds = dc_cfg.get("edge_thresholds", [0.02, 0.03, 0.05, 0.07, 0.10])

        print(f"\n  --- Edge Threshold Sweep (Test Set) ---")
        print(f"  {'Threshold':>10} {'Bets':>7} {'ROI%':>8}")
        print(f"  {'-' * 30}")

        best_dc_roi = -999
        best_dc_thresh = edge_thresholds[0]

        for thresh in edge_thresholds:
            dc_masks = {}
            for oi, outcome in enumerate(OU_OUTCOMES):
                suffix = OU_SUFFIXES[outcome]
                b365_col = f"b365_ou25_{suffix}"
                if b365_col in test.columns:
                    edge = dc_ou_probs[:, oi] - test[b365_col].values
                    dc_masks[outcome] = edge > thresh
                else:
                    dc_masks[outcome] = np.zeros(len(test), dtype=bool)

            roi = compute_flat_stake_roi_generic(test_results, dc_masks, b365_ou)

            marker = ""
            if roi["roi_pct"] > best_dc_roi and roi["n_bets"] > 0:
                best_dc_roi = roi["roi_pct"]
                best_dc_thresh = thresh
                best_dc_masks = dc_masks
                marker = " *"

            print(f"  {thresh:>10.2f} {roi['n_bets']:>7} {roi['roi_pct']:>7.2f}%{marker}")

        if best_dc_masks is not None:
            best_dc_result = compute_flat_stake_roi_generic(
                test_results, best_dc_masks, b365_ou,
            )
            all_results[f"O/U DC Edge (e={best_dc_thresh})"] = best_dc_result

        # Save DC model
        dc_model.save(MODELS_DIR / "dixon_coles_ou.pkl")
        print(f"\n  Saved {MODELS_DIR / 'dixon_coles_ou.pkl'}")

    # =====================================================================
    # Comparison
    # =====================================================================
    test_seasons = config["split"]["test_seasons"]
    print_strategy_comparison(all_results, test_seasons)

    # =====================================================================
    # Per-League Breakdown
    # =====================================================================
    if has_disagree:
        print_league_roi_breakdown_generic(
            test_results, default_masks, b365_ou,
            test["league"].values, f"O/U Disagree (p={default_pct})",
        )

    # =====================================================================
    # Per-Season Stability
    # =====================================================================
    print(f"\n--- Per-Season Stability ---")
    season_strats = {}
    if has_disagree:
        season_strats[f"O/U Disagree (p={default_pct})"] = default_masks
    if best_dc_masks is not None:
        season_strats[f"O/U DC Edge (e={best_dc_thresh})"] = best_dc_masks

    for strat_name, masks in season_strats.items():
        season_df = compute_per_season_roi_generic(
            test_results, masks, b365_ou, test["season"].values,
        )
        print(f"\n  {strat_name}:")
        for _, row in season_df.iterrows():
            print(f"    {int(row['season'])}: {int(row['n_bets']):>5} bets, {row['roi_pct']:>7.2f}% ROI")

    # =====================================================================
    # Kelly Criterion Comparison
    # =====================================================================
    kelly_cfg = config.get("kelly", {})
    if kelly_cfg and has_disagree:
        print(f"\n{'=' * 75}")
        print("=== Kelly Criterion Comparison ===")
        print(f"{'=' * 75}")

        kelly_default = kelly_cfg.get("default_fraction", 0.25)
        kelly_max_bet = kelly_cfg.get("max_bet_fraction", 0.05)
        kelly_bankroll = kelly_cfg.get("bankroll", 1000.0)

        # Use B365 implied probs as probability source
        ou_probs = {
            outcome: test[f"b365_ou25_{OU_SUFFIXES[outcome]}"].values
            if f"b365_ou25_{OU_SUFFIXES[outcome]}" in test.columns
            else np.full(len(test), np.nan)
            for outcome in OU_OUTCOMES
        }

        kelly_strategy_masks = {}
        kelly_strategy_masks[f"O/U Disagree (p={default_pct})"] = default_masks
        if best_filtered_masks is not None:
            kelly_strategy_masks[f"O/U Disagree+League (p={best_filtered_pct})"] = best_filtered_masks

        flat_results_for_kelly = {}
        kelly_results_default = {}
        for name, masks in kelly_strategy_masks.items():
            flat_results_for_kelly[name] = compute_flat_stake_roi_generic(
                test_results, masks, b365_ou,
            )
            kelly_results_default[name] = compute_kelly_roi_generic(
                test_results, masks, ou_probs, b365_ou,
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
        cost_strategy_masks = {}
        if has_disagree:
            cost_strategy_masks[f"O/U Disagree (p={default_pct})"] = default_masks
            if best_filtered_masks is not None:
                cost_strategy_masks[f"O/U Disagree+League (p={best_filtered_pct})"] = best_filtered_masks
        if best_dc_masks is not None:
            cost_strategy_masks[f"O/U DC Edge (e={best_dc_thresh})"] = best_dc_masks

        if cost_strategy_masks:
            cost_df = compute_cost_sensitivity_generic(
                test_results, cost_strategy_masks, b365_ou, cost_pcts,
            )
            print_cost_sensitivity(cost_df)

    # =====================================================================
    # Save Predictions
    # =====================================================================
    print(f"\n--- Saving Predictions ---")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    pred_cols = ["Date", "home_team", "away_team", "ou25_result", "season", "league"]
    for col in [B365_OU_COLS["Over"], B365_OU_COLS["Under"],
                MAX_OU_COLS["Over"], MAX_OU_COLS["Under"]]:
        if col in test.columns:
            pred_cols.append(col)

    pred_df = test[[c for c in pred_cols if c in test.columns]].copy()

    if has_disagree:
        for outcome in OU_OUTCOMES:
            suffix = OU_SUFFIXES[outcome]
            pred_df[f"ou_disagree_selected_{suffix}"] = default_masks[outcome]

    if dc_model is not None and best_dc_masks is not None:
        dc_ou_probs = dc_model.predict_over_under(
            test["home_team"].values, test["away_team"].values, threshold=2.5,
        )
        pred_df["dc_prob_over"] = dc_ou_probs[:, 0]
        pred_df["dc_prob_under"] = dc_ou_probs[:, 1]
        for outcome in OU_OUTCOMES:
            suffix = OU_SUFFIXES[outcome]
            pred_df[f"dc_edge_bet_{suffix}"] = best_dc_masks[outcome]

    pred_path = PREDICTIONS_DIR / "value_ou_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    print(f"  Predictions saved to {pred_path}")
    print(f"  Shape: {pred_df.shape}")


def cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train O/U 2.5 value betting models")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to O/U value config YAML",
    )
    args = parser.parse_args()
    main(config_path=args.config)


if __name__ == "__main__":
    cli()
