"""CLV training orchestrator: predict line movement and run value bet selector."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.clv_metrics import (
    compute_clv_bet_roi,
    compute_clv_correlation,
    compute_direction_accuracy,
    compute_mae,
    compute_value_bet_roi,
    print_clv_report,
)
from src.models.baseline import GradientBoostingModel
from src.models.clv import CLVModel
from src.models.train import load_config, split_by_season


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "clv_config.yaml"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

OUTCOMES = ["H", "D", "A"]
OUTCOME_SUFFIXES = {"H": "h", "D": "d", "A": "a"}


def build_clv_features(config: dict, df: pd.DataFrame) -> list[str]:
    """Build feature list for CLV model from config, filtering to columns present in df.

    Args:
        config: CLV config dict with features groups.
        df: DataFrame to check column availability against.

    Returns:
        List of feature column names.
    """
    feat_cfg = config["features"]
    cols = []
    for group in ["opening_odds", "odds_disagreement", "multi_book", "non_odds", "venue", "rolling_extended"]:
        cols.extend(feat_cfg.get(group, []))

    available = [c for c in cols if c in df.columns]
    missing = set(cols) - set(available)
    if missing:
        print(f"  WARNING: Missing CLV features: {sorted(missing)}")
    return available


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CLV targets (closing - opening Pinnacle prob) and odds disagreement features.

    Args:
        df: DataFrame with psc_prob_h/d/a and ps_prob_h/d/a columns.

    Returns:
        DataFrame with added clv_h/d/a, odds_disagree_h/d/a, and per-book
        disagreement columns.
    """
    df = df.copy()
    for suffix in ["h", "d", "a"]:
        # CLV targets: closing - opening Pinnacle implied probability
        df[f"clv_{suffix}"] = df[f"psc_prob_{suffix}"] - df[f"ps_prob_{suffix}"]
        # Odds disagreement features: B365 - Pinnacle opening
        df[f"odds_disagree_{suffix}"] = df[f"b365_prob_{suffix}"] - df[f"ps_prob_{suffix}"]

        # Per-book disagreement (book_prob - Pinnacle opening)
        for book in ["bw", "wh", "vc", "iw"]:
            book_col = f"{book}_prob_{suffix}"
            if book_col in df.columns:
                df[f"odds_disagree_{book}_{suffix}"] = df[book_col] - df[f"ps_prob_{suffix}"]

        # Best-across-books disagreement (most generous book vs Pinnacle)
        if f"min_book_prob_{suffix}" in df.columns:
            df[f"odds_disagree_best_{suffix}"] = (
                df[f"min_book_prob_{suffix}"] - df[f"ps_prob_{suffix}"]
            )
    return df


def _train_clv_model(
    outcome: str,
    config: dict,
    feature_cols: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> CLVModel:
    """Train a single CLV regression model for one outcome.

    Args:
        outcome: "H", "D", or "A".
        config: CLV config dict.
        feature_cols: List of feature column names.
        train: Training DataFrame.
        val: Validation DataFrame.

    Returns:
        Trained CLVModel.
    """
    suffix = OUTCOME_SUFFIXES[outcome]
    target_col = f"clv_{suffix}"

    # Drop rows where target is NaN
    train_valid = train.dropna(subset=[target_col])
    val_valid = val.dropna(subset=[target_col])

    model = CLVModel(config["gradient_boosting"])
    model.fit(
        train_valid[feature_cols],
        train_valid[target_col],
        X_val=val_valid[feature_cols] if len(val_valid) > 0 else None,
        y_val=val_valid[target_col] if len(val_valid) > 0 else None,
    )
    return model


def _load_outcome_model(path: Path) -> GradientBoostingModel | None:
    """Load the pre-trained outcome model for Signal 2, or None if not found."""
    if not path.exists():
        print(f"  WARNING: Outcome model not found at {path}")
        print("  Multi-signal betting will use CLV-only mode.")
        return None
    return GradientBoostingModel.load(path)


def main(config_path: Path | None = None) -> None:
    """Run the full CLV training and value-betting evaluation pipeline."""
    config_path = config_path or DEFAULT_CONFIG
    config = load_config(config_path)

    # Load data
    print(f"Loading features from {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"  Total matches: {len(df)}")

    # Drop rows with missing FTR
    df = df.dropna(subset=["FTR"])
    print(f"  After dropping null FTR: {len(df)}")

    # Compute targets and odds disagreement features
    df = compute_targets(df)

    # Check CLV data availability
    clv_cols = ["clv_h", "clv_d", "clv_a"]
    clv_available = df[clv_cols].notna().all(axis=1).sum()
    print(f"  Matches with complete CLV data: {clv_available} ({clv_available / len(df) * 100:.1f}%)")

    # Split by season
    train, val, test = split_by_season(df, config)
    print(f"\n=== Data Splits ===")
    print(f"  Train: {len(train)} matches (seasons {config['split']['train_seasons']})")
    print(f"  Val:   {len(val)} matches (seasons {config['split']['val_seasons']})")
    print(f"  Test:  {len(test)} matches (seasons {config['split']['test_seasons']})")

    # Build feature set
    feature_cols = build_clv_features(config, df)
    print(f"  CLV features: {len(feature_cols)}")

    # Train 3 CLV models (one per outcome)
    print(f"\n{'=' * 60}")
    print("=== Training CLV Models ===")
    print(f"{'=' * 60}")

    clv_models = {}
    for outcome in OUTCOMES:
        print(f"\n--- Training CLV model for {outcome} ---")
        model = _train_clv_model(outcome, config, feature_cols, train, val)

        best_iter = (
            model.model.best_iteration_
            if hasattr(model.model, "best_iteration_") and model.model.best_iteration_ > 0
            else config["gradient_boosting"]["n_estimators"]
        )
        print(f"  Best iteration: {best_iter}")

        imp = model.get_feature_importance()
        print(f"  Top 10 features:")
        for _, row in imp.head(10).iterrows():
            print(f"    {row['feature']:<35} {row['importance']:>6}")

        clv_models[outcome] = model

    # Generate predictions on test set
    y_true = {}
    y_pred = {}
    for outcome in OUTCOMES:
        suffix = OUTCOME_SUFFIXES[outcome]
        target_col = f"clv_{suffix}"
        y_true[outcome] = test[target_col].values
        y_pred[outcome] = clv_models[outcome].predict(test[feature_cols])

    # CLV prediction quality report
    print_clv_report(
        y_true, y_pred, test["FTR"],
        test["B365H"].values, test["B365D"].values, test["B365A"].values,
        model_name="CLV Prediction Quality",
    )

    # --- Value Bet Selector ---
    print(f"\n{'=' * 60}")
    print("=== Value Bet Selector ===")
    print(f"{'=' * 60}")

    betting_cfg = config.get("betting", {})
    clv_threshold = betting_cfg.get("min_predicted_clv", 0.02)
    edge_threshold = betting_cfg.get("min_edge_vs_b365", 0.03)
    min_signals = betting_cfg.get("min_agreement_signals", 2)

    b365_h = test["B365H"].values
    b365_d = test["B365D"].values
    b365_a = test["B365A"].values

    # 1. Naive betting (bet on most likely outcome)
    print(f"\n--- Naive Betting (bet home on every match) ---")
    valid_naive = np.isfinite(b365_h) & (b365_h > 0)
    n_naive = int(valid_naive.sum())
    naive_wins = (test["FTR"].values[valid_naive] == "H")
    naive_returned = float(np.sum(naive_wins * b365_h[valid_naive]))
    naive_roi = (naive_returned - n_naive) / n_naive * 100 if n_naive > 0 else 0.0
    print(f"  Bets: {n_naive}  ROI: {naive_roi:.2f}%")

    # 2. CLV-only betting
    print(f"\n--- CLV-Only Betting (threshold={clv_threshold}) ---")
    clv_roi = compute_clv_bet_roi(
        test["FTR"].values, y_pred,
        b365_h, b365_d, b365_a,
        clv_threshold=clv_threshold,
    )
    print(f"  Bets: {clv_roi['n_bets']}  ROI: {clv_roi['roi_pct']:.2f}%")
    for outcome in OUTCOMES:
        r = clv_roi["by_outcome"][outcome]
        print(f"    {outcome}: {r['n_bets']} bets, ROI {r['roi_pct']:.2f}%")

    # 3. Multi-signal betting (CLV + outcome model)
    outcome_model = _load_outcome_model(MODELS_DIR / "gbm_all_league.pkl")

    if outcome_model is not None:
        # Get outcome model features — load the model config to build the feature list
        outcome_feature_names = outcome_model.feature_names
        # Check which features are available in test set
        available_outcome_feats = [f for f in outcome_feature_names if f in test.columns]
        if len(available_outcome_feats) < len(outcome_feature_names):
            missing = set(outcome_feature_names) - set(available_outcome_feats)
            print(f"  WARNING: Missing outcome model features: {sorted(missing)}")

        outcome_probs = outcome_model.predict_proba(test[available_outcome_feats])

        print(f"\n--- Multi-Signal Betting (CLV>{clv_threshold} + edge>{edge_threshold}, min_signals={min_signals}) ---")
        multi_roi = compute_value_bet_roi(
            test["FTR"].values,
            outcome_probs,
            b365_h, b365_d, b365_a,
            y_pred,
            clv_threshold=clv_threshold,
            edge_threshold=edge_threshold,
            min_signals=min_signals,
        )
        print(f"  Bets: {multi_roi['n_bets']}  ROI: {multi_roi['roi_pct']:.2f}%")
        for outcome in OUTCOMES:
            r = multi_roi["by_outcome"][outcome]
            print(f"    {outcome}: {r['n_bets']} bets, ROI {r['roi_pct']:.2f}%")

        # Comparison table
        print(f"\n{'=' * 60}")
        print("=== COMPARISON TABLE ===")
        print(f"{'=' * 60}")
        print(f"  {'Strategy':<35} {'Bets':>6} {'ROI%':>8}")
        print(f"  {'-' * 50}")
        print(f"  {'Naive (bet H every match)':<35} {n_naive:>6} {naive_roi:>7.2f}%")
        print(f"  {f'CLV-only (>{clv_threshold})':<35} {clv_roi['n_bets']:>6} {clv_roi['roi_pct']:>7.2f}%")
        print(f"  {f'Multi-signal (>={min_signals} signals)':<35} {multi_roi['n_bets']:>6} {multi_roi['roi_pct']:>7.2f}%")
    else:
        print("\n  Skipping multi-signal betting (no outcome model available).")
        multi_roi = None

    # Per-league breakdown
    print(f"\n{'=' * 60}")
    print("=== Per-League CLV Betting Breakdown ===")
    print(f"{'=' * 60}")
    print(f"  {'League':<28} {'N':>6} {'CLV Bets':>9} {'CLV ROI%':>9}")
    print(f"  {'-' * 55}")

    league_rows = []
    for league in sorted(test["league"].unique()):
        mask = test["league"].values == league
        if mask.sum() < 10:
            continue

        league_pred_clv = {o: y_pred[o][mask] for o in OUTCOMES}
        league_roi = compute_clv_bet_roi(
            test["FTR"].values[mask], league_pred_clv,
            b365_h[mask], b365_d[mask], b365_a[mask],
            clv_threshold=clv_threshold,
        )
        league_rows.append({
            "league": league,
            "n_matches": int(mask.sum()),
            "n_bets": league_roi["n_bets"],
            "roi_pct": league_roi["roi_pct"],
        })
        print(f"  {league:<28} {mask.sum():>6} {league_roi['n_bets']:>9} {league_roi['roi_pct']:>8.2f}%")

    # Save models
    print(f"\n--- Saving Models ---")
    for outcome in OUTCOMES:
        suffix = OUTCOME_SUFFIXES[outcome]
        model_path = MODELS_DIR / f"clv_{suffix}.pkl"
        clv_models[outcome].save(model_path)
        print(f"  Saved {model_path}")

    # Save predictions
    pred_df = test[["Date", "home_team", "away_team", "FTR", "season", "league",
                     "B365H", "B365D", "B365A"]].copy()
    for outcome in OUTCOMES:
        suffix = OUTCOME_SUFFIXES[outcome]
        pred_df[f"clv_true_{suffix}"] = y_true[outcome]
        pred_df[f"clv_pred_{suffix}"] = y_pred[outcome]

    pred_path = PREDICTIONS_DIR / "clv_predictions.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(pred_path, index=False)
    print(f"\n  Predictions saved to {pred_path}")
    print(f"  Shape: {pred_df.shape}")


def cli():
    """CLI entry point with --config argument."""
    parser = argparse.ArgumentParser(description="Train CLV prediction models")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to CLV config YAML",
    )
    args = parser.parse_args()
    main(config_path=args.config)


if __name__ == "__main__":
    cli()
