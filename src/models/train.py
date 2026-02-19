"""Training orchestrator: load data, split, train, evaluate, save."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import (
    compute_log_loss,
    compute_roi,
    print_evaluation_report,
)
from src.models.baseline import GradientBoostingModel, OddsBaseline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model_config.yaml"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"


def load_config(path: Path) -> dict:
    """Load model config from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def split_by_season(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by season into train/val/test sets.

    Args:
        df: Full feature DataFrame with a 'season' column.
        config: Config dict with split.train_seasons, val_seasons, test_seasons.

    Returns:
        Tuple of (train, val, test) DataFrames.
    """
    split_cfg = config["split"]
    train = df[df["season"].isin(split_cfg["train_seasons"])].copy()
    val = df[df["season"].isin(split_cfg["val_seasons"])].copy()
    test = df[df["season"].isin(split_cfg["test_seasons"])].copy()
    return train, val, test


def build_feature_sets(config: dict, df: pd.DataFrame) -> dict[str, list[str]]:
    """Build named feature lists from config groups, filtering to columns present in df.

    All feature sets use only pre-match information to avoid look-ahead bias.

    Returns:
        Dict with keys 'all_league', 'tier1', 'no_odds' mapping to feature lists.
    """
    feat_cfg = config["features"]

    def _resolve(groups: list[str]) -> list[str]:
        cols = []
        for group in groups:
            cols.extend(feat_cfg.get(group, []))
        available = [c for c in cols if c in df.columns]
        missing = set(cols) - set(available)
        if missing:
            print(f"  WARNING: Missing features: {sorted(missing)}")
        return available

    return {
        "all_league": _resolve([
            "odds_b365", "odds_pinnacle_open",
            "non_odds", "venue", "rolling_extended",
        ]),
        "tier1": _resolve([
            "odds_b365", "odds_pinnacle_open",
            "non_odds", "venue", "rolling_extended",
            "tier1_xg",
        ]),
        "no_odds": _resolve([
            "non_odds", "venue", "rolling_extended",
        ]),
    }


def print_league_breakdown(
    y_true: pd.Series,
    y_prob: np.ndarray,
    odds_df: pd.DataFrame,
    league_series: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    """Print per-league log-loss and ROI breakdown.

    Args:
        y_true: Series of outcome labels (H/D/A).
        y_prob: (N, 3) probability array.
        odds_df: DataFrame with B365H, B365D, B365A columns.
        league_series: Series of league names aligned with y_true.
        model_name: Name for display.

    Returns:
        DataFrame with per-league metrics.
    """
    print(f"\n--- League Breakdown: {model_name} ---")
    print(f"  {'League':<28} {'N':>6} {'LogLoss':>9} {'ROI%':>8} {'Bets':>6}")
    print(f"  {'-'*60}")

    rows = []
    for league in sorted(league_series.unique()):
        mask = league_series.values == league
        if mask.sum() < 10:
            continue
        ll = compute_log_loss(y_true.iloc[mask], y_prob[mask])
        roi = compute_roi(
            y_true.iloc[mask], y_prob[mask],
            odds_df["B365H"].values[mask],
            odds_df["B365D"].values[mask],
            odds_df["B365A"].values[mask],
        )
        rows.append({
            "league": league,
            "n_matches": int(mask.sum()),
            "log_loss": ll,
            "roi_pct": roi["roi_pct"],
            "n_bets": roi["n_bets"],
        })
        print(f"  {league:<28} {mask.sum():>6} {ll:>9.4f} {roi['roi_pct']:>7.2f}% {roi['n_bets']:>6}")

    return pd.DataFrame(rows)


def _train_and_eval_gbm(
    name: str,
    config: dict,
    feature_cols: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, dict, GradientBoostingModel]:
    """Train LightGBM and evaluate on val + test."""
    print(f"\n{'=' * 55}")
    print(f"=== {name} ===")
    print(f"{'=' * 55}")
    print(f"  Features ({len(feature_cols)}): {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")

    model = GradientBoostingModel(config["gradient_boosting"])
    model.fit(
        train[feature_cols], train["FTR"],
        X_val=val[feature_cols], y_val=val["FTR"],
    )

    best_iter = model.model.best_iteration_ if model.model.best_iteration_ > 0 else config["gradient_boosting"]["n_estimators"]
    print(f"  Best iteration: {best_iter}")

    # Feature importance
    imp = model.get_feature_importance()
    print(f"\n  Feature importance (top 15):")
    for _, row in imp.head(15).iterrows():
        print(f"    {row['feature']:<35} {row['importance']:>6}")

    print("\n[Validation Set]")
    val_probs = model.predict_proba(val[feature_cols])
    print_evaluation_report(val["FTR"], val_probs, val, f"{name} (val)")

    print("\n[Test Set]")
    test_probs = model.predict_proba(test[feature_cols])
    metrics = print_evaluation_report(test["FTR"], test_probs, test, f"{name} (test)")

    return test_probs, metrics, model


def main(config_path: Path | None = None) -> None:
    """Run the full training pipeline."""
    config_path = config_path or DEFAULT_CONFIG
    config = load_config(config_path)

    # Load data
    print(f"Loading features from {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"  Total matches: {len(df)}")

    # Drop rows with missing target
    df = df.dropna(subset=["FTR"])
    print(f"  After dropping null FTR: {len(df)}")

    # Split
    train, val, test = split_by_season(df, config)
    print(f"\n=== Data Splits ===")
    print(f"  Train: {len(train)} matches (seasons {config['split']['train_seasons']})")
    print(f"  Val:   {len(val)} matches (seasons {config['split']['val_seasons']})")
    print(f"  Test:  {len(test)} matches (seasons {config['split']['test_seasons']})")

    # Build feature sets from config (pre-match features only)
    feature_sets = build_feature_sets(config, df)
    for name, cols in feature_sets.items():
        print(f"  Feature set '{name}': {len(cols)} features")

    # Tier 1 subsets
    tier1_leagues = config.get("tier1_leagues", [])
    train_t1 = train[train["league"].isin(tier1_leagues)].copy()
    val_t1 = val[val["league"].isin(tier1_leagues)].copy()
    test_t1 = test[test["league"].isin(tier1_leagues)].copy()
    print(f"\n  Tier 1 leagues: {tier1_leagues}")
    print(f"  Tier 1 train/val/test: {len(train_t1)}/{len(val_t1)}/{len(test_t1)}")

    # Collect test-set metrics for comparison table
    results = {}
    test_predictions = {}

    # --- 1. OddsBaseline (B365) ---
    print(f"\n{'=' * 55}")
    print("=== OddsBaseline (B365 implied probabilities) ===")
    print(f"{'=' * 55}")
    odds_baseline = OddsBaseline()
    odds_test_probs = odds_baseline.predict_proba(test)
    results["OddsBaseline"] = print_evaluation_report(
        test["FTR"], odds_test_probs, test, "OddsBaseline (test)",
    )
    test_predictions["odds_b365"] = odds_test_probs

    # --- 2. GBM All-League ---
    gbm_all_probs, gbm_all_metrics, gbm_all_model = _train_and_eval_gbm(
        "GBM All-League", config, feature_sets["all_league"],
        train, val, test,
    )
    results["GBM AllLeague"] = gbm_all_metrics
    test_predictions["gbm_all"] = gbm_all_probs

    # --- 3. GBM Tier 1 (top 5 leagues, with xG rolling features) ---
    gbm_t1_probs, gbm_t1_metrics, gbm_t1_model = _train_and_eval_gbm(
        "GBM Tier1", config, feature_sets["tier1"],
        train_t1, val_t1, test_t1,
    )
    results["GBM Tier1"] = gbm_t1_metrics
    test_predictions["gbm_tier1"] = gbm_t1_probs

    # --- 4. GBM No-Odds ---
    gbm_no_odds_probs, gbm_no_odds_metrics, gbm_no_odds_model = _train_and_eval_gbm(
        "GBM No-Odds", config, feature_sets["no_odds"],
        train, val, test,
    )
    results["GBM NoOdds"] = gbm_no_odds_metrics
    test_predictions["gbm_no_odds"] = gbm_no_odds_probs

    # --- Comparison table ---
    print(f"\n{'=' * 80}")
    print("=== COMPARISON TABLE (Test Set) ===")
    print(f"{'=' * 80}")
    names = list(results.keys())
    header = f"{'Metric':<12}" + "".join(f"{n:>15}" for n in names)
    print(f"\n{header}")
    print("-" * (12 + 15 * len(names)))
    for metric, key in [("Log-loss", "log_loss"), ("Brier", "brier_score"), ("Accuracy", "accuracy")]:
        vals = "".join(f"{results[n][key]:>15.4f}" for n in names)
        print(f"{metric:<12}{vals}")
    roi_vals = "".join(f"{results[n]['roi']['roi_pct']:>14.2f}%" for n in names)
    print(f"{'ROI %':<12}{roi_vals}")
    bets_vals = "".join(f"{results[n]['roi']['n_bets']:>15}" for n in names)
    print(f"{'Bets':<12}{bets_vals}")
    print("\n  * GBM Tier1 evaluated on Tier 1 leagues only "
          f"({len(test_t1)} matches vs {len(test)} full test set)")
    print("  * All features are pre-match only (no closing odds or CLV)")

    # --- League breakdown for GBM All-League ---
    print_league_breakdown(
        test["FTR"], gbm_all_probs, test, test["league"], "GBM All-League",
    )

    # Also show OddsBaseline league breakdown for comparison
    print_league_breakdown(
        test["FTR"], odds_test_probs, test, test["league"], "OddsBaseline",
    )

    # --- Save models ---
    gbm_all_model.save(MODELS_DIR / "gbm_all_league.pkl")
    gbm_t1_model.save(MODELS_DIR / "gbm_tier1.pkl")
    gbm_no_odds_model.save(MODELS_DIR / "gbm_no_odds.pkl")
    print(f"\nModels saved to {MODELS_DIR}/")

    # --- Save test predictions ---
    pred_df = test[["Date", "home_team", "away_team", "FTR", "season", "league",
                     "B365H", "B365D", "B365A"]].copy()
    for suffix, probs in [
        ("odds_b365", test_predictions["odds_b365"]),
        ("gbm_all", test_predictions["gbm_all"]),
        ("gbm_no_odds", test_predictions["gbm_no_odds"]),
    ]:
        pred_df[f"pred_h_{suffix}"] = probs[:, 0]
        pred_df[f"pred_d_{suffix}"] = probs[:, 1]
        pred_df[f"pred_a_{suffix}"] = probs[:, 2]

    # Tier 1 predictions (only for Tier 1 subset)
    tier1_pred_df = test_t1[["Date", "home_team", "away_team", "FTR", "season", "league",
                              "B365H", "B365D", "B365A"]].copy()
    tier1_pred_df["pred_h_gbm_t1"] = gbm_t1_probs[:, 0]
    tier1_pred_df["pred_d_gbm_t1"] = gbm_t1_probs[:, 1]
    tier1_pred_df["pred_a_gbm_t1"] = gbm_t1_probs[:, 2]

    pred_path = PREDICTIONS_DIR / "enhanced_predictions.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")
    print(f"  Shape: {pred_df.shape}")

    tier1_pred_path = PREDICTIONS_DIR / "tier1_predictions.parquet"
    tier1_pred_df.to_parquet(tier1_pred_path, index=False)
    print(f"Tier 1 predictions saved to {tier1_pred_path}")
    print(f"  Shape: {tier1_pred_df.shape}")


def cli():
    """CLI entry point with --config argument."""
    parser = argparse.ArgumentParser(description="Train match outcome models")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to model config YAML",
    )
    args = parser.parse_args()
    main(config_path=args.config)
