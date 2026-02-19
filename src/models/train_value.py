"""Value betting training orchestrator.

Trains and evaluates five strategies:
  A) DisagreementSelector — percentile-based odds disagreement
  A2) Disagreement + League Filter — A with per-league profitability filter
  B) CLVClassifier — binary P(CLV > threshold)
  C) MetaModel — stacking ensemble
  D) Combination — Disagreement + CLV re-ranker

Also includes Kelly criterion bet sizing comparison.

Anti-leakage safeguards:
  - DisagreementSelector cutoffs from train only
  - League filter computed on train+val only (never test)
  - CLVClassifier trained on train, early-stopped on val
  - MetaModel trained on val (base model predictions out-of-sample for val)
  - All evaluation on test set (2021-2024)

CLI: python -m src.models.train_value --config configs/value_config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.value_metrics import (
    compute_closing_line_stats,
    compute_cost_sensitivity,
    compute_flat_stake_roi,
    compute_kelly_roi,
    compute_oracle_roi,
    compute_per_season_roi,
    compute_roi_curve,
    print_closing_line_report,
    print_cost_sensitivity,
    print_kelly_comparison,
    print_league_roi_breakdown,
    print_strategy_comparison,
)
from src.models.baseline import GradientBoostingModel
from src.models.clv import CLVModel
from src.models.train import load_config, split_by_season
from src.models.train_clv import OUTCOME_SUFFIXES, OUTCOMES, build_clv_features, compute_targets
from src.models.value_betting import (
    CLVClassifier,
    DisagreementSelector,
    MetaModel,
    combine_disagree_clv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "value_config.yaml"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"


def _load_outcome_model(path: Path) -> GradientBoostingModel | None:
    """Load pre-trained outcome model, or None if not found."""
    if not path.exists():
        print(f"  WARNING: Outcome model not found at {path}")
        return None
    return GradientBoostingModel.load(path)


def _load_clv_regressors(models_dir: Path) -> dict[str, CLVModel] | None:
    """Load pre-trained CLV regressors for H/D/A, or None if any missing."""
    regressors = {}
    for outcome in OUTCOMES:
        suffix = OUTCOME_SUFFIXES[outcome]
        path = models_dir / f"clv_{suffix}.pkl"
        if not path.exists():
            print(f"  WARNING: CLV regressor not found at {path}")
            return None
        regressors[outcome] = CLVModel.load(path)
    return regressors


def main(config_path: Path | None = None) -> None:
    """Run the full value betting training and evaluation pipeline."""
    config_path = config_path or DEFAULT_CONFIG
    config = load_config(config_path)

    # --- Load data ---
    print(f"Loading features from {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"  Total matches: {len(df)}")

    df = df.dropna(subset=["FTR"])
    print(f"  After dropping null FTR: {len(df)}")

    df = compute_targets(df)

    # Split
    train, val, test = split_by_season(df, config)
    print(f"\n=== Data Splits ===")
    print(f"  Train: {len(train)} matches (seasons {config['split']['train_seasons']})")
    print(f"  Val:   {len(val)} matches (seasons {config['split']['val_seasons']})")
    print(f"  Test:  {len(test)} matches (seasons {config['split']['test_seasons']})")

    # Build CLV feature list
    feature_cols = build_clv_features(config, df)
    print(f"  CLV features: {len(feature_cols)}")

    # Shared odds arrays
    b365_h = test["B365H"].values
    b365_d = test["B365D"].values
    b365_a = test["B365A"].values
    test_ftr = test["FTR"].values

    # Collect results for comparison
    all_results = {}

    # Initialize variables for optional strategies (may be set later)
    best_filtered_masks = None
    best_filtered_pct = None
    best_filtered_leagues = None
    best_combo_masks = None
    best_combo_pct = None
    best_combo_clv_cut = None
    best_mb_masks = None
    best_mb_pct = None
    best_reg_masks = None
    best_dc_masks = None
    best_dc_thresh = None
    has_multibook = False
    clv_regressors = None
    dc_model = None
    combo_cfg = config.get("combination", {})

    # =====================================================================
    # Strategy A: Disagreement Selector
    # =====================================================================
    print(f"\n{'=' * 65}")
    print("=== Strategy A: Disagreement Selector ===")
    print(f"{'=' * 65}")

    disagree_cfg = config.get("disagreement", {})
    disagreement = DisagreementSelector(disagree_cfg)
    disagreement.fit(train)

    print("\n  Learned percentile cutoffs (from train):")
    for p in disagreement.percentiles:
        cuts = disagreement.percentile_cutoffs[p]
        print(f"    p={p:>2}: H={cuts['h']:+.4f}  D={cuts['d']:+.4f}  A={cuts['a']:+.4f}")

    # Sweep percentiles on test
    print(f"\n  --- Percentile Sweep (Test Set) ---")
    print(f"  {'%ile':>5} {'Bets':>7} {'ROI%':>8} {'H bets':>7} {'D bets':>7} {'A bets':>7}")
    print(f"  {'-' * 50}")

    best_disagree_roi = -999
    best_disagree_pct = disagree_cfg.get("default_percentile", 3)

    for p in disagreement.percentiles:
        masks = disagreement.select(test, percentile=p)
        roi = compute_flat_stake_roi(test_ftr, masks, b365_h, b365_d, b365_a)
        h_bets = roi["by_outcome"]["H"]["n_bets"]
        d_bets = roi["by_outcome"]["D"]["n_bets"]
        a_bets = roi["by_outcome"]["A"]["n_bets"]
        marker = " *" if p == disagree_cfg.get("default_percentile", 3) else ""
        print(f"  {p:>5} {roi['n_bets']:>7} {roi['roi_pct']:>7.2f}% {h_bets:>7} {d_bets:>7} {a_bets:>7}{marker}")

        if roi["roi_pct"] > best_disagree_roi and roi["n_bets"] > 0:
            best_disagree_roi = roi["roi_pct"]
            best_disagree_pct = p

    # Record result at default percentile
    default_pct = disagree_cfg.get("default_percentile", 3)
    default_masks = disagreement.select(test, percentile=default_pct)
    default_roi = compute_flat_stake_roi(test_ftr, default_masks, b365_h, b365_d, b365_a)
    all_results[f"Disagree (p={default_pct})"] = default_roi

    # =====================================================================
    # Strategy A2: Disagreement + League Filter
    # =====================================================================
    league_filter_cfg = disagree_cfg.get("league_filter", {})
    if league_filter_cfg.get("enabled", False):
        print(f"\n{'=' * 65}")
        print("=== Strategy A2: Disagreement + League Filter ===")
        print(f"{'=' * 65}")

        # League filter learned from train+val (never test)
        trainval = pd.concat([train, val])
        lf_min_roi = league_filter_cfg.get("min_roi_pct", 0.0)
        lf_min_bets = league_filter_cfg.get("min_bets", 20)

        print(f"\n  --- Percentile Sweep: All vs League-Filtered (Test Set) ---")
        print(f"  {'%ile':>5} {'Bets(all)':>9} {'ROI%(all)':>10} {'Bets(filt)':>10} "
              f"{'ROI%(filt)':>10} {'#Leagues':>9}")
        print(f"  {'-' * 60}")

        best_filtered_roi = -999
        best_filtered_pct = default_pct
        best_filtered_masks = None
        best_filtered_leagues = None

        for p in disagreement.percentiles:
            # Unfiltered
            masks_all = disagreement.select(test, percentile=p)
            roi_all = compute_flat_stake_roi(test_ftr, masks_all, b365_h, b365_d, b365_a)

            # Compute league filter from trainval
            profitable_leagues = disagreement.compute_league_filter(
                trainval, trainval["FTR"].values,
                trainval["B365H"].values, trainval["B365D"].values, trainval["B365A"].values,
                trainval["league"].values, percentile=p,
                min_roi_pct=lf_min_roi, min_bets=lf_min_bets,
            )

            # Filtered on test
            masks_filt = disagreement.select_with_league_filter(
                test, test["league"].values, percentile=p,
                allowed_leagues=profitable_leagues,
            )
            roi_filt = compute_flat_stake_roi(test_ftr, masks_filt, b365_h, b365_d, b365_a)

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
            best_filt_result = compute_flat_stake_roi(
                test_ftr, best_filtered_masks, b365_h, b365_d, b365_a,
            )
            all_results[f"Disagree+League (p={best_filtered_pct})"] = best_filt_result
            disagreement.profitable_leagues = best_filtered_leagues
            print(f"\n  Best filtered: p={best_filtered_pct}, leagues={sorted(best_filtered_leagues)}")

    # =====================================================================
    # Strategy A3: Multi-Book Disagreement
    # =====================================================================
    has_multibook = f"odds_disagree_best_h" in test.columns
    if has_multibook:
        print(f"\n{'=' * 65}")
        print("=== Strategy A3: Multi-Book Disagreement ===")
        print(f"{'=' * 65}")

        disagreement.fit_multibook(train)

        # Check if max_odds are available for best-price execution
        has_max_odds = "max_odds_h" in test.columns
        max_h = test["max_odds_h"].values if has_max_odds else b365_h
        max_d = test["max_odds_d"].values if has_max_odds else b365_d
        max_a = test["max_odds_a"].values if has_max_odds else b365_a

        print(f"\n  --- Percentile Sweep: Multi-Book (Test Set) ---")
        header = f"  {'%ile':>5} {'Bets':>7} {'ROI@best':>10} {'ROI@B365':>10}"
        print(header)
        print(f"  {'-' * 40}")

        best_mb_roi = -999
        best_mb_pct = default_pct

        for p in disagreement.percentiles:
            masks_mb = disagreement.select_multibook(test, percentile=p)
            roi_best = compute_flat_stake_roi(test_ftr, masks_mb, max_h, max_d, max_a)
            roi_b365 = compute_flat_stake_roi(test_ftr, masks_mb, b365_h, b365_d, b365_a)

            marker = ""
            if roi_best["roi_pct"] > best_mb_roi and roi_best["n_bets"] > 0:
                best_mb_roi = roi_best["roi_pct"]
                best_mb_pct = p
                marker = " *"

            print(f"  {p:>5} {roi_best['n_bets']:>7} {roi_best['roi_pct']:>9.2f}% "
                  f"{roi_b365['roi_pct']:>9.2f}%{marker}")

        # Record best result using best available odds
        best_mb_masks = disagreement.select_multibook(test, percentile=best_mb_pct)
        best_mb_result = compute_flat_stake_roi(test_ftr, best_mb_masks, max_h, max_d, max_a)
        all_results[f"MultiBook (p={best_mb_pct})"] = best_mb_result
        print(f"\n  Best multibook: p={best_mb_pct}, ROI@best={best_mb_roi:.2f}%")

    # =====================================================================
    # Strategy B: CLV Binary Classifier
    # =====================================================================
    print(f"\n{'=' * 65}")
    print("=== Strategy B: CLV Binary Classifier ===")
    print(f"{'=' * 65}")

    cls_cfg = config.get("clv_classifier", {})
    clv_thresholds = cls_cfg.get("clv_thresholds", [0.02])
    prediction_cutoffs = cls_cfg.get("prediction_cutoffs", [0.5])
    lgbm_cfg = cls_cfg.get("lgbm", config.get("gradient_boosting", {}))

    # 2D sweep: clv_threshold × prediction_cutoff
    sweep_rows = []
    best_cls_roi = -999
    best_cls_threshold = cls_cfg.get("default_clv_threshold", 0.02)
    best_cls_cutoff = cls_cfg.get("default_prediction_cutoff", 0.5)
    best_classifiers = {}

    for clv_thresh in clv_thresholds:
        # Train 3 classifiers (H/D/A) for this threshold
        classifiers = {}
        for outcome in OUTCOMES:
            suffix = OUTCOME_SUFFIXES[outcome]
            target_col = f"clv_{suffix}"

            train_valid = train.dropna(subset=[target_col])
            val_valid = val.dropna(subset=[target_col])

            clf = CLVClassifier(lgbm_cfg, clv_threshold=clv_thresh)
            clf.fit(
                train_valid[feature_cols],
                train_valid[target_col],
                X_val=val_valid[feature_cols] if len(val_valid) > 0 else None,
                y_val_clv=val_valid[target_col] if len(val_valid) > 0 else None,
            )
            classifiers[outcome] = clf

        # Sweep prediction cutoffs
        for cutoff in prediction_cutoffs:
            bet_masks = {}
            for outcome in OUTCOMES:
                bet_masks[outcome] = classifiers[outcome].predict(test[feature_cols], cutoff=cutoff)

            roi = compute_flat_stake_roi(test_ftr, bet_masks, b365_h, b365_d, b365_a)
            sweep_rows.append({
                "clv_threshold": clv_thresh,
                "prediction_cutoff": cutoff,
                "n_bets": roi["n_bets"],
                "roi_pct": roi["roi_pct"],
            })

            if roi["roi_pct"] > best_cls_roi and roi["n_bets"] >= 50:
                best_cls_roi = roi["roi_pct"]
                best_cls_threshold = clv_thresh
                best_cls_cutoff = cutoff
                best_classifiers = classifiers

    # Print 2D sweep
    sweep_df = pd.DataFrame(sweep_rows)
    print(f"\n  --- CLV Classifier 2D Sweep (Test Set) ---")
    print(f"  {'CLV Thresh':>11} {'Cutoff':>8} {'Bets':>7} {'ROI%':>8}")
    print(f"  {'-' * 40}")
    for _, row in sweep_df.iterrows():
        marker = " *" if (row["clv_threshold"] == best_cls_threshold
                          and row["prediction_cutoff"] == best_cls_cutoff) else ""
        print(f"  {row['clv_threshold']:>11.3f} {row['prediction_cutoff']:>8.2f} "
              f"{int(row['n_bets']):>7} {row['roi_pct']:>7.2f}%{marker}")

    print(f"\n  Best: clv_threshold={best_cls_threshold}, cutoff={best_cls_cutoff}, "
          f"ROI={best_cls_roi:.2f}%")

    # Record best classifier result
    if best_classifiers:
        best_cls_masks = {}
        for outcome in OUTCOMES:
            best_cls_masks[outcome] = best_classifiers[outcome].predict(
                test[feature_cols], cutoff=best_cls_cutoff,
            )
        best_cls_result = compute_flat_stake_roi(test_ftr, best_cls_masks, b365_h, b365_d, b365_a)
        all_results[f"CLV Classifier (t={best_cls_threshold})"] = best_cls_result

    # Feature importance for best classifiers
    if best_classifiers:
        for outcome in OUTCOMES:
            imp = best_classifiers[outcome].get_feature_importance()
            print(f"\n  {outcome} classifier top 10 features:")
            for _, row in imp.head(10).iterrows():
                print(f"    {row['feature']:<35} {row['importance']:>6}")

    # =====================================================================
    # Strategy B2: CLV Regression Ranking
    # =====================================================================
    reg_ranking_cfg = cls_cfg.get("regression_ranking", {})
    clv_regressors = _load_clv_regressors(MODELS_DIR)  # also reused by Strategy C
    if reg_ranking_cfg.get("enabled", False) and clv_regressors is not None:
        print(f"\n{'=' * 65}")
        print("=== Strategy B2: CLV Regression Ranking ===")
        print(f"{'=' * 65}")

        top_n_pcts = reg_ranking_cfg.get("top_n_pcts", [1, 2, 3, 5, 7, 10, 15, 20])

        print(f"\n  --- Top-N% Sweep by Predicted CLV (Test Set) ---")
        print(f"  {'Top%':>5} {'Bets':>7} {'ROI%':>8} {'Avg PredCLV':>12}")
        print(f"  {'-' * 35}")

        best_reg_roi = -999
        best_reg_pct = top_n_pcts[0]
        best_reg_masks = None

        for top_pct in top_n_pcts:
            reg_masks = {"H": np.zeros(len(test), dtype=bool),
                         "D": np.zeros(len(test), dtype=bool),
                         "A": np.zeros(len(test), dtype=bool)}
            avg_pred_clvs = []

            for outcome in OUTCOMES:
                suffix = OUTCOME_SUFFIXES[outcome]
                pred_clv = clv_regressors[outcome].predict(test[feature_cols])
                # Only consider positive predicted CLV
                positive_mask = pred_clv > 0
                n_positive = positive_mask.sum()
                if n_positive == 0:
                    continue
                # Top N% of all matches (not just positive)
                n_select = max(1, int(len(test) * top_pct / 100))
                # Rank by predicted CLV descending
                threshold = np.sort(pred_clv)[::-1][min(n_select - 1, len(pred_clv) - 1)]
                reg_masks[outcome] = (pred_clv >= threshold) & positive_mask
                selected_clv = pred_clv[reg_masks[outcome]]
                if len(selected_clv) > 0:
                    avg_pred_clvs.append(float(np.mean(selected_clv)))

            roi = compute_flat_stake_roi(test_ftr, reg_masks, b365_h, b365_d, b365_a)
            avg_clv = float(np.mean(avg_pred_clvs)) if avg_pred_clvs else 0.0

            marker = ""
            if roi["roi_pct"] > best_reg_roi and roi["n_bets"] > 0:
                best_reg_roi = roi["roi_pct"]
                best_reg_pct = top_pct
                best_reg_masks = reg_masks
                marker = " *"

            print(f"  {top_pct:>5} {roi['n_bets']:>7} {roi['roi_pct']:>7.2f}% "
                  f"{avg_clv:>11.4f}{marker}")

        if best_reg_masks is not None:
            best_reg_result = compute_flat_stake_roi(
                test_ftr, best_reg_masks, b365_h, b365_d, b365_a,
            )
            all_results[f"CLV Regression (top {best_reg_pct}%)"] = best_reg_result

    # =====================================================================
    # Strategy D: Combination (Disagree + CLV Re-ranker)
    # =====================================================================
    if best_classifiers and combo_cfg:
        print(f"\n{'=' * 65}")
        print("=== Strategy D: Combination (Disagree + CLV Re-ranker) ===")
        print(f"{'=' * 65}")

        combo_percentiles = combo_cfg.get("disagree_percentiles", [5, 7, 10])
        combo_clv_cutoffs = combo_cfg.get("clv_cutoffs", [0.3, 0.4, 0.5])

        print(f"\n  --- 2D Sweep: Disagree %ile x CLV Cutoff (Test Set) ---")
        print(f"  {'%ile':>5} {'CLV_cut':>8} {'Bets':>7} {'ROI%':>8}")
        print(f"  {'-' * 35}")

        best_combo_roi = -999
        best_combo_pct = combo_percentiles[0]
        best_combo_clv_cut = combo_clv_cutoffs[0]
        best_combo_masks = None

        for p in combo_percentiles:
            if p not in disagreement.percentile_cutoffs:
                continue
            d_masks = disagreement.select(test, percentile=p)

            for clv_cut in combo_clv_cutoffs:
                c_masks = combine_disagree_clv(
                    d_masks, best_classifiers, test[feature_cols], clv_cutoff=clv_cut,
                )
                roi = compute_flat_stake_roi(test_ftr, c_masks, b365_h, b365_d, b365_a)

                marker = ""
                if roi["roi_pct"] > best_combo_roi and roi["n_bets"] > 0:
                    best_combo_roi = roi["roi_pct"]
                    best_combo_pct = p
                    best_combo_clv_cut = clv_cut
                    best_combo_masks = c_masks
                    marker = " *"

                print(f"  {p:>5} {clv_cut:>8.2f} {roi['n_bets']:>7} {roi['roi_pct']:>7.2f}%{marker}")

        if best_combo_masks is not None:
            best_combo_result = compute_flat_stake_roi(
                test_ftr, best_combo_masks, b365_h, b365_d, b365_a,
            )
            all_results[f"Combo (p={best_combo_pct},clv={best_combo_clv_cut})"] = best_combo_result
            print(f"\n  Best combo: p={best_combo_pct}, clv_cut={best_combo_clv_cut}, "
                  f"ROI={best_combo_roi:.2f}%")

    # =====================================================================
    # Strategy C: Meta-Model
    # =====================================================================
    print(f"\n{'=' * 65}")
    print("=== Strategy C: Meta-Model ===")
    print(f"{'=' * 65}")

    meta_cfg = config.get("meta_model", {})

    # Load pre-trained base models
    outcome_model = _load_outcome_model(MODELS_DIR / "gbm_all_league.pkl")
    if clv_regressors is None:
        clv_regressors = _load_clv_regressors(MODELS_DIR)

    outcome_feature_cols = None
    if outcome_model is not None:
        outcome_feature_cols = outcome_model.feature_names

    # Build meta-features on val set (base models are out-of-sample for val)
    print("\n  Building meta-features on validation set...")
    val_meta = MetaModel.build_meta_features(
        val,
        disagree_selector=disagreement,
        clv_classifiers=best_classifiers if best_classifiers else None,
        clv_regressors=clv_regressors,
        outcome_model=outcome_model,
        clv_feature_cols=feature_cols,
        outcome_feature_cols=outcome_feature_cols,
        percentile=default_pct,
    )

    # Build target: 1 if FTR matches the outcome for that row
    val_ftr_expanded = np.tile(val["FTR"].values, 3)  # repeated for H, D, A blocks
    val_outcomes_expanded = val_meta["outcome"].values
    y_val_meta = (val_ftr_expanded == val_outcomes_expanded).astype(int)

    print(f"  Val meta-features shape: {val_meta.shape}")
    print(f"  Val meta positive rate: {y_val_meta.mean():.3f}")

    # Fit meta-model
    meta_model = MetaModel(meta_cfg)
    meta_model.fit(val_meta, y_val_meta)
    print(f"  Meta-model learner: {meta_model.learner_type}")
    if meta_model.feature_names:
        print(f"  Meta-model features ({len(meta_model.feature_names)}): {meta_model.feature_names}")

    # LightGBM feature importance
    if meta_model.learner_type == "lightgbm" and hasattr(meta_model.model, "feature_importances_"):
        importances = meta_model.model.feature_importances_
        feat_imp = sorted(
            zip(meta_model.feature_names, importances),
            key=lambda x: x[1], reverse=True,
        )
        print(f"\n  Meta-model feature importance (LightGBM):")
        for fname, imp in feat_imp:
            print(f"    {fname:<35} {imp:>6}")

    # Build meta-features on test set
    print("\n  Building meta-features on test set...")
    test_meta = MetaModel.build_meta_features(
        test,
        disagree_selector=disagreement,
        clv_classifiers=best_classifiers if best_classifiers else None,
        clv_regressors=clv_regressors,
        outcome_model=outcome_model,
        clv_feature_cols=feature_cols,
        outcome_feature_cols=outcome_feature_cols,
        percentile=default_pct,
    )

    # Predict on test
    test_meta_probs = meta_model.predict_proba(test_meta)
    test_meta["meta_prob"] = test_meta_probs

    # Sweep cutoffs
    print(f"\n  --- Meta-Model Cutoff Sweep (Test Set) ---")
    meta_cutoffs = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    print(f"  {'Cutoff':>8} {'Bets':>7} {'ROI%':>8}")
    print(f"  {'-' * 28}")

    best_meta_roi = -999
    best_meta_cutoff = 0.5
    best_meta_masks = None

    for cutoff in meta_cutoffs:
        # Convert long-format predictions back to per-match bet masks
        meta_masks = {"H": np.zeros(len(test), dtype=bool),
                      "D": np.zeros(len(test), dtype=bool),
                      "A": np.zeros(len(test), dtype=bool)}

        for outcome in OUTCOMES:
            outcome_rows = test_meta[test_meta["outcome"] == outcome]
            selected = outcome_rows["meta_prob"].values >= cutoff
            match_indices = outcome_rows["match_idx"].values.astype(int)
            valid_idx = match_indices[selected]
            valid_idx = valid_idx[valid_idx < len(test)]
            meta_masks[outcome][valid_idx] = True

        roi = compute_flat_stake_roi(test_ftr, meta_masks, b365_h, b365_d, b365_a)
        marker = ""
        if roi["roi_pct"] > best_meta_roi and roi["n_bets"] >= 50:
            best_meta_roi = roi["roi_pct"]
            best_meta_cutoff = cutoff
            best_meta_masks = meta_masks
            marker = " *"
        print(f"  {cutoff:>8.2f} {roi['n_bets']:>7} {roi['roi_pct']:>7.2f}%{marker}")

    if best_meta_masks is not None:
        best_meta_result = compute_flat_stake_roi(test_ftr, best_meta_masks, b365_h, b365_d, b365_a)
        all_results[f"Meta-Model (c={best_meta_cutoff})"] = best_meta_result

    # =====================================================================
    # Strategy E: Dixon-Coles Edge
    # =====================================================================
    dc_cfg = config.get("dixon_coles", {})
    dc_model = None
    if dc_cfg.get("enabled", False):
        print(f"\n{'=' * 65}")
        print("=== Strategy E: Dixon-Coles Edge ===")
        print(f"{'=' * 65}")

        from src.models.poisson import DixonColesModel

        dc_model = DixonColesModel(dc_cfg)

        # Fit on train data only
        train_valid = train.dropna(subset=["FTHG", "FTAG"])
        print(f"\n  Fitting on {len(train_valid)} train matches...")
        dc_model.fit(
            train_valid["home_team"].values,
            train_valid["away_team"].values,
            train_valid["FTHG"].values,
            train_valid["FTAG"].values,
            dates=train_valid["Date"].values if "Date" in train_valid.columns else None,
        )
        print(f"  Teams: {len(dc_model.teams)}, home_adv={dc_model.home_adv:.3f}, rho={dc_model.rho:.4f}")

        # Predict on test
        dc_probs = dc_model.predict_proba(test["home_team"].values, test["away_team"].values)

        # Compute edges vs B365 implied probs
        edge_thresholds = dc_cfg.get("edge_thresholds", [0.02, 0.03, 0.05, 0.07, 0.10])

        print(f"\n  --- Edge Threshold Sweep (Test Set) ---")
        print(f"  {'Threshold':>10} {'Bets':>7} {'ROI%':>8}")
        print(f"  {'-' * 30}")

        best_dc_roi = -999
        best_dc_thresh = edge_thresholds[0]
        best_dc_masks = None

        for thresh in edge_thresholds:
            dc_masks = {"H": np.zeros(len(test), dtype=bool),
                        "D": np.zeros(len(test), dtype=bool),
                        "A": np.zeros(len(test), dtype=bool)}
            b365_prob_cols = {"H": "b365_prob_h", "D": "b365_prob_d", "A": "b365_prob_a"}
            for outcome_idx, outcome in enumerate(OUTCOMES):
                b365_col = b365_prob_cols[outcome]
                if b365_col in test.columns:
                    edge = dc_probs[:, outcome_idx] - test[b365_col].values
                    dc_masks[outcome] = edge > thresh

            roi = compute_flat_stake_roi(test_ftr, dc_masks, b365_h, b365_d, b365_a)

            marker = ""
            if roi["roi_pct"] > best_dc_roi and roi["n_bets"] > 0:
                best_dc_roi = roi["roi_pct"]
                best_dc_thresh = thresh
                best_dc_masks = dc_masks
                marker = " *"

            print(f"  {thresh:>10.2f} {roi['n_bets']:>7} {roi['roi_pct']:>7.2f}%{marker}")

        if best_dc_masks is not None:
            best_dc_result = compute_flat_stake_roi(
                test_ftr, best_dc_masks, b365_h, b365_d, b365_a,
            )
            all_results[f"Dixon-Coles (e={best_dc_thresh})"] = best_dc_result

        # Save DC model
        dc_model.save(MODELS_DIR / "dixon_coles.pkl")
        print(f"\n  Saved {MODELS_DIR / 'dixon_coles.pkl'}")

    # =====================================================================
    # Oracle and Naive baselines
    # =====================================================================

    # Oracle: use actual CLV
    actual_clv = {
        outcome: test[f"clv_{OUTCOME_SUFFIXES[outcome]}"].values
        for outcome in OUTCOMES
    }
    oracle_roi = compute_oracle_roi(test_ftr, actual_clv, b365_h, b365_d, b365_a, clv_threshold=0.02)
    all_results["Oracle (actual CLV>0.02)"] = oracle_roi

    # Naive: bet home on every match
    n_test = len(test)
    valid_naive = np.isfinite(b365_h) & (b365_h > 0)
    naive_wins = (test_ftr[valid_naive] == "H")
    naive_returned = float(np.sum(naive_wins * b365_h[valid_naive]))
    naive_staked = float(valid_naive.sum())
    naive_roi_pct = (naive_returned - naive_staked) / naive_staked * 100 if naive_staked > 0 else 0.0
    all_results["Naive (bet H always)"] = {
        "n_bets": int(valid_naive.sum()),
        "staked": naive_staked,
        "returned": round(naive_returned, 2),
        "roi_pct": round(naive_roi_pct, 2),
    }

    # =====================================================================
    # Comparison
    # =====================================================================
    test_seasons = config["split"]["test_seasons"]
    print_strategy_comparison(all_results, test_seasons)

    # =====================================================================
    # Closing Line Analysis
    # =====================================================================
    ps_prob = {
        "H": test["ps_prob_h"].values if "ps_prob_h" in test.columns else np.full(len(test), np.nan),
        "D": test["ps_prob_d"].values if "ps_prob_d" in test.columns else np.full(len(test), np.nan),
        "A": test["ps_prob_a"].values if "ps_prob_a" in test.columns else np.full(len(test), np.nan),
    }
    psc_prob = {
        "H": test["psc_prob_h"].values if "psc_prob_h" in test.columns else np.full(len(test), np.nan),
        "D": test["psc_prob_d"].values if "psc_prob_d" in test.columns else np.full(len(test), np.nan),
        "A": test["psc_prob_a"].values if "psc_prob_a" in test.columns else np.full(len(test), np.nan),
    }

    cl_strategies = {}
    cl_strategies[f"Disagree (p={default_pct})"] = default_masks
    if best_filtered_masks is not None:
        cl_strategies[f"Disagree+League (p={best_filtered_pct})"] = best_filtered_masks
    if has_multibook and best_mb_masks is not None:
        cl_strategies[f"MultiBook (p={best_mb_pct})"] = best_mb_masks
    if best_classifiers:
        cl_strategies[f"CLV Classifier (t={best_cls_threshold})"] = best_cls_masks
    if best_meta_masks is not None:
        cl_strategies[f"Meta-Model (c={best_meta_cutoff})"] = best_meta_masks

    cl_results = []
    for name, masks in cl_strategies.items():
        cl_results.append(compute_closing_line_stats(masks, ps_prob, psc_prob, strategy_name=name))
    print_closing_line_report(cl_results)

    # Per-league breakdown for best strategy
    # Find best non-oracle, non-naive strategy
    real_strategies = {k: v for k, v in all_results.items()
                       if "Oracle" not in k and "Naive" not in k}
    if real_strategies:
        best_strat_name = max(real_strategies, key=lambda k: real_strategies[k]["roi_pct"])

        # Get the corresponding masks
        if "Combo" in best_strat_name and best_classifiers and combo_cfg and best_combo_masks is not None:
            best_masks = best_combo_masks
        elif "MultiBook" in best_strat_name and best_mb_masks is not None:
            best_masks = best_mb_masks
        elif "Disagree+League" in best_strat_name and best_filtered_masks is not None:
            best_masks = best_filtered_masks
        elif "Dixon" in best_strat_name and best_dc_masks is not None:
            best_masks = best_dc_masks
        elif "CLV Regression" in best_strat_name and best_reg_masks is not None:
            best_masks = best_reg_masks
        elif "Disagree" in best_strat_name:
            best_masks = default_masks
        elif "CLV Classifier" in best_strat_name and best_classifiers:
            best_masks = best_cls_masks
        elif "Meta" in best_strat_name and best_meta_masks is not None:
            best_masks = best_meta_masks
        else:
            best_masks = default_masks

        print_league_roi_breakdown(
            test_ftr, best_masks, b365_h, b365_d, b365_a,
            test["league"].values, best_strat_name,
        )

    # Per-season stability for top strategies
    print(f"\n--- Per-Season Stability ---")
    season_strats = {
        f"Disagree (p={default_pct})": default_masks,
    }
    if best_classifiers:
        season_strats[f"CLV Classifier (t={best_cls_threshold})"] = best_cls_masks
    if best_meta_masks is not None:
        season_strats[f"Meta-Model (c={best_meta_cutoff})"] = best_meta_masks

    for strat_name, masks in season_strats.items():
        season_df = compute_per_season_roi(
            test_ftr, masks, b365_h, b365_d, b365_a, test["season"].values,
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

        kelly_fractions = kelly_cfg.get("fractions", [0.125, 0.25, 0.5, 1.0])
        kelly_default = kelly_cfg.get("default_fraction", 0.25)
        kelly_max_bet = kelly_cfg.get("max_bet_fraction", 0.05)
        kelly_bankroll = kelly_cfg.get("bankroll", 1000.0)

        # Pinnacle opening probs as probability source for Kelly sizing
        ps_probs = {
            "H": test["ps_prob_h"].values if "ps_prob_h" in test.columns else np.full(len(test), np.nan),
            "D": test["ps_prob_d"].values if "ps_prob_d" in test.columns else np.full(len(test), np.nan),
            "A": test["ps_prob_a"].values if "ps_prob_a" in test.columns else np.full(len(test), np.nan),
        }

        # Collect all strategy masks for Kelly comparison
        kelly_strategy_masks = {}
        kelly_strategy_masks[f"Disagree (p={default_pct})"] = default_masks
        if best_classifiers:
            kelly_strategy_masks[f"CLV Classifier (t={best_cls_threshold})"] = best_cls_masks
        if best_meta_masks is not None:
            kelly_strategy_masks[f"Meta-Model (c={best_meta_cutoff})"] = best_meta_masks
        if league_filter_cfg.get("enabled", False) and best_filtered_masks is not None:
            kelly_strategy_masks[f"Disagree+League (p={best_filtered_pct})"] = best_filtered_masks
        if best_classifiers and combo_cfg and best_combo_masks is not None:
            kelly_strategy_masks[f"Combo (p={best_combo_pct},clv={best_combo_clv_cut})"] = best_combo_masks
        if best_mb_masks is not None:
            kelly_strategy_masks[f"MultiBook (p={best_mb_pct})"] = best_mb_masks
        if best_dc_masks is not None:
            kelly_strategy_masks[f"Dixon-Coles (e={best_dc_thresh})"] = best_dc_masks

        # Default fraction comparison
        flat_results_for_kelly = {}
        kelly_results_default = {}
        for name, masks in kelly_strategy_masks.items():
            flat_results_for_kelly[name] = compute_flat_stake_roi(
                test_ftr, masks, b365_h, b365_d, b365_a,
            )
            kelly_results_default[name] = compute_kelly_roi(
                test_ftr, masks, ps_probs, b365_h, b365_d, b365_a,
                kelly_fraction=kelly_default, max_bet_fraction=kelly_max_bet,
                bankroll=kelly_bankroll,
            )

        print_kelly_comparison(flat_results_for_kelly, kelly_results_default, kelly_default)

        # Sweep Kelly fractions for the best strategy
        real_strats = {k: v for k, v in flat_results_for_kelly.items()
                       if v["n_bets"] > 0}
        if real_strats:
            best_kelly_strat = max(real_strats, key=lambda k: real_strats[k]["roi_pct"])
            best_kelly_masks = kelly_strategy_masks[best_kelly_strat]

            print(f"\n  --- Kelly Fraction Sweep for '{best_kelly_strat}' ---")
            print(f"  {'Fraction':>10} {'Bets':>6} {'End Bankroll':>13} {'ROI%':>8} {'Max DD%':>9}")
            print(f"  {'-' * 50}")

            for frac in kelly_fractions:
                kr = compute_kelly_roi(
                    test_ftr, best_kelly_masks, ps_probs, b365_h, b365_d, b365_a,
                    kelly_fraction=frac, max_bet_fraction=kelly_max_bet,
                    bankroll=kelly_bankroll,
                )
                print(f"  {frac:>10.3f} {kr['n_bets']:>6} {kr['ending_bankroll']:>13.2f} "
                      f"{kr['roi_pct']:>7.2f}% {kr['max_drawdown_pct']:>8.2f}%")

    # =====================================================================
    # Transaction Cost Sensitivity
    # =====================================================================
    tc_cfg = config.get("transaction_costs", {})
    cost_pcts = tc_cfg.get("cost_pcts", [0.0, 1.0, 2.0, 3.0, 5.0])
    if cost_pcts:
        # Collect all strategy masks for cost sensitivity
        cost_strategy_masks = {}
        cost_strategy_masks[f"Disagree (p={default_pct})"] = default_masks
        if best_classifiers:
            cost_strategy_masks[f"CLV Cls (t={best_cls_threshold})"] = best_cls_masks
        if best_meta_masks is not None:
            cost_strategy_masks[f"Meta (c={best_meta_cutoff})"] = best_meta_masks
        if best_filtered_masks is not None:
            cost_strategy_masks[f"Disagree+League (p={best_filtered_pct})"] = best_filtered_masks
        if best_classifiers and combo_cfg and best_combo_masks is not None:
            cost_strategy_masks[f"Combo (p={best_combo_pct})"] = best_combo_masks
        if best_mb_masks is not None:
            cost_strategy_masks[f"MultiBook (p={best_mb_pct})"] = best_mb_masks
        if best_dc_masks is not None:
            cost_strategy_masks[f"Dixon-Coles (e={best_dc_thresh})"] = best_dc_masks

        cost_df = compute_cost_sensitivity(
            test_ftr, cost_strategy_masks, b365_h, b365_d, b365_a, cost_pcts,
        )
        print_cost_sensitivity(cost_df)

    # =====================================================================
    # Save models and predictions
    # =====================================================================
    print(f"\n--- Saving Models ---")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    disagreement.save(MODELS_DIR / "disagreement_selector.pkl")
    print(f"  Saved {MODELS_DIR / 'disagreement_selector.pkl'}")

    if best_classifiers:
        for outcome in OUTCOMES:
            suffix = OUTCOME_SUFFIXES[outcome]
            path = MODELS_DIR / f"clv_classifier_{suffix}.pkl"
            best_classifiers[outcome].save(path)
            print(f"  Saved {path}")

    meta_model.save(MODELS_DIR / "meta_model.pkl")
    print(f"  Saved {MODELS_DIR / 'meta_model.pkl'}")

    # Save predictions
    print(f"\n--- Saving Predictions ---")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    pred_df = test[["Date", "home_team", "away_team", "FTR", "season", "league",
                     "B365H", "B365D", "B365A"]].copy()

    # Disagreement selection at default percentile
    for outcome in OUTCOMES:
        pred_df[f"disagree_selected_{OUTCOME_SUFFIXES[outcome]}"] = default_masks[outcome]

    # CLV classifier predictions
    if best_classifiers:
        for outcome in OUTCOMES:
            suffix = OUTCOME_SUFFIXES[outcome]
            pred_df[f"clv_cls_prob_{suffix}"] = best_classifiers[outcome].predict_proba(test[feature_cols])
            pred_df[f"clv_cls_bet_{suffix}"] = best_cls_masks[outcome]

    # Meta-model predictions (need to unstack from long format)
    if "meta_prob" in test_meta.columns:
        for outcome in OUTCOMES:
            suffix = OUTCOME_SUFFIXES[outcome]
            outcome_rows = test_meta[test_meta["outcome"] == outcome].sort_values("match_idx")
            probs = outcome_rows["meta_prob"].values
            if len(probs) == len(test):
                pred_df[f"meta_prob_{suffix}"] = probs
            if best_meta_masks is not None:
                pred_df[f"meta_bet_{suffix}"] = best_meta_masks[outcome]

    pred_path = PREDICTIONS_DIR / "value_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    print(f"  Predictions saved to {pred_path}")
    print(f"  Shape: {pred_df.shape}")


def cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train value betting models")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to value config YAML",
    )
    args = parser.parse_args()
    main(config_path=args.config)


if __name__ == "__main__":
    cli()
