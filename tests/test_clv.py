"""Unit tests for CLV prediction model, metrics, and training pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.clv_metrics import (
    compute_clv_bet_roi,
    compute_clv_correlation,
    compute_direction_accuracy,
    compute_mae,
    compute_profitable_clv,
    compute_value_bet_roi,
)
from src.models.clv import CLVModel
from src.models.train_clv import build_clv_features, compute_targets


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clv_sample_df():
    """Synthetic DataFrame with columns needed for CLV training."""
    rng = np.random.RandomState(42)
    n = 300
    seasons = np.repeat([2016, 2017, 2018, 2019, 2020, 2021], 50)

    df = pd.DataFrame({
        "season": seasons,
        "Date": pd.date_range("2016-01-01", periods=n, freq="D"),
        "FTR": rng.choice(["H", "D", "A"], size=n, p=[0.44, 0.27, 0.29]),
        "home_team": "TeamA",
        "away_team": "TeamB",
        "league": rng.choice(["EPL", "La_Liga", "Serie_A"], n),
        # B365 odds
        "B365H": rng.uniform(1.3, 4.0, n),
        "B365D": rng.uniform(2.5, 5.0, n),
        "B365A": rng.uniform(1.5, 8.0, n),
        # B365 implied probs
        "b365_prob_h": rng.uniform(0.3, 0.7, n),
        "b365_prob_d": rng.uniform(0.15, 0.35, n),
        "b365_prob_a": rng.uniform(0.1, 0.4, n),
        # Pinnacle opening
        "ps_prob_h": rng.uniform(0.3, 0.7, n),
        "ps_prob_d": rng.uniform(0.15, 0.35, n),
        "ps_prob_a": rng.uniform(0.1, 0.4, n),
        # Pinnacle closing
        "psc_prob_h": rng.uniform(0.3, 0.7, n),
        "psc_prob_d": rng.uniform(0.15, 0.35, n),
        "psc_prob_a": rng.uniform(0.1, 0.4, n),
        # Non-odds features
        "elo_diff": rng.randn(n) * 100,
        "h_elo_expected": rng.rand(n),
        "form_diff_5": rng.randn(n),
        "form_diff_10": rng.randn(n),
        "goals_diff_5": rng.randn(n),
        "goals_diff_10": rng.randn(n),
        "sot_diff_5": rng.randn(n),
        "sot_diff_10": rng.randn(n),
        "rest_diff": rng.randint(-5, 6, n).astype(float),
        "h_is_early_season": rng.choice([0, 1], n),
        "a_is_early_season": rng.choice([0, 1], n),
        "h_rolling5_ppg": rng.uniform(0.5, 2.5, n),
        "a_rolling5_ppg": rng.uniform(0.5, 2.5, n),
        "h_rolling5_goals_for": rng.uniform(0.5, 3.0, n),
        "a_rolling5_goals_for": rng.uniform(0.5, 3.0, n),
        "h_rolling5_goals_against": rng.uniform(0.3, 2.5, n),
        "a_rolling5_goals_against": rng.uniform(0.3, 2.5, n),
        "h_rolling5_win_rate": rng.uniform(0, 1, n),
        "a_rolling5_win_rate": rng.uniform(0, 1, n),
        "h_rolling5_clean_sheet_rate": rng.uniform(0, 1, n),
        "a_rolling5_clean_sheet_rate": rng.uniform(0, 1, n),
    })
    return df


@pytest.fixture
def clv_config():
    return {
        "n_estimators": 50,
        "learning_rate": 0.1,
        "max_depth": 3,
        "num_leaves": 15,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
    }


@pytest.fixture
def clv_features():
    return [
        "b365_prob_h", "b365_prob_d", "b365_prob_a",
        "ps_prob_h", "ps_prob_d", "ps_prob_a",
        "elo_diff", "h_elo_expected",
        "form_diff_5", "form_diff_10",
    ]


@pytest.fixture
def split_config():
    return {
        "split": {
            "train_seasons": [2016, 2017, 2018],
            "val_seasons": [2019],
            "test_seasons": [2020, 2021],
        }
    }


# ---------------------------------------------------------------------------
# CLVModel tests
# ---------------------------------------------------------------------------

class TestCLVModel:
    def test_fit_predict_shape(self, clv_sample_df, clv_config, clv_features):
        """Predictions should be 1-D float array."""
        df = compute_targets(clv_sample_df)
        model = CLVModel(clv_config)
        X = df[clv_features]
        y = df["clv_h"]
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert preds.dtype == np.float64

    def test_predictions_are_continuous(self, clv_sample_df, clv_config, clv_features):
        """Regression output should have more than a few unique values."""
        df = compute_targets(clv_sample_df)
        model = CLVModel(clv_config)
        model.fit(df[clv_features], df["clv_h"])
        preds = model.predict(df[clv_features])
        assert len(np.unique(preds)) > 10

    def test_with_early_stopping(self, clv_sample_df, clv_config, clv_features, split_config):
        """Early stopping should work with validation set."""
        from src.models.train import split_by_season
        df = compute_targets(clv_sample_df)
        train, val, test = split_by_season(df, split_config)
        model = CLVModel(clv_config)
        model.fit(
            train[clv_features], train["clv_h"],
            X_val=val[clv_features], y_val=val["clv_h"],
        )
        preds = model.predict(test[clv_features])
        assert preds.shape == (len(test),)
        assert not np.isnan(preds).any()

    def test_feature_importance(self, clv_sample_df, clv_config, clv_features):
        df = compute_targets(clv_sample_df)
        model = CLVModel(clv_config)
        model.fit(df[clv_features], df["clv_h"])
        imp = model.get_feature_importance()
        assert len(imp) == len(clv_features)
        assert "feature" in imp.columns
        assert "importance" in imp.columns

    def test_save_load_roundtrip(self, clv_sample_df, clv_config, clv_features):
        df = compute_targets(clv_sample_df)
        model = CLVModel(clv_config)
        model.fit(df[clv_features], df["clv_h"])
        original_preds = model.predict(df[clv_features])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "clv_model.pkl"
            model.save(path)
            loaded = CLVModel.load(path)
            loaded_preds = loaded.predict(df[clv_features])

        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_handles_nan_features(self, clv_config):
        """NaN features should be imputed with median."""
        X = pd.DataFrame({
            "f1": [1.0, np.nan, 3.0, 4.0, 5.0] * 20,
            "f2": [np.nan, 2.0, 3.0, 4.0, 5.0] * 20,
        })
        y = np.random.RandomState(42).randn(100) * 0.05
        model = CLVModel(clv_config)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)
        assert not np.isnan(preds).any()

    def test_save_contains_model_type(self, clv_sample_df, clv_config, clv_features):
        """Saved model should have model_type='clv_regression'."""
        import joblib
        df = compute_targets(clv_sample_df)
        model = CLVModel(clv_config)
        model.fit(df[clv_features], df["clv_h"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "clv.pkl"
            model.save(path)
            data = joblib.load(path)

        assert data["model_type"] == "clv_regression"
        assert data["feature_names"] == clv_features


# ---------------------------------------------------------------------------
# Direction accuracy tests
# ---------------------------------------------------------------------------

class TestDirectionAccuracy:
    def test_perfect_prediction(self):
        y_true = np.array([0.05, -0.03, 0.01, -0.02])
        y_pred = np.array([0.10, -0.01, 0.02, -0.05])
        assert compute_direction_accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_inverse_prediction(self):
        y_true = np.array([0.05, -0.03, 0.01, -0.02])
        y_pred = np.array([-0.10, 0.01, -0.02, 0.05])
        assert compute_direction_accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_half_correct(self):
        y_true = np.array([0.05, -0.03, 0.01, -0.02])
        y_pred = np.array([0.10, 0.01, 0.02, 0.05])  # 2/4 correct
        assert compute_direction_accuracy(y_true, y_pred) == pytest.approx(0.5)

    def test_excludes_zero_true(self):
        """Zero-movement matches should be excluded."""
        y_true = np.array([0.0, 0.05, -0.03])
        y_pred = np.array([0.10, 0.01, -0.01])
        # Only 2 non-zero entries, both correct
        assert compute_direction_accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_all_zero_returns_zero(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([0.1, -0.1, 0.0])
        assert compute_direction_accuracy(y_true, y_pred) == 0.0


# ---------------------------------------------------------------------------
# CLV correlation tests
# ---------------------------------------------------------------------------

class TestCLVCorrelation:
    def test_perfect_correlation(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        corr = compute_clv_correlation(y_true, y_pred)
        assert corr["pearson"] == pytest.approx(1.0, abs=1e-6)
        assert corr["spearman"] == pytest.approx(1.0, abs=1e-6)

    def test_inverse_correlation(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        corr = compute_clv_correlation(y_true, y_pred)
        assert corr["pearson"] == pytest.approx(-1.0, abs=1e-6)
        assert corr["spearman"] == pytest.approx(-1.0, abs=1e-6)

    def test_zero_correlation(self):
        """Uncorrelated data should have near-zero correlation."""
        rng = np.random.RandomState(42)
        y_true = rng.randn(1000)
        y_pred = rng.randn(1000)
        corr = compute_clv_correlation(y_true, y_pred)
        assert abs(corr["pearson"]) < 0.1
        assert abs(corr["spearman"]) < 0.1

    def test_handles_nan(self):
        y_true = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        corr = compute_clv_correlation(y_true, y_pred)
        # Should compute on valid pairs only (indices 0, 3, 4)
        assert corr["pearson"] == pytest.approx(1.0, abs=1e-6)

    def test_too_few_values(self):
        y_true = np.array([1.0, np.nan])
        y_pred = np.array([np.nan, 2.0])
        corr = compute_clv_correlation(y_true, y_pred)
        assert corr["pearson"] == 0.0
        assert corr["spearman"] == 0.0


# ---------------------------------------------------------------------------
# MAE tests
# ---------------------------------------------------------------------------

class TestMAE:
    def test_perfect(self):
        y_true = np.array([0.05, -0.03, 0.01])
        y_pred = np.array([0.05, -0.03, 0.01])
        assert compute_mae(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)

    def test_known_mae(self):
        y_true = np.array([0.10, 0.00, -0.10])
        y_pred = np.array([0.05, 0.05, -0.05])
        # Errors: 0.05, 0.05, 0.05 → MAE = 0.05
        assert compute_mae(y_true, y_pred) == pytest.approx(0.05)

    def test_nonnegative(self):
        rng = np.random.RandomState(42)
        assert compute_mae(rng.randn(100), rng.randn(100)) >= 0


# ---------------------------------------------------------------------------
# Profitable CLV tests
# ---------------------------------------------------------------------------

class TestProfitableCLV:
    def test_no_bets_below_threshold(self):
        y_pred_clv = np.array([0.01, 0.005, -0.02])
        y_true_clv = np.array([0.02, 0.01, -0.01])
        ftr = np.array(["H", "D", "A"])
        odds = np.array([2.0, 3.0, 4.0])
        result = compute_profitable_clv(y_true_clv, y_pred_clv, ftr, odds, threshold=0.02)
        assert result["n_bets"] == 0

    def test_bets_above_threshold(self):
        y_pred_clv = np.array([0.05, 0.03, 0.01])
        y_true_clv = np.array([0.04, 0.02, 0.01])
        ftr = np.array(["H", "D", "A"])
        odds = np.array([2.0, 3.0, 4.0])
        result = compute_profitable_clv(y_true_clv, y_pred_clv, ftr, odds, threshold=0.02)
        assert result["n_bets"] == 2  # indices 0 and 1

    def test_handles_nan_odds(self):
        y_pred_clv = np.array([0.05, 0.05])
        y_true_clv = np.array([0.03, 0.03])
        ftr = np.array(["H", "D"])
        odds = np.array([2.0, np.nan])
        result = compute_profitable_clv(y_true_clv, y_pred_clv, ftr, odds, threshold=0.02)
        assert result["n_bets"] == 1


# ---------------------------------------------------------------------------
# CLV bet ROI tests
# ---------------------------------------------------------------------------

class TestCLVBetROI:
    def test_winning_bets(self):
        """All bets win — ROI should be positive."""
        ftr = np.array(["H", "H", "H"])
        pred_clv = {
            "H": np.array([0.05, 0.05, 0.05]),
            "D": np.array([-0.01, -0.01, -0.01]),
            "A": np.array([-0.01, -0.01, -0.01]),
        }
        odds_h = np.array([2.5, 2.5, 2.5])
        odds_d = np.array([3.0, 3.0, 3.0])
        odds_a = np.array([4.0, 4.0, 4.0])
        result = compute_clv_bet_roi(ftr, pred_clv, odds_h, odds_d, odds_a, clv_threshold=0.02)
        assert result["n_bets"] == 3  # only H bets triggered
        assert result["roi_pct"] > 0

    def test_no_bets(self):
        """No predicted CLV above threshold — no bets placed."""
        ftr = np.array(["H", "D", "A"])
        pred_clv = {
            "H": np.array([0.01, 0.01, 0.01]),
            "D": np.array([0.01, 0.01, 0.01]),
            "A": np.array([0.01, 0.01, 0.01]),
        }
        result = compute_clv_bet_roi(
            ftr, pred_clv,
            np.array([2.0, 3.0, 4.0]),
            np.array([3.0, 3.0, 3.0]),
            np.array([4.0, 4.0, 4.0]),
            clv_threshold=0.02,
        )
        assert result["n_bets"] == 0
        assert result["roi_pct"] == 0.0

    def test_per_outcome_breakdown(self):
        """Each outcome should have independent bet counting."""
        ftr = np.array(["H", "D", "A", "H"])
        pred_clv = {
            "H": np.array([0.05, 0.00, 0.00, 0.05]),
            "D": np.array([0.00, 0.05, 0.00, 0.00]),
            "A": np.array([0.00, 0.00, 0.05, 0.00]),
        }
        result = compute_clv_bet_roi(
            ftr, pred_clv,
            np.array([2.0, 3.0, 4.0, 2.0]),
            np.array([3.0, 2.5, 3.0, 3.0]),
            np.array([4.0, 4.0, 2.0, 4.0]),
            clv_threshold=0.02,
        )
        assert result["by_outcome"]["H"]["n_bets"] == 2
        assert result["by_outcome"]["D"]["n_bets"] == 1
        assert result["by_outcome"]["A"]["n_bets"] == 1


# ---------------------------------------------------------------------------
# Value bet ROI (multi-signal) tests
# ---------------------------------------------------------------------------

class TestValueBetROI:
    def test_both_signals_agree(self):
        """When both CLV and edge signal agree, bet should be placed."""
        ftr = np.array(["H", "H"])
        outcome_probs = np.array([
            [0.60, 0.20, 0.20],  # high home prob
            [0.60, 0.20, 0.20],
        ])
        pred_clv = {
            "H": np.array([0.05, 0.05]),  # both above threshold
            "D": np.array([-0.01, -0.01]),
            "A": np.array([-0.01, -0.01]),
        }
        # B365 odds imply ~40% for H (1/2.5=0.4), so edge = 0.60-0.40 = 0.20 > 0.03
        result = compute_value_bet_roi(
            ftr, outcome_probs,
            np.array([2.5, 2.5]), np.array([3.5, 3.5]), np.array([4.0, 4.0]),
            pred_clv, clv_threshold=0.02, edge_threshold=0.03, min_signals=2,
        )
        assert result["n_bets"] == 2
        assert result["roi_pct"] > 0  # both win at 2.5 odds

    def test_only_one_signal_no_bet(self):
        """When only CLV signal fires but not edge, no bet with min_signals=2."""
        ftr = np.array(["H"])
        outcome_probs = np.array([
            [0.35, 0.35, 0.30],  # low home prob, no edge
        ])
        pred_clv = {
            "H": np.array([0.05]),  # CLV fires
            "D": np.array([-0.01]),
            "A": np.array([-0.01]),
        }
        # B365 implied = 1/2.5 = 0.4, edge = 0.35-0.40 = -0.05 < 0.03 → no edge signal
        result = compute_value_bet_roi(
            ftr, outcome_probs,
            np.array([2.5]), np.array([3.5]), np.array([4.0]),
            pred_clv, clv_threshold=0.02, edge_threshold=0.03, min_signals=2,
        )
        assert result["n_bets"] == 0

    def test_min_signals_one(self):
        """With min_signals=1, single signal should trigger bet."""
        ftr = np.array(["H"])
        outcome_probs = np.array([[0.35, 0.35, 0.30]])
        pred_clv = {
            "H": np.array([0.05]),
            "D": np.array([-0.01]),
            "A": np.array([-0.01]),
        }
        result = compute_value_bet_roi(
            ftr, outcome_probs,
            np.array([2.5]), np.array([3.5]), np.array([4.0]),
            pred_clv, clv_threshold=0.02, edge_threshold=0.03, min_signals=1,
        )
        assert result["n_bets"] >= 1  # CLV signal for H fires


# ---------------------------------------------------------------------------
# Target computation tests
# ---------------------------------------------------------------------------

class TestComputeTargets:
    def test_clv_values(self):
        """CLV should be closing - opening Pinnacle prob."""
        df = pd.DataFrame({
            "psc_prob_h": [0.50, 0.45],
            "psc_prob_d": [0.30, 0.25],
            "psc_prob_a": [0.20, 0.30],
            "ps_prob_h": [0.48, 0.47],
            "ps_prob_d": [0.28, 0.26],
            "ps_prob_a": [0.24, 0.27],
            "b365_prob_h": [0.49, 0.46],
            "b365_prob_d": [0.29, 0.24],
            "b365_prob_a": [0.22, 0.30],
        })
        result = compute_targets(df)
        np.testing.assert_allclose(result["clv_h"], [0.02, -0.02])
        np.testing.assert_allclose(result["clv_d"], [0.02, -0.01])
        np.testing.assert_allclose(result["clv_a"], [-0.04, 0.03])

    def test_odds_disagreement(self):
        """Odds disagreement should be B365 - Pinnacle opening."""
        df = pd.DataFrame({
            "psc_prob_h": [0.50],
            "psc_prob_d": [0.30],
            "psc_prob_a": [0.20],
            "ps_prob_h": [0.48],
            "ps_prob_d": [0.28],
            "ps_prob_a": [0.24],
            "b365_prob_h": [0.50],
            "b365_prob_d": [0.30],
            "b365_prob_a": [0.20],
        })
        result = compute_targets(df)
        assert result["odds_disagree_h"].iloc[0] == pytest.approx(0.02)
        assert result["odds_disagree_d"].iloc[0] == pytest.approx(0.02)
        assert result["odds_disagree_a"].iloc[0] == pytest.approx(-0.04)

    def test_handles_nan_odds(self):
        """NaN in odds should propagate to NaN targets."""
        df = pd.DataFrame({
            "psc_prob_h": [np.nan],
            "psc_prob_d": [0.30],
            "psc_prob_a": [0.20],
            "ps_prob_h": [0.48],
            "ps_prob_d": [0.28],
            "ps_prob_a": [0.24],
            "b365_prob_h": [0.50],
            "b365_prob_d": [0.30],
            "b365_prob_a": [0.20],
        })
        result = compute_targets(df)
        assert np.isnan(result["clv_h"].iloc[0])
        assert not np.isnan(result["clv_d"].iloc[0])

    def test_does_not_modify_input(self):
        """compute_targets should return a copy, not modify input."""
        df = pd.DataFrame({
            "psc_prob_h": [0.50], "psc_prob_d": [0.30], "psc_prob_a": [0.20],
            "ps_prob_h": [0.48], "ps_prob_d": [0.28], "ps_prob_a": [0.24],
            "b365_prob_h": [0.50], "b365_prob_d": [0.30], "b365_prob_a": [0.20],
        })
        original_cols = set(df.columns)
        compute_targets(df)
        assert set(df.columns) == original_cols


# ---------------------------------------------------------------------------
# Build CLV features tests
# ---------------------------------------------------------------------------

class TestBuildCLVFeatures:
    def test_resolves_available_features(self):
        config = {
            "features": {
                "opening_odds": ["b365_prob_h", "ps_prob_h"],
                "odds_disagreement": ["odds_disagree_h"],
                "non_odds": ["elo_diff"],
                "venue": ["h_venue_rolling3_ppg"],
                "rolling_extended": ["h_rolling10_ppg"],
            }
        }
        df = pd.DataFrame({
            "b365_prob_h": [1], "ps_prob_h": [1],
            "odds_disagree_h": [1], "elo_diff": [1],
            "h_venue_rolling3_ppg": [1], "h_rolling10_ppg": [1],
        })
        features = build_clv_features(config, df)
        assert len(features) == 6
        assert "b365_prob_h" in features
        assert "odds_disagree_h" in features

    def test_filters_missing_columns(self):
        config = {
            "features": {
                "opening_odds": ["b365_prob_h", "MISSING"],
                "odds_disagreement": [],
                "non_odds": ["elo_diff"],
                "venue": [],
                "rolling_extended": [],
            }
        }
        df = pd.DataFrame({"b365_prob_h": [1], "elo_diff": [1]})
        features = build_clv_features(config, df)
        assert "MISSING" not in features
        assert "b365_prob_h" in features
        assert "elo_diff" in features


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestCLVIntegration:
    def test_full_pipeline_synthetic(self, clv_sample_df, clv_config, clv_features, split_config):
        """End-to-end: compute targets, split, train, predict, evaluate."""
        from src.models.train import split_by_season

        df = compute_targets(clv_sample_df)
        train, val, test = split_by_season(df, split_config)

        # Train CLV model for home outcome
        model = CLVModel(clv_config)
        train_valid = train.dropna(subset=["clv_h"])
        val_valid = val.dropna(subset=["clv_h"])
        model.fit(
            train_valid[clv_features], train_valid["clv_h"],
            X_val=val_valid[clv_features], y_val=val_valid["clv_h"],
        )

        # Predict on test
        test_valid = test.dropna(subset=["clv_h"])
        preds = model.predict(test_valid[clv_features])
        assert preds.shape == (len(test_valid),)

        # Metrics
        dir_acc = compute_direction_accuracy(test_valid["clv_h"].values, preds)
        assert 0 <= dir_acc <= 1

        corr = compute_clv_correlation(test_valid["clv_h"].values, preds)
        assert -1 <= corr["pearson"] <= 1

        mae = compute_mae(test_valid["clv_h"].values, preds)
        assert mae >= 0

    def test_clv_bet_roi_integration(self, clv_sample_df, clv_config, clv_features, split_config):
        """Train models for all outcomes, then compute CLV bet ROI."""
        from src.models.train import split_by_season

        df = compute_targets(clv_sample_df)
        train, val, test = split_by_season(df, split_config)

        pred_clv = {}
        for outcome, suffix in [("H", "h"), ("D", "d"), ("A", "a")]:
            target = f"clv_{suffix}"
            train_v = train.dropna(subset=[target])
            model = CLVModel(clv_config)
            model.fit(train_v[clv_features], train_v[target])
            pred_clv[outcome] = model.predict(test[clv_features])

        roi = compute_clv_bet_roi(
            test["FTR"].values, pred_clv,
            test["B365H"].values, test["B365D"].values, test["B365A"].values,
            clv_threshold=0.02,
        )
        assert "n_bets" in roi
        assert "roi_pct" in roi
        assert "by_outcome" in roi
        for outcome in ["H", "D", "A"]:
            assert outcome in roi["by_outcome"]
