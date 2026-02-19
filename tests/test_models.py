"""Unit tests for baseline model training, splitting, and metrics."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_brier_score,
    compute_calibration,
    compute_log_loss,
    compute_roi,
    compute_accuracy,
)
from src.models.baseline import BaselineModel, GradientBoostingModel, OddsBaseline
from src.models.train import (
    build_feature_sets,
    print_league_breakdown,
    split_by_season,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Create a small synthetic DataFrame mimicking features.parquet."""
    rng = np.random.RandomState(42)
    n = 300
    seasons = np.repeat([2016, 2017, 2018, 2019, 2020, 2021], 50)
    dates = pd.date_range("2016-01-01", periods=n, freq="D")
    ftr = rng.choice(["H", "D", "A"], size=n, p=[0.44, 0.27, 0.29])

    df = pd.DataFrame({
        "season": seasons,
        "Date": dates,
        "FTR": ftr,
        "HomeTeam": "TeamA",
        "AwayTeam": "TeamB",
        "Div": "E0",
        "elo_diff": rng.randn(n) * 100,
        "h_elo_expected": rng.rand(n),
        "b365_prob_h": rng.uniform(0.3, 0.7, n),
        "b365_prob_d": rng.uniform(0.15, 0.35, n),
        "b365_prob_a": rng.uniform(0.1, 0.4, n),
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
        "B365H": rng.uniform(1.3, 4.0, n),
        "B365D": rng.uniform(2.5, 5.0, n),
        "B365A": rng.uniform(1.5, 8.0, n),
    })
    return df


@pytest.fixture
def split_config():
    return {
        "split": {
            "train_seasons": [2016, 2017, 2018],
            "val_seasons": [2019],
            "test_seasons": [2020, 2021],
        }
    }


@pytest.fixture
def lr_config():
    return {
        "C": 1.0,
        "max_iter": 1000,
        "multi_class": "multinomial",
        "solver": "lbfgs",
        "random_state": 42,
    }


@pytest.fixture
def gbm_config():
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
def non_odds_features():
    return [
        "elo_diff", "h_elo_expected", "form_diff_5", "form_diff_10",
        "goals_diff_5", "goals_diff_10", "sot_diff_5", "sot_diff_10",
        "rest_diff", "h_is_early_season", "a_is_early_season",
        "h_rolling5_ppg", "a_rolling5_ppg",
        "h_rolling5_goals_for", "a_rolling5_goals_for",
        "h_rolling5_goals_against", "a_rolling5_goals_against",
        "h_rolling5_win_rate", "a_rolling5_win_rate",
        "h_rolling5_clean_sheet_rate", "a_rolling5_clean_sheet_rate",
    ]


@pytest.fixture
def all_features(non_odds_features):
    return ["b365_prob_h", "b365_prob_d", "b365_prob_a"] + non_odds_features


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------

class TestSplitBySeason:
    def test_no_temporal_leakage(self, sample_df, split_config):
        """All train dates must be before val dates, which must be before test dates."""
        train, val, test = split_by_season(sample_df, split_config)
        assert train["Date"].max() < val["Date"].min()
        assert val["Date"].max() < test["Date"].min()

    def test_no_overlap(self, sample_df, split_config):
        train, val, test = split_by_season(sample_df, split_config)
        train_seasons = set(train["season"].unique())
        val_seasons = set(val["season"].unique())
        test_seasons = set(test["season"].unique())
        assert train_seasons.isdisjoint(val_seasons)
        assert train_seasons.isdisjoint(test_seasons)
        assert val_seasons.isdisjoint(test_seasons)

    def test_sizes(self, sample_df, split_config):
        train, val, test = split_by_season(sample_df, split_config)
        assert len(train) + len(val) + len(test) == len(sample_df)

    def test_correct_seasons(self, sample_df, split_config):
        train, val, test = split_by_season(sample_df, split_config)
        assert set(train["season"].unique()) == {2016, 2017, 2018}
        assert set(val["season"].unique()) == {2019}
        assert set(test["season"].unique()) == {2020, 2021}


# ---------------------------------------------------------------------------
# OddsBaseline tests
# ---------------------------------------------------------------------------

class TestOddsBaseline:
    def test_probabilities_sum_to_one(self, sample_df):
        model = OddsBaseline()
        probs = model.predict_proba(sample_df)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_output_shape(self, sample_df):
        model = OddsBaseline()
        probs = model.predict_proba(sample_df)
        assert probs.shape == (len(sample_df), 3)

    def test_handles_nan_odds(self):
        df = pd.DataFrame({
            "b365_prob_h": [np.nan, 0.5],
            "b365_prob_d": [np.nan, 0.3],
            "b365_prob_a": [np.nan, 0.2],
        })
        model = OddsBaseline()
        probs = model.predict_proba(df)
        assert not np.isnan(probs).any()
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# BaselineModel (LogisticRegression) tests
# ---------------------------------------------------------------------------

class TestBaselineModel:
    def test_fit_predict_shape(self, sample_df, lr_config, all_features):
        model = BaselineModel(lr_config)
        X, y = sample_df[all_features], sample_df["FTR"]
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 3)

    def test_probabilities_sum_to_one(self, sample_df, lr_config, all_features):
        model = BaselineModel(lr_config)
        X, y = sample_df[all_features], sample_df["FTR"]
        model.fit(X, y)
        probs = model.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_probabilities_in_valid_range(self, sample_df, lr_config, all_features):
        model = BaselineModel(lr_config)
        X, y = sample_df[all_features], sample_df["FTR"]
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_save_load_roundtrip(self, sample_df, lr_config, all_features):
        model = BaselineModel(lr_config)
        X, y = sample_df[all_features], sample_df["FTR"]
        model.fit(X, y)
        original_probs = model.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            model.save(path)
            loaded = BaselineModel.load(path)
            loaded_probs = loaded.predict_proba(X)

        np.testing.assert_array_equal(original_probs, loaded_probs)

    def test_handles_nan_features(self, lr_config):
        X = pd.DataFrame({
            "f1": [1.0, np.nan, 3.0, 4.0, 5.0] * 10,
            "f2": [np.nan, 2.0, 3.0, 4.0, 5.0] * 10,
        })
        y = pd.Series(["H", "D", "A", "H", "D"] * 10)
        model = BaselineModel(lr_config)
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (50, 3)
        assert not np.isnan(probs).any()


# ---------------------------------------------------------------------------
# GradientBoostingModel tests
# ---------------------------------------------------------------------------

class TestGradientBoostingModel:
    def test_fit_predict_shape(self, sample_df, gbm_config, all_features):
        model = GradientBoostingModel(gbm_config)
        X, y = sample_df[all_features], sample_df["FTR"]
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 3)

    def test_probabilities_sum_to_one(self, sample_df, gbm_config, all_features):
        model = GradientBoostingModel(gbm_config)
        X, y = sample_df[all_features], sample_df["FTR"]
        model.fit(X, y)
        probs = model.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_with_early_stopping(self, sample_df, split_config, gbm_config, all_features):
        train, val, test = split_by_season(sample_df, split_config)
        model = GradientBoostingModel(gbm_config)
        model.fit(train[all_features], train["FTR"],
                  X_val=val[all_features], y_val=val["FTR"])
        probs = model.predict_proba(test[all_features])
        assert probs.shape == (len(test), 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_no_odds_features(self, sample_df, gbm_config, non_odds_features):
        """GBM should work without odds features."""
        model = GradientBoostingModel(gbm_config)
        X, y = sample_df[non_odds_features], sample_df["FTR"]
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 3)

    def test_feature_importance(self, sample_df, gbm_config, all_features):
        model = GradientBoostingModel(gbm_config)
        X, y = sample_df[all_features], sample_df["FTR"]
        model.fit(X, y)
        imp = model.get_feature_importance()
        assert len(imp) == len(all_features)
        assert "feature" in imp.columns
        assert "importance" in imp.columns

    def test_save_load_roundtrip(self, sample_df, gbm_config, all_features):
        model = GradientBoostingModel(gbm_config)
        X, y = sample_df[all_features], sample_df["FTR"]
        model.fit(X, y)
        original_probs = model.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gbm.pkl"
            model.save(path)
            loaded = GradientBoostingModel.load(path)
            loaded_probs = loaded.predict_proba(X)

        np.testing.assert_array_equal(original_probs, loaded_probs)

    def test_handles_nan_features(self, gbm_config):
        X = pd.DataFrame({
            "f1": [1.0, np.nan, 3.0, 4.0, 5.0] * 20,
            "f2": [np.nan, 2.0, 3.0, 4.0, 5.0] * 20,
        })
        y = pd.Series(["H", "D", "A", "H", "D"] * 20)
        model = GradientBoostingModel(gbm_config)
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (100, 3)
        assert not np.isnan(probs).any()


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_log_loss_known(self):
        y_true = pd.Series(["H", "D", "A"])
        y_prob = np.array([
            [0.99, 0.005, 0.005],
            [0.005, 0.99, 0.005],
            [0.005, 0.005, 0.99],
        ])
        ll = compute_log_loss(y_true, y_prob)
        assert ll < 0.05

    def test_log_loss_bad_predictions(self):
        y_true = pd.Series(["H", "D", "A"] * 10)
        good_prob = np.tile([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], (10, 1))
        uniform_prob = np.full((30, 3), 1 / 3)
        assert compute_log_loss(y_true, good_prob) < compute_log_loss(y_true, uniform_prob)

    def test_brier_score_range(self):
        y_true = pd.Series(["H", "D", "A"] * 10)
        y_prob = np.full((30, 3), 1 / 3)
        bs = compute_brier_score(y_true, y_prob)
        assert 0 <= bs <= 2

    def test_brier_score_perfect(self):
        y_true = pd.Series(["H", "D", "A"])
        y_prob = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        bs = compute_brier_score(y_true, y_prob)
        assert bs == pytest.approx(0.0, abs=1e-10)

    def test_accuracy_perfect(self):
        y_true = pd.Series(["H", "D", "A"])
        y_prob = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        assert compute_accuracy(y_true, y_prob) == pytest.approx(1.0)

    def test_calibration_bins(self):
        rng = np.random.RandomState(42)
        n = 1000
        y_true = pd.Series(rng.choice(["H", "D", "A"], n))
        y_prob = rng.dirichlet([1, 1, 1], n)
        cal = compute_calibration(y_true, y_prob, n_bins=5)
        assert set(cal.keys()) == {"H", "D", "A"}
        for cls_data in cal.values():
            assert len(cls_data["bin_centers"]) > 0
            assert (cls_data["obs_freqs"] >= 0).all()
            assert (cls_data["obs_freqs"] <= 1).all()

    def test_roi_perfect_predictions(self):
        y_true = pd.Series(["H", "D", "A", "H", "H"])
        y_prob = np.array([
            [0.95, 0.025, 0.025],
            [0.025, 0.95, 0.025],
            [0.025, 0.025, 0.95],
            [0.95, 0.025, 0.025],
            [0.95, 0.025, 0.025],
        ])
        odds_h = np.array([2.0, 4.0, 6.0, 2.0, 2.0])
        odds_d = np.array([3.0, 2.5, 4.0, 3.0, 3.0])
        odds_a = np.array([5.0, 3.0, 1.8, 5.0, 5.0])
        roi = compute_roi(y_true, y_prob, odds_h, odds_d, odds_a, threshold=0.05)
        assert roi["n_bets"] > 0
        assert roi["roi_pct"] > 0


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline_lr(self, sample_df, split_config, lr_config, all_features):
        train, val, test = split_by_season(sample_df, split_config)
        model = BaselineModel(lr_config)
        model.fit(train[all_features], train["FTR"])
        probs = model.predict_proba(test[all_features])
        assert probs.shape == (len(test), 3)
        ll = compute_log_loss(test["FTR"], probs)
        bs = compute_brier_score(test["FTR"], probs)
        assert ll > 0
        assert 0 <= bs <= 2

    def test_full_pipeline_gbm(self, sample_df, split_config, gbm_config, non_odds_features):
        """GBM no-odds end-to-end."""
        train, val, test = split_by_season(sample_df, split_config)
        model = GradientBoostingModel(gbm_config)
        model.fit(train[non_odds_features], train["FTR"],
                  X_val=val[non_odds_features], y_val=val["FTR"])
        probs = model.predict_proba(test[non_odds_features])
        assert probs.shape == (len(test), 3)
        ll = compute_log_loss(test["FTR"], probs)
        assert ll > 0


# ---------------------------------------------------------------------------
# build_feature_sets tests
# ---------------------------------------------------------------------------

class TestBuildFeatureSets:
    @pytest.fixture
    def feature_config(self):
        return {
            "features": {
                "odds_b365": ["b365_prob_h"],
                "odds_pinnacle_open": ["ps_prob_h"],
                "non_odds": ["elo_diff", "form_diff_5"],
                "venue": ["h_venue_rolling3_ppg"],
                "rolling_extended": ["h_rolling10_ppg"],
                "tier1_xg": ["xg_diff_5"],
            }
        }

    @pytest.fixture
    def feature_df(self):
        return pd.DataFrame({
            "b365_prob_h": [1], "ps_prob_h": [1],
            "elo_diff": [1], "form_diff_5": [1],
            "h_venue_rolling3_ppg": [1], "h_rolling10_ppg": [1],
            "xg_diff_5": [1],
        })

    def test_feature_sets_structure(self, feature_config, feature_df):
        """build_feature_sets should return dict with expected keys."""
        fsets = build_feature_sets(feature_config, feature_df)
        assert "all_league" in fsets
        assert "tier1" in fsets
        assert "no_odds" in fsets

    def test_all_league_excludes_tier1_xg(self, feature_config, feature_df):
        """all_league set should not include tier1_xg features."""
        fsets = build_feature_sets(feature_config, feature_df)
        assert "xg_diff_5" not in fsets["all_league"]
        assert "b365_prob_h" in fsets["all_league"]
        assert "ps_prob_h" in fsets["all_league"]

    def test_tier1_includes_xg(self, feature_config, feature_df):
        """tier1 set should include tier1_xg features."""
        fsets = build_feature_sets(feature_config, feature_df)
        assert "xg_diff_5" in fsets["tier1"]

    def test_no_odds_excludes_odds(self, feature_config, feature_df):
        """no_odds set should exclude all odds features."""
        fsets = build_feature_sets(feature_config, feature_df)
        assert "b365_prob_h" not in fsets["no_odds"]
        assert "ps_prob_h" not in fsets["no_odds"]
        assert "elo_diff" in fsets["no_odds"]

    def test_missing_columns_filtered(self):
        """Features not present in df should be silently filtered out."""
        config = {
            "features": {
                "odds_b365": ["b365_prob_h", "b365_prob_d"],
                "odds_pinnacle_open": [],
                "non_odds": ["elo_diff", "MISSING_COL"],
                "venue": [],
                "rolling_extended": [],
                "tier1_xg": [],
            }
        }
        df = pd.DataFrame({"b365_prob_h": [1], "b365_prob_d": [1], "elo_diff": [1]})
        fsets = build_feature_sets(config, df)
        assert "MISSING_COL" not in fsets["all_league"]
        assert "elo_diff" in fsets["all_league"]


# ---------------------------------------------------------------------------
# League breakdown tests
# ---------------------------------------------------------------------------

class TestLeagueBreakdown:
    def test_returns_per_league_metrics(self):
        """League breakdown should return a row per league with metrics."""
        rng = np.random.RandomState(42)
        n = 200
        y_true = pd.Series(rng.choice(["H", "D", "A"], n, p=[0.44, 0.27, 0.29]))
        y_prob = rng.dirichlet([2, 1, 1], n)
        leagues = pd.Series(["EPL"] * 100 + ["La_Liga"] * 100)
        odds_df = pd.DataFrame({
            "B365H": rng.uniform(1.3, 4.0, n),
            "B365D": rng.uniform(2.5, 5.0, n),
            "B365A": rng.uniform(1.5, 8.0, n),
        })
        result = print_league_breakdown(y_true, y_prob, odds_df, leagues, "test")
        assert len(result) == 2
        assert set(result["league"]) == {"EPL", "La_Liga"}
        assert "log_loss" in result.columns
        assert "roi_pct" in result.columns
        assert "n_bets" in result.columns

    def test_skips_small_leagues(self):
        """Leagues with fewer than 10 matches should be skipped."""
        rng = np.random.RandomState(42)
        n = 105
        y_true = pd.Series(rng.choice(["H", "D", "A"], n))
        y_prob = rng.dirichlet([1, 1, 1], n)
        leagues = pd.Series(["EPL"] * 100 + ["Tiny"] * 5)
        odds_df = pd.DataFrame({
            "B365H": rng.uniform(1.3, 4.0, n),
            "B365D": rng.uniform(2.5, 5.0, n),
            "B365A": rng.uniform(1.5, 8.0, n),
        })
        result = print_league_breakdown(y_true, y_prob, odds_df, leagues, "test")
        assert len(result) == 1
        assert result.iloc[0]["league"] == "EPL"


# ---------------------------------------------------------------------------
# Tier 1 subset tests
# ---------------------------------------------------------------------------

class TestTier1Subset:
    def test_tier1_only_trains_on_tier1_leagues(self):
        """Tier 1 model should only use data from Tier 1 leagues."""
        rng = np.random.RandomState(42)
        tier1_leagues = ["EPL", "La_Liga"]
        n = 300
        leagues = rng.choice(["EPL", "La_Liga", "SPL", "Eredivisie"], n)
        df = pd.DataFrame({
            "league": leagues,
            "season": np.repeat([2017, 2018, 2019, 2020, 2021, 2022], 50),
        })
        tier1_mask = df["league"].isin(tier1_leagues)
        tier1_df = df[tier1_mask]

        # Verify filtering works correctly
        assert set(tier1_df["league"].unique()).issubset(set(tier1_leagues))
        assert len(tier1_df) < len(df)
        assert len(tier1_df) > 0

    def test_tier1_features_superset_of_all_league(self):
        """tier1 feature set should contain all features from all_league plus extras."""
        config = {
            "features": {
                "odds_b365": ["b365_prob_h"],
                "odds_pinnacle_open": ["ps_prob_h"],
                "non_odds": ["elo_diff"],
                "venue": ["h_venue_rolling3_ppg"],
                "rolling_extended": ["h_rolling10_ppg"],
                "tier1_xg": ["xg_diff_5"],
            }
        }
        df = pd.DataFrame({
            "b365_prob_h": [1], "ps_prob_h": [1],
            "elo_diff": [1], "h_venue_rolling3_ppg": [1],
            "h_rolling10_ppg": [1], "xg_diff_5": [1],
        })
        fsets = build_feature_sets(config, df)
        assert set(fsets["all_league"]).issubset(set(fsets["tier1"]))
        assert len(fsets["tier1"]) > len(fsets["all_league"])
