"""Unit tests for value betting models and metrics."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.value_metrics import (
    adjust_odds_for_costs,
    compute_closing_line_stats,
    compute_cost_sensitivity,
    compute_flat_stake_roi,
    compute_kelly_fraction,
    compute_kelly_roi,
    compute_oracle_roi,
    compute_per_season_roi,
    compute_roi_curve,
)
from src.models.train_clv import compute_targets
from src.models.value_betting import (
    CLVClassifier,
    DisagreementSelector,
    MetaModel,
    combine_disagree_clv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Synthetic DataFrame with all columns needed for value betting."""
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
    return compute_targets(df)


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
def clv_features():
    return [
        "b365_prob_h", "b365_prob_d", "b365_prob_a",
        "ps_prob_h", "ps_prob_d", "ps_prob_a",
        "odds_disagree_h", "odds_disagree_d", "odds_disagree_a",
        "elo_diff", "h_elo_expected",
        "form_diff_5", "form_diff_10",
    ]


@pytest.fixture
def lgbm_config():
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


# ---------------------------------------------------------------------------
# TestDisagreementSelector
# ---------------------------------------------------------------------------

class TestDisagreementSelector:
    def test_fit_learns_cutoffs(self, sample_df):
        """fit() should populate percentile_cutoffs dict."""
        sel = DisagreementSelector({"percentiles": [3, 5, 10]})
        sel.fit(sample_df)
        assert sel.percentile_cutoffs is not None
        assert 3 in sel.percentile_cutoffs
        assert "h" in sel.percentile_cutoffs[3]
        assert "d" in sel.percentile_cutoffs[3]
        assert "a" in sel.percentile_cutoffs[3]

    def test_cutoff_monotonicity(self, sample_df):
        """Lower percentile should have lower (more negative) cutoff."""
        sel = DisagreementSelector({"percentiles": [1, 5, 10, 20]})
        sel.fit(sample_df)
        for suffix in ["h", "d", "a"]:
            cuts = [sel.percentile_cutoffs[p][suffix] for p in [1, 5, 10, 20]]
            for i in range(len(cuts) - 1):
                assert cuts[i] <= cuts[i + 1], f"Not monotonic for {suffix}: {cuts}"

    def test_select_returns_bool_masks(self, sample_df):
        """select() should return boolean arrays."""
        sel = DisagreementSelector({"percentiles": [3, 5, 10], "default_percentile": 5})
        sel.fit(sample_df)
        masks = sel.select(sample_df, percentile=5)
        assert set(masks.keys()) == {"H", "D", "A"}
        for outcome in ["H", "D", "A"]:
            assert masks[outcome].dtype == bool
            assert len(masks[outcome]) == len(sample_df)

    def test_top3_subset_of_top10(self, sample_df):
        """Top 3% selections should be a subset of top 10%."""
        sel = DisagreementSelector({"percentiles": [3, 10]})
        sel.fit(sample_df)
        masks_3 = sel.select(sample_df, percentile=3)
        masks_10 = sel.select(sample_df, percentile=10)
        for outcome in ["H", "D", "A"]:
            # Every bet selected at p=3 should also be selected at p=10
            assert np.all(masks_3[outcome] <= masks_10[outcome])

    def test_training_cutoffs_used(self, sample_df):
        """Cutoffs should be computed from training data, not applied data."""
        rng = np.random.RandomState(99)
        sel = DisagreementSelector({"percentiles": [50]})
        sel.fit(sample_df)  # Learn cutoffs from sample_df

        # Create different test data with shifted disagreement
        test_df = sample_df.copy()
        test_df["odds_disagree_h"] = sample_df["odds_disagree_h"] + 10.0
        masks = sel.select(test_df, percentile=50)
        # With huge shift, nothing should be selected (all values far above cutoff)
        assert masks["H"].sum() == 0

    def test_invalid_percentile_raises(self, sample_df):
        """Requesting unfitted percentile should raise ValueError."""
        sel = DisagreementSelector({"percentiles": [3, 5]})
        sel.fit(sample_df)
        with pytest.raises(ValueError, match="Percentile 99"):
            sel.select(sample_df, percentile=99)

    def test_save_load_roundtrip(self, sample_df):
        """Save/load should preserve cutoffs."""
        sel = DisagreementSelector({"percentiles": [3, 5, 10]})
        sel.fit(sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "disagree.pkl"
            sel.save(path)
            loaded = DisagreementSelector.load(path)

        assert loaded.percentile_cutoffs == sel.percentile_cutoffs
        assert loaded.percentiles == sel.percentiles

    def test_nan_handling(self):
        """NaN in odds_disagree should be handled gracefully."""
        df = pd.DataFrame({
            "odds_disagree_h": [0.01, np.nan, -0.05, 0.02, -0.03],
            "odds_disagree_d": [0.02, 0.01, np.nan, -0.01, 0.03],
            "odds_disagree_a": [-0.02, 0.01, 0.03, np.nan, -0.01],
        })
        sel = DisagreementSelector({"percentiles": [20, 50]})
        sel.fit(df)
        assert sel.percentile_cutoffs is not None
        masks = sel.select(df, percentile=50)
        assert len(masks["H"]) == 5

    def test_default_percentile(self, sample_df):
        """select() with no percentile should use default_percentile."""
        sel = DisagreementSelector({"percentiles": [3, 5], "default_percentile": 5})
        sel.fit(sample_df)
        masks_default = sel.select(sample_df)
        masks_5 = sel.select(sample_df, percentile=5)
        for outcome in ["H", "D", "A"]:
            np.testing.assert_array_equal(masks_default[outcome], masks_5[outcome])


# ---------------------------------------------------------------------------
# TestCLVClassifier
# ---------------------------------------------------------------------------

class TestCLVClassifier:
    def test_fit_predict_shape(self, sample_df, clv_features, lgbm_config):
        """predict_proba should return 1-D array of correct length."""
        clf = CLVClassifier(lgbm_config, clv_threshold=0.02)
        clf.fit(sample_df[clv_features], sample_df["clv_h"])
        probs = clf.predict_proba(sample_df[clv_features])
        assert probs.shape == (len(sample_df),)

    def test_probabilities_in_range(self, sample_df, clv_features, lgbm_config):
        """Predicted probabilities should be in [0, 1]."""
        clf = CLVClassifier(lgbm_config, clv_threshold=0.02)
        clf.fit(sample_df[clv_features], sample_df["clv_h"])
        probs = clf.predict_proba(sample_df[clv_features])
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_predict_returns_bool(self, sample_df, clv_features, lgbm_config):
        """predict() should return boolean array."""
        clf = CLVClassifier(lgbm_config, clv_threshold=0.02)
        clf.fit(sample_df[clv_features], sample_df["clv_h"])
        preds = clf.predict(sample_df[clv_features], cutoff=0.5)
        assert preds.dtype == bool

    def test_binarization_works(self, sample_df, clv_features, lgbm_config):
        """Model should be trained on binarized CLV target."""
        clv_vals = sample_df["clv_h"].values
        # Use realistic thresholds that produce a mix of 0s and 1s
        low_thresh = float(np.nanpercentile(clv_vals, 25))
        high_thresh = float(np.nanpercentile(clv_vals, 75))

        clf_low = CLVClassifier(lgbm_config, clv_threshold=low_thresh)
        clf_low.fit(sample_df[clv_features], sample_df["clv_h"])

        clf_high = CLVClassifier(lgbm_config, clv_threshold=high_thresh)
        clf_high.fit(sample_df[clv_features], sample_df["clv_h"])

        preds_low = clf_low.predict(sample_df[clv_features], cutoff=0.5)
        preds_high = clf_high.predict(sample_df[clv_features], cutoff=0.5)

        # Lower threshold → more positives → more bets selected
        assert preds_low.sum() >= preds_high.sum()

    def test_different_thresholds(self, sample_df, clv_features, lgbm_config):
        """Different CLV thresholds should produce different models."""
        clf_002 = CLVClassifier(lgbm_config, clv_threshold=0.02)
        clf_002.fit(sample_df[clv_features], sample_df["clv_h"])
        clf_005 = CLVClassifier(lgbm_config, clv_threshold=0.05)
        clf_005.fit(sample_df[clv_features], sample_df["clv_h"])

        p1 = clf_002.predict_proba(sample_df[clv_features])
        p2 = clf_005.predict_proba(sample_df[clv_features])
        # Not identical
        assert not np.allclose(p1, p2)

    def test_early_stopping(self, sample_df, clv_features, lgbm_config, split_config):
        """Early stopping should work with validation set."""
        from src.models.train import split_by_season
        train, val, test = split_by_season(sample_df, split_config)

        clf = CLVClassifier(lgbm_config, clv_threshold=0.02)
        clf.fit(
            train[clv_features], train["clv_h"],
            X_val=val[clv_features], y_val_clv=val["clv_h"],
        )
        probs = clf.predict_proba(test[clv_features])
        assert probs.shape == (len(test),)
        assert not np.isnan(probs).any()

    def test_feature_importance(self, sample_df, clv_features, lgbm_config):
        """Feature importance should have correct length."""
        clf = CLVClassifier(lgbm_config, clv_threshold=0.02)
        clf.fit(sample_df[clv_features], sample_df["clv_h"])
        imp = clf.get_feature_importance()
        assert len(imp) == len(clv_features)
        assert "feature" in imp.columns
        assert "importance" in imp.columns

    def test_save_load_roundtrip(self, sample_df, clv_features, lgbm_config):
        """Save/load should preserve predictions."""
        clf = CLVClassifier(lgbm_config, clv_threshold=0.02)
        clf.fit(sample_df[clv_features], sample_df["clv_h"])
        original = clf.predict_proba(sample_df[clv_features])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "clf.pkl"
            clf.save(path)
            loaded = CLVClassifier.load(path)
            loaded_probs = loaded.predict_proba(sample_df[clv_features])

        np.testing.assert_array_equal(original, loaded_probs)
        assert loaded.clv_threshold == 0.02

    def test_nan_handling(self, lgbm_config):
        """NaN features should be imputed."""
        X = pd.DataFrame({
            "f1": [1.0, np.nan, 3.0, 4.0, 5.0] * 20,
            "f2": [np.nan, 2.0, 3.0, 4.0, 5.0] * 20,
        })
        y_clv = np.random.RandomState(42).randn(100) * 0.05
        clf = CLVClassifier(lgbm_config, clv_threshold=0.02)
        clf.fit(X, y_clv)
        probs = clf.predict_proba(X)
        assert probs.shape == (100,)
        assert not np.isnan(probs).any()


# ---------------------------------------------------------------------------
# TestMetaModel
# ---------------------------------------------------------------------------

class TestMetaModel:
    def _make_meta_df(self, n=100):
        """Create a synthetic meta-features DataFrame."""
        rng = np.random.RandomState(42)
        return pd.DataFrame({
            "match_idx": np.repeat(np.arange(n // 3), 3)[:n],
            "outcome": (["H", "D", "A"] * (n // 3 + 1))[:n],
            "outcome_code": ([0, 1, 2] * (n // 3 + 1))[:n],
            "disagree_raw": rng.randn(n) * 0.02,
            "disagree_rank": rng.rand(n),
            "clv_classifier_prob": rng.rand(n),
            "clv_regressor_pred": rng.randn(n) * 0.02,
            "outcome_model_prob": rng.rand(n) * 0.5 + 0.1,
            "outcome_model_edge": rng.randn(n) * 0.05,
            "b365_odds": rng.uniform(1.5, 5.0, n),
            "b365_implied": rng.uniform(0.2, 0.6, n),
            "ps_opening_prob": rng.uniform(0.2, 0.6, n),
        })

    def test_fit_predict_shape(self):
        """fit/predict should work and return correct shape."""
        meta_df = self._make_meta_df(99)
        y = np.random.RandomState(42).choice([0, 1], size=99, p=[0.67, 0.33])
        model = MetaModel({"learner": "logistic_regression",
                           "logistic_regression": {"C": 1.0, "random_state": 42}})
        model.fit(meta_df, y)
        probs = model.predict_proba(meta_df)
        assert probs.shape == (99,)

    def test_probabilities_in_range(self):
        """Probabilities should be in [0, 1]."""
        meta_df = self._make_meta_df(99)
        y = np.random.RandomState(42).choice([0, 1], size=99)
        model = MetaModel({"learner": "logistic_regression",
                           "logistic_regression": {"C": 1.0, "random_state": 42}})
        model.fit(meta_df, y)
        probs = model.predict_proba(meta_df)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_lr_learner(self):
        """Logistic regression learner should fit and have scaler."""
        meta_df = self._make_meta_df(99)
        y = np.random.RandomState(42).choice([0, 1], size=99)
        model = MetaModel({"learner": "logistic_regression",
                           "logistic_regression": {"C": 1.0, "random_state": 42}})
        model.fit(meta_df, y)
        assert model.scaler is not None
        assert model.learner_type == "logistic_regression"

    def test_lightgbm_learner(self):
        """LightGBM learner should fit without scaler."""
        meta_df = self._make_meta_df(99)
        y = np.random.RandomState(42).choice([0, 1], size=99)
        model = MetaModel({"learner": "lightgbm",
                           "lightgbm": {"n_estimators": 20, "max_depth": 2,
                                        "random_state": 42, "verbose": -1}})
        model.fit(meta_df, y)
        assert model.scaler is None
        assert model.learner_type == "lightgbm"

    def test_build_meta_features_shape(self, sample_df):
        """build_meta_features should produce 3× rows (one per outcome)."""
        n = len(sample_df)
        meta_df = MetaModel.build_meta_features(sample_df)
        assert len(meta_df) == 3 * n
        assert set(meta_df["outcome"].unique()) == {"H", "D", "A"}

    def test_build_meta_features_with_none_models(self, sample_df):
        """Should work gracefully when all base models are None."""
        meta_df = MetaModel.build_meta_features(
            sample_df,
            disagree_selector=None,
            clv_classifiers=None,
            clv_regressors=None,
            outcome_model=None,
        )
        assert len(meta_df) == 3 * len(sample_df)
        assert "match_idx" in meta_df.columns
        assert "outcome" in meta_df.columns

    def test_save_load_roundtrip(self):
        """Save/load should preserve predictions."""
        meta_df = self._make_meta_df(99)
        y = np.random.RandomState(42).choice([0, 1], size=99)
        model = MetaModel({"learner": "logistic_regression",
                           "logistic_regression": {"C": 1.0, "random_state": 42}})
        model.fit(meta_df, y)
        original = model.predict_proba(meta_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meta.pkl"
            model.save(path)
            loaded = MetaModel.load(path)
            loaded_probs = loaded.predict_proba(meta_df)

        np.testing.assert_array_almost_equal(original, loaded_probs)


# ---------------------------------------------------------------------------
# TestValueMetrics
# ---------------------------------------------------------------------------

class TestValueMetrics:
    def test_flat_stake_roi_all_wins(self):
        """All bets win at odds 2.0 → ROI = 100%."""
        ftr = np.array(["H", "H", "H"])
        masks = {
            "H": np.array([True, True, True]),
            "D": np.array([False, False, False]),
            "A": np.array([False, False, False]),
        }
        result = compute_flat_stake_roi(
            ftr, masks,
            np.array([2.0, 2.0, 2.0]),
            np.array([3.0, 3.0, 3.0]),
            np.array([4.0, 4.0, 4.0]),
        )
        assert result["n_bets"] == 3
        assert result["roi_pct"] == pytest.approx(100.0)

    def test_flat_stake_roi_no_bets(self):
        """No bets → ROI = 0, n_bets = 0."""
        ftr = np.array(["H", "D", "A"])
        masks = {
            "H": np.array([False, False, False]),
            "D": np.array([False, False, False]),
            "A": np.array([False, False, False]),
        }
        result = compute_flat_stake_roi(
            ftr, masks,
            np.array([2.0, 3.0, 4.0]),
            np.array([3.0, 3.0, 3.0]),
            np.array([4.0, 4.0, 4.0]),
        )
        assert result["n_bets"] == 0
        assert result["roi_pct"] == 0.0

    def test_flat_stake_per_outcome(self):
        """Per-outcome breakdown should be correct."""
        ftr = np.array(["H", "D", "A"])
        masks = {
            "H": np.array([True, False, False]),
            "D": np.array([False, True, False]),
            "A": np.array([False, False, True]),
        }
        result = compute_flat_stake_roi(
            ftr, masks,
            np.array([2.0, 3.0, 4.0]),
            np.array([3.0, 3.5, 3.0]),
            np.array([4.0, 4.0, 2.5]),
        )
        assert result["by_outcome"]["H"]["n_bets"] == 1
        assert result["by_outcome"]["H"]["roi_pct"] == pytest.approx(100.0)
        assert result["by_outcome"]["D"]["n_bets"] == 1
        assert result["by_outcome"]["D"]["roi_pct"] == pytest.approx(250.0)
        assert result["by_outcome"]["A"]["n_bets"] == 1
        assert result["by_outcome"]["A"]["roi_pct"] == pytest.approx(150.0)

    def test_roi_curve_shape(self):
        """compute_roi_curve should return DataFrame with correct columns."""
        ftr = np.array(["H", "D", "A", "H", "D"])
        scores = {
            "H": np.array([0.8, 0.3, 0.2, 0.7, 0.1]),
            "D": np.array([0.1, 0.6, 0.5, 0.2, 0.7]),
            "A": np.array([0.1, 0.1, 0.3, 0.1, 0.2]),
        }
        cutoffs = [0.3, 0.5, 0.7]
        result = compute_roi_curve(
            ftr, scores,
            np.array([2.0, 3.0, 4.0, 2.5, 3.5]),
            np.array([3.0, 2.5, 3.0, 3.5, 2.8]),
            np.array([5.0, 6.0, 3.0, 4.0, 5.0]),
            cutoffs,
        )
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["cutoff", "n_bets", "roi_pct"]
        assert len(result) == 3

    def test_per_season_roi(self):
        """Per-season ROI should cover each season."""
        ftr = np.array(["H", "H", "D", "A"])
        masks = {
            "H": np.array([True, True, True, True]),
            "D": np.array([False, False, False, False]),
            "A": np.array([False, False, False, False]),
        }
        seasons = np.array([2021, 2021, 2022, 2022])
        result = compute_per_season_roi(
            ftr, masks,
            np.array([2.0, 2.5, 3.0, 4.0]),
            np.array([3.0, 3.0, 3.0, 3.0]),
            np.array([4.0, 4.0, 4.0, 4.0]),
            seasons,
        )
        assert len(result) == 2
        assert set(result["season"]) == {2021, 2022}

    def test_oracle_roi(self):
        """Oracle with actual CLV should bet correctly."""
        ftr = np.array(["H", "D", "A"])
        actual_clv = {
            "H": np.array([0.05, -0.01, -0.01]),  # positive CLV for match 0 (H)
            "D": np.array([-0.01, 0.05, -0.01]),   # positive CLV for match 1 (D)
            "A": np.array([-0.01, -0.01, 0.05]),   # positive CLV for match 2 (A)
        }
        result = compute_oracle_roi(
            ftr, actual_clv,
            np.array([2.0, 3.0, 4.0]),
            np.array([3.0, 3.5, 3.0]),
            np.array([4.0, 4.0, 2.5]),
        )
        # All 3 bets should be placed (one per outcome), all win
        assert result["n_bets"] == 3
        assert result["roi_pct"] > 0


# ---------------------------------------------------------------------------
# TestBuildMetaFeatures
# ---------------------------------------------------------------------------

class TestBuildMetaFeatures:
    def test_correct_row_count(self, sample_df):
        """Should produce exactly 3× matches rows."""
        meta_df = MetaModel.build_meta_features(sample_df)
        assert len(meta_df) == 3 * len(sample_df)

    def test_handles_none_base_models(self, sample_df):
        """Should work with all None models, just producing basic features."""
        meta_df = MetaModel.build_meta_features(
            sample_df,
            disagree_selector=None,
            clv_classifiers=None,
            clv_regressors=None,
            outcome_model=None,
        )
        assert "match_idx" in meta_df.columns
        assert "outcome" in meta_df.columns
        assert "outcome_code" in meta_df.columns
        # Should still have odds-based features
        if "b365_implied" in meta_df.columns:
            assert not meta_df["b365_implied"].isna().all()


# ---------------------------------------------------------------------------
# TestValueIntegration
# ---------------------------------------------------------------------------

class TestValueIntegration:
    def test_full_pipeline_synthetic(self, sample_df, split_config, clv_features, lgbm_config):
        """End-to-end: disagreement + classifier + meta on synthetic data."""
        from src.models.train import split_by_season

        train, val, test = split_by_season(sample_df, split_config)

        # Strategy A: Disagreement
        sel = DisagreementSelector({"percentiles": [5, 10, 20], "default_percentile": 10})
        sel.fit(train)
        disagree_masks = sel.select(test, percentile=10)
        roi_a = compute_flat_stake_roi(
            test["FTR"].values, disagree_masks,
            test["B365H"].values, test["B365D"].values, test["B365A"].values,
        )
        assert "roi_pct" in roi_a

        # Strategy B: CLV Classifier
        classifiers = {}
        for outcome, suffix in [("H", "h"), ("D", "d"), ("A", "a")]:
            clf = CLVClassifier(lgbm_config, clv_threshold=0.02)
            target = f"clv_{suffix}"
            train_valid = train.dropna(subset=[target])
            clf.fit(train_valid[clv_features], train_valid[target])
            classifiers[outcome] = clf

        cls_masks = {}
        for outcome in ["H", "D", "A"]:
            cls_masks[outcome] = classifiers[outcome].predict(test[clv_features], cutoff=0.5)
        roi_b = compute_flat_stake_roi(
            test["FTR"].values, cls_masks,
            test["B365H"].values, test["B365D"].values, test["B365A"].values,
        )
        assert "roi_pct" in roi_b

        # Strategy C: Meta-Model
        val_meta = MetaModel.build_meta_features(
            val,
            disagree_selector=sel,
            clv_classifiers=classifiers,
            clv_feature_cols=clv_features,
            percentile=10,
        )
        val_ftr_expanded = np.tile(val["FTR"].values, 3)
        y_val_meta = (val_ftr_expanded == val_meta["outcome"].values).astype(int)

        meta_model = MetaModel({"learner": "logistic_regression",
                                "logistic_regression": {"C": 1.0, "random_state": 42}})
        meta_model.fit(val_meta, y_val_meta)

        test_meta = MetaModel.build_meta_features(
            test,
            disagree_selector=sel,
            clv_classifiers=classifiers,
            clv_feature_cols=clv_features,
            percentile=10,
        )
        probs = meta_model.predict_proba(test_meta)
        assert probs.shape == (len(test_meta),)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_meta_model_no_leakage(self, sample_df, split_config, clv_features, lgbm_config):
        """Meta-model should be trained on val, not train or test."""
        from src.models.train import split_by_season

        train, val, test = split_by_season(sample_df, split_config)

        # Train disagreement on train
        sel = DisagreementSelector({"percentiles": [10], "default_percentile": 10})
        sel.fit(train)

        # Train classifiers on train
        classifiers = {}
        for outcome, suffix in [("H", "h"), ("D", "d"), ("A", "a")]:
            clf = CLVClassifier(lgbm_config, clv_threshold=0.02)
            target = f"clv_{suffix}"
            train_valid = train.dropna(subset=[target])
            clf.fit(train_valid[clv_features], train_valid[target])
            classifiers[outcome] = clf

        # Meta-model trained on val (out-of-sample for base models)
        val_meta = MetaModel.build_meta_features(
            val, disagree_selector=sel, clv_classifiers=classifiers,
            clv_feature_cols=clv_features, percentile=10,
        )
        val_ftr_expanded = np.tile(val["FTR"].values, 3)
        y_val_meta = (val_ftr_expanded == val_meta["outcome"].values).astype(int)

        meta_model = MetaModel({"learner": "logistic_regression",
                                "logistic_regression": {"C": 1.0, "random_state": 42}})
        meta_model.fit(val_meta, y_val_meta)

        # Evaluate on test (completely held out)
        test_meta = MetaModel.build_meta_features(
            test, disagree_selector=sel, clv_classifiers=classifiers,
            clv_feature_cols=clv_features, percentile=10,
        )
        probs = meta_model.predict_proba(test_meta)

        # Verify test seasons are truly held out
        train_seasons = set(split_config["split"]["train_seasons"])
        val_seasons = set(split_config["split"]["val_seasons"])
        test_seasons = set(split_config["split"]["test_seasons"])
        assert train_seasons.isdisjoint(test_seasons)
        assert val_seasons.isdisjoint(test_seasons)
        assert probs.shape == (len(test_meta),)


# ---------------------------------------------------------------------------
# TestLeagueFilter
# ---------------------------------------------------------------------------

class TestLeagueFilter:
    def test_compute_league_filter_returns_set_of_strings(self, sample_df):
        """compute_league_filter should return a set of strings."""
        sel = DisagreementSelector({"percentiles": [10, 20]})
        sel.fit(sample_df)
        result = sel.compute_league_filter(
            sample_df, sample_df["FTR"].values,
            sample_df["B365H"].values, sample_df["B365D"].values, sample_df["B365A"].values,
            sample_df["league"].values, percentile=20,
            min_roi_pct=-100.0, min_bets=1,
        )
        assert isinstance(result, set)
        for item in result:
            assert isinstance(item, str)

    def test_strict_filter_fewer_leagues(self, sample_df):
        """Strict min_roi_pct should produce fewer or equal leagues than lenient."""
        sel = DisagreementSelector({"percentiles": [20]})
        sel.fit(sample_df)

        lenient = sel.compute_league_filter(
            sample_df, sample_df["FTR"].values,
            sample_df["B365H"].values, sample_df["B365D"].values, sample_df["B365A"].values,
            sample_df["league"].values, percentile=20,
            min_roi_pct=-100.0, min_bets=1,
        )
        strict = sel.compute_league_filter(
            sample_df, sample_df["FTR"].values,
            sample_df["B365H"].values, sample_df["B365D"].values, sample_df["B365A"].values,
            sample_df["league"].values, percentile=20,
            min_roi_pct=50.0, min_bets=1,
        )
        assert len(strict) <= len(lenient)

    def test_select_with_league_filter_only_allowed(self, sample_df):
        """select_with_league_filter should only select from allowed leagues."""
        sel = DisagreementSelector({"percentiles": [20]})
        sel.fit(sample_df)

        allowed = {"EPL"}
        masks = sel.select_with_league_filter(
            sample_df, sample_df["league"].values,
            percentile=20, allowed_leagues=allowed,
        )
        leagues = sample_df["league"].values
        for outcome in ["H", "D", "A"]:
            selected_leagues = set(leagues[masks[outcome]])
            assert selected_leagues <= allowed, (
                f"Found leagues {selected_leagues} outside allowed {allowed}"
            )


# ---------------------------------------------------------------------------
# TestCombinationStrategy
# ---------------------------------------------------------------------------

class TestCombinationStrategy:
    def test_combination_is_subset_of_disagreement(self, sample_df, clv_features, lgbm_config):
        """Combination masks should be a subset of disagreement masks (intersection)."""
        sel = DisagreementSelector({"percentiles": [20]})
        sel.fit(sample_df)
        d_masks = sel.select(sample_df, percentile=20)

        # Train a CLV classifier
        classifiers = {}
        for outcome, suffix in [("H", "h"), ("D", "d"), ("A", "a")]:
            clf = CLVClassifier(lgbm_config, clv_threshold=0.02)
            target = f"clv_{suffix}"
            valid = sample_df.dropna(subset=[target])
            clf.fit(valid[clv_features], valid[target])
            classifiers[outcome] = clf

        combo_masks = combine_disagree_clv(d_masks, classifiers, sample_df[clv_features], clv_cutoff=0.3)
        for outcome in ["H", "D", "A"]:
            # Every combo bet should also be a disagreement bet
            assert np.all(combo_masks[outcome] <= d_masks[outcome])

    def test_higher_clv_cutoff_fewer_bets(self, sample_df, clv_features, lgbm_config):
        """Higher CLV cutoff should produce fewer or equal bets (monotonicity)."""
        sel = DisagreementSelector({"percentiles": [20]})
        sel.fit(sample_df)
        d_masks = sel.select(sample_df, percentile=20)

        classifiers = {}
        for outcome, suffix in [("H", "h"), ("D", "d"), ("A", "a")]:
            clf = CLVClassifier(lgbm_config, clv_threshold=0.02)
            target = f"clv_{suffix}"
            valid = sample_df.dropna(subset=[target])
            clf.fit(valid[clv_features], valid[target])
            classifiers[outcome] = clf

        combo_low = combine_disagree_clv(d_masks, classifiers, sample_df[clv_features], clv_cutoff=0.2)
        combo_high = combine_disagree_clv(d_masks, classifiers, sample_df[clv_features], clv_cutoff=0.8)

        total_low = sum(m.sum() for m in combo_low.values())
        total_high = sum(m.sum() for m in combo_high.values())
        assert total_high <= total_low


# ---------------------------------------------------------------------------
# TestKellyCriterion
# ---------------------------------------------------------------------------

class TestKellyCriterion:
    def test_positive_edge_positive_fraction(self):
        """Positive edge should give positive Kelly fraction."""
        # prob=0.6, odds=2.5 → b=1.5, f=(1.5*0.6-0.4)/1.5 = 0.333
        f = compute_kelly_fraction(0.6, 2.5)
        assert f > 0

    def test_negative_edge_negative_fraction(self):
        """Negative edge should give negative Kelly fraction."""
        # prob=0.2, odds=2.0 → b=1.0, f=(1.0*0.2-0.8)/1.0 = -0.6
        f = compute_kelly_fraction(0.2, 2.0)
        assert f < 0

    def test_zero_edge(self):
        """Fair odds should give zero Kelly fraction."""
        # prob=0.5, odds=2.0 → b=1.0, f=(1.0*0.5-0.5)/1.0 = 0.0
        f = compute_kelly_fraction(0.5, 2.0)
        assert f == pytest.approx(0.0)

    def test_zero_odds_returns_zero(self):
        """Odds <= 1 should return 0."""
        assert compute_kelly_fraction(0.5, 1.0) == 0.0
        assert compute_kelly_fraction(0.5, 0.5) == 0.0

    def test_all_wins_bankroll_grows(self):
        """If all bets win, bankroll should grow."""
        ftr = np.array(["H", "H", "H"])
        masks = {
            "H": np.array([True, True, True]),
            "D": np.array([False, False, False]),
            "A": np.array([False, False, False]),
        }
        probs = {
            "H": np.array([0.6, 0.6, 0.6]),
            "D": np.array([0.3, 0.3, 0.3]),
            "A": np.array([0.1, 0.1, 0.1]),
        }
        result = compute_kelly_roi(
            ftr, masks, probs,
            np.array([2.5, 2.5, 2.5]),
            np.array([3.0, 3.0, 3.0]),
            np.array([5.0, 5.0, 5.0]),
            kelly_fraction=0.25, max_bet_fraction=0.05, bankroll=1000.0,
        )
        assert result["ending_bankroll"] > result["starting_bankroll"]
        assert result["n_bets"] == 3

    def test_no_bets_bankroll_unchanged(self):
        """No bets → bankroll stays the same."""
        ftr = np.array(["H", "D"])
        masks = {
            "H": np.array([False, False]),
            "D": np.array([False, False]),
            "A": np.array([False, False]),
        }
        probs = {
            "H": np.array([0.5, 0.5]),
            "D": np.array([0.3, 0.3]),
            "A": np.array([0.2, 0.2]),
        }
        result = compute_kelly_roi(
            ftr, masks, probs,
            np.array([2.0, 3.0]),
            np.array([3.0, 3.0]),
            np.array([4.0, 4.0]),
            bankroll=500.0,
        )
        assert result["ending_bankroll"] == 500.0
        assert result["n_bets"] == 0
        assert result["roi_pct"] == 0.0

    def test_max_bet_fraction_respected(self):
        """Kelly stake should never exceed max_bet_fraction of bankroll."""
        ftr = np.array(["H"])
        masks = {
            "H": np.array([True]),
            "D": np.array([False]),
            "A": np.array([False]),
        }
        # Very high prob → large raw Kelly → should be capped
        probs = {
            "H": np.array([0.95]),
            "D": np.array([0.03]),
            "A": np.array([0.02]),
        }
        bankroll = 1000.0
        max_bet = 0.02  # 2% cap
        result = compute_kelly_roi(
            ftr, masks, probs,
            np.array([2.0]),
            np.array([10.0]),
            np.array([10.0]),
            kelly_fraction=1.0,  # Full Kelly
            max_bet_fraction=max_bet,
            bankroll=bankroll,
        )
        # Max stake would be 0.02 * 1000 = 20
        # Win at odds 2.0 → profit = 20 * (2.0-1.0) = 20
        # Ending bankroll = 1020
        assert result["total_staked"] <= max_bet * bankroll + 0.01
        assert result["ending_bankroll"] == pytest.approx(1020.0)

    def test_bankroll_never_negative(self):
        """Bankroll should never go negative even with all losses."""
        rng = np.random.RandomState(42)
        n = 50
        # All away wins but we bet on home
        ftr = np.array(["A"] * n)
        masks = {
            "H": np.ones(n, dtype=bool),
            "D": np.zeros(n, dtype=bool),
            "A": np.zeros(n, dtype=bool),
        }
        probs = {
            "H": np.full(n, 0.55),  # slight perceived edge
            "D": np.full(n, 0.25),
            "A": np.full(n, 0.20),
        }
        result = compute_kelly_roi(
            ftr, masks, probs,
            rng.uniform(1.5, 3.0, n),
            rng.uniform(3.0, 5.0, n),
            rng.uniform(3.0, 8.0, n),
            kelly_fraction=0.25, max_bet_fraction=0.05, bankroll=1000.0,
        )
        assert result["ending_bankroll"] >= 0


# ---------------------------------------------------------------------------
# TestTransactionCosts
# ---------------------------------------------------------------------------

class TestTransactionCosts:
    def test_zero_cost_no_change(self):
        """Zero cost should not change odds."""
        odds = np.array([1.5, 2.0, 3.0, 5.0])
        adjusted = adjust_odds_for_costs(odds, cost_pct=0.0)
        np.testing.assert_array_equal(odds, adjusted)

    def test_cost_reduces_odds(self):
        """Positive cost should reduce odds (but keep >= 1.0)."""
        odds = np.array([1.5, 2.0, 3.0, 5.0])
        adjusted = adjust_odds_for_costs(odds, cost_pct=2.0)
        # effective = 1 + (odds-1) * 0.98
        expected = 1.0 + (odds - 1.0) * 0.98
        np.testing.assert_array_almost_equal(adjusted, expected)
        assert np.all(adjusted < odds)
        assert np.all(adjusted >= 1.0)

    def test_100pct_cost_returns_evens(self):
        """100% cost should collapse all odds to 1.0 (no profit possible)."""
        odds = np.array([1.5, 2.0, 10.0])
        adjusted = adjust_odds_for_costs(odds, cost_pct=100.0)
        np.testing.assert_array_almost_equal(adjusted, [1.0, 1.0, 1.0])

    def test_floor_at_one(self):
        """Adjusted odds should never go below 1.0."""
        odds = np.array([1.01, 1.001])
        adjusted = adjust_odds_for_costs(odds, cost_pct=50.0)
        assert np.all(adjusted >= 1.0)

    def test_higher_cost_lower_roi(self):
        """Higher transaction costs should produce lower or equal ROI."""
        ftr = np.array(["H", "H", "D", "A", "H"])
        masks = {
            "H": np.array([True, True, True, True, True]),
            "D": np.array([False, False, False, False, False]),
            "A": np.array([False, False, False, False, False]),
        }
        b365_h = np.array([2.0, 2.5, 3.0, 2.0, 1.8])
        b365_d = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        b365_a = np.array([4.0, 4.0, 4.0, 4.0, 4.0])

        roi_0 = compute_flat_stake_roi(ftr, masks, b365_h, b365_d, b365_a)["roi_pct"]
        adj_h = adjust_odds_for_costs(b365_h, 3.0)
        adj_d = adjust_odds_for_costs(b365_d, 3.0)
        adj_a = adjust_odds_for_costs(b365_a, 3.0)
        roi_3 = compute_flat_stake_roi(ftr, masks, adj_h, adj_d, adj_a)["roi_pct"]
        assert roi_3 <= roi_0

    def test_cost_sensitivity_shape(self):
        """compute_cost_sensitivity should return correct DataFrame shape."""
        ftr = np.array(["H", "D", "A"])
        masks_a = {
            "H": np.array([True, False, False]),
            "D": np.array([False, True, False]),
            "A": np.array([False, False, True]),
        }
        masks_b = {
            "H": np.array([True, True, False]),
            "D": np.array([False, False, False]),
            "A": np.array([False, False, True]),
        }
        strategies = {"Strat A": masks_a, "Strat B": masks_b}
        cost_pcts = [0.0, 1.0, 2.0]
        df = compute_cost_sensitivity(
            ftr, strategies,
            np.array([2.0, 3.0, 4.0]),
            np.array([3.0, 3.5, 3.0]),
            np.array([4.0, 4.0, 2.5]),
            cost_pcts,
        )
        assert len(df) == 6  # 2 strategies × 3 cost levels
        assert set(df.columns) == {"strategy", "cost_pct", "n_bets", "roi_pct"}


# ---------------------------------------------------------------------------
# TestMultiBookDisagreement
# ---------------------------------------------------------------------------

class TestMultiBookDisagreement:
    def test_select_multibook_returns_masks(self, sample_df):
        """select_multibook should return boolean masks when best columns exist."""
        # Add multibook columns
        rng = np.random.RandomState(42)
        df = sample_df.copy()
        for suffix in ["h", "d", "a"]:
            df[f"odds_disagree_best_{suffix}"] = rng.randn(len(df)) * 0.05
        sel = DisagreementSelector({"percentiles": [5, 10, 20]})
        sel.fit(df)
        sel.fit_multibook(df)
        masks = sel.select_multibook(df, percentile=10)
        assert set(masks.keys()) == {"H", "D", "A"}
        for outcome in ["H", "D", "A"]:
            assert masks[outcome].dtype == bool
            assert len(masks[outcome]) == len(df)

    def test_multibook_more_bets_than_single(self, sample_df):
        """Multibook at same percentile should select >= single-book bets."""
        rng = np.random.RandomState(42)
        df = sample_df.copy()
        # Make best-book disagree slightly more negative than B365 disagree
        for suffix in ["h", "d", "a"]:
            df[f"odds_disagree_best_{suffix}"] = df[f"odds_disagree_{suffix}"] - 0.01
        sel = DisagreementSelector({"percentiles": [10, 20]})
        sel.fit(df)
        sel.fit_multibook(df)

        for p in [10, 20]:
            masks_single = sel.select(df, percentile=p)
            masks_multi = sel.select_multibook(df, percentile=p)
            total_single = sum(m.sum() for m in masks_single.values())
            total_multi = sum(m.sum() for m in masks_multi.values())
            assert total_multi >= total_single

    def test_multibook_nan_handled(self):
        """NaN in best-book disagreement should not cause errors."""
        df = pd.DataFrame({
            "odds_disagree_h": [0.01, -0.05, 0.02, -0.03, 0.01],
            "odds_disagree_d": [0.02, 0.01, -0.01, 0.03, -0.02],
            "odds_disagree_a": [-0.02, 0.01, 0.03, -0.01, 0.02],
            "odds_disagree_best_h": [np.nan, -0.06, 0.01, -0.04, 0.00],
            "odds_disagree_best_d": [0.01, np.nan, -0.02, 0.02, -0.03],
            "odds_disagree_best_a": [-0.03, 0.00, np.nan, -0.02, 0.01],
        })
        sel = DisagreementSelector({"percentiles": [20, 50]})
        sel.fit(df)
        sel.fit_multibook(df)
        masks = sel.select_multibook(df, percentile=50)
        assert len(masks["H"]) == 5

    def test_save_load_preserves_multibook(self, sample_df):
        """Save/load should preserve multibook_cutoffs."""
        rng = np.random.RandomState(42)
        df = sample_df.copy()
        for suffix in ["h", "d", "a"]:
            df[f"odds_disagree_best_{suffix}"] = rng.randn(len(df)) * 0.05
        sel = DisagreementSelector({"percentiles": [5, 10]})
        sel.fit(df)
        sel.fit_multibook(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sel.pkl"
            sel.save(path)
            loaded = DisagreementSelector.load(path)

        assert loaded.multibook_cutoffs is not None
        assert loaded.multibook_cutoffs == sel.multibook_cutoffs


# ---------------------------------------------------------------------------
# TestClosingLineAnalysis
# ---------------------------------------------------------------------------

class TestClosingLineAnalysis:
    def test_correct_structure(self):
        """compute_closing_line_stats should return expected keys."""
        masks = {
            "H": np.array([True, False, True]),
            "D": np.array([False, True, False]),
            "A": np.array([False, False, True]),
        }
        ps = {"H": np.array([0.5, 0.3, 0.4]),
              "D": np.array([0.3, 0.4, 0.3]),
              "A": np.array([0.2, 0.3, 0.3])}
        psc = {"H": np.array([0.52, 0.31, 0.42]),
               "D": np.array([0.28, 0.42, 0.29]),
               "A": np.array([0.20, 0.27, 0.29])}
        result = compute_closing_line_stats(masks, ps, psc, strategy_name="Test")
        assert "strategy" in result
        assert "n_bets" in result
        assert "avg_clv" in result
        assert "pct_beat_closing" in result
        assert "by_outcome" in result
        assert result["strategy"] == "Test"

    def test_avg_clv_in_range(self):
        """avg_clv should be in [-1, 1] for reasonable inputs."""
        rng = np.random.RandomState(42)
        n = 100
        masks = {
            "H": rng.choice([True, False], n),
            "D": rng.choice([True, False], n),
            "A": rng.choice([True, False], n),
        }
        ps = {"H": rng.uniform(0.2, 0.6, n),
              "D": rng.uniform(0.2, 0.4, n),
              "A": rng.uniform(0.1, 0.4, n)}
        psc = {"H": ps["H"] + rng.randn(n) * 0.02,
               "D": ps["D"] + rng.randn(n) * 0.02,
               "A": ps["A"] + rng.randn(n) * 0.02}
        result = compute_closing_line_stats(masks, ps, psc)
        assert -1.0 <= result["avg_clv"] <= 1.0

    def test_empty_masks_return_zeros(self):
        """Empty masks should return zeros."""
        masks = {
            "H": np.array([False, False]),
            "D": np.array([False, False]),
            "A": np.array([False, False]),
        }
        ps = {"H": np.array([0.5, 0.3]),
              "D": np.array([0.3, 0.4]),
              "A": np.array([0.2, 0.3])}
        psc = {"H": np.array([0.52, 0.31]),
               "D": np.array([0.28, 0.42]),
               "A": np.array([0.20, 0.27])}
        result = compute_closing_line_stats(masks, ps, psc)
        assert result["n_bets"] == 0
        assert result["avg_clv"] == 0.0
        assert result["pct_beat_closing"] == 0.0

    def test_all_positive_clv(self):
        """When all CLV is positive, pct_beat_closing should be 100%."""
        masks = {
            "H": np.array([True, True]),
            "D": np.array([False, False]),
            "A": np.array([False, False]),
        }
        ps = {"H": np.array([0.4, 0.5]),
              "D": np.array([0.3, 0.3]),
              "A": np.array([0.3, 0.2])}
        # Closing probs higher than opening for all selected bets
        psc = {"H": np.array([0.45, 0.55]),
               "D": np.array([0.30, 0.28]),
               "A": np.array([0.25, 0.17])}
        result = compute_closing_line_stats(masks, ps, psc)
        assert result["pct_beat_closing"] == 100.0
        assert result["avg_clv"] > 0


# ---------------------------------------------------------------------------
# TestDixonColesModel
# ---------------------------------------------------------------------------

class TestDixonColesModel:
    def test_predict_proba_valid(self):
        """predict_proba should return valid probabilities summing to ~1."""
        from src.models.poisson import DixonColesModel

        model = DixonColesModel({"max_goals": 5, "decay_rate": 0.003, "max_iter": 100})
        # Create simple training data
        rng = np.random.RandomState(42)
        n = 100
        teams = ["A", "B", "C", "D"]
        home = rng.choice(teams, n)
        away = rng.choice(teams, n)
        hg = rng.poisson(1.5, n)
        ag = rng.poisson(1.2, n)
        model.fit(home, away, hg, ag)

        probs = model.predict_proba(np.array(["A", "B"]), np.array(["C", "D"]))
        assert probs.shape == (2, 3)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=0.01)

    def test_unknown_teams_return_prior(self):
        """Unknown teams should return prior [0.437, 0.265, 0.298]."""
        from src.models.poisson import DixonColesModel, PRIOR_PROBS

        model = DixonColesModel({"max_goals": 5, "max_iter": 50})
        rng = np.random.RandomState(42)
        n = 50
        home = rng.choice(["A", "B"], n)
        away = rng.choice(["A", "B"], n)
        model.fit(home, away, rng.poisson(1.5, n), rng.poisson(1.2, n))

        probs = model.predict_proba(np.array(["Unknown"]), np.array(["Team"]))
        np.testing.assert_array_almost_equal(probs[0], PRIOR_PROBS)

    def test_save_load_roundtrip(self):
        """Save/load should preserve predictions."""
        from src.models.poisson import DixonColesModel

        model = DixonColesModel({"max_goals": 5, "max_iter": 50})
        rng = np.random.RandomState(42)
        n = 50
        teams = ["A", "B", "C"]
        home = rng.choice(teams, n)
        away = rng.choice(teams, n)
        model.fit(home, away, rng.poisson(1.5, n), rng.poisson(1.2, n))
        original = model.predict_proba(np.array(["A"]), np.array(["B"]))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dc.pkl"
            model.save(path)
            loaded = DixonColesModel.load(path)
            loaded_probs = loaded.predict_proba(np.array(["A"]), np.array(["B"]))

        np.testing.assert_array_almost_equal(original, loaded_probs)

    def test_nan_goals_skipped(self):
        """NaN goals should be filtered out gracefully."""
        from src.models.poisson import DixonColesModel

        model = DixonColesModel({"max_goals": 5, "max_iter": 50})
        home = np.array(["A", "B", "A", "B", "A"] * 10)
        away = np.array(["B", "A", "B", "A", "B"] * 10)
        hg = np.array([1.0, 2.0, np.nan, 0.0, 1.0] * 10)
        ag = np.array([0.0, np.nan, 1.0, 0.0, 2.0] * 10)
        model.fit(home, away, hg, ag)
        assert model.teams is not None
        probs = model.predict_proba(np.array(["A"]), np.array(["B"]))
        assert probs.shape == (1, 3)
        assert np.all(np.isfinite(probs))


# ---------------------------------------------------------------------------
# TestCLVRegressionRanking
# ---------------------------------------------------------------------------

class TestCLVRegressionRanking:
    def test_lower_top_pct_fewer_bets(self, sample_df, clv_features, lgbm_config):
        """Lower top_pct should produce fewer or equal bets."""
        from src.models.clv import CLVModel

        rng = np.random.RandomState(42)
        # Train a simple regressor
        model = CLVModel(lgbm_config)
        valid = sample_df.dropna(subset=["clv_h"])
        model.fit(valid[clv_features], valid["clv_h"])

        pred_clv = model.predict(sample_df[clv_features])

        counts = []
        for top_pct in [1, 5, 10, 20]:
            n_select = max(1, int(len(sample_df) * top_pct / 100))
            threshold = np.sort(pred_clv)[::-1][min(n_select - 1, len(pred_clv) - 1)]
            selected = (pred_clv >= threshold) & (pred_clv > 0)
            counts.append(int(selected.sum()))

        # Monotonically non-decreasing
        for i in range(len(counts) - 1):
            assert counts[i] <= counts[i + 1], f"Not monotonic: {counts}"

    def test_only_positive_clv_selected(self, sample_df, clv_features, lgbm_config):
        """Only positive predicted CLV should be selected."""
        from src.models.clv import CLVModel

        model = CLVModel(lgbm_config)
        valid = sample_df.dropna(subset=["clv_h"])
        model.fit(valid[clv_features], valid["clv_h"])

        pred_clv = model.predict(sample_df[clv_features])
        n_select = max(1, int(len(sample_df) * 5 / 100))
        threshold = np.sort(pred_clv)[::-1][min(n_select - 1, len(pred_clv) - 1)]
        selected = (pred_clv >= threshold) & (pred_clv > 0)

        # All selected should have positive predicted CLV
        if selected.sum() > 0:
            assert np.all(pred_clv[selected] > 0)
