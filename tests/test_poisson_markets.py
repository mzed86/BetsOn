"""Tests for Dixon-Coles Over/Under and BTTS predictions."""

import numpy as np
import pytest

from src.models.poisson import DixonColesModel


@pytest.fixture
def fitted_dc_model():
    """Fit a small Dixon-Coles model on synthetic data."""
    rng = np.random.RandomState(42)
    n = 200
    teams = ["TeamA", "TeamB", "TeamC", "TeamD"]
    home = rng.choice(teams, n)
    away = rng.choice(teams, n)
    # Ensure home != away
    for i in range(n):
        while away[i] == home[i]:
            away[i] = rng.choice(teams)
    home_goals = rng.poisson(1.4, n)
    away_goals = rng.poisson(1.1, n)

    model = DixonColesModel({"max_goals": 6, "decay_rate": 0.0, "max_iter": 200})
    model.fit(home, away, home_goals.astype(float), away_goals.astype(float))
    return model, teams


class TestPredictOverUnder:
    """Tests for predict_over_under()."""

    def test_output_shape(self, fitted_dc_model):
        model, teams = fitted_dc_model
        home = np.array(["TeamA", "TeamB", "TeamC"])
        away = np.array(["TeamB", "TeamC", "TeamD"])
        result = model.predict_over_under(home, away)
        assert result.shape == (3, 2)

    def test_sums_to_one(self, fitted_dc_model):
        model, teams = fitted_dc_model
        home = np.array(["TeamA", "TeamB", "TeamC"])
        away = np.array(["TeamB", "TeamC", "TeamD"])
        result = model.predict_over_under(home, away)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_values_in_range(self, fitted_dc_model):
        model, teams = fitted_dc_model
        home = np.array(["TeamA", "TeamB"])
        away = np.array(["TeamB", "TeamC"])
        result = model.predict_over_under(home, away)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_unknown_teams_return_prior(self, fitted_dc_model):
        model, _ = fitted_dc_model
        home = np.array(["Unknown1"])
        away = np.array(["Unknown2"])
        result = model.predict_over_under(home, away)
        np.testing.assert_allclose(result[0], [0.50, 0.50])

    def test_threshold_semantics(self, fitted_dc_model):
        """With threshold=2.5: 0+0=0 goals is Under, 2+1=3 goals is Over."""
        model, teams = fitted_dc_model
        # Test that the function respects the threshold
        home = np.array(["TeamA"])
        away = np.array(["TeamB"])
        result = model.predict_over_under(home, away, threshold=2.5)
        # P(Over) + P(Under) = 1
        assert abs(result[0, 0] + result[0, 1] - 1.0) < 1e-6
        # Both probabilities should be meaningful (not degenerate)
        assert result[0, 0] > 0.01
        assert result[0, 1] > 0.01

    def test_different_thresholds(self, fitted_dc_model):
        """Higher threshold should increase P(Under)."""
        model, _ = fitted_dc_model
        home = np.array(["TeamA"])
        away = np.array(["TeamB"])
        result_25 = model.predict_over_under(home, away, threshold=2.5)
        result_35 = model.predict_over_under(home, away, threshold=3.5)
        # P(Under 3.5) >= P(Under 2.5)
        assert result_35[0, 1] >= result_25[0, 1]

    def test_single_match(self, fitted_dc_model):
        model, _ = fitted_dc_model
        result = model.predict_over_under(np.array(["TeamA"]), np.array(["TeamB"]))
        assert result.shape == (1, 2)


class TestPredictBTTS:
    """Tests for predict_btts()."""

    def test_output_shape(self, fitted_dc_model):
        model, _ = fitted_dc_model
        home = np.array(["TeamA", "TeamB", "TeamC"])
        away = np.array(["TeamB", "TeamC", "TeamD"])
        result = model.predict_btts(home, away)
        assert result.shape == (3, 2)

    def test_sums_to_one(self, fitted_dc_model):
        model, _ = fitted_dc_model
        home = np.array(["TeamA", "TeamB", "TeamC"])
        away = np.array(["TeamB", "TeamC", "TeamD"])
        result = model.predict_btts(home, away)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_values_in_range(self, fitted_dc_model):
        model, _ = fitted_dc_model
        home = np.array(["TeamA", "TeamB"])
        away = np.array(["TeamB", "TeamC"])
        result = model.predict_btts(home, away)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_unknown_teams_return_prior(self, fitted_dc_model):
        model, _ = fitted_dc_model
        home = np.array(["Unknown1"])
        away = np.array(["Unknown2"])
        result = model.predict_btts(home, away)
        np.testing.assert_allclose(result[0], [0.55, 0.45])

    def test_btts_meaningful(self, fitted_dc_model):
        """BTTS probabilities should be reasonable (not degenerate)."""
        model, _ = fitted_dc_model
        home = np.array(["TeamA"])
        away = np.array(["TeamB"])
        result = model.predict_btts(home, away)
        # Both should be well away from 0
        assert result[0, 0] > 0.1  # BTTS Yes
        assert result[0, 1] > 0.1  # BTTS No

    def test_single_match(self, fitted_dc_model):
        model, _ = fitted_dc_model
        result = model.predict_btts(np.array(["TeamA"]), np.array(["TeamB"]))
        assert result.shape == (1, 2)
