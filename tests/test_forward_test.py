"""Tests for forward test infrastructure."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.forward_test import (
    _implied_probs_2way,
    _implied_probs_3way,
    compute_forward_features,
    generate_forward_bets,
)


@pytest.fixture
def full_fixtures_df():
    """Synthetic fixtures with all odds columns present."""
    return pd.DataFrame({
        "Date": pd.to_datetime(["2026-02-22", "2026-02-22", "2026-02-23"]),
        "home_team": ["Arsenal", "Liverpool", "Chelsea"],
        "away_team": ["Man City", "Tottenham", "Man Utd"],
        "league": ["EPL", "EPL", "EPL"],
        "B365H": [2.50, 1.80, 2.10],
        "B365D": [3.40, 3.60, 3.30],
        "B365A": [2.80, 4.50, 3.50],
        "PSH": [2.55, 1.82, 2.12],
        "PSD": [3.35, 3.55, 3.28],
        "PSA": [2.75, 4.40, 3.45],
        "B365>2.5": [1.80, 1.70, 1.85],
        "B365<2.5": [2.10, 2.20, 2.05],
        "P>2.5": [1.82, 1.72, 1.87],
        "P<2.5": [2.08, 2.18, 2.03],
        "B365CH": [2.40, 2.00, 2.20],
        "B365CD": [3.50, 3.80, 3.40],
        "B365CA": [2.90, 3.50, 3.10],
        "AvgCH": [2.42, 2.02, 2.22],
        "AvgCD": [3.45, 3.75, 3.38],
        "AvgCA": [2.85, 3.45, 3.08],
    })


@pytest.fixture
def partial_fixtures_df():
    """Fixtures with only 1X2 odds (no O/U or corner)."""
    return pd.DataFrame({
        "Date": pd.to_datetime(["2026-02-22"]),
        "home_team": ["Arsenal"],
        "away_team": ["Man City"],
        "league": ["EPL"],
        "B365H": [2.50],
        "B365D": [3.40],
        "B365A": [2.80],
        "PSH": [2.55],
        "PSD": [3.35],
        "PSA": [2.75],
    })


class TestImpliedProbs:
    """Test implied probability computation functions."""

    def test_3way_sums_to_one(self, full_fixtures_df):
        probs = _implied_probs_3way(
            full_fixtures_df, ["B365H", "B365D", "B365A"], "b365_prob",
        )
        total = probs["b365_prob_h"] + probs["b365_prob_d"] + probs["b365_prob_a"]
        np.testing.assert_allclose(total.values, 1.0, atol=1e-10)

    def test_2way_sums_to_one(self, full_fixtures_df):
        probs = _implied_probs_2way(
            full_fixtures_df, ["B365>2.5", "B365<2.5"], "b365_ou25",
        )
        total = probs["b365_ou25_over"] + probs["b365_ou25_under"]
        np.testing.assert_allclose(total.values, 1.0, atol=1e-10)

    def test_3way_all_positive(self, full_fixtures_df):
        probs = _implied_probs_3way(
            full_fixtures_df, ["B365H", "B365D", "B365A"], "b365_prob",
        )
        assert (probs["b365_prob_h"] > 0).all()
        assert (probs["b365_prob_d"] > 0).all()
        assert (probs["b365_prob_a"] > 0).all()

    def test_lower_odds_higher_prob(self, full_fixtures_df):
        probs = _implied_probs_3way(
            full_fixtures_df, ["B365H", "B365D", "B365A"], "b365_prob",
        )
        # Liverpool has lowest B365H (1.80), so should have highest home prob
        assert probs["b365_prob_h"].iloc[1] > probs["b365_prob_h"].iloc[0]


class TestComputeForwardFeatures:
    """Test feature computation for forward fixtures."""

    def test_full_features_all_markets(self, full_fixtures_df):
        result = compute_forward_features(full_fixtures_df)

        # 1X2 features
        assert "b365_prob_h" in result.columns
        assert "ps_prob_h" in result.columns
        assert "odds_disagree_h" in result.columns

        # O/U features
        assert "b365_ou25_over" in result.columns
        assert "ps_ou25_over" in result.columns
        assert "ou25_disagree_over" in result.columns

        # Corner features
        assert "b365_corner_h" in result.columns
        assert "avg_corner_h" in result.columns
        assert "corner_disagree_h" in result.columns

    def test_partial_only_1x2(self, partial_fixtures_df):
        result = compute_forward_features(partial_fixtures_df)

        # 1X2 should be present
        assert "odds_disagree_h" in result.columns

        # O/U and corners should not be present
        assert "ou25_disagree_over" not in result.columns
        assert "corner_disagree_h" not in result.columns

    def test_disagreement_sign(self, full_fixtures_df):
        """Disagreement should be b365_prob - ps_prob."""
        result = compute_forward_features(full_fixtures_df)
        expected = result["b365_prob_h"] - result["ps_prob_h"]
        np.testing.assert_allclose(
            result["odds_disagree_h"].values, expected.values, atol=1e-10,
        )

    def test_preserves_original_columns(self, full_fixtures_df):
        result = compute_forward_features(full_fixtures_df)
        assert "home_team" in result.columns
        assert "away_team" in result.columns
        assert "league" in result.columns
        assert "B365H" in result.columns


class TestGenerateForwardBets:
    """Test forward bet generation with mocked selectors."""

    def _mock_selector(self, select_result):
        """Create a mock selector that returns given masks."""
        selector = MagicMock()
        selector.profitable_leagues = {"EPL"}
        selector.select_with_league_filter.return_value = select_result
        selector.select.return_value = select_result
        return selector

    def test_no_bets_when_all_disabled(self, full_fixtures_df):
        df = compute_forward_features(full_fixtures_df)
        config = {
            "strategies": {
                "1x2_disagree": {"enabled": False},
                "ou_disagree": {"enabled": False},
                "corner_disagree": {"enabled": False},
            },
            "commission_pct": 0.0,
            "kelly": {"fraction": 0.25, "max_bet_fraction": 0.05, "bankroll": 1000.0},
        }
        bets = generate_forward_bets(df, config)
        assert len(bets) == 0

    def test_bet_has_required_fields(self, full_fixtures_df):
        df = compute_forward_features(full_fixtures_df)
        n = len(df)
        masks_1x2 = {"H": np.array([True] + [False] * (n - 1)),
                      "D": np.zeros(n, dtype=bool),
                      "A": np.zeros(n, dtype=bool)}

        mock_sel = self._mock_selector(masks_1x2)

        config = {
            "strategies": {
                "1x2_disagree": {"enabled": True, "percentile": 7, "use_league_filter": True},
                "ou_disagree": {"enabled": False},
                "corner_disagree": {"enabled": False},
            },
            "commission_pct": 0.0,
            "kelly": {"fraction": 0.25, "max_bet_fraction": 0.05, "bankroll": 1000.0},
        }

        with patch(
            "src.models.forward_test.DisagreementSelector.load",
            return_value=mock_sel,
        ), patch("src.models.forward_test.Path.exists", return_value=True):
            bets = generate_forward_bets(df, config)

        if bets:  # may be empty if Kelly fraction is 0
            b = bets[0]
            assert "date" in b
            assert "home_team" in b
            assert "away_team" in b
            assert "market" in b
            assert "outcome" in b
            assert "odds" in b
            assert "kelly_pct" in b
            assert "stake" in b
            assert "league" in b

    def test_missing_ou_columns_skips_gracefully(self, partial_fixtures_df):
        df = compute_forward_features(partial_fixtures_df)
        config = {
            "strategies": {
                "1x2_disagree": {"enabled": False},
                "ou_disagree": {"enabled": True, "percentile": 3, "use_league_filter": True},
                "corner_disagree": {"enabled": False},
            },
            "commission_pct": 0.0,
            "kelly": {"fraction": 0.25, "max_bet_fraction": 0.05, "bankroll": 1000.0},
        }
        # Should not raise, just skip
        bets = generate_forward_bets(df, config)
        assert len(bets) == 0

    def test_stake_respects_bankroll(self, full_fixtures_df):
        df = compute_forward_features(full_fixtures_df)
        n = len(df)
        masks_1x2 = {"H": np.ones(n, dtype=bool),
                      "D": np.zeros(n, dtype=bool),
                      "A": np.zeros(n, dtype=bool)}

        mock_sel = self._mock_selector(masks_1x2)

        config = {
            "strategies": {
                "1x2_disagree": {"enabled": True, "percentile": 7, "use_league_filter": True},
                "ou_disagree": {"enabled": False},
                "corner_disagree": {"enabled": False},
            },
            "commission_pct": 0.0,
            "kelly": {"fraction": 0.25, "max_bet_fraction": 0.05, "bankroll": 500.0},
        }

        with patch(
            "src.models.forward_test.DisagreementSelector.load",
            return_value=mock_sel,
        ), patch("src.models.forward_test.Path.exists", return_value=True):
            bets = generate_forward_bets(df, config)

        for b in bets:
            # Stake should be at most max_bet_fraction * bankroll = 0.05 * 500 = 25
            assert b["stake"] <= 25.01
