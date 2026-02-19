"""Tests for portfolio revenue simulation."""

import numpy as np
import pandas as pd
import pytest

from src.models.portfolio import (
    compute_revenue_projections,
    simulate_portfolio_kelly,
)


@pytest.fixture
def sample_bets():
    """Synthetic bet stream for testing."""
    dates = pd.date_range("2023-01-01", periods=10, freq="7D")
    return [
        {
            "date": dates[0],
            "match_idx": 0,
            "home_team": "TeamA",
            "away_team": "TeamB",
            "market": "1X2",
            "outcome": "H",
            "odds": 2.50,
            "result": "H",
            "estimated_prob": 0.45,
            "season": 2023,
            "league": "EPL",
        },
        {
            "date": dates[0],
            "match_idx": 1,
            "home_team": "TeamC",
            "away_team": "TeamD",
            "market": "O/U",
            "outcome": "Over",
            "odds": 1.90,
            "result": "Under",
            "estimated_prob": 0.55,
            "season": 2023,
            "league": "EPL",
        },
        {
            "date": dates[1],
            "match_idx": 2,
            "home_team": "TeamE",
            "away_team": "TeamF",
            "market": "Corners",
            "outcome": "H",
            "odds": 2.80,
            "result": "H",
            "estimated_prob": 0.40,
            "season": 2023,
            "league": "La_Liga",
        },
        {
            "date": dates[2],
            "match_idx": 3,
            "home_team": "TeamG",
            "away_team": "TeamH",
            "market": "1X2",
            "outcome": "D",
            "odds": 3.40,
            "result": "H",
            "estimated_prob": 0.35,
            "season": 2023,
            "league": "EPL",
        },
        {
            "date": dates[3],
            "match_idx": 4,
            "home_team": "TeamI",
            "away_team": "TeamJ",
            "market": "O/U",
            "outcome": "Under",
            "odds": 2.10,
            "result": "Under",
            "estimated_prob": 0.50,
            "season": 2023,
            "league": "Bundesliga",
        },
    ]


class TestSimulatePortfolioKelly:
    """Test the portfolio Kelly simulation."""

    def test_empty_bets(self):
        result = simulate_portfolio_kelly([])
        assert result["summary"]["n_bets"] == 0
        assert result["summary"]["ending_bankroll"] == 1000.0
        assert result["bet_log"] == []
        assert result["per_market"] == {}

    def test_returns_expected_keys(self, sample_bets):
        result = simulate_portfolio_kelly(sample_bets)
        assert "summary" in result
        assert "bet_log" in result
        assert "bankroll_curve" in result
        assert "per_market" in result

    def test_summary_fields(self, sample_bets):
        result = simulate_portfolio_kelly(sample_bets)
        s = result["summary"]
        assert "n_bets" in s
        assert "starting_bankroll" in s
        assert "ending_bankroll" in s
        assert "profit" in s
        assert "roi_pct" in s
        assert "max_drawdown_pct" in s
        assert "total_staked" in s
        assert "months" in s
        assert "bets_per_month" in s

    def test_profit_consistency(self, sample_bets):
        result = simulate_portfolio_kelly(sample_bets, bankroll=1000.0)
        s = result["summary"]
        assert abs(s["profit"] - (s["ending_bankroll"] - s["starting_bankroll"])) < 0.01

    def test_bet_log_records_all_placed_bets(self, sample_bets):
        result = simulate_portfolio_kelly(sample_bets, commission_pct=0.0)
        s = result["summary"]
        assert len(result["bet_log"]) == s["n_bets"]

    def test_per_market_breakdown(self, sample_bets):
        result = simulate_portfolio_kelly(sample_bets, commission_pct=0.0)
        pm = result["per_market"]
        total_from_markets = sum(m["n_bets"] for m in pm.values())
        assert total_from_markets == result["summary"]["n_bets"]

    def test_bankroll_curve_monotonic_dates(self, sample_bets):
        result = simulate_portfolio_kelly(sample_bets)
        curve = result["bankroll_curve"]
        dates = [c["date"] for c in curve]
        for i in range(1, len(dates)):
            assert dates[i] >= dates[i - 1]

    def test_commission_reduces_profit(self, sample_bets):
        r0 = simulate_portfolio_kelly(sample_bets, commission_pct=0.0)
        r5 = simulate_portfolio_kelly(sample_bets, commission_pct=5.0)
        # With commission, less profit or more loss
        assert r5["summary"]["ending_bankroll"] <= r0["summary"]["ending_bankroll"]

    def test_same_date_bets_use_same_bankroll(self):
        """All bets on the same date should use pre-date bankroll."""
        bets = [
            {
                "date": pd.Timestamp("2023-01-01"),
                "match_idx": 0,
                "home_team": "A",
                "away_team": "B",
                "market": "1X2",
                "outcome": "H",
                "odds": 2.0,
                "result": "H",
                "estimated_prob": 0.60,
                "season": 2023,
                "league": "EPL",
            },
            {
                "date": pd.Timestamp("2023-01-01"),
                "match_idx": 1,
                "home_team": "C",
                "away_team": "D",
                "market": "1X2",
                "outcome": "H",
                "odds": 2.0,
                "result": "H",
                "estimated_prob": 0.60,
                "season": 2023,
                "league": "EPL",
            },
        ]
        result = simulate_portfolio_kelly(bets, commission_pct=0.0, bankroll=1000.0)
        log = result["bet_log"]
        # Both bets placed, both with same stake (same pre-date bankroll)
        assert len(log) == 2
        assert log[0]["stake"] == log[1]["stake"]

    def test_kelly_fraction_scales_stakes(self, sample_bets):
        r_quarter = simulate_portfolio_kelly(sample_bets, kelly_fraction=0.25, commission_pct=0.0)
        r_half = simulate_portfolio_kelly(sample_bets, kelly_fraction=0.5, commission_pct=0.0)
        # Half Kelly stakes more
        assert r_half["summary"]["total_staked"] > r_quarter["summary"]["total_staked"]

    def test_max_bet_fraction_caps_stake(self):
        """A bet with very high Kelly should be capped at max_bet_fraction."""
        bets = [
            {
                "date": pd.Timestamp("2023-01-01"),
                "match_idx": 0,
                "home_team": "A",
                "away_team": "B",
                "market": "1X2",
                "outcome": "H",
                "odds": 10.0,
                "result": "H",
                "estimated_prob": 0.90,
                "season": 2023,
                "league": "EPL",
            },
        ]
        result = simulate_portfolio_kelly(
            bets, commission_pct=0.0, kelly_fraction=1.0,
            max_bet_fraction=0.05, bankroll=1000.0,
        )
        log = result["bet_log"]
        assert len(log) == 1
        # Stake should be at most 5% of 1000 = 50
        assert log[0]["stake"] <= 50.01


class TestRevenueProjections:
    """Test revenue projection scaling."""

    def test_basic_projection(self):
        summary = {
            "starting_bankroll": 1000.0,
            "ending_bankroll": 1100.0,
            "months": 12.0,
        }
        proj = compute_revenue_projections(summary, [1000, 5000])
        assert len(proj) == 2
        assert proj.iloc[0]["bankroll"] == 1000
        assert proj.iloc[0]["profit"] == pytest.approx(100.0, abs=0.01)
        assert proj.iloc[1]["bankroll"] == 5000
        assert proj.iloc[1]["profit"] == pytest.approx(500.0, abs=0.01)

    def test_zero_starting_bankroll(self):
        summary = {
            "starting_bankroll": 0,
            "ending_bankroll": 0,
            "months": 12.0,
        }
        proj = compute_revenue_projections(summary, [1000])
        assert len(proj) == 0

    def test_monthly_yearly_relationship(self):
        summary = {
            "starting_bankroll": 1000.0,
            "ending_bankroll": 1200.0,
            "months": 6.0,
        }
        proj = compute_revenue_projections(summary, [1000])
        row = proj.iloc[0]
        assert row["yearly"] == pytest.approx(row["monthly"] * 12, abs=0.1)

    def test_scaling_is_linear(self):
        summary = {
            "starting_bankroll": 1000.0,
            "ending_bankroll": 1100.0,
            "months": 12.0,
        }
        proj = compute_revenue_projections(summary, [1000, 10000])
        # 10x bankroll should give 10x profit
        assert proj.iloc[1]["profit"] == pytest.approx(proj.iloc[0]["profit"] * 10, abs=0.01)
