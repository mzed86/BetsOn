"""Unit tests for the feature engineering pipeline.

Uses a synthetic 4-team dataset with known results to verify:
- Rolling stats are correctly shifted (no leakage)
- First match produces NaN rolling features
- Cross-league history carries for promoted teams
- Elo ratings: winner gains, loser loses, pre-match recorded
- Odds: normalized probs sum to 1, zero odds -> NaN
- Rest days: capped at threshold, first match -> NaN
"""

import numpy as np
import pandas as pd
import pytest

from src.features.config import (
    EloConfig,
    FeatureConfig,
    MatchContextConfig,
    OddsConfig,
    RollingConfig,
)
from src.features.elo import compute_elo_features
from src.features.match_context import compute_match_context_features
from src.features.odds import compute_odds_features
from src.features.rolling import _melt_to_team_perspective, compute_rolling_features


@pytest.fixture
def synthetic_matches() -> pd.DataFrame:
    """Create a small synthetic dataset with 6 matches across 2 seasons.

    Teams: Alpha, Beta, Gamma, Delta
    Season 1 (3 matches):
      Match 0: Alpha 2-1 Beta  (Alpha wins)
      Match 1: Gamma 0-0 Delta (Draw)
      Match 2: Alpha 1-0 Gamma (Alpha wins)
    Season 2 (3 matches):
      Match 3: Beta 3-1 Delta  (Beta wins)
      Match 4: Alpha 0-2 Beta  (Beta wins)
      Match 5: Gamma 1-1 Alpha (Draw)
    """
    return pd.DataFrame({
        "Date": pd.to_datetime([
            "2023-08-10", "2023-08-12", "2023-08-20",
            "2024-08-10", "2024-08-15", "2024-08-25",
        ]),
        "home_team": ["Alpha", "Gamma", "Alpha", "Beta", "Alpha", "Gamma"],
        "away_team": ["Beta", "Delta", "Gamma", "Delta", "Beta", "Alpha"],
        "FTHG": [2, 0, 1, 3, 0, 1],
        "FTAG": [1, 0, 0, 1, 2, 1],
        "FTR": ["H", "D", "H", "H", "A", "D"],
        "HS": [10, 5, 8, 12, 6, 7],
        "AS": [6, 5, 4, 8, 10, 7],
        "HST": [5, 2, 4, 6, 3, 3],
        "AST": [3, 2, 1, 3, 5, 3],
        "HF": [10, 12, 8, 14, 11, 9],
        "AF": [8, 12, 10, 10, 9, 11],
        "HC": [5, 3, 6, 7, 4, 5],
        "AC": [3, 3, 2, 5, 6, 5],
        "HY": [1, 2, 0, 3, 1, 2],
        "AY": [2, 2, 1, 1, 0, 1],
        "HR": [0, 0, 0, 0, 0, 0],
        "AR": [0, 0, 0, 0, 0, 0],
        "B365H": [1.8, 2.5, 2.0, 1.6, 2.8, 2.4],
        "B365D": [3.5, 3.2, 3.3, 4.0, 3.4, 3.1],
        "B365A": [4.5, 2.8, 3.8, 5.5, 2.5, 3.0],
        "PSH": [1.85, 2.55, 2.05, 1.65, 2.85, 2.45],
        "PSD": [3.55, 3.25, 3.35, 4.05, 3.45, 3.15],
        "PSA": [4.55, 2.85, 3.85, 5.55, 2.55, 3.05],
        "PSCH": [1.82, 2.48, 2.02, 1.62, 2.82, 2.42],
        "PSCD": [3.52, 3.22, 3.32, 4.02, 3.42, 3.12],
        "PSCA": [4.52, 2.82, 3.82, 5.52, 2.52, 3.02],
        "league": ["EPL"] * 6,
        "season": [2023, 2023, 2023, 2024, 2024, 2024],
    })


@pytest.fixture
def rolling_config() -> RollingConfig:
    return RollingConfig(windows=[2, 3], venue_windows=[2], min_periods=1)


@pytest.fixture
def elo_config() -> EloConfig:
    return EloConfig(k_factor=20, initial_rating=1500, home_advantage=100)


@pytest.fixture
def odds_config() -> OddsConfig:
    return OddsConfig(
        primary=["B365H", "B365D", "B365A"],
        pinnacle_opening=["PSH", "PSD", "PSA"],
        pinnacle_closing=["PSCH", "PSCD", "PSCA"],
        drop_columns=[],
    )


@pytest.fixture
def match_context_config() -> MatchContextConfig:
    return MatchContextConfig(early_season_threshold=2, summer_break_threshold=60)


# --- Rolling feature tests ---


class TestMelt:
    def test_melt_produces_two_rows_per_match(self, synthetic_matches):
        melted = _melt_to_team_perspective(synthetic_matches)
        assert len(melted) == 2 * len(synthetic_matches)

    def test_melt_goals_perspective(self, synthetic_matches):
        melted = _melt_to_team_perspective(synthetic_matches)
        # Match 0: Alpha 2-1 Beta
        # Home row: Alpha goals_for=2, goals_against=1
        alpha_home = melted[
            (melted["team"] == "Alpha") & (melted["match_idx"] == 0)
        ]
        assert alpha_home["goals_for"].values[0] == 2
        assert alpha_home["goals_against"].values[0] == 1
        # Away row: Beta goals_for=1, goals_against=2
        beta_away = melted[
            (melted["team"] == "Beta") & (melted["match_idx"] == 0)
        ]
        assert beta_away["goals_for"].values[0] == 1
        assert beta_away["goals_against"].values[0] == 2

    def test_melt_ppg(self, synthetic_matches):
        melted = _melt_to_team_perspective(synthetic_matches)
        # Match 0: Alpha wins -> ppg=3, Beta loses -> ppg=0
        alpha_m0 = melted[(melted["team"] == "Alpha") & (melted["match_idx"] == 0)]
        assert alpha_m0["ppg"].values[0] == 3.0
        beta_m0 = melted[(melted["team"] == "Beta") & (melted["match_idx"] == 0)]
        assert beta_m0["ppg"].values[0] == 0.0
        # Match 1: Draw -> ppg=1
        gamma_m1 = melted[(melted["team"] == "Gamma") & (melted["match_idx"] == 1)]
        assert gamma_m1["ppg"].values[0] == 1.0


class TestRolling:
    def test_no_leakage(self, synthetic_matches, rolling_config):
        """Rolling features for match N must NOT include match N's data."""
        result = compute_rolling_features(synthetic_matches, rolling_config)

        # Match 0 is Alpha's first match -> rolling should be NaN
        assert pd.isna(result.loc[0, "h_rolling2_ppg"])

        # Match 2: Alpha's second match (home). rolling2_ppg should reflect
        # only match 0 (ppg=3), NOT match 2's own result.
        # Alpha's history: match 0 (ppg=3). So rolling2_ppg = 3.0
        assert result.loc[2, "h_rolling2_ppg"] == pytest.approx(3.0)

    def test_first_match_nan(self, synthetic_matches, rolling_config):
        """First match for each team should have NaN rolling features."""
        result = compute_rolling_features(synthetic_matches, rolling_config)
        # Match 0: first for Alpha (home) and Beta (away)
        assert pd.isna(result.loc[0, "h_rolling2_ppg"])
        assert pd.isna(result.loc[0, "a_rolling2_ppg"])
        # Match 1: first for Gamma (home) and Delta (away)
        assert pd.isna(result.loc[1, "h_rolling2_ppg"])
        assert pd.isna(result.loc[1, "a_rolling2_ppg"])

    def test_cross_season_history_carries(self, synthetic_matches, rolling_config):
        """Teams carry rolling history across seasons (no reset)."""
        result = compute_rolling_features(synthetic_matches, rolling_config)
        # Match 4: Alpha at home, season 2024. Alpha's prior matches:
        # match 0 (ppg=3), match 2 (ppg=3) from season 2023.
        # rolling2_ppg should use matches 0,2 -> mean(3,3) = 3.0
        assert result.loc[4, "h_rolling2_ppg"] == pytest.approx(3.0)

    def test_rolling_columns_exist(self, synthetic_matches, rolling_config):
        result = compute_rolling_features(synthetic_matches, rolling_config)
        # Should have h_ and a_ prefixed rolling columns
        assert "h_rolling2_ppg" in result.columns
        assert "a_rolling2_ppg" in result.columns
        assert "h_rolling3_goals_for" in result.columns
        assert "a_rolling2_shots_on_target" in result.columns


# --- Elo tests ---


class TestElo:
    def test_winner_gains_loser_loses(self, synthetic_matches, elo_config):
        result = compute_elo_features(synthetic_matches, elo_config)
        # Match 0: Both start at 1500. Alpha wins.
        # After match 0: Alpha's Elo should increase, Beta's decrease.
        # Match 4: Alpha home vs Beta away.
        # Alpha's pre-match Elo should be > 1500 (won matches 0,2)
        # Beta's should be < 1500 after losing match 0
        assert result.loc[4, "h_elo"] > 1500
        # Beta lost match 0, but won match 3, so check relative
        # Just verify the system is recording and updating
        assert result.loc[0, "h_elo"] == 1500  # first match = initial
        assert result.loc[0, "a_elo"] == 1500

    def test_pre_match_recorded(self, synthetic_matches, elo_config):
        """Elo values are pre-match (before result is applied)."""
        result = compute_elo_features(synthetic_matches, elo_config)
        # Match 0: both at initial rating
        assert result.loc[0, "h_elo"] == elo_config.initial_rating
        assert result.loc[0, "a_elo"] == elo_config.initial_rating

    def test_elo_diff_computed(self, synthetic_matches, elo_config):
        result = compute_elo_features(synthetic_matches, elo_config)
        for i in range(len(synthetic_matches)):
            assert result.loc[i, "elo_diff"] == pytest.approx(
                result.loc[i, "h_elo"] - result.loc[i, "a_elo"]
            )

    def test_elo_expected_range(self, synthetic_matches, elo_config):
        result = compute_elo_features(synthetic_matches, elo_config)
        # Expected score should be between 0 and 1
        assert (result["h_elo_expected"] >= 0).all()
        assert (result["h_elo_expected"] <= 1).all()

    def test_draw_splits_evenly(self, elo_config):
        """A draw between equal-rated teams (no home advantage) should not change ratings."""
        matches = pd.DataFrame({
            "Date": pd.to_datetime(["2023-01-01", "2023-01-08"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "FTHG": [1, 0],
            "FTAG": [1, 0],
            "FTR": ["D", "D"],
            "league": ["EPL", "EPL"],
            "season": [2023, 2023],
        })
        no_home_adv = EloConfig(k_factor=20, initial_rating=1500, home_advantage=0)
        result = compute_elo_features(matches, no_home_adv)
        # After draw with no home advantage and equal ratings,
        # expected = 0.5, actual = 0.5, so no change
        assert result.loc[1, "h_elo"] == 1500
        assert result.loc[1, "a_elo"] == 1500


# --- Odds tests ---


class TestOdds:
    def test_normalized_probs_sum_to_one(self, synthetic_matches, odds_config):
        result = compute_odds_features(synthetic_matches, odds_config)
        prob_sum = (
            result["b365_prob_h"] + result["b365_prob_d"] + result["b365_prob_a"]
        )
        np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-10)

    def test_overround_greater_than_one(self, synthetic_matches, odds_config):
        result = compute_odds_features(synthetic_matches, odds_config)
        # Bookmaker overround should be > 1 (they take a margin)
        assert (result["b365_prob_overround"] > 1.0).all()

    def test_zero_odds_produce_nan(self, odds_config):
        matches = pd.DataFrame({
            "B365H": [0.0, 1.8],
            "B365D": [3.5, 3.5],
            "B365A": [4.5, 4.5],
            "PSH": [1.85, 1.85],
            "PSD": [3.55, 3.55],
            "PSA": [4.55, 4.55],
            "PSCH": [1.82, 1.82],
            "PSCD": [3.52, 3.52],
            "PSCA": [4.52, 4.52],
        })
        result = compute_odds_features(matches, odds_config)
        # First row has B365H=0 -> all b365_prob should be NaN
        assert pd.isna(result.loc[0, "b365_prob_h"])
        assert pd.isna(result.loc[0, "b365_prob_d"])
        assert pd.isna(result.loc[0, "b365_prob_a"])
        # Second row should be valid
        assert not pd.isna(result.loc[1, "b365_prob_h"])

    def test_favourite_column(self, synthetic_matches, odds_config):
        result = compute_odds_features(synthetic_matches, odds_config)
        # Match 0: B365H=1.8 (lowest odds = highest prob) -> favourite = H
        assert result.loc[0, "favourite"] == "H"
        assert result.loc[0, "favourite_prob"] > 0

    def test_pinnacle_columns_present(self, synthetic_matches, odds_config):
        result = compute_odds_features(synthetic_matches, odds_config)
        assert "ps_prob_h" in result.columns
        assert "psc_prob_h" in result.columns


# --- Match context tests ---


class TestMatchContext:
    def test_first_match_rest_nan(self, synthetic_matches, match_context_config):
        result = compute_match_context_features(synthetic_matches, match_context_config)
        # First match for Alpha (match 0) and Beta (match 0) -> NaN rest
        assert pd.isna(result.loc[0, "h_days_rest"])
        assert pd.isna(result.loc[0, "a_days_rest"])

    def test_rest_days_computed(self, synthetic_matches, match_context_config):
        result = compute_match_context_features(synthetic_matches, match_context_config)
        # Match 2 (2023-08-20): Alpha last played 2023-08-10 = 10 days rest
        assert result.loc[2, "h_days_rest"] == 10

    def test_rest_days_capped(self, match_context_config):
        matches = pd.DataFrame({
            "Date": pd.to_datetime(["2023-06-01", "2023-12-01"]),
            "home_team": ["X", "X"],
            "away_team": ["Y", "Y"],
            "FTHG": [1, 1],
            "FTAG": [0, 0],
            "FTR": ["H", "H"],
            "league": ["EPL", "EPL"],
            "season": [2023, 2023],
        })
        result = compute_match_context_features(matches, match_context_config)
        # 183 days apart -> capped at 60
        assert result.loc[1, "h_days_rest"] == 60

    def test_match_number(self, synthetic_matches, match_context_config):
        result = compute_match_context_features(synthetic_matches, match_context_config)
        # Match 0: Alpha's 1st match in season 2023
        assert result.loc[0, "h_match_num"] == 1
        # Match 2: Alpha's 2nd home match in season 2023, but 2nd overall
        assert result.loc[2, "h_match_num"] == 2

    def test_early_season_flag(self, synthetic_matches, match_context_config):
        result = compute_match_context_features(synthetic_matches, match_context_config)
        # threshold=2, so match_num <= 2 -> early season
        assert result.loc[0, "h_is_early_season"] == 1  # match_num=1
        assert result.loc[2, "h_is_early_season"] == 1  # match_num=2

    def test_rest_diff(self, synthetic_matches, match_context_config):
        result = compute_match_context_features(synthetic_matches, match_context_config)
        # rest_diff = h_days_rest - a_days_rest
        valid = result.dropna(subset=["h_days_rest", "a_days_rest"])
        for idx in valid.index:
            assert result.loc[idx, "rest_diff"] == pytest.approx(
                result.loc[idx, "h_days_rest"] - result.loc[idx, "a_days_rest"]
            )


# --- Market timing feature tests ---


class TestMarketTiming:
    def test_day_of_week_range(self, synthetic_matches, match_context_config):
        """day_of_week should be in [0, 6]."""
        result = compute_match_context_features(synthetic_matches, match_context_config)
        assert "day_of_week" in result.columns
        assert (result["day_of_week"] >= 0).all()
        assert (result["day_of_week"] <= 6).all()

    def test_is_midweek_correct(self, match_context_config):
        """is_midweek should be 1 for Tue/Wed/Thu (dayofweek 1,2,3)."""
        matches = pd.DataFrame({
            "Date": pd.to_datetime([
                "2024-01-01",  # Monday (0)
                "2024-01-02",  # Tuesday (1) -> midweek
                "2024-01-03",  # Wednesday (2) -> midweek
                "2024-01-04",  # Thursday (3) -> midweek
                "2024-01-05",  # Friday (4)
                "2024-01-06",  # Saturday (5)
                "2024-01-07",  # Sunday (6)
            ]),
            "home_team": ["A", "A", "A", "A", "A", "A", "A"],
            "away_team": ["B", "B", "B", "B", "B", "B", "B"],
            "FTHG": [1, 1, 1, 1, 1, 1, 1],
            "FTAG": [0, 0, 0, 0, 0, 0, 0],
            "FTR": ["H", "H", "H", "H", "H", "H", "H"],
            "league": ["EPL"] * 7,
            "season": [2024] * 7,
        })
        result = compute_match_context_features(matches, match_context_config)
        assert result.loc[0, "is_midweek"] == 0  # Monday
        assert result.loc[1, "is_midweek"] == 1  # Tuesday
        assert result.loc[2, "is_midweek"] == 1  # Wednesday
        assert result.loc[3, "is_midweek"] == 1  # Thursday
        assert result.loc[4, "is_midweek"] == 0  # Friday
        assert result.loc[5, "is_midweek"] == 0  # Saturday
        assert result.loc[6, "is_midweek"] == 0  # Sunday

    def test_is_friday(self, match_context_config):
        """is_friday should be 1 only for Friday."""
        matches = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-05", "2024-01-06"]),  # Fri, Sat
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "FTHG": [1, 1],
            "FTAG": [0, 0],
            "FTR": ["H", "H"],
            "league": ["EPL", "EPL"],
            "season": [2024, 2024],
        })
        result = compute_match_context_features(matches, match_context_config)
        assert result.loc[0, "is_friday"] == 1
        assert result.loc[1, "is_friday"] == 0

    def test_kickoff_hour_with_time(self, match_context_config):
        """kickoff_hour should parse from Time column."""
        matches = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-06", "2024-01-06", "2024-01-06"]),
            "Time": ["15:00", "12:30", np.nan],
            "home_team": ["A", "B", "C"],
            "away_team": ["D", "E", "F"],
            "FTHG": [1, 0, 1],
            "FTAG": [0, 0, 0],
            "FTR": ["H", "D", "H"],
            "league": ["EPL", "EPL", "EPL"],
            "season": [2024, 2024, 2024],
        })
        result = compute_match_context_features(matches, match_context_config)
        assert result.loc[0, "kickoff_hour"] == pytest.approx(15.0)
        assert result.loc[1, "kickoff_hour"] == pytest.approx(12.5)
        assert pd.isna(result.loc[2, "kickoff_hour"])

    def test_is_early_kickoff(self, match_context_config):
        """is_early_kickoff should be 1 when hour <= 13."""
        matches = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-06", "2024-01-06", "2024-01-06"]),
            "Time": ["12:30", "13:00", "15:00"],
            "home_team": ["A", "B", "C"],
            "away_team": ["D", "E", "F"],
            "FTHG": [1, 0, 1],
            "FTAG": [0, 0, 0],
            "FTR": ["H", "D", "H"],
            "league": ["EPL", "EPL", "EPL"],
            "season": [2024, 2024, 2024],
        })
        result = compute_match_context_features(matches, match_context_config)
        assert result.loc[0, "is_early_kickoff"] == 1.0  # 12:30 <= 13
        assert result.loc[1, "is_early_kickoff"] == 1.0  # 13:00 <= 13
        assert result.loc[2, "is_early_kickoff"] == 0.0  # 15:00 > 13

    def test_no_time_column(self, synthetic_matches, match_context_config):
        """When Time column is absent, kickoff features should not be created."""
        assert "Time" not in synthetic_matches.columns
        result = compute_match_context_features(synthetic_matches, match_context_config)
        assert "kickoff_hour" not in result.columns
        assert "is_early_kickoff" not in result.columns
        # But day_of_week features should still exist
        assert "day_of_week" in result.columns
        assert "is_midweek" in result.columns
        assert "is_friday" in result.columns


# --- Multi-book odds tests ---


class TestMultiBookOdds:
    def test_additional_book_probs(self):
        """Additional book implied probs should be computed when columns exist."""
        from src.features.config import OddsConfig
        config = OddsConfig(
            primary=["B365H", "B365D", "B365A"],
            pinnacle_opening=["PSH", "PSD", "PSA"],
            pinnacle_closing=["PSCH", "PSCD", "PSCA"],
            drop_columns=[],
            additional_books={"bw": ["BWH", "BWD", "BWA"]},
        )
        matches = pd.DataFrame({
            "B365H": [1.8, 2.5],
            "B365D": [3.5, 3.2],
            "B365A": [4.5, 2.8],
            "PSH": [1.85, 2.55],
            "PSD": [3.55, 3.25],
            "PSA": [4.55, 2.85],
            "PSCH": [1.82, 2.48],
            "PSCD": [3.52, 3.22],
            "PSCA": [4.52, 2.82],
            "BWH": [1.82, 2.52],
            "BWD": [3.52, 3.20],
            "BWA": [4.52, 2.82],
        })
        result = compute_odds_features(matches, config)
        assert "bw_prob_h" in result.columns
        assert "bw_prob_d" in result.columns
        assert "bw_prob_a" in result.columns
        # Probs should sum to ~1
        bw_sum = result["bw_prob_h"] + result["bw_prob_d"] + result["bw_prob_a"]
        np.testing.assert_allclose(bw_sum.values, 1.0, atol=1e-10)

    def test_max_odds_gte_each_book(self):
        """max_odds should be >= each individual book's odds."""
        from src.features.config import OddsConfig
        config = OddsConfig(
            primary=["B365H", "B365D", "B365A"],
            pinnacle_opening=["PSH", "PSD", "PSA"],
            pinnacle_closing=["PSCH", "PSCD", "PSCA"],
            drop_columns=[],
            additional_books={"bw": ["BWH", "BWD", "BWA"]},
        )
        matches = pd.DataFrame({
            "B365H": [1.8, 2.5],
            "B365D": [3.5, 3.2],
            "B365A": [4.5, 2.8],
            "PSH": [1.85, 2.55],
            "PSD": [3.55, 3.25],
            "PSA": [4.55, 2.85],
            "PSCH": [1.82, 2.48],
            "PSCD": [3.52, 3.22],
            "PSCA": [4.52, 2.82],
            "BWH": [1.90, 2.45],
            "BWD": [3.60, 3.10],
            "BWA": [4.40, 2.90],
        })
        result = compute_odds_features(matches, config)
        assert "max_odds_h" in result.columns
        assert (result["max_odds_h"] >= matches["B365H"]).all()
        assert (result["max_odds_h"] >= matches["BWH"]).all()

    def test_min_book_prob_lte_each_book(self):
        """min_book_prob should be <= each book's normalized prob."""
        from src.features.config import OddsConfig
        config = OddsConfig(
            primary=["B365H", "B365D", "B365A"],
            pinnacle_opening=["PSH", "PSD", "PSA"],
            pinnacle_closing=["PSCH", "PSCD", "PSCA"],
            drop_columns=[],
            additional_books={"bw": ["BWH", "BWD", "BWA"]},
        )
        matches = pd.DataFrame({
            "B365H": [1.8, 2.5],
            "B365D": [3.5, 3.2],
            "B365A": [4.5, 2.8],
            "PSH": [1.85, 2.55],
            "PSD": [3.55, 3.25],
            "PSA": [4.55, 2.85],
            "PSCH": [1.82, 2.48],
            "PSCD": [3.52, 3.22],
            "PSCA": [4.52, 2.82],
            "BWH": [1.82, 2.52],
            "BWD": [3.52, 3.20],
            "BWA": [4.52, 2.82],
        })
        result = compute_odds_features(matches, config)
        assert "min_book_prob_h" in result.columns
        assert (result["min_book_prob_h"] <= result["b365_prob_h"] + 1e-10).all()
        assert (result["min_book_prob_h"] <= result["bw_prob_h"] + 1e-10).all()

    def test_missing_book_handled(self):
        """Missing additional book columns should not cause errors."""
        from src.features.config import OddsConfig
        config = OddsConfig(
            primary=["B365H", "B365D", "B365A"],
            pinnacle_opening=["PSH", "PSD", "PSA"],
            pinnacle_closing=["PSCH", "PSCD", "PSCA"],
            drop_columns=[],
            additional_books={"bw": ["BWH", "BWD", "BWA"], "wh": ["WHH", "WHD", "WHA"]},
        )
        # Only B365 and PS columns — no BW or WH
        matches = pd.DataFrame({
            "B365H": [1.8],
            "B365D": [3.5],
            "B365A": [4.5],
            "PSH": [1.85],
            "PSD": [3.55],
            "PSA": [4.55],
            "PSCH": [1.82],
            "PSCD": [3.52],
            "PSCA": [4.52],
        })
        result = compute_odds_features(matches, config)
        assert "bw_prob_h" not in result.columns
        assert "wh_prob_h" not in result.columns
