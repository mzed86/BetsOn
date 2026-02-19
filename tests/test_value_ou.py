"""Tests for Over/Under 2.5 value betting pipeline components."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from src.evaluation.value_metrics import (
    compute_cost_sensitivity_generic,
    compute_flat_stake_roi_generic,
    compute_kelly_roi_generic,
    compute_per_season_roi_generic,
)
from src.models.value_betting import GenericDisagreementSelector


@pytest.fixture
def ou_synthetic_df():
    """Synthetic O/U data for testing."""
    rng = np.random.RandomState(42)
    n = 300

    df = pd.DataFrame({
        "ou25_disagree_over": rng.normal(-0.02, 0.05, n),
        "ou25_disagree_under": rng.normal(-0.02, 0.05, n),
        "ou25_result": rng.choice(["Over", "Under"], n, p=[0.48, 0.52]),
        "b365_ou25_over": 1.5 + rng.random(n) * 1.5,
        "b365_ou25_under": 1.5 + rng.random(n) * 1.5,
        "league": rng.choice(["EPL", "La_Liga", "Bundesliga"], n),
        "season": rng.choice([2022, 2023, 2024], n),
    })
    return df


class TestGenericDisagreementSelectorOU:
    """Test GenericDisagreementSelector with O/U outcomes."""

    def test_fit_learns_cutoffs(self, ou_synthetic_df):
        selector = GenericDisagreementSelector(
            outcomes=["Over", "Under"],
            outcome_suffixes={"Over": "over", "Under": "under"},
            config={"percentiles": [5, 10, 20], "default_percentile": 10},
            disagree_prefix="ou25_disagree",
        )
        selector.fit(ou_synthetic_df)

        assert selector.percentile_cutoffs is not None
        assert 5 in selector.percentile_cutoffs
        assert 10 in selector.percentile_cutoffs
        assert "over" in selector.percentile_cutoffs[5]
        assert "under" in selector.percentile_cutoffs[5]

    def test_cutoff_monotonicity(self, ou_synthetic_df):
        selector = GenericDisagreementSelector(
            outcomes=["Over", "Under"],
            outcome_suffixes={"Over": "over", "Under": "under"},
            config={"percentiles": [5, 10, 20], "default_percentile": 10},
            disagree_prefix="ou25_disagree",
        )
        selector.fit(ou_synthetic_df)

        # Higher percentile → higher (or equal) cutoff
        for suffix in ["over", "under"]:
            assert selector.percentile_cutoffs[5][suffix] <= selector.percentile_cutoffs[10][suffix]
            assert selector.percentile_cutoffs[10][suffix] <= selector.percentile_cutoffs[20][suffix]

    def test_select_returns_bool_masks(self, ou_synthetic_df):
        selector = GenericDisagreementSelector(
            outcomes=["Over", "Under"],
            outcome_suffixes={"Over": "over", "Under": "under"},
            config={"percentiles": [5, 10, 20], "default_percentile": 10},
            disagree_prefix="ou25_disagree",
        )
        selector.fit(ou_synthetic_df)
        masks = selector.select(ou_synthetic_df, percentile=10)

        assert "Over" in masks
        assert "Under" in masks
        assert masks["Over"].dtype == bool or np.issubdtype(masks["Over"].dtype, np.bool_)
        assert len(masks["Over"]) == len(ou_synthetic_df)

    def test_narrower_percentile_fewer_bets(self, ou_synthetic_df):
        selector = GenericDisagreementSelector(
            outcomes=["Over", "Under"],
            outcome_suffixes={"Over": "over", "Under": "under"},
            config={"percentiles": [5, 10, 20], "default_percentile": 10},
            disagree_prefix="ou25_disagree",
        )
        selector.fit(ou_synthetic_df)

        masks5 = selector.select(ou_synthetic_df, percentile=5)
        masks20 = selector.select(ou_synthetic_df, percentile=20)

        total5 = sum(m.sum() for m in masks5.values())
        total20 = sum(m.sum() for m in masks20.values())
        assert total5 <= total20

    def test_unfitted_raises(self, ou_synthetic_df):
        selector = GenericDisagreementSelector(
            outcomes=["Over", "Under"],
            outcome_suffixes={"Over": "over", "Under": "under"},
            config={},
            disagree_prefix="ou25_disagree",
        )
        with pytest.raises(RuntimeError):
            selector.select(ou_synthetic_df)

    def test_invalid_percentile_raises(self, ou_synthetic_df):
        selector = GenericDisagreementSelector(
            outcomes=["Over", "Under"],
            outcome_suffixes={"Over": "over", "Under": "under"},
            config={"percentiles": [5, 10], "default_percentile": 5},
            disagree_prefix="ou25_disagree",
        )
        selector.fit(ou_synthetic_df)
        with pytest.raises(ValueError):
            selector.select(ou_synthetic_df, percentile=99)

    def test_save_load_roundtrip(self, ou_synthetic_df):
        selector = GenericDisagreementSelector(
            outcomes=["Over", "Under"],
            outcome_suffixes={"Over": "over", "Under": "under"},
            config={"percentiles": [5, 10], "default_percentile": 5},
            disagree_prefix="ou25_disagree",
        )
        selector.fit(ou_synthetic_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "selector.pkl"
            selector.save(path)
            loaded = GenericDisagreementSelector.load(path)

            assert loaded.outcomes == selector.outcomes
            assert loaded.outcome_suffixes == selector.outcome_suffixes
            assert loaded.disagree_prefix == selector.disagree_prefix
            # Check cutoffs match
            for p in selector.percentiles:
                for suffix in ["over", "under"]:
                    assert abs(loaded.percentile_cutoffs[p][suffix]
                               - selector.percentile_cutoffs[p][suffix]) < 1e-10


class TestGenericROIFunctions:
    """Test generic ROI functions with O/U data."""

    def test_flat_stake_roi_basic(self, ou_synthetic_df):
        results = ou_synthetic_df["ou25_result"].values
        masks = {
            "Over": np.ones(len(ou_synthetic_df), dtype=bool),
            "Under": np.zeros(len(ou_synthetic_df), dtype=bool),
        }
        odds = {
            "Over": ou_synthetic_df["b365_ou25_over"].values,
            "Under": ou_synthetic_df["b365_ou25_under"].values,
        }
        roi = compute_flat_stake_roi_generic(results, masks, odds)
        assert "n_bets" in roi
        assert "roi_pct" in roi
        assert roi["n_bets"] > 0

    def test_flat_stake_roi_no_bets(self, ou_synthetic_df):
        results = ou_synthetic_df["ou25_result"].values
        masks = {
            "Over": np.zeros(len(ou_synthetic_df), dtype=bool),
            "Under": np.zeros(len(ou_synthetic_df), dtype=bool),
        }
        odds = {
            "Over": ou_synthetic_df["b365_ou25_over"].values,
            "Under": ou_synthetic_df["b365_ou25_under"].values,
        }
        roi = compute_flat_stake_roi_generic(results, masks, odds)
        assert roi["n_bets"] == 0
        assert roi["roi_pct"] == 0.0

    def test_per_season_roi(self, ou_synthetic_df):
        results = ou_synthetic_df["ou25_result"].values
        masks = {
            "Over": np.ones(len(ou_synthetic_df), dtype=bool),
            "Under": np.zeros(len(ou_synthetic_df), dtype=bool),
        }
        odds = {
            "Over": ou_synthetic_df["b365_ou25_over"].values,
            "Under": ou_synthetic_df["b365_ou25_under"].values,
        }
        season_df = compute_per_season_roi_generic(
            results, masks, odds, ou_synthetic_df["season"].values,
        )
        assert "season" in season_df.columns
        assert "n_bets" in season_df.columns
        assert "roi_pct" in season_df.columns
        assert len(season_df) == 3  # 3 unique seasons

    def test_cost_sensitivity(self, ou_synthetic_df):
        results = ou_synthetic_df["ou25_result"].values
        masks = {
            "Over": np.ones(len(ou_synthetic_df), dtype=bool),
            "Under": np.zeros(len(ou_synthetic_df), dtype=bool),
        }
        odds = {
            "Over": ou_synthetic_df["b365_ou25_over"].values,
            "Under": ou_synthetic_df["b365_ou25_under"].values,
        }
        cost_df = compute_cost_sensitivity_generic(
            results, {"test_strat": masks}, odds, [0.0, 2.0, 5.0],
        )
        assert "strategy" in cost_df.columns
        assert len(cost_df) == 3

    def test_kelly_roi(self, ou_synthetic_df):
        results = ou_synthetic_df["ou25_result"].values
        masks = {
            "Over": np.ones(len(ou_synthetic_df), dtype=bool),
            "Under": np.zeros(len(ou_synthetic_df), dtype=bool),
        }
        odds = {
            "Over": ou_synthetic_df["b365_ou25_over"].values,
            "Under": ou_synthetic_df["b365_ou25_under"].values,
        }
        probs = {
            "Over": np.full(len(ou_synthetic_df), 0.5),
            "Under": np.full(len(ou_synthetic_df), 0.5),
        }
        kelly = compute_kelly_roi_generic(results, masks, probs, odds)
        assert "n_bets" in kelly
        assert "ending_bankroll" in kelly
        assert "max_drawdown_pct" in kelly


class TestLeagueFilter:
    """Test league filter with O/U data."""

    def test_compute_league_filter(self, ou_synthetic_df):
        selector = GenericDisagreementSelector(
            outcomes=["Over", "Under"],
            outcome_suffixes={"Over": "over", "Under": "under"},
            config={"percentiles": [5, 10, 20], "default_percentile": 10},
            disagree_prefix="ou25_disagree",
        )
        selector.fit(ou_synthetic_df)

        odds = {
            "Over": ou_synthetic_df["b365_ou25_over"].values,
            "Under": ou_synthetic_df["b365_ou25_under"].values,
        }
        leagues = selector.compute_league_filter(
            ou_synthetic_df, ou_synthetic_df["ou25_result"].values,
            odds, ou_synthetic_df["league"].values,
            percentile=10, min_roi_pct=-100, min_bets=1,
        )
        assert isinstance(leagues, set)

    def test_select_with_league_filter(self, ou_synthetic_df):
        selector = GenericDisagreementSelector(
            outcomes=["Over", "Under"],
            outcome_suffixes={"Over": "over", "Under": "under"},
            config={"percentiles": [10], "default_percentile": 10},
            disagree_prefix="ou25_disagree",
        )
        selector.fit(ou_synthetic_df)

        masks_all = selector.select(ou_synthetic_df, percentile=10)
        masks_filt = selector.select_with_league_filter(
            ou_synthetic_df, ou_synthetic_df["league"].values,
            percentile=10, allowed_leagues={"EPL"},
        )

        # Filtered should have fewer or equal bets
        total_all = sum(m.sum() for m in masks_all.values())
        total_filt = sum(m.sum() for m in masks_filt.values())
        assert total_filt <= total_all
