"""Tests for corner match result value betting pipeline components."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from src.evaluation.value_metrics import compute_flat_stake_roi_generic
from src.models.value_betting import GenericDisagreementSelector


@pytest.fixture
def corner_synthetic_df():
    """Synthetic corner data for testing."""
    rng = np.random.RandomState(42)
    n = 300

    df = pd.DataFrame({
        "corner_disagree_h": rng.normal(-0.02, 0.05, n),
        "corner_disagree_d": rng.normal(-0.02, 0.05, n),
        "corner_disagree_a": rng.normal(-0.02, 0.05, n),
        "corner_ftr": rng.choice(["H", "D", "A"], n, p=[0.42, 0.22, 0.36]),
        "b365_corner_h": 1.5 + rng.random(n) * 2.0,
        "b365_corner_d": 2.5 + rng.random(n) * 2.0,
        "b365_corner_a": 1.5 + rng.random(n) * 2.0,
        "league": rng.choice(["EPL", "La_Liga", "Bundesliga", "Serie_A"], n),
        "season": rng.choice([2022, 2023, 2024], n),
    })
    return df


class TestGenericDisagreementSelectorCorners:
    """Test GenericDisagreementSelector with corner H/D/A outcomes."""

    def test_fit_and_select(self, corner_synthetic_df):
        selector = GenericDisagreementSelector(
            outcomes=["H", "D", "A"],
            outcome_suffixes={"H": "h", "D": "d", "A": "a"},
            config={"percentiles": [5, 10, 20], "default_percentile": 10},
            disagree_prefix="corner_disagree",
        )
        selector.fit(corner_synthetic_df)

        assert selector.percentile_cutoffs is not None
        masks = selector.select(corner_synthetic_df, percentile=10)
        assert "H" in masks and "D" in masks and "A" in masks
        assert len(masks["H"]) == len(corner_synthetic_df)

    def test_three_outcomes_cutoffs(self, corner_synthetic_df):
        selector = GenericDisagreementSelector(
            outcomes=["H", "D", "A"],
            outcome_suffixes={"H": "h", "D": "d", "A": "a"},
            config={"percentiles": [5, 10], "default_percentile": 5},
            disagree_prefix="corner_disagree",
        )
        selector.fit(corner_synthetic_df)

        # All three suffixes present
        for p in [5, 10]:
            assert "h" in selector.percentile_cutoffs[p]
            assert "d" in selector.percentile_cutoffs[p]
            assert "a" in selector.percentile_cutoffs[p]

    def test_narrower_percentile_fewer_bets(self, corner_synthetic_df):
        selector = GenericDisagreementSelector(
            outcomes=["H", "D", "A"],
            outcome_suffixes={"H": "h", "D": "d", "A": "a"},
            config={"percentiles": [5, 10, 20], "default_percentile": 10},
            disagree_prefix="corner_disagree",
        )
        selector.fit(corner_synthetic_df)

        masks5 = selector.select(corner_synthetic_df, percentile=5)
        masks20 = selector.select(corner_synthetic_df, percentile=20)

        total5 = sum(m.sum() for m in masks5.values())
        total20 = sum(m.sum() for m in masks20.values())
        assert total5 <= total20

    def test_roi_computation(self, corner_synthetic_df):
        selector = GenericDisagreementSelector(
            outcomes=["H", "D", "A"],
            outcome_suffixes={"H": "h", "D": "d", "A": "a"},
            config={"percentiles": [10], "default_percentile": 10},
            disagree_prefix="corner_disagree",
        )
        selector.fit(corner_synthetic_df)
        masks = selector.select(corner_synthetic_df, percentile=10)

        odds = {
            "H": corner_synthetic_df["b365_corner_h"].values,
            "D": corner_synthetic_df["b365_corner_d"].values,
            "A": corner_synthetic_df["b365_corner_a"].values,
        }
        roi = compute_flat_stake_roi_generic(
            corner_synthetic_df["corner_ftr"].values, masks, odds,
        )
        assert "n_bets" in roi
        assert "roi_pct" in roi
        assert roi["n_bets"] > 0

    def test_save_load_roundtrip(self, corner_synthetic_df):
        selector = GenericDisagreementSelector(
            outcomes=["H", "D", "A"],
            outcome_suffixes={"H": "h", "D": "d", "A": "a"},
            config={"percentiles": [5, 10], "default_percentile": 5},
            disagree_prefix="corner_disagree",
        )
        selector.fit(corner_synthetic_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "selector.pkl"
            selector.save(path)
            loaded = GenericDisagreementSelector.load(path)

            assert loaded.outcomes == ["H", "D", "A"]
            assert loaded.disagree_prefix == "corner_disagree"
            for p in [5, 10]:
                for suffix in ["h", "d", "a"]:
                    assert abs(loaded.percentile_cutoffs[p][suffix]
                               - selector.percentile_cutoffs[p][suffix]) < 1e-10

    def test_league_filter(self, corner_synthetic_df):
        selector = GenericDisagreementSelector(
            outcomes=["H", "D", "A"],
            outcome_suffixes={"H": "h", "D": "d", "A": "a"},
            config={"percentiles": [10], "default_percentile": 10},
            disagree_prefix="corner_disagree",
        )
        selector.fit(corner_synthetic_df)

        odds = {
            "H": corner_synthetic_df["b365_corner_h"].values,
            "D": corner_synthetic_df["b365_corner_d"].values,
            "A": corner_synthetic_df["b365_corner_a"].values,
        }
        leagues = selector.compute_league_filter(
            corner_synthetic_df, corner_synthetic_df["corner_ftr"].values,
            odds, corner_synthetic_df["league"].values,
            percentile=10, min_roi_pct=-100, min_bets=1,
        )
        assert isinstance(leagues, set)

    def test_select_with_league_filter_restricts(self, corner_synthetic_df):
        selector = GenericDisagreementSelector(
            outcomes=["H", "D", "A"],
            outcome_suffixes={"H": "h", "D": "d", "A": "a"},
            config={"percentiles": [10], "default_percentile": 10},
            disagree_prefix="corner_disagree",
        )
        selector.fit(corner_synthetic_df)

        masks_all = selector.select(corner_synthetic_df, percentile=10)
        masks_filt = selector.select_with_league_filter(
            corner_synthetic_df, corner_synthetic_df["league"].values,
            percentile=10, allowed_leagues={"EPL"},
        )

        total_all = sum(m.sum() for m in masks_all.values())
        total_filt = sum(m.sum() for m in masks_filt.values())
        assert total_filt <= total_all

        # Filtered should only include EPL matches
        for outcome in ["H", "D", "A"]:
            selected_idx = np.where(masks_filt[outcome])[0]
            for idx in selected_idx:
                assert corner_synthetic_df.iloc[idx]["league"] == "EPL"
