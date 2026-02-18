"""Feature engineering pipeline orchestrator.

Loads raw match data, runs all feature modules, computes differentials,
and outputs a model-ready Parquet file.
"""

import logging
import time
from pathlib import Path

import pandas as pd

from src.features.config import FeatureConfig, load_config
from src.features.elo import compute_elo_features
from src.features.match_context import compute_match_context_features
from src.features.odds import compute_odds_features
from src.features.rolling import compute_rolling_features

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
INPUT_PATH = DATA_DIR / "processed" / "matches_combined.parquet"
OUTPUT_PATH = DATA_DIR / "processed" / "features.parquet"


def _load_and_clean(config: FeatureConfig) -> pd.DataFrame:
    """Load matches_combined.parquet and apply basic cleaning.

    Drops rows with null FTR and columns with >90% null (from config).
    """
    logger.info("Loading %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    logger.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    # Drop rows with null match result
    null_ftr = df["FTR"].isnull().sum()
    if null_ftr > 0:
        df = df.dropna(subset=["FTR"]).reset_index(drop=True)
        logger.info("Dropped %d rows with null FTR", null_ftr)

    # Drop columns with >90% null (from config)
    drop_cols = [c for c in config.odds.drop_columns if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info("Dropped %d high-null columns", len(drop_cols))

    return df


def _compute_differentials(features: pd.DataFrame) -> pd.DataFrame:
    """Compute home-minus-away differential features."""
    diff = pd.DataFrame(index=features.index)

    # Rolling stat differentials
    for col in features.columns:
        if col.startswith("h_rolling") and "_ppg" in col:
            suffix = col[2:]  # strip "h_"
            a_col = f"a_{suffix}"
            if a_col in features.columns:
                diff[f"form_diff_{suffix.split('_')[0].replace('rolling', '')}"] = (
                    features[col] - features[a_col]
                )
        elif col.startswith("h_rolling") and "_goals_for" in col:
            suffix = col[2:]
            a_col = f"a_{suffix}"
            if a_col in features.columns:
                window = suffix.split("_")[0].replace("rolling", "")
                diff[f"goals_diff_{window}"] = features[col] - features[a_col]
        elif col.startswith("h_rolling") and "_shots_on_target" in col:
            suffix = col[2:]
            a_col = f"a_{suffix}"
            if a_col in features.columns:
                window = suffix.split("_")[0].replace("rolling", "")
                diff[f"sot_diff_{window}"] = features[col] - features[a_col]
        elif col.startswith("h_rolling") and "_xg_for" in col:
            suffix = col[2:]
            a_col = f"a_{suffix}"
            if a_col in features.columns:
                window = suffix.split("_")[0].replace("rolling", "")
                diff[f"xg_diff_{window}"] = features[col] - features[a_col]

    return diff


def build(config: FeatureConfig | None = None) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Args:
        config: Optional FeatureConfig override. If None, loads from YAML.

    Returns:
        The feature-engineered DataFrame.
    """
    if config is None:
        config = load_config()

    matches = _load_and_clean(config)

    # Run each feature module
    steps = [
        ("Rolling features", lambda: compute_rolling_features(matches, config.rolling)),
        ("Elo features", lambda: compute_elo_features(matches, config.elo)),
        ("Odds features", lambda: compute_odds_features(matches, config.odds)),
        ("Match context", lambda: compute_match_context_features(matches, config.match_context)),
    ]

    feature_parts = [matches]
    for name, func in steps:
        logger.info("--- %s ---", name)
        start = time.time()
        part = func()
        elapsed = time.time() - start
        logger.info("%s completed in %.1fs", name, elapsed)
        feature_parts.append(part)

    # Combine all features
    features = pd.concat(feature_parts, axis=1)

    # Compute differentials
    logger.info("--- Differential features ---")
    diffs = _compute_differentials(features)
    features = pd.concat([features, diffs], axis=1)
    logger.info("Differential features: %d columns", len(diffs.columns))

    logger.info("Final shape: %d rows x %d columns", len(features), len(features.columns))
    return features


def main() -> None:
    """CLI entry point: build features and write to Parquet."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    total_start = time.time()
    features = build()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Wrote %s (%d rows x %d cols)", OUTPUT_PATH, len(features), len(features.columns))

    total_elapsed = time.time() - total_start
    logger.info("Feature pipeline finished in %.1fs", total_elapsed)
