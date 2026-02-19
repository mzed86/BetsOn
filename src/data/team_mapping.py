"""Team name normalization between football-data.co.uk and Understat."""

from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "team_mapping.yaml"


@lru_cache(maxsize=1)
def _load_config() -> dict:
    """Load and cache the team mapping YAML config."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_mapping(source: str) -> dict[str, str]:
    """Build a flat {source_name: canonical_name} dict for a given source.

    Args:
        source: Either "football_data" or "understat".

    Returns:
        Dictionary mapping source-specific team names to canonical names.
    """
    config = _load_config()
    mapping = {}
    for league_cfg in config.values():
        if source in league_cfg:
            mapping.update(league_cfg[source])
    return mapping


def normalize_team_name(name: str, source: str, league: str | None = None) -> str:
    """Normalize a single team name to its canonical form.

    Args:
        name: Team name from the source data.
        source: Either "football_data" or "understat".
        league: Optional league filter (unused currently, reserved for future
                disambiguation if two leagues map the same source name differently).

    Returns:
        Canonical team name, or the original name if no mapping exists.
    """
    mapping = build_mapping(source)
    return mapping.get(name, name)


def normalize_series(series: pd.Series, source: str, league: str | None = None) -> pd.Series:
    """Vectorized team name normalization for a pandas Series.

    Args:
        series: pandas Series of team names.
        source: Either "football_data" or "understat".
        league: Optional league filter (reserved for future use).

    Returns:
        Series with normalized team names.
    """
    mapping = build_mapping(source)
    return series.map(lambda x: mapping.get(x, x))
