"""Load feature engineering config from YAML into typed dataclasses."""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "feature_config.yaml"


@dataclass(frozen=True)
class RollingConfig:
    windows: list[int] = field(default_factory=lambda: [5, 10])
    venue_windows: list[int] = field(default_factory=lambda: [3, 6])
    min_periods: int = 1


@dataclass(frozen=True)
class EloConfig:
    k_factor: float = 20
    initial_rating: float = 1500
    home_advantage: float = 100


@dataclass(frozen=True)
class OddsConfig:
    primary: list[str] = field(default_factory=lambda: ["B365H", "B365D", "B365A"])
    pinnacle_opening: list[str] = field(default_factory=lambda: ["PSH", "PSD", "PSA"])
    pinnacle_closing: list[str] = field(default_factory=lambda: ["PSCH", "PSCD", "PSCA"])
    drop_columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MatchContextConfig:
    early_season_threshold: int = 5
    summer_break_threshold: int = 60


@dataclass(frozen=True)
class FeatureConfig:
    rolling: RollingConfig = field(default_factory=RollingConfig)
    elo: EloConfig = field(default_factory=EloConfig)
    odds: OddsConfig = field(default_factory=OddsConfig)
    match_context: MatchContextConfig = field(default_factory=MatchContextConfig)
    tier1_leagues: list[str] = field(
        default_factory=lambda: ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"]
    )


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> FeatureConfig:
    """Load and cache the feature config from YAML.

    Args:
        path: Optional path override. Defaults to configs/feature_config.yaml.

    Returns:
        Parsed FeatureConfig dataclass.
    """
    config_path = path or CONFIG_PATH
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return FeatureConfig(
        rolling=RollingConfig(**raw.get("rolling", {})),
        elo=EloConfig(**raw.get("elo", {})),
        odds=OddsConfig(**raw.get("odds", {})),
        match_context=MatchContextConfig(**raw.get("match_context", {})),
        tier1_leagues=raw.get("tier1_leagues", ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"]),
    )
