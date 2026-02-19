"""Scraper for Understat xG data using the understat async Python library.

Fetches league-level match results, team stats, and player stats with
expected goals (xG) and expected assists (xA) data.

Usage:
    python -m scrapers.understat_scraper [--leagues EPL La_Liga] [--seasons 2023 2024]
"""

import argparse
import asyncio
import logging
import time
from pathlib import Path

import aiohttp
import pandas as pd
import yaml
from understat import Understat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "scraping_config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "understat"


def load_config() -> dict:
    """Load the understat section of the scraping config."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return config["understat"]


def resolve_leagues(config: dict, requested: list[str] | None) -> list[str]:
    """Return the list of league names to scrape."""
    if requested is None:
        return config["leagues"]
    valid = set(config["leagues"])
    result = []
    for league in requested:
        if league in valid:
            result.append(league)
        else:
            logger.warning("Unknown Understat league '%s', skipping", league)
    return result


def resolve_seasons(config: dict, requested: list[str] | None) -> list[int]:
    """Return the list of season start-years to scrape."""
    if requested is None:
        return config["seasons"]
    return [int(s) for s in requested]


async def fetch_league_season(
    understat: Understat, league: str, season: int, delay: float
) -> dict[str, pd.DataFrame]:
    """Fetch matches, teams, and players for one league-season."""
    results = {}

    # Match results with xG
    logger.info("Fetching matches: %s %d", league, season)
    matches = await understat.get_league_results(league, season)
    if matches:
        results["matches"] = pd.json_normalize(matches)
    await asyncio.sleep(delay)

    # Team-level stats
    logger.info("Fetching teams: %s %d", league, season)
    teams_raw = await understat.get_teams(league, season)
    if teams_raw:
        # Flatten the nested team data
        rows = []
        for team in teams_raw:
            row = {"id": team["id"], "title": team["title"]}
            # History contains per-match aggregated stats
            if "history" in team:
                for stat_key in ("xG", "xGA", "scored", "missed", "pts", "wins", "draws", "loses"):
                    values = [float(m.get(stat_key, 0)) for m in team["history"]]
                    row[f"total_{stat_key}"] = sum(values)
                row["matches_played"] = len(team["history"])
            rows.append(row)
        results["teams"] = pd.DataFrame(rows)
    await asyncio.sleep(delay)

    # Player-level stats
    logger.info("Fetching players: %s %d", league, season)
    players = await understat.get_league_players(league, season)
    if players:
        results["players"] = pd.json_normalize(players)
    await asyncio.sleep(delay)

    return results


async def scrape_async(
    leagues: list[str] | None = None, seasons: list[str] | None = None
) -> None:
    """Run the Understat scraper."""
    config = load_config()
    delay = config.get("request_delay_seconds", 2.0)

    league_list = resolve_leagues(config, leagues)
    season_list = resolve_seasons(config, seasons)

    total = len(league_list) * len(season_list)
    downloaded = 0
    failed = 0

    logger.info(
        "Starting Understat scraper: %d leagues x %d seasons = %d league-seasons",
        len(league_list),
        len(season_list),
        total,
    )

    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        for league in league_list:
            league_dir = OUTPUT_DIR / league
            league_dir.mkdir(parents=True, exist_ok=True)

            for season in season_list:
                file_prefix = league_dir / str(season)
                matches_path = Path(f"{file_prefix}_matches.csv")
                teams_path = Path(f"{file_prefix}_teams.csv")
                players_path = Path(f"{file_prefix}_players.csv")

                # Skip if all three files already exist
                if matches_path.exists() and teams_path.exists() and players_path.exists():
                    logger.info("Already exists: %s %d — skipping", league, season)
                    downloaded += 1
                    continue

                try:
                    data = await fetch_league_season(understat, league, season, delay)

                    for key, path in [
                        ("matches", matches_path),
                        ("teams", teams_path),
                        ("players", players_path),
                    ]:
                        if key in data and not data[key].empty:
                            data[key].to_csv(path, index=False)
                            logger.info("Saved %s (%d rows)", path.name, len(data[key]))

                    downloaded += 1

                except Exception as e:
                    failed += 1
                    logger.error("Error fetching %s %d: %s", league, season, e)

    logger.info(
        "Done. Downloaded: %d, Failed: %d, Total: %d",
        downloaded,
        failed,
        total,
    )


def scrape(leagues: list[str] | None = None, seasons: list[str] | None = None) -> None:
    """Synchronous entry point that runs the async scraper."""
    asyncio.run(scrape_async(leagues, seasons))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download xG data from Understat"
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=None,
        help="Leagues to scrape (e.g. EPL La_Liga). Defaults to all configured.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Season start years (e.g. 2023 2024). Defaults to all configured.",
    )
    args = parser.parse_args()
    scrape(leagues=args.leagues, seasons=args.seasons)


if __name__ == "__main__":
    main()
