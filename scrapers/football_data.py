"""Scraper for football-data.co.uk historical match results and betting odds.

Downloads CSV files containing match results, statistics, and bookmaker odds
for configured leagues and seasons.

Usage:
    python -m scrapers.football_data [--leagues EPL Bundesliga] [--seasons 2324 2425]
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "scraping_config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "football_data"


def load_config() -> dict:
    """Load the football-data section of the scraping config."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return config["football_data"]


def build_url(base_url: str, season_code: str, league_code: str) -> str:
    """Construct the download URL for a given season and league."""
    return f"{base_url}/{season_code}/{league_code}.csv"


def download_csv(url: str, timeout: int = 30) -> pd.DataFrame | None:
    """Download a CSV from a URL and return as a DataFrame."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    # football-data.co.uk uses various encodings
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(
                pd.io.common.StringIO(response.content.decode(encoding)),
                on_bad_lines="skip",
            )
            # Drop fully empty rows/columns that sometimes appear at the end
            df = df.dropna(how="all").dropna(axis=1, how="all")
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    logger.warning("Could not decode CSV from %s", url)
    return None


def resolve_leagues(config: dict, requested: list[str] | None) -> dict[str, str]:
    """Map requested league display names back to league codes.

    If requested is None, return all configured leagues.
    """
    code_to_name = config["leagues"]  # e.g. {"E0": "EPL", ...}
    name_to_code = {v: k for k, v in code_to_name.items()}

    if requested is None:
        return code_to_name

    result = {}
    for name in requested:
        if name in name_to_code:
            result[name_to_code[name]] = name
        elif name in code_to_name:
            # User passed the raw code directly
            result[name] = code_to_name[name]
        else:
            logger.warning("Unknown league '%s', skipping", name)
    return result


def resolve_seasons(config: dict, requested: list[str] | None) -> list[str]:
    """Return the list of season codes to scrape."""
    if requested is None:
        return config["seasons"]
    return requested


def scrape(leagues: list[str] | None = None, seasons: list[str] | None = None) -> None:
    """Run the football-data.co.uk scraper."""
    config = load_config()
    base_url = config["base_url"]
    delay = config.get("request_delay_seconds", 1.0)

    league_map = resolve_leagues(config, leagues)
    season_list = resolve_seasons(config, seasons)

    total = len(league_map) * len(season_list)
    downloaded = 0
    failed = 0

    logger.info(
        "Starting football-data.co.uk scraper: %d leagues x %d seasons = %d files",
        len(league_map),
        len(season_list),
        total,
    )

    for league_code, league_name in league_map.items():
        league_dir = OUTPUT_DIR / league_name
        league_dir.mkdir(parents=True, exist_ok=True)

        for season_code in season_list:
            output_path = league_dir / f"{season_code}.csv"

            if output_path.exists():
                logger.info("Already exists: %s — skipping", output_path)
                downloaded += 1
                continue

            url = build_url(base_url, season_code, league_code)
            logger.info("Downloading %s %s -> %s", league_name, season_code, url)

            try:
                df = download_csv(url)
                if df is not None and not df.empty:
                    df.to_csv(output_path, index=False)
                    downloaded += 1
                    logger.info(
                        "Saved %s (%d rows, %d cols)",
                        output_path.name,
                        len(df),
                        len(df.columns),
                    )
                else:
                    failed += 1
                    logger.warning("Empty or unparseable: %s", url)
            except requests.RequestException as e:
                failed += 1
                logger.error("HTTP error for %s: %s", url, e)

            time.sleep(delay)

    logger.info(
        "Done. Downloaded: %d, Failed: %d, Total: %d",
        downloaded,
        failed,
        total,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download match data from football-data.co.uk"
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=None,
        help="League names to scrape (e.g. EPL Bundesliga). Defaults to all configured.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Season codes to scrape (e.g. 2324 2425). Defaults to all configured.",
    )
    args = parser.parse_args()
    scrape(leagues=args.leagues, seasons=args.seasons)


if __name__ == "__main__":
    main()
