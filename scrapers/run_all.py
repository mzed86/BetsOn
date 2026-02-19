"""Run all scrapers sequentially with config-driven defaults.

Usage:
    python -m scrapers.run_all
"""

import logging

from scrapers.football_data import scrape as scrape_football_data
from scrapers.understat_scraper import scrape as scrape_understat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== Running football-data.co.uk scraper ===")
    scrape_football_data()

    logger.info("=== Running Understat scraper ===")
    scrape_understat()

    logger.info("=== All scrapers complete ===")


if __name__ == "__main__":
    main()
