"""CLI orchestrator for the data processing pipeline.

Usage:
    python -m src.data.process                          # Run all steps
    python -m src.data.process --steps football_data    # Just football-data
    python -m src.data.process --steps understat merge  # Understat + merge
"""

import argparse
import logging
import time

from src.data import football_data_processor, understat_processor, merge

STEPS = {
    "football_data": ("Football-data processing", football_data_processor.process),
    "understat": ("Understat processing", understat_processor.process),
    "merge": ("Merging datasets", merge.merge),
}

ALL_STEPS = list(STEPS.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="Data processing pipeline")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=ALL_STEPS,
        help="Which processing steps to run (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    total_start = time.time()

    for step_name in args.steps:
        description, func = STEPS[step_name]
        logging.info("=== %s ===", description)
        start = time.time()
        func()
        elapsed = time.time() - start
        logging.info("=== %s completed in %.1fs ===", description, elapsed)

    total_elapsed = time.time() - total_start
    logging.info("Pipeline finished in %.1fs", total_elapsed)


if __name__ == "__main__":
    main()
