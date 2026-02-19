"""Scraper for The Odds API: fetch upcoming fixture odds for forward testing.

Pulls upcoming match odds from the-odds-api.com (free tier: 500 requests/month)
and formats them into a CSV compatible with forward_test.py.

Requires THE_ODDS_API_KEY environment variable.

Usage:
    python -m scrapers.odds_api
    python -m scrapers.odds_api --leagues soccer_epl soccer_germany_bundesliga
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"

# Mapping: The Odds API sport key -> our league name (must match selector profitable_leagues)
LEAGUE_MAP = {
    "soccer_epl": "EPL",
    "soccer_efl_champ": "EFL_Championship",
    "soccer_england_league1": "EFL_League1",
    "soccer_england_league2": "EFL_League2",
    "soccer_spl": "SPL",
    "soccer_germany_bundesliga": "Bundesliga",
    "soccer_germany_bundesliga2": "Bundesliga2",
    "soccer_italy_serie_a": "Serie_A",
    "soccer_italy_serie_b": "Serie_B",
    "soccer_spain_la_liga": "La_Liga",
    "soccer_spain_segunda_division": "La_Liga2",
    "soccer_france_ligue_one": "Ligue_1",
    "soccer_france_ligue_two": "Ligue_2",
    "soccer_netherlands_eredivisie": "Eredivisie",
    "soccer_portugal_primeira_liga": "Liga_Portugal",
    "soccer_turkey_super_league": "Super_Lig",
}

# Bookmaker key mapping for The Odds API
BOOKMAKER_MAP = {
    "bet365": "B365",
    "pinnacle": "PS",
}

API_BASE = "https://api.the-odds-api.com/v4"


def get_api_key() -> str:
    """Get API key from environment."""
    key = os.environ.get("THE_ODDS_API_KEY", "")
    if not key:
        raise RuntimeError(
            "THE_ODDS_API_KEY environment variable not set.\n"
            "Get a free key at https://the-odds-api.com/"
        )
    return key


def fetch_odds(
    sport_key: str,
    api_key: str,
    markets: str = "h2h,totals",
    regions: str = "uk,eu",
    odds_format: str = "decimal",
) -> list[dict]:
    """Fetch upcoming odds for a sport from The Odds API.

    Args:
        sport_key: Sport identifier (e.g. "soccer_epl").
        api_key: API key.
        markets: Comma-separated markets (h2h, totals, spreads).
        regions: Comma-separated regions for bookmaker filtering.
        odds_format: "decimal" or "american".

    Returns:
        List of event dicts from the API.
    """
    url = f"{API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }
    resp = requests.get(url, params=params, timeout=30)

    # Log quota usage from headers
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    logger.info("  API quota: %s used, %s remaining", used, remaining)

    resp.raise_for_status()
    return resp.json()


def _extract_bookmaker_odds(
    bookmakers: list[dict],
    bookmaker_key: str,
    market_key: str,
) -> dict | None:
    """Extract odds from a specific bookmaker for a specific market.

    Returns dict like {"H": 2.5, "D": 3.4, "A": 2.8} for h2h,
    or {"Over": 1.8, "Under": 2.1} for totals.
    """
    for bm in bookmakers:
        if bm["key"] != bookmaker_key:
            continue
        for mkt in bm.get("markets", []):
            if mkt["key"] != market_key:
                continue
            outcomes = {}
            for o in mkt["outcomes"]:
                name = o["name"]
                price = o["price"]
                if market_key == "h2h":
                    # Map: home_team -> H, Draw -> D, away_team -> A
                    # Names are actual team names or "Draw"
                    outcomes[name] = price
                elif market_key == "totals":
                    # "Over" or "Under" with a point value
                    outcomes[name] = price
            return outcomes
    return None


def _compute_avg_odds(
    bookmakers: list[dict],
    market_key: str,
    exclude_keys: set | None = None,
) -> dict | None:
    """Compute average odds across all bookmakers for a market."""
    exclude_keys = exclude_keys or set()
    all_odds = {}
    counts = {}

    for bm in bookmakers:
        if bm["key"] in exclude_keys:
            continue
        for mkt in bm.get("markets", []):
            if mkt["key"] != market_key:
                continue
            for o in mkt["outcomes"]:
                name = o["name"]
                price = o["price"]
                all_odds.setdefault(name, 0.0)
                counts.setdefault(name, 0)
                all_odds[name] += price
                counts[name] += 1

    if not all_odds:
        return None
    return {name: all_odds[name] / counts[name] for name in all_odds}


def events_to_fixtures_df(
    events: list[dict],
    league_name: str,
) -> pd.DataFrame:
    """Convert API events to a fixtures DataFrame for forward_test.py.

    Args:
        events: List of event dicts from The Odds API.
        league_name: Our internal league name.

    Returns:
        DataFrame with columns matching forward_test.py expectations.
    """
    rows = []
    for event in events:
        home = event["home_team"]
        away = event["away_team"]
        commence = event.get("commence_time", "")
        bookmakers = event.get("bookmakers", [])

        if not bookmakers:
            continue

        row = {
            "Date": commence[:10] if commence else "",
            "home_team": home,
            "away_team": away,
            "league": league_name,
        }

        # --- 1X2 (h2h) ---
        # Pinnacle (sharp reference)
        ps_h2h = _extract_bookmaker_odds(bookmakers, "pinnacle", "h2h")
        if ps_h2h:
            row["PSH"] = ps_h2h.get(home)
            row["PSD"] = ps_h2h.get("Draw")
            row["PSA"] = ps_h2h.get(away)

        # B365 if available; otherwise use average of soft books as proxy
        b365_h2h = _extract_bookmaker_odds(bookmakers, "bet365", "h2h")
        if b365_h2h:
            row["B365H"] = b365_h2h.get(home)
            row["B365D"] = b365_h2h.get("Draw")
            row["B365A"] = b365_h2h.get(away)
        else:
            avg_h2h = _compute_avg_odds(bookmakers, "h2h", exclude_keys={"pinnacle"})
            if avg_h2h:
                row["B365H"] = avg_h2h.get(home)
                row["B365D"] = avg_h2h.get("Draw")
                row["B365A"] = avg_h2h.get(away)

        # --- O/U 2.5 (totals) ---
        ps_totals = _extract_bookmaker_odds(bookmakers, "pinnacle", "totals")
        if ps_totals:
            row["P>2.5"] = ps_totals.get("Over")
            row["P<2.5"] = ps_totals.get("Under")

        b365_totals = _extract_bookmaker_odds(bookmakers, "bet365", "totals")
        if b365_totals:
            row["B365>2.5"] = b365_totals.get("Over")
            row["B365<2.5"] = b365_totals.get("Under")
        else:
            avg_totals = _compute_avg_odds(bookmakers, "totals", exclude_keys={"pinnacle"})
            if avg_totals:
                row["B365>2.5"] = avg_totals.get("Over")
                row["B365<2.5"] = avg_totals.get("Under")

        # --- Corner odds ---
        # The Odds API doesn't carry corner markets.
        # Use avg soft-book h2h as proxy for corner odds (AvgCH/CD/CA)
        # and B365/avg h2h as proxy for B365CH/CD/CA.
        avg_h2h_for_corners = _compute_avg_odds(bookmakers, "h2h", exclude_keys={"pinnacle"})
        if avg_h2h_for_corners:
            row["AvgCH"] = avg_h2h_for_corners.get(home)
            row["AvgCD"] = avg_h2h_for_corners.get("Draw")
            row["AvgCA"] = avg_h2h_for_corners.get(away)
        # Use B365 (or avg) for B365 corner columns too
        b365_for_corners = b365_h2h or avg_h2h_for_corners
        if b365_for_corners:
            row["B365CH"] = b365_for_corners.get(home)
            row["B365CD"] = b365_for_corners.get("Draw")
            row["B365CA"] = b365_for_corners.get(away)

        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def scrape_upcoming_fixtures(
    league_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Scrape upcoming fixtures with odds from The Odds API.

    Args:
        league_keys: List of sport keys to fetch. Defaults to all in LEAGUE_MAP.

    Returns:
        Combined DataFrame of upcoming fixtures.
    """
    api_key = get_api_key()
    league_keys = league_keys or list(LEAGUE_MAP.keys())

    all_dfs = []
    for sport_key in league_keys:
        league_name = LEAGUE_MAP.get(sport_key, sport_key)
        logger.info("Fetching %s (%s)...", league_name, sport_key)

        try:
            events = fetch_odds(sport_key, api_key)
            if not events:
                logger.info("  No upcoming events")
                continue
            df = events_to_fixtures_df(events, league_name)
            if not df.empty:
                logger.info("  Got %d fixtures", len(df))
                all_dfs.append(df)
            else:
                logger.info("  No fixtures with odds")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                logger.info("  Sport not available: %s", sport_key)
            else:
                logger.error("  HTTP error: %s", e)
        except requests.RequestException as e:
            logger.error("  Request error: %s", e)

    if not all_dfs:
        logger.warning("No fixtures found across any league")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)
    return combined


def main() -> None:
    """Fetch upcoming odds and save to CSV."""
    parser = argparse.ArgumentParser(description="Fetch upcoming fixture odds")
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=None,
        help="Sport keys to fetch (e.g. soccer_epl soccer_germany_bundesliga)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: outputs/predictions/upcoming_fixtures.csv)",
    )
    args = parser.parse_args()

    df = scrape_upcoming_fixtures(args.leagues)

    if df.empty:
        logger.warning("No fixtures to save")
        return

    output_path = args.output or OUTPUT_DIR / "upcoming_fixtures.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d fixtures to %s", len(df), output_path)

    # Also save a dated copy for history
    date_str = datetime.now().strftime("%Y%m%d")
    history_dir = PROJECT_ROOT / "outputs" / "forward_test" / "fixtures"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"fixtures_{date_str}.csv"
    df.to_csv(history_path, index=False)
    logger.info("Saved dated copy to %s", history_path)


if __name__ == "__main__":
    main()
