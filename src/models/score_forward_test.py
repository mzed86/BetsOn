"""Score forward test predictions against actual results.

Loads past predictions from outputs/forward_test/predictions/,
fetches actual results from football-data.co.uk current season data,
and produces a running P&L tally with statistical significance testing.

CLI: python -m src.models.score_forward_test
"""

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "forward_test" / "predictions"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "forward_test" / "results"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "football_data"


def load_all_predictions() -> pd.DataFrame:
    """Load all dated prediction files and combine."""
    pred_files = sorted(PREDICTIONS_DIR.glob("bets_*.csv"))
    if not pred_files:
        print("No prediction files found in", PREDICTIONS_DIR)
        return pd.DataFrame()

    dfs = []
    for f in pred_files:
        df = pd.read_csv(f)
        df["prediction_file"] = f.name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    return combined


def load_current_season_results() -> pd.DataFrame:
    """Load results from football-data.co.uk current season CSVs."""
    # Find the most recent season code
    all_results = []
    for league_dir in RAW_DATA_DIR.iterdir():
        if not league_dir.is_dir():
            continue
        csvs = sorted(league_dir.glob("*.csv"))
        if not csvs:
            continue
        # Take the latest season file
        latest = csvs[-1]
        try:
            df = pd.read_csv(latest)
            if "FTR" in df.columns and "HomeTeam" in df.columns:
                df["league"] = league_dir.name
                all_results.append(df)
        except Exception:
            continue

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    # Parse dates
    for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y"]:
        try:
            combined["Date"] = pd.to_datetime(combined["Date"], format=fmt)
            break
        except (ValueError, TypeError):
            continue

    return combined


def score_predictions(predictions: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Match predictions against actual results and compute P&L.

    Args:
        predictions: DataFrame from load_all_predictions.
        results: DataFrame from load_current_season_results.

    Returns:
        Predictions DataFrame with added 'actual_result', 'won', 'pnl' columns.
    """
    if predictions.empty or results.empty:
        return predictions

    scored = predictions.copy()
    scored["actual_result"] = None
    scored["won"] = None
    scored["pnl"] = None

    for idx, bet in scored.iterrows():
        # Find matching result
        match_mask = (
            (results["Date"] == bet["date"])
            & (results["HomeTeam"] == bet["home_team"])
        )
        matches = results[match_mask]
        if matches.empty:
            continue

        match = matches.iloc[0]

        # Determine actual result based on market
        if bet["market"] == "1X2":
            actual = match.get("FTR")
        elif bet["market"] == "O/U":
            fthg = match.get("FTHG")
            ftag = match.get("FTAG")
            if pd.notna(fthg) and pd.notna(ftag):
                actual = "Over" if (fthg + ftag) > 2.5 else "Under"
            else:
                actual = None
        elif bet["market"] == "Corners":
            hc = match.get("HC")
            ac = match.get("AC")
            if pd.notna(hc) and pd.notna(ac):
                if hc > ac:
                    actual = "H"
                elif hc == ac:
                    actual = "D"
                else:
                    actual = "A"
            else:
                actual = None
        else:
            actual = None

        if actual is None:
            continue

        scored.at[idx, "actual_result"] = actual
        won = actual == bet["outcome"]
        scored.at[idx, "won"] = won

        stake = bet["stake"]
        if won:
            # Approximate PnL (odds already adjusted for commission in forward_test)
            scored.at[idx, "pnl"] = round(stake * (bet["odds"] - 1) * 0.962, 2)  # ~3.8% commission
        else:
            scored.at[idx, "pnl"] = -round(stake, 2)

    return scored


def compute_significance(resolved: pd.DataFrame) -> dict:
    """Compute statistical significance of forward test results.

    Uses a one-sample t-test on profit-per-unit-staked to test whether
    the mean return is significantly greater than zero.

    Returns dict with p_value, t_stat, confidence interval, verdict,
    and estimated bets remaining to reach significance.
    """
    if len(resolved) < 2:
        return {
            "n_bets": len(resolved),
            "roi_pct": None,
            "weighted_roi_pct": None,
            "mean_return": None,
            "std_return": None,
            "t_stat": None,
            "p_value": None,
            "ci_lower_pct": None,
            "ci_upper_pct": None,
            "verdict": "Not enough data",
            "bets_needed": None,
            "bets_remaining": None,
        }

    # Profit per unit staked for each bet (normalise by stake to remove
    # Kelly sizing effects — we want to test the edge, not the sizing)
    returns = resolved["pnl"].values / resolved["stake"].values

    n = len(returns)
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=1))
    roi_pct = mean_return * 100

    # Stake-weighted ROI reflects actual cash flow
    total_staked = float(resolved["stake"].sum())
    total_pnl = float(resolved["pnl"].sum())
    weighted_roi_pct = (total_pnl / total_staked * 100) if total_staked > 0 else 0.0

    # One-sided t-test: H0: mean_return <= 0, H1: mean_return > 0
    if std_return == 0:
        t_stat = float("inf") if mean_return > 0 else float("-inf")
        p_value = 0.0 if mean_return > 0 else 1.0
    else:
        t_stat = mean_return / (std_return / math.sqrt(n))
        p_value = 1 - stats.t.cdf(t_stat, df=n - 1)

    # 95% confidence interval for ROI
    se = std_return / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lower = (mean_return - t_crit * se) * 100
    ci_upper = (mean_return + t_crit * se) * 100

    # Estimate bets needed to reach significance at current observed ROI
    if mean_return > 0 and std_return > 0:
        # n needed for z=1.645 (one-sided 95%) given observed effect size
        bets_needed = math.ceil((1.645 * std_return / mean_return) ** 2)
        bets_remaining = max(0, bets_needed - n)
    else:
        bets_needed = None
        bets_remaining = None

    # Human-readable verdict — use stake-weighted ROI so the verdict
    # reflects actual cash-flow reality, not the unweighted per-bet mean
    if p_value < 0.01 and weighted_roi_pct > 0:
        verdict = "SIGNIFICANT (p < 0.01) — strong evidence of edge"
    elif p_value < 0.05 and weighted_roi_pct > 0:
        verdict = "SIGNIFICANT (p < 0.05) — evidence of edge"
    elif p_value < 0.10 and weighted_roi_pct > 0:
        verdict = "MARGINAL (p < 0.10) — weak evidence, keep going"
    elif weighted_roi_pct > 0:
        verdict = "NOT YET SIGNIFICANT — positive but inconclusive"
    else:
        verdict = "NEGATIVE — no evidence of edge"

    return {
        "n_bets": n,
        "roi_pct": round(roi_pct, 2),
        "weighted_roi_pct": round(weighted_roi_pct, 2),
        "mean_return": round(mean_return, 4),
        "std_return": round(std_return, 4),
        "t_stat": round(t_stat, 3),
        "p_value": round(p_value, 4),
        "ci_lower_pct": round(ci_lower, 2),
        "ci_upper_pct": round(ci_upper, 2),
        "verdict": verdict,
        "bets_needed": bets_needed,
        "bets_remaining": bets_remaining,
    }


def print_forward_test_report(scored: pd.DataFrame) -> None:
    """Print running forward test report with significance testing."""
    resolved = scored[scored["won"].notna()].copy()
    pending = scored[scored["won"].isna()]

    print(f"\n{'=' * 70}")
    print("=== FORWARD TEST REPORT ===")
    print(f"{'=' * 70}")
    print(f"  Total predictions: {len(scored)}")
    print(f"  Resolved:          {len(resolved)}")
    print(f"  Pending:           {len(pending)}")

    if resolved.empty:
        print("\n  No resolved bets yet.")
        return

    total_staked = resolved["stake"].sum()
    total_pnl = resolved["pnl"].sum()
    n_wins = resolved["won"].sum()
    n_bets = len(resolved)
    roi = total_pnl / total_staked * 100 if total_staked > 0 else 0

    print(f"\n  Resolved bets:     {n_bets}")
    print(f"  Wins:              {int(n_wins)} ({n_wins/n_bets*100:.1f}%)")
    print(f"  Total staked:      {total_staked:.2f}")
    print(f"  Total PnL:         {total_pnl:+.2f}")
    print(f"  ROI:               {roi:+.2f}%")

    # Per-market breakdown
    print(f"\n  --- Per-Market ---")
    print(f"  {'Market':<12} {'Bets':>6} {'Wins':>6} {'PnL':>10} {'ROI%':>8}")
    print(f"  {'-' * 45}")
    for market in sorted(resolved["market"].unique()):
        m = resolved[resolved["market"] == market]
        m_staked = m["stake"].sum()
        m_pnl = m["pnl"].sum()
        m_wins = m["won"].sum()
        m_roi = m_pnl / m_staked * 100 if m_staked > 0 else 0
        print(f"  {market:<12} {len(m):>6} {int(m_wins):>6} {m_pnl:>+10.2f} {m_roi:>+7.2f}%")

    # Statistical significance
    sig = compute_significance(resolved)
    print(f"\n  --- Statistical Significance ---")
    print(f"  Test:              one-sided t-test (H0: mean profit <= 0)")
    print(f"  ROI per bet:       {sig['roi_pct']:+.2f}%")
    print(f"  95% CI:            [{sig['ci_lower_pct']:+.2f}%, {sig['ci_upper_pct']:+.2f}%]")
    print(f"  t-statistic:       {sig['t_stat']:.3f}")
    print(f"  p-value:           {sig['p_value']:.4f}")
    print(f"  Verdict:           {sig['verdict']}")
    if sig["bets_remaining"] is not None:
        if sig["bets_remaining"] == 0:
            print(f"  Bets to 95% sig:   already there!")
        else:
            weeks = math.ceil(sig["bets_remaining"] / 120)
            print(f"  Bets to 95% sig:   ~{sig['bets_remaining']} more (~{weeks} weeks at current rate)")


def main() -> None:
    """Score forward test predictions and print report."""
    predictions = load_all_predictions()
    if predictions.empty:
        print("No predictions to score.")
        return

    results = load_current_season_results()
    if results.empty:
        print("No results data available. Run the scraper first:")
        print("  python -m scrapers.football_data --seasons 2526")
        return

    scored = score_predictions(predictions, results)

    print_forward_test_report(scored)

    # Save scored results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = RESULTS_DIR / f"scored_{date_str}.csv"
    scored.to_csv(out_path, index=False)
    print(f"\n  Saved scored results to {out_path}")

    # Save running tally (with significance stats)
    resolved = scored[scored["won"].notna()]
    if not resolved.empty:
        sig = compute_significance(resolved)
        tally = {
            "as_of": date_str,
            "total_bets": len(resolved),
            "wins": int(resolved["won"].sum()),
            "total_staked": round(resolved["stake"].sum(), 2),
            "total_pnl": round(resolved["pnl"].sum(), 2),
            "roi_pct": round(resolved["pnl"].sum() / resolved["stake"].sum() * 100, 2)
            if resolved["stake"].sum() > 0 else 0.0,
            "significance": sig,
        }
        tally_path = RESULTS_DIR / "running_tally.json"
        with open(tally_path, "w") as f:
            json.dump(tally, f, indent=2)
        print(f"  Saved running tally to {tally_path}")


if __name__ == "__main__":
    main()
