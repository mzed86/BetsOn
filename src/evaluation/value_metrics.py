"""Value betting evaluation metrics.

Functions for computing ROI, sweeping cutoffs, per-season stability,
oracle performance, and formatted comparison tables.
"""

import numpy as np
import pandas as pd


OUTCOMES = ["H", "D", "A"]
B365_COLS = {"H": 0, "D": 1, "A": 2}  # column indices for stacked odds


# =====================================================================
# Generic (outcome-agnostic) functions
# =====================================================================

def compute_flat_stake_roi_generic(
    results: np.ndarray | pd.Series,
    bet_masks: dict[str, np.ndarray],
    odds: dict[str, np.ndarray],
) -> dict:
    """Outcome-generic flat-stake ROI computation.

    Args:
        results: Array of actual outcomes (e.g. "Over"/"Under" or "H"/"D"/"A").
        bet_masks: Dict mapping outcome label to boolean mask of selected bets.
        odds: Dict mapping outcome label to decimal odds array.

    Returns:
        Dict with n_bets, staked, returned, roi_pct, and by_outcome breakdown.
    """
    results = np.asarray(results)
    total_bets = 0
    total_staked = 0.0
    total_returned = 0.0
    by_outcome = {}

    for outcome in bet_masks:
        mask = np.asarray(bet_masks[outcome], dtype=bool)
        o = np.asarray(odds[outcome], dtype=np.float64)
        valid = mask & np.isfinite(o) & (o > 0)
        n_bets = int(valid.sum())

        if n_bets == 0:
            by_outcome[outcome] = {"n_bets": 0, "staked": 0.0, "returned": 0.0, "roi_pct": 0.0}
            continue

        staked = float(n_bets)
        wins = results[valid] == outcome
        returned = float(np.sum(wins * o[valid]))

        by_outcome[outcome] = {
            "n_bets": n_bets,
            "staked": staked,
            "returned": round(returned, 2),
            "roi_pct": round((returned - staked) / staked * 100, 2),
        }
        total_bets += n_bets
        total_staked += staked
        total_returned += returned

    return {
        "n_bets": total_bets,
        "staked": total_staked,
        "returned": round(total_returned, 2),
        "roi_pct": round((total_returned - total_staked) / total_staked * 100, 2) if total_staked > 0 else 0.0,
        "by_outcome": by_outcome,
    }


def compute_per_season_roi_generic(
    results: np.ndarray | pd.Series,
    bet_masks: dict[str, np.ndarray],
    odds: dict[str, np.ndarray],
    seasons: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Outcome-generic per-season ROI computation.

    Args:
        results: Actual outcomes array.
        bet_masks: Dict outcome -> boolean bet masks.
        odds: Dict outcome -> decimal odds arrays.
        seasons: Season identifier for each match.

    Returns:
        DataFrame with columns [season, n_bets, roi_pct].
    """
    results = np.asarray(results)
    seasons = np.asarray(seasons)

    rows = []
    for season in sorted(np.unique(seasons)):
        s_mask = seasons == season
        season_bet_masks = {
            outcome: np.asarray(bet_masks[outcome], dtype=bool) & s_mask
            for outcome in bet_masks
        }
        roi = compute_flat_stake_roi_generic(results, season_bet_masks, odds)
        rows.append({"season": season, "n_bets": roi["n_bets"], "roi_pct": roi["roi_pct"]})

    return pd.DataFrame(rows)


def compute_cost_sensitivity_generic(
    results: np.ndarray | pd.Series,
    strategy_masks: dict[str, dict[str, np.ndarray]],
    odds: dict[str, np.ndarray],
    cost_pcts: list[float],
) -> pd.DataFrame:
    """Outcome-generic transaction cost sensitivity sweep.

    Args:
        results: Actual outcomes array.
        strategy_masks: Dict strategy_name -> bet_masks dict.
        odds: Dict outcome -> decimal odds arrays.
        cost_pcts: List of cost percentages to sweep.

    Returns:
        DataFrame with columns [strategy, cost_pct, n_bets, roi_pct].
    """
    results = np.asarray(results)
    rows = []
    for cost_pct in cost_pcts:
        adj_odds = {
            outcome: adjust_odds_for_costs(odds[outcome], cost_pct)
            for outcome in odds
        }
        for name, masks in strategy_masks.items():
            roi = compute_flat_stake_roi_generic(results, masks, adj_odds)
            rows.append({
                "strategy": name,
                "cost_pct": cost_pct,
                "n_bets": roi["n_bets"],
                "roi_pct": roi["roi_pct"],
            })
    return pd.DataFrame(rows)


def compute_kelly_roi_generic(
    results: np.ndarray | pd.Series,
    bet_masks: dict[str, np.ndarray],
    estimated_probs: dict[str, np.ndarray],
    odds: dict[str, np.ndarray],
    kelly_fraction: float = 0.25,
    max_bet_fraction: float = 0.05,
    bankroll: float = 1000.0,
) -> dict:
    """Outcome-generic sequential Kelly bankroll simulation.

    Args:
        results: Actual outcomes array.
        bet_masks: Dict outcome -> boolean mask.
        estimated_probs: Dict outcome -> probability array.
        odds: Dict outcome -> decimal odds array.
        kelly_fraction: Fractional Kelly multiplier.
        max_bet_fraction: Maximum fraction of bankroll per bet.
        bankroll: Starting bankroll.

    Returns:
        Dict with n_bets, starting_bankroll, ending_bankroll, roi_pct,
        max_drawdown_pct, total_staked.
    """
    results = np.asarray(results)

    bets = []
    for outcome in bet_masks:
        mask = np.asarray(bet_masks[outcome], dtype=bool)
        o = np.asarray(odds[outcome], dtype=np.float64)
        probs = np.asarray(estimated_probs[outcome], dtype=np.float64)
        valid = mask & np.isfinite(o) & (o > 1.0) & np.isfinite(probs)
        indices = np.where(valid)[0]
        for idx in indices:
            bets.append((int(idx), outcome, probs[idx], o[idx]))

    if not bets:
        return {
            "n_bets": 0,
            "starting_bankroll": bankroll,
            "ending_bankroll": bankroll,
            "roi_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "total_staked": 0.0,
        }

    bets.sort(key=lambda x: x[0])

    starting_bankroll = bankroll
    current_bankroll = bankroll
    peak_bankroll = bankroll
    max_drawdown_pct = 0.0
    total_staked = 0.0
    n_bets = 0

    i = 0
    while i < len(bets):
        match_idx = bets[i][0]
        pre_match_bankroll = current_bankroll

        match_bets = []
        while i < len(bets) and bets[i][0] == match_idx:
            match_bets.append(bets[i])
            i += 1

        for _, outcome, prob, o_val in match_bets:
            raw_f = compute_kelly_fraction(prob, o_val)
            if raw_f <= 0:
                continue
            f = min(raw_f * kelly_fraction, max_bet_fraction)
            stake = f * pre_match_bankroll
            if stake <= 0:
                continue
            total_staked += stake
            n_bets += 1
            if results[match_idx] == outcome:
                current_bankroll += stake * (o_val - 1.0)
            else:
                current_bankroll -= stake

        if current_bankroll > peak_bankroll:
            peak_bankroll = current_bankroll
        dd = (peak_bankroll - current_bankroll) / peak_bankroll * 100 if peak_bankroll > 0 else 0.0
        if dd > max_drawdown_pct:
            max_drawdown_pct = dd

    roi_pct = (current_bankroll - starting_bankroll) / total_staked * 100 if total_staked > 0 else 0.0

    return {
        "n_bets": n_bets,
        "starting_bankroll": starting_bankroll,
        "ending_bankroll": round(current_bankroll, 2),
        "roi_pct": round(roi_pct, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "total_staked": round(total_staked, 2),
    }


def print_league_roi_breakdown_generic(
    results: np.ndarray | pd.Series,
    bet_masks: dict[str, np.ndarray],
    odds: dict[str, np.ndarray],
    league_series: np.ndarray | pd.Series,
    strategy_name: str,
) -> pd.DataFrame:
    """Outcome-generic per-league ROI breakdown.

    Args:
        results: Actual outcomes array.
        bet_masks: Dict outcome -> boolean bet masks.
        odds: Dict outcome -> decimal odds arrays.
        league_series: League identifier for each match.
        strategy_name: Name for display.

    Returns:
        DataFrame with per-league metrics.
    """
    results = np.asarray(results)
    leagues = np.asarray(league_series)

    print(f"\n--- Per-League: {strategy_name} ---")
    print(f"  {'League':<28} {'Bets':>6} {'ROI%':>8}")
    print(f"  {'-' * 45}")

    rows = []
    for league in sorted(np.unique(leagues)):
        l_mask = leagues == league
        league_masks = {
            outcome: np.asarray(bet_masks[outcome], dtype=bool) & l_mask
            for outcome in bet_masks
        }
        roi = compute_flat_stake_roi_generic(results, league_masks, odds)
        if roi["n_bets"] > 0:
            rows.append({"league": league, "n_bets": roi["n_bets"], "roi_pct": roi["roi_pct"]})
            print(f"  {league:<28} {roi['n_bets']:>6} {roi['roi_pct']:>7.2f}%")

    return pd.DataFrame(rows)


def adjust_odds_for_costs(
    odds: np.ndarray,
    cost_pct: float = 0.0,
) -> np.ndarray:
    """Reduce odds to simulate transaction costs (slippage, commission, etc.).

    Applies cost as a percentage reduction to the profit portion of odds:
        effective_odds = 1 + (odds - 1) * (1 - cost_pct / 100)

    This models a world where you capture (100 - cost_pct)% of the advertised
    edge.  A 2% cost on odds 3.00 gives effective odds 2.96.

    Args:
        odds: Decimal odds array (e.g. 2.50 means +150).
        cost_pct: Percentage cost applied to the profit portion (0-100).

    Returns:
        Adjusted odds array (always >= 1.0).
    """
    odds = np.asarray(odds, dtype=np.float64)
    if cost_pct == 0.0:
        return odds
    factor = 1.0 - cost_pct / 100.0
    adjusted = 1.0 + (odds - 1.0) * factor
    return np.maximum(adjusted, 1.0)


def compute_cost_sensitivity(
    ftr: np.ndarray | pd.Series,
    strategy_masks: dict[str, dict[str, np.ndarray]],
    b365_h: np.ndarray,
    b365_d: np.ndarray,
    b365_a: np.ndarray,
    cost_pcts: list[float],
) -> pd.DataFrame:
    """Sweep transaction cost levels across multiple strategies.

    Thin wrapper around compute_cost_sensitivity_generic for H/D/A markets.

    Args:
        ftr: Match results (H/D/A).
        strategy_masks: Dict strategy_name -> bet_masks dict.
        b365_h/d/a: B365 decimal odds arrays.
        cost_pcts: List of cost percentages to sweep.

    Returns:
        DataFrame with columns [strategy, cost_pct, n_bets, roi_pct].
    """
    odds = {
        "H": np.asarray(b365_h, dtype=np.float64),
        "D": np.asarray(b365_d, dtype=np.float64),
        "A": np.asarray(b365_a, dtype=np.float64),
    }
    return compute_cost_sensitivity_generic(ftr, strategy_masks, odds, cost_pcts)


def print_cost_sensitivity(df: pd.DataFrame) -> None:
    """Print formatted cost sensitivity table.

    Args:
        df: DataFrame from compute_cost_sensitivity().
    """
    strategies = df["strategy"].unique()
    cost_pcts = sorted(df["cost_pct"].unique())

    # Header
    cost_headers = "".join(f"{c:>7.1f}%" for c in cost_pcts)
    print(f"\n{'=' * (40 + 8 * len(cost_pcts))}")
    print("=== TRANSACTION COST SENSITIVITY (ROI% at each cost level) ===")
    print(f"{'=' * (40 + 8 * len(cost_pcts))}")
    print(f"  {'Strategy':<32} {'Bets':>6} {cost_headers}")
    print(f"  {'-' * (38 + 8 * len(cost_pcts))}")

    for strat in strategies:
        sdf = df[df["strategy"] == strat].sort_values("cost_pct")
        n_bets = sdf["n_bets"].iloc[0]
        roi_vals = "".join(f"{r:>7.2f}%" for r in sdf["roi_pct"].values)
        print(f"  {strat:<32} {n_bets:>6} {roi_vals}")

    # Find breakeven cost for each strategy
    print(f"\n  Breakeven cost (ROI crosses 0%):")
    for strat in strategies:
        sdf = df[df["strategy"] == strat].sort_values("cost_pct")
        roi_at_zero = sdf[sdf["cost_pct"] == 0.0]["roi_pct"].values
        if len(roi_at_zero) == 0 or roi_at_zero[0] <= 0:
            print(f"    {strat:<32} already negative at 0% cost")
            continue
        # Linear interpolation to find breakeven
        costs = sdf["cost_pct"].values
        rois = sdf["roi_pct"].values
        breakeven = None
        for i in range(len(rois) - 1):
            if rois[i] >= 0 and rois[i + 1] < 0:
                # Linear interp
                breakeven = costs[i] + (0 - rois[i]) * (costs[i + 1] - costs[i]) / (rois[i + 1] - rois[i])
                break
        if breakeven is not None:
            print(f"    {strat:<32} ~{breakeven:.1f}%")
        else:
            print(f"    {strat:<32} >{costs[-1]:.1f}% (still profitable)")


def compute_flat_stake_roi(
    ftr: np.ndarray | pd.Series,
    bet_masks: dict[str, np.ndarray],
    b365_h: np.ndarray,
    b365_d: np.ndarray,
    b365_a: np.ndarray,
) -> dict:
    """Compute flat-stake ROI from boolean bet masks per outcome.

    Thin wrapper around compute_flat_stake_roi_generic for H/D/A markets.

    Args:
        ftr: Match results (H/D/A).
        bet_masks: Dict {"H": bool_mask, "D": bool_mask, "A": bool_mask}.
        b365_h/d/a: B365 decimal odds arrays.

    Returns:
        Dict with n_bets, staked, returned, roi_pct, and by_outcome breakdown.
    """
    odds = {
        "H": np.asarray(b365_h, dtype=np.float64),
        "D": np.asarray(b365_d, dtype=np.float64),
        "A": np.asarray(b365_a, dtype=np.float64),
    }
    return compute_flat_stake_roi_generic(ftr, bet_masks, odds)


def compute_roi_curve(
    ftr: np.ndarray | pd.Series,
    scores: dict[str, np.ndarray],
    b365_h: np.ndarray,
    b365_d: np.ndarray,
    b365_a: np.ndarray,
    cutoffs: list[float],
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Sweep cutoffs on continuous scores and compute ROI at each.

    Args:
        ftr: Match results (H/D/A).
        scores: Dict outcome -> continuous score array (e.g. probabilities).
        b365_h/d/a: B365 decimal odds.
        cutoffs: List of cutoff values to sweep.
        higher_is_better: If True, bet when score >= cutoff. If False, <= cutoff.

    Returns:
        DataFrame with columns [cutoff, n_bets, roi_pct].
    """
    ftr = np.asarray(ftr)
    odds_map = {
        "H": np.asarray(b365_h, dtype=np.float64),
        "D": np.asarray(b365_d, dtype=np.float64),
        "A": np.asarray(b365_a, dtype=np.float64),
    }

    rows = []
    for cutoff in cutoffs:
        bet_masks = {}
        for outcome in OUTCOMES:
            s = np.asarray(scores[outcome], dtype=np.float64)
            if higher_is_better:
                bet_masks[outcome] = s >= cutoff
            else:
                bet_masks[outcome] = s <= cutoff

        roi = compute_flat_stake_roi(ftr, bet_masks, b365_h, b365_d, b365_a)
        rows.append({
            "cutoff": cutoff,
            "n_bets": roi["n_bets"],
            "roi_pct": roi["roi_pct"],
        })

    return pd.DataFrame(rows)


def compute_per_season_roi(
    ftr: np.ndarray | pd.Series,
    bet_masks: dict[str, np.ndarray],
    b365_h: np.ndarray,
    b365_d: np.ndarray,
    b365_a: np.ndarray,
    seasons: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Compute ROI per season for stability analysis.

    Thin wrapper around compute_per_season_roi_generic for H/D/A markets.

    Args:
        ftr: Match results.
        bet_masks: Dict outcome -> boolean bet masks.
        b365_h/d/a: B365 decimal odds.
        seasons: Season identifier for each match.

    Returns:
        DataFrame with columns [season, n_bets, roi_pct].
    """
    odds = {
        "H": np.asarray(b365_h, dtype=np.float64),
        "D": np.asarray(b365_d, dtype=np.float64),
        "A": np.asarray(b365_a, dtype=np.float64),
    }
    return compute_per_season_roi_generic(ftr, bet_masks, odds, seasons)


def compute_oracle_roi(
    ftr: np.ndarray | pd.Series,
    actual_clv: dict[str, np.ndarray],
    b365_h: np.ndarray,
    b365_d: np.ndarray,
    b365_a: np.ndarray,
    clv_threshold: float = 0.02,
) -> dict:
    """Profit ceiling using actual CLV (oracle strategy).

    Args:
        ftr: Match results.
        actual_clv: Dict outcome -> actual CLV arrays.
        b365_h/d/a: B365 decimal odds.
        clv_threshold: Minimum actual CLV to bet.

    Returns:
        ROI dict (same structure as compute_flat_stake_roi).
    """
    bet_masks = {}
    for outcome in OUTCOMES:
        suffix = {"H": "h", "D": "d", "A": "a"}[outcome]
        clv = np.asarray(actual_clv[outcome], dtype=np.float64)
        bet_masks[outcome] = clv > clv_threshold

    return compute_flat_stake_roi(ftr, bet_masks, b365_h, b365_d, b365_a)


def print_strategy_comparison(results: dict[str, dict], test_seasons: list[int]) -> None:
    """Print formatted comparison table for all strategies.

    Args:
        results: Dict strategy_name -> ROI result dict.
        test_seasons: List of test season years (for per-year avg).
    """
    n_years = len(test_seasons) if test_seasons else 1
    print(f"\n{'=' * 65}")
    print("=== STRATEGY COMPARISON ===")
    print(f"{'=' * 65}")
    print(f"  {'Strategy':<35} {'Bets':>7} {'ROI%':>8} {'Bets/yr':>8}")
    print(f"  {'-' * 60}")

    for name, res in results.items():
        n_bets = res["n_bets"]
        roi = res["roi_pct"]
        bets_per_yr = n_bets / n_years
        print(f"  {name:<35} {n_bets:>7} {roi:>7.2f}% {bets_per_yr:>8.0f}")


def compute_kelly_fraction(estimated_prob: float, decimal_odds: float) -> float:
    """Compute raw Kelly fraction for a single bet.

    Args:
        estimated_prob: Estimated probability of winning.
        decimal_odds: Decimal odds (e.g. 2.5 means +150).

    Returns:
        Raw Kelly fraction (can be negative; caller should clamp to [0, max]).
    """
    if decimal_odds <= 1.0:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - estimated_prob
    return (b * estimated_prob - q) / b


def compute_kelly_roi(
    ftr: np.ndarray | pd.Series,
    bet_masks: dict[str, np.ndarray],
    estimated_probs: dict[str, np.ndarray],
    b365_h: np.ndarray,
    b365_d: np.ndarray,
    b365_a: np.ndarray,
    kelly_fraction: float = 0.25,
    max_bet_fraction: float = 0.05,
    bankroll: float = 1000.0,
) -> dict:
    """Sequential bankroll simulation with Kelly sizing.

    Thin wrapper around compute_kelly_roi_generic for H/D/A markets.

    Args:
        ftr: Match results (H/D/A).
        bet_masks: Dict {"H": bool_mask, "D": bool_mask, "A": bool_mask}.
        estimated_probs: Dict {"H": prob_array, "D": prob_array, "A": prob_array}.
        b365_h/d/a: B365 decimal odds arrays.
        kelly_fraction: Fractional Kelly multiplier (e.g. 0.25 = quarter Kelly).
        max_bet_fraction: Maximum fraction of bankroll per bet.
        bankroll: Starting bankroll.

    Returns:
        Dict with n_bets, starting_bankroll, ending_bankroll, roi_pct,
        max_drawdown_pct, total_staked.
    """
    odds = {
        "H": np.asarray(b365_h, dtype=np.float64),
        "D": np.asarray(b365_d, dtype=np.float64),
        "A": np.asarray(b365_a, dtype=np.float64),
    }
    return compute_kelly_roi_generic(
        ftr, bet_masks, estimated_probs, odds,
        kelly_fraction=kelly_fraction,
        max_bet_fraction=max_bet_fraction,
        bankroll=bankroll,
    )


def print_kelly_comparison(flat_results: dict, kelly_results: dict, kelly_fraction: float) -> None:
    """Print side-by-side flat vs Kelly comparison.

    Args:
        flat_results: Dict strategy_name -> flat ROI result dict.
        kelly_results: Dict strategy_name -> Kelly ROI result dict.
        kelly_fraction: Kelly fraction used.
    """
    print(f"\n{'=' * 75}")
    print(f"=== KELLY CRITERION COMPARISON (fraction={kelly_fraction}) ===")
    print(f"{'=' * 75}")
    print(f"  {'Strategy':<35} {'Bets':>6} {'Flat ROI%':>10} {'Kelly ROI%':>11} {'Kelly DD%':>10}")
    print(f"  {'-' * 72}")

    for name in flat_results:
        flat = flat_results[name]
        kelly = kelly_results.get(name, {})
        n_bets = flat.get("n_bets", 0)
        flat_roi = flat.get("roi_pct", 0.0)
        kelly_roi = kelly.get("roi_pct", 0.0)
        kelly_dd = kelly.get("max_drawdown_pct", 0.0)
        print(f"  {name:<35} {n_bets:>6} {flat_roi:>9.2f}% {kelly_roi:>10.2f}% {kelly_dd:>9.2f}%")


def compute_closing_line_stats(
    bet_masks: dict[str, np.ndarray],
    ps_prob: dict[str, np.ndarray],
    psc_prob: dict[str, np.ndarray],
    strategy_name: str = "",
) -> dict:
    """Compute closing line value statistics for selected bets.

    For each selected bet: CLV = psc_prob - ps_prob.  Positive CLV means the
    closing line moved in our favour (the market agreed with our bet).

    Args:
        bet_masks: Dict outcome -> boolean mask of selected bets.
        ps_prob: Dict outcome -> Pinnacle opening probabilities.
        psc_prob: Dict outcome -> Pinnacle closing probabilities.
        strategy_name: Name for labelling.

    Returns:
        Dict with strategy, n_bets, avg_clv, pct_beat_closing, by_outcome.
    """
    total_clv_sum = 0.0
    total_beat = 0
    total_n = 0
    by_outcome = {}

    for outcome in OUTCOMES:
        mask = np.asarray(bet_masks.get(outcome, np.array([])), dtype=bool)
        ps = np.asarray(ps_prob.get(outcome, np.array([])), dtype=np.float64)
        psc = np.asarray(psc_prob.get(outcome, np.array([])), dtype=np.float64)

        if len(mask) == 0 or mask.sum() == 0:
            by_outcome[outcome] = {"n_bets": 0, "avg_clv": 0.0, "pct_beat_closing": 0.0}
            continue

        clv = psc[mask] - ps[mask]
        valid = np.isfinite(clv)
        n = int(valid.sum())
        if n == 0:
            by_outcome[outcome] = {"n_bets": 0, "avg_clv": 0.0, "pct_beat_closing": 0.0}
            continue

        clv_valid = clv[valid]
        avg_clv = float(np.mean(clv_valid))
        beat = int((clv_valid > 0).sum())

        by_outcome[outcome] = {
            "n_bets": n,
            "avg_clv": round(avg_clv, 5),
            "pct_beat_closing": round(beat / n * 100, 1),
        }
        total_clv_sum += clv_valid.sum()
        total_beat += beat
        total_n += n

    return {
        "strategy": strategy_name,
        "n_bets": total_n,
        "avg_clv": round(total_clv_sum / total_n, 5) if total_n > 0 else 0.0,
        "pct_beat_closing": round(total_beat / total_n * 100, 1) if total_n > 0 else 0.0,
        "by_outcome": by_outcome,
    }


def print_closing_line_report(cl_results: list[dict]) -> None:
    """Print formatted closing line analysis table.

    Args:
        cl_results: List of dicts from compute_closing_line_stats().
    """
    print(f"\n{'=' * 65}")
    print("=== CLOSING LINE ANALYSIS ===")
    print(f"{'=' * 65}")
    print(f"  {'Strategy':<35} {'Bets':>6} {'Avg CLV':>9} {'% Beat CL':>10}")
    print(f"  {'-' * 62}")

    for r in cl_results:
        name = r["strategy"]
        print(f"  {name:<35} {r['n_bets']:>6} {r['avg_clv']:>+8.4f} "
              f"{r['pct_beat_closing']:>9.1f}%")

    print(f"\n  Interpretation:")
    print(f"    Avg CLV > 0 → market moved in your favour (good signal)")
    print(f"    % Beat CL > 50% → you beat the closing line more often than not")
    print(f"    CLV is the strongest predictor of long-term betting profitability")


def print_league_roi_breakdown(
    ftr: np.ndarray | pd.Series,
    bet_masks: dict[str, np.ndarray],
    b365_h: np.ndarray,
    b365_d: np.ndarray,
    b365_a: np.ndarray,
    league_series: np.ndarray | pd.Series,
    strategy_name: str,
) -> pd.DataFrame:
    """Print per-league ROI breakdown for a given strategy.

    Thin wrapper around print_league_roi_breakdown_generic for H/D/A markets.

    Args:
        ftr: Match results.
        bet_masks: Dict outcome -> boolean bet masks.
        b365_h/d/a: B365 decimal odds.
        league_series: League identifier for each match.
        strategy_name: Name for display.

    Returns:
        DataFrame with per-league metrics.
    """
    odds = {
        "H": np.asarray(b365_h, dtype=np.float64),
        "D": np.asarray(b365_d, dtype=np.float64),
        "A": np.asarray(b365_a, dtype=np.float64),
    }
    return print_league_roi_breakdown_generic(ftr, bet_masks, odds, league_series, strategy_name)
