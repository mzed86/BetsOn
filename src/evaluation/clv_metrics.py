"""CLV-specific evaluation metrics.

Measures for assessing how well the CLV model predicts line movement and
whether those predictions translate into profitable betting.
"""

import numpy as np
import pandas as pd
from scipy import stats


OUTCOMES = ["H", "D", "A"]
OUTCOME_IDX = {c: i for i, c in enumerate(OUTCOMES)}


def compute_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions that correctly predict the sign of line movement.

    Args:
        y_true: Actual CLV values (closing - opening prob).
        y_pred: Predicted CLV values.

    Returns:
        Accuracy in [0, 1].  Baseline is ~50% (random sign prediction).
        Samples where y_true == 0 are excluded.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    # Exclude NaN and zero-movement matches
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    nonzero = valid & (y_true != 0)
    if nonzero.sum() == 0:
        return 0.0
    correct = np.sign(y_true[nonzero]) == np.sign(y_pred[nonzero])
    return float(correct.mean())


def compute_clv_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Pearson and Spearman correlation between predicted and actual CLV.

    Args:
        y_true: Actual CLV values.
        y_pred: Predicted CLV values.

    Returns:
        Dict with 'pearson' and 'spearman' correlation coefficients.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    # Filter out NaN pairs
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 3:
        return {"pearson": 0.0, "spearman": 0.0}
    pearson_r, _ = stats.pearsonr(y_true[valid], y_pred[valid])
    spearman_r, _ = stats.spearmanr(y_true[valid], y_pred[valid])
    return {"pearson": float(pearson_r), "spearman": float(spearman_r)}


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error of CLV predictions.

    Args:
        y_true: Actual CLV values.
        y_pred: Predicted CLV values.

    Returns:
        MAE (in probability points, e.g. 0.02 = 2pp).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(y_true[valid] - y_pred[valid])))


def compute_profitable_clv(
    y_true_clv: np.ndarray,
    y_pred_clv: np.ndarray,
    ftr: pd.Series | np.ndarray,
    b365_odds: np.ndarray,
    threshold: float = 0.02,
) -> dict:
    """ROI from betting on a single outcome where predicted CLV exceeds threshold.

    When the model predicts the line will move favorably (CLV > threshold),
    we bet at B365 odds.  If the outcome wins, we collect the payout.

    Args:
        y_true_clv: Actual CLV for this outcome (unused for betting, for analysis).
        y_pred_clv: Predicted CLV for this outcome.
        ftr: Match results (H/D/A).
        b365_odds: B365 decimal odds for this outcome.
        threshold: Minimum predicted CLV to trigger a bet.

    Returns:
        Dict with n_bets, staked, returned, roi_pct, avg_pred_clv, avg_true_clv.
    """
    y_pred_clv = np.asarray(y_pred_clv, dtype=np.float64)
    y_true_clv = np.asarray(y_true_clv, dtype=np.float64)
    b365_odds = np.asarray(b365_odds, dtype=np.float64)
    ftr = np.asarray(ftr)

    bet_mask = y_pred_clv > threshold
    valid_mask = bet_mask & np.isfinite(b365_odds) & (b365_odds > 0)
    n_bets = int(valid_mask.sum())

    if n_bets == 0:
        return {
            "n_bets": 0, "staked": 0.0, "returned": 0.0, "roi_pct": 0.0,
            "avg_pred_clv": 0.0, "avg_true_clv": 0.0,
        }

    staked = float(n_bets)
    # We don't know which specific outcome this is for from ftr alone,
    # so the caller must pass the correct outcome_label to filter wins.
    # For simplicity, this function is called per-outcome with pre-filtered ftr.
    returned = float(np.sum(b365_odds[valid_mask]))  # placeholder — see note below

    return {
        "n_bets": n_bets,
        "staked": staked,
        "returned": round(returned, 2),
        "roi_pct": round((returned - staked) / staked * 100, 2),
        "avg_pred_clv": round(float(y_pred_clv[valid_mask].mean()), 4),
        "avg_true_clv": round(float(y_true_clv[valid_mask].mean()), 4),
    }


def compute_clv_bet_roi(
    ftr: pd.Series | np.ndarray,
    pred_clv: dict[str, np.ndarray],
    b365_odds_h: np.ndarray,
    b365_odds_d: np.ndarray,
    b365_odds_a: np.ndarray,
    clv_threshold: float = 0.02,
) -> dict:
    """ROI from CLV-only betting: bet on outcome where predicted CLV > threshold.

    Args:
        ftr: Match results (H/D/A).
        pred_clv: Dict mapping outcome -> predicted CLV array {"H": ..., "D": ..., "A": ...}.
        b365_odds_h: B365 home odds.
        b365_odds_d: B365 draw odds.
        b365_odds_a: B365 away odds.
        clv_threshold: Minimum predicted CLV to bet.

    Returns:
        Dict with n_bets, staked, returned, roi_pct, and per-outcome breakdown.
    """
    ftr = np.asarray(ftr)
    odds_map = {"H": np.asarray(b365_odds_h), "D": np.asarray(b365_odds_d), "A": np.asarray(b365_odds_a)}

    total_bets = 0
    total_staked = 0.0
    total_returned = 0.0
    by_outcome = {}

    for outcome in OUTCOMES:
        clv = np.asarray(pred_clv[outcome], dtype=np.float64)
        odds = odds_map[outcome]
        bet_mask = (clv > clv_threshold) & np.isfinite(odds) & (odds > 0)
        n_bets = int(bet_mask.sum())

        if n_bets == 0:
            by_outcome[outcome] = {"n_bets": 0, "staked": 0.0, "returned": 0.0, "roi_pct": 0.0}
            continue

        staked = float(n_bets)
        wins = ftr[bet_mask] == outcome
        returned = float(np.sum(wins * odds[bet_mask]))

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


def compute_value_bet_roi(
    ftr: pd.Series | np.ndarray,
    outcome_probs: np.ndarray,
    b365_odds_h: np.ndarray,
    b365_odds_d: np.ndarray,
    b365_odds_a: np.ndarray,
    pred_clv: dict[str, np.ndarray],
    clv_threshold: float = 0.02,
    edge_threshold: float = 0.03,
    min_signals: int = 2,
) -> dict:
    """Multi-signal value betting: bet only when multiple signals agree.

    Signals:
        1. CLV signal: predicted CLV > clv_threshold
        2. Edge signal: outcome model prob > B365 implied prob + edge_threshold
        3. Odds disagreement: pred_clv contains B365-Pinnacle disagreement implicitly
           (if odds_disagree features are in the CLV model inputs)

    We bet on outcome X when at least `min_signals` of {CLV, Edge} agree.
    (Odds disagreement is captured within the CLV model's features.)

    Args:
        ftr: Match results (H/D/A).
        outcome_probs: (N, 3) array of outcome model probs [P(H), P(D), P(A)].
        b365_odds_h/d/a: B365 decimal odds.
        pred_clv: Dict {"H": ..., "D": ..., "A": ...} of predicted CLV arrays.
        clv_threshold: Min predicted CLV for signal 1.
        edge_threshold: Min edge for signal 2.
        min_signals: Min signals that must agree to bet.

    Returns:
        Dict with n_bets, staked, returned, roi_pct, and per-outcome breakdown.
    """
    ftr = np.asarray(ftr)
    odds_arr = np.column_stack([
        np.asarray(b365_odds_h, dtype=np.float64),
        np.asarray(b365_odds_d, dtype=np.float64),
        np.asarray(b365_odds_a, dtype=np.float64),
    ])
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = np.where(odds_arr > 0, 1.0 / odds_arr, 0.0)

    total_bets = 0
    total_staked = 0.0
    total_returned = 0.0
    by_outcome = {}

    for outcome, idx in OUTCOME_IDX.items():
        clv = np.asarray(pred_clv[outcome], dtype=np.float64)

        # Signal 1: CLV prediction
        clv_signal = clv > clv_threshold
        # Signal 2: Outcome model edge
        edge_signal = outcome_probs[:, idx] - implied[:, idx] > edge_threshold

        signal_count = clv_signal.astype(int) + edge_signal.astype(int)
        bet_mask = (signal_count >= min_signals) & np.isfinite(odds_arr[:, idx]) & (odds_arr[:, idx] > 0)
        n_bets = int(bet_mask.sum())

        if n_bets == 0:
            by_outcome[outcome] = {"n_bets": 0, "staked": 0.0, "returned": 0.0, "roi_pct": 0.0}
            continue

        staked = float(n_bets)
        wins = ftr[bet_mask] == outcome
        returned = float(np.sum(wins * odds_arr[bet_mask, idx]))

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


def print_clv_report(
    y_true: dict[str, np.ndarray],
    y_pred: dict[str, np.ndarray],
    ftr: pd.Series,
    b365_odds_h: np.ndarray,
    b365_odds_d: np.ndarray,
    b365_odds_a: np.ndarray,
    model_name: str = "CLV Model",
) -> dict:
    """Print a formatted CLV evaluation report.

    Args:
        y_true: Dict mapping outcome -> actual CLV array.
        y_pred: Dict mapping outcome -> predicted CLV array.
        ftr: Match results (H/D/A).
        b365_odds_h/d/a: B365 decimal odds.
        model_name: Name for display.

    Returns:
        Dict of computed metrics.
    """
    print(f"\n{'=' * 60}")
    print(f"=== {model_name} ===")
    print(f"{'=' * 60}")

    metrics = {}
    for outcome in OUTCOMES:
        yt = np.asarray(y_true[outcome])
        yp = np.asarray(y_pred[outcome])

        dir_acc = compute_direction_accuracy(yt, yp)
        corr = compute_clv_correlation(yt, yp)
        mae = compute_mae(yt, yp)

        metrics[outcome] = {
            "direction_accuracy": dir_acc,
            "pearson": corr["pearson"],
            "spearman": corr["spearman"],
            "mae": mae,
        }

        print(f"\n  {outcome}: direction_acc={dir_acc:.4f}  pearson={corr['pearson']:.4f}"
              f"  spearman={corr['spearman']:.4f}  MAE={mae:.4f}")

    # CLV-only betting ROI
    clv_roi = compute_clv_bet_roi(ftr, y_pred, b365_odds_h, b365_odds_d, b365_odds_a)
    metrics["clv_roi"] = clv_roi

    print(f"\n  CLV-only betting (threshold=0.02):")
    print(f"    Bets: {clv_roi['n_bets']}  ROI: {clv_roi['roi_pct']:.2f}%")
    for outcome in OUTCOMES:
        r = clv_roi["by_outcome"][outcome]
        print(f"      {outcome}: {r['n_bets']} bets, ROI {r['roi_pct']:.2f}%")

    return metrics
