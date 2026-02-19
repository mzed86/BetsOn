"""Evaluation metrics for match outcome prediction models."""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


CLASSES = ["H", "D", "A"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
# sklearn's log_loss sorts labels alphabetically; precompute reorder indices
_SORTED_CLASSES = sorted(CLASSES)  # ["A", "D", "H"]
_SORTED_IDX = [CLASSES.index(c) for c in _SORTED_CLASSES]


def compute_log_loss(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Compute multi-class log-loss.

    Args:
        y_true: Series of outcome labels (H/D/A).
        y_prob: (N, 3) array of probabilities [P(H), P(D), P(A)].

    Returns:
        Log-loss value.
    """
    return log_loss(y_true, y_prob[:, _SORTED_IDX], labels=_SORTED_CLASSES)


def compute_brier_score(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Compute multi-class Brier score: mean of per-class (p_i - y_i)^2.

    Args:
        y_true: Series of outcome labels (H/D/A).
        y_prob: (N, 3) array of probabilities [P(H), P(D), P(A)].

    Returns:
        Brier score (lower is better, range [0, 2]).
    """
    y_onehot = np.zeros_like(y_prob)
    for i, label in enumerate(y_true):
        y_onehot[i, CLASS_IDX[label]] = 1.0
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def compute_calibration(
    y_true: pd.Series, y_prob: np.ndarray, n_bins: int = 10
) -> dict:
    """Compute calibration data per outcome class.

    Bins predicted probabilities and computes observed frequency in each bin.

    Args:
        y_true: Series of outcome labels (H/D/A).
        y_prob: (N, 3) array of probabilities.
        n_bins: Number of bins.

    Returns:
        Dict keyed by class label, each containing:
            - bin_centers: array of bin midpoints
            - pred_probs: mean predicted probability per bin
            - obs_freqs: observed frequency per bin
            - counts: number of samples per bin
    """
    result = {}
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for cls_name, cls_idx in CLASS_IDX.items():
        probs = y_prob[:, cls_idx]
        actuals = (y_true.values == cls_name).astype(float)

        bin_centers = []
        pred_probs = []
        obs_freqs = []
        counts = []

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            if hi == bin_edges[-1]:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs >= lo) & (probs < hi)

            n = mask.sum()
            if n == 0:
                continue

            bin_centers.append((lo + hi) / 2)
            pred_probs.append(float(probs[mask].mean()))
            obs_freqs.append(float(actuals[mask].mean()))
            counts.append(int(n))

        result[cls_name] = {
            "bin_centers": np.array(bin_centers),
            "pred_probs": np.array(pred_probs),
            "obs_freqs": np.array(obs_freqs),
            "counts": np.array(counts),
        }

    return result


def compute_roi(
    y_true: pd.Series,
    y_prob: np.ndarray,
    odds_h: np.ndarray,
    odds_d: np.ndarray,
    odds_a: np.ndarray,
    threshold: float = 0.05,
    stake: float = 1.0,
) -> dict:
    """Simulate flat-stake betting where model_prob - implied_prob > threshold.

    Args:
        y_true: Series of outcome labels (H/D/A).
        y_prob: (N, 3) array of probabilities.
        odds_h: Array of decimal odds for home win.
        odds_d: Array of decimal odds for draw.
        odds_a: Array of decimal odds for away win.
        threshold: Minimum edge (model_prob - implied_prob) to place bet.
        stake: Flat stake per bet.

    Returns:
        Dict with n_bets, total_staked, total_return, roi_pct, and
        per-outcome breakdown.
    """
    odds_arr = np.column_stack([odds_h, odds_d, odds_a])
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = np.where(odds_arr > 0, 1.0 / odds_arr, 0.0)
    edge = y_prob - implied

    results = {"by_outcome": {}}
    total_bets = 0
    total_staked = 0.0
    total_return = 0.0

    for cls_name, cls_idx in CLASS_IDX.items():
        bet_mask = edge[:, cls_idx] > threshold
        # Also require valid odds
        valid_mask = bet_mask & np.isfinite(odds_arr[:, cls_idx])
        n_bets = int(valid_mask.sum())

        if n_bets == 0:
            results["by_outcome"][cls_name] = {
                "n_bets": 0,
                "staked": 0.0,
                "returned": 0.0,
                "roi_pct": 0.0,
            }
            continue

        staked = n_bets * stake
        wins = (y_true.values[valid_mask] == cls_name)
        returned = float(np.sum(wins * odds_arr[valid_mask, cls_idx] * stake))

        results["by_outcome"][cls_name] = {
            "n_bets": n_bets,
            "staked": staked,
            "returned": round(returned, 2),
            "roi_pct": round((returned - staked) / staked * 100, 2) if staked > 0 else 0.0,
        }

        total_bets += n_bets
        total_staked += staked
        total_return += returned

    results["n_bets"] = total_bets
    results["total_staked"] = total_staked
    results["total_return"] = round(total_return, 2)
    results["roi_pct"] = (
        round((total_return - total_staked) / total_staked * 100, 2)
        if total_staked > 0
        else 0.0
    )

    return results


def compute_accuracy(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Compute classification accuracy from probability predictions.

    Args:
        y_true: Series of outcome labels (H/D/A).
        y_prob: (N, 3) array of probabilities.

    Returns:
        Accuracy as a fraction.
    """
    pred_labels = np.array(CLASSES)[np.argmax(y_prob, axis=1)]
    return float(np.mean(pred_labels == y_true.values))


def print_evaluation_report(
    y_true: pd.Series,
    y_prob: np.ndarray,
    odds_df: pd.DataFrame,
    model_name: str,
) -> dict:
    """Print a formatted evaluation report and return metrics dict.

    Args:
        y_true: Series of outcome labels (H/D/A).
        y_prob: (N, 3) array of probabilities.
        odds_df: DataFrame with columns B365H, B365D, B365A.
        model_name: Name for display.

    Returns:
        Dict of computed metrics.
    """
    ll = compute_log_loss(y_true, y_prob)
    bs = compute_brier_score(y_true, y_prob)
    acc = compute_accuracy(y_true, y_prob)

    roi = compute_roi(
        y_true,
        y_prob,
        odds_df["B365H"].values,
        odds_df["B365D"].values,
        odds_df["B365A"].values,
    )

    print(f"\n--- {model_name} ---")
    print(f"  Log-loss:  {ll:.4f}")
    print(f"  Brier:     {bs:.4f}")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  ROI (5% edge threshold):")
    print(f"    Bets placed: {roi['n_bets']}")
    print(f"    Total staked: {roi['total_staked']:.0f}")
    print(f"    Total return: {roi['total_return']:.2f}")
    print(f"    ROI: {roi['roi_pct']:.2f}%")
    for cls_name in CLASSES:
        r = roi["by_outcome"][cls_name]
        print(f"      {cls_name}: {r['n_bets']} bets, ROI {r['roi_pct']:.2f}%")

    return {"log_loss": ll, "brier_score": bs, "accuracy": acc, "roi": roi}
