"""Plot portfolio bankroll curve and per-market breakdown.

CLI: python -m src.evaluation.plot_portfolio
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PORTFOLIO_DIR = PROJECT_ROOT / "outputs" / "portfolio"


def plot_bankroll_curve(
    curve_path: Path | None = None,
    log_path: Path | None = None,
    output_path: Path | None = None,
) -> None:
    """Plot bankroll over time with per-market annotations."""
    curve_path = curve_path or PORTFOLIO_DIR / "bankroll_curve.parquet"
    log_path = log_path or PORTFOLIO_DIR / "per_bet_log.parquet"
    output_path = output_path or PORTFOLIO_DIR / "bankroll_curve.png"

    curve = pd.read_parquet(curve_path)
    curve["date"] = pd.to_datetime(curve["date"])

    log = pd.read_parquet(log_path)
    log["date"] = pd.to_datetime(log["date"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[3, 1.2, 1.2])
    fig.suptitle("Portfolio Simulation: 2021-2024 (3.8% commission, quarter-Kelly)",
                 fontsize=14, fontweight="bold", y=0.98)

    # --- Panel 1: Bankroll curve ---
    ax1 = axes[0]
    ax1.plot(curve["date"], curve["bankroll"], color="#2563eb", linewidth=1.5, label="Bankroll")
    ax1.axhline(y=1000, color="#94a3b8", linestyle="--", linewidth=0.8, label="Starting bankroll")

    # Shade drawdown periods
    peak = curve["bankroll"].cummax()
    drawdown = (peak - curve["bankroll"]) / peak * 100
    dd_mask = drawdown > 0
    ax1.fill_between(
        curve["date"], curve["bankroll"], peak,
        where=dd_mask, alpha=0.15, color="#ef4444", label="Drawdown",
    )

    # Mark max drawdown point
    max_dd_idx = drawdown.idxmax()
    ax1.annotate(
        f"Max DD: {drawdown.iloc[max_dd_idx]:.1f}%",
        xy=(curve["date"].iloc[max_dd_idx], curve["bankroll"].iloc[max_dd_idx]),
        xytext=(30, -30), textcoords="offset points",
        fontsize=9, color="#ef4444",
        arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.2),
    )

    # End annotation
    final = curve["bankroll"].iloc[-1]
    profit = final - 1000
    ax1.annotate(
        f"Final: {final:,.0f} (+{profit:,.0f})",
        xy=(curve["date"].iloc[-1], final),
        xytext=(-120, 20), textcoords="offset points",
        fontsize=10, fontweight="bold", color="#16a34a",
        arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.2),
    )

    ax1.set_ylabel("Bankroll", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # --- Panel 2: Cumulative PnL by market ---
    ax2 = axes[1]
    market_colors = {"1X2": "#2563eb", "O/U": "#16a34a", "Corners": "#f59e0b"}

    for market in ["1X2", "O/U", "Corners"]:
        mlog = log[log["market"] == market].sort_values("date")
        if mlog.empty:
            continue
        cum_pnl = mlog["pnl"].cumsum()
        ax2.plot(mlog["date"], cum_pnl, color=market_colors[market],
                 linewidth=1.3, label=market)

    ax2.axhline(y=0, color="#94a3b8", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Cumulative PnL", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # --- Panel 3: Monthly bet count ---
    ax3 = axes[2]
    log["month"] = log["date"].dt.to_period("M")
    monthly = log.groupby(["month", "market"]).size().unstack(fill_value=0)
    months = monthly.index.to_timestamp()

    bottom = pd.Series(0.0, index=monthly.index)
    for market in ["1X2", "O/U", "Corners"]:
        if market in monthly.columns:
            ax3.bar(months, monthly[market], bottom=bottom.values,
                    width=20, color=market_colors[market], label=market, alpha=0.8)
            bottom += monthly[market]

    ax3.set_ylabel("Bets/month", fontsize=11)
    ax3.set_xlabel("Date", fontsize=11)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_bankroll_curve()
