"""Figures and the markdown report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import drawdown_series, rolling_beta  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path.name


def fig_growth(bt: pd.DataFrame, bench: pd.DataFrame, split_date: str | None, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    nav = (1 + bt["ret"]).cumprod()
    ax.plot(nav, label="Long Pelosi / Short Cramer (net, beta-hedged)", lw=2, color="black")
    for c in bench.columns:
        ax.plot((1 + bench[c].reindex(bt.index).fillna(0)).cumprod(), label=c, lw=1, alpha=0.8)
    if split_date:
        ax.axvline(pd.Timestamp(split_date), color="red", ls="--", lw=1)
        ax.text(pd.Timestamp(split_date), ax.get_ylim()[1] * 0.95, "  OOS →", color="red")
    ax.set_yscale("log")
    ax.set_title("Growth of $1 (log scale)")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, path)


def fig_legs(bt: pd.DataFrame, path: Path) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax = axes[0]
    for col, lab in [("ret_long", "Long leg (Pelosi)"), ("ret_short", "Short leg (−Cramer)"),
                     ("ret_hedge", "SPY hedge overlay"), ("ret", "Net portfolio")]:
        ax.plot((1 + bt[col]).cumprod() - 1, label=lab, lw=1.5 if col == "ret" else 1)
    ax.set_title("Cumulative contribution by book")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(bt["n_long"], label="# long names", color="tab:green")
    ax.set_ylabel("# long")
    ax2 = ax.twinx()
    ax2.plot(bt["n_short"], label="# short names", color="tab:red", alpha=0.7)
    ax2.set_ylabel("# short")
    ax2.grid(False)
    ax.set_title("Number of open positions")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    return _save(fig, path)


def fig_drawdown(bt: pd.DataFrame, bench: pd.Series, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(bt.index, drawdown_series(bt["ret"]), 0, color="black", alpha=0.5, label="Strategy")
    ax.plot(drawdown_series(bench.reindex(bt.index).fillna(0)), color="tab:blue", lw=1, label="SPY")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Drawdown")
    ax.legend(fontsize=8)
    return _save(fig, path)


def fig_beta(bt: pd.DataFrame, bench: pd.DataFrame, tol: float, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in bench.columns:
        rb = rolling_beta(bt["ret"], bench[c].reindex(bt.index), 252)
        ax.plot(rb, label=f"realised 1y beta vs {c}", lw=1)
    ax.plot(bt["net_beta_exante"].rolling(21).mean(), color="black", ls=":", lw=1, label="ex-ante net beta (21d avg)")
    ax.axhspan(-tol, tol, color="green", alpha=0.12, label=f"±{tol:.2f} band")
    ax.set_ylim(-0.6, 0.6)
    ax.set_title("Market beta of the portfolio")
    ax.legend(fontsize=8, ncol=3)
    return _save(fig, path)


def fig_rolling_sharpe(bt: pd.DataFrame, path: Path, window: int = 252) -> str:
    ex = bt["ret"] - bt["rf"]
    rs = ex.rolling(window).mean() / ex.rolling(window).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(rs, color="black")
    ax.axhline(0, color="grey", lw=1)
    ax.set_title(f"Rolling {window}-day Sharpe")
    return _save(fig, path)


def fig_grid(grid: pd.DataFrame, path: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.scatter(grid["is_sharpe"], grid["oos_sharpe"], s=18, alpha=0.7)
    ax.scatter(grid["is_sharpe"].iloc[0], grid["oos_sharpe"].iloc[0], s=90, color="red", label="selected (best IS)")
    lim = [min(grid[["is_sharpe", "oos_sharpe"]].min()) - 0.2, max(grid[["is_sharpe", "oos_sharpe"]].max()) + 0.2]
    ax.plot(lim, lim, color="grey", lw=1, ls="--")
    ax.set_xlabel("In-sample Sharpe")
    ax.set_ylabel("Out-of-sample Sharpe")
    ax.set_title(f"Parameter grid: IS vs OOS (Spearman ρ = {grid.attrs.get('is_oos_spearman', np.nan):.2f})")
    ax.legend(fontsize=8)
    ax = axes[1]
    piv = grid.pivot_table(index="pelosi_hold_days", columns="cramer_hold_days", values="oos_sharpe", aggfunc="mean")
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("Cramer hold (sessions)")
    ax.set_ylabel("Pelosi max hold (sessions)")
    ax.set_title("OOS Sharpe (avg over beta windows)")
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def fig_placebo(actual: float, placebos: dict[str, pd.Series], path: Path) -> str:
    fig, axes = plt.subplots(1, len(placebos), figsize=(4.5 * len(placebos), 3.6), squeeze=False)
    for ax, (name, s) in zip(axes[0], placebos.items()):
        ax.hist(s.dropna(), bins=30, color="grey", alpha=0.8)
        ax.axvline(actual, color="red", lw=2, label=f"actual = {actual:.2f}")
        p = float((s >= actual).mean())
        ax.set_title(f"{name}\np(placebo ≥ actual) = {p:.3f}", fontsize=9)
        ax.set_xlabel("Sharpe")
        ax.legend(fontsize=8)
    return _save(fig, path)


def fig_lag(lag: pd.DataFrame, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    colors = ["tab:red"] + ["tab:blue"] * (len(lag) - 1)
    ax.barh(lag["variant"], lag["sharpe"], color=colors)
    ax.set_xlabel("Sharpe (full L/S portfolio)")
    ax.set_title("Cost of the disclosure lag (red = lookahead, not implementable)")
    ax.invert_yaxis()
    return _save(fig, path)


def fig_walk_forward(wf_ret: pd.Series, bench: pd.Series, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot((1 + wf_ret).cumprod(), color="black", lw=2, label="Walk-forward OOS (params re-chosen yearly)")
    ax.plot((1 + bench.reindex(wf_ret.index).fillna(0)).cumprod(), lw=1, label="SPY")
    ax.set_title("Walk-forward out-of-sample growth of $1")
    ax.legend(fontsize=8)
    return _save(fig, path)


def fig_yearly(yr: pd.DataFrame, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    yr.plot.bar(ax=ax, width=0.85)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Calendar-year returns")
    ax.legend(fontsize=8, ncol=4)
    ax.axhline(0, color="black", lw=0.8)
    return _save(fig, path)


def fig_lag_hist(tx: pd.DataFrame, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    lag = tx["disclosure_lag_days"].clip(upper=120)
    ax.hist(lag, bins=40, color="tab:purple", alpha=0.85)
    ax.axvline(45, color="red", ls="--", label="STOCK Act 45-day limit")
    ax.set_xlabel("days from trade to public filing (clipped at 120)")
    ax.set_title("Pelosi PTRs: how late does the information arrive?")
    ax.legend(fontsize=8)
    return _save(fig, path)


# ---------------------------------------------------------------- diversification figures
def fig_frontier(strat: pd.Series, bases: dict[str, pd.Series], rf: pd.Series, weights, path: Path) -> str:
    """Risk/return of blends: each curve moves from 100 % base (w=0) to 100 % strategy (w=1)."""
    from .diversification import blend
    from .metrics import ann_vol, cagr

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, base in bases.items():
        pts = [(ann_vol(blend(strat, base, w, rf, "fund")), cagr(blend(strat, base, w, rf, "fund"))) for w in weights]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", ms=3, label=f"{name} → strategy")
        for w, x, y in zip(weights, xs, ys):
            if w in (0.0, 0.2, 0.5):
                ax.annotate(f"{w:.0%}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.scatter([ann_vol(strat)], [cagr(strat)], color="black", zorder=5, label="100 % L/S book")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("annualised volatility")
    ax.set_ylabel("CAGR")
    ax.set_title("Blending the L/S book into conventional portfolios (labels = weight in the book)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def fig_rolling_corr(corrs: dict[str, pd.Series], path: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for name, s in corrs.items():
        ax.plot(s, lw=1.2, label=name)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-1, 1)
    ax.set_title("Rolling 1-year correlation of the L/S book with conventional assets")
    ax.legend(fontsize=8, ncol=len(corrs))
    return _save(fig, path)


def fig_conditional(cond: pd.DataFrame, path: Path) -> str:
    q = cond.iloc[:5]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(q))
    ax.bar(x - 0.2, q["spy_avg"], width=0.4, label="SPY avg monthly return", color="tab:blue")
    ax.bar(x + 0.2, q["strat_avg"], width=0.4, label="L/S book avg monthly return", color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i).replace(" (", "\n(") for i in q.index], fontsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("Strategy performance conditional on the SPY month (quintiles)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def fig_candidates(rets: pd.DataFrame, split_date: str, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in rets.columns:
        ax.plot((1 + rets[c]).cumprod(), lw=2 if c.startswith("Baseline") else 1.2, label=c,
                color="black" if c.startswith("Baseline") else None)
    ax.axvline(pd.Timestamp(split_date), color="red", ls="--", lw=1)
    ax.text(pd.Timestamp(split_date), ax.get_ylim()[1] * 0.97, "  OOS →", color="red")
    ax.set_title("Sharpe-improvement candidates: growth of $1 (all chosen before looking at OOS)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def fig_candidate_bars(df: pd.DataFrame, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(df))
    ax.barh(y - 0.2, df["is_sharpe"], height=0.4, label="in-sample 2018–21", color="tab:blue")
    ax.barh(y + 0.2, df["oos_sharpe"], height=0.4, label="out-of-sample 2022–24", color="tab:orange")
    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Sharpe")
    ax.set_title("Candidate modifications: does the IS improvement survive OOS?")
    ax.legend(fontsize=8)
    return _save(fig, path)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)
