"""Diversification value of the L/S book and in-sample -> out-of-sample tests of Sharpe-improving changes.

Two questions are answered here:

1. Does adding the beta-neutral book to a conventional portfolio (SPY, 60/40, bonds) improve
   that portfolio's Sharpe / drawdown, and how much of the book can one hold before its own
   risk dominates? (correlations, blends, conditional performance, crisis episodes)
2. Which modelling changes raise the book's own Sharpe, and do they survive when chosen
   in-sample and evaluated out-of-sample? Every candidate is pre-declared here, not picked
   after looking at the OOS numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SampleSplit, StrategyParams
from .metrics import alpha_beta, ann_vol, cagr, max_drawdown, sharpe
from .portfolio import run_backtest
from .robustness import Inputs

ANN = 252


# --------------------------------------------------------------------------- building blocks
def sixty_forty(spy: pd.Series, agg: pd.Series) -> pd.Series:
    """Daily-rebalanced 60 % SPY / 40 % AGG."""
    return (0.6 * spy.fillna(0.0) + 0.4 * agg.fillna(0.0)).rename("60/40")


def monthly(r: pd.Series) -> pd.Series:
    return (1 + r).resample("ME").prod() - 1


def correlation_table(strat: pd.Series, assets: pd.DataFrame) -> pd.DataFrame:
    """Daily and monthly correlations of the strategy with each asset (same dates)."""
    df = pd.concat([strat.rename("strategy"), assets], axis=1).loc[strat.index]
    daily = df.corr().loc["strategy"].drop("strategy")
    mth = df.apply(monthly).corr().loc["strategy"].drop("strategy")
    return pd.DataFrame({"corr_daily": daily, "corr_monthly": mth})


def blend(strat: pd.Series, base: pd.Series, w: float, rf: pd.Series, mode: str) -> pd.Series:
    """Portfolio with weight ``w`` in the strategy.

    mode="fund":    w in the strategy, (1-w) in the base portfolio (sell base to buy the book).
    mode="overlay": 100 % base plus w of the book financed at the T-bill rate (the book's own
                    cash already earns rf, so the financing charge nets that out).
    """
    if mode == "fund":
        return (1 - w) * base + w * strat
    if mode == "overlay":
        return base + w * (strat - rf.reindex(strat.index).fillna(0.0))
    raise ValueError(mode)


def _row(r: pd.Series, rf: pd.Series, spy: pd.Series) -> dict:
    m = monthly(r)
    return dict(
        cagr=cagr(r), vol=ann_vol(r), sharpe=sharpe(r, rf), max_dd=max_drawdown(r),
        worst_month=float(m.min()), beta_spy=alpha_beta(r, spy.reindex(r.index), rf)["beta"],
    )


def blend_table(strat: pd.Series, base: pd.Series, rf: pd.Series, spy: pd.Series,
                weights: list[float], mode: str) -> pd.DataFrame:
    rows = {w: _row(blend(strat, base, w, rf, mode), rf, spy) for w in weights}
    return pd.DataFrame(rows).T.rename_axis("weight_in_strategy")


def best_weight(strat: pd.Series, base: pd.Series, rf: pd.Series, mode: str,
                grid: np.ndarray | None = None) -> float:
    """Sharpe-maximising weight over a grid (call this on IS data only, then apply OOS)."""
    grid = np.arange(0.0, 1.0001, 0.05) if grid is None else grid
    sh = [sharpe(blend(strat, base, w, rf, mode), rf) for w in grid]
    return float(grid[int(np.nanargmax(sh))])


def is_oos_blend(strat: pd.Series, base: pd.Series, rf: pd.Series, spy: pd.Series,
                 split: SampleSplit, mode: str) -> dict:
    """Pick the strategy weight in-sample, report base vs blend out-of-sample."""
    is_idx = strat.loc[split.is_start: split.is_end].index
    oos_idx = strat.loc[split.oos_start: split.oos_end].index
    w = best_weight(strat.loc[is_idx], base.loc[is_idx], rf, mode)
    out = {"weight_chosen_is": w}
    for tag, idx in (("is", is_idx), ("oos", oos_idx)):
        b, m = base.loc[idx], blend(strat.loc[idx], base.loc[idx], w, rf, mode)
        out[f"{tag}_sharpe_base"] = sharpe(b, rf)
        out[f"{tag}_sharpe_blend"] = sharpe(m, rf)
        out[f"{tag}_maxdd_base"] = max_drawdown(b)
        out[f"{tag}_maxdd_blend"] = max_drawdown(m)
        out[f"{tag}_cagr_base"] = cagr(b)
        out[f"{tag}_cagr_blend"] = cagr(m)
    return out


def bootstrap_sharpe_gain(strat: pd.Series, base: pd.Series, rf: pd.Series, w: float, mode: str,
                          n: int = 2000, block: int = 21, seed: int = 0) -> dict:
    """Stationary block bootstrap of Sharpe(blend) - Sharpe(base), resampling the two series jointly."""
    rng = np.random.default_rng(seed)
    rfa = rf.reindex(strat.index).fillna(0.0)
    m = blend(strat, base, w, rf, mode)
    ex = np.column_stack([(m - rfa).values, (base - rfa).values])
    ex = ex[~np.isnan(ex).any(axis=1)]
    T = len(ex)
    gains = np.empty(n)
    for i in range(n):
        idx = []
        while len(idx) < T:
            s = rng.integers(0, T)
            idx.extend(range(s, min(s + block, T)))
        smp = ex[np.array(idx[:T])]
        sh = smp.mean(axis=0) / smp.std(axis=0, ddof=1) * np.sqrt(ANN)
        gains[i] = sh[0] - sh[1]
    point = (ex.mean(axis=0) / ex.std(axis=0, ddof=1) * np.sqrt(ANN))
    return dict(gain=float(point[0] - point[1]), ci_low=float(np.percentile(gains, 2.5)),
                ci_high=float(np.percentile(gains, 97.5)), p_gain_le_0=float((gains <= 0).mean()))


# --------------------------------------------------------------------------- conditional behaviour
def conditional_table(strat: pd.Series, spy: pd.Series, rf: pd.Series) -> pd.DataFrame:
    """Strategy monthly returns conditioned on the SPY month (quintiles + up/down)."""
    ms, mm = monthly(strat), monthly(spy.reindex(strat.index))
    mrf = monthly(rf.reindex(strat.index).fillna(0.0))
    df = pd.DataFrame({"strat": ms, "spy": mm, "rf": mrf}).dropna()
    df["bucket"] = pd.qcut(df["spy"], 5, labels=["Q1 (worst SPY months)", "Q2", "Q3", "Q4", "Q5 (best SPY months)"])
    g = df.groupby("bucket", observed=True)
    out = g.agg(n_months=("strat", "size"), spy_avg=("spy", "mean"), strat_avg=("strat", "mean"),
                strat_hit=("strat", lambda x: (x > 0).mean()))
    for lab, mask in (("SPY down months", df["spy"] < 0), ("SPY up months", df["spy"] >= 0), ("All months", df["spy"].notna())):
        sub = df[mask]
        out.loc[lab] = [len(sub), sub["spy"].mean(), sub["strat"].mean(), (sub["strat"] > 0).mean()]
    return out


def episode_table(strat: pd.Series, assets: pd.DataFrame, episodes: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Cumulative returns over named stress windows."""
    rows = {}
    for name, s, e in episodes:
        idx = strat.loc[s:e].index
        row = {"strategy": (1 + strat.loc[idx]).prod() - 1}
        for c in assets.columns:
            row[c] = (1 + assets[c].reindex(idx).fillna(0.0)).prod() - 1
        rows[f"{name} ({s} → {e})"] = row
    return pd.DataFrame(rows).T


def rolling_correlation(strat: pd.Series, base: pd.Series, window: int = 252) -> pd.Series:
    return strat.rolling(window).corr(base.reindex(strat.index))


# --------------------------------------------------------------------------- leg-level diversification
def leg_diversification(inp: Inputs, params: StrategyParams, start: str, end: str, rf: pd.Series) -> dict:
    """Correlation between the two hedged legs and the Sharpe of each vs. the combination."""
    long = run_backtest(inp.tx, inp.signals, inp.md, params.with_(short_notional=0.0), start, end)["ret"]
    short = run_backtest(inp.tx, inp.signals, inp.md, params.with_(long_notional=0.0), start, end)["ret"]
    both = run_backtest(inp.tx, inp.signals, inp.md, params, start, end)["ret"]
    rfa = rf.reindex(both.index).fillna(0.0)
    ex_l, ex_s = long - rfa, short - rfa
    rho = float(ex_l.corr(ex_s))
    s_l, s_s = sharpe(long, rf), sharpe(short, rf)
    # textbook Sharpe of two legs combined at equal risk, given rho (no costs interaction)
    theo = (s_l + s_s) / np.sqrt(2 * (1 + rho)) if rho > -1 else np.nan
    return dict(corr_legs_daily=rho, corr_legs_monthly=float(monthly(ex_l).corr(monthly(ex_s))),
                sharpe_long=s_l, sharpe_short=s_s, sharpe_combined=sharpe(both, rf),
                sharpe_equal_risk_theory=float(theo), vol_long=ann_vol(long), vol_short=ann_vol(short))


def short_notional_sweep(inp: Inputs, params: StrategyParams, split: SampleSplit, rf: pd.Series,
                         grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)) -> pd.DataFrame:
    """How much Cramer short per unit of Pelosi long? IS and OOS Sharpe for each ratio."""
    rows = {}
    for k in grid:
        bt = run_backtest(inp.tx, inp.signals, inp.md, params.with_(short_notional=k), split.is_start, split.oos_end)
        r = bt["ret"]
        rows[k] = dict(
            is_sharpe=sharpe(r.loc[split.is_start: split.is_end], rf),
            oos_sharpe=sharpe(r.loc[split.oos_start: split.oos_end], rf),
            full_sharpe=sharpe(r, rf), full_vol=ann_vol(r), full_max_dd=max_drawdown(r),
        )
    return pd.DataFrame(rows).T.rename_axis("short_notional")


# --------------------------------------------------------------------------- Sharpe-improvement candidates
def candidate_params(base: StrategyParams) -> list[tuple[str, str, StrategyParams]]:
    """Pre-declared candidate changes (name, rationale, params). Nothing here was chosen on OOS data."""
    return [
        ("Baseline (pre-specified)", "as reported", base),
        ("Vol-target 10 %", "de-lever when trailing 63d vol is high; classic Sharpe-improver for fat-tailed books",
         base.with_(vol_target=0.10)),
        ("Vol-target 8 %", "same, lower target", base.with_(vol_target=0.08)),
        ("Hedge with QQQ", "hedge the mega-cap-growth book with its natural index", base.with_(hedge_tickers=("QQQ",))),
        ("Two-factor hedge SPY+QQQ", "neutralise market AND growth-vs-market exposure", base.with_(hedge_tickers=("SPY", "QQQ"))),
        ("Two-factor hedge SPY+IWM", "neutralise market AND size exposure", base.with_(hedge_tickers=("SPY", "IWM"))),
        ("Cramer hold 63d (lower turnover)", "cut short-leg costs; IS grid preferred 63", base.with_(cramer_hold_days=63)),
        ("Cramer hold 126d", "even lower turnover", base.with_(cramer_hold_days=126)),
        ("No Cramer leg (hedged Pelosi only)", "is the short leg worth its costs?", base.with_(short_notional=0.0)),
        ("Half-size Cramer leg", "shrink the leg that is break-even after costs", base.with_(short_notional=0.5)),
        ("Pelosi: don't exit on disclosed sales", "her sales carry no timing information", base.with_(pelosi_close_on_sale=False)),
        ("Pelosi: hold 504d", "longer hold on the leg that supplies the return", base.with_(pelosi_hold_days=504)),
        ("Pelosi: no single-name cap", "let her concentration through", base.with_(pelosi_max_weight=1.0)),
        ("Combo: vol-target 10 % + Cramer 63d + no sale-exit",
         "the structural changes that do not require choosing a hedge index post hoc",
         base.with_(vol_target=0.10, cramer_hold_days=63, pelosi_close_on_sale=False)),
    ]


def evaluate_candidates(inp: Inputs, base: StrategyParams, split: SampleSplit, rf: pd.Series,
                        bench: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, why, p in candidate_params(base):
        bt = run_backtest(inp.tx, inp.signals, inp.md, p, split.is_start, split.oos_end)
        r = bt["ret"]
        r_is, r_oos = r.loc[split.is_start: split.is_end], r.loc[split.oos_start: split.oos_end]
        ab_spy = alpha_beta(r, bench["SPY"].reindex(r.index), rf)
        ab_qqq = alpha_beta(r, bench["QQQ"].reindex(r.index), rf)
        rows.append(dict(
            candidate=name, rationale=why,
            is_sharpe=sharpe(r_is, rf), oos_sharpe=sharpe(r_oos, rf), full_sharpe=sharpe(r, rf),
            full_cagr=cagr(r), full_vol=ann_vol(r), full_max_dd=max_drawdown(r),
            beta_spy=ab_spy["beta"], beta_qqq=ab_qqq["beta"], alpha_t_spy=ab_spy["alpha_t"],
            turnover_ann=float(bt["turnover"].mean() * ANN), avg_leverage=float(bt["leverage"].mean()),
        ))
    df = pd.DataFrame(rows).set_index("candidate")
    b = df.loc["Baseline (pre-specified)"]
    df["is_gain"] = df["is_sharpe"] - b["is_sharpe"]
    df["oos_gain"] = df["oos_sharpe"] - b["oos_sharpe"]
    return df


def candidate_returns(inp: Inputs, base: StrategyParams, start: str, end: str, names: list[str]) -> pd.DataFrame:
    """Daily return series for a subset of candidates (for plotting)."""
    lookup = {n: p for n, _, p in candidate_params(base)}
    return pd.DataFrame({n: run_backtest(inp.tx, inp.signals, inp.md, lookup[n], start, end)["ret"] for n in names})
