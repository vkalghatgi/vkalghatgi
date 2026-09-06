"""Overfitting controls: IS/OOS grid, walk-forward, placebos, lag sensitivity, bootstrap."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PARAM_GRID, SampleSplit, StrategyParams
from .metrics import cagr, max_drawdown, sharpe
from .portfolio import MarketData, run_backtest
from .signals import cramer_weights, pelosi_weights


@dataclass
class Inputs:
    tx: pd.DataFrame
    signals: pd.DataFrame
    md: MarketData


def _stats(r: pd.Series, rf: pd.Series) -> dict:
    return dict(sharpe=sharpe(r, rf), cagr=cagr(r), max_dd=max_drawdown(r), n_days=len(r))


# --------------------------------------------------------------------------- grid search
def grid_search(inp: Inputs, base: StrategyParams, split: SampleSplit, grid: dict | None = None) -> pd.DataFrame:
    """Evaluate every parameter combination in-sample and out-of-sample.

    Selection is done on IS Sharpe only; OOS columns are reported, never used
    for selection. The Spearman correlation between IS and OOS Sharpe across
    the grid tells you whether IS performance predicts OOS performance at all.
    """
    grid = grid or PARAM_GRID
    keys = list(grid)
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        p = base.with_(**dict(zip(keys, combo)))
        bt = run_backtest(inp.tx, inp.signals, inp.md, p, start=split.is_start, end=split.oos_end)
        is_r = bt.loc[split.is_start: split.is_end, "ret"]
        oos_r = bt.loc[split.oos_start: split.oos_end, "ret"]
        row = dict(zip(keys, combo))
        row.update({f"is_{k}": v for k, v in _stats(is_r, bt["rf"]).items()})
        row.update({f"oos_{k}": v for k, v in _stats(oos_r, bt["rf"]).items()})
        row["is_mean_abs_beta"] = float(bt.loc[split.is_start: split.is_end, "net_beta_exante"].abs().mean())
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("is_sharpe", ascending=False).reset_index(drop=True)
    df.attrs["is_oos_spearman"] = float(df[["is_sharpe", "oos_sharpe"]].corr(method="spearman").iloc[0, 1])
    return df


def select_params(base: StrategyParams, grid_df: pd.DataFrame, grid: dict | None = None) -> StrategyParams:
    grid = grid or PARAM_GRID
    best = grid_df.iloc[0]
    return base.with_(**{k: type(grid[k][0])(best[k]) for k in grid})


# --------------------------------------------------------------------------- walk-forward
def walk_forward(
    inp: Inputs, base: StrategyParams, first_oos_year: int, last_year: int, min_train_years: int = 2,
    grid: dict | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Expanding-window walk-forward: each year's parameters are chosen using only prior years.

    Returns the stitched OOS return series and a table of chosen parameters per year.
    """
    grid = grid or PARAM_GRID
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    # run every combo once over the full horizon, then slice - identical results, far cheaper
    runs = {}
    for combo in combos:
        p = base.with_(**dict(zip(keys, combo)))
        runs[combo] = run_backtest(inp.tx, inp.signals, inp.md, p)
    rf = next(iter(runs.values()))["rf"]
    pieces, chosen = [], []
    for y in range(first_oos_year, last_year + 1):
        tr_start, tr_end = f"{y - min_train_years - 10}-01-01", f"{y - 1}-12-31"
        scores = {c: sharpe(bt.loc[tr_start: tr_end, "ret"], rf) for c, bt in runs.items()}
        best = max(scores, key=lambda c: (np.nan_to_num(scores[c], nan=-9), c))
        seg = runs[best].loc[f"{y}-01-01": f"{y}-12-31", "ret"]
        pieces.append(seg)
        chosen.append(dict(year=y, **dict(zip(keys, best)), train_sharpe=scores[best],
                           oos_year_return=(1 + seg).prod() - 1, oos_year_sharpe=sharpe(seg, rf)))
    return pd.concat(pieces), pd.DataFrame(chosen)


# --------------------------------------------------------------------------- lag sensitivity
def lag_sensitivity(inp: Inputs, params: StrategyParams, start: str, end: str) -> pd.DataFrame:
    """How much does the disclosure lag cost? Compare the *illegal* trade-date entry with filing-date entry and extra delays."""
    rows = []
    variants = [
        ("trade date (LOOKAHEAD - not implementable)", dict(use_transaction_date=True), 0),
        ("filing date + 1 session (baseline)", {}, 0),
        ("filing date + 5 sessions", {}, 5),
        ("filing date + 21 sessions", {}, 21),
        ("filing date + 45 sessions", {}, 45),
    ]
    for label, kw, lag in variants:
        p = params.with_(pelosi_extra_lag_days=lag)
        bt = run_backtest(inp.tx, inp.signals, inp.md, p, start=start, end=end, **kw)
        long_only = run_backtest(inp.tx, inp.signals, inp.md, p.with_(short_notional=0.0), start=start, end=end, **kw)
        rows.append(dict(variant=label, **_stats(bt["ret"], bt["rf"]),
                         long_leg_sharpe=sharpe(long_only["ret"], bt["rf"]), long_leg_cagr=cagr(long_only["ret"])))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- rule sensitivity
def rule_sensitivity(inp: Inputs, params: StrategyParams, start: str, end: str, bench: pd.DataFrame) -> pd.DataFrame:
    """One-at-a-time changes to the discretionary modelling choices (not the tuned grid parameters)."""
    from .metrics import alpha_beta

    variants = [
        ("baseline", {}),
        ("Cramer: fade new buys AND reiterated holds", dict(cramer_signal_types=("start_long", "hold_long"))),
        ("Cramer: only high-confidence calls (≥0.9)", dict(cramer_min_confidence=0.9)),
        ("Pelosi: ignore option trades (stock only)", dict(pelosi_include_options=False)),
        ("Pelosi: weight by disclosed $ range midpoint", dict(pelosi_weighting="amount")),
        ("Pelosi: no per-name cap (100%)", dict(pelosi_max_weight=1.0)),
        ("Pelosi: do not close on disclosed sales", dict(pelosi_close_on_sale=False)),
        ("Beta: leg-return-series method", dict(beta_method="leg")),
        ("Hedge instrument: QQQ instead of SPY", dict(hedge_tickers=("QQQ",))),
        ("Costs doubled (20 bp, 200 bp borrow)", dict(cost_bps=20.0, borrow_bps_annual=200.0)),
        ("No costs, no borrow", dict(cost_bps=0.0, borrow_bps_annual=0.0)),
    ]
    rows = []
    for label, kw in variants:
        bt = run_backtest(inp.tx, inp.signals, inp.md, params.with_(**kw), start=start, end=end)
        ab = alpha_beta(bt["ret"], bench["SPY"].reindex(bt.index), bt["rf"])
        rows.append(dict(variant=label, **_stats(bt["ret"], bt["rf"]), spy_beta=ab["beta"], alpha_ann=ab["alpha_ann"],
                         alpha_t=ab["alpha_t"], avg_n_long=bt["n_long"].mean(), avg_n_short=bt["n_short"].mean()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- placebo tests
def placebo_pelosi_time_shift(inp: Inputs, params: StrategyParams, start: str, end: str, n: int, seed: int = 0) -> pd.DataFrame:
    """Keep Pelosi's tickers, sizes and holding logic but shift every filing by a random 6-24 months.

    If the strategy is just "long mega-cap tech", the shifted versions do as well.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        shift_days = int(rng.integers(180, 730)) * int(rng.choice([-1, 1]))
        tx = inp.tx.copy()
        tx["filing_date"] = tx["filing_date"] + pd.Timedelta(days=shift_days)
        tx["txn_date"] = tx["txn_date"] + pd.Timedelta(days=shift_days)
        bt = run_backtest(tx, inp.signals, inp.md, params, start=start, end=end)
        rows.append(dict(trial=i, shift_days=shift_days, **_stats(bt["ret"], bt["rf"])))
    return pd.DataFrame(rows)


def placebo_cramer_random_tickers(inp: Inputs, params: StrategyParams, start: str, end: str, n: int, seed: int = 0) -> pd.DataFrame:
    """Keep Cramer's *dates* and signal counts but replace each ticker with a random one from the priced universe.

    Tests whether fading Cramer beats fading a random basket of similar size.
    """
    rng = np.random.default_rng(seed)
    universe = np.array([c for c in inp.md.prices.columns if not c.startswith("^") and c in set(inp.signals.ticker_symbol)])
    rows = []
    for i in range(n):
        s = inp.signals.copy()
        s["ticker_symbol"] = rng.choice(universe, size=len(s), replace=True)
        bt = run_backtest(inp.tx, s, inp.md, params, start=start, end=end)
        rows.append(dict(trial=i, **_stats(bt["ret"], bt["rf"])))
    return pd.DataFrame(rows)


def placebo_cramer_shuffle_dates(inp: Inputs, params: StrategyParams, start: str, end: str, n: int, seed: int = 0) -> pd.DataFrame:
    """Stricter Cramer placebo: keep his exact names and mention frequencies, permute which date each name was called on.

    Isolates *timing*: if fading Cramer only works because of which names he likes, this placebo does as well.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        s = inp.signals.copy()
        s["ticker_symbol"] = rng.permutation(s["ticker_symbol"].values)
        bt = run_backtest(inp.tx, s, inp.md, params, start=start, end=end)
        rows.append(dict(trial=i, **_stats(bt["ret"], bt["rf"])))
    return pd.DataFrame(rows)


def placebo_pelosi_random_tickers(inp: Inputs, params: StrategyParams, start: str, end: str, n: int, seed: int = 0,
                                  candidate_pool: list[str] | None = None) -> pd.DataFrame:
    """Keep Pelosi's dates/sizes/holding logic but swap each ticker for a random large-cap."""
    rng = np.random.default_rng(seed)
    pool = np.array(candidate_pool)
    rows = []
    for i in range(n):
        tx = inp.tx.copy()
        mapping = {t: rng.choice(pool) for t in tx.ticker.dropna().unique()}
        tx["ticker"] = tx["ticker"].map(lambda t: mapping.get(t, t) if pd.notna(t) else t)
        bt = run_backtest(tx, inp.signals, inp.md, params, start=start, end=end)
        rows.append(dict(trial=i, **_stats(bt["ret"], bt["rf"])))
    return pd.DataFrame(rows)


def placebo_pvalue(actual: float, placebo: pd.Series) -> float:
    """One-sided: share of placebo trials at least as good as the actual result."""
    return float((placebo >= actual).mean())


# --------------------------------------------------------------------------- bootstrap
def block_bootstrap_sharpe(r: pd.Series, rf: pd.Series, n: int = 2000, block: int = 21, seed: int = 0) -> dict:
    """Stationary block bootstrap CI for the annualised Sharpe ratio and P(Sharpe <= 0)."""
    rng = np.random.default_rng(seed)
    ex = (r - rf.reindex(r.index).fillna(0.0)).dropna().values
    T = len(ex)
    out = np.empty(n)
    for i in range(n):
        idx = []
        while len(idx) < T:
            s = rng.integers(0, T)
            idx.extend(range(s, min(s + block, T)))
        sample = ex[np.array(idx[:T])]
        sd = sample.std(ddof=1)
        out[i] = sample.mean() / sd * np.sqrt(252) if sd > 0 else 0.0
    return dict(sharpe=float(ex.mean() / ex.std(ddof=1) * np.sqrt(252)),
                ci_low=float(np.percentile(out, 2.5)), ci_high=float(np.percentile(out, 97.5)),
                p_sharpe_le_0=float((out <= 0).mean()))


# --------------------------------------------------------------------------- data-availability check
def exposure_coverage(inp: Inputs, params: StrategyParams, start: str, end: str) -> pd.DataFrame:
    """Per-year: how often each leg actually had positions (sparse legs weaken the L/S story)."""
    cal = inp.md.calendar
    wl = pelosi_weights(inp.tx, cal, params)
    ws = cramer_weights(inp.signals, cal, params)
    df = pd.DataFrame({"long_populated": (wl.sum(axis=1) > 0), "short_populated": (ws.sum(axis=1) > 0),
                       "n_long": (wl > 0).sum(axis=1), "n_short": (ws > 0).sum(axis=1)}, index=cal)
    df = df.loc[start:end]
    g = df.groupby(df.index.year)
    return pd.DataFrame({"pct_days_long_populated": g["long_populated"].mean(),
                         "pct_days_short_populated": g["short_populated"].mean(),
                         "avg_n_long": g["n_long"].mean(), "avg_n_short": g["n_short"].mean()})
