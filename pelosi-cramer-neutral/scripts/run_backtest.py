"""Run the full backtesting suite and write results/report.md + figures + tables.

Usage: python scripts/run_backtest.py [--placebo-n 100] [--boot-n 2000] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pcn import metrics as M  # noqa: E402
from pcn import report as R  # noqa: E402
from pcn import robustness as RB  # noqa: E402
from pcn.config import (  # noqa: E402
    BENCHMARKS, DATA_DIR, DEFAULT_PARAMS, DEFAULT_SPLIT, HEDGE_TICKER, PARAM_GRID, RESULTS_DIR, RISK_FREE_TICKER,
)
from pcn.data.cramer import cross_check, load_cramer_signals, load_thestreet_calls  # noqa: E402
from pcn.data.prices import clean_prices, risk_free_daily, to_returns  # noqa: E402
from pcn.portfolio import MarketData, run_backtest  # noqa: E402

JOINT_START, JOINT_END = "2018-01-02", "2024-12-31"      # both legs have data
PELOSI_START, PELOSI_END = "2015-01-13", "2026-09-04"    # Pelosi-only leg


def load_inputs() -> RB.Inputs:
    tx = pd.read_csv(DATA_DIR / "pelosi_transactions.csv", parse_dates=["filing_date", "txn_date", "notif_date"])
    sig = load_cramer_signals(DATA_DIR / "cramer_signals.csv")
    px = clean_prices(pd.read_parquet(DATA_DIR / "prices.parquet").astype(float))
    cal = px[HEDGE_TICKER].dropna().index
    px = px.reindex(cal)
    rf = risk_free_daily(px[RISK_FREE_TICKER], cal)
    return RB.Inputs(tx=tx, signals=sig, md=MarketData(prices=px, rf_daily=rf, calendar=cal))


def bench_returns(md: MarketData) -> pd.DataFrame:
    return to_returns(md.prices[list(BENCHMARKS)])


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.1%}"


def main(placebo_n: int, boot_n: int) -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig_dir, tab_dir = RESULTS_DIR / "figures", RESULTS_DIR / "tables"
    fig_dir.mkdir(exist_ok=True)
    tab_dir.mkdir(exist_ok=True)

    inp = load_inputs()
    md, tx, sig = inp.md, inp.tx, inp.signals
    bench = bench_returns(md)
    rf = md.rf_daily
    split = DEFAULT_SPLIT
    figs: dict[str, str] = {}

    # ------------------------------------------------------------------ 0. data facts
    tx_t = tx[tx.ticker.notna()]
    lag_stats = tx_t["disclosure_lag_days"].describe()
    late = (tx_t["disclosure_lag_days"] > 45).mean()
    xchk = cross_check(sig, load_thestreet_calls(DATA_DIR / "cramer_thestreet_calls.csv"))
    figs["lag_hist"] = R.fig_lag_hist(tx_t, fig_dir / "pelosi_disclosure_lag.png")
    coverage = RB.exposure_coverage(inp, DEFAULT_PARAMS, JOINT_START, JOINT_END)
    coverage.to_csv(tab_dir / "exposure_coverage.csv")

    # ------------------------------------------------------------------ 1. pre-specified parameters, full joint window
    bt0 = run_backtest(tx, sig, md, DEFAULT_PARAMS, JOINT_START, JOINT_END)
    s0 = M.summary(bt0["ret"], rf, bench.loc[bt0.index], DEFAULT_PARAMS.beta_tolerance)
    s0_is = M.summary(bt0.loc[split.is_start: split.is_end, "ret"], rf, bench, DEFAULT_PARAMS.beta_tolerance)
    s0_oos = M.summary(bt0.loc[split.oos_start: split.oos_end, "ret"], rf, bench, DEFAULT_PARAMS.beta_tolerance)
    bt0.to_csv(tab_dir / "daily_default_params.csv")
    long0 = run_backtest(tx, sig, md, DEFAULT_PARAMS.with_(short_notional=0.0), JOINT_START, JOINT_END)
    short0 = run_backtest(tx, sig, md, DEFAULT_PARAMS.with_(long_notional=0.0), JOINT_START, JOINT_END)
    unhedged0 = run_backtest(tx, sig, md, DEFAULT_PARAMS.with_(beta_hedge=False), JOINT_START, JOINT_END)
    nocost = DEFAULT_PARAMS.with_(cost_bps=0.0, borrow_bps_annual=0.0)
    gross0 = run_backtest(tx, sig, md, nocost, JOINT_START, JOINT_END)
    long0g = run_backtest(tx, sig, md, nocost.with_(short_notional=0.0), JOINT_START, JOINT_END)
    short0g = run_backtest(tx, sig, md, nocost.with_(long_notional=0.0), JOINT_START, JOINT_END)
    legs = pd.DataFrame(
        {
            name: {
                "CAGR": M.cagr(b["ret"]), "Vol": M.ann_vol(b["ret"]), "Sharpe": M.sharpe(b["ret"], rf),
                "MaxDD": M.max_drawdown(b["ret"]), "Beta vs SPY": M.alpha_beta(b["ret"], bench["SPY"], rf)["beta"],
                "Alpha ann.": M.alpha_beta(b["ret"], bench["SPY"], rf)["alpha_ann"],
                "Alpha t": M.alpha_beta(b["ret"], bench["SPY"], rf)["alpha_t"],
            }
            for name, b in [("L/S net (default)", bt0), ("Long Pelosi only (hedged)", long0),
                            ("Short Cramer only (hedged)", short0), ("L/S unhedged (no SPY overlay)", unhedged0),
                            ("L/S before costs & borrow", gross0),
                            ("Long Pelosi only (hedged), before costs", long0g),
                            ("Short Cramer only (hedged), before costs & borrow", short0g)]
        }
    ).T
    legs.to_csv(tab_dir / "leg_decomposition.csv")
    figs["growth"] = R.fig_growth(bt0, bench[["SPY", "IWM", "QQQ"]], split.oos_start, fig_dir / "growth.png")
    figs["legs"] = R.fig_legs(bt0, fig_dir / "legs.png")
    figs["dd"] = R.fig_drawdown(bt0, bench["SPY"], fig_dir / "drawdown.png")
    figs["beta"] = R.fig_beta(bt0, bench[["SPY", "IWM"]], DEFAULT_PARAMS.beta_tolerance, fig_dir / "beta.png")
    figs["rsharpe"] = R.fig_rolling_sharpe(bt0, fig_dir / "rolling_sharpe.png")
    yr = M.yearly_table(bt0["ret"], bench[["SPY", "IWM", "QQQ"]])
    yr.to_csv(tab_dir / "yearly_returns.csv")
    figs["yearly"] = R.fig_yearly(yr, fig_dir / "yearly.png")

    # ------------------------------------------------------------------ 2. IS grid search -> OOS
    print("grid search ...")
    grid = RB.grid_search(inp, DEFAULT_PARAMS, split)
    grid.to_csv(tab_dir / "grid_is_oos.csv", index=False)
    sel = RB.select_params(DEFAULT_PARAMS, grid)
    bt_sel = run_backtest(tx, sig, md, sel, JOINT_START, JOINT_END)
    s_sel_is = M.summary(bt_sel.loc[split.is_start: split.is_end, "ret"], rf, bench, sel.beta_tolerance)
    s_sel_oos = M.summary(bt_sel.loc[split.oos_start: split.oos_end, "ret"], rf, bench, sel.beta_tolerance)
    figs["grid"] = R.fig_grid(grid, fig_dir / "grid_is_oos.png")

    # ------------------------------------------------------------------ 3. walk-forward
    print("walk-forward ...")
    wf_ret, wf_tab = RB.walk_forward(inp, DEFAULT_PARAMS, first_oos_year=2020, last_year=2024)
    wf_tab.to_csv(tab_dir / "walk_forward.csv", index=False)
    s_wf = M.summary(wf_ret, rf, bench, DEFAULT_PARAMS.beta_tolerance)
    figs["wf"] = R.fig_walk_forward(wf_ret, bench["SPY"], fig_dir / "walk_forward.png")

    # ------------------------------------------------------------------ 4. disclosure-lag sensitivity
    print("lag sensitivity ...")
    lag = RB.lag_sensitivity(inp, DEFAULT_PARAMS, JOINT_START, JOINT_END)
    lag.to_csv(tab_dir / "lag_sensitivity.csv", index=False)
    figs["lag"] = R.fig_lag(lag, fig_dir / "lag_sensitivity.png")

    # ------------------------------------------------------------------ 4b. rule sensitivity
    print("rule sensitivity ...")
    rules = RB.rule_sensitivity(inp, DEFAULT_PARAMS, JOINT_START, JOINT_END, bench)
    rules.to_csv(tab_dir / "rule_sensitivity.csv", index=False)

    # ------------------------------------------------------------------ 5. placebos
    print(f"placebos (n={placebo_n}) ...")
    large_caps = [t for t in sig.ticker_symbol.value_counts().index[:150] if t in md.prices.columns]
    pl_shift = RB.placebo_pelosi_time_shift(inp, DEFAULT_PARAMS, JOINT_START, JOINT_END, placebo_n)
    pl_rand_p = RB.placebo_pelosi_random_tickers(inp, DEFAULT_PARAMS, JOINT_START, JOINT_END, placebo_n, candidate_pool=large_caps)
    pl_rand_c = RB.placebo_cramer_random_tickers(inp, DEFAULT_PARAMS, JOINT_START, JOINT_END, placebo_n)
    pl_shuf_c = RB.placebo_cramer_shuffle_dates(inp, DEFAULT_PARAMS, JOINT_START, JOINT_END, placebo_n)
    for name, df in [("placebo_pelosi_timeshift", pl_shift), ("placebo_pelosi_random", pl_rand_p),
                     ("placebo_cramer_random", pl_rand_c), ("placebo_cramer_shuffle", pl_shuf_c)]:
        df.to_csv(tab_dir / f"{name}.csv", index=False)
    placebo_p = {
        "Pelosi filings time-shifted ±6-24m": RB.placebo_pvalue(s0["sharpe"], pl_shift["sharpe"]),
        "Pelosi tickers → random large caps": RB.placebo_pvalue(s0["sharpe"], pl_rand_p["sharpe"]),
        "Cramer tickers → random universe names": RB.placebo_pvalue(s0["sharpe"], pl_rand_c["sharpe"]),
        "Cramer tickers shuffled across dates": RB.placebo_pvalue(s0["sharpe"], pl_shuf_c["sharpe"]),
    }
    figs["placebo"] = R.fig_placebo(
        s0["sharpe"],
        {"Pelosi time-shift": pl_shift["sharpe"], "Pelosi random tickers": pl_rand_p["sharpe"],
         "Cramer random tickers": pl_rand_c["sharpe"], "Cramer names shuffled across dates": pl_shuf_c["sharpe"]},
        fig_dir / "placebo.png",
    )

    # ------------------------------------------------------------------ 6. bootstrap
    boot = RB.block_bootstrap_sharpe(bt0["ret"], rf, n=boot_n)
    boot_oos = RB.block_bootstrap_sharpe(bt0.loc[split.oos_start: split.oos_end, "ret"], rf, n=boot_n)

    # ------------------------------------------------------------------ 7. Pelosi-only leg over its full history
    bt_pel_full = run_backtest(tx, sig, md, DEFAULT_PARAMS.with_(short_notional=0.0), PELOSI_START, PELOSI_END)
    s_pel_full = M.summary(bt_pel_full["ret"], rf, bench.loc[bt_pel_full.index], DEFAULT_PARAMS.beta_tolerance)
    bt_pel_full.to_csv(tab_dir / "daily_pelosi_leg_full_history.csv")
    # long Pelosi *unhedged* for reference (what the "Pelosi tracker" ETFs effectively do)
    pel_unh = run_backtest(tx, sig, md, DEFAULT_PARAMS.with_(short_notional=0.0, beta_hedge=False), PELOSI_START, PELOSI_END)
    s_pel_unh = M.summary(pel_unh["ret"], rf, bench.loc[pel_unh.index], DEFAULT_PARAMS.beta_tolerance)

    # ------------------------------------------------------------------ 8. write report
    summary_json = {
        "default_full": s0, "default_is": s0_is, "default_oos": s0_oos,
        "selected_params": sel.__dict__, "selected_is": s_sel_is, "selected_oos": s_sel_oos,
        "walk_forward": s_wf, "bootstrap_full": boot, "bootstrap_oos": boot_oos, "placebo_p": placebo_p,
        "pelosi_leg_full_history_hedged": s_pel_full, "pelosi_leg_full_history_unhedged": s_pel_unh,
        "cramer_cross_check": xchk,
        "disclosure_lag": {k: float(v) for k, v in lag_stats.items()}, "share_filed_after_45d": float(late),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary_json, indent=2, default=str))
    write_report(
        RESULTS_DIR / "report.md", figs, s0, s0_is, s0_oos, legs, grid, sel, s_sel_is, s_sel_oos, wf_tab, s_wf, lag,
        placebo_p, boot, boot_oos, s_pel_full, s_pel_unh, xchk, lag_stats, late, coverage, yr, tx_t, sig, md, split,
        rules, placebo_n,
    )
    print(f"done in {time.time() - t0:.0f}s -> {RESULTS_DIR / 'report.md'}")


def write_report(path, figs, s0, s0_is, s0_oos, legs, grid, sel, s_sel_is, s_sel_oos, wf_tab, s_wf, lag, placebo_p,
                 boot, boot_oos, s_pel_full, s_pel_unh, xchk, lag_stats, late, coverage, yr, tx_t, sig, md, split,
                 rules, placebo_n):
    F = lambda name: f"figures/{figs[name]}"  # noqa: E731
    rules_fmt = rules.copy()
    for c in ["cagr", "max_dd", "alpha_ann"]:
        rules_fmt[c] = rules_fmt[c].map(pct)
    for c in ["sharpe", "spy_beta", "alpha_t"]:
        rules_fmt[c] = rules_fmt[c].map(lambda v: f"{v:.2f}")
    rules_fmt["avg_n_long"] = rules_fmt["avg_n_long"].map(lambda v: f"{v:.1f}")
    rules_fmt["avg_n_short"] = rules_fmt["avg_n_short"].map(lambda v: f"{v:.0f}")
    rules_fmt = rules_fmt.drop(columns=["n_days"])
    is_oos_rows = pd.DataFrame(
        {
            "Pre-specified params · IS": s0_is, "Pre-specified params · OOS": s0_oos,
            "IS-optimised params · IS": s_sel_is, "IS-optimised params · OOS": s_sel_oos,
            "Walk-forward (yearly re-selection) · OOS only": s_wf,
        }
    ).loc[["start", "end", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "SPY_beta", "SPY_alpha_ann",
           "SPY_alpha_t", "SPY_excess_cagr", "IWM_excess_cagr", "QQQ_excess_cagr"]]
    for r in ["cagr", "ann_vol", "max_drawdown", "SPY_alpha_ann", "SPY_excess_cagr", "IWM_excess_cagr", "QQQ_excess_cagr"]:
        is_oos_rows.loc[r] = is_oos_rows.loc[r].map(pct)
    for r in ["sharpe", "sortino", "SPY_beta", "SPY_alpha_t"]:
        is_oos_rows.loc[r] = is_oos_rows.loc[r].map(lambda v: f"{float(v):.2f}")
    is_oos_rows.index.name = "metric"

    legs_fmt = legs.copy()
    for c in ["CAGR", "Vol", "MaxDD", "Alpha ann."]:
        legs_fmt[c] = legs_fmt[c].map(pct)
    for c in ["Sharpe", "Beta vs SPY", "Alpha t"]:
        legs_fmt[c] = legs_fmt[c].map(lambda v: f"{v:.2f}")
    legs_fmt.index.name = "book"

    grid_top = grid.head(10)[list(PARAM_GRID) + ["is_sharpe", "oos_sharpe", "is_cagr", "oos_cagr", "oos_max_dd", "is_mean_abs_beta"]]
    lag_fmt = lag.copy()
    for c in ["cagr", "max_dd", "long_leg_cagr"]:
        lag_fmt[c] = lag_fmt[c].map(pct)
    for c in ["sharpe", "long_leg_sharpe"]:
        lag_fmt[c] = lag_fmt[c].map(lambda v: f"{v:.2f}")
    lag_fmt = lag_fmt.drop(columns=["n_days"])

    cov_fmt = coverage.copy()
    cov_fmt["pct_days_long_populated"] = cov_fmt["pct_days_long_populated"].map(pct)
    cov_fmt["pct_days_short_populated"] = cov_fmt["pct_days_short_populated"].map(pct)
    cov_fmt["avg_n_long"] = cov_fmt["avg_n_long"].map(lambda v: f"{v:.1f}")
    cov_fmt["avg_n_short"] = cov_fmt["avg_n_short"].map(lambda v: f"{v:.0f}")
    cov_fmt.index.name = "year"
    yr_fmt = yr.copy().map(pct)
    yr_fmt.index.name = "year"

    n_buy = int(((tx_t.txn_type == "P") & tx_t.instrument.isin(["stock", "call", "exercise"])).sum())
    n_sell = int((tx_t.txn_type == "S").sum())
    n_opt = int(tx_t.instrument.isin(["call", "exercise"]).sum())
    beta_ok = s0["spy_rolling_beta_1y_pct_within_tol"]

    md_txt = f"""# Long Pelosi / Short Cramer — market-neutral backtest

_Generated by `scripts/run_backtest.py`. All numbers are net of {DEFAULT_PARAMS.cost_bps:.0f} bp one-way
transaction costs and a {DEFAULT_PARAMS.borrow_bps_annual:.0f} bp/yr stock-borrow fee; cash collateral earns the
13-week T-bill rate. Sharpe ratios are computed on returns in excess of T-bills._

## 0. Key findings

* **Joint sample 2018-01 → 2024-12 (7 y), pre-specified parameters, net of costs:** CAGR **{s0['cagr']:.1%}**,
  vol {s0['ann_vol']:.1%}, **Sharpe {s0['sharpe']:.2f}**, max drawdown **{s0['max_drawdown']:.0%}**
  (2022), alpha vs SPY {s0['SPY_alpha_ann']:+.1%}/yr (*t* = {s0['SPY_alpha_t']:.2f}). Bootstrap 95 % CI for the Sharpe:
  [{boot['ci_low']:.2f}, {boot['ci_high']:.2f}] — **not statistically distinguishable from zero.**
* **Beta neutral in the mandated sense:** full-sample β to SPY = {s0['SPY_beta']:.3f} (s.e. {s0['SPY_beta_se']:.3f}); ex-ante β is
  zero every day; realised rolling 1-year β stayed inside ±0.10 on {beta_ok:.0%} of days and peaked at
  {s0['spy_rolling_beta_1y_max_abs']:.2f}. β to Russell 2000 {s0['IWM_beta']:+.2f}, to QQQ {s0['QQQ_beta']:+.2f} — the book is market-neutral but
  carries a large-cap-growth tilt (long NVDA/AAPL/MSFT vs. a broad short basket), which is what produced the 2022 drawdown.
* **Versus benchmarks:** the L/S book returned {s0['SPY_excess_cagr']:+.1%}/yr vs SPY, {s0['IWM_excess_cagr']:+.1%}/yr vs the Russell 2000 and
  {s0['QQQ_excess_cagr']:+.1%}/yr vs QQQ over the same window, at roughly SPY's volatility and with a comparable drawdown — an
  unlevered beta-zero book cannot be expected to beat a bull market's beta.
* **Where the P&L comes from:** the hedged Pelosi leg (Sharpe {legs.loc['Long Pelosi only (hedged)', 'Sharpe']:.2f}) supplies most of it; the hedged
  Cramer short leg is about break-even after costs (Sharpe {legs.loc['Short Cramer only (hedged)', 'Sharpe']:.2f};
  {legs.loc['Short Cramer only (hedged), before costs & borrow', 'Sharpe']:.2f} before costs). Excluding Pelosi's *option* trades
  collapses the strategy (Sharpe {rules.set_index('variant').loc['Pelosi: ignore option trades (stock only)', 'sharpe']:.2f}) — the
  return is concentrated in her disclosed call purchases on a handful of mega-caps.
* **The disclosure lag does not hurt.** Acting on the filing date beats the (illegal, not implementable) trade-date entry
  (Sharpe {lag.iloc[1]['sharpe']:.2f} vs {lag.iloc[0]['sharpe']:.2f}), and waiting a further 5–45 sessions is no worse. There is no
  short-lived private information in these filings to be late to.
* **Placebos:** time-shifting Pelosi's filings (p = {placebo_p['Pelosi filings time-shifted ±6-24m']:.2f}) or swapping her names for random
  large caps (p = {placebo_p['Pelosi tickers → random large caps']:.2f}) does about as well ⇒ her leg is indistinguishable from a
  concentrated large-cap-tech bet. Fading Cramer's *actual* names on their *actual* dates, however, beats shuffling
  the same names across dates in every one of {placebo_n} trials (p = {placebo_p['Cramer tickers shuffled across dates']:.3f}) ⇒ there is a genuine
  post-mention reversal in his picks; it is simply too small to pay for a 21-day-turnover short book.
* **In-sample vs out-of-sample:** pre-specified parameters gave IS Sharpe {s0_is['sharpe']:.2f} → OOS {s0_oos['sharpe']:.2f};
  IS-optimised parameters {s_sel_is['sharpe']:.2f} → {s_sel_oos['sharpe']:.2f}; walk-forward OOS {s_wf['sharpe']:.2f}. IS→OOS Spearman across the
  grid = {grid.attrs['is_oos_spearman']:.2f}. Tuning buys almost nothing; the decay from IS to OOS is the honest number.
* **Fragility:** doubling costs (20 bp, 200 bp borrow) takes the Sharpe to
  {rules.set_index('variant').loc['Costs doubled (20 bp, 200 bp borrow)', 'sharpe']:.2f}; removing costs lifts it to
  {rules.set_index('variant').loc['No costs, no borrow', 'sharpe']:.2f}. Hedging with QQQ instead of SPY (removing the growth tilt) gives
  {rules.set_index('variant').loc['Hedge instrument: QQQ instead of SPY', 'sharpe']:.2f} — but that is a post-hoc choice and is reported as a sensitivity, not a result.

## 1. Is this actually tradeable? (point-in-time discipline)

**Yes, but only on the filing date, not the trade date.** Under the STOCK Act a member must file a Periodic
Transaction Report (PTR) within 30 days of being notified of a trade and no later than **45 days** after the
trade. The information is public only once the Clerk posts the PDF. This backtest therefore:

* stamps every Pelosi transaction with **both** its trade date and the Clerk's **filing date**, and only acts on the
  first trading session *after* the filing date (`filing_date + 1 session`);
* enters Cramer fades on the session *after* the episode airs (Mad Money airs 6 pm ET, after the close);
* never uses a price, a beta estimate or a parameter that was not observable at the time (betas are trailing;
  parameters are chosen in-sample only, see §5).

Observed disclosure lag for the {len(tx_t)} Pelosi transactions with tickers: median **{lag_stats['50%']:.0f} days**,
mean {lag_stats['mean']:.0f}, max {lag_stats['max']:.0f}; **{late:.0%}** were filed after the 45-day statutory limit
(the strategy simply trades later in those cases — it never assumes a filing was on time).

![]({F('lag_hist')})

§6 quantifies what the lag costs: the same rules applied on the (unknowable) trade date are shown in red as a
lookahead reference only.

## 2. Data

| Leg | Source | Span | Rows |
|---|---|---|---|
| Long: Nancy Pelosi PTRs | House Clerk financial-disclosure index + PTR PDFs (parsed with `pdfplumber`) | {tx_t.filing_date.min().date()} → {tx_t.filing_date.max().date()} | {len(tx_t)} transactions ({n_buy} buys incl. {n_opt} call purchases/exercises, {n_sell} sells) across {tx_t.ticker.nunique()} tickers |
| Short: Jim Cramer bullish calls | Kull (2026) transcript-derived *Mad Money* signals, CC-BY-4.0 | {sig.signal_date.min().date()} → {sig.signal_date.max().date()} | {len(sig):,} calls ({(sig.signal_type == 'start_long').sum():,} new buys, {(sig.signal_type == 'hold_long').sum():,} holds) across {sig.ticker_symbol.nunique():,} tickers |
| Cross-check | TheStreet *Mad Money* screener scrape (offline since 2022) | {xchk['overlap_start']} → {xchk['overlap_end']} | {xchk['thestreet_bullish_in_window']:,} bullish calls |
| Prices | Yahoo Finance (Pelosi names, benchmarks, ^IRX) + FIGI-keyed adjusted closes shipped with Kull (2026) for the Cramer universe, incl. delisted names | 2014-06 → 2026-09 | {md.prices.shape[1]:,} series |

*Cramer source agreement:* {xchk['kull_found_in_thestreet_pct']}% of transcript calls appear in the screener within ±1 day,
and {xchk['thestreet_found_in_kull_pct']}% of screener bullish calls appear in the transcript data. The two were built
independently (LLM transcript extraction vs. TheStreet's editorial log), so partial overlap is expected; the
transcript data is used because it is longer and includes the segment/confidence metadata.

**Why the joint window is 2018-01 → 2024-12.** Pelosi PTRs are available electronically from Jan-2015 and continue
to today; machine-readable Cramer calls exist from Jan-2018 to Dec-2024. A long/short book needs both legs, so the
headline results cover **7 years, 2018-01-02 → 2024-12-31**. The Pelosi leg alone is also reported over 2015 → 2026
(§7). Pre-2015 PTRs exist only as scanned paper filings and are excluded as unreliable.

Known gaps: `WORK` (Slack, acquired 2021) and old-Hertz (`HTZ`, 2015 options) have no Yahoo history, and `BFET` is
private — these Pelosi positions are dropped. The Cramer universe is survivorship-bias free (1,385/1,386 tickers priced).

Leg population by year (default parameters):

{R.md_table(cov_fmt.reset_index())}

## 3. Strategy rules

* **Long leg (Pelosi):** every disclosed purchase of stock, exercised call, or purchased call option on a listed
  ticker opens/extends an equal-weighted long (max {DEFAULT_PARAMS.pelosi_max_weight:.0%} per name) for up to
  `pelosi_hold_days` sessions; a disclosed sale of that ticker closes it the session after the sale is filed.
* **Short leg (Cramer):** every new bullish call (`start_long`) opens an equal-weighted short held
  `cramer_hold_days` sessions; repeated calls on the same name extend the position.
* **Notionals:** 100% long / 100% short of NAV when both legs are populated (the long leg is often under-invested
  because Pelosi holds few names and the cap binds).
* **Beta neutrality:** trailing `beta_window`-day betas of each leg to SPY (shrunk toward 1) give the ex-ante net beta;
  an SPY overlay brings it to 0 every day. Target is 0; the mandate is |β| ≤ ±0.10.

Pre-specified ("prior") parameters, fixed before looking at any result:
`pelosi_hold_days={DEFAULT_PARAMS.pelosi_hold_days}`, `cramer_hold_days={DEFAULT_PARAMS.cramer_hold_days}`,
`beta_window={DEFAULT_PARAMS.beta_window}`.

## 4. Headline results — full joint window (pre-specified parameters, 2018-01 → 2024-12)

{R.md_table(M.format_summary(s0, list(BENCHMARKS)))}

**Beta neutrality check:** full-sample beta to SPY = {s0['SPY_beta']:.3f} (s.e. {s0['SPY_beta_se']:.3f}); realised
rolling 1-year beta stayed within ±{DEFAULT_PARAMS.beta_tolerance:.2f} on **{beta_ok:.0%}** of days
(max |β| = {s0['spy_rolling_beta_1y_max_abs']:.2f}). Beta to the Russell 2000 = {s0['IWM_beta']:.3f}, to QQQ = {s0['QQQ_beta']:.3f}.

### Versus equity benchmarks

{R.md_table(M.format_benchmarks(s0, BENCHMARKS))}

![]({F('growth')})
![]({F('dd')})
![]({F('beta')})
![]({F('rsharpe')})

### Calendar-year returns

{R.md_table(yr_fmt.reset_index())}

![]({F('yearly')})

### Where does the P&L come from? (each book beta-hedged on its own)

{R.md_table(legs_fmt.reset_index())}

![]({F('legs')})

## 5. In-sample vs out-of-sample

Split: **IS = {split.is_start} → {split.is_end}**, **OOS = {split.oos_start} → {split.oos_end}**. The parameter grid
({' × '.join(f'{k}∈{v}' for k, v in PARAM_GRID.items())}, {len(grid)} combinations) was searched on IS Sharpe only.

{R.md_table(is_oos_rows.reset_index())}

* IS-optimised parameters: `{ {k: getattr(sel, k) for k in PARAM_GRID} }`.
* Spearman rank correlation between IS and OOS Sharpe across the {len(grid)} grid points: **{grid.attrs['is_oos_spearman']:.2f}**
  (near zero or negative ⇒ in-sample ranking has no predictive value; the "best IS" point is then just noise).
* Walk-forward: parameters re-selected each January from all prior years (first OOS year 2020); the stitched
  OOS series has Sharpe **{s_wf['sharpe']:.2f}**, CAGR {s_wf['cagr']:.1%}, max DD {s_wf['max_drawdown']:.1%}.

Top-10 grid points by IS Sharpe:

{R.md_table(grid_top)}

![]({F('grid')})
![]({F('wf')})

Walk-forward selections:

{R.md_table(wf_tab)}

## 6. Cost of the disclosure lag, rule sensitivity and placebo tests

### Disclosure lag

{R.md_table(lag_fmt)}

![]({F('lag')})

### Modelling-choice sensitivity (one change at a time from the pre-specified baseline, full joint window)

{R.md_table(rules_fmt)}

### Placebos ({placebo_n} trials each; p = share of placebo Sharpes ≥ actual {s0['sharpe']:.2f})

| Placebo | What it breaks | p-value |
|---|---|---|
| Pelosi filings time-shifted ±6–24 months | timing of her trades (keeps the names) | {placebo_p['Pelosi filings time-shifted ±6-24m']:.3f} |
| Pelosi tickers → random large caps | her stock selection (keeps the dates) | {placebo_p['Pelosi tickers → random large caps']:.3f} |
| Cramer tickers → random universe names | his stock selection (keeps the dates and counts) | {placebo_p['Cramer tickers → random universe names']:.3f} |
| Cramer tickers shuffled across dates | his *timing* only (keeps exactly his names and how often he mentions them) | {placebo_p['Cramer tickers shuffled across dates']:.3f} |

![]({F('placebo')})

### Bootstrap (stationary block bootstrap, 21-day blocks)

| Sample | Sharpe | 95% CI | P(Sharpe ≤ 0) |
|---|---|---|---|
| Full joint window | {boot['sharpe']:.2f} | [{boot['ci_low']:.2f}, {boot['ci_high']:.2f}] | {boot['p_sharpe_le_0']:.3f} |
| OOS only | {boot_oos['sharpe']:.2f} | [{boot_oos['ci_low']:.2f}, {boot_oos['ci_high']:.2f}] | {boot_oos['p_sharpe_le_0']:.3f} |

## 7. Pelosi leg over its full history (2015-01 → 2026-09, no Cramer leg)

| | Beta-hedged (SPY overlay) | Unhedged long-only |
|---|---|---|
| CAGR | {s_pel_full['cagr']:.1%} | {s_pel_unh['cagr']:.1%} |
| Vol | {s_pel_full['ann_vol']:.1%} | {s_pel_unh['ann_vol']:.1%} |
| Sharpe | {s_pel_full['sharpe']:.2f} | {s_pel_unh['sharpe']:.2f} |
| Max drawdown | {s_pel_full['max_drawdown']:.1%} | {s_pel_unh['max_drawdown']:.1%} |
| Beta vs SPY | {s_pel_full['SPY_beta']:.2f} | {s_pel_unh['SPY_beta']:.2f} |
| Alpha vs SPY (ann., t) | {s_pel_full['SPY_alpha_ann']:+.1%} ({s_pel_full['SPY_alpha_t']:.2f}) | {s_pel_unh['SPY_alpha_ann']:+.1%} ({s_pel_unh['SPY_alpha_t']:.2f}) |
| SPY CAGR same period | {s_pel_full['SPY_cagr']:.1%} | {s_pel_unh['SPY_cagr']:.1%} |
| QQQ CAGR same period | {s_pel_full['QQQ_cagr']:.1%} | {s_pel_unh['QQQ_cagr']:.1%} |

## 8. Caveats

* **Option leverage is not modelled.** Pelosi's most publicised trades are deep-in-the-money LEAPS. A disclosed call
  purchase is treated as a *delta-one long in the underlying* with equal weight; replicating the option payoff would
  require strike/expiry-level pricing that is not available point-in-time from PTRs (only ranges are disclosed).
* **Dollar sizes are ranges** ($1,001–$15,000 … $5m–$25m); equal weighting is the default, amount-weighting is an
  option (`pelosi_weighting="amount"`).
* **Short-leg frictions are simplified**: a flat borrow fee, no hard-to-borrow constraints, no short-sale
  restrictions. Small-cap Cramer names would in practice be more expensive or impossible to borrow.
* **Concentration.** The long leg frequently holds fewer than 5 names; the beta hedge neutralises market risk but
  not idiosyncratic risk (NVDA/AAPL dominate).
* **Sample size.** 7 joint years and ~{n_buy} long entries is a small sample; every inference above should be read with
  the bootstrap CIs and placebo p-values, not the point estimates.
"""
    path.write_text(md_txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebo-n", type=int, default=100)
    ap.add_argument("--boot-n", type=int, default=2000)
    ap.add_argument("--quick", action="store_true", help="tiny placebo/bootstrap counts for smoke testing")
    a = ap.parse_args()
    if a.quick:
        main(placebo_n=5, boot_n=200)
    else:
        main(placebo_n=a.placebo_n, boot_n=a.boot_n)
