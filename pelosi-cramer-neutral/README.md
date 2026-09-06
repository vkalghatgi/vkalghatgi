# Long Pelosi / Short Cramer — beta-neutral backtest

A point-in-time-correct backtest of the meme: **go long what Nancy Pelosi discloses buying, short what Jim Cramer
tells you to buy, hedge the residual market beta to zero.**

The full results write-up (statistics, IS/OOS, walk-forward, placebos, bootstrap, figures) is in
[`results/report.md`](results/report.md). This file explains what was built, why the sample is what it is, and
how to reproduce it.

---

## TL;DR

Joint window 2018-01 → 2024-12 (the seven years for which both legs have machine-readable, point-in-time data),
pre-specified parameters, net of 10 bp costs and 100 bp borrow:

| | L/S book | SPY | Russell 2000 (IWM) | QQQ |
|---|---|---|---|---|
| CAGR | **6.6 %** | 13.7 % | 6.8 % | 19.3 % |
| Vol | 13.7 % | 19 % | 24 % | 24 % |
| Sharpe (excess of T-bills) | **0.37** | 0.64 | 0.30 | 0.76 |
| Max drawdown | **−30 %** | −34 % | −41 % | −35 % |
| Beta of the book to … | — | **0.01** (s.e. 0.02) | −0.12 | +0.12 |
| Alpha of the book vs … (ann., *t*) | — | +4.9 % (0.9) | +5.9 % (1.2) | +2.7 % (0.5) |

* Bootstrap 95 % CI for the Sharpe is **[−0.43, 1.14]**: not distinguishable from zero. IS Sharpe 0.50 → OOS 0.21;
  walk-forward OOS 0.30.
* Beta-neutral as mandated: ex-ante β = 0 daily; realised rolling 1-year β inside ±0.10 on 82 % of days, max 0.20.
  The residual risk is a large-cap-growth tilt (long NVDA/AAPL/MSFT vs a broad short basket) — hence the 2022 drawdown.
* The Pelosi leg supplies the return (hedged Sharpe 0.29) and it lives almost entirely in her disclosed **call-option**
  trades (drop them → Sharpe 0.07). Time-shifting her filings (p = 0.44) or swapping her names for random large caps
  (p = 0.26) does about as well: the leg is a concentrated mega-cap-tech bet, not evidence of information.
* The Cramer short leg has real *timing* content — fading his names on his dates beats fading the same names shuffled
  across dates in 200/200 placebo trials, and is Sharpe 0.61 before costs — but net of a 21-day-turnover short book's
  costs it is break-even (0.12).
* The **disclosure lag does not hurt**: entering on the filing date (Sharpe 0.37) beats the illegal trade-date entry
  (0.22), and waiting a further 5–45 sessions is no worse. There is no short-lived private information to be late to.
* Fragile to costs: doubling them takes the Sharpe to 0.03.

**Diversification and Sharpe improvement** ([`results/diversification.md`](results/diversification.md)):

* Correlation with SPY +0.01, 60/40 0.00, bonds −0.06, gold −0.07; rolling 1-yr correlation with SPY stays in
  [−0.24, +0.27]. Only loadings: +0.22 to QQQ, −0.22 to IWM (the growth residual).
* Moving 20 % of a 60/40 into the book: Sharpe 0.57 → 0.65, max DD −21.7 % → −18.6 %, CAGR 9.0 → 8.8 %. As a
  25–50 % overlay on 60/40: Sharpe 0.57 → 0.65–0.68 with CAGR 9.0 → 10.3–11.5 %. Bootstrap CI on the Sharpe gain
  includes zero (P(gain ≤ 0) = 0.28). The IS-optimal weight (35 %) still helps OOS (0.11 → 0.20).
* Positive in fast crashes (COVID +12.5 %, regional-bank stress +5.2 %, Q4-2018 +0.8 %), negative in the slow
  2022 growth bear (−8.8 %). A diversifier, not a tail hedge.
* Of 13 pre-declared Sharpe-improving changes, 8 beat the baseline in both IS and OOS. **Vol-targeting to 10 %**
  (IS 0.66 / OOS 0.39 vs 0.50 / 0.21, max DD −20 %) and a **slower Cramer leg** (63d) are the a-priori
  defensible ones; a pre-declared combo of both plus no-exit-on-sales reaches IS 0.88 / OOS 0.70 / full 0.81.
* The **growth tilt is the return**: a two-factor SPY+QQQ hedge that truly neutralises QQQ beta leaves Sharpe 0.12.
  The IS-optimal leg mix (drop the Cramer leg, IS 0.66) is the *worst* OOS choice (−0.19): the short leg carried
  2022.

---

## 1. Is it actually possible? (the disclosure-lag problem)

Members of Congress do **not** disclose in real time. Under the STOCK Act (2012) a Periodic Transaction Report
(PTR) must be filed within 30 days of the member being *notified* of the trade and no later than **45 days after
the trade**. The report is public only when the Clerk of the House posts the PDF.

This codebase treats that as a hard constraint:

| | Information becomes public | Earliest trade in the backtest |
|---|---|---|
| Pelosi purchase / sale | Clerk `FilingDate` of the PTR (parsed from the House index, *not* the transaction date) | first session **after** the filing date (`filing_date + 1`) |
| Cramer bullish call | Episode air date (6 pm ET, after the close) | next session's close |
| Beta estimates | — | trailing windows only |
| Strategy parameters | — | selected in-sample only; walk-forward re-selects yearly using prior years |

Every Pelosi record carries **both** `txn_date` and `filing_date` (`data/pelosi_transactions.csv`). Observed lag:
median 25 days, mean 28, max 244; ~3 % of filings breached the 45-day limit — the backtest simply trades later in
those cases. `results/tables/lag_sensitivity.csv` quantifies the cost of the lag against a lookahead reference that
trades on the transaction date (shown only to measure the lag; it is not implementable).

## 2. Data and why the joint sample is 2018 → 2024

| Leg | Source | Reliable span | Notes |
|---|---|---|---|
| **Pelosi** | [House Clerk financial-disclosure index](https://disclosures-clerk.house.gov) (`{year}FD.zip`) + PTR PDFs, parsed with `pdfplumber` | **2015-01 → today** (63 PTRs, 183 ticker-level transactions) | Pre-2015 PTRs exist only as scanned paper forms and are excluded. |
| **Cramer** | Kull, A. (2026) *What Cramer Holds vs What He Recommends*, transcript-derived signal dataset, [CC-BY-4.0](https://github.com/andreskull/cramer-mad-money-research) | **2018-01 → 2024-12** (16,701 on-air calls, 1,386 tickers) | Long-side calls only (`start_long`, `hold_long`). |
| Cramer cross-check | TheStreet *Mad Money* screener scrape ([gaborvecsei](https://github.com/gaborvecsei/Mad-Money-Backtesting)) | 2020-01 → 2022-06 | Screener is offline since 2022. ~55–60 % pairwise overlap with the transcript data. |
| Prices | Yahoo Finance for Pelosi names, benchmarks and `^IRX`; FIGI-keyed adjusted closes shipped with Kull (2026) for the whole Cramer universe | 2014-06 → 2026-09 | The FIGI matrix includes delisted names, so the short leg is survivorship-bias free. Both sources agree to ~1e-7 on overlapping returns. Known defects (unadjusted reverse splits, sub-penny placeholder quotes) are handled in `pcn/data/prices.py::clean_prices` and audited in `data/price_jump_audit.csv`. |

A long/short book needs both legs, so headline statistics cover **2018-01-02 → 2024-12-31**. The Pelosi leg is
also reported alone over 2015 → 2026. No Cramer dataset was found that goes back further than 2018 in a
machine-readable, point-in-time form; if one exists it can be dropped into `data/cramer_signals.csv` with the same
columns and everything re-runs.

## 3. Strategy

* **Long leg:** each disclosed purchase of stock, exercised call, or purchased call option opens an equal-weighted
  long (cap 20 %/name) held up to `pelosi_hold_days`; a disclosed sale closes it the session after it is filed.
* **Short leg:** each new bullish Cramer call opens an equal-weighted short held `cramer_hold_days` sessions.
* **Neutralisation:** holdings-based trailing betas (`Σ wᵢ βᵢ`, βᵢ shrunk toward 1) give the ex-ante net beta; an
  SPY overlay brings it to the 0 target every day. Mandate: |β| ≤ 0.10. Realised rolling betas are reported.
  (`hedge_tickers` accepts several instruments for a joint multi-factor hedge; `vol_target` scales the book —
  both are off in the baseline and used only in the diversification analysis.)
* **Frictions:** 10 bp one-way costs on all turnover, 100 bp/yr borrow, cash collateral earns 13-week T-bills.

Pre-specified parameters (fixed before any result was seen): `pelosi_hold_days=252`, `cramer_hold_days=21`,
`beta_window=126`.

## 4. Overfitting controls

1. **Chronological IS/OOS split** (IS 2018–2021, OOS 2022–2024); a 48-point grid is searched on IS Sharpe only and
   the OOS column is *reported, never selected on*. The Spearman correlation of IS vs OOS Sharpe across the grid is
   printed — if it is ~0, in-sample tuning is noise.
2. **Walk-forward**: parameters re-chosen every January from all prior years; only the stitched OOS series is scored.
3. **Placebos** (200 trials each): time-shift Pelosi's filings by ±6–24 months; replace her tickers with random
   large caps; replace Cramer's tickers with random universe names. Reported as p = P(placebo ≥ actual).
4. **Stationary block bootstrap** (21-day blocks) confidence interval for the Sharpe ratio.
5. **One-at-a-time rule sensitivity** for every discretionary modelling choice (signal types, options handling,
   weighting, hedge instrument, cost levels).
6. **Disclosure-lag sensitivity** (+1/+5/+21/+45 sessions after filing, plus the lookahead reference).

## 5. Reproduce

```bash
cd pelosi-cramer-neutral
pip install -r requirements.txt
python scripts/fetch_data.py        # optional: re-download PTRs, Cramer data, prices (Yahoo is slow/throttled)
python scripts/run_backtest.py      # ~15 min; writes results/report.md, results/figures, results/tables
python scripts/run_diversification.py  # ~30 s; writes results/diversification.md (blends, IS/OOS candidates)
python -m pytest tests              # parser, point-in-time and hedging guarantees
```

The committed `data/` folder already contains everything `run_backtest.py` needs, so the report can be
regenerated offline.

## 6. Layout

```
pcn/
  config.py          StrategyParams (all tunables), benchmarks, IS/OOS split, grid
  data/house_clerk.py  Clerk index + PTR PDF parser  -> transactions with txn_date AND filing_date
  data/cramer.py       Kull transcript signals + TheStreet cross-check
  data/prices.py       Yahoo + FIGI price panel, cleaning, jump audit, T-bill conversion
  signals.py         point-in-time target weights for each leg
  portfolio.py       L/S engine, holdings-based (multi-factor) beta, hedge overlay, vol target, costs/borrow
  metrics.py         Sharpe, Sortino, MDD, alpha/beta (HAC), IR, rolling beta ...
  robustness.py      grid, walk-forward, lag sensitivity, rule sensitivity, placebos, bootstrap
  diversification.py correlations, blends/overlays, conditional returns, leg mix, IS->OOS candidate tests
  report.py          figures + markdown tables
scripts/fetch_data.py, scripts/run_backtest.py, scripts/run_diversification.py
data/                committed inputs (transactions, signals, price panel)
results/             report.md, diversification.md, figures/, tables/
tests/
```

## 7. Caveats (read before believing any number)

* Option leverage is **not** modelled — a disclosed LEAPS purchase is a delta-one long in the underlying.
* Disclosed sizes are ranges; equal weighting is the default.
* Borrow is a flat fee; no hard-to-borrow or locate constraints on the Cramer shorts.
* The long book usually holds < 6 names; the beta hedge removes market risk, not idiosyncratic or factor risk
  (2022 shows the growth-factor exposure).
* Seven joint years and ~110 long entries is a small sample. Use the bootstrap CIs and placebo p-values.

## Attribution

Cramer signals and the FIGI price matrix: Kull, A. (2026). *What Cramer Holds vs What He Recommends: 16,701 Mad
Money Recommendations (2018–2024)*, CC-BY-4.0, <https://github.com/andreskull/cramer-mad-money-research>.
Pelosi transactions: Office of the Clerk, U.S. House of Representatives, public financial-disclosure records.
