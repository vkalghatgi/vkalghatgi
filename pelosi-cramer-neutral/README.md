# Long Pelosi / Short Cramer — beta-neutral backtest

A point-in-time-correct backtest of the meme: **go long what Nancy Pelosi discloses buying, short what Jim Cramer
tells you to buy, hedge the residual market beta to zero.**

The full results write-up (statistics, IS/OOS, walk-forward, placebos, bootstrap, figures) is in
[`results/report.md`](results/report.md). This file explains what was built, why the sample is what it is, and
how to reproduce it.

---

## TL;DR

See the headline block at the top of [`results/report.md`](results/report.md). In one paragraph: over the seven
years for which both legs have machine-readable data (2018-01 → 2024-12) the beta-neutral book has a positive but
statistically weak net Sharpe (bootstrap CI straddles zero), an alpha *t*-stat around 1, a ~30 % max drawdown
(2022, when the mega-cap-tech long book fell while the broad short book did not), and a full-sample SPY beta of
≈0 with the realised rolling 1-year beta inside ±0.10 on most days. Almost all of the P&L comes from the Pelosi
leg; fading Cramer is roughly a zero-alpha, cost-consuming diversifier. The **disclosure lag does not hurt** the
Pelosi leg — trading on the (illegal) trade date is no better than trading on the filing date — which is itself
evidence that the leg's return is a large-cap-tech tilt rather than short-lived private information.

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
python -m pytest tests              # parser + point-in-time guarantees
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
  portfolio.py       L/S engine, holdings-based beta, SPY overlay, costs/borrow
  metrics.py         Sharpe, Sortino, MDD, alpha/beta (HAC), IR, rolling beta ...
  robustness.py      grid, walk-forward, lag sensitivity, rule sensitivity, placebos, bootstrap
  report.py          figures + markdown tables
scripts/fetch_data.py, scripts/run_backtest.py
data/                committed inputs (transactions, signals, price panel)
results/             report.md, figures/, tables/
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
