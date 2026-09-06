"""Fetch and normalise every input dataset.

Usage:  python scripts/fetch_data.py [--no-prices]

Outputs (all under data/):
    pelosi_ptr_index.csv        one row per Pelosi PTR filing (filing date, DocID)
    pelosi_transactions.csv     parsed transactions with txn_date AND filing_date
    cramer_signals.csv          one row per Mad Money directional call (Kull 2026, CC-BY-4.0)
    cramer_thestreet_calls.csv  TheStreet screener scrape (2020-2022) for cross-checking
    cache/prices.parquet        adjusted daily closes for the whole universe + benchmarks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from pcn.config import BENCHMARKS, CACHE_DIR, DATA_DIR, HEDGE_TICKER, RISK_FREE_TICKER  # noqa: E402
from pcn.data.cramer import build_cramer_signals, build_thestreet_calls  # noqa: E402
from pcn.data.house_clerk import build_pelosi_transactions  # noqa: E402
from pcn.data.prices import (  # noqa: E402
    audit_jumps, clean_prices, combine_price_sources, download_yahoo_sequential, load_figi_prices,
)

PRICE_START = "2014-06-01"
PRICE_END = "2026-09-05"


def main(with_prices: bool = True) -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print("== House Clerk: Pelosi PTRs")
    filings, tx = build_pelosi_transactions(CACHE_DIR)
    filings.to_csv(DATA_DIR / "pelosi_ptr_index.csv", index=False)
    tx.to_csv(DATA_DIR / "pelosi_transactions.csv", index=False)
    print(f"   {len(filings)} filings, {len(tx)} transactions, "
          f"{tx.ticker.notna().sum()} with tickers, "
          f"{tx.filing_date.min().date()} → {tx.filing_date.max().date()}")

    print("== Cramer: transcript signals (Kull 2026)")
    cr = build_cramer_signals(CACHE_DIR)
    cr.to_csv(DATA_DIR / "cramer_signals.csv", index=False)
    print(f"   {len(cr)} signals, {cr.ticker_symbol.nunique()} tickers, "
          f"{cr.signal_date.min().date()} → {cr.signal_date.max().date()}")

    print("== Cramer: TheStreet screener (cross-check)")
    ts = build_thestreet_calls(CACHE_DIR)
    ts.to_csv(DATA_DIR / "cramer_thestreet_calls.csv", index=False)
    print(f"   {len(ts)} calls, {ts.date.min().date()} → {ts.date.max().date()}")

    if not with_prices:
        return
    print("== Prices: FIGI matrix (Kull 2026) for the Cramer universe")
    figi = load_figi_prices(CACHE_DIR / "cramer_snapshot_full.csv", CACHE_DIR)
    print(f"   {figi.shape[1]} tickers, {figi.index.min().date()} → {figi.index.max().date()}")

    print("== Prices: Yahoo for Pelosi tickers, benchmarks, T-bill (sequential, throttled)")
    yahoo_set = sorted(set(tx.ticker.dropna()) | set(BENCHMARKS) | {HEDGE_TICKER, RISK_FREE_TICKER, "^VIX"})
    yahoo = download_yahoo_sequential(yahoo_set, PRICE_START, PRICE_END, CACHE_DIR / "yahoo_prices.parquet")
    missing = sorted(t for t in yahoo_set if t not in yahoo.columns)
    print(f"   {len([t for t in yahoo_set if t in yahoo.columns])}/{len(yahoo_set)} series; missing on Yahoo: {missing}")

    panel = clean_prices(combine_price_sources(yahoo, figi, prefer_yahoo=yahoo_set))
    panel.astype("float32").to_parquet(DATA_DIR / "prices.parquet")
    print(f"   combined panel: {panel.shape[1]} series, {panel.index.min().date()} → {panel.index.max().date()}")
    jumps = audit_jumps(panel)
    jumps.to_csv(DATA_DIR / "price_jump_audit.csv", index=False)
    print(f"   audit: {len(jumps)} persistent one-day jumps > +200% remain (see data/price_jump_audit.csv)")
    no_px = sorted((set(tx.ticker.dropna()) | set(cr.ticker_symbol)) - set(panel.columns))
    pd.Series(no_px, name="ticker").to_csv(DATA_DIR / "tickers_missing_prices.csv", index=False)
    print(f"   tickers with no prices at all: {no_px}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-prices", action="store_true")
    a = ap.parse_args()
    main(with_prices=not a.no_prices)
