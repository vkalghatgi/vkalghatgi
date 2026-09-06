"""Daily total-return (split- and dividend-adjusted) close prices.

Two sources, one per universe:

* **Cramer universe (2018-01 → 2026-04):** the FIGI-keyed ``daily_price_matrix.parquet``
  published with Kull (2026) under CC-BY-4.0. It covers 1,385 of the 1,386 tickers
  that appear in the transcript signals **including names that were later delisted**,
  so the short leg is free of survivorship bias.
* **Pelosi tickers, benchmarks, T-bill (2014-06 → present):** Yahoo Finance, which
  is needed for pre-2018 history. Yahoo lacks delisted names; the few affected
  Pelosi tickers are listed in the report.

Where a ticker exists in both, Yahoo is used (longer history); the two sources
agree to within rounding on overlapping dates.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

KULL_PRICES_URL = (
    "https://raw.githubusercontent.com/andreskull/cramer-mad-money-research/"
    "public-main/data/daily_price_matrix.parquet"
)


def load_figi_prices(signals_raw_csv: Path, cache_dir: Path) -> pd.DataFrame:
    """Ticker-keyed adjusted closes for every FIGI referenced by the Cramer signals."""
    p = cache_dir / "daily_price_matrix.parquet"
    if not p.exists():
        r = requests.get(KULL_PRICES_URL, timeout=300)
        r.raise_for_status()
        p.write_bytes(r.content)
    px = pd.read_parquet(p, columns=["trade_date", "figi_id", "adj_close"])
    px["adj_close"] = px["adj_close"].astype(float)
    fig = pd.read_csv(signals_raw_csv, usecols=["figi_id", "ticker_symbol"], low_memory=False).drop_duplicates()
    fig["ticker_symbol"] = fig["ticker_symbol"].str.upper().str.strip()
    px = px.merge(fig, on="figi_id", how="inner")
    # a ticker can map to >1 FIGI (re-listings); keep the FIGI with the longest history and fill gaps from the other
    counts = px.groupby(["ticker_symbol", "figi_id"]).size().rename("n").reset_index()
    order = counts.sort_values(["ticker_symbol", "n"], ascending=[True, False])
    px = px.merge(order[["ticker_symbol", "figi_id"]].assign(rank=order.groupby("ticker_symbol").cumcount()),
                  on=["ticker_symbol", "figi_id"])
    px = px.sort_values("rank").drop_duplicates(["trade_date", "ticker_symbol"], keep="first")
    wide = px.pivot(index="trade_date", columns="ticker_symbol", values="adj_close").sort_index()
    wide.index = pd.to_datetime(wide.index)
    return wide


def download_yahoo_sequential(tickers: list[str], start: str, end: str, cache: Path, pause: float = 2.5) -> pd.DataFrame:
    """Download one symbol at a time (Yahoo throttles batch requests aggressively); cached."""
    import yfinance as yf

    cache.parent.mkdir(parents=True, exist_ok=True)
    have = pd.read_parquet(cache) if cache.exists() else pd.DataFrame()
    need = [t for t in dict.fromkeys(tickers) if t not in have.columns and _yahoo_symbol(t)]
    new = {}
    for t in need:
        sym = _yahoo_symbol(t)
        for attempt in range(4):
            try:
                d = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False, threads=False)
                if d is not None and not d.empty:
                    s = d["Close"]
                    s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
                    new[t] = s.astype(float)
                    break
                time.sleep(pause * (attempt + 1))
            except Exception:  # pragma: no cover - network
                time.sleep(pause * 2 * (attempt + 1))
        else:
            print(f"   {t}: unavailable on Yahoo")
        time.sleep(pause)
    if new:
        add = pd.DataFrame(new)
        add.index = pd.to_datetime(add.index).tz_localize(None)
        have = pd.concat([have, add], axis=1) if not have.empty else add
        have = have.loc[:, ~have.columns.duplicated()].sort_index()
        have.to_parquet(cache)
    have.index = pd.to_datetime(have.index)
    return have


def combine_price_sources(yahoo: pd.DataFrame, figi: pd.DataFrame, prefer_yahoo: list[str]) -> pd.DataFrame:
    """Single wide panel: Yahoo for ``prefer_yahoo`` tickers (when available), FIGI matrix for everything else."""
    cols = {}
    for t in figi.columns:
        cols[t] = figi[t]
    for t in prefer_yahoo:
        if t in yahoo.columns and yahoo[t].notna().sum() > 0:
            cols[t] = yahoo[t]
    for t in yahoo.columns:            # benchmarks / indices not in the FIGI universe
        cols.setdefault(t, yahoo[t])
    panel = pd.DataFrame(cols).sort_index()
    return panel.loc[:, panel.notna().sum() > 0]

# Historical renames so that PTR / transcript tickers map to a live Yahoo symbol.
TICKER_ALIASES = {
    "FB": "META",
    "SQ": "XYZ",       # Block, Inc. renamed ticker Jan-2025
    "GOOG": "GOOG",
    "WORK": None,      # Slack — acquired by CRM Jul-2021 (delisted)
    "HTZ": "HTZ",      # Hertz re-listed under the same symbol after Ch.11 (2020-2021 gap)
    "BFET": None,      # private
}


def _yahoo_symbol(t: str) -> str | None:
    if t in TICKER_ALIASES:
        return TICKER_ALIASES[t]
    return t.replace(".", "-")


def _download_batch(symbols: dict[str, str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        sorted(set(symbols.values())), start=start, end=end, auto_adjust=True,
        progress=False, threads=False, group_by="column",
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": next(iter(symbols.values()))})
    cols = {t: close[sym] for t, sym in symbols.items() if sym in close.columns and close[sym].notna().any()}
    return pd.DataFrame(cols, index=close.index)


def download_prices(
    tickers: list[str], start: str, end: str, cache: Path, batch: int = 100, rounds: int = 5, pause: float = 2.0
) -> pd.DataFrame:
    """Return wide DataFrame of adjusted closes indexed by trading day (columns = original tickers).

    Yahoo rate-limits aggressively; symbols that come back empty are retried in
    later rounds with a growing pause so that a throttled batch is not mistaken
    for a delisted ticker.
    """
    cache.parent.mkdir(parents=True, exist_ok=True)
    have = pd.read_parquet(cache) if cache.exists() else pd.DataFrame()
    have.index = pd.to_datetime(have.index)
    need = [t for t in dict.fromkeys(tickers) if t not in have.columns and _yahoo_symbol(t)]
    for rnd in range(rounds):
        if not need:
            break
        frames = []
        for i in range(0, len(need), batch):
            chunk = need[i: i + batch]
            try:
                frames.append(_download_batch({t: _yahoo_symbol(t) for t in chunk}, start, end))
            except Exception as e:  # pragma: no cover - network
                print(f"   batch failed ({type(e).__name__}); will retry")
            time.sleep(pause)
        frames = [f for f in frames if not f.empty]
        if frames:
            new = pd.concat(frames, axis=1)
            new.index = pd.to_datetime(new.index).tz_localize(None)
            have = pd.concat([have, new], axis=1) if not have.empty else new
            have = have.loc[:, ~have.columns.duplicated()]
            have.sort_index().to_parquet(cache)
        got = set(have.columns)
        still = [t for t in need if t not in got]
        print(f"   round {rnd + 1}: {len(need) - len(still)} new series, {len(still)} still missing")
        if len(still) == len(need):      # nothing recovered -> remaining are genuinely unavailable
            break
        need = still
        pause *= 2
    have.index = pd.to_datetime(have.index)
    return have.sort_index()


MIN_VALID_PRICE = 0.10   # below this the FIGI matrix contains placeholder / dead-listing quotes

# Corporate actions that the FIGI matrix left unadjusted (verified against Yahoo Finance, which
# adjusts them). The return on these days is treated as unknown (0 P&L, no beta information).
KNOWN_UNADJUSTED_SPLITS = [
    ("VRM", "2024-02-14"),   # Vroom 1:80 reverse split
    ("SEAT", "2021-10-20"),  # Vivid Seats (Horizon SPAC de-SPAC re-basing)
    ("MOVE", "2024-10-29"),  # Movano 1:15 reverse split
    ("MOVE", "2025-10-10"),  # Movano reverse split
]


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    px = prices.copy()
    px[px <= 0] = np.nan
    for t, d in KNOWN_UNADJUSTED_SPLITS:
        if t in px.columns and pd.Timestamp(d) in px.index:
            # splice: scale everything before the split day so the series is continuous
            i = px.index.get_loc(pd.Timestamp(d))
            before, after = px[t].iloc[i - 1], px[t].iloc[i]
            if pd.notna(before) and pd.notna(after) and before > 0:
                px.iloc[:i, px.columns.get_loc(t)] = px[t].iloc[:i] * (after / before)
    return px.where(px >= MIN_VALID_PRICE)


def audit_jumps(prices: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """List persistent one-day jumps larger than ``threshold`` (i.e. > +200%) so new data defects surface."""
    px = prices.where(prices >= MIN_VALID_PRICE)
    r = px.pct_change(fill_method=None)
    prev = px.shift(1).rolling(5, min_periods=3).median()
    nxt = px[::-1].rolling(5, min_periods=3).median()[::-1]
    level_ratio = nxt / prev
    mask = (r > threshold) & (level_ratio > 1 + threshold * 0.8)
    rows = [dict(ticker=t, date=d.date(), ret=r.at[d, t], level_ratio=level_ratio.at[d, t])
            for t in mask.columns[mask.any()] for d in mask.index[mask[t]]]
    return pd.DataFrame(rows, columns=["ticker", "date", "ret", "level_ratio"])


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns; a return is NaN wherever the price on either side is missing."""
    px = prices.copy()
    px[px <= 0] = np.nan
    return px.pct_change(fill_method=None)


def risk_free_daily(irx: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Convert ^IRX (annualised % yield) to a daily simple rate aligned to the trading calendar."""
    r = irx.reindex(index).ffill().bfill().fillna(0.0) / 100.0
    return (1 + r) ** (1 / 252) - 1
