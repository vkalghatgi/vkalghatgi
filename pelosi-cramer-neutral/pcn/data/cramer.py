"""Jim Cramer / *Mad Money* recommendation datasets.

Two independent public sources are used:

1. **Transcript-derived signals (primary), 2018-01 → 2024-12.**
   Kull, A. (2026). *What Cramer Holds vs What He Recommends: 16,701 Mad Money
   Recommendations (2018–2024)*. Data licensed CC-BY-4.0.
   https://github.com/andreskull/cramer-mad-money-research
   One row per on-air directional call; ``signal_type`` is ``start_long`` (a new
   buy/positive call) or ``hold_long`` (a reiterated hold). Long-side calls only.

2. **TheStreet "Mad Money" screener scrape (cross-check), 2020-01 → 2022-06.**
   https://github.com/gaborvecsei/Mad-Money-Backtesting (screener is now offline).
   Includes ``buy/positive/negative/sell`` calls, used to validate the primary
   dataset over the overlapping window and to test the "long Cramer sells" variant.

Point-in-time note
------------------
*Mad Money* airs at 6 pm ET after the close. A call made on ``signal_date`` is
therefore first tradeable at the next session's open. The backtest enters at the
**next session's close** (one more session of slippage than strictly required).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

KULL_URL = (
    "https://raw.githubusercontent.com/andreskull/cramer-mad-money-research/"
    "public-main/data/cramer_snapshot_full.csv"
)
THESTREET_URL = (
    "https://raw.githubusercontent.com/gaborvecsei/Mad-Money-Backtesting/master/"
    "mad_money_recommendations.csv"
)

SLIM_COLUMNS = [
    "signal_id", "signal_date", "ticker_symbol", "signal_type", "hold_subtype",
    "show_segment_heuristic", "section_raw", "confidence_score", "vix_at_signal", "gics_sector",
]


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest


def build_cramer_signals(cache_dir: Path) -> pd.DataFrame:
    """Download the Kull dataset and collapse it to one row per signal."""
    raw = _download(KULL_URL, cache_dir / "cramer_snapshot_full.csv")
    df = pd.read_csv(raw, low_memory=False)
    df = df.drop_duplicates("signal_id")[SLIM_COLUMNS].copy()
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["ticker_symbol"] = df["ticker_symbol"].str.upper().str.strip()
    return df.sort_values(["signal_date", "ticker_symbol"]).reset_index(drop=True)


def build_thestreet_calls(cache_dir: Path) -> pd.DataFrame:
    raw = _download(THESTREET_URL, cache_dir / "mad_money_recommendations.csv")
    df = pd.read_csv(raw)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].str.upper().str.strip()
    return df[["date", "symbol", "call", "segment", "current_price"]].sort_values(["date", "symbol"]).reset_index(drop=True)


def load_cramer_signals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["signal_date"])
    return df


def load_thestreet_calls(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def cross_check(kull: pd.DataFrame, thestreet: pd.DataFrame) -> dict:
    """How well do the two independent sources agree on Cramer's bullish calls?"""
    lo, hi = thestreet["date"].min(), thestreet["date"].max()
    k = kull[(kull.signal_date >= lo) & (kull.signal_date <= hi)]
    k_pairs = set(zip(k.signal_date.dt.normalize(), k.ticker_symbol))
    t_bull = thestreet[thestreet.call.isin(["buy", "positive"])]
    t_pairs = set(zip(t_bull.date.dt.normalize(), t_bull.symbol))
    # allow +/- 1 day tolerance (screener dates are sometimes the next morning)
    t_pairs_tol = set()
    for d, s in t_pairs:
        for off in (-1, 0, 1):
            t_pairs_tol.add((d + pd.Timedelta(days=off), s))
    k_pairs_tol = set()
    for d, s in k_pairs:
        for off in (-1, 0, 1):
            k_pairs_tol.add((d + pd.Timedelta(days=off), s))
    return {
        "overlap_start": str(lo.date()),
        "overlap_end": str(hi.date()),
        "kull_signals_in_window": len(k_pairs),
        "thestreet_bullish_in_window": len(t_pairs),
        "kull_found_in_thestreet_pct": round(100 * sum(p in t_pairs_tol for p in k_pairs) / max(len(k_pairs), 1), 1),
        "thestreet_found_in_kull_pct": round(100 * sum(p in k_pairs_tol for p in t_pairs) / max(len(t_pairs), 1), 1),
    }
