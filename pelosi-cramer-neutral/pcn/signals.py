"""Turn disclosures / on-air calls into daily target weights, point-in-time correctly.

Convention
----------
``weights.loc[D, ticker]`` is the target weight held **at the close of day D**.
It earns the return from D's close to D+1's close. A piece of information that
becomes public on calendar date ``P`` is first *actionable* on the first trading
session strictly after ``P`` (``next_session(P)``) — the filing/broadcast can
happen after the close, so the same-day close is never assumed tradeable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyParams


def next_session(dates: pd.Series, calendar: pd.DatetimeIndex, extra_lag: int = 0) -> pd.Series:
    """First trading day strictly after each date, shifted by ``extra_lag`` sessions."""
    idx = calendar.searchsorted(pd.to_datetime(dates).values, side="right") + extra_lag
    out = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    ok = idx < len(calendar)
    out[ok] = calendar[idx[ok]]
    return out


def _intervals_to_weights(
    intervals: pd.DataFrame, calendar: pd.DatetimeIndex, size_col: str | None = None, max_weight: float = 1.0
) -> pd.DataFrame:
    """intervals: columns [ticker, start, end, (size)] with inclusive trading-day bounds."""
    if intervals.empty:
        return pd.DataFrame(index=calendar, dtype=float)
    tickers = sorted(intervals.ticker.unique())
    pos = {t: i for i, t in enumerate(tickers)}
    m = np.zeros((len(calendar), len(tickers)))
    cal_pos = {d: i for i, d in enumerate(calendar)}
    for r in intervals.itertuples(index=False):
        if pd.isna(r.start) or r.start not in cal_pos:
            continue
        i0 = cal_pos[r.start]
        i1 = cal_pos[r.end] if (pd.notna(r.end) and r.end in cal_pos) else len(calendar) - 1
        sz = getattr(r, size_col) if size_col else 1.0
        m[i0: i1 + 1, pos[r.ticker]] += sz
    m = np.where(m > 0, m, 0.0)
    tot = m.sum(axis=1, keepdims=True)
    w = np.divide(m, tot, out=np.zeros_like(m), where=tot > 0)
    w = np.minimum(w, max_weight)   # cap => leg may be < 100% invested when few names are held
    return pd.DataFrame(w, index=calendar, columns=tickers)


# --------------------------------------------------------------------------- Pelosi
def pelosi_events(tx: pd.DataFrame, params: StrategyParams, use_transaction_date: bool = False) -> pd.DataFrame:
    """One row per (public) buy or sell event with the date it became knowable.

    ``use_transaction_date=True`` deliberately *violates* point-in-time discipline
    (acts on the trade date instead of the filing date). It is only used to
    quantify the cost of the disclosure lag - never as a tradeable strategy.
    """
    t = tx[tx.ticker.notna()].copy()
    bullish = {"stock", "exercise"} | ({"call"} if params.pelosi_include_options else set())
    t["event"] = np.where((t.txn_type == "P") & t.instrument.isin(bullish), "buy",
                          np.where(t.txn_type == "S", "sell", None))
    t = t[t.event.notna()]
    t["public_date"] = t["txn_date"] if use_transaction_date else t["filing_date"]
    t["size"] = (t.amount_low + t.amount_high) / 2.0
    return t[["ticker", "event", "public_date", "txn_date", "filing_date", "size", "instrument"]].reset_index(drop=True)


def pelosi_weights(
    tx: pd.DataFrame, calendar: pd.DatetimeIndex, params: StrategyParams, use_transaction_date: bool = False
) -> pd.DataFrame:
    ev = pelosi_events(tx, params, use_transaction_date)
    ev["action_date"] = next_session(ev.public_date, calendar, params.pelosi_extra_lag_days)
    ev = ev.dropna(subset=["action_date"]).sort_values(["action_date", "event"])

    intervals = []
    for tkr, g in ev.groupby("ticker"):
        open_lots: list[dict] = []
        for r in g.itertuples(index=False):
            if r.event == "buy":
                i0 = calendar.get_loc(r.action_date)
                i1 = min(i0 + params.pelosi_hold_days, len(calendar) - 1)
                open_lots.append(dict(ticker=tkr, start=r.action_date, end=calendar[i1], size=r.size))
            elif r.event == "sell" and params.pelosi_close_on_sale:
                # a disclosed sale closes every lot still open on that date
                for lot in open_lots:
                    if lot["end"] >= r.action_date:
                        lot["end"] = calendar[max(calendar.get_loc(r.action_date) - 1, 0)]
        intervals.extend(open_lots)
    iv = pd.DataFrame(intervals, columns=["ticker", "start", "end", "size"])
    iv = iv[iv.end >= iv.start]
    size_col = "size" if params.pelosi_weighting == "amount" else None
    return _intervals_to_weights(iv, calendar, size_col, params.pelosi_max_weight)


# --------------------------------------------------------------------------- Cramer
def cramer_weights(
    signals: pd.DataFrame, calendar: pd.DatetimeIndex, params: StrategyParams, hold_days: int | None = None
) -> pd.DataFrame:
    """Equal-weighted basket of names Cramer was bullish on, each held ``cramer_hold_days`` sessions."""
    s = signals[signals.signal_type.isin(params.cramer_signal_types)]
    s = s[s.confidence_score.fillna(1.0) >= params.cramer_min_confidence].copy()
    s["action_date"] = next_session(s.signal_date, calendar, params.cramer_extra_lag_days)
    s = s.dropna(subset=["action_date"])
    hd = params.cramer_hold_days if hold_days is None else hold_days
    idx = calendar.get_indexer(s.action_date)
    end_idx = np.minimum(idx + hd, len(calendar) - 1)
    iv = pd.DataFrame({"ticker": s.ticker_symbol.values, "start": s.action_date.values, "end": calendar[end_idx]})
    # overlapping calls on the same name are one position (extend the holding window)
    iv = iv.sort_values(["ticker", "start"])
    merged = []
    for tkr, g in iv.groupby("ticker"):
        cur = None
        for r in g.itertuples(index=False):
            if cur is None or r.start > cur["end"]:
                if cur is not None:
                    merged.append(cur)
                cur = dict(ticker=tkr, start=r.start, end=r.end)
            else:
                cur["end"] = max(cur["end"], r.end)
        merged.append(cur)
    return _intervals_to_weights(pd.DataFrame(merged), calendar, None, 1.0)


def restrict_to_priced(
    weights: pd.DataFrame, prices: pd.DataFrame, max_weight: float = 1.0
) -> tuple[pd.DataFrame, list[str]]:
    """Drop tickers without price history; zero weights where no price exists that day; renormalise."""
    have = [t for t in weights.columns if t in prices.columns]
    dropped = [t for t in weights.columns if t not in prices.columns]
    w = weights[have].copy()
    avail = prices[have].reindex(w.index).notna()
    w = w.where(avail, 0.0)
    tot = w.sum(axis=1)
    target = weights.sum(axis=1)
    scale = (target / tot).replace([np.inf, -np.inf], 0).fillna(0)
    return w.mul(scale, axis=0).clip(upper=max_weight), dropped
