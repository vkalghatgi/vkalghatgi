"""Daily long/short portfolio engine with a trailing-beta SPY overlay.

Accounting (per trading day t, NAV normalised to 1):

    r_p(t) = r_long(t) - r_short(t) + h(t-1) * r_SPY(t)
             + rf(t) * cash(t-1) - borrow(t) - costs(t)

* ``r_long`` / ``r_short`` are the returns of the equal-weighted baskets scaled to
  their notionals (``long_notional``, ``short_notional``) when populated.
* ``h`` is the SPY hedge weight chosen at the close of t-1 from *trailing* leg
  betas, so that ex-ante net beta = ``beta_target``.
* ``cash = 1 - gross_long + gross_short`` (short proceeds held as collateral).
* costs = ``cost_bps`` x turnover (target-vs-drifted weights, all books incl. hedge).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyParams
from .data.prices import to_returns
from .signals import cramer_weights, pelosi_weights, restrict_to_priced


@dataclass
class MarketData:
    prices: pd.DataFrame          # adjusted closes, columns = tickers
    rf_daily: pd.Series           # daily risk-free simple rate
    calendar: pd.DatetimeIndex    # trading days (from SPY)

    @property
    def returns(self) -> pd.DataFrame:
        if not hasattr(self, "_ret"):
            self._ret = to_returns(self.prices)
        return self._ret


def basket_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (basket return, turnover) for target weights held at each close.

    Basket return on day t uses weights from t-1. Turnover on day t is the
    distance between the target at t and the weights drifted from t-1 to t.
    """
    if weights.shape[1] == 0:
        z = pd.Series(0.0, index=weights.index)
        return z, z.copy()
    r = returns.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    w_prev = weights.shift(1).fillna(0.0)
    ret = (w_prev * r).sum(axis=1)
    drifted = w_prev * (1 + r)
    gross_prev = w_prev.sum(axis=1)
    drifted = drifted.div((1 + ret).where(gross_prev > 0, 1.0), axis=0)
    turnover = (weights - drifted).abs().sum(axis=1)
    return ret, turnover


def trailing_beta(leg: pd.Series, mkt: pd.Series, window: int, prior: float = 1.0, prior_obs: int = 20) -> pd.Series:
    """Rolling OLS beta using only data up to and including each date, shrunk toward ``prior``.

    Days where the leg was empty (return exactly 0 with no positions) are excluded
    from the estimate by the caller via NaNs.
    """
    cov = leg.rolling(window, min_periods=max(20, window // 4)).cov(mkt)
    var = mkt.rolling(window, min_periods=max(20, window // 4)).var()
    n = leg.rolling(window, min_periods=1).count()
    b = cov / var
    shrunk = (n * b + prior_obs * prior) / (n + prior_obs)
    return shrunk.fillna(prior)


_STOCK_BETA_CACHE: dict[tuple, pd.DataFrame] = {}


def stock_betas(returns: pd.DataFrame, mkt: pd.Series, window: int, prior: float = 1.0, prior_obs: int = 20) -> pd.DataFrame:
    """Trailing beta of every stock to the market, shrunk toward ``prior`` (cached per window)."""
    key = (id(returns), window, prior, prior_obs)
    if key not in _STOCK_BETA_CACHE:
        minp = max(20, window // 4)
        cov = returns.rolling(window, min_periods=minp).cov(mkt)
        var = mkt.rolling(window, min_periods=minp).var()
        n = returns.rolling(window, min_periods=1).count()
        b = cov.div(var, axis=0)
        shrunk = (n * b + prior_obs * prior) / (n + prior_obs)
        # single-name betas outside [-1, 4] are estimation noise / data artefacts, not information
        _STOCK_BETA_CACHE[key] = shrunk.fillna(prior).clip(lower=-1.0, upper=4.0)
    return _STOCK_BETA_CACHE[key]


def holdings_beta(weights: pd.DataFrame, betas: pd.DataFrame) -> pd.Series:
    """Beta-dollars of a book: sum_i w_i(t) * beta_i(t) (unit-notional beta = this / gross)."""
    if weights.shape[1] == 0:
        return pd.Series(0.0, index=weights.index)
    b = betas.reindex(index=weights.index, columns=weights.columns).fillna(1.0)
    return (weights * b).sum(axis=1)


def run_backtest(
    tx: pd.DataFrame,
    signals: pd.DataFrame,
    md: MarketData,
    params: StrategyParams,
    start: str | None = None,
    end: str | None = None,
    use_transaction_date: bool = False,
    long_weights: pd.DataFrame | None = None,
    short_weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cal = md.calendar
    rets = md.returns

    wl = pelosi_weights(tx, cal, params, use_transaction_date) if long_weights is None else long_weights
    ws = cramer_weights(signals, cal, params) if short_weights is None else short_weights
    wl, _ = restrict_to_priced(wl, md.prices, params.pelosi_max_weight)
    ws, _ = restrict_to_priced(ws, md.prices, 1.0)
    wl = wl * params.long_notional
    ws = ws * params.short_notional

    r_long, to_long = basket_returns(wl, rets)
    r_short, to_short = basket_returns(ws, rets)
    gross_long = wl.sum(axis=1)
    gross_short = ws.sum(axis=1)
    n_long = (wl > 0).sum(axis=1)
    n_short = (ws > 0).sum(axis=1)

    spy = rets[params.hedge_ticker].reindex(cal).fillna(0.0)   # hedge instrument returns (SPY by default)
    gl_prev, gs_prev = gross_long.shift(1), gross_short.shift(1)
    if params.beta_method == "holdings":
        # beta of what we actually hold tonight: sum of position weight x trailing stock beta
        sb = stock_betas(rets, spy, params.beta_window)
        bd_long, bd_short = holdings_beta(wl, sb), holdings_beta(ws, sb)
        beta_long = (bd_long / gross_long).where(gross_long > 0, 1.0)
        beta_short = (bd_short / gross_short).where(gross_short > 0, 1.0)
    else:
        # beta of the leg's own trailing return series (unit-notional, NaN when the leg is empty)
        unit_long = (r_long / gl_prev).where(gl_prev > 0)
        unit_short = (r_short / gs_prev).where(gs_prev > 0)
        beta_long = trailing_beta(unit_long, spy, params.beta_window)
        beta_short = trailing_beta(unit_short, spy, params.beta_window)

    # hedge decided at the close of t using information through t
    net_beta_exante = gross_long * beta_long - gross_short * beta_short
    hedge = (-(net_beta_exante - params.beta_target)) if params.beta_hedge else pd.Series(0.0, index=cal)
    hedge = hedge.fillna(0.0).clip(-params.max_hedge, params.max_hedge)
    r_hedge = hedge.shift(1).fillna(0.0) * spy
    hedge_drift = hedge.shift(1).fillna(0.0) * (1 + spy)
    to_hedge = (hedge - hedge_drift).abs()

    cash = (1.0 - gross_long + gross_short).shift(1).fillna(1.0)
    r_cash = md.rf_daily.reindex(cal).fillna(0.0) * cash if params.earn_cash_rf else pd.Series(0.0, index=cal)
    borrow = gs_prev.fillna(0.0) * params.borrow_bps_annual / 1e4 / 252
    costs = (to_long + to_short + to_hedge) * params.cost_bps / 1e4

    out = pd.DataFrame(
        {
            "ret": r_long - r_short + r_hedge + r_cash - borrow - costs,
            "ret_gross": r_long - r_short + r_hedge,
            "ret_long": r_long,
            "ret_short": -r_short,
            "ret_hedge": r_hedge,
            "ret_cash": r_cash,
            "borrow": borrow,
            "costs": costs,
            "turnover": to_long + to_short + to_hedge,
            "gross_long": gross_long,
            "gross_short": gross_short,
            "hedge_w": hedge,
            "n_long": n_long,
            "n_short": n_short,
            "beta_long_est": beta_long,
            "beta_short_est": beta_short,
            "net_beta_exante": net_beta_exante + hedge,
            "spy": spy,
            "rf": md.rf_daily.reindex(cal).fillna(0.0),
        },
        index=cal,
    )
    if start:
        out = out[out.index >= pd.Timestamp(start)]
    if end:
        out = out[out.index <= pd.Timestamp(end)]
    return out


def leg_only(bt_params: StrategyParams, which: str) -> StrategyParams:
    """Parameters for a single hedged leg (the other leg switched off)."""
    if which == "long":
        return bt_params.with_(short_notional=0.0)
    if which == "short":
        return bt_params.with_(long_notional=0.0)
    raise ValueError(which)
