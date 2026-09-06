"""Guarantees that no signal is acted on before it was public and that P&L uses yesterday's weights."""
import numpy as np
import pandas as pd

from pcn.config import StrategyParams
from pcn.data.prices import risk_free_daily
from pcn.portfolio import MarketData, basket_returns, run_backtest
from pcn.signals import cramer_weights, next_session, pelosi_weights

CAL = pd.bdate_range("2020-01-01", "2020-12-31")


def _tx(rows):
    df = pd.DataFrame(rows, columns=["ticker", "txn_type", "txn_date", "filing_date", "instrument", "amount_low", "amount_high"])
    df["txn_date"] = pd.to_datetime(df["txn_date"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df


def test_next_session_is_strictly_after_public_date():
    d = pd.Series(pd.to_datetime(["2020-03-06", "2020-03-07", "2020-03-09"]))  # Fri, Sat, Mon
    out = next_session(d, CAL)
    assert list(out.dt.strftime("%Y-%m-%d")) == ["2020-03-09", "2020-03-09", "2020-03-10"]
    assert (out > d).all()


def test_pelosi_position_opens_after_filing_not_after_trade():
    tx = _tx([("AAPL", "P", "2020-01-15", "2020-02-14", "stock", 1e6, 5e6)])
    p = StrategyParams(pelosi_hold_days=10)
    w = pelosi_weights(tx, CAL, p)
    assert w.loc["2020-01-16":"2020-02-14", "AAPL"].eq(0).all()      # nothing between trade and filing
    assert w.loc["2020-02-18", "AAPL"] > 0                             # first session after the filing (17th is a holiday-free Monday? use >=)
    first = w.index[w["AAPL"] > 0][0]
    assert first == CAL[CAL.searchsorted(pd.Timestamp("2020-02-14"), side="right")]
    assert (w["AAPL"] > 0).sum() == 11                                  # entry day + hold_days


def test_disclosed_sale_closes_position_only_once_filed():
    tx = _tx([
        ("AAPL", "P", "2020-01-15", "2020-01-20", "stock", 1e6, 5e6),
        ("AAPL", "S", "2020-03-02", "2020-04-01", "stock", 1e6, 5e6),   # sold in March, filed in April
    ])
    w = pelosi_weights(tx, CAL, StrategyParams(pelosi_hold_days=252))
    assert w.loc["2020-03-31", "AAPL"] > 0          # still long: sale not yet public
    assert w.loc["2020-04-01", "AAPL"] > 0          # filing day itself: still held at the close (filed possibly after hours)
    assert w.loc["2020-04-02", "AAPL"] == 0          # exited at the close of the session after the filing


def test_cramer_entry_is_next_session_and_positions_merge():
    sig = pd.DataFrame({
        "signal_date": pd.to_datetime(["2020-06-01", "2020-06-03"]), "ticker_symbol": ["XYZ", "XYZ"],
        "signal_type": ["start_long", "start_long"], "confidence_score": [0.9, 0.9],
    })
    w = cramer_weights(sig, CAL, StrategyParams(cramer_hold_days=5))
    assert w.loc["2020-06-01", "XYZ"] == 0 and w.loc["2020-06-02", "XYZ"] == 1.0
    # second call on 06-03 extends: 06-04 + 5 sessions -> last day 06-11
    assert w.loc["2020-06-11", "XYZ"] == 1.0 and w.loc["2020-06-12", "XYZ"] == 0


def test_basket_return_uses_previous_close_weights():
    idx = CAL[:4]
    w = pd.DataFrame({"A": [0.0, 1.0, 1.0, 0.0]}, index=idx)
    r = pd.DataFrame({"A": [0.10, 0.20, 0.30, 0.40]}, index=idx)
    ret, to = basket_returns(w, r)
    assert list(np.round(ret.values, 6)) == [0.0, 0.0, 0.30, 0.40]   # day-2 weight earns day-3 return, etc.
    assert to.iloc[1] == 1.0 and to.iloc[3] == 1.0


def test_hedge_uses_only_trailing_information():
    """Shifting future market returns must not change today's hedge weight."""
    rng = np.random.default_rng(0)
    idx = CAL
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (len(idx), 3)), axis=0), index=idx, columns=["SPY", "AAPL", "XYZ"])
    px["^IRX"] = 1.0
    md = MarketData(prices=px, rf_daily=risk_free_daily(px["^IRX"], idx), calendar=idx)
    tx = _tx([("AAPL", "P", "2020-01-15", "2020-01-20", "stock", 1e6, 5e6)])
    sig = pd.DataFrame({"signal_date": [pd.Timestamp("2020-01-21")], "ticker_symbol": ["XYZ"],
                        "signal_type": ["start_long"], "confidence_score": [0.9]})
    p = StrategyParams(beta_window=40, cramer_hold_days=200)
    a = run_backtest(tx, sig, md, p)
    px2 = px.copy()
    px2.loc["2020-09-01":, ["SPY", "AAPL", "XYZ"]] *= np.cumprod(1 + rng.normal(0, 0.03, (len(px2.loc["2020-09-01":]), 3)), axis=0)
    md2 = MarketData(prices=px2, rf_daily=md.rf_daily, calendar=idx)
    b = run_backtest(tx, sig, md2, p)
    pd.testing.assert_series_equal(a.loc[:"2020-08-31", "hedge_w"], b.loc[:"2020-08-31", "hedge_w"])
    pd.testing.assert_series_equal(a.loc[:"2020-08-31", "ret"], b.loc[:"2020-08-31", "ret"])
