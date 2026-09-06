"""Multi-factor hedge betas and vol targeting behave as advertised and never use future data."""
import numpy as np
import pandas as pd

from pcn.diversification import blend
from pcn.portfolio import stock_betas, stock_factor_betas, vol_target_scale

IDX = pd.bdate_range("2015-01-01", "2019-12-31")


def _synthetic():
    rng = np.random.default_rng(1)
    T = len(IDX)
    spy = pd.Series(rng.normal(0, 0.01, T), IDX)
    growth = pd.Series(rng.normal(0, 0.006, T), IDX)
    qqq = spy + growth                                 # QQQ = market + growth factor
    # stock A: beta 1.0 to SPY, 0 to growth ; stock B: 0.5 SPY + 1.0 growth  => on (SPY, QQQ): B = (-0.5, 1.0)
    a = 1.0 * spy + pd.Series(rng.normal(0, 0.01, T), IDX)
    b = 0.5 * spy + 1.0 * growth + pd.Series(rng.normal(0, 0.01, T), IDX)
    rets = pd.DataFrame({"A": a, "B": b, "SPY": spy, "QQQ": qqq})
    return rets, pd.DataFrame({"SPY": spy, "QQQ": qqq})


def test_two_factor_betas_recover_true_loadings():
    rets, factors = _synthetic()
    fb = stock_factor_betas(rets, factors, window=504)
    last = IDX[-1]
    assert abs(fb["SPY"].loc[last, "A"] - 1.0) < 0.15
    assert abs(fb["QQQ"].loc[last, "A"] - 0.0) < 0.15
    assert abs(fb["SPY"].loc[last, "B"] - (-0.5)) < 0.15
    assert abs(fb["QQQ"].loc[last, "B"] - 1.0) < 0.15


def test_single_factor_path_matches_stock_betas():
    rets, factors = _synthetic()
    one = stock_factor_betas(rets, factors[["SPY"]], window=126)["SPY"]
    ref = stock_betas(rets, factors["SPY"], 126)
    pd.testing.assert_frame_equal(one, ref)


def test_factor_betas_are_causal():
    rets, factors = _synthetic()
    fb_full = stock_factor_betas(rets, factors, window=126)["QQQ"]
    cut = IDX[600]
    fb_cut = stock_factor_betas(rets.loc[:cut].copy(), factors.loc[:cut].copy(), window=126)["QQQ"]
    pd.testing.assert_series_equal(fb_full.loc[:cut, "B"], fb_cut["B"])


def test_vol_target_scale_uses_only_past_and_respects_cap():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0, 0.02, len(IDX)), IDX)
    lev = vol_target_scale(r, target=0.10, window=63, max_lev=1.5)
    assert lev.max() <= 1.5 + 1e-12
    assert (lev.iloc[:30] == 1.0).all()                         # no estimate yet -> unit leverage
    # leverage at t must not change if returns after t change
    r2 = r.copy()
    r2.iloc[700:] = 0.0
    lev2 = vol_target_scale(r2, target=0.10, window=63, max_lev=1.5)
    pd.testing.assert_series_equal(lev.iloc[:700], lev2.iloc[:700])
    # realised vol of the scaled series is close to target
    scaled = lev.shift(1).fillna(1.0) * r
    assert abs(scaled.iloc[100:].std() * np.sqrt(252) - 0.10) < 0.02


def test_blend_modes():
    rf = pd.Series(0.0001, IDX)
    strat = pd.Series(0.001, IDX)
    base = pd.Series(0.0005, IDX)
    assert np.isclose(blend(strat, base, 0.0, rf, "fund").iloc[0], 0.0005)
    assert np.isclose(blend(strat, base, 1.0, rf, "fund").iloc[0], 0.001)
    assert np.isclose(blend(strat, base, 0.5, rf, "overlay").iloc[0], 0.0005 + 0.5 * (0.001 - 0.0001))
