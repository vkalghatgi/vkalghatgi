"""Performance and risk statistics for daily return series."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

ANN = 252


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    growth = (1 + r).prod()
    years = len(r) / ANN
    return growth ** (1 / years) - 1 if growth > 0 else -1.0


def ann_vol(r: pd.Series) -> float:
    return r.dropna().std(ddof=1) * np.sqrt(ANN)


def sharpe(r: pd.Series, rf: pd.Series | None = None) -> float:
    ex = r - (rf.reindex(r.index).fillna(0.0) if rf is not None else 0.0)
    ex = ex.dropna()
    sd = ex.std(ddof=1)
    return ex.mean() / sd * np.sqrt(ANN) if sd > 0 else np.nan


def sortino(r: pd.Series, rf: pd.Series | None = None) -> float:
    ex = (r - (rf.reindex(r.index).fillna(0.0) if rf is not None else 0.0)).dropna()
    dn = ex[ex < 0]
    dd = np.sqrt((dn ** 2).sum() / len(ex)) if len(ex) else np.nan
    return ex.mean() / dd * np.sqrt(ANN) if dd and dd > 0 else np.nan


def drawdown_series(r: pd.Series) -> pd.Series:
    nav = (1 + r.fillna(0)).cumprod()
    return nav / nav.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return drawdown_series(r).min()


def max_drawdown_duration(r: pd.Series) -> int:
    dd = drawdown_series(r)
    longest = cur = 0
    for v in dd.values:
        cur = cur + 1 if v < 0 else 0
        longest = max(longest, cur)
    return int(longest)


def calmar(r: pd.Series) -> float:
    mdd = max_drawdown(r)
    return cagr(r) / abs(mdd) if mdd < 0 else np.nan


def hit_rate(r: pd.Series) -> float:
    r = r.dropna()
    return (r > 0).mean() if len(r) else np.nan


def alpha_beta(r: pd.Series, bench: pd.Series, rf: pd.Series | None = None, lags: int = 5) -> dict:
    """OLS of excess strategy returns on excess benchmark returns with Newey-West (HAC) errors."""
    df = pd.concat([r, bench], axis=1, keys=["r", "b"]).dropna()
    if rf is not None:
        f = rf.reindex(df.index).fillna(0.0)
        df["r"] -= f
        df["b"] -= f
    if len(df) < 30:
        return dict(alpha_ann=np.nan, alpha_t=np.nan, beta=np.nan, beta_se=np.nan, r2=np.nan, corr=np.nan)
    X = sm.add_constant(df["b"])
    res = sm.OLS(df["r"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return dict(
        alpha_ann=float(res.params["const"] * ANN),
        alpha_t=float(res.tvalues["const"]),
        beta=float(res.params["b"]),
        beta_se=float(res.bse["b"]),
        r2=float(res.rsquared),
        corr=float(df["r"].corr(df["b"])),
    )


def information_ratio(r: pd.Series, bench: pd.Series) -> float:
    act = (r - bench.reindex(r.index)).dropna()
    sd = act.std(ddof=1)
    return act.mean() / sd * np.sqrt(ANN) if sd > 0 else np.nan


def rolling_beta(r: pd.Series, bench: pd.Series, window: int = 252) -> pd.Series:
    df = pd.concat([r, bench], axis=1, keys=["r", "b"]).dropna()
    cov = df["r"].rolling(window).cov(df["b"])
    var = df["b"].rolling(window).var()
    return cov / var


def summary(r: pd.Series, rf: pd.Series, benchmarks: pd.DataFrame, beta_tol: float = 0.10) -> dict:
    """Headline statistics plus benchmark-relative statistics for every benchmark column."""
    r = r.dropna()
    out = {
        "start": str(r.index.min().date()),
        "end": str(r.index.max().date()),
        "years": round(len(r) / ANN, 2),
        "total_return": (1 + r).prod() - 1,
        "cagr": cagr(r),
        "ann_vol": ann_vol(r),
        "sharpe": sharpe(r, rf),
        "sortino": sortino(r, rf),
        "max_drawdown": max_drawdown(r),
        "max_dd_days": max_drawdown_duration(r),
        "calmar": calmar(r),
        "hit_rate_daily": hit_rate(r),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurt()),
        "worst_day": float(r.min()),
        "best_day": float(r.max()),
    }
    for b in benchmarks.columns:
        bb = benchmarks[b].reindex(r.index)
        ab = alpha_beta(r, bb, rf)
        out[f"{b}_cagr"] = cagr(bb)
        out[f"{b}_excess_cagr"] = out["cagr"] - cagr(bb)
        out[f"{b}_excess_total"] = out["total_return"] - ((1 + bb.fillna(0)).prod() - 1)
        out[f"{b}_beta"] = ab["beta"]
        out[f"{b}_beta_se"] = ab["beta_se"]
        out[f"{b}_alpha_ann"] = ab["alpha_ann"]
        out[f"{b}_alpha_t"] = ab["alpha_t"]
        out[f"{b}_corr"] = ab["corr"]
        out[f"{b}_info_ratio"] = information_ratio(r, bb)
        out[f"{b}_sharpe"] = sharpe(bb, rf)
        out[f"{b}_max_dd"] = max_drawdown(bb.fillna(0))
    if "SPY" in benchmarks.columns:
        rb = rolling_beta(r, benchmarks["SPY"].reindex(r.index)).dropna()
        out["spy_rolling_beta_1y_max_abs"] = float(rb.abs().max()) if len(rb) else np.nan
        out["spy_rolling_beta_1y_pct_within_tol"] = float((rb.abs() <= beta_tol).mean()) if len(rb) else np.nan
    return out


def yearly_table(r: pd.Series, benchmarks: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([r.rename("strategy"), benchmarks.reindex(r.index)], axis=1)
    return df.groupby(df.index.year).apply(lambda g: (1 + g).prod() - 1)


def format_summary(s: dict, benchmarks: list[str]) -> pd.DataFrame:
    rows = [
        ("Period", f"{s['start']} → {s['end']} ({s['years']} y)"),
        ("Total return", f"{s['total_return']:.1%}"),
        ("CAGR", f"{s['cagr']:.2%}"),
        ("Annualised vol", f"{s['ann_vol']:.2%}"),
        ("Sharpe (vs T-bill)", f"{s['sharpe']:.2f}"),
        ("Sortino", f"{s['sortino']:.2f}"),
        ("Max drawdown", f"{s['max_drawdown']:.1%} ({s['max_dd_days']} days)"),
        ("Calmar", f"{s['calmar']:.2f}"),
        ("Daily hit rate", f"{s['hit_rate_daily']:.1%}"),
        ("Skew / kurtosis", f"{s['skew']:.2f} / {s['kurtosis']:.1f}"),
        ("Worst / best day", f"{s['worst_day']:.2%} / {s['best_day']:.2%}"),
    ]
    if "spy_rolling_beta_1y_max_abs" in s:
        rows.append(("Rolling 1y |beta| to SPY: max / % within ±0.10",
                     f"{s['spy_rolling_beta_1y_max_abs']:.2f} / {s['spy_rolling_beta_1y_pct_within_tol']:.0%}"))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def format_benchmarks(s: dict, benchmarks: dict[str, str]) -> pd.DataFrame:
    rows = []
    for b, name in benchmarks.items():
        if f"{b}_beta" not in s:
            continue
        rows.append(
            {
                "Benchmark": f"{b} ({name})",
                "Bench CAGR": f"{s[f'{b}_cagr']:.2%}",
                "Bench Sharpe": f"{s[f'{b}_sharpe']:.2f}",
                "Bench MaxDD": f"{s[f'{b}_max_dd']:.1%}",
                "Strategy − Bench CAGR": f"{s[f'{b}_excess_cagr']:+.2%}",
                "Beta (s.e.)": f"{s[f'{b}_beta']:.3f} ({s[f'{b}_beta_se']:.3f})",
                "Alpha ann. (t)": f"{s[f'{b}_alpha_ann']:+.2%} ({s[f'{b}_alpha_t']:.2f})",
                "Corr": f"{s[f'{b}_corr']:.2f}",
                "Info ratio": f"{s[f'{b}_info_ratio']:.2f}",
            }
        )
    return pd.DataFrame(rows)
