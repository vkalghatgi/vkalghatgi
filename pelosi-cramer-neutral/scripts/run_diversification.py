"""Diversification analysis + IS/OOS test of Sharpe-improvement candidates.

Run after `scripts/run_backtest.py`. Writes results/diversification.md and figures/tables.

    python scripts/run_diversification.py [--boot-n 2000]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pcn import diversification as D  # noqa: E402
from pcn import metrics as M  # noqa: E402
from pcn import report as R  # noqa: E402
from pcn.config import BENCHMARKS, DEFAULT_PARAMS, DEFAULT_SPLIT, DIVERSIFIERS, RESULTS_DIR  # noqa: E402
from pcn.data.prices import to_returns  # noqa: E402
from pcn.portfolio import run_backtest  # noqa: E402
from scripts.run_backtest import JOINT_END, JOINT_START, load_inputs  # noqa: E402

EPISODES = [
    ("Q4-2018 sell-off", "2018-09-20", "2018-12-24"),
    ("COVID crash", "2020-02-19", "2020-03-23"),
    ("2022 bear market", "2022-01-03", "2022-10-12"),
    ("Regional-bank stress", "2023-02-02", "2023-03-13"),
    ("Aug-2024 vol shock", "2024-07-16", "2024-08-05"),
]
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.1%}"


def main(boot_n: int) -> None:
    t0 = time.time()
    fig_dir, tab_dir = RESULTS_DIR / "figures", RESULTS_DIR / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    inp = load_inputs()
    md, rf, split = inp.md, inp.md.rf_daily, DEFAULT_SPLIT
    bench = to_returns(md.prices[list(BENCHMARKS)])
    div = to_returns(md.prices[list(DIVERSIFIERS)])

    bt = run_backtest(inp.tx, inp.signals, inp.md, DEFAULT_PARAMS, JOINT_START, JOINT_END)
    strat = bt["ret"].rename("strategy")
    idx = strat.index
    spy, agg = bench["SPY"].reindex(idx), div["AGG"].reindex(idx)
    b6040 = D.sixty_forty(spy, agg)
    assets = pd.concat([bench.reindex(idx), div.reindex(idx), b6040], axis=1)
    figs: dict[str, str] = {}

    # ---------------------------------------------------------------- 1. correlations
    corr = D.correlation_table(strat, assets)
    corr.to_csv(tab_dir / "div_correlations.csv")
    figs["rolling_corr"] = R.fig_rolling_corr(
        {n: D.rolling_correlation(strat, assets[n]) for n in ["SPY", "QQQ", "IWM", "AGG", "60/40"]},
        fig_dir / "div_rolling_corr.png")

    # ---------------------------------------------------------------- 2. blends
    bases = {"SPY": spy, "60/40": b6040, "QQQ": bench["QQQ"].reindex(idx)}
    blends_fund = {n: D.blend_table(strat, b, rf, spy, WEIGHTS, "fund") for n, b in bases.items()}
    blends_over = {n: D.blend_table(strat, b, rf, spy, [0.0, 0.25, 0.5, 0.75, 1.0], "overlay") for n, b in bases.items()}
    for n, t in blends_fund.items():
        t.to_csv(tab_dir / f"div_blend_fund_{n.replace('/', '')}.csv")
    for n, t in blends_over.items():
        t.to_csv(tab_dir / f"div_blend_overlay_{n.replace('/', '')}.csv")
    figs["frontier"] = R.fig_frontier(strat, bases, rf, np.arange(0, 1.0001, 0.1).round(2).tolist(), fig_dir / "div_frontier.png")

    isoos = {n: D.is_oos_blend(strat, b, rf, spy, split, "fund") for n, b in bases.items()}
    isoos_df = pd.DataFrame(isoos).T
    isoos_df.to_csv(tab_dir / "div_blend_is_oos.csv")
    boot = {n: D.bootstrap_sharpe_gain(strat, b, rf, 0.20, "fund", n=boot_n) for n, b in bases.items()}
    boot_df = pd.DataFrame(boot).T
    boot_df.to_csv(tab_dir / "div_blend_bootstrap.csv")

    # ---------------------------------------------------------------- 3. conditional / episodes
    cond = D.conditional_table(strat, spy, rf)
    cond.to_csv(tab_dir / "div_conditional.csv")
    figs["conditional"] = R.fig_conditional(cond, fig_dir / "div_conditional.png")
    epis = D.episode_table(strat, assets[["SPY", "QQQ", "IWM", "AGG", "60/40"]], EPISODES)
    epis.to_csv(tab_dir / "div_episodes.csv")

    # ---------------------------------------------------------------- 4. inside the book
    legs = D.leg_diversification(inp, DEFAULT_PARAMS, JOINT_START, JOINT_END, rf)
    sweep = D.short_notional_sweep(inp, DEFAULT_PARAMS, split, rf)
    sweep.to_csv(tab_dir / "div_short_notional_sweep.csv")

    # ---------------------------------------------------------------- 5. Sharpe-improvement candidates
    cands = D.evaluate_candidates(inp, DEFAULT_PARAMS, split, rf, bench)
    cands.to_csv(tab_dir / "div_candidates.csv")
    figs["cand_bars"] = R.fig_candidate_bars(cands, fig_dir / "div_candidates_is_oos.png")
    plot_names = ["Baseline (pre-specified)", "Vol-target 10 %", "Hedge with QQQ", "Two-factor hedge SPY+QQQ",
                  "No Cramer leg (hedged Pelosi only)", "Combo: vol-target 10 % + Cramer 63d + no sale-exit"]
    cand_rets = D.candidate_returns(inp, DEFAULT_PARAMS, JOINT_START, JOINT_END, plot_names)
    figs["cand_growth"] = R.fig_candidates(cand_rets, split.oos_start, fig_dir / "div_candidates_growth.png")

    # the best IS candidate (excluding the baseline), and how it did OOS
    non_base = cands.drop(index="Baseline (pre-specified)")
    best_is = non_base["is_sharpe"].idxmax()
    best_row = cands.loc[best_is]
    base_row = cands.loc["Baseline (pre-specified)"]
    survivors = non_base[(non_base["is_gain"] > 0) & (non_base["oos_gain"] > 0)]

    write_report(RESULTS_DIR / "diversification.md", figs, corr, blends_fund, blends_over, isoos_df, boot_df, cond, epis,
                 legs, sweep, cands, best_is, best_row, base_row, survivors, strat, spy, b6040, rf, split, bt)
    print(f"done in {time.time() - t0:.0f}s -> {RESULTS_DIR / 'diversification.md'}")


def write_report(path, figs, corr, blends_fund, blends_over, isoos, boot, cond, epis, legs, sweep, cands,
                 best_is, best_row, base_row, survivors, strat, spy, b6040, rf, split, bt) -> None:
    F = "figures/"
    s_strat, s_spy, s_6040 = M.sharpe(strat, rf), M.sharpe(spy, rf), M.sharpe(b6040, rf)
    b20 = blends_fund["60/40"].loc[0.2]
    b0 = blends_fund["60/40"].loc[0.0]
    s20 = blends_fund["SPY"].loc[0.2]
    s0 = blends_fund["SPY"].loc[0.0]
    ov = blends_over["60/40"].loc[0.5]
    down = cond.loc["SPY down months"]
    up = cond.loc["SPY up months"]

    L: list[str] = []
    w = L.append
    w("# Diversification value and Sharpe-improvement analysis\n")
    w(f"Companion to `report.md`. Same book (long Pelosi / short Cramer, beta-hedged with SPY, pre-specified "
      f"parameters, net of costs), same joint window {strat.index.min().date()} → {strat.index.max().date()}; "
      f"IS = {split.is_start[:4]}–{split.is_end[:4]}, OOS = {split.oos_start[:4]}–{split.oos_end[:4]}.\n")

    rc_spy = D.rolling_correlation(strat, spy)
    vt, qh, tf, combo = (cands.loc["Vol-target 10 %"], cands.loc["Hedge with QQQ"],
                         cands.loc["Two-factor hedge SPY+QQQ"], cands.loc[cands.index[-1]])
    nosale = cands.loc["Pelosi: don't exit on disclosed sales"]
    sw_is_best = float(sweep["is_sharpe"].idxmax())
    sw_oos_best = float(sweep["oos_sharpe"].idxmax())
    ov25, ov50 = blends_over["60/40"].loc[0.25], blends_over["60/40"].loc[0.5]

    w("## Key findings\n")
    w(f"* **The book is uncorrelated with everything conventional.** Daily correlation with SPY "
      f"{corr.loc['SPY', 'corr_daily']:+.2f}, with a 60/40 portfolio {corr.loc['60/40', 'corr_daily']:+.2f}, with "
      f"bonds (AGG) {corr.loc['AGG', 'corr_daily']:+.2f}, long Treasuries (TLT) {corr.loc['TLT', 'corr_daily']:+.2f}, "
      f"gold {corr.loc['GLD', 'corr_daily']:+.2f}. Monthly correlations are similar (SPY {corr.loc['SPY', 'corr_monthly']:+.2f}). "
      f"The rolling 1-year correlation with SPY stays within [{rc_spy.min():+.2f}, {rc_spy.max():+.2f}]. The only "
      f"non-trivial loadings are +{corr.loc['QQQ', 'corr_daily']:.2f} to QQQ and {corr.loc['IWM', 'corr_daily']:+.2f} to "
      f"IWM: the large-cap-growth residual.")
    w(f"* **Diversification benefit is real but small, because the book's own Sharpe is small.** Moving 20 % of a "
      f"60/40 portfolio into the book changes Sharpe {b0['sharpe']:.2f} → {b20['sharpe']:.2f}, max drawdown "
      f"{pct(b0['max_dd'])} → {pct(b20['max_dd'])}, CAGR {pct(b0['cagr'])} → {pct(b20['cagr'])}. For SPY: Sharpe "
      f"{s0['sharpe']:.2f} → {s20['sharpe']:.2f}, max drawdown {pct(s0['max_dd'])} → {pct(s20['max_dd'])}, CAGR "
      f"{pct(s0['cagr'])} → {pct(s20['cagr'])}. Run as an overlay (keep the 60/40, add 25–50 % of the book on top) the "
      f"Sharpe goes {b0['sharpe']:.2f} → {ov25['sharpe']:.2f}–{ov50['sharpe']:.2f} with CAGR {pct(b0['cagr'])} → "
      f"{pct(ov25['cagr'])}–{pct(ov50['cagr'])}. Block-bootstrap 95 % CI on the Sharpe gain from a 20 % funded "
      f"allocation out of 60/40: [{boot.loc['60/40', 'ci_low']:+.2f}, {boot.loc['60/40', 'ci_high']:+.2f}], "
      f"P(gain ≤ 0) = {boot.loc['60/40', 'p_gain_le_0']:.2f}. Zero correlation is necessary but not sufficient: a "
      f"Sharpe-{s_strat:.2f} diversifier can only lift a Sharpe-{s_6040:.2f} portfolio by a few hundredths.")
    w(f"* **The IS-chosen allocation survives OOS, weakly.** The weight that maximised the 60/40 blend's IS Sharpe "
      f"({isoos.loc['60/40', 'weight_chosen_is']:.0%} in the book), held through 2022–24, gives Sharpe "
      f"{isoos.loc['60/40', 'oos_sharpe_base']:.2f} → {isoos.loc['60/40', 'oos_sharpe_blend']:.2f} and max drawdown "
      f"{pct(isoos.loc['60/40', 'oos_maxdd_base'])} → {pct(isoos.loc['60/40', 'oos_maxdd_blend'])}, in a period in "
      f"which the book itself had Sharpe {M.sharpe(strat.loc[split.oos_start:], rf):.2f}. For SPY the IS weight is "
      f"{isoos.loc['SPY', 'weight_chosen_is']:.0%} and OOS Sharpe goes {isoos.loc['SPY', 'oos_sharpe_base']:.2f} → "
      f"{isoos.loc['SPY', 'oos_sharpe_blend']:.2f}.")
    w(f"* **Mildly counter-cyclical in fast crashes, not in a slow growth bear.** In SPY down months the book averaged "
      f"{pct(down['strat_avg'])} per month (hit rate {pct(down['strat_hit'])}) against {pct(up['strat_avg'])} in up "
      f"months, and it lost money on average in the best SPY quintile ({pct(cond.iloc[4]['strat_avg'])}). It made "
      f"{pct(epis.iloc[1]['strategy'])} through the COVID crash (Cramer's names fell harder than Pelosi's), "
      f"{pct(epis.iloc[3]['strategy'])} through the regional-bank stress and {pct(epis.iloc[0]['strategy'])} in Q4-2018 — "
      f"but {pct(epis.iloc[2]['strategy'])} across the 2022 bear market, because its residual exposure is long mega-cap "
      f"growth and that is what de-rated. It diversifies; it is not a tail hedge.")
    w(f"* **Inside the book, the two legs diversify each other** (daily correlation of the two hedged legs "
      f"{legs['corr_legs_daily']:+.2f}), so the combination (Sharpe {legs['sharpe_combined']:.2f}) beats either leg "
      f"alone (Pelosi {legs['sharpe_long']:.2f}, Cramer {legs['sharpe_short']:.2f}) even though the Cramer leg is "
      f"near break-even after costs. The short-notional sweep is a textbook overfitting warning: in-sample the best "
      f"ratio is {sw_is_best:.2f}× short per unit long (i.e. drop the Cramer leg, IS Sharpe "
      f"{sweep.loc[sw_is_best, 'is_sharpe']:.2f}) — and that choice has the *worst* OOS Sharpe of the sweep "
      f"({sweep.loc[sw_is_best, 'oos_sharpe']:.2f}). OOS Sharpe rises monotonically with the short weight (best at "
      f"{sw_oos_best:.2f}×, {sweep.loc[sw_oos_best, 'oos_sharpe']:.2f}), because the Cramer leg is what carried 2022. "
      f"The pre-specified 1:1 is a sensible middle.")
    w(f"* **Sharpe improvers, tested IS → OOS.** {len(survivors)} of {len(cands) - 1} pre-declared changes beat the "
      f"baseline (IS {base_row['is_sharpe']:.2f} / OOS {base_row['oos_sharpe']:.2f}) in *both* halves. They fall into "
      f"three groups. (i) **Volatility targeting** (10 %: IS {vt['is_sharpe']:.2f}, OOS {vt['oos_sharpe']:.2f}, max "
      f"drawdown {pct(vt['full_max_dd'])} vs {pct(base_row['full_max_dd'])}) — the change that requires no view about "
      f"the signals, is standard for a neutral book, and works by de-levering in 2020 and 2022 (average leverage "
      f"{vt['avg_leverage']:.2f}×). (ii) **A slower Cramer leg** (63d/126d holds: OOS "
      f"{cands.loc['Cramer hold 63d (lower turnover)', 'oos_sharpe']:.2f}/{cands.loc['Cramer hold 126d', 'oos_sharpe']:.2f}), "
      f"which halves turnover; the IS grid in `report.md` had already pointed here. (iii) **More Pelosi concentration** "
      f"(no exit on disclosed sales OOS {nosale['oos_sharpe']:.2f}, no "
      f"single-name cap OOS {cands.loc['Pelosi: no single-name cap', 'oos_sharpe']:.2f}) — real improvements in this "
      f"sample, but each is a larger bet on NVDA/AAPL/MSFT in 2023–24 rather than new evidence of information. The "
      f"pre-declared combo of (i)+(ii)+no-sale-exit reaches IS {combo['is_sharpe']:.2f}, OOS {combo['oos_sharpe']:.2f}, "
      f"full {combo['full_sharpe']:.2f}, max drawdown {pct(combo['full_max_dd'])}.")
    w(f"* **The growth tilt *is* the return.** A two-factor SPY+QQQ hedge that drives realised QQQ beta to "
      f"{tf['beta_qqq']:+.2f} leaves a Sharpe of {tf['full_sharpe']:.2f} (IS {tf['is_sharpe']:.2f}, OOS "
      f"{tf['oos_sharpe']:.2f}); hedging size as well (SPY+IWM) gives {cands.loc['Two-factor hedge SPY+IWM', 'full_sharpe']:.2f}. "
      f"Hedging with QQQ *instead of* SPY does the opposite — Sharpe {qh['full_sharpe']:.2f}, better in both halves and in "
      f"every calendar year — but it does not remove the QQQ beta ({qh['beta_qqq']:+.2f}, same as baseline). It works "
      f"because the overlay is on average net *long* the index (mean hedge {bt['hedge_w'].mean():+.2f} of NAV: the "
      f"Pelosi leg averages only {bt['gross_long'].mean():.2f} gross while the short leg is {bt['gross_short'].mean():.2f} "
      f"with a higher estimated beta), so replacing SPY by QQQ adds "
      f"a long QQQ-minus-SPY position, and in 2022 the overlay happened to be short when QQQ fell more. That is a "
      f"growth bet layered on a growth bet, not a better hedge. Once mega-cap-growth exposure is neutralised there is "
      f"little idiosyncratic Pelosi/Cramer alpha left to diversify anything with.")
    w("")

    # ---------------------------------------------------------------- 1
    w("## 1. Correlation with conventional assets\n")
    w("Daily and monthly return correlations over the joint window. `60/40` = daily-rebalanced 60 % SPY / 40 % AGG.\n")
    ct = corr.copy()
    ct.index = [f"{i} ({(BENCHMARKS | DIVERSIFIERS).get(i, 'SPY/AGG blend')})" for i in ct.index]
    w(R.md_table(ct.rename_axis("asset").reset_index()))
    w(f"\n![]({F}{figs['rolling_corr']})\n")
    w("The rolling correlation with SPY oscillates around zero with no trend; the ex-ante hedge does what it is "
      "meant to. Correlation with QQQ is mildly positive and with IWM mildly negative, the signature of the "
      "large-cap-growth residual documented in `report.md`.\n")

    # ---------------------------------------------------------------- 2
    w("## 2. Blending the book into a conventional portfolio\n")
    w("### 2a. Funded allocation (sell the base portfolio to buy the book)\n")
    for n, t in blends_fund.items():
        w(f"**Base = {n}**\n")
        tt = t.copy()
        for c in ["cagr", "vol", "max_dd", "worst_month"]:
            tt[c] = tt[c].map(pct)
        tt.index = [f"{i:.0%}" for i in tt.index]
        w(R.md_table(tt.rename_axis("weight in book").reset_index(), "{:.2f}"))
        w("")
    w(f"![]({F}{figs['frontier']})\n")
    w("### 2b. Overlay (keep 100 % of the base, add the book on top, financed at T-bills)\n")
    w("Because the book is beta-neutral and holds its own cash, it can be run as an overlay; this is how a "
      "beta-neutral sleeve is actually used in a multi-strategy fund.\n")
    for n, t in blends_over.items():
        w(f"**Base = {n}**\n")
        tt = t.copy()
        for c in ["cagr", "vol", "max_dd", "worst_month"]:
            tt[c] = tt[c].map(pct)
        tt.index = [f"+{i:.0%}" for i in tt.index]
        w(R.md_table(tt.rename_axis("book overlay").reset_index(), "{:.2f}"))
        w("")
    w(f"A 50 % overlay on 60/40 takes Sharpe {blends_over['60/40'].loc[0.0, 'sharpe']:.2f} → {ov['sharpe']:.2f} and "
      f"CAGR {pct(blends_over['60/40'].loc[0.0, 'cagr'])} → {pct(ov['cagr'])} for vol "
      f"{pct(blends_over['60/40'].loc[0.0, 'vol'])} → {pct(ov['vol'])}.\n")

    w("### 2c. Is the improvement real? IS-chosen weight applied OOS, and a bootstrap\n")
    tt = isoos.copy()
    tt["weight_chosen_is"] = tt["weight_chosen_is"].map(pct)
    for c in [c for c in tt.columns if "maxdd" in c or "cagr" in c]:
        tt[c] = tt[c].map(pct)
    w(R.md_table(tt.rename_axis("base").reset_index(), "{:.2f}"))
    w("\nStationary block bootstrap (21-day blocks, joint resampling) of Sharpe(80 % base + 20 % book) − Sharpe(base):\n")
    w(R.md_table(boot.rename_axis("base").reset_index(), "{:.2f}"))
    w("\nThe point estimates of the gain are positive for every base but none of the intervals excludes zero; "
      "with a Sharpe-0.4 diversifier over seven years that is the expected outcome, not evidence against it.\n")
    w(f"Rolling 1-year correlation with SPY ranges {rc_spy.min():+.2f} to {rc_spy.max():+.2f}; the diversification "
      f"is not an artefact of one sub-period.\n")

    # ---------------------------------------------------------------- 3
    w("## 3. When does the book make money?\n")
    ct = cond.copy()
    ct["n_months"] = ct["n_months"].astype(int)
    for c in ["spy_avg", "strat_avg", "strat_hit"]:
        ct[c] = ct[c].map(pct)
    w(R.md_table(ct.rename_axis("SPY month bucket").reset_index()))
    w(f"\n![]({F}{figs['conditional']})\n")
    w("Cumulative returns through named stress windows:\n")
    et = epis.copy()
    for c in et.columns:
        et[c] = et[c].map(pct)
    w(R.md_table(et.rename_axis("episode").reset_index()))
    w("\nThe pattern is that of a market-neutral-but-not-style-neutral book: slightly better when the market falls "
      "than when it rallies (the short leg's high-beta names fall harder in fast crashes), but exposed to "
      "growth-vs-value rotations, which is what 2022 was.\n")

    # ---------------------------------------------------------------- 4
    w("## 4. Diversification inside the book\n")
    w(R.md_table(pd.DataFrame([legs]).T.rename(columns={0: "value"}).rename_axis("statistic").reset_index()))
    w(f"\nWith leg correlation {legs['corr_legs_daily']:+.2f} the textbook Sharpe of an equal-risk combination is "
      f"{legs['sharpe_equal_risk_theory']:.2f}; the realised {legs['sharpe_combined']:.2f} is in line. The Cramer "
      f"leg earns roughly nothing on its own but adds an uncorrelated return stream, which is precisely the case "
      f"where a weak signal is still worth holding.\n")
    w("Short notional per unit of long notional (pre-specified value is 1.0):\n")
    st = sweep.copy()
    st["full_vol"] = st["full_vol"].map(pct)
    st["full_max_dd"] = st["full_max_dd"].map(pct)
    st.index = [f"{i:.2f}" for i in st.index]
    w(R.md_table(st.rename_axis("short_notional").reset_index(), "{:.2f}"))
    w(f"\nIS and OOS disagree completely on this dial: IS Sharpe falls monotonically as the short weight rises "
      f"(2018–21 was a period in which shorting anything was expensive), OOS Sharpe rises monotonically with it "
      f"(2022 was the Cramer leg's best year). Anyone who had optimised the leg mix in-sample would have dropped "
      f"the short leg and then taken the worst OOS outcome available. The pre-specified 1:1 was not chosen with "
      f"either half in view and sits in the middle of both rankings.\n")

    # ---------------------------------------------------------------- 5
    w("## 5. Sharpe-improvement candidates, in-sample → out-of-sample\n")
    w("Every candidate below was declared before its OOS number was computed (see `pcn/diversification.py`). "
      "Treat the OOS column as the only honest one; the `full` column mixes the two halves. A change that helps IS "
      "and hurts OOS is noise.\n")
    ct = cands.copy()
    for c in ["full_cagr", "full_vol", "full_max_dd"]:
        ct[c] = ct[c].map(pct)
    ct = ct[["rationale", "is_sharpe", "oos_sharpe", "full_sharpe", "is_gain", "oos_gain", "full_cagr", "full_vol",
             "full_max_dd", "beta_spy", "beta_qqq", "alpha_t_spy", "turnover_ann", "avg_leverage"]]
    w(R.md_table(ct.rename_axis("candidate").reset_index(), "{:.2f}"))
    w(f"\n![]({F}{figs['cand_bars']})\n")
    w(f"![]({F}{figs['cand_growth']})\n")
    w("### Reading the table\n")
    w(f"* **Volatility targeting** is the cleanest improvement: it needs no view on the signals or on which index to "
      f"hedge with, it is standard practice for a leveraged neutral book, and it improves both halves "
      f"(IS {vt['is_sharpe']:.2f}, OOS {vt['oos_sharpe']:.2f} vs {base_row['is_sharpe']:.2f}/{base_row['oos_sharpe']:.2f}) "
      f"while cutting the max drawdown from {pct(base_row['full_max_dd'])} to {pct(vt['full_max_dd'])}. Average leverage "
      f"{vt['avg_leverage']:.2f}× — it is mostly *de*-levering in 2020 and 2022, when the book's residual growth "
      f"exposure was most volatile. The 8 % and 10 % targets have identical Sharpe because the 2× leverage cap never "
      f"binds; only the scale differs. The scaling is applied to the net return series (positions, hedge and costs "
      f"scale together; excess cash earns T-bills), which ignores the small extra cost of changing leverage.")
    w(f"* **Hedge-instrument changes.** Hedging with QQQ instead of SPY has the highest single-change improvement in "
      f"both halves (IS {qh['is_sharpe']:.2f}, OOS {qh['oos_sharpe']:.2f}), yet its realised QQQ beta "
      f"({qh['beta_qqq']:+.2f}) is the same as the baseline's, because single-name betas to QQQ are about the same as "
      f"to SPY and so the overlay is the same size. What changes is *which* index the overlay holds. The overlay is "
      f"net long on average ({bt['hedge_w'].mean():+.2f} of NAV — the Pelosi leg is frequently under-populated), so "
      f"the switch adds a long QQQ-minus-SPY position that paid in 2018–21 and 2023–24, and in 2022 the overlay was "
      f"short at the moment QQQ fell hardest. Both are growth bets, not hedge refinements; a QQQ hedge is "
      f"a-priori defensible for the Pelosi leg (mega-cap tech) but not for the Cramer leg (broad universe). The "
      f"two-factor hedges, which really do neutralise growth (SPY+QQQ, QQQ beta {tf['beta_qqq']:+.2f}) or size "
      f"(SPY+IWM), *remove* most of the return and add turnover. This is the clearest evidence in the study about "
      f"what the book actually is.")
    w(f"* **Turnover reduction on the Cramer leg** (63d / 126d holds) helps in both halves and cuts annual turnover "
      f"from {base_row['turnover_ann']:.0f}× to {cands.loc['Cramer hold 63d (lower turnover)', 'turnover_ann']:.0f}× / "
      f"{cands.loc['Cramer hold 126d', 'turnover_ann']:.0f}×. Dropping or halving the Cramer leg looks excellent IS "
      f"(0.66 / 0.63) and fails OOS ({cands.loc['No Cramer leg (hedged Pelosi only)', 'oos_sharpe']:.2f} / "
      f"{cands.loc['Half-size Cramer leg', 'oos_sharpe']:.2f}): after costs the leg adds little return but it is the "
      f"leg that made money in 2022 (+21 % while the Pelosi leg lost 51 %; Section 4).")
    w(f"* **Pelosi-leg construction changes.** Not exiting on disclosed sales (OOS "
      f"{nosale['oos_sharpe']:.2f}) and removing the single-name cap (OOS "
      f"{cands.loc['Pelosi: no single-name cap', 'oos_sharpe']:.2f}) both survive; a longer hold "
      f"({cands.loc['Pelosi: hold 504d', 'oos_sharpe']:.2f}) does not. The survivors work by keeping more NVDA/AAPL/MSFT "
      f"through 2023 (her disclosed NVDA sales filed in July and October 2022 closed the position under the baseline rule, "
      f"months before the 2023 rally). They raise "
      f"the Sharpe in this sample; they are a bigger bet on the same three names, not new evidence of information.")
    w(f"* **Combination.** The pre-declared combo (vol-target 10 % + Cramer 63d + no sale-exit) gives IS "
      f"{combo['is_sharpe']:.2f}, OOS {combo['oos_sharpe']:.2f}, full {combo['full_sharpe']:.2f}, max drawdown "
      f"{pct(combo['full_max_dd'])}, at {pct(combo['full_vol'])} vol. Note that a Sharpe of 0.8 over seven years still "
      f"has a standard error of about 0.4, and that the combo was declared knowing the *IS* grid results. It is a fair "
      f"estimate of what a carefully built version of this idea would have delivered, not a forecast.\n")

    w("## 6. What would actually raise the Sharpe\n")
    w("The analysis above bounds what parameter changes can do. The structural options are:\n")
    w("1. **Volatility-target the book and slow the short leg.** Both are a-priori choices, both improve IS and OOS, "
      "and together they take the max drawdown from roughly −30 % to −20 % without changing what the book bets on.")
    w(f"2. **Run it as an overlay on a diversified portfolio, not as a stand-alone fund.** Section 2b shows the book is "
      f"worth about +{ov25['sharpe'] - b0['sharpe']:.2f} to +{ov50['sharpe'] - b0['sharpe']:.2f} of Sharpe to a 60/40 "
      f"investor at a 25–50 % overlay. That is its realistic use — a small uncorrelated sleeve.")
    w("3. **Model the options as options.** The Pelosi return lives in disclosed deep-in-the-money LEAP calls. Treating "
      "them as delta-one equity understates both return and risk; a delta-adjusted replication would raise CAGR but "
      "not obviously Sharpe, and requires historical option data that is not in this study.")
    w("4. **Do not neutralise the growth tilt** unless you have another source of return; Section 5 shows the tilt is "
      "most of the return. Equivalently, an investor already long mega-cap growth gets *less* diversification from "
      "this book than the correlation table suggests.")
    w("5. **Accept that the confidence intervals will not shrink.** Seven years of a Sharpe-0.4 process gives a "
      "standard error of about 0.38 on the Sharpe. No amount of re-parameterisation changes that; only more data "
      "(or a genuinely different signal) does.\n")

    path.write_text("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot-n", type=int, default=2000)
    a = ap.parse_args()
    main(a.boot_n)
