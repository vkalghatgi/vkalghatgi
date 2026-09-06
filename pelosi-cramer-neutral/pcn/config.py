from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = ROOT / "results"

BENCHMARKS = {
    "SPY": "S&P 500",
    "IWM": "Russell 2000",
    "QQQ": "Nasdaq 100",
    "DIA": "Dow Jones Industrial",
    "RSP": "S&P 500 Equal Weight",
    "VTI": "US Total Market",
}
# Non-equity assets used only in the diversification analysis
DIVERSIFIERS = {
    "AGG": "US Aggregate Bonds",
    "TLT": "20y+ Treasuries",
    "GLD": "Gold",
}
RISK_FREE_TICKER = "^IRX"  # 13-week T-bill discount yield (annualised, %)
HEDGE_TICKER = "SPY"


@dataclass(frozen=True)
class StrategyParams:
    """Everything that can be tuned. Anything tuned must be tuned in-sample only."""

    # --- Pelosi (long) leg -------------------------------------------------
    pelosi_hold_days: int = 252          # max calendar holding (trading days) after entry
    pelosi_close_on_sale: bool = True    # exit when a disclosed sale of the same ticker is filed
    pelosi_include_options: bool = True  # treat disclosed call purchases/exercises as long-equity signals
    pelosi_weighting: str = "equal"      # "equal" | "amount" (midpoint of disclosed range)
    pelosi_max_weight: float = 0.20      # cap on any single name within the leg
    pelosi_extra_lag_days: int = 0       # additional trading-day delay after the filing date

    # --- Cramer (short) leg ------------------------------------------------
    cramer_hold_days: int = 21           # trading days each short is held
    cramer_signal_types: tuple[str, ...] = ("start_long",)  # which transcript signal types to fade
    cramer_min_confidence: float = 0.0
    cramer_max_positions: int = 150      # cap on simultaneously-open shorts (equal-weighted)
    cramer_extra_lag_days: int = 0

    # --- Portfolio ---------------------------------------------------------
    long_notional: float = 1.0           # fraction of NAV in the long leg when populated
    short_notional: float = 1.0          # fraction of NAV in the short leg when populated
    beta_window: int = 126               # trailing window (trading days) for beta estimates
    beta_method: str = "holdings"        # "holdings": sum(w_i * stock beta_i) | "leg": beta of the leg's return series
    beta_target: float = 0.0
    beta_hedge: bool = True              # hedge residual beta with the hedge instrument(s)
    hedge_tickers: tuple[str, ...] = ("SPY",)  # one instrument = market beta; several = multi-factor hedge
    vol_target: float | None = None      # if set, scale the whole book to this annualised vol (trailing 63d estimate)
    max_leverage: float = 2.0            # cap on the vol-target scaling factor
    max_hedge: float = 1.5               # |overlay| cannot exceed 150% of NAV (safety rail against bad data)
    beta_tolerance: float = 0.10         # reporting threshold |beta| <= tol

    # --- Frictions ---------------------------------------------------------
    cost_bps: float = 10.0               # one-way transaction cost per unit of turnover
    borrow_bps_annual: float = 100.0     # stock-borrow fee on short notional
    earn_cash_rf: bool = True            # collateral earns T-bill rate

    def with_(self, **kw) -> "StrategyParams":
        return replace(self, **kw)


@dataclass
class SampleSplit:
    """In-sample / out-of-sample definition (dates are inclusive)."""

    is_start: str = "2018-01-01"
    is_end: str = "2021-12-31"
    oos_start: str = "2022-01-01"
    oos_end: str = "2024-12-31"


DEFAULT_PARAMS = StrategyParams()
DEFAULT_SPLIT = SampleSplit()

# Parameter grid searched *in-sample only*
PARAM_GRID: dict[str, list] = {
    "pelosi_hold_days": [63, 126, 252, 504],
    "cramer_hold_days": [5, 21, 63, 126],
    "beta_window": [63, 126, 252],
}
