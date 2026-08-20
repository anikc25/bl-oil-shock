"""Central configuration. Every tunable lives here, nothing is hardcoded downstream."""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

TRADING_DAYS = 252

# 10 NIFTY50 names chosen for a spread of oil sensitivity:
# upstream/energy & metals (oil-levered) vs FMCG/paints (oil is an input cost).
TICKERS = [
    "ONGC.NS",        # upstream oil & gas   -> expected positive oil beta
    "HINDALCO.NS",    # aluminium, energy-intensive
    "JSWSTEEL.NS",    # steel
    "VEDL.NS",        # diversified metals & oil
    "SUNPHARMA.NS",   # pharma, low sensitivity
    "SHREECEM.NS",    # cement, petcoke/freight cost exposure
    "NESTLEIND.NS",   # FMCG
    "BRITANNIA.NS",   # FMCG
    "HINDUNILVR.NS",  # FMCG, crude-linked input costs
    "ASIANPAINT.NS",  # paints, crude derivatives are a direct raw material
]

OIL_TICKER = "BZ=F"      # Brent crude front-month future
BENCHMARK = "^NSEI"      # NIFTY 50 index, used for reporting only


@dataclass
class Config:
    tickers: list = field(default_factory=lambda: list(TICKERS))
    oil_ticker: str = OIL_TICKER
    start: str = "2018-01-01"
    end: str = "2026-08-01"

    # --- Black-Litterman parameters ---
    risk_aversion: float = 2.5      # A (delta). Idzorek's standard value.
    tau: float = 0.05               # prior scaling
    view_confidence: float = 0.25    # c in Omega = diag(P tau Sigma P')/c
    oil_shock_annual: float = -0.20 # scenario: Brent falls 20% over a year

    # --- backtest ---
    train_window: int = 250         # ~1 year of trailing data for estimation
    rebalance_freq: int = 21        # rebalance monthly
    gross_exposure: float = 1.0     # match leverage across all strategies
    cost_bps: float = 10.0          # 10bps per unit of turnover
    rf_annual: float = 0.065        # ~India 10Y, for Sharpe/Sortino

    use_posterior_cov: bool = False # w = Sigma^-1 mu vs (Sigma+V)^-1 mu

    @property
    def n_assets(self) -> int:
        return len(self.tickers)


DEFAULT = Config()
