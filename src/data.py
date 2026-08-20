"""Data acquisition: download once, cache to CSV, and guarantee column order."""

from pathlib import Path
import numpy as np
import pandas as pd

from .config import DATA_RAW, TRADING_DAYS


def _cache_path(start: str, end: str) -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    return DATA_RAW / f"prices_{start}_{end}.csv"


def download_prices(tickers, oil_ticker, start, end, force=False) -> pd.DataFrame:
    """
    Download adjusted closes for stocks + oil, cache to disk.

    CRITICAL: yfinance returns columns sorted ALPHABETICALLY, not in the
    order you passed them. Every covariance matrix, pick matrix and weight
    vector downstream assumes `tickers` order. Reindexing here is the single
    most important line in this file.
    """
    path = _cache_path(start, end)
    cols = list(tickers) + [oil_ticker]

    if path.exists() and not force:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        raw = yf.download(cols, start=start, end=end,
                          auto_adjust=True, progress=False)
        df = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        df.to_csv(path)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Tickers missing from download: {missing}")

    df = df[cols]                       # <-- enforce order
    df = df.ffill().dropna(how="any")
    if df.empty:
        raise ValueError("No overlapping data after alignment.")
    return df


def split_prices(df: pd.DataFrame, tickers, oil_ticker):
    """Return (stock_prices, oil_prices) with identical, ordered indices."""
    return df[list(tickers)].copy(), df[oil_ticker].copy()


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Simple returns, NOT log returns.

    Portfolio return r_p = w'r only holds for simple returns. Log returns are
    additive across TIME, not across ASSETS. The original notebook used
    log returns and then compounded them with (1+r).cumprod(), which is
    wrong twice over.
    """
    return prices.pct_change().dropna()


def market_cap_weights(tickers, fallback_equal=True) -> np.ndarray:
    """
    Market-cap weights normalised over the 10-stock universe.

    Caveat to state in your report: yfinance reports CURRENT market cap, so
    using it as the prior for a backtest starting years earlier is a mild
    look-ahead. Acceptable for a static universe of large caps, but say so.
    """
    import yfinance as yf
    caps = {}
    for t in tickers:
        try:
            caps[t] = yf.Ticker(t).info.get("marketCap", np.nan)
        except Exception:
            caps[t] = np.nan

    s = pd.Series(caps).reindex(tickers).astype(float)
    if s.isna().all():
        if not fallback_equal:
            raise RuntimeError("No market caps retrieved.")
        print("WARNING: no market caps retrieved -> falling back to equal weights")
        return np.ones(len(tickers)) / len(tickers)
    if s.isna().any():
        print(f"WARNING: market cap missing for {list(s[s.isna()].index)} "
              f"-> filled with universe mean")
        s = s.fillna(s.mean())
    return (s / s.sum()).values


def save_market_caps(w, tickers, path=None):
    path = path or (DATA_RAW / "market_caps.csv")
    pd.Series(w, index=tickers, name="weight").to_csv(path)


def load_cached_market_caps(tickers, path=None):
    path = path or (DATA_RAW / "market_caps.csv")
    if Path(path).exists():
        s = pd.read_csv(path, index_col=0)["weight"].reindex(tickers)
        if not s.isna().any():
            return s.values
    return None


# ----------------------------------------------------------------------
# Synthetic generator: lets tests and demos run with no network access.
# ----------------------------------------------------------------------
def synthetic_data(tickers, n_days=1500, seed=0, betas=None):
    """Factor model: r_i = beta_i * r_oil + idiosyncratic + drift."""
    rng = np.random.default_rng(seed)
    n = len(tickers)
    if betas is None:
        betas = np.linspace(0.9, -0.35, n)

    oil = rng.normal(0.0, 0.021, n_days)
    idio = rng.normal(0.0, 0.013, (n_days, n))
    drift = rng.uniform(0.0002, 0.0006, n)
    r = betas * oil[:, None] + idio + drift

    idx = pd.bdate_range("2018-01-01", periods=n_days)
    returns = pd.DataFrame(r, index=idx, columns=list(tickers))
    oil_ret = pd.Series(oil, index=idx, name="oil")
    return returns, oil_ret, pd.Series(betas, index=list(tickers))
