"""Performance metrics. All annualised, all taking SIMPLE returns."""

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


def wealth_curve(returns) -> np.ndarray:
    """Cumulative wealth starting at 1.0 so a day-1 loss is counted."""
    r = np.asarray(returns, float)
    return np.concatenate([[1.0], np.cumprod(1.0 + r)])


def max_drawdown(returns) -> float:
    w = wealth_curve(returns)
    peak = np.maximum.accumulate(w)
    return float(((w - peak) / peak).min())


def downside_deviation(excess, freq=TRADING_DAYS) -> float:
    """
    sqrt( mean( min(r - MAR, 0)^2 ) ), averaged over ALL periods.

    NOT np.std(r[r < 0]). Taking the std of the negative subset de-means those
    observations and understates downside risk -- it is the single most common
    Sortino implementation bug.
    """
    d = np.minimum(np.asarray(excess, float), 0.0)
    return float(np.sqrt(np.mean(d ** 2)) * np.sqrt(freq))


def perf_stats(returns, rf_annual=0.065, freq=TRADING_DAYS) -> dict:
    r = np.asarray(returns, float)
    r = r[~np.isnan(r)]
    keys = ["Ann. Return", "Ann. Vol", "Sharpe", "Sortino", "Max DD",
            "Calmar", "Hit Rate", "Obs"]
    if len(r) < 2:
        return dict.fromkeys(keys, np.nan)

    rf_period = (1 + rf_annual) ** (1 / freq) - 1
    excess = r - rf_period

    total = float(np.prod(1 + r))
    ann_ret = total ** (freq / len(r)) - 1 if total > 0 else -1.0
    sd = r.std(ddof=1)
    ann_vol = sd * np.sqrt(freq)

    sharpe = excess.mean() / sd * np.sqrt(freq) if sd > 0 else np.nan
    dd = downside_deviation(excess, freq)
    sortino = excess.mean() * freq / dd if dd > 0 else np.nan
    mdd = max_drawdown(r)

    return {
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max DD": mdd,
        "Calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,
        "Hit Rate": float((r > 0).mean()),
        "Obs": len(r),
    }


def stats_table(returns_df: pd.DataFrame, rf_annual=0.065,
                freq=TRADING_DAYS) -> pd.DataFrame:
    return pd.DataFrame(
        {c: perf_stats(returns_df[c], rf_annual, freq) for c in returns_df.columns}
    ).T


def drawdown_series(returns, index=None) -> pd.Series:
    w = wealth_curve(returns)[1:]
    peak = np.maximum.accumulate(w)
    dd = (w - peak) / peak
    return pd.Series(dd, index=index if index is not None else range(len(dd)))


def turnover(w_new, w_old) -> float:
    """Sum of absolute weight changes -- the cost driver."""
    return float(np.abs(np.asarray(w_new) - np.asarray(w_old)).sum())
