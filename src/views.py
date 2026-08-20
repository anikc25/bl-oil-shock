"""
Oil-shock views.

This module is what turns the project from "generic Black-Litterman" into
"Black-Litterman under oil price shocks". Views are DERIVED from estimated
oil betas and one explicit scenario input, never hardcoded. That means you
can answer "where did Q come from?" with a regression instead of a shrug.
"""

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


def estimate_oil_betas(R: pd.DataFrame, oil_returns: pd.Series) -> pd.DataFrame:
    """
    Univariate OLS of each stock's returns on oil returns.

        r_i,t = alpha_i + beta_i * r_oil,t + eps_i,t

    Returns a frame with beta, t-stat and R^2. Closed-form OLS, no loops
    over observations, so it is fast enough for a rolling backtest.
    """
    oil = oil_returns.reindex(R.index).astype(float)
    x = oil.values - oil.values.mean()
    sxx = float(x @ x)
    n = len(x)
    if sxx <= 0 or n < 10:
        raise ValueError("Degenerate oil series in this window.")

    out = {}
    for col in R.columns:
        y = R[col].values.astype(float)
        yc = y - y.mean()
        beta = float(x @ yc) / sxx
        resid = yc - beta * x
        sse = float(resid @ resid)
        sst = float(yc @ yc)
        se = np.sqrt(sse / (n - 2) / sxx) if n > 2 else np.nan
        out[col] = {
            "beta": beta,
            "t_stat": beta / se if se and se > 0 else np.nan,
            "r_squared": 1 - sse / sst if sst > 0 else np.nan,
        }
    return pd.DataFrame(out).T.reindex(R.columns)


def build_views(betas: pd.Series, tickers, oil_shock_annual: float,
                basket_size: int = 3, prior_returns=None):
    """
    Three views, all functions of (beta, shock). All returns are ANNUAL,
    matching the annualised Sigma used elsewhere.

    CRITICAL: Q in Black-Litterman is a TOTAL expected return, not an
    incremental one. `beta * shock` is only the shock's marginal impact. Used
    raw as Q, it tells the model "this stock will return 0.9% in total" when
    equilibrium says 3.8% -- a bearish view smuggled in by units, not economics.
    That bias is systematic: every asset's posterior falls below its prior,
    the portfolio de-risks, and the model looks broken when it is merely
    mis-specified.

    So when `prior_returns` (Pi) is supplied we anchor:

      View 1 (relative): (Pi_hi - Pi_lo) + (beta_hi - beta_lo) * shock
      View 2 (absolute):  Pi_top + beta_top * shock
      View 3 (absolute):  Pi_bot + beta_bot * shock

    This has a clean null property: shock = 0 gives Q = Pi exactly, so the
    posterior collapses to the prior and BL reproduces market-cap weights.
    Without anchoring, shock = 0 instead asserts "every asset returns exactly
    zero", which is a strong bearish view rather than no view at all.

    P rows are NORMALISED so each leg sums to 1. With raw +1/-1 entries a
    5-vs-5 view means "the SUM of five stocks minus the SUM of five others",
    which makes Q five times larger than the spread you actually have in mind.
    """
    tickers = list(tickers)
    n = len(tickers)
    b = betas.reindex(tickers).astype(float)
    if b.isna().any():
        raise ValueError("Betas missing for some tickers.")

    if prior_returns is None:
        pi = pd.Series(0.0, index=tickers)
    else:
        pi = pd.Series(np.asarray(prior_returns, float).reshape(-1), index=tickers)

    ranked = b.sort_values(ascending=False)
    k = min(basket_size, n // 2)
    high, low = list(ranked.index[:k]), list(ranked.index[-k:])
    pos = {t: i for i, t in enumerate(tickers)}

    p1 = np.zeros(n)
    for t in high:
        p1[pos[t]] = 1.0 / k
    for t in low:
        p1[pos[t]] = -1.0 / k
    shock_spread = (b[high].mean() - b[low].mean()) * oil_shock_annual
    q1 = (pi[high].mean() - pi[low].mean()) + shock_spread

    top, bot = ranked.index[0], ranked.index[-1]
    p2 = np.zeros(n); p2[pos[top]] = 1.0
    p3 = np.zeros(n); p3[pos[bot]] = 1.0

    P = np.vstack([p1, p2, p3])
    Q = np.array([
        q1,
        pi[top] + b[top] * oil_shock_annual,
        pi[bot] + b[bot] * oil_shock_annual,
    ])

    labels = [
        f"[{'/'.join(t.replace('.NS','') for t in high)}] vs "
        f"[{'/'.join(t.replace('.NS','') for t in low)}]: spread {q1:+.2%} "
        f"(shock contributes {shock_spread:+.2%})",
        f"{top.replace('.NS','')} returns {Q[1]:+.2%} "
        f"(equilibrium {pi[top]:+.2%}, beta {b[top]:+.2f})",
        f"{bot.replace('.NS','')} returns {Q[2]:+.2%} "
        f"(equilibrium {pi[bot]:+.2%}, beta {b[bot]:+.2f})",
    ]
    return P, Q, labels


def describe_views(P, Q, tickers) -> pd.DataFrame:
    """Human-readable table of the pick matrix, for the report."""
    df = pd.DataFrame(P, columns=[t.replace(".NS", "") for t in tickers],
                      index=[f"View {i+1}" for i in range(P.shape[0])])
    df["Q (annual)"] = Q
    return df
