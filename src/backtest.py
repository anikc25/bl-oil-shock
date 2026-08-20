"""
Walk-forward backtesting.

Every weight vector is estimated on a trailing window and held forward, so
every return is genuinely out-of-sample. This replaces the single-cutoff
split, which is one draw from a distribution and therefore trivially
cherry-picked: in the original notebook, moving the cutoff back four weeks
turned a Sharpe of 0.81 into 0.002.
"""

import numpy as np
import pandas as pd

from .bl import annualise_cov, implied_equilibrium_returns, run_bl
from .config import TRADING_DAYS
from .metrics import stats_table, turnover
from .views import build_views, estimate_oil_betas


def _apply_costs(period_returns, w_new, w_old, cost_bps):
    """Charge turnover cost on the first day of the holding period."""
    r = np.asarray(period_returns, float).copy()
    if cost_bps and len(r):
        r[0] -= turnover(w_new, w_old) * cost_bps / 10_000.0
    return r


def walk_forward(returns: pd.DataFrame, oil_returns: pd.Series,
                 w_mkt: np.ndarray, cfg, return_weights=False):
    """
    Rolling estimate-and-hold loop.

    For each rebalance date t:
      1. Estimate Sigma and oil betas on returns[t-train_window : t]
      2. Build views from those betas + the shock scenario
      3. Solve BL for weights
      4. Hold for `rebalance_freq` days, recording realised returns
    """
    tickers = list(returns.columns)
    n = len(tickers)
    dates = returns.index

    w_mkt = np.asarray(w_mkt, float)
    w_eq = np.ones(n) / n
    g = cfg.gross_exposure
    w_mkt_s = w_mkt * (g / np.abs(w_mkt).sum())
    w_eq_s = w_eq * (g / np.abs(w_eq).sum())

    prev = {"BL": np.zeros(n), "MarketCap": np.zeros(n), "EqualWeight": np.zeros(n)}
    chunks = {"BL": [], "MarketCap": [], "EqualWeight": []}
    idx_chunks, weight_log = [], []

    start = cfg.train_window
    while start + cfg.rebalance_freq <= len(dates):
        train = returns.iloc[start - cfg.train_window:start]
        test = returns.iloc[start:start + cfg.rebalance_freq]
        oil_train = oil_returns.iloc[start - cfg.train_window:start]

        try:
            Sigma = annualise_cov(train.cov().values)
            betas = estimate_oil_betas(train, oil_train)["beta"]
            # Anchor views to this window's equilibrium returns so that Q is a
            # TOTAL expected return, not just the shock's marginal impact.
            Pi = implied_equilibrium_returns(Sigma, w_mkt, cfg.risk_aversion)
            P, Q, _ = build_views(betas, tickers, cfg.oil_shock_annual,
                                  prior_returns=Pi)
            res = run_bl(Sigma, w_mkt, P, Q,
                         risk_aversion=cfg.risk_aversion,
                         tau=cfg.tau, confidence=cfg.view_confidence,
                         gross_exposure=g,
                         use_posterior_cov=cfg.use_posterior_cov)
            w_bl = res["weights"]
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"  [skip {dates[start].date()}] {type(e).__name__}: {e}")
            start += cfg.rebalance_freq
            continue

        for name, w in (("BL", w_bl), ("MarketCap", w_mkt_s),
                        ("EqualWeight", w_eq_s)):
            chunks[name].append(_apply_costs(test.values @ w, w, prev[name],
                                             cfg.cost_bps))
            prev[name] = w

        idx_chunks.append(test.index)
        weight_log.append(pd.Series(w_bl, index=tickers, name=dates[start]))
        start += cfg.rebalance_freq

    if not idx_chunks:
        raise ValueError(
            f"No rebalances completed. Need > {cfg.train_window + cfg.rebalance_freq} "
            f"observations, have {len(dates)}. Extend `start` in the config."
        )

    idx = pd.DatetimeIndex(np.concatenate([i.values for i in idx_chunks]))
    out = pd.DataFrame({k: np.concatenate(v) for k, v in chunks.items()}, index=idx)
    if return_weights:
        return out, pd.DataFrame(weight_log)
    return out


def run_backtest(returns, oil_returns, w_mkt, cfg):
    """Backtest + summary table in one call."""
    bt, weights = walk_forward(returns, oil_returns, w_mkt, cfg,
                               return_weights=True)
    stats = stats_table(bt, rf_annual=cfg.rf_annual)
    return bt, stats, weights


def _clone(cfg, **kw):
    from dataclasses import replace
    return replace(cfg, **kw)


def sensitivity_grid(returns, oil_returns, w_mkt, cfg,
                     confidences=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
                     taus=(0.025, 0.05, 0.10)):
    """
    Sweep the CONFIDENCE parameters -- Omega scaling and tau.

    Note you should expect tau to have no effect under the He-Litterman Omega
    (it cancels). Reporting that is a finding, not a bug: it demonstrates you
    understand the parameterisation rather than sweeping knobs blindly.
    """
    rows = []
    for tau in taus:
        for c in confidences:
            cfg_i = _clone(cfg, tau=tau, view_confidence=c)
            try:
                bt = walk_forward(returns, oil_returns, w_mkt, cfg_i)
                s = stats_table(bt, rf_annual=cfg.rf_annual).loc["BL"]
                rows.append({"tau": tau, "confidence": c, **s.to_dict()})
            except ValueError as e:
                print(f"  [skip tau={tau}, c={c}] {e}")
    return pd.DataFrame(rows)


def shock_scenarios(returns, oil_returns, w_mkt, cfg,
                    shocks=(-0.40, -0.20, -0.10, 0.0, 0.10, 0.20, 0.40)):
    """How does the strategy behave across assumed oil shock magnitudes?"""
    rows = []
    for s in shocks:
        cfg_i = _clone(cfg, oil_shock_annual=s)
        try:
            bt = walk_forward(returns, oil_returns, w_mkt, cfg_i)
            stat = stats_table(bt, rf_annual=cfg.rf_annual).loc["BL"]
            rows.append({"Oil shock": s, **stat.to_dict()})
        except ValueError as e:
            print(f"  [skip shock={s}] {e}")
    return pd.DataFrame(rows)


def rolling_beta_history(returns, oil_returns, window=250, step=21):
    """Time series of oil betas -- shows regime shifts in oil sensitivity."""
    rows = []
    for start in range(window, len(returns), step):
        tr = returns.iloc[start - window:start]
        try:
            b = estimate_oil_betas(tr, oil_returns.iloc[start - window:start])["beta"]
            rows.append(b.rename(returns.index[start]))
        except ValueError:
            continue
    return pd.DataFrame(rows)
