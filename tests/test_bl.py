"""
Correctness tests. Run with:  pytest -v

These are the checks an interviewer would want to see. In particular
test_no_views_recovers_market and test_infinite_omega_recovers_prior are the
two canonical Black-Litterman sanity properties -- if either fails, the
implementation is wrong regardless of how good the backtest looks.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import bl, metrics, views
from src.backtest import walk_forward
from src.config import Config
from src.data import simple_returns, synthetic_data

TICKERS = [f"T{i}.NS" for i in range(10)]


@pytest.fixture(scope="module")
def synth():
    r, oil, betas = synthetic_data(TICKERS, n_days=1200, seed=42)
    return r, oil, betas


@pytest.fixture
def market_weights():
    rng = np.random.default_rng(7)
    w = rng.dirichlet(np.ones(10) * 5)
    return w


# ---------------------------------------------------------------- core BL
def test_no_views_recovers_market(synth, market_weights):
    """With zero views, w = (1/A) Sigma^-1 Pi must reproduce w_mkt exactly."""
    r, _, _ = synth
    Sigma = bl.annualise_cov(r.cov().values)
    Pi = bl.implied_equilibrium_returns(Sigma, market_weights, 2.5)
    w = bl.optimal_weights(Sigma, Pi, 2.5)
    np.testing.assert_allclose(w, market_weights, atol=1e-10)


def test_infinite_omega_recovers_prior(synth, market_weights):
    """Infinitely uncertain views must leave the prior untouched."""
    r, _, _ = synth
    Sigma = bl.annualise_cov(r.cov().values)
    Pi = bl.implied_equilibrium_returns(Sigma, market_weights, 2.5)
    P = np.array([[1.0] + [0.0] * 9, [0.0] * 9 + [1.0]])
    Q = np.array([0.5, -0.5])
    mu, _ = bl.posterior(Sigma, Pi, P, Q, np.diag([1e12, 1e12]), 0.05)
    np.testing.assert_allclose(mu, Pi, atol=1e-8)


def test_tight_omega_matches_view(synth, market_weights):
    """A near-certain absolute view should pull that asset's mu onto Q."""
    r, _, _ = synth
    Sigma = bl.annualise_cov(r.cov().values)
    Pi = bl.implied_equilibrium_returns(Sigma, market_weights, 2.5)
    P = np.array([[1.0] + [0.0] * 9])
    Q = np.array([0.42])
    mu, _ = bl.posterior(Sigma, Pi, P, Q, np.diag([1e-12]), 0.05)
    assert abs(mu[0] - 0.42) < 1e-4


def test_tau_cancels_under_he_litterman(synth, market_weights):
    """Documented property: proportional Omega makes the posterior tau-free."""
    r, _, _ = synth
    Sigma = bl.annualise_cov(r.cov().values)
    Pi = bl.implied_equilibrium_returns(Sigma, market_weights, 2.5)
    P = np.array([[1.0, -1.0] + [0.0] * 8])
    Q = np.array([0.05])
    out = []
    for tau in (0.01, 0.05, 0.5):
        Om = bl.he_litterman_omega(P, Sigma, tau, 1.0)
        out.append(bl.posterior(Sigma, Pi, P, Q, Om, tau)[0])
    np.testing.assert_allclose(out[0], out[1], atol=1e-10)
    np.testing.assert_allclose(out[1], out[2], atol=1e-10)


def test_higher_confidence_moves_toward_view(synth, market_weights):
    """Raising c must move the posterior closer to Q, monotonically."""
    r, _, _ = synth
    Sigma = bl.annualise_cov(r.cov().values)
    Pi = bl.implied_equilibrium_returns(Sigma, market_weights, 2.5)
    P = np.array([[1.0] + [0.0] * 9])
    Q = np.array([0.60])
    gaps = []
    for c in (0.1, 1.0, 10.0, 100.0):
        Om = bl.he_litterman_omega(P, Sigma, 0.05, c)
        mu, _ = bl.posterior(Sigma, Pi, P, Q, Om, 0.05)
        gaps.append(abs(mu[0] - 0.60))
    assert all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1))


def test_gross_exposure_normalisation(synth, market_weights):
    r, _, _ = synth
    Sigma = bl.annualise_cov(r.cov().values)
    mu = bl.implied_equilibrium_returns(Sigma, market_weights, 2.5) * 3
    w = bl.optimal_weights(Sigma, mu, 2.5, gross_exposure=1.0)
    assert abs(np.abs(w).sum() - 1.0) < 1e-12


def test_dimension_mismatch_raises(synth, market_weights):
    r, _, _ = synth
    Sigma = bl.annualise_cov(r.cov().values)
    Pi = bl.implied_equilibrium_returns(Sigma, market_weights, 2.5)
    with pytest.raises(ValueError):
        bl.posterior(Sigma, Pi, np.ones((1, 5)), np.array([0.1]),
                     np.diag([0.01]), 0.05)


# ---------------------------------------------------------------- views
def test_beta_recovery(synth):
    """OLS must recover the betas the synthetic data was built with."""
    r, oil, true_betas = synth
    est = views.estimate_oil_betas(r, oil)["beta"]
    np.testing.assert_allclose(est.values, true_betas.values, atol=0.05)


def test_view_rows_normalised(synth):
    r, oil, _ = synth
    b = views.estimate_oil_betas(r, oil)["beta"]
    P, Q, _ = views.build_views(b, TICKERS, -0.20)
    assert P.shape == (3, 10)
    assert abs(P[0].sum()) < 1e-12          # relative view is cash-neutral
    assert abs(np.abs(P[0]).sum() - 2.0) < 1e-12
    assert abs(P[1].sum() - 1.0) < 1e-12    # absolute views sum to 1


def test_view_sign_flips_with_shock(synth):
    """A negative oil shock must flip the sign of a positive-shock view."""
    r, oil, _ = synth
    b = views.estimate_oil_betas(r, oil)["beta"]
    _, q_up, _ = views.build_views(b, TICKERS, +0.20)
    _, q_dn, _ = views.build_views(b, TICKERS, -0.20)
    np.testing.assert_allclose(q_up, -q_dn, atol=1e-12)


# ---------------------------------------------------------------- metrics
def test_max_drawdown_simple_case():
    assert abs(metrics.max_drawdown([0.5, -0.5]) - (-0.5)) < 1e-12
    assert metrics.max_drawdown([0.01, 0.01, 0.01]) == 0.0


def test_drawdown_counts_first_day_loss():
    """Wealth starts at 1.0, so an immediate loss is captured."""
    assert metrics.max_drawdown([-0.10, 0.05]) < -0.099


def test_downside_deviation_not_subset_std():
    """Our estimator must differ from the naive std-of-negatives version."""
    r = np.array([0.02, -0.01, 0.03, -0.05, 0.01])
    ours = metrics.downside_deviation(r, freq=1)
    naive = np.std(r[r < 0])
    assert ours > naive


def test_sharpe_zero_rf_matches_manual(synth):
    r = np.array([0.01, -0.005, 0.02, 0.0, -0.01])
    s = metrics.perf_stats(r, rf_annual=0.0, freq=252)["Sharpe"]
    manual = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert abs(s - manual) < 1e-10


# ---------------------------------------------------------------- backtest
def test_walk_forward_is_out_of_sample(synth, market_weights):
    """Backtest must start only after the first full training window."""
    r, oil, _ = synth
    cfg = Config(tickers=TICKERS, train_window=250, rebalance_freq=21,
                 cost_bps=0.0)
    bt = walk_forward(r, oil, market_weights, cfg)
    assert bt.index[0] >= r.index[250]
    assert len(bt) > 0
    assert list(bt.columns) == ["BL", "MarketCap", "EqualWeight"]


def test_costs_reduce_returns(synth, market_weights):
    r, oil, _ = synth
    free = walk_forward(r, oil, market_weights,
                        Config(tickers=TICKERS, cost_bps=0.0))
    costly = walk_forward(r, oil, market_weights,
                          Config(tickers=TICKERS, cost_bps=50.0))
    assert costly["BL"].sum() < free["BL"].sum()


def test_insufficient_data_raises(synth, market_weights):
    r, oil, _ = synth
    with pytest.raises(ValueError, match="No rebalances"):
        walk_forward(r.iloc[:100], oil.iloc[:100], market_weights,
                     Config(tickers=TICKERS, train_window=250))


def test_column_order_preserved(synth):
    """Guards against the yfinance alphabetical-reorder bug."""
    r, _, _ = synth
    assert list(r.columns) == TICKERS


def test_simple_returns_reconstruct_prices():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]},
                          index=pd.bdate_range("2024-01-01", periods=3))
    ret = simple_returns(prices)
    rebuilt = 100.0 * np.cumprod(1 + ret["A"].values)
    np.testing.assert_allclose(rebuilt, [110.0, 99.0])


# ------------------------------------------------- prior-anchored views
def test_zero_shock_recovers_market(synth, market_weights):
    """
    THE null test for anchored views: a zero oil shock is 'no information',
    so Q must equal Pi, the posterior must equal the prior, and BL weights
    must reproduce market-cap weights. Without anchoring this fails, because
    Q = beta*0 = 0 asserts every asset returns exactly zero.
    """
    r, oil, _ = synth
    Sigma = bl.annualise_cov(r.cov().values)
    Pi = bl.implied_equilibrium_returns(Sigma, market_weights, 2.5)
    b = views.estimate_oil_betas(r, oil)["beta"]

    P, Q, _ = views.build_views(b, TICKERS, 0.0, prior_returns=Pi)
    mu, _ = bl.posterior(Sigma, Pi, P, Q,
                         bl.he_litterman_omega(P, Sigma, 0.05, 1.0), 0.05)
    np.testing.assert_allclose(mu, Pi, atol=1e-10)

    w = bl.optimal_weights(Sigma, mu, 2.5)
    np.testing.assert_allclose(w, market_weights, atol=1e-9)


def test_unanchored_zero_shock_is_biased(synth, market_weights):
    """Documents the bug: without anchoring, a zero shock is NOT neutral."""
    r, oil, _ = synth
    Sigma = bl.annualise_cov(r.cov().values)
    Pi = bl.implied_equilibrium_returns(Sigma, market_weights, 2.5)
    b = views.estimate_oil_betas(r, oil)["beta"]

    P, Q, _ = views.build_views(b, TICKERS, 0.0)   # no prior_returns
    mu, _ = bl.posterior(Sigma, Pi, P, Q,
                         bl.he_litterman_omega(P, Sigma, 0.05, 1.0), 0.05)
    assert not np.allclose(mu, Pi, atol=1e-6)
    assert mu.mean() < Pi.mean()   # systematically bearish


def test_anchored_views_are_not_uniformly_bearish(synth, market_weights):
    """
    A negative oil shock must RAISE the posterior for negative-beta names.
    The unanchored version pushed all ten assets down at once.
    """
    r, oil, _ = synth
    Sigma = bl.annualise_cov(r.cov().values)
    Pi = bl.implied_equilibrium_returns(Sigma, market_weights, 2.5)
    b = views.estimate_oil_betas(r, oil)["beta"]

    P, Q, _ = views.build_views(b, TICKERS, -0.20, prior_returns=Pi)
    mu, _ = bl.posterior(Sigma, Pi, P, Q,
                         bl.he_litterman_omega(P, Sigma, 0.05, 1.0), 0.05)
    assert (mu > Pi).any() and (mu < Pi).any()
