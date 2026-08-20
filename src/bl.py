"""
Black-Litterman core.

All inputs and outputs are ANNUALISED. Mixing a daily covariance with a
risk-aversion coefficient of 2.5 (an annual-scale number) is the scale error
that makes views overwhelm the prior.
"""

import numpy as np

from .config import TRADING_DAYS


def annualise_cov(daily_cov: np.ndarray, freq: int = TRADING_DAYS) -> np.ndarray:
    return daily_cov * freq


def implied_equilibrium_returns(Sigma, w_mkt, risk_aversion):
    """
    Reverse optimisation: Pi = A * Sigma * w_mkt.

    Rather than estimating expected returns from historical means (famously
    noisy), we ask what returns would make the observed market portfolio
    optimal. This is the Black-Litterman prior.
    """
    w = np.asarray(w_mkt, float).reshape(-1)
    return risk_aversion * (np.asarray(Sigma, float) @ w)


def he_litterman_omega(P, Sigma, tau, confidence=1.0):
    """
    Omega = diag(P tau Sigma P') / confidence.

    Each view's uncertainty is proportional to the variance of its own view
    portfolio, so a view on a volatile spread is automatically treated as
    less precise. `confidence` > 1 tightens the views.

    IMPORTANT PROPERTY: with this proportional spec, tau cancels exactly out
    of the posterior mean. Omega is proportional to tau, so inv(Omega) carries
    1/tau; both terms of the posterior scale by 1/tau while V scales by tau.
    Only `confidence` moves the result. If you want tau itself to matter, pass
    an absolute Omega instead. Interviewers like this question.
    """
    P = np.atleast_2d(np.asarray(P, float))
    var = np.diag(P @ (tau * np.asarray(Sigma, float)) @ P.T)
    return np.diag(np.maximum(var, 1e-12) / confidence)


def posterior(Sigma, Pi, P, Q, Omega, tau):
    """
    Standard Black-Litterman posterior.

        V  = [ (tau*Sigma)^-1 + P' Omega^-1 P ]^-1
        mu = V [ (tau*Sigma)^-1 Pi + P' Omega^-1 Q ]

    Returns (mu, V). Uses solve() rather than inv() where possible for
    numerical stability.
    """
    Sigma = np.asarray(Sigma, float)
    P = np.atleast_2d(np.asarray(P, float))
    Pi = np.asarray(Pi, float).reshape(-1, 1)
    Q = np.asarray(Q, float).reshape(-1, 1)
    Omega = np.atleast_2d(np.asarray(Omega, float))

    if P.shape[1] != Sigma.shape[0]:
        raise ValueError(f"P has {P.shape[1]} columns but Sigma is "
                         f"{Sigma.shape[0]}x{Sigma.shape[0]}")
    if Q.shape[0] != P.shape[0]:
        raise ValueError("Q length must equal number of views (rows of P)")

    inv_tau_sigma = np.linalg.inv(tau * Sigma)
    inv_omega = np.linalg.inv(Omega)

    V = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)
    mu = V @ (inv_tau_sigma @ Pi + P.T @ inv_omega @ Q)
    return mu.reshape(-1), V


def optimal_weights(Sigma, mu, risk_aversion, posterior_cov=None,
                    gross_exposure=None, long_only=False):
    """
    Unconstrained mean-variance: w = (1/A) * S^-1 * mu.

    posterior_cov : pass V to use (Sigma + V), the fuller BL formulation that
                    accounts for estimation error in mu. None uses Sigma.
    gross_exposure: rescale so sum(|w|) equals this. ESSENTIAL for fair
                    benchmarking -- the original notebook compared a BL
                    portfolio with ~2.7x gross exposure against benchmarks at
                    1.0x, so most of its "outperformance" was just leverage.
    long_only     : clip negatives and renormalise (crude but shows the
                    no-shorting case Indian retail mandates often require).
    """
    S = np.asarray(Sigma, float)
    if posterior_cov is not None:
        S = S + np.asarray(posterior_cov, float)

    w = np.linalg.solve(S, np.asarray(mu, float).reshape(-1)) / risk_aversion

    if long_only:
        w = np.clip(w, 0, None)
        if w.sum() <= 0:
            raise ValueError("Long-only clipping removed all exposure.")

    if gross_exposure is not None:
        gross = np.abs(w).sum()
        if gross > 0:
            w = w * (gross_exposure / gross)
    return w


def run_bl(Sigma, w_mkt, P, Q, risk_aversion, tau, confidence,
           gross_exposure=None, use_posterior_cov=False):
    """Convenience wrapper: prior -> Omega -> posterior -> weights."""
    Pi = implied_equilibrium_returns(Sigma, w_mkt, risk_aversion)
    Omega = he_litterman_omega(P, Sigma, tau, confidence)
    mu, V = posterior(Sigma, Pi, P, Q, Omega, tau)
    w = optimal_weights(Sigma, mu, risk_aversion,
                        posterior_cov=V if use_posterior_cov else None,
                        gross_exposure=gross_exposure)
    return {"Pi": Pi, "Omega": Omega, "mu": mu, "V": V, "weights": w}
