# Black–Litterman Portfolio Optimisation under Oil Price Shocks

A Bayesian asset-allocation framework applied to 10 NIFTY 50 stocks, where
investor views are **derived from estimated crude-oil betas** rather than
assumed. Evaluated with a walk-forward out-of-sample backtest including
transaction costs.

## Motivation

Indian equities are unusually exposed to crude: India imports roughly 85% of
its oil, so a shock propagates in opposite directions across the index —
upstream producers and metals gain, while FMCG and paints face input-cost
pressure. That divergence is exactly the kind of structured, non-consensus
belief Black–Litterman is designed to express, which makes it a better test
case than an arbitrary "I like this stock" view.

## Method

1. **Prior.** Reverse-optimise market-cap weights into implied equilibrium
   returns, `Π = A·Σ·w_mkt`, avoiding noisy historical mean estimates.
2. **Views.** Regress each stock's returns on Brent returns to get an oil
   beta, then map a shock scenario `s` into three views:
   - relative: top-3 beta basket vs bottom-3 basket, `Q₁ = (β̄_hi − β̄_lo)·s`
   - absolute: highest-beta stock, `Q₂ = β_max·s`
   - absolute: lowest-beta stock, `Q₃ = β_min·s`
3. **Uncertainty.** He–Litterman `Ω = diag(P·τΣ·Pᵀ)/c`, so volatile views are
   automatically down-weighted.
4. **Posterior.** `μ = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]`
5. **Allocation.** `w = (1/A)·Σ⁻¹·μ`, rescaled to fixed gross exposure.

## Design decisions worth defending

| Decision | Why |
|---|---|
| Simple returns, not log | `r_p = wᵀr` only holds for simple returns; log returns are additive across time, not across assets |
| Annualised Σ, Π, Q | `A = 2.5` is an annual-scale constant; pairing it with a daily covariance makes views swamp the prior |
| Explicit column reindex after download | `yfinance` returns columns alphabetically, silently misaligning Σ, P and `w_mkt` |
| Walk-forward, not one split | A single cutoff is one draw; on this data, shifting it four weeks moved Sharpe from 0.81 to 0.002 |
| Gross exposure matched at 1.0 | Unconstrained BL weights are levered; comparing them to a 1.0x benchmark measures leverage, not skill |
| Turnover costs at 10bps | BL rebalances aggressively; a costless backtest flatters it |
| Sortino via `√(mean(min(r−MAR,0)²))` | `std(r[r<0])` de-means the negative subset and understates downside risk |
| Views anchored to Π | Q is a *total* expected return; raw `β·s` is only the shock's marginal impact, which smuggles in a systematic bearish view |

## Structure

```
bl-oil-shock/
├── src/
│   ├── config.py      # every tunable, one dataclass
│   ├── data.py        # download, cache, column alignment, synthetic mode
│   ├── views.py       # oil beta regression -> P, Q
│   ├── bl.py          # prior, Omega, posterior, weights
│   ├── metrics.py     # Sharpe, Sortino, drawdown, turnover
│   ├── backtest.py    # walk-forward, sensitivity, scenarios
│   └── plots.py       # all figures
├── scripts/run_pipeline.py
├── tests/test_bl.py   # 19 tests
├── notebooks/report.ipynb
├── data/raw/          # cached prices (gitignored)
└── results/{figures,tables}/
```

## Setup

```bash
git clone <your-repo-url> && cd bl-oil-shock
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
pytest -v                                 # verify the implementation first
python scripts/run_pipeline.py            # full pipeline on live data
python scripts/run_pipeline.py --synthetic   # no network required
python scripts/run_pipeline.py --shock -0.30 --confidence 2.0
python scripts/run_pipeline.py --refresh  # bypass the price cache
```

Prices cache to `data/raw/` on first run, so subsequent runs are offline
and reproducible.

## Tests

The suite encodes the two canonical Black–Litterman correctness properties:

- with **zero views**, the optimiser must reproduce market-cap weights exactly
- with **Ω → ∞**, the posterior must collapse back to the prior

It also verifies OLS beta recovery against known synthetic betas, monotonic
convergence toward views as confidence rises, and that turnover costs reduce
returns.

## The null property

With views anchored to the prior, a **zero oil shock gives Q = Π exactly**, so
the posterior collapses to the prior and BL reproduces market-cap weights.
`test_zero_shock_recovers_market` pins this down. Unanchored, a zero shock
instead asserts "every asset returns exactly 0%" — a strong bearish view
disguised as no view, which pushed all ten posterior returns below their
priors and made the portfolio de-risk for the wrong reason.

## A note on τ

Under the He–Litterman specification, `Ω ∝ τ`, so `Ω⁻¹` carries `1/τ`. Both
terms of the posterior mean scale by `1/τ` while `V` scales by `τ`, and **τ
cancels exactly**. Only `confidence` moves the result. `test_tau_cancels_under_
he_litterman` pins this down. If you want τ to bite, supply an absolute Ω.

## Limitations

- Market caps come from `yfinance` as *current* values, a mild look-ahead in
  the prior. Acceptable for a stable large-cap universe; stated rather than hidden.
- Univariate oil betas ignore the market factor; a two-factor model
  (market + oil) would isolate oil exposure more cleanly.
- Costs are a flat turnover charge, with no market-impact or borrow costs for
  short legs.
- A 10-stock universe is a sandbox, not an investable strategy.

## 👤 Author

 **Anik Chakraborty**
   MSc Economics, IIT Kanpur

 **Aryan Anand**
   MSc Economics, IIT Kanpur
