#!/usr/bin/env python3
"""
End-to-end pipeline.

    python scripts/run_pipeline.py                  # full run on real data
    python scripts/run_pipeline.py --synthetic      # no network needed
    python scripts/run_pipeline.py --shock -0.30    # override the scenario
    python scripts/run_pipeline.py --refresh        # bypass the price cache
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import plots
from src.backtest import (rolling_beta_history, run_backtest, sensitivity_grid,
                          shock_scenarios)
from src.bl import annualise_cov, implied_equilibrium_returns, run_bl
from src.config import TABLES, Config
from src.data import (download_prices, load_cached_market_caps,
                      market_cap_weights, save_market_caps, simple_returns,
                      split_prices, synthetic_data)
from src.views import build_views, describe_views, estimate_oil_betas


def banner(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def save_table(df, name):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / name)
    print(f"  saved results/tables/{name}")


def parse_args():
    p = argparse.ArgumentParser(description="Black-Litterman under oil shocks")
    p.add_argument("--synthetic", action="store_true",
                   help="run on generated data (no network)")
    p.add_argument("--refresh", action="store_true", help="re-download prices")
    p.add_argument("--shock", type=float, default=None,
                   help="annual oil shock, e.g. -0.20")
    p.add_argument("--confidence", type=float, default=None)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    over = {k: v for k, v in {
        "oil_shock_annual": args.shock, "view_confidence": args.confidence,
        "start": args.start, "end": args.end}.items() if v is not None}
    if over:
        cfg = replace(cfg, **over)

    # ---------------------------------------------------------- 1. DATA
    banner("1. DATA")
    if args.synthetic:
        returns, oil_ret, _ = synthetic_data(cfg.tickers, n_days=1600, seed=1)
        w_mkt = np.random.default_rng(3).dirichlet(np.ones(cfg.n_assets) * 5)
        print("  synthetic mode: 1600 simulated days")
    else:
        raw = download_prices(cfg.tickers, cfg.oil_ticker, cfg.start, cfg.end,
                              force=args.refresh)
        prices, oil_px = split_prices(raw, cfg.tickers, cfg.oil_ticker)
        returns = simple_returns(prices)
        oil_ret = oil_px.pct_change().reindex(returns.index).fillna(0.0)
        w_mkt = load_cached_market_caps(cfg.tickers)
        if w_mkt is None:
            w_mkt = market_cap_weights(cfg.tickers)
            save_market_caps(w_mkt, cfg.tickers)

    assert list(returns.columns) == cfg.tickers, "COLUMN ORDER CORRUPTED"
    print(f"  {returns.index[0].date()} -> {returns.index[-1].date()}  "
          f"({len(returns)} trading days x {cfg.n_assets} stocks)")
    print(f"  column order verified against config: OK")
    print(f"  market-cap prior: "
          f"{dict(zip([t.replace('.NS','') for t in cfg.tickers], w_mkt.round(3)))}")

    # ------------------------------------------------- 2. OIL SENSITIVITY
    banner("2. OIL SENSITIVITY (full sample, reporting only)")
    beta_df = estimate_oil_betas(returns, oil_ret)
    print(beta_df.round(3).to_string())
    save_table(beta_df, "oil_betas.csv")

    # ---------------------------------------------------------- 3. VIEWS
    banner(f"3. VIEWS UNDER A {cfg.oil_shock_annual:+.0%} OIL SHOCK")
    _Sig_full = annualise_cov(returns.iloc[-cfg.train_window:].cov().values)
    _Pi_full = implied_equilibrium_returns(_Sig_full, w_mkt, cfg.risk_aversion)
    P, Q, labels = build_views(beta_df["beta"], cfg.tickers,
                               cfg.oil_shock_annual, prior_returns=_Pi_full)
    for i, lab in enumerate(labels, 1):
        print(f"  View {i}: {lab}")
    view_tbl = describe_views(P, Q, cfg.tickers)
    save_table(view_tbl.round(4), "views.csv")

    # ------------------------------------------- 4. SINGLE-PERIOD EXAMPLE
    banner("4. ILLUSTRATIVE ALLOCATION (final training window)")
    train = returns.iloc[-cfg.train_window:]
    Sigma = annualise_cov(train.cov().values)
    res = run_bl(Sigma, w_mkt, P, Q, cfg.risk_aversion, cfg.tau,
                 cfg.view_confidence, gross_exposure=cfg.gross_exposure,
                 use_posterior_cov=cfg.use_posterior_cov)
    alloc = pd.DataFrame({
        "Prior Pi (ann.)": res["Pi"],
        "Posterior mu (ann.)": res["mu"],
        "Market weight": w_mkt,
        "BL weight": res["weights"],
    }, index=[t.replace(".NS", "") for t in cfg.tickers])
    print(alloc.round(4).to_string())
    print(f"\n  gross exposure {np.abs(res['weights']).sum():.3f} | "
          f"net {res['weights'].sum():+.3f}")
    save_table(alloc.round(6), "allocation.csv")

    # ------------------------------------------------------ 5. BACKTEST
    banner("5. WALK-FORWARD OUT-OF-SAMPLE BACKTEST")
    print(f"  train {cfg.train_window}d | rebalance {cfg.rebalance_freq}d | "
          f"costs {cfg.cost_bps:.0f}bps | gross {cfg.gross_exposure}")
    bt, stats, weights = run_backtest(returns, oil_ret, w_mkt, cfg)
    print(f"  {len(weights)} rebalances, {len(bt)} out-of-sample days\n")
    print(stats.round(4).to_string())
    save_table(stats, "performance.csv")
    save_table(bt, "oos_returns.csv")
    save_table(weights, "weights_history.csv")

    bl_s, mk_s = stats.loc["BL"], stats.loc["MarketCap"]
    print(f"\n  BL vs market cap: Sharpe {bl_s['Sharpe']:.3f} vs "
          f"{mk_s['Sharpe']:.3f} | return {bl_s['Ann. Return']:.2%} vs "
          f"{mk_s['Ann. Return']:.2%} | maxDD {bl_s['Max DD']:.2%} vs "
          f"{mk_s['Max DD']:.2%}")

    # --------------------------------------------------- 6. SENSITIVITY
    banner("6. SENSITIVITY TO CONFIDENCE PARAMETERS")
    sens = sensitivity_grid(returns, oil_ret, w_mkt, cfg)
    print(sens.round(4).to_string(index=False))
    save_table(sens, "sensitivity.csv")
    if sens["tau"].nunique() > 1:
        spread = sens.groupby("confidence")["Sharpe"].std().max()
        if spread < 1e-8:
            print("\n  NOTE: tau has no effect, as expected -- it cancels out "
                  "of the posterior\n        under the He-Litterman Omega. "
                  "Only `confidence` moves results.")

    # ----------------------------------------------------- 7. SCENARIOS
    banner("7. OIL SHOCK SCENARIO SWEEP")
    sc = shock_scenarios(returns, oil_ret, w_mkt, cfg)
    print(sc.round(4).to_string(index=False))
    save_table(sc, "shock_scenarios.csv")

    # --------------------------------------------------------- 8. PLOTS
    if not args.no_plots:
        banner("8. FIGURES")
        plots.plot_wealth(bt)
        plots.plot_drawdown(bt)
        plots.plot_oil_betas(beta_df)
        plots.plot_weights(res["weights"], w_mkt, cfg.tickers)
        plots.plot_sensitivity(sens)
        plots.plot_shock_curve(sc)
        plots.plot_weight_evolution(weights)
        plots.plot_rolling_betas(
            rolling_beta_history(returns, oil_ret, cfg.train_window,
                                 cfg.rebalance_freq))

    banner("DONE")
    print("  tables  -> results/tables/")
    print("  figures -> results/figures/")
    return bt, stats, sens


if __name__ == "__main__":
    main()
