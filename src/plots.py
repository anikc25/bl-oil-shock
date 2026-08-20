"""Figures for the report. Every function saves to results/figures/."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES
from .metrics import drawdown_series, wealth_curve

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(FIGURES.parents[1])}")
    return path


def plot_wealth(bt: pd.DataFrame, name="01_cumulative_wealth.png"):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for c in bt.columns:
        ax.plot(bt.index, wealth_curve(bt[c])[1:], label=c, linewidth=1.8)
    ax.set_title("Walk-forward out-of-sample cumulative wealth (leverage matched)")
    ax.set_ylabel("Growth of 1"); ax.set_xlabel("Date"); ax.legend()
    return _save(fig, name)


def plot_drawdown(bt: pd.DataFrame, name="02_drawdown.png"):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for c in bt.columns:
        d = drawdown_series(bt[c], bt.index)
        ax.fill_between(d.index, d.values, 0, alpha=0.25)
        ax.plot(d.index, d.values, label=c, linewidth=1.3)
    ax.set_title("Drawdown"); ax.set_ylabel("Drawdown"); ax.legend()
    return _save(fig, name)


def plot_oil_betas(beta_df: pd.DataFrame, name="03_oil_betas.png"):
    d = beta_df.sort_values("beta")
    colors = ["#c0392b" if b < 0 else "#27ae60" for b in d["beta"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([t.replace(".NS", "") for t in d.index], d["beta"], color=colors)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Beta to Brent crude returns")
    ax.set_title("Estimated oil sensitivity (full sample)")
    for i, (b, t) in enumerate(zip(d["beta"], d["t_stat"])):
        ax.text(b, i, f"  t={t:.1f}", va="center",
                ha="left" if b >= 0 else "right", fontsize=8)
    return _save(fig, name)


def plot_weights(w_bl, w_mkt, tickers, name="04_weights.png"):
    x = np.arange(len(tickers)); wd = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - wd/2, w_mkt, wd, label="Market cap (prior)")
    ax.bar(x + wd/2, w_bl, wd, label="Black-Litterman (posterior)")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(".NS", "") for t in tickers], rotation=45, ha="right")
    ax.set_ylabel("Weight"); ax.set_title("Prior vs posterior allocation")
    ax.legend()
    return _save(fig, name)


def plot_sensitivity(sens: pd.DataFrame, metric="Sharpe",
                     name="05_sensitivity.png"):
    piv = sens.pivot(index="tau", columns="confidence", values=metric)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto")
    fig.colorbar(im, ax=ax, label=metric)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("View confidence (c)"); ax.set_ylabel("tau")
    ax.set_title(f"Out-of-sample {metric} sensitivity")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center",
                    va="center", fontsize=8)
    return _save(fig, name)


def plot_shock_curve(sc: pd.DataFrame, name="06_shock_scenarios.png"):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(sc["Oil shock"], sc["Ann. Return"], "o-", color="#2c3e50",
             label="Ann. return")
    ax1.set_xlabel("Assumed annual oil shock"); ax1.set_ylabel("Annualised return")
    ax1.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(sc["Oil shock"], sc["Sharpe"], "s--", color="#e67e22", label="Sharpe")
    ax2.set_ylabel("Sharpe")
    ax1.set_title("Performance across oil shock scenarios")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    return _save(fig, name)


def plot_weight_evolution(weights: pd.DataFrame, name="07_weight_evolution.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    for c in weights.columns:
        ax.plot(weights.index, weights[c], label=c.replace(".NS", ""), linewidth=1.2)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_title("BL weights through time (each point a rebalance)")
    ax.set_ylabel("Weight")
    ax.legend(ncol=5, fontsize=7)
    return _save(fig, name)


def plot_rolling_betas(rb: pd.DataFrame, name="08_rolling_betas.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    for c in rb.columns:
        ax.plot(rb.index, rb[c], label=c.replace(".NS", ""), linewidth=1.1)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_title("Rolling 250-day oil betas")
    ax.set_ylabel("Beta"); ax.legend(ncol=5, fontsize=7)
    return _save(fig, name)
