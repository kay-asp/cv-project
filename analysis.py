"""
Statistical analysis and visualisation.

Responsibilities:
  - Spearman correlation between CRS and degradation
  - Scatter plot of CRS vs degradation per species
  - Baseline comparison plots (uniform, Gaussian, real Grad-CAM)
  - Summary tables / figures for reporting
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

MARK_COLOR = "#2a6fdb"
INK_MUTED = "#555555"

def select_qualifying_species(per_species_df, min_test_n=5):
    """Species that can take part in the correlation.
 
    A species qualifies when it has at least min_test_n shallow test crops
    AND min_test_n deep test crops (so both accuracies are not just noise),
    and has a CRS value.
 
    Args:
        per_species_df: results table with n_shallow_test, n_deep_test, crs.
        min_test_n: minimum test crops on each side.
 
    Returns:
        list of species names.
    """
    df = per_species_df
    ok = ((df["n_shallow_test"] >= min_test_n)
          & (df["n_deep_test"] >= min_test_n)
          & df["crs"].notna())
    keep = df.index[ok].tolist()
    print(f"{len(keep)} / {len(df)} species qualify "
          f"(>= {min_test_n} shallow and deep test crops, CRS available)")
    return keep
    
def _paired(x_scores, y_scores, species=None):
    """Align two species->value dicts, keeping species where both are finite."""
    names = species if species is not None else sorted(set(x_scores) & set(y_scores))
    x = pd.Series({s: x_scores.get(s, np.nan) for s in names}, dtype=float)
    y = pd.Series({s: y_scores.get(s, np.nan) for s in names}, dtype=float)
    ok = x.notna() & y.notna()
    return x[ok], y[ok]

def spearman_correlation(crs_scores, degradation_scores, species=None):
    """Compute Spearman's ρ and p-value between CRS and degradation.

    Args:
        crs_scores: dict species -> CRS
        degradation_scores: dict species -> degradation
        species: optional list to restrict to (e.g. select_qualifying_species).

    Returns:
        rho (float), p_value (float)
    """
    x, y = _paired(crs_scores, degradation_scores, species)
    if len(x) < 3:
        raise ValueError(f"Only {len(x)} species with both values - can't correlate")
    rho, p = spearmanr(x, y)
    print(f"Spearman over {len(x)} species")
    return float(rho), float(p)


def plot_crs_vs_degradation(crs_scores, degradation_scores, species=None,
                            rho=None, p_value=None, n_labels=5, save_path=None):
    """Scatter plot: CRS (x) vs degradation (y) for each species.
    
    Labels the n_labels species furthest from the centre of the cloud, so the
    interesting outliers are named without cluttering the plot.

    Args:
        crs_scores: dict species -> CRS
        degradation_scores: dict species -> degradation
        save_path: If provided, save the figure to this path.
    """
    x, y = _paired(crs_scores, degradation_scores, species)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.axhline(0, color="#bbbbbb", linewidth=1, zorder=0)
    ax.scatter(x, y, s=28, color=MARK_COLOR, alpha=0.7,
               edgecolor="white", linewidth=0.8)
 
    z = ((x - x.mean()) / x.std()) ** 2 + ((y - y.mean()) / y.std()) ** 2
    for s in z.sort_values(ascending=False).index[:n_labels]:
        ax.annotate(s, (x[s], y[s]), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color=INK_MUTED)
 
    title = "Context reliance vs depth degradation"
    if rho is not None:
        title += f"\nSpearman ρ = {rho:.3f}, p = {p_value:.3g}, n = {len(x)} species"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Context Reliance Score (shallow test, Grad-CAM)")
    ax.set_ylabel("Degradation (shallow acc − deep acc)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_baseline_comparison(crs_scores, uniform_crs, gaussian_crs, species=None,
                             save_path=None):
    """Per-species CRS distribution for Grad-CAM vs the two trivial maps.
 
    If Grad-CAM's CRS sits near the uniform map's, the model's attribution
    spreads onto background no more than chance would. Above it = genuinely
    leaning on context; below it = focused on the organism.
    """
    names = species if species is not None else sorted(crs_scores)
    cols = {"Grad-CAM": crs_scores, "Uniform random": uniform_crs,
            "Centre Gaussian": gaussian_crs}
    values = [pd.Series({s: d.get(s, np.nan) for s in names}).dropna().values
              for d in cols.values()]
 
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(values, widths=0.5, showfliers=False,
               medianprops=dict(color=MARK_COLOR, linewidth=2),
               boxprops=dict(color=INK_MUTED), whiskerprops=dict(color=INK_MUTED),
               capprops=dict(color=INK_MUTED))
    rng = np.random.default_rng(0)
    for i, v in enumerate(values, start=1):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=8,
                   color=MARK_COLOR, alpha=0.35, linewidth=0)
    ax.set_xticks(range(1, len(cols) + 1), list(cols))
    ax.set_ylabel("Per-species CRS (fraction of map outside mask)")
    ax.set_ylim(0, 1)
    ax.set_title("CRS: Grad-CAM vs trivial attribution maps", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_control_comparison(main_degradation, control_degradation, save_path=None):
    """Compare degradation of shallow-only model vs all-depths control.

    Helps separate distribution shift from intrinsic difficulty.
    """
    raise NotImplementedError


def generate_summary_table(crs_scores, degradation_scores, rho, p_value,
                           baselines=None, species=None):
    """Create a summary DataFrame with per-species CRS, degradation,
    and the overall Spearman result.

    Returns:
        pd.DataFrame
    """
    rows = {"Grad-CAM": (crs_scores, rho, p_value)}
    for name, scores in (baselines or {}).items():
        r, p = spearman_correlation(scores, degradation_scores, species)
        rows[name] = (scores, r, p)
 
    out = []
    for name, (scores, r, p) in rows.items():
        x, _ = _paired(scores, degradation_scores, species)
        out.append({"method": name, "n_species": len(x), "mean_crs": x.mean(),
                    "spearman_rho": r, "p_value": p})
    return pd.DataFrame(out).set_index("method")
