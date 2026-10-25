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
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.axhline(0, color="#bbbbbb", linewidth=0.8, zorder=0)
    ax.scatter(x, y, s=12, color=MARK_COLOR, alpha=0.7,
               edgecolor="white", linewidth=0.4)
 
    z = ((x - x.mean()) / x.std()) ** 2 + ((y - y.mean()) / y.std()) ** 2
    for s in z.sort_values(ascending=False).index[:n_labels]:
        ax.annotate(s, (x[s], y[s]), xytext=(5, 5), textcoords="offset points",
                    fontsize=6, color=INK_MUTED)
 
    title = "Context reliance vs depth degradation"
    if rho is not None:
        title += f"\nρ = {rho:.3f}, p = {p_value:.3g}, n = {len(x)}"
    ax.set_title(title, fontsize=8, pad=6)
    ax.set_xlabel("Context Reliance Score", fontsize=8)
    ax.set_ylabel("Degradation", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300)
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
 
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.boxplot(values, widths=0.5, showfliers=False,
               medianprops=dict(color=MARK_COLOR, linewidth=2),
               boxprops=dict(color=INK_MUTED), whiskerprops=dict(color=INK_MUTED),
               capprops=dict(color=INK_MUTED))
    rng = np.random.default_rng(0)
    for i, v in enumerate(values, start=1):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=4,
                   color=MARK_COLOR, alpha=0.35, linewidth=0)
    ax.set_xticks(range(1, len(cols) + 1), list(cols))
    ax.set_ylabel("Per-species CRS", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("CRS: Grad-CAM vs trivial attribution maps", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300)
    plt.show()
    return fig


def _binned_accuracy(df, bin_col="bin_mid", species_weighted=False):
    """Mean accuracy per depth bin, optionally weighting species equally."""
    if not species_weighted:
        return df.groupby(bin_col)["correct"].mean()
    per = df.groupby([bin_col, "species"])["correct"].mean()
    return per.groupby(bin_col).mean()


def plot_accuracy_vs_depth(predictions, depth_df, depth_threshold=None,
                           n_bins=15, int_preds=None, int_modes=None,
                           species_weighted=False, save_path=None):
    """Line graph of accuracy vs depth for each model.

    Args:
        predictions: DataFrame from evaluation.save_results with columns
            model, split, image_id, correct.
        depth_df: DataFrame with columns image_id, depth_m.
        depth_threshold: If given, draw a vertical line at this depth
            (the shallow/deep training cutoff) for the shallow model.
        n_bins: Number of equal-frequency depth bins.
        int_preds: Optional intervention predictions DataFrame with columns
            model, split, mode, image_id, correct.
        int_modes: List of intervention modes to overlay, e.g.
            ['grey_bg', 'blur_bg'].  Accepts a single string too.
        species_weighted: If True, compute per-species accuracy within each
            depth bin first, then average across species so every species
            counts equally regardless of image count.
        save_path: If provided, save the figure to this path.
    """
    df = predictions.merge(depth_df[["image_id", "depth_m"]], on="image_id", how="left")
    df = df.dropna(subset=["depth_m"])

    df["depth_bin"] = pd.qcut(df["depth_m"], n_bins, duplicates="drop")
    df["bin_mid"] = df["depth_bin"].apply(lambda b: b.mid)

    colors = {"shallow": MARK_COLOR, "control": "#eb6834"}
    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    for model in df["model"].unique():
        sub = df[df["model"] == model]
        acc = _binned_accuracy(sub, species_weighted=species_weighted)
        ax.plot(acc.index, acc.values, marker="o", markersize=3,
                linewidth=1.2, label=model.capitalize(),
                color=colors.get(model, INK_MUTED))

    if int_preds is not None and int_modes is not None:
        if isinstance(int_modes, str):
            int_modes = [int_modes]
        mode_style = {
            "grey_bg": ("Bg removed", "#8b5cf6", "s"),
            "blur_bg": ("Bg blurred", "#eb6834", "^"),
        }
        for mode in int_modes:
            label, color, marker = mode_style.get(mode, (mode, INK_MUTED, "D"))
            ip = int_preds[(int_preds["model"] == "shallow")
                           & (int_preds["mode"] == mode)]
            ip = ip.merge(depth_df[["image_id", "depth_m"]],
                          on="image_id", how="left")
            ip = ip.dropna(subset=["depth_m"])
            ip["depth_bin"] = pd.cut(ip["depth_m"],
                                     bins=df["depth_bin"].cat.categories)
            ip["bin_mid"] = ip["depth_bin"].apply(lambda b: b.mid)
            acc_int = _binned_accuracy(ip, species_weighted=species_weighted)
            ax.plot(acc_int.index, acc_int.values, marker=marker,
                    markersize=3, linewidth=1.2, linestyle="--",
                    label=f"Shallow ({label})", color=color)

    if depth_threshold is not None:
        ax.axvline(depth_threshold, color=INK_MUTED, linestyle="--",
                   linewidth=0.8, label=f"Depth threshold ({depth_threshold} m)")

    ax.set_xlabel("Depth (m)", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=8)
    title = "Accuracy vs depth"
    if species_weighted:
        title += " (species-weighted)"
    ax.set_title(title, fontsize=8, pad=6)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300)
    plt.show()
    return fig


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
