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

def select_qualifying_species(per_species_df, min_test_n=5, crs_col="shallow_crs"):
    """Species that can take part in the correlation.
 
    A species qualifies when it has at least min_test_n shallow test crops
    AND min_test_n deep test crops (so both accuracies are not just noise),
    and has a CRS value.
 
    Args:
        per_species_df: results table with n_shallow_test, n_deep_test and crs_col.
        min_test_n: minimum test crops on each side.
        crs_col: the CRS column that must be present.
 
    Returns:
        list of species names.
    """
    df = per_species_df
    ok = ((df["n_shallow_test"] >= min_test_n)
          & (df["n_deep_test"] >= min_test_n)
          & df[crs_col].notna())
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


def bootstrap_spearman(image_crs, degradation_scores, species=None, n_boot=2000, seed=42):
    """Bootstrap CI for Spearman(species mean CRS, degradation).

    Species CRS from a handful of crops is noisy. Each resample redraws every
    species' crops with replacement, recomputes its mean CRS, and recorrelates
    with degradation, so the CI shows how much ρ moves with that noise.
    (Degradation is held fixed: only the CRS side is resampled.)

    Args:
        image_crs: per-image CRS DataFrame (species, crs) for the shallow split.
        degradation_scores: dict species -> degradation.
        species: optional list to restrict to.
        n_boot: number of resamples.
        seed: random seed.

    Returns:
        dict with rho (on the full data), ci_low, ci_high (95% percentile) and n.
    """
    from scipy.stats import rankdata
    from attribution import _bootstrap_means

    df = image_crs.dropna(subset=["crs"])
    names = species if species is not None else sorted(df["species"].unique())
    groups = {s: v.to_numpy() for s, v in df.groupby("species")["crs"]}
    names = [s for s in names if s in groups and np.isfinite(degradation_scores.get(s, np.nan))]
    if len(names) < 3:
        raise ValueError(f"Only {len(names)} species with both values - can't correlate")

    rng = np.random.default_rng(seed)
    boot = np.column_stack([_bootstrap_means(groups[s], n_boot, rng) for s in names])  # (n_boot, n_species)
    deg = rankdata([degradation_scores[s] for s in names])
    ranks = rankdata(boot, axis=1)
    rhos = np.array([np.corrcoef(r, deg)[0, 1] for r in ranks])

    rho = float(spearmanr([groups[s].mean() for s in names], deg)[0])
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    print(f"Spearman ρ = {rho:.3f}, 95% bootstrap CI [{lo:.3f}, {hi:.3f}] over {len(names)} species "
          f"(crops resampled within species, {n_boot} resamples)")
    return {"rho": rho, "ci_low": float(lo), "ci_high": float(hi), "n": len(names)}


def bootstrap_spearman_species(crs_scores, degradation_scores, species=None, n_boot=2000, seed=42):
    """Bootstrap CI for Spearman(CRS, degradation), resampling species.

    Species are the unit of analysis, so each resample draws species with
    replacement (keeping each species' CRS and degradation together) and
    recorrelates. Unlike bootstrap_spearman, this reflects which species
    happened to be sampled, the same unit the intervention CIs resample.

    Args:
        crs_scores: dict species -> CRS
        degradation_scores: dict species -> degradation
        species: optional list to restrict to.
        n_boot: number of resamples.
        seed: random seed.

    Returns:
        dict with rho (on the full data), ci_low, ci_high (95% percentile) and n.
    """
    x, y = _paired(crs_scores, degradation_scores, species)
    if len(x) < 3:
        raise ValueError(f"Only {len(x)} species with both values - can't correlate")
    x, y = x.to_numpy(), y.to_numpy()

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    rhos = np.array([spearmanr(x[i], y[i])[0] for i in idx])
    rhos = rhos[np.isfinite(rhos)]

    rho = float(spearmanr(x, y)[0])
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    print(f"Spearman ρ = {rho:.3f}, 95% bootstrap CI [{lo:.3f}, {hi:.3f}] over {len(x)} species "
          f"(species resampled, {n_boot} resamples)")
    return {"rho": rho, "ci_low": float(lo), "ci_high": float(hi), "n": len(x)}


def plot_crs_vs_degradation(crs_scores, degradation_scores, species=None,
                            rho=None, p_value=None, n_labels=5, ylabel="Degradation",
                            save_path=None):
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
 
    title = ("Context reliance vs depth degradation" if ylabel == "Degradation"
             else f"Context reliance vs {ylabel.lower()}")
    if rho is not None:
        title += f"\nρ = {rho:.3f}, p = {p_value:.3g}, n = {len(x)}"
    ax.set_title(title, fontsize=8, pad=6)
    ax.set_xlabel("Context Reliance Score", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300)
    plt.show()
    return fig


def crs_depth_model(predictions, per_species_df, species=None, crs_col="shallow_crs",
                    depth_df=None, depth_threshold=None, models=("shallow",)):
    """Crop-level test: does accuracy fall more at depth for high-CRS species?

    Fits correct ~ depth * crs on every test crop instead of one degradation
    value per species, so species with more crops count more and depth can be
    continuous. crs is the species' shallow CRS (standardised over the species
    used), never the crop's own CRS: a deep crop's CRS is itself affected by
    the shift and by whether the prediction was wrong. Two fits:

      - GEE (logistic, independence working correlation, robust SEs clustered
        by species): crops of one species aren't treated as independent.
      - Bayesian mixed logistic model with a random intercept per species
        (variational fit): each species keeps its own baseline accuracy.
        p-values are normal approximations from the posterior mean and SD.

    depth is the deep-split indicator (0/1), or with depth_df the depth in km
    relative to depth_threshold. The depth:crs coefficient is the hypothesis
    test (log-odds of a correct prediction): negative = high-CRS species lose
    more accuracy at depth. With both models, a shallow_model indicator is
    added and depth:crs:shallow_model asks whether that holds more for the
    shallow-trained model than for the control.

    Args:
        predictions: predictions.csv (model, split, image_id, species, correct).
        per_species_df: results table indexed by species, with crs_col.
        species: optional list to restrict to.
        crs_col: species-level CRS column ('shallow_crs', 'shallow_crs_excess', ...).
        depth_df: optional DataFrame with image_id, depth_m for continuous depth.
        depth_threshold: shallow/deep cutoff in m (needed with depth_df).
        models: which models' predictions to fit.

    Returns:
        DataFrame with columns method, term, coef, se, ci_low, ci_high, p.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy.stats import norm
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    df = predictions[predictions["model"].isin(models)].copy()
    if species is not None:
        df = df[df["species"].isin(species)]
    crs = per_species_df[crs_col].reindex(df["species"].unique()).dropna()
    df["crs"] = df["species"].map((crs - crs.mean()) / crs.std())
    df["correct"] = df["correct"].astype(int)
    if depth_df is None:
        df["depth"] = (df["split"] == "deep").astype(float)
        depth_label = "deep split (0/1)"
    else:
        if depth_threshold is None:
            raise ValueError("depth_df needs depth_threshold")
        depth = depth_df[["image_id", "depth_m"]].drop_duplicates("image_id")
        df = df.merge(depth, on="image_id", how="left")
        df["depth"] = (df["depth_m"] - depth_threshold) / 1000
        depth_label = f"km relative to {depth_threshold} m"
    df = df.dropna(subset=["crs", "depth"])

    formula, key = "correct ~ depth * crs", "depth:crs"
    if len(models) > 1:
        df["shallow_model"] = (df["model"] == "shallow").astype(float)
        formula, key = "correct ~ depth * crs * shallow_model", "depth:crs:shallow_model"

    gee = smf.gee(formula, groups="species", data=df, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Independence()).fit()
    ci = gee.conf_int()
    rows = [{"method": "GEE", "term": t, "coef": gee.params[t], "se": gee.bse[t],
             "ci_low": ci.loc[t, 0], "ci_high": ci.loc[t, 1], "p": gee.pvalues[t]}
            for t in gee.params.index]

    glmm = BinomialBayesMixedGLM.from_formula(formula, {"species": "0 + C(species)"}, df).fit_vb()
    for t, m, s in zip(glmm.model.exog_names, glmm.fe_mean, glmm.fe_sd):
        rows.append({"method": "Mixed (species intercept)", "term": t, "coef": m, "se": s,
                     "ci_low": m - 1.96 * s, "ci_high": m + 1.96 * s, "p": 2 * norm.sf(abs(m / s))})

    out = pd.DataFrame(rows)
    print(f"{formula}: {len(df)} crops, {df['species'].nunique()} species, "
          f"depth = {depth_label}, crs = {crs_col} (standardised)")
    for _, r in out[out["term"] == key].iterrows():
        print(f"  {r['method']}: {key} = {r['coef']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}], "
              f"p = {r['p']:.3g}")
    return out


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
    ax.set_ylabel("Mean per-species accuracy" if species_weighted else "Accuracy (pooled over crops)",
                  fontsize=8)
    title = "Per-species accuracy vs depth" if species_weighted else "Accuracy vs depth"
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


def plot_crs_vs_depth(image_crs_by_model, depth_df, crs_col="crs", depth_threshold=None,
                      n_bins=15, species=None, species_weighted=False, show_chance=True,
                      save_path=None):
    """Line graph of mean CRS vs depth for each model, on the shallow + deep test crops.

    Args:
        image_crs_by_model: dict model name -> image_crs table (both splits),
            e.g. {'shallow': image_crs.csv, 'control': image_crs_control.csv}.
        depth_df: DataFrame with columns image_id, depth_m.
        crs_col: 'crs' (true class), 'crs_pred' (predicted class), or an _excess column.
        depth_threshold: If given, draw a vertical line at the shallow/deep cutoff.
        n_bins: Number of equal-frequency depth bins (shared by all models).
        species: optional list to restrict to.
        species_weighted: average per-species means within each bin, as in
            plot_accuracy_vs_depth.
        show_chance: also draw the flat-map CRS (1 - coverage), which is the same
            for every model; if CRS follows it, the change is mask size, not attention.
        save_path: If provided, save the figure to this path.
    """
    depth = depth_df[["image_id", "depth_m"]].drop_duplicates("image_id")
    df = pd.concat([d.assign(model=m) for m, d in image_crs_by_model.items()], ignore_index=True)
    if species is not None:
        df = df[df["species"].isin(species)]
    df = df.merge(depth, on="image_id", how="left").dropna(subset=["depth_m", crs_col])
    df["bin_mid"] = pd.qcut(df["depth_m"], n_bins, duplicates="drop").apply(lambda b: b.mid)

    def binned(sub, col):
        if not species_weighted:
            return sub.groupby("bin_mid", observed=True)[col].mean()
        return sub.groupby(["bin_mid", "species"], observed=True)[col].mean().groupby("bin_mid").mean()

    colors = {"shallow": MARK_COLOR, "control": "#eb6834"}
    labels = {"shallow": "Shallow-trained", "control": "Control (all depths)"}
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for model in image_crs_by_model:
        crs = binned(df[df["model"] == model], crs_col)
        ax.plot(crs.index, crs.values, marker="o", markersize=3, linewidth=1.2,
                label=labels.get(model, model), color=colors.get(model, INK_MUTED))
    if show_chance and not crs_col.endswith("_excess"):
        first = df[df["model"] == next(iter(image_crs_by_model))]
        chance = binned(first, "crs_chance")
        ax.plot(chance.index, chance.values, linestyle=":", linewidth=1.2, color="#999999",
                label="Chance (flat map)")
    if depth_threshold is not None:
        ax.axvline(depth_threshold, color=INK_MUTED, linestyle="--", linewidth=0.8,
                   label=f"Depth threshold ({depth_threshold} m)")

    which = "predicted class" if crs_col.startswith("crs_pred") else "true class"
    ax.set_xlabel("Depth (m)", fontsize=8)
    ax.set_ylabel(f"Mean {crs_col.replace('_', ' ').replace('crs', 'CRS')}", fontsize=8)
    ax.set_title(f"Context reliance vs depth ({which})"
                 + (" (species-weighted)" if species_weighted else ""), fontsize=8, pad=6)
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


def crs_depth_breakdown(image_crs_by_model, depth_df, species=None, n_bins=15, min_each=3):
    """Is the rise in CRS at depth a change within species, or errors and species mix?

    plot_crs_vs_depth pools every crop in a bin, so a bin's CRS also moves with its
    share of misclassified crops (which have higher CRS) and with which species
    live at that depth. Per bin this reports the pooled CRS, the share of wrong
    crops, and CRS over correct crops with each species weighted equally. It also
    compares each species with itself: mean correct-crop CRS on deep minus shallow
    test crops, for species with >= min_each correct crops on each side, tested
    with a Wilcoxon signed-rank test.

    Args:
        image_crs_by_model: dict model name -> image_crs table (both splits).
        depth_df: DataFrame with columns image_id, depth_m.
        species: optional list to restrict to.
        n_bins: Number of equal-frequency depth bins (as in plot_crs_vs_depth).
        min_each: minimum correct crops per split for the within-species test.

    Returns:
        (dict model -> per-bin DataFrame, dict model -> (median difference, p, n species))
    """
    from scipy.stats import wilcoxon

    depth = depth_df[["image_id", "depth_m"]].drop_duplicates("image_id")
    tables, within = {}, {}
    for model, ic in image_crs_by_model.items():
        df = ic if species is None else ic[ic["species"].isin(species)]
        df = df.merge(depth, on="image_id", how="left").dropna(subset=["depth_m", "crs"])
        df["bin_mid"] = pd.qcut(df["depth_m"], n_bins, duplicates="drop").apply(lambda b: round(b.mid))
        right = df[df["correct"]]
        tables[model] = pd.DataFrame({
            "crs_all": df.groupby("bin_mid", observed=True)["crs"].mean(),
            "share_wrong": 1 - df.groupby("bin_mid", observed=True)["correct"].mean(),
            "crs_correct_species_weighted": right.groupby(["bin_mid", "species"], observed=True)["crs"]
                                                 .mean().groupby("bin_mid").mean()})
        g = right.groupby(["species", "split"])["crs"].agg(["mean", "count"]).unstack("split")
        ok = (g[("count", "shallow")] >= min_each) & (g[("count", "deep")] >= min_each)
        diffs = (g.loc[ok, ("mean", "deep")] - g.loc[ok, ("mean", "shallow")]).dropna()
        within[model] = (float(diffs.median()), float(wilcoxon(diffs).pvalue), len(diffs))
    return tables, within


def crs_depth_consistency(image_crs, species=None, min_n=3):
    """Is species CRS a stable property, or noise? Shallow vs deep species CRS.

    Correlates each species' mean CRS on its correctly classified shallow test
    crops with the same on its correctly classified deep crops (species with
    >= min_n correct crops on each side). Organism size alone makes CRS
    consistent (small organisms leave more background), so the correlation is
    also reported partialled on shallow mask coverage (the flat-map CRS,
    crs_chance) and on excess CRS (crs - chance). A high value means CRS ranks
    species the same way on separate images; it does not mean shallow attention
    causes deep attention (both come from the same frozen model).

    Args:
        image_crs: per-image CRS table (both splits) with species, split,
            correct, crs, crs_chance and crs_excess.
        species: optional list to restrict to.
        min_n: minimum correct crops per split.

    Returns:
        dict with rho, p, partial_rho, partial_p, excess_rho, excess_p and n.
    """
    from scipy.stats import pearsonr

    df = image_crs if species is None else image_crs[image_crs["species"].isin(species)]
    df = df[df["correct"]].dropna(subset=["crs"])
    g = df.groupby(["species", "split"])[["crs", "crs_chance", "crs_excess"]].mean().unstack("split")
    n = df.groupby(["species", "split"]).size().unstack("split")
    ok = (n["shallow"] >= min_n) & (n["deep"] >= min_n)
    g = g[ok.reindex(g.index, fill_value=False)].dropna()

    rho, p = spearmanr(g[("crs", "shallow")], g[("crs", "deep")])
    rk = pd.DataFrame({"sh": g[("crs", "shallow")], "dp": g[("crs", "deep")],
                       "cov": g[("crs_chance", "shallow")]}).rank()
    resid = lambda y, x: y - np.polyval(np.polyfit(x, y, 1), x)
    pr, pp = pearsonr(resid(rk["sh"], rk["cov"]), resid(rk["dp"], rk["cov"]))
    ex, exp_ = spearmanr(g[("crs_excess", "shallow")], g[("crs_excess", "deep")])
    print(f"Shallow vs deep species CRS (correct crops): rho = {rho:.3f}, p = {p:.2g}, n = {len(g)}; "
          f"partial on mask coverage = {pr:.3f} (p = {pp:.2g}); excess CRS rho = {ex:.3f} (p = {exp_:.2g})")
    return {"rho": float(rho), "p": float(p), "partial_rho": float(pr), "partial_p": float(pp),
            "excess_rho": float(ex), "excess_p": float(exp_), "n": len(g)}


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
