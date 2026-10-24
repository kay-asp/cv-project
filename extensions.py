"""
Extension analyses - no retraining needed.

  1. Background intervention: grey out / blur the background (or remove the
     organism) on the test crops and re-measure accuracy. A causal test of
     whether the models use background, instead of reading it off Grad-CAM.
  2. Background shift: how much the background colour / brightness changes
     between shallow and deep crops of each species, and whether species whose
     background shifts most AND whose model relies on context degrade most.
  3. Per-image CRS: are the crops the model gets wrong at depth the ones with
     high context reliance?

Every function takes loaders built with data.get_dataloaders(..., coco=coco),
which yield (images, labels, masks).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from torchvision.transforms.functional import gaussian_blur

import config
import data
from attribution import _gradcam_batch

MODES = ("none", "grey_bg", "blur_bg", "grey_organism")
MODE_LABELS = {"none": "Original", "grey_bg": "Background greyed",
               "blur_bg": "Background blurred", "grey_organism": "Organism greyed"}
SPLIT_COLORS = {"shallow": "#2a78d6", "deep": "#eb6834"}
INK_MUTED = "#555555"


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


# ── 1. Background intervention ───────────────────────────────────────

def dilate_masks(masks, px):
    """Grow each organism mask by px pixels (so its edges aren't cut off).

    Args:
        masks: (B, H, W) tensor, 1 = organism.
        px: pixels to grow by; 0 returns the masks unchanged (as float).
    """
    masks = masks.float()
    if px <= 0:
        return masks
    return F.max_pool2d(masks[:, None], kernel_size=2 * px + 1, stride=1, padding=px)[:, 0]


def apply_intervention(images, masks, mode, dilate_px=3, blur_sigma=10.0):
    """Edit a batch of normalised crops.

    Modes:
        'none':          unchanged.
        'grey_bg':       background set to ImageNet-mean grey (0 after
                         normalisation), organism untouched.
        'blur_bg':       background heavily Gaussian-blurred, organism sharp.
                         Milder than grey: keeps the background's colour and
                         brightness but removes its texture and detail.
        'grey_organism': organism greyed out, background untouched - can the
                         model still name the species from background alone?

    Each removal errs on the side of removing too much, so nothing leaks:
    the *_bg modes use the exact mask (growing it would keep a ring of real
    background around the organism, which a model can still read), while
    grey_organism greys the mask grown by dilate_px (so organism edge pixels
    don't survive in the "background only" image).

    Args:
        images: (B, 3, H, W) normalised tensor.
        masks: (B, H, W) tensor, 1 = organism, same device as images.
        dilate_px: mask growth used by grey_organism only.
    """
    if mode == "none":
        return images
    grow = dilate_px if mode == "grey_organism" else 0
    organism = dilate_masks(masks, grow)[:, None].to(images.dtype)        # (B, 1, H, W)
    if mode == "grey_bg":
        return images * organism
    if mode == "blur_bg":
        # Blur only background pixels (normalised convolution), so the
        # organism's colour doesn't smear out into a halo around it
        k = int(4 * blur_sigma) | 1                                       # odd kernel size
        bg = 1 - organism
        num = gaussian_blur(images * bg, [k, k], [blur_sigma, blur_sigma])
        den = gaussian_blur(bg, [k, k], [blur_sigma, blur_sigma])
        blurred = num / den.clamp_min(1e-6)
        return images * organism + blurred * bg
    if mode == "grey_organism":
        return images * (1 - organism)
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def run_interventions(models, loaders, species_list, modes=MODES, dilate_px=3, blur_sigma=10.0):
    """Predict every test crop under every intervention, for every model.

    Each batch is loaded from disk once and reused for every (model, mode),
    since loading crops + masks is the slow part on Colab.

    Args:
        models: dict name -> model, e.g. {'shallow': classifier, 'control': control}.
        loaders: dict split -> mask loader, e.g. {'shallow': ..., 'deep': ...}.
        species_list: class order.

    Returns:
        DataFrame, one row per (model, split, mode, crop): model, split, mode,
        image_id, ann_id, species, pred, confidence, correct.
    """
    device = _device()
    for m in models.values():
        m.to(device).eval()

    out = []
    for split, loader in loaders.items():
        ds = loader.dataset
        rec = {(name, mode): ([], [], []) for name in models for mode in modes}
        for images, labels, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            for mode in modes:
                with torch.no_grad():
                    x = apply_intervention(images, masks, mode, dilate_px, blur_sigma)
                    x = x.contiguous(memory_format=torch.channels_last)
                    for name, m in models.items():
                        with torch.autocast(device.type, dtype=torch.float16,
                                            enabled=device.type == "cuda"):
                            probs = m(x).float().softmax(dim=1)
                        conf, pred = probs.max(dim=1)
                        lab, prd, cnf = rec[(name, mode)]
                        lab.append(labels)
                        prd.append(pred.cpu())
                        cnf.append(conf.cpu())
        for (name, mode), (lab, prd, cnf) in rec.items():
            lab, prd = torch.cat(lab).tolist(), torch.cat(prd).tolist()
            out.append(pd.DataFrame({
                "model": name, "split": split, "mode": mode,
                "image_id": ds.image_ids, "ann_id": ds.ann_ids,
                "species": [species_list[i] for i in lab],
                "pred": [species_list[i] for i in prd],
                "confidence": torch.cat(cnf).tolist(),
                "correct": [a == b for a, b in zip(lab, prd)],
            }))
        print(f"{split}: {len(ds)} crops x {len(modes)} modes x {len(models)} models done")
    return pd.concat(out, ignore_index=True)


def intervention_per_species(preds, species=None):
    """Per-species accuracy for every (model, split, mode).

    Returns:
        DataFrame indexed by species, columns '<model>_<split>_<mode>'.
    """
    acc = preds.groupby(["model", "split", "mode", "species"])["correct"].mean()
    wide = acc.unstack(["model", "split", "mode"])
    wide.columns = ["_".join(c) for c in wide.columns]
    if species is not None:
        wide = wide.reindex(species)
    return wide


def intervention_summary(preds, species=None, modes=MODES):
    """Mean per-species accuracy for each model x split x mode, plus how much
    shallow->deep degradation is left under each mode.

    Mean per-species accuracy (not pooled over crops), to match how the rest
    of the notebook reports accuracy.
    """
    wide = intervention_per_species(preds, species)
    rows = []
    for model in preds["model"].unique():
        for mode in modes:
            sh, dp = f"{model}_shallow_{mode}", f"{model}_deep_{mode}"
            if sh not in wide or dp not in wide:
                continue
            both = wide[[sh, dp]].dropna()
            rows.append({"model": model, "mode": MODE_LABELS.get(mode, mode),
                         "shallow_acc": both[sh].mean(), "deep_acc": both[dp].mean(),
                         "degradation": (both[sh] - both[dp]).mean(),
                         "n_species": len(both)})
    return pd.DataFrame(rows).set_index(["model", "mode"])


def background_dependence(preds, crs_scores, model="shallow", species=None):
    """Does removing the background hurt the species Grad-CAM says rely on it?

    Per species: drop = accuracy(original) - accuracy(background greyed), on
    shallow test crops. If CRS measures real reliance, high-CRS species should
    show bigger drops (positive Spearman rho). Also reported for blur.

    Returns:
        (per-species DataFrame, dict mode -> (rho, p, n))
    """
    wide = intervention_per_species(preds, species)
    df = pd.DataFrame({"crs": pd.Series(crs_scores)}).reindex(wide.index)
    stats = {}
    for mode in ("grey_bg", "blur_bg"):
        col = f"drop_{mode}"
        df[col] = wide[f"{model}_shallow_none"] - wide[f"{model}_shallow_{mode}"]
        ok = df[["crs", col]].dropna()
        rho, p = spearmanr(ok["crs"], ok[col])
        stats[mode] = (float(rho), float(p), len(ok))
        print(f"CRS vs accuracy drop ({MODE_LABELS[mode]}): "
              f"rho = {rho:.3f}, p = {p:.3g}, n = {len(ok)} species")
    return df, stats


def bootstrap_intervention(preds, species, n_boot=2000, seed=config.RANDOM_SEED,
                           modes=("grey_bg", "blur_bg")):
    """95% bootstrap confidence intervals for the intervention effects.

    Species are resampled with replacement n_boot times (species are the unit
    of analysis everywhere else, so they are here too). For each model and mode:

      degradation:  mean(shallow acc - deep acc) under the edit.
      reduction:    how much the edit shrinks degradation vs the original
                    images (positive = less degradation once edited).
      retention_gap: (deep acc kept) - (shallow acc kept), where "kept" =
                    accuracy under the edit / original accuracy. Positive =
                    the edit hurt shallow crops proportionally more, i.e. the
                    background was helping more in shallow water. This one is
                    not distorted by the accuracy floor, unlike reduction.

    Plus a 'shallow - control' row per mode: the main model's retention_gap
    minus the control's. If its interval is above 0, the shallow-trained model
    used background more at shallow depths than a model that saw all depths.

    Returns:
        DataFrame with columns model, mode, metric, estimate, ci_low, ci_high,
        and significant (interval excludes 0).
    """
    wide = intervention_per_species(preds, species).dropna()
    models = [m for m in ("shallow", "control") if f"{m}_shallow_none" in wide]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(wide), size=(n_boot, len(wide)))
    full = np.arange(len(wide))[None, :]

    def means(col, ix):
        return wide[col].values[ix].mean(axis=1)

    def metrics(model, mode, ix):
        sh0, dp0 = means(f"{model}_shallow_none", ix), means(f"{model}_deep_none", ix)
        sh, dp = means(f"{model}_shallow_{mode}", ix), means(f"{model}_deep_{mode}", ix)
        return {"degradation": sh - dp, "reduction": (sh0 - dp0) - (sh - dp),
                "retention_gap": dp / dp0 - sh / sh0}

    rows = []

    def add(model, mode, metric, est, boot):
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"model": model, "mode": MODE_LABELS.get(mode, mode), "metric": metric,
                     "estimate": float(est[0]), "ci_low": lo, "ci_high": hi,
                     "significant": bool(lo > 0 or hi < 0)})

    for mode in modes:
        per_model = {}
        for model in models:
            est, boot = metrics(model, mode, full), metrics(model, mode, idx)
            per_model[model] = (est, boot)
            for metric in ("degradation", "reduction", "retention_gap"):
                add(model, mode, metric, est[metric], boot[metric])
        if len(models) == 2:
            (e1, b1), (e2, b2) = per_model["shallow"], per_model["control"]
            add("shallow - control", mode, "retention_gap",
                e1["retention_gap"] - e2["retention_gap"],
                b1["retention_gap"] - b2["retention_gap"])
    print(f"Bootstrap over {len(wide)} species, {n_boot} resamples")
    return pd.DataFrame(rows)


def plot_intervention_summary(summary, save_path=None):
    """One panel per model: mean per-species accuracy under each mode,
    shallow vs deep test crops."""
    models = summary.index.get_level_values("model").unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5),
                             sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        s = summary.loc[model]
        x = np.arange(len(s))
        w = 0.36
        for offset, split in ((-w / 2, "shallow"), (w / 2, "deep")):
            vals = s[f"{split}_acc"].values
            bars = ax.bar(x + offset, vals, width=w - 0.04, color=SPLIT_COLORS[split],
                          label=f"{split.capitalize()} test crops")
            ax.bar_label(bars, fmt="%.2f", fontsize=8, color=INK_MUTED, padding=2)
        ax.set_xticks(x, s.index, rotation=15)
        ax.set_title(f"{model.capitalize()} model", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel("Mean per-species accuracy")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("Accuracy when background or organism is removed", fontsize=12, y=1.04)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.99))
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
    return fig


def show_interventions(mask_loader, idxs=(0,), dilate_px=3, blur_sigma=10.0):
    """Visual check: each crop under every mode, side by side."""
    ds = mask_loader.dataset
    mean = torch.tensor(data.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(data.IMAGENET_STD).view(1, 3, 1, 1)
    for i in idxs:
        img, label, mask = ds[i]
        fig, axes = plt.subplots(1, len(MODES), figsize=(4 * len(MODES), 4))
        for ax, mode in zip(axes, MODES):
            x = apply_intervention(img[None], mask[None], mode, dilate_px, blur_sigma)
            rgb = (x * std + mean).clamp(0, 1)[0].permute(1, 2, 0).numpy()
            ax.imshow(rgb)
            ax.set_title(MODE_LABELS[mode], fontsize=10)
            ax.axis("off")
        fig.suptitle(mask_loader.species_list[label], fontsize=11)
        plt.tight_layout()
        plt.show()


# ── 2. Background shift ──────────────────────────────────────────────

def colour_stats(mask_loader, dilate_px=3):
    """Mean colour and brightness of each crop's background and organism.

    Background = pixels outside the mask grown by dilate_px (so organism edge
    pixels don't leak in). Values are in [0, 1] RGB, after undoing the
    ImageNet normalisation. Luminance = 0.299 R + 0.587 G + 0.114 B.

    Returns:
        DataFrame, one row per crop: image_id, ann_id, species, label,
        bg_r, bg_g, bg_b, bg_lum, org_r, org_g, org_b, org_lum.
    """
    ds = mask_loader.dataset
    mean = torch.tensor(data.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(data.IMAGENET_STD).view(1, 3, 1, 1)
    lum_w = torch.tensor([0.299, 0.587, 0.114])
    bg_all, org_all = [], []
    for images, labels, masks in mask_loader:
        rgb = (images * std + mean).clamp(0, 1)                      # (B, 3, H, W)
        org = masks.float()[:, None]
        bg = 1 - dilate_masks(masks, dilate_px)[:, None]
        for region, store in ((bg, bg_all), (org, org_all)):
            n = region.sum(dim=(2, 3))                                # (B, 1)
            m = (rgb * region).sum(dim=(2, 3)) / n.clamp_min(1)       # (B, 3)
            m[n[:, 0] == 0] = float("nan")                            # empty region -> nan
            store.append(m)
    bg_all, org_all = torch.cat(bg_all), torch.cat(org_all)
    out = pd.DataFrame({"image_id": ds.image_ids, "ann_id": ds.ann_ids,
                        "species": [mask_loader.species_list[l] for l in ds.labels]})
    for prefix, t in (("bg", bg_all), ("org", org_all)):
        for j, c in enumerate("rgb"):
            out[f"{prefix}_{c}"] = t[:, j].numpy()
        out[f"{prefix}_lum"] = (t @ lum_w).numpy()
    return out


def species_shift(shallow_stats, deep_stats):
    """Per-species change in background (and organism) appearance, shallow -> deep.

    colour_shift: distance between the species' mean shallow and mean deep
        colour (Euclidean in RGB, 0-1 scale; max possible ~1.73).
    lum_shift: deep minus shallow mean brightness (negative = darker at depth).

    The organism columns are a control: lighting changes at depth affect the
    organism too, so a degradation link to background shift only means
    something if it isn't equally explained by organism shift.
    """
    cols = [f"{p}_{c}" for p in ("bg", "org") for c in ("r", "g", "b", "lum")]
    sh = shallow_stats.groupby("species")[cols].mean()
    dp = deep_stats.groupby("species")[cols].mean()
    sh, dp = sh.align(dp, join="inner")
    out = pd.DataFrame(index=sh.index)
    for p in ("bg", "org"):
        rgb = [f"{p}_r", f"{p}_g", f"{p}_b"]
        out[f"{p}_colour_shift"] = np.sqrt(((dp[rgb] - sh[rgb]) ** 2).sum(axis=1))
        out[f"{p}_lum_shift"] = dp[f"{p}_lum"] - sh[f"{p}_lum"]
    return out


def shift_analysis(per_species_df, species=None):
    """Does background shift predict degradation, alone and combined with CRS?

    Needs per_species_df columns: degradation, crs, bg_colour_shift,
    org_colour_shift.

    Reports:
      - Spearman of degradation with background shift and organism shift.
      - 2x2 table: mean degradation for low/high CRS x low/high background
        shift (median splits). The hypothesis predicts the high/high cell
        degrades most.
      - Regression on rank-standardised predictors:
        degradation ~ crs * bg_shift + org_shift. The crs:bg_shift term is the
        "both high together" effect.

    Returns:
        (spearman DataFrame, 2x2 DataFrame, fitted statsmodels result)
    """
    import statsmodels.formula.api as smf

    df = per_species_df if species is None else per_species_df.reindex(species)
    df = df[["degradation", "crs", "bg_colour_shift", "org_colour_shift"]].dropna()
    print(f"Shift analysis over {len(df)} species")

    rows = []
    for col in ("bg_colour_shift", "org_colour_shift", "crs"):
        rho, p = spearmanr(df[col], df["degradation"])
        rows.append({"predictor": col, "spearman_rho": rho, "p_value": p})
    spear = pd.DataFrame(rows).set_index("predictor")

    crs_hi = np.where(df["crs"] > df["crs"].median(), "high CRS", "low CRS")
    shift_hi = np.where(df["bg_colour_shift"] > df["bg_colour_shift"].median(),
                        "high bg shift", "low bg shift")
    grid = (df.assign(crs_group=crs_hi, shift_group=shift_hi)
              .groupby(["crs_group", "shift_group"])["degradation"]
              .agg(["mean", "count"]).unstack("shift_group"))

    z = lambda s: (s.rank() - s.rank().mean()) / s.rank().std()
    reg_df = pd.DataFrame({"degradation": df["degradation"], "crs": z(df["crs"]),
                           "bg_shift": z(df["bg_colour_shift"]),
                           "org_shift": z(df["org_colour_shift"])})
    fit = smf.ols("degradation ~ crs * bg_shift + org_shift", data=reg_df).fit()
    return spear, grid, fit


# ── 3. Per-image CRS: correct vs incorrect ───────────────────────────

def compute_image_crs(model, mask_loader, target_layer=config.GRADCAM_TARGET_LAYER):
    """Grad-CAM CRS for every crop, plus whether the model got it right.

    Explained for the true species (same as compute_species_crs), so a wrong
    prediction's CRS says how much of the evidence for the right species
    came from background.

    Returns:
        DataFrame, one row per crop: image_id, ann_id, species, pred, correct, crs.
    """
    device = _device()
    model = model.to(device).eval()
    labels_all, preds_all, crs_all = [], [], []
    for images, labels, masks in mask_loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16,
                                             enabled=device.type == "cuda"):
            preds = model(images).argmax(dim=1)
        cams = _gradcam_batch(model, images, labels, target_layer)
        total = cams.sum(dim=(1, 2))
        outside = (cams * (masks == 0)).sum(dim=(1, 2))
        crs = torch.where(total > 0, outside / total.clamp_min(1e-12),
                          torch.full_like(total, float("nan")))
        labels_all.append(labels.cpu())
        preds_all.append(preds.cpu())
        crs_all.append(crs.float().cpu())

    ds, sl = mask_loader.dataset, mask_loader.species_list
    labels, preds = torch.cat(labels_all).tolist(), torch.cat(preds_all).tolist()
    return pd.DataFrame({"image_id": ds.image_ids, "ann_id": ds.ann_ids,
                         "species": [sl[i] for i in labels], "pred": [sl[i] for i in preds],
                         "correct": [a == b for a, b in zip(labels, preds)],
                         "crs": torch.cat(crs_all).double().numpy()})


def crs_correct_vs_incorrect(image_crs, species=None, min_each=3, label=""):
    """Is CRS higher on wrong predictions than right ones?

    Two tests:
      - Pooled: all correct crops vs all incorrect crops (Mann-Whitney U).
        Simple, but confounded: hard species may just have higher CRS overall.
      - Within species (the fair one): for each species with >= min_each
        correct AND >= min_each incorrect crops, mean CRS(incorrect) minus
        mean CRS(correct). Wilcoxon signed-rank test on those differences.

    Returns:
        dict of results; also prints a short summary.
    """
    df = image_crs.dropna(subset=["crs"])
    if species is not None:
        df = df[df["species"].isin(species)]
    right, wrong = df.loc[df["correct"], "crs"], df.loc[~df["correct"], "crs"]
    _, p_pooled = mannwhitneyu(wrong, right, alternative="two-sided")

    g = df.groupby(["species", "correct"])["crs"].agg(["mean", "count"]).unstack("correct")
    ok = (g[("count", True)] >= min_each) & (g[("count", False)] >= min_each)
    diffs = (g.loc[ok, ("mean", False)] - g.loc[ok, ("mean", True)]).dropna()
    p_within = wilcoxon(diffs).pvalue if len(diffs) >= 5 else float("nan")

    res = {"split": label, "n_correct": len(right), "n_incorrect": len(wrong),
           "median_crs_correct": right.median(), "median_crs_incorrect": wrong.median(),
           "p_pooled": p_pooled, "n_species_within": len(diffs),
           "median_within_species_diff": diffs.median() if len(diffs) else float("nan"),
           "p_within": p_within}
    print(f"[{label}] median CRS correct {res['median_crs_correct']:.3f} vs incorrect "
          f"{res['median_crs_incorrect']:.3f} (pooled p = {p_pooled:.3g}); within "
          f"{len(diffs)} species, median diff = {res['median_within_species_diff']:.3f} "
          f"(p = {p_within:.3g})")
    return res


def plot_crs_correct_vs_incorrect(image_crs_by_split, species=None, save_path=None):
    """Box plot of per-image CRS, correct vs incorrect, for each split.

    Args:
        image_crs_by_split: dict split -> compute_image_crs DataFrame.
    """
    groups, labels, colors = [], [], []
    for split, df in image_crs_by_split.items():
        df = df.dropna(subset=["crs"])
        if species is not None:
            df = df[df["species"].isin(species)]
        for correct, word in ((True, "correct"), (False, "wrong")):
            v = df.loc[df["correct"] == correct, "crs"].values
            groups.append(v)
            labels.append(f"{split.capitalize()}\n{word}\n(n={len(v)})")
            colors.append(SPLIT_COLORS.get(split, INK_MUTED))

    fig, ax = plt.subplots(figsize=(1.8 * len(groups) + 1, 4.8))
    bp = ax.boxplot(groups, widths=0.55, showfliers=False, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set(facecolor=c, edgecolor=c)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set_color(INK_MUTED)
    ax.set_xticks(range(1, len(groups) + 1), labels)
    ax.set_ylabel("Per-image CRS (true species)")
    ax.set_ylim(0, 1)
    ax.set_title("Context reliance on correct vs wrong predictions", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
    return fig


# ── 4. Within-species CRS variance ─────────────────────────────────────

def crs_variance_analysis(image_crs_df, degradation_scores, species=None, min_images=5):
    """Test whether within-species CRS variance predicts degradation.

    Species where CRS is highly variable (some images organism-focused,
    others background-reliant) may be more vulnerable to depth shift
    than species with consistently moderate CRS.

    Args:
        image_crs_df: compute_image_crs() DataFrame (shallow split).
        degradation_scores: dict species -> degradation.
        species: optional list to restrict to.
        min_images: minimum scored images to compute a reliable std.

    Returns:
        (per-species DataFrame with crs mean/std/iqr/degradation,
         dict metric -> (rho, p, n))
    """
    df = image_crs_df.dropna(subset=["crs"])
    if species is not None:
        df = df[df["species"].isin(species)]

    grouped = df.groupby("species")["crs"]
    stats = grouped.agg(["mean", "std", "count"])
    stats["iqr"] = grouped.quantile(0.75) - grouped.quantile(0.25)
    stats = stats[stats["count"] >= min_images]

    deg = pd.Series(degradation_scores).reindex(stats.index).dropna()
    stats = stats.reindex(deg.index).dropna()
    stats["degradation"] = deg

    results = {}
    for metric in ("std", "iqr", "mean"):
        rho, p = spearmanr(stats[metric], stats["degradation"])
        results[metric] = (float(rho), float(p), len(stats))
        print(f"CRS {metric} vs degradation: rho = {rho:.3f}, p = {p:.3g}, n = {len(stats)} species")

    return stats, results


def plot_crs_variance_vs_degradation(var_df, metric="std", rho=None, p_value=None,
                                     n_labels=5, save_path=None):
    """Scatter: per-species CRS variance (or IQR) vs degradation."""
    x, y = var_df[metric], var_df["degradation"]
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.axhline(0, color="#bbbbbb", linewidth=1, zorder=0)
    ax.scatter(x, y, s=28, color=SPLIT_COLORS["shallow"], alpha=0.7,
               edgecolor="white", linewidth=0.8)

    z = ((x - x.mean()) / x.std()) ** 2 + ((y - y.mean()) / y.std()) ** 2
    for s in z.sort_values(ascending=False).index[:n_labels]:
        ax.annotate(s, (x[s], y[s]), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color=INK_MUTED)

    metric_label = {"std": "CRS standard deviation", "iqr": "CRS interquartile range",
                    "mean": "CRS mean"}[metric]
    title = "Within-species CRS variability vs depth degradation"
    if rho is not None:
        title += f"\nSpearman ρ = {rho:.3f}, p = {p_value:.3g}, n = {len(x)} species"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"{metric_label} (shallow test, Grad-CAM)")
    ax.set_ylabel("Degradation (shallow acc − deep acc)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
    return fig
