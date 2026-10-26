"""
Grad-CAM attribution and Context Reliance Score (CRS) computation.

Responsibilities:
  - Compute Grad-CAM heatmaps on the final convolutional layer
  - Compute per-image CRS (fraction of attribution outside organism mask)
  - Aggregate CRS per species
  - Generate baseline attribution maps (uniform, centre-Gaussian)

How CRS is scored:
  - On the raw Grad-CAM map (ReLU output, not min-max normalised): CRS is a
    ratio, and subtracting each map's minimum would change it.
  - At the heatmap's own resolution (7x7 for layer4): the mask is area-pooled
    down to the heatmap grid, rather than the heatmap being upsampled to the
    mask, so coarse 32 px cells don't smear onto the background around small
    organisms.
  - Against the organism mask grown by config.CRS_DILATE_PX, the same margin
    the background interventions use.
  - Each score comes with its chance level, the CRS of a flat map on the
    same mask (= 1 - organism coverage), and crs_excess = crs - chance.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config


def _gradcam_batch(model, images, target_classes, target_layer=config.GRADCAM_TARGET_LAYER,
                   raw=False, return_logits=False):
    """Grad-CAM for a batch of images, one target class per image.

    Args:
        model: Trained classifier (already on the same device as images).
        images: Preprocessed input tensor (B, 3, H, W).
        target_classes: Long tensor (B,) of class indices to explain.
        target_layer: Name of the convolutional layer to hook.
        raw: If True, return the un-normalised ReLU map at the layer's own
            resolution (B, h, w), which is what CRS is computed on.
        return_logits: If True, also return the (B, num_classes) fp32 logits
            of the same forward pass, so predictions need no second pass.

    Returns:
        Tensor (B, H, W) with each heatmap min-max normalised to [0, 1]
        (a heatmap that is all zero stays all zero), or (B, h, w) if raw;
        a (heatmap, logits) tuple if return_logits.
    """
    layer = dict(model.named_modules())[target_layer]
    acts = {}
    def hook(module, inputs, out):
        # Start the autograd graph here: with parameter grads switched off, the
        # layers before target_layer save nothing for backward, which keeps
        # memory close to a plain inference pass
        out = out.detach().requires_grad_(True)
        acts["a"] = out
        return out

    handle = layer.register_forward_hook(hook)
    requires_grad = [p.requires_grad for p in model.parameters()]
    model.requires_grad_(False)
    try:
        model.eval()
        # fp32 (no autocast) so the gradients are numerically stable
        with torch.enable_grad():
            logits = model(images)
            score = logits.gather(1, target_classes[:, None]).sum()
            # Gradients w.r.t. the activations only, so no parameter .grad builds up.
            # In eval mode each sample's logit depends only on its own activations,
            # so summing the scores gives every image its own gradient.
            grads, = torch.autograd.grad(score, acts["a"])
    finally:
        handle.remove()
        for p, rg in zip(model.parameters(), requires_grad):
            p.requires_grad_(rg)

    a = acts["a"].detach()
    weights = grads.mean(dim=(2, 3), keepdim=True)          # (B, C, 1, 1)
    cam = F.relu((weights * a).sum(dim=1))                  # (B, h, w)
    if not raw:
        cam = F.interpolate(cam[:, None], size=images.shape[-2:], mode="bilinear",
                            align_corners=False)[:, 0]
        lo = cam.amin(dim=(1, 2), keepdim=True)
        hi = cam.amax(dim=(1, 2), keepdim=True)
        span = hi - lo
        cam = torch.where(span > 0, (cam - lo) / span.clamp_min(1e-12), torch.zeros_like(cam))
    return (cam, logits.detach()) if return_logits else cam


def compute_gradcam(model, image, target_class, target_layer=config.GRADCAM_TARGET_LAYER,
                    raw=False):
    """Compute Grad-CAM heatmap for a given image and class.

    Args:
        model: Trained classifier.
        image: Preprocessed input tensor, (3, H, W) or (1, 3, H, W).
        target_class: Class index to explain.
        target_layer: Name of the convolutional layer to hook.
        raw: If True, return the un-normalised (h, w) map for compute_crs;
            otherwise an (H, W) map in [0, 1] for display.

    Returns:
        heatmap: numpy array, (H, W) in [0, 1], or (h, w) if raw.
    """
    device = next(model.parameters()).device
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    target = torch.tensor([int(target_class)], device=device)
    return _gradcam_batch(model, image, target, target_layer, raw=raw)[0].cpu().numpy()


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


def crs_from_cams(cams, masks, dilate_px=config.CRS_DILATE_PX):
    """CRS and its chance level for a batch, scored on the heatmap's own grid.

    The mask is dilated, then area-pooled onto the heatmap grid, giving the
    organism fraction of each cell; a cell's attribution counts as context in
    proportion to the part of it that is background.

    Args:
        cams: (B, h, w) non-negative heatmaps, un-normalised (raw=True Grad-CAM).
        masks: (B, H, W) organism masks, 1 = organism, with H >= h and W >= w.
        dilate_px: Pixels to grow the mask by first.

    Returns:
        (crs, chance), float64 tensors of shape (B,). crs is nan where the
        heatmap is all zero (so blank maps are left out of species means
        instead of counting as 0). chance is the CRS a flat heatmap gets on
        the same mask, i.e. 1 - dilated organism coverage.
    """
    masks = masks.to(cams.device)
    if masks.shape[-2] < cams.shape[-2] or masks.shape[-1] < cams.shape[-1]:
        raise ValueError(f"mask shape {tuple(masks.shape[-2:])} is smaller than "
                         f"heatmap shape {tuple(cams.shape[-2:])}")
    organism = F.adaptive_avg_pool2d(dilate_masks(masks, dilate_px)[:, None],
                                     cams.shape[-2:])[:, 0].double()
    cams = cams.double()
    total = cams.sum(dim=(1, 2))
    outside = (cams * (1 - organism)).sum(dim=(1, 2))
    crs = torch.where(total > 0, outside / total.clamp_min(1e-12),
                      torch.full_like(total, float("nan")))
    chance = 1 - organism.mean(dim=(1, 2))
    return crs, chance


def compute_crs(heatmap, mask, dilate_px=config.CRS_DILATE_PX):
    """Compute Context Reliance Score for a single image.

    CRS = (attribution mass outside mask) / (total attribution mass),
    scored on the heatmap's grid (see crs_from_cams).

    Args:
        heatmap: Raw Grad-CAM heatmap (h, w), e.g. compute_gradcam(..., raw=True).
        mask: Binary segmentation mask (H, W), 1 = organism.
        dilate_px: Pixels to grow the mask by first.

    Returns:
        float in [0, 1], or nan if the heatmap has no attribution mass.
    """
    heatmap = torch.as_tensor(np.asarray(heatmap) if not torch.is_tensor(heatmap) else heatmap)
    mask = torch.as_tensor(np.asarray(mask) if not torch.is_tensor(mask) else mask)
    crs, _ = crs_from_cams(heatmap[None].cpu(), mask[None].cpu(), dilate_px)
    return float(crs[0])


def compute_image_crs(model, mask_loader, target_layer=config.GRADCAM_TARGET_LAYER,
                      dilate_px=config.CRS_DILATE_PX):
    """Grad-CAM CRS for every crop, plus whether the model got it right.

    The one place CRS is computed: species CRS (species_crs_table) is
    aggregated from this table, so per-species and per-image numbers always
    agree. Each crop is explained for its true species, so a wrong
    prediction's CRS says how much of the evidence for the right species came
    from background. The prediction comes from the same fp32 forward pass.

    crs_pred explains the predicted species instead: how much of what the
    model actually decided on came from background. It equals crs on correct
    crops; on wrong crops it takes a second Grad-CAM pass.

    Args:
        model: Trained classifier.
        mask_loader: Loader built with data.get_dataloaders(..., coco=coco); its
            mask_mode decides which annotations count as organism.
        target_layer: Name of the convolutional layer to hook.
        dilate_px: Pixels to grow the masks by before scoring.

    Returns:
        DataFrame, one row per crop: image_id, ann_id, species, pred, correct,
        crs (nan for an empty Grad-CAM map), crs_chance (flat-map CRS,
        = 1 - dilated coverage), crs_excess, crs_pred, crs_pred_excess.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    labels_all, preds_all, crs_all, chance_all, crs_pred_all = [], [], [], [], []
    for images, labels, masks in mask_loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        cams, logits = _gradcam_batch(model, images, labels, target_layer, raw=True,
                                      return_logits=True)
        crs, chance = crs_from_cams(cams, masks, dilate_px)
        preds = logits.argmax(dim=1)

        # Predicted-class CRS: same as crs where the prediction is right, re-explained where it's wrong
        crs_pred = crs.clone()
        wrong = preds != labels
        if wrong.any():
            cams_wrong = _gradcam_batch(model, images[wrong], preds[wrong], target_layer, raw=True)
            crs_pred[wrong] = crs_from_cams(cams_wrong, masks[wrong], dilate_px)[0]

        labels_all.append(labels.cpu())
        preds_all.append(preds.cpu())
        crs_all.append(crs.cpu())
        chance_all.append(chance.cpu())
        crs_pred_all.append(crs_pred.cpu())

    ds, sl = mask_loader.dataset, mask_loader.species_list
    labels, preds = torch.cat(labels_all).tolist(), torch.cat(preds_all).tolist()
    crs, chance = torch.cat(crs_all).numpy(), torch.cat(chance_all).numpy()
    crs_pred = torch.cat(crs_pred_all).numpy()
    df = pd.DataFrame({"image_id": ds.image_ids, "ann_id": ds.ann_ids,
                       "species": [sl[i] for i in labels], "pred": [sl[i] for i in preds],
                       "correct": [a == b for a, b in zip(labels, preds)],
                       "crs": crs, "crs_chance": chance, "crs_excess": crs - chance,
                       "crs_pred": crs_pred, "crs_pred_excess": crs_pred - chance})

    # Empty maps are dropped from every CRS analysis; say whether that hits wrong crops harder
    empty = df["crs"].isna()
    n_right, n_wrong = int(df["correct"].sum()), int((~df["correct"]).sum())
    print(f"CRS: scored {int((~empty).sum())} / {len(df)} crops. Empty Grad-CAM maps: "
          f"{int((empty & df['correct']).sum())} / {n_right} correct, "
          f"{int((empty & ~df['correct']).sum())} / {n_wrong} wrong; "
          f"{int(df['crs_pred'].isna().sum())} empty predicted-class maps")
    return df


def _bootstrap_means(values, n_boot, rng):
    """n_boot bootstrap means of a 1-D array (resampling with replacement)."""
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return values[idx].mean(axis=1)


def species_crs_table(image_crs, species_list, prefix="", n_boot=2000, seed=config.RANDOM_SEED):
    """Per-species CRS from a compute_image_crs table (empty maps left out).

    Args:
        image_crs: compute_image_crs() DataFrame.
        species_list: Species to report (index order of the result).
        prefix: Added to every column name, e.g. 'shallow_' or 'deep_' so the
            two splits can sit side by side in results.csv.
        n_boot: Bootstrap resamples (over each species' crops) for the CI.
        seed: Random seed for the bootstrap.

    Returns:
        DataFrame indexed by species with columns crs (mean), crs_ci_low /
        crs_ci_high (95% percentile bootstrap CI of the mean; nan with fewer
        than 2 crops), crs_median, crs_chance (mean flat-map CRS, = 1 - mean
        dilated coverage), crs_excess (mean of crs - chance), crs_pred and
        crs_pred_excess (the same for the predicted-class CRS, still grouped
        by true species; only if image_crs has crs_pred) and n_crs (crops
        scored), each with prefix in front. A species with no scored crops
        has nan values.
    """
    df = image_crs.dropna(subset=["crs"])
    g = df.groupby("species")
    out = g[["crs", "crs_chance", "crs_excess"]].mean()
    out["crs_median"] = g["crs"].median()
    out["n_crs"] = g.size()

    rng = np.random.default_rng(seed)
    ci = {}
    for species, vals in g["crs"]:
        if len(vals) >= 2:
            ci[species] = np.percentile(_bootstrap_means(vals.to_numpy(), n_boot, rng), [2.5, 97.5])
    ci = pd.DataFrame(ci, index=["crs_ci_low", "crs_ci_high"]).T
    out = out.join(ci)
    cols = ["crs", "crs_ci_low", "crs_ci_high", "crs_median", "crs_chance", "crs_excess"]
    if "crs_pred" in image_crs.columns:
        # Its own empty maps are dropped separately (they needn't be the true-class ones)
        pred = image_crs.dropna(subset=["crs_pred"]).groupby("species")
        out = out.join(pred[["crs_pred", "crs_pred_excess"]].mean(), how="outer")
        cols += ["crs_pred", "crs_pred_excess"]
    out = out.reindex(species_list)
    out.index.name = "species"
    out["n_crs"] = out["n_crs"].fillna(0).astype(int)
    out = out[cols + ["n_crs"]]
    return out.add_prefix(prefix)


def compute_species_crs(model, dataloader, species_list, target_layer=config.GRADCAM_TARGET_LAYER,
                        dilate_px=config.CRS_DILATE_PX):
    """Average CRS over all shallow-depth images for each species.

    Shorthand for species_crs_table(compute_image_crs(...)); use the two
    steps directly when the per-image table is needed too.

    Args:
        model: Trained classifier.
        dataloader: Shallow-depth test DataLoader with masks, i.e. built with
            data.get_dataloaders(..., coco=coco), yielding (images, labels, masks).
        species_list: List of species to evaluate (label index = position).
        target_layer: Name of the convolutional layer to hook.
        dilate_px: Pixels to grow the masks by before scoring.

    Returns:
        species_crs_table DataFrame.
    """
    return species_crs_table(compute_image_crs(model, dataloader, target_layer, dilate_px),
                             species_list)


def generate_uniform_baseline(shape):
    """Flat attribution map (baseline for CRS comparison).

    A map that "looks everywhere equally". Its CRS is exactly the fraction of
    the crop that is background, so it tells us what CRS a meaningless
    attribution gets purely from how big the organism is in the crop. (A
    uniform random map has this as its expected value; the flat map gives it
    without sampling noise.)

    Returns:
        numpy array of ones with the given shape.
    """
    return np.ones(shape)


def generate_gaussian_baseline(shape, sigma=None):
    """Centre-Gaussian attribution map (captures centre-bias baseline).

    A map that only "looks at the middle". Because every crop is centred on
    its organism, this is what CRS looks like for a model that ignores the
    image content and just attends to the centre.

    Returns:
        numpy array of given shape with Gaussian blob centred on the image.
    """
    h, w = shape
    if sigma is None:
        sigma = 0.25 * min(h, w)
    y = np.arange(h) - (h - 1) / 2
    x = np.arange(w) - (w - 1) / 2
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))


def compute_species_baseline_crs(mask_loader, species_list, kind, sigma=None,
                                 dilate_px=config.CRS_DILATE_PX, batch_size=512):
    """Per-species CRS of a trivial baseline map, on the same images and masks
    as compute_species_crs, so the numbers are directly comparable.

    The baseline map is built at IMG_SIZE and area-pooled to GRADCAM_SIZE, so
    it is scored on the same grid (and dilated masks) as Grad-CAM.

    Args:
        mask_loader: Same shallow-test loader passed to compute_species_crs
            (built with data.get_dataloaders(..., coco=coco)).
        species_list: Class order (label index = position).
        kind: 'uniform' or 'gaussian'.
        sigma: Gaussian width in pixels (see generate_gaussian_baseline).
        dilate_px: Pixels to grow the masks by before scoring.
        batch_size: Masks scored at once.

    Returns:
        dict mapping species -> mean baseline CRS (nan if no images).
    """
    if kind not in ("uniform", "gaussian"):
        raise ValueError(f"kind must be 'uniform' or 'gaussian', got {kind!r}")
    ds = mask_loader.dataset
    full = (generate_gaussian_baseline(config.IMG_SIZE, sigma) if kind == "gaussian"
            else generate_uniform_baseline(config.IMG_SIZE))
    cam = F.adaptive_avg_pool2d(torch.from_numpy(full)[None, None], config.GRADCAM_SIZE)[0]

    crs = np.empty(len(ds))
    for start in range(0, len(ds), batch_size):
        idx = range(start, min(start + batch_size, len(ds)))
        masks = torch.stack([torch.from_numpy(ds.get_mask(i)) for i in idx])
        crs[start:start + len(idx)] = crs_from_cams(cam.expand(len(idx), -1, -1), masks,
                                                    dilate_px)[0].numpy()

    labels = np.asarray(ds.labels)
    result = {}
    for idx, species in enumerate(species_list):
        vals = crs[(labels == idx) & np.isfinite(crs)]
        result[species] = float(vals.mean()) if len(vals) else float("nan")
    print(f"{kind} baseline CRS: scored {int(np.isfinite(crs).sum())} / {len(crs)} images")
    return result
