"""
Grad-CAM attribution and Context Reliance Score (CRS) computation.

Responsibilities:
  - Compute Grad-CAM heatmaps on the final convolutional layer
  - Compute per-image CRS (fraction of attribution outside organism mask)
  - Aggregate CRS per species
  - Generate baseline attribution maps (uniform random, centre-Gaussian)
"""

import numpy as np
import torch
import torch.nn.functional as F

import config


def _gradcam_batch(model, images, target_classes, target_layer=config.GRADCAM_TARGET_LAYER):
    """Grad-CAM for a batch of images, one target class per image.

    Args:
        model: Trained classifier (already on the same device as images).
        images: Preprocessed input tensor (B, 3, H, W).
        target_classes: Long tensor (B,) of class indices to explain.
        target_layer: Name of the convolutional layer to hook.

    Returns:
        Tensor (B, H, W) with each heatmap min-max normalised to [0, 1]
        (a heatmap that is all zero stays all zero).
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
    cam = F.relu((weights * a).sum(dim=1, keepdim=True))    # (B, 1, h, w)
    cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]

    lo = cam.amin(dim=(1, 2), keepdim=True)
    hi = cam.amax(dim=(1, 2), keepdim=True)
    span = hi - lo
    return torch.where(span > 0, (cam - lo) / span.clamp_min(1e-12), torch.zeros_like(cam))


def compute_gradcam(model, image, target_class, target_layer=config.GRADCAM_TARGET_LAYER):
    """Compute Grad-CAM heatmap for a given image and class.

    Args:
        model: Trained classifier.
        image: Preprocessed input tensor, (3, H, W) or (1, 3, H, W).
        target_class: Class index to explain.
        target_layer: Name of the convolutional layer to hook.

    Returns:
        heatmap: numpy array (H, W) with attribution values in [0, 1].
    """
    device = next(model.parameters()).device
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    target = torch.tensor([int(target_class)], device=device)
    return _gradcam_batch(model, image, target, target_layer)[0].cpu().numpy()


def compute_crs(heatmap, mask):
    """Compute Context Reliance Score for a single image.

    CRS = (attribution mass outside mask) / (total attribution mass)

    Args:
        heatmap: Grad-CAM heatmap (H, W).
        mask: Binary segmentation mask (H, W), 1 = organism.

    Returns:
        float in [0, 1], or nan if the heatmap has no attribution mass
        (so blank maps are left out of species means instead of counting as 0).
    """
    heatmap = heatmap.detach().cpu().numpy() if torch.is_tensor(heatmap) else np.asarray(heatmap)
    mask = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    if heatmap.shape != mask.shape:
        raise ValueError(f"heatmap shape {heatmap.shape} doesn't match mask shape {mask.shape}")

    heatmap = heatmap.astype(np.float64)
    total = heatmap.sum()
    if total <= 0:
        return float("nan")
    return float((heatmap * (mask == 0)).sum() / total)


def compute_species_crs(model, dataloader, species_list, target_layer=config.GRADCAM_TARGET_LAYER):
    """Average CRS over all shallow-depth images for each species.

    Each image is explained for its true label: the CRS measures how much of
    the evidence for the correct species comes from outside the organism.

    Args:
        model: Trained classifier.
        dataloader: Shallow-depth test DataLoader with masks, i.e. built with
            data.get_dataloaders(..., coco=coco), yielding (images, labels, masks).
        species_list: List of species to evaluate (label index = position).
        target_layer: Name of the convolutional layer to hook.

    Returns:
        dict mapping species -> mean CRS. A species with no scored images
        maps to float('nan').
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_labels, all_crs = [], []
    for images, labels, masks in dataloader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        cams = _gradcam_batch(model, images, labels, target_layer)
        total = cams.sum(dim=(1, 2))
        outside = (cams * (masks == 0)).sum(dim=(1, 2))
        crs = torch.where(total > 0, outside / total.clamp_min(1e-12),
                          torch.full_like(total, float("nan")))

        all_labels.append(labels.cpu())
        all_crs.append(crs.cpu())

    labels = torch.cat(all_labels).numpy()
    crs = torch.cat(all_crs).double().numpy()
    n_scored = int(np.isfinite(crs).sum())
    print(f"CRS: scored {n_scored} / {len(crs)} images "
          f"({len(crs) - n_scored} had an empty Grad-CAM map)")

    result = {}
    for idx, species in enumerate(species_list):
        vals = crs[(labels == idx) & np.isfinite(crs)]
        result[species] = float(vals.mean()) if len(vals) else float("nan")
    return result


def generate_uniform_baseline(shape, rng=None):
    """Uniform random attribution map (baseline for CRS comparison).
    
    A map that "looks everywhere equally". Its CRS is roughly the fraction of
    the crop that is background, so it tells us what CRS a meaningless
    attribution gets purely from how big the organism is in the crop.

    Returns:
        numpy array of given shape with uniform random values.
    """
    rng = rng if rng is not None else np.random.default_rng()
    return rng.random(shape)
    


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
    
def compute_species_baseline_crs(mask_loader, species_list, kind, seed=config.RANDOM_SEED,
                                 sigma=None):
    """Per-species CRS of a trivial baseline map, on the same images and masks
    as compute_species_crs, so the numbers are directly comparable.
 
    Args:
        mask_loader: Same shallow-test loader passed to compute_species_crs
            (built with data.get_dataloaders(..., coco=coco)).
        species_list: Class order (label index = position).
        kind: 'uniform' or 'gaussian'.
        seed: Random seed for the uniform maps.
        sigma: Gaussian width in pixels (see generate_gaussian_baseline).
 
    Returns:
        dict mapping species -> mean baseline CRS (nan if no images).
    """
    import data  # local import: data imports config/torch, avoid circular imports
 
    if kind not in ("uniform", "gaussian"):
        raise ValueError(f"kind must be 'uniform' or 'gaussian', got {kind!r}")
    ds = mask_loader.dataset
    rng = np.random.default_rng(seed)
    gauss = generate_gaussian_baseline(config.IMG_SIZE, sigma) if kind == "gaussian" else None
 
    crs = np.empty(len(ds.ann_ids))
    for i, (image_id, ann_id, bbox) in enumerate(zip(ds.image_ids, ds.ann_ids, ds.bboxes)):
        mask = data.get_crop_mask(ds.coco, image_id, ann_id, bbox)
        heatmap = gauss if kind == "gaussian" else generate_uniform_baseline(mask.shape, rng)
        crs[i] = compute_crs(heatmap, mask)
 
    labels = np.asarray(ds.labels)
    result = {}
    for idx, species in enumerate(species_list):
        vals = crs[(labels == idx) & np.isfinite(crs)]
        result[species] = float(vals.mean()) if len(vals) else float("nan")
    print(f"{kind} baseline CRS: scored {int(np.isfinite(crs).sum())} / {len(crs)} images")
    return result
