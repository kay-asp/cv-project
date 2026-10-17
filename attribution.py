"""
Grad-CAM attribution and Context Reliance Score (CRS) computation.

Responsibilities:
  - Compute Grad-CAM heatmaps on the final convolutional layer
  - Compute per-image CRS (fraction of attribution outside organism mask)
  - Aggregate CRS per species
  - Generate baseline attribution maps (uniform random, centre-Gaussian)
"""

from collections import defaultdict

import numpy as np
import torch

import config


def compute_gradcam(model, image, target_class, target_layer=config.GRADCAM_TARGET_LAYER):
    """Compute Grad-CAM heatmap for a given image and class.

    Args:
        model: Trained classifier.
        image: Preprocessed input tensor.
        target_class: Class index to explain.
        target_layer: Name of the convolutional layer to hook.

    Returns:
        heatmap: numpy array (H, W) with attribution values in [0, 1].
    """
    raise NotImplementedError


def compute_crs(heatmap, mask):
    """Compute Context Reliance Score for a single image.

    CRS = (attribution mass outside mask) / (total attribution mass)

    Args:
        heatmap: Grad-CAM heatmap (H, W).
        mask: Binary segmentation mask (H, W), 1 = organism.

    Returns:
        float in [0, 1]
    """
    raise NotImplementedError


def compute_species_crs(model, dataset, species_list):
    """Average CRS over all shallow-depth images for each species.

    Args:
        model: Trained classifier.
        dataset: Shallow-depth test dataset with masks, e.g.
            data.MaskedCropDataset. Yields (image_tensor, label, mask) and has
            a .species_list mapping label index -> species name (the model's
            class order).
        species_list: List of species to evaluate; images of other species
            are skipped.

    Returns:
        dict mapping species -> mean CRS. A species with no images in the
        dataset maps to float('nan').
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    wanted = set(species_list)
    scores = defaultdict(list)
    for i in range(len(dataset)):
        image, label, mask = dataset[i]
        label = int(label)
        species = dataset.species_list[label]
        if species not in wanted:
            continue
        # Explain the true class: CRS asks where the evidence for the
        # organism's own label comes from, right or wrong prediction.
        heatmap = compute_gradcam(model, image.unsqueeze(0).to(device), label)
        scores[species].append(compute_crs(heatmap, mask))

    return {species: float(np.mean(scores[species])) if scores[species] else float("nan")
            for species in species_list}


def generate_uniform_baseline(shape):
    """Uniform random attribution map (baseline for CRS comparison).

    Returns:
        numpy array of given shape with uniform random values.
    """
    raise NotImplementedError


def generate_gaussian_baseline(shape, sigma=None):
    """Centre-Gaussian attribution map (captures centre-bias baseline).

    Returns:
        numpy array of given shape with Gaussian blob centred on the image.
    """
    raise NotImplementedError
