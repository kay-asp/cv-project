"""
Grad-CAM attribution and Context Reliance Score (CRS) computation.

Responsibilities:
  - Compute Grad-CAM heatmaps on the final convolutional layer
  - Compute per-image CRS (fraction of attribution outside organism mask)
  - Aggregate CRS per species
  - Generate baseline attribution maps (uniform random, centre-Gaussian)
"""

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
        dataset: Shallow-depth test dataset with masks.
        species_list: List of species to evaluate.

    Returns:
        dict mapping species -> mean CRS
    """
    raise NotImplementedError


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
