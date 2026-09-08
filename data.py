"""
Data loading, cropping, and depth-based splitting for FathomNet imagery.

Responsibilities:
  - Download / load FathomNet images and metadata
  - Crop around bounding boxes with a habitat margin
  - Load and align segmentation masks
  - Split data into shallow (within training depth) and deep (beyond it)
  - Provide PyTorch Datasets / DataLoaders
"""

import config


def load_fathomnet_metadata():
    """Load image-level labels, bounding boxes, depth metadata, and mask paths.

    Returns:
        pd.DataFrame with columns: image_id, image_path, species, depth_m,
                                    bbox (x, y, w, h), mask_path
    """
    raise NotImplementedError


def crop_to_bounding_box(image, bbox, margin=config.CROP_MARGIN):
    """Crop image around a bounding box with a surrounding habitat margin.

    Args:
        image: PIL Image or numpy array.
        bbox: (x, y, w, h) bounding box of the organism.
        margin: Fractional margin to add on each side.

    Returns:
        Cropped image, adjusted mask (if provided).
    """
    raise NotImplementedError


def split_shallow_deep(metadata_df, depth_threshold):
    """Split dataset into shallow (≤ threshold) and deep (> threshold).

    Args:
        metadata_df: DataFrame with a 'depth_m' column.
        depth_threshold: Maximum depth considered 'shallow'.

    Returns:
        shallow_df, deep_df
    """
    raise NotImplementedError


def get_dataloaders(metadata_df, split="train", batch_size=config.BATCH_SIZE):
    """Build PyTorch DataLoaders with standard augmentation / normalisation.

    Args:
        metadata_df: DataFrame for this split.
        split: 'train' | 'val' | 'test' — controls augmentation.
        batch_size: Batch size.

    Returns:
        torch.utils.data.DataLoader
    """
    raise NotImplementedError
