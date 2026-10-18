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
import os
import pandas as pd
from pycocotools.coco import COCO
from fathomnet.api import images as fn_images

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import hashlib
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_fathomnet_metadata(coco_path, depth_path):
    """Load coco json, fetch depth from FathomNet API, and build a full
    metadata DataFrame. Caches depth lookups so the API is only hit once.

    Args:
        coco_path: Path to train.json or test.json
        depth_path: Where to save/load the depth CSV cache

    Returns:
        coco: pycocotools COCO object (for mask retrieval later)
        df: DataFrame with columns:
            image_id, ann_id, species, bbox, depth_m, width, height
    """
    

    # Load coco json
    coco = COCO(coco_path)
    cat_map = {c['id']: c['name'] for c in coco.dataset['categories']}

    # Build annotation rows
    rows = []
    for ann in coco.dataset['annotations']:
        img = coco.loadImgs(ann['image_id'])[0]
        rows.append({
            'image_id': ann['image_id'],
            'file_name': img['file_name'],   # added: uuid needed to download crops
            'ann_id':   ann['id'],
            'species':  cat_map[ann['category_id']],
            'bbox':     ann['bbox'],
            'width':    img['width'],
            'height':   img['height'],
        })
    df = pd.DataFrame(rows)

    # Fetch / load depth metadata
    if os.path.exists(depth_path):
        depth_df = pd.read_csv(depth_path)
        print(f"Loaded depth cache: {len(depth_df)} entries")
    else:
        depth_df = pd.DataFrame(columns=['image_id', 'depth_m'])

    # Find images we still need to look up
    cached_ids = set(depth_df['image_id'])
    all_imgs = coco.loadImgs(coco.getImgIds())
    missing = [img for img in all_imgs if img['id'] not in cached_ids]
    print(f"Need to fetch depth for {len(missing)} / {len(all_imgs)} images")

    new_rows = []
    for i, img in enumerate(missing):
        uuid = img['file_name'].replace('.png', '')
        try:
            result = fn_images.find_by_uuid(uuid)
            depth = result.depthMeters
        except Exception:
            depth = None
        new_rows.append({'image_id': img['id'], 'depth_m': depth})

        # Checkpoint every 500 to avoid losing progress
        if (i + 1) % 500 == 0:
            progress = pd.concat([depth_df, pd.DataFrame(new_rows)], ignore_index=True)
            progress.to_csv(depth_path, index=False)
            print(f"  fetched {i + 1} / {len(missing)} (saved checkpoint)")

    if new_rows:
        depth_df = pd.concat([depth_df, pd.DataFrame(new_rows)], ignore_index=True)
        depth_df.to_csv(depth_path, index=False)
        print(f"Saved depth cache: {len(depth_df)} entries")

    # Merge depth data and annotations data
    df = df.merge(depth_df[['image_id', 'depth_m']], on='image_id', how='left')

    return df, coco

def load_merged_metadata(df_train, df_test, coco_train, coco_test):
    """Load train and test, merge into one COCO object and one DataFrame."""
    
    # Merge DataFrames
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Merge COCO objects
    merged = {
        'images':      coco_train.dataset['images'] + coco_test.dataset['images'],
        'annotations': coco_train.dataset['annotations'] + coco_test.dataset['annotations'],
        'categories':  coco_train.dataset['categories'],
    }
    coco = COCO()
    coco.dataset = merged
    coco.createIndex()

    return df, coco

def display_image(image, masks=None, bboxes=None, labels=None, title=None,
                  show_original=False, figsize=None):
    """Display an image with optional mask overlays, bounding boxes and labels.

    Args:
        image: PIL Image or (H, W, 3) array.
        masks: one (H, W) mask or a list of masks (numpy or torch), 1 = organism.
            Each mask gets its own colour; bboxes[i] is drawn in mask i's colour.
        bboxes: list of (x, y, w, h) boxes.
        labels: list of strings, written above the matching box.
        title: title of the first panel.
        show_original: also show the plain image on the left, to compare
            against the overlay.
        figsize: defaults to (7 * number of panels, 7).
    """
    if masks is not None and not isinstance(masks, (list, tuple)):
        masks = [masks]
    masks = [np.asarray(m) for m in masks] if masks is not None else []
    bboxes = bboxes or []
    labels = labels or []

    n = max(len(masks), len(bboxes), 1)
    colors = plt.cm.Set2(np.linspace(0, 1, n))

    n_panels = 2 if show_original else 1
    fig, axes = plt.subplots(1, n_panels, figsize=figsize or (7 * n_panels, 7))
    axes = np.atleast_1d(axes)

    if show_original:
        axes[0].imshow(image)
        axes[0].set_title(title or "")

    ax = axes[-1]
    ax.imshow(image)
    if masks:
        overlay = np.zeros((*masks[0].shape[:2], 4))  # RGBA
        for mask, color in zip(masks, colors):
            overlay[mask == 1] = [*color[:3], 0.4]
        ax.imshow(overlay)

    for k, (bbox, color) in enumerate(zip(bboxes, colors)):
        x, y, w, h = bbox
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2,
                                       edgecolor=color, facecolor='none'))
        if k < len(labels):
            ax.text(x, y - 5, labels[k], color='white', fontsize=9,
                    bbox=dict(facecolor=color, alpha=0.7, pad=2))

    if not show_original:
        ax.set_title(title or "")
    elif masks:
        organism = np.any([m == 1 for m in masks], axis=0).mean()
        ax.set_title(f"+ mask overlay ({organism:.0%} organism)")

    for a in axes:
        a.axis('off')
    plt.tight_layout()
    plt.show()


def show_sample(coco, df, idx=0):
    """Display an image with bounding boxes, labels, and mask overlay."""
    row = df.iloc[idx]
    img = get_image(coco, row['image_id'])

    anns = coco.loadAnns(coco.getAnnIds(imgIds=row['image_id']))
    cat_map = {c['id']: c['name'] for c in coco.dataset['categories']}
    display_image(img,
                  masks=[coco.annToMask(a) for a in anns],
                  bboxes=[a['bbox'] for a in anns],
                  labels=[cat_map[a['category_id']] for a in anns],
                  title=f"image_id: {row['image_id']} | depth: {row.get('depth_m', '?')}m",
                  figsize=(14, 10))


def crop_to_bounding_box(image, bbox, margin=config.CROP_MARGIN, mask=None):
    """Crop image around a bounding box with a surrounding habitat margin.

    Args:
        image: PIL Image or numpy array.
        bbox: (x, y, w, h) bounding box of the organism.
        margin: Fractional margin to add on each side.

    Returns:
        Cropped image, adjusted mask (if provided).
    """
    img_w, img_h = image.size
    x, y, w, h = bbox  #x,y is is top left corner of box, w,h is its width and height

    pad_w = margin * w  #margin is a fraction of box's own size
    pad_h = margin * h

    #expand box by padding amount, check never goes outside image
    left = int(max(0, x - pad_w))
    top = int(max(0, y - pad_h))
    right = int(min(img_w, x + w + pad_w))
    bottom = int(min(img_h, y + h + pad_h))

    if right <= left or bottom <= top:
        raise ValueError(
            f"Degenerate crop box {bbox} on a {img_w}x{img_h} image "
            f"(check bbox is (x, y, w, h) in that image's own coordinates)."
        )
    
    cropped_image = image.crop((left, top, right, bottom))

    #since image cropped, mask must also be cropped to keep them aligned
    cropped_mask = None
    if mask is not None:
        if mask.shape[:2] != (img_h, img_w):
            raise ValueError(
                f"mask shape {mask.shape[:2]} doesn't match image size "
                f"({img_h}, {img_w}) — pass the mask for the same "
                f"(uncropped) image as `image`."
            )
        cropped_mask = mask[top:bottom, left:right]

    return cropped_image, cropped_mask, (left, top, right, bottom)


def split_shallow_deep(metadata_df, depth_threshold):
    """Split dataset into shallow (≤ threshold) and deep (> threshold)"""

    #number of rows with missing depth
    n_missing = metadata_df['depth_m'].isna().sum()

    #if depth is missing, drop row
    if n_missing:
        print(f"Dropping {n_missing} rows with missing depth_m "
              f"({n_missing / len(metadata_df):.1%} of data)")
    df = metadata_df.dropna(subset=['depth_m'])

    shallow_df = df[df['depth_m'] <= depth_threshold].reset_index(drop=True) #shallow split
    deep_df = df[df['depth_m'] > depth_threshold].reset_index(drop=True) #deep split

    print(f"Threshold {depth_threshold}m -> shallow: {len(shallow_df)} "
          f"({len(shallow_df) / len(df):.1%}), deep: {len(deep_df)} "
          f"({len(deep_df) / len(df):.1%})")
    
    return shallow_df, deep_df

def summarise_depth_distribution(metadata_df):
    #print percentiles and species counts by depth to help pick shallow/deep threshold.
    depths = metadata_df['depth_m'].dropna()  #pull out just depth column and drop missing vals
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    rows = []

    for p in percentiles:
        cut = depths.quantile(p/100) #find depth value below which p% of the data falls
        #eg. depths.quantile(0.25) gives depth where 25% of imgs are shallower than that and 75% are deeper
        n_shallow = (depths <= cut).sum()
        n_deep = (depths > cut).sum()
        rows.append({
            'percentile': p,
            'depth_m': round(cut, 1),
            'n_shallow': n_shallow,
            'n_deep': n_deep,
        })  #collect one row per percentile into a table
    
    summary = pd.DataFrame(rows)
    print(depths.describe())
    print()
    print(summary.to_string(index=False))
    return summary




# ══════════════════════════════════════════════════════════════════════
# Added for the Modeling phase: species filtering, crop cache, DataLoaders
# ══════════════════════════════════════════════════════════════════════

def restrict_to_common_species(shallow_df, deep_df, min_n=None):
    """Keep only species with >= min_n annotations on BOTH sides of the depth
    split, so every class has something to compare shallow vs deep on.

    Without this, shallow_df has ~1,200 species but only a fraction also
    appear in deep water, which inflates num_classes and makes deep accuracy
    undefined for most classes.
    """
    if min_n is None:
        min_n = getattr(config, "MIN_IMAGES_PER_SPECIES", 10)
    sh = shallow_df["species"].value_counts()
    dp = deep_df["species"].value_counts()
    keep = sorted(set(sh[sh >= min_n].index) & set(dp[dp >= min_n].index))
    shallow_df = shallow_df[shallow_df["species"].isin(keep)].reset_index(drop=True)
    deep_df = deep_df[deep_df["species"].isin(keep)].reset_index(drop=True)
    print(f"Species with >= {min_n} annotations in BOTH splits: {len(keep)}")
    print(f"Kept -> shallow: {len(shallow_df)} annotations | deep: {len(deep_df)} annotations")
    return shallow_df, deep_df


def _crop_file(image_id, ann_id):
    return f"{image_id}_{ann_id}.jpg"


def save_crop(crop, path):
    """Resize a crop to config.IMG_SIZE and save it as JPEG """
    
    crop = crop.convert("RGB").resize(config.IMG_SIZE, Image.BILINEAR)
    crop.save(path, quality=getattr(config, "CROP_JPEG_QUALITY", 95))


def _download_image(uuid, retries=3):
    """Fetch one full FathomNet image as RGB PIL; None if it keeps failing."""
    for attempt in range(retries):
        try:
            fn_img = fn_images.find_by_uuid(uuid)
            resp = requests.get(fn_img.url, timeout=30)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            time.sleep(1.5 ** attempt)
    return None


def get_image(coco, image_id):
    """Download one full FathomNet image (with retries) as an RGB PIL image."""
    file_name = coco.loadImgs(int(image_id))[0]['file_name']
    img = _download_image(file_name.replace('.png', ''))
    if img is None:
        raise RuntimeError(f"Could not download image {image_id}")
    return img


def prepare_crops(df, workers=32, limit=None):
    """Download each image once, crop every annotation on it (with the habitat
    margin) and save the crops as IMG_SIZE JPEGs. Resumable: crops already on
    disk are skipped, so it is safe to re-run after a disconnect.

    limit - for smoke test
    returns (n_saved, n_failed) annotation counts for this call.
    """
    
    os.makedirs(config.CROP_DIR, exist_ok=True)
    existing = set(os.listdir(config.CROP_DIR))
    todo = df[[_crop_file(i, a) not in existing for i, a in zip(df["image_id"], df["ann_id"])]]
    groups = list(todo.groupby("image_id"))
    if limit:
        groups = groups[:limit]
    print(f"{len(todo)} crops missing across {len(groups)} images to fetch")

    def work(item):
        image_id, g = item
        img = _download_image(g["file_name"].iloc[0].replace(".png", ""))
        if img is None:
            return 0, len(g)
        saved = 0
        for r in g.itertuples():
            try:
                crop, _, _ = crop_to_bounding_box(img, r.bbox)
                save_crop(crop, os.path.join(config.CROP_DIR, _crop_file(r.image_id, r.ann_id)))
                saved += 1
            except ValueError:      # degenerate bbox -> skip
                pass
        return saved, len(g) - saved

    n_saved = n_failed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, item) for item in groups]
        for k, fut in enumerate(as_completed(futures), 1):
            s, f = fut.result()
            n_saved += s
            n_failed += f
            if k % 500 == 0:
                print(f"  {k}/{len(groups)} images | saved {n_saved} crops | failed {n_failed}")
    print(f"Done: saved {n_saved} crops, {n_failed} failed/skipped")
    return n_saved, n_failed


def archive_dir(src_dir, archive_path):
    """Pack every file in src_dir (e.g. CROP_DIR, MASK_DIR) into the tar at
    archive_path (e.g. CROP_ARCHIVE, MASK_ARCHIVE) on the Shared Drive."""
    with tarfile.open(archive_path, "w") as tar:
        for name in os.listdir(src_dir):
            if not name.endswith(".tmp"):   # skip half-written files
                tar.add(os.path.join(src_dir, name), arcname=name)
    print(f"Saved {src_dir} to {archive_path}")


def restore_dir(archive_path, dest_dir):
    """Extract the tar at archive_path into dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(archive_path) as tar:
        tar.extractall(dest_dir)
    print(f"Restored {len(os.listdir(dest_dir))} files to {dest_dir}")


def drop_missing_crops(df):
    """Drop rows whose crop is not on disk (e.g. the source image is no longer
    downloadable from FathomNet), so every DataLoader sees the same rows."""
    existing = set(os.listdir(config.CROP_DIR)) if os.path.isdir(config.CROP_DIR) else set()
    has_crop = [_crop_file(i, a) in existing for i, a in zip(df["image_id"], df["ann_id"])]
    n_missing = len(has_crop) - sum(has_crop)
    if n_missing:
        print(f"Dropping {n_missing} rows with no crop on disk")
    return df[has_crop].reset_index(drop=True)


def _split_of(image_id, seed, val_frac, test_frac):
    """Deterministic train/val/test assignment from the image id alone.

    Because it depends only on (seed, image_id), every DataFrame gets the same
    assignment: the control model's train split never overlaps the deep test
    split, and all annotations from one image land in the same split.
    """
    u = int(hashlib.md5(f"{seed}-{image_id}".encode()).hexdigest(), 16) % 10_000 / 10_000
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"


class CropDataset(Dataset):
    """Reads pre-cropped PNGs; yields (image_tensor, integer_label)."""

    def __init__(self, df, species_to_idx, crop_dir, transform):
        self.paths = [os.path.join(crop_dir, _crop_file(r.image_id, r.ann_id))
                      for r in df.itertuples()]
        self.labels = [species_to_idx[s] for s in df["species"]]
        self.image_ids = df["image_id"].tolist()
        self.ann_ids = df["ann_id"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def _mask_file(image_id, ann_id):
    return f"{image_id}_{ann_id}.png"


def get_crop_mask(coco, image_id, ann_id, bbox):
    """Organism mask aligned pixel-for-pixel with the saved crop.

    Same steps as the notebook's crop check (annToMask -> crop_to_bounding_box),
    then the IMG_SIZE squash save_crop applied to the image. annToMask only
    needs the COCO metadata, so no image download: a blank placeholder of the
    full image size stands in for the image when computing the crop box.

    Cached: the first call saves the mask to MASK_DIR as a PNG, later calls
    just load it.

    Returns:
        uint8 array of shape IMG_SIZE (H, W), 1 = organism.
    """
    path = os.path.join(config.MASK_DIR, _mask_file(image_id, ann_id))
    if os.path.exists(path):
        return np.array(Image.open(path), dtype=np.uint8)

    ann = coco.loadAnns([int(ann_id)])[0]
    full_mask = coco.annToMask(ann)
    h, w = full_mask.shape
    _, mask, _ = crop_to_bounding_box(Image.new("L", (w, h)), bbox, mask=full_mask)
    # Same resize call as save_crop; NEAREST keeps the mask binary
    mask = Image.fromarray(mask).resize(config.IMG_SIZE, Image.NEAREST)

    # Write to a temp file then rename, so an interrupted save never leaves a
    # half-written PNG that later calls would read as the cached mask
    os.makedirs(config.MASK_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    mask.save(tmp, format="PNG")
    os.replace(tmp, path)
    return np.array(mask, dtype=np.uint8)


def prepare_masks(mask_loader):
    """Compute and cache the mask of every crop in a loader built with
    get_dataloaders(..., coco=coco). Resumable: cached masks are skipped.

    Returns:
        Number of masks newly saved (0 if all were already cached).
    """
    ds = mask_loader.dataset
    existing = set(os.listdir(config.MASK_DIR)) if os.path.isdir(config.MASK_DIR) else set()
    n_new = 0
    for image_id, ann_id, bbox in zip(ds.image_ids, ds.ann_ids, ds.bboxes):
        if _mask_file(image_id, ann_id) not in existing:
            get_crop_mask(ds.coco, image_id, ann_id, bbox)
            n_new += 1
    print(f"Masks: {n_new} newly saved, {len(ds) - n_new} already cached in {config.MASK_DIR}")
    return n_new


class MaskedCropDataset(CropDataset):
    """CropDataset that also yields each crop's organism mask (for CRS)."""

    def __init__(self, df, species_to_idx, crop_dir, transform, coco):
        super().__init__(df, species_to_idx, crop_dir, transform)
        self.bboxes = df["bbox"].tolist()
        self.coco = coco

    def __getitem__(self, i):
        img, label = super().__getitem__(i)
        mask = get_crop_mask(self.coco, self.image_ids[i], self.ann_ids[i], self.bboxes[i])
        return img, label, torch.from_numpy(mask)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def show_saved_crops(mask_loader, idxs=(0,)):
    """Display saved crops with their masks exactly as attribution sees them.

    Unlike the crop check in the notebook (which re-crops the original
    downloaded image), this reads
    the 224x224 crops from disk and the masks from get_crop_mask, so it checks
    that what Grad-CAM / CRS receive is aligned.

    Args:
        mask_loader: DataLoader built with get_dataloaders(..., coco=coco).
        idxs: positions in mask_loader.dataset to show.
    """
    ds = mask_loader.dataset
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    for i in idxs:
        img, label, mask = ds[i]
        img = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()  # undo Normalize
        display_image(img, masks=mask.numpy(), show_original=True,
                      title=f"{mask_loader.species_list[label]} — saved crop "
                            f"(image_id {ds.image_ids[i]}, ann_id {ds.ann_ids[i]})")


def _build_transform(train):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(), norm])
    return transforms.Compose([transforms.Resize(config.IMG_SIZE),
                               transforms.ToTensor(), norm])


def get_dataloaders(metadata_df, split="train", batch_size=None,
                    species_list=None, coco=None):
    """Build PyTorch DataLoaders with standard augmentation / normalisation.

    Args:
        metadata_df: DataFrame for this depth group (run prepare_crops on it first).
        split: 'train' | 'val' | 'test' (image-level, deterministic) or 'all'
            (every row, no augmentation).
        batch_size: Batch size (default: BATCH_SIZE for train, else EVAL_BATCH_SIZE).
        species_list: class order; defaults to sorted unique species of
            metadata_df. Label index = position in this list.
        coco: COCO object; if given, batches also include crop-aligned masks
            (for attribution). Not allowed with split='train', whose random
            flip would misalign image and mask.

    Returns:
        DataLoader yielding (images, labels), or (images, labels, masks) when
        coco is given. Has a .species_list attribute.
    """
    if split not in ("train", "val", "test", "all"):
        raise ValueError(f"split must be train/val/test/all, got {split!r}")
    if coco is not None and split == "train":
        raise ValueError("masks are only supported for val/test/all (train augmentation flips images)")
    crop_dir = config.CROP_DIR
    species_list = list(species_list) if species_list is not None else sorted(metadata_df["species"].unique())
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    df = metadata_df[metadata_df["species"].isin(species_to_idx)]

    if split != "all":
        seed = getattr(config, "RANDOM_SEED", 42)
        test_frac = getattr(config, "TEST_SPLIT", 0.2)
        val_frac = getattr(config, "VAL_SPLIT", 0.1)
        assign = {i: _split_of(i, seed, val_frac, test_frac) for i in df["image_id"].unique()}
        df = df[df["image_id"].map(assign) == split]

    existing = set(os.listdir(crop_dir)) if os.path.isdir(crop_dir) else set()
    has_crop = [_crop_file(i, a) in existing for i, a in zip(df["image_id"], df["ann_id"])]
    if not all(has_crop):
        print(f"Warning: {len(has_crop) - sum(has_crop)} rows have no crop on disk and are skipped "
              f"(run prepare_crops first)")
    df = df[has_crop]
    if len(df) == 0:
        raise RuntimeError(f"No samples for split={split!r} - did prepare_crops run?")

    if batch_size is None:
        batch_size = config.BATCH_SIZE if split == "train" else config.EVAL_BATCH_SIZE

    transform = _build_transform(train=(split == "train"))
    if coco is not None:
        ds = MaskedCropDataset(df, species_to_idx, crop_dir, transform, coco)
    else:
        ds = CropDataset(df, species_to_idx, crop_dir, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                        num_workers=config.NUM_WORKERS or os.cpu_count(),
                        persistent_workers=True, prefetch_factor=4,
                        pin_memory=torch.cuda.is_available())
    loader.species_list = species_list
    print(f"{split}: {len(ds)} crops, {len(species_list)} classes")
    return loader
