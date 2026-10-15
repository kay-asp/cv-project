"""
Shared configuration: paths, hyperparameters, depth thresholds.
"""

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR = "data/"
MASK_DIR = "data/masks/"
MODEL_SAVE_PATH = "checkpoints/"
RESULTS_DIR = "results/"

# ADDED — notebook cell 10 calls data.load_fathomnet_metadata(config.TRAIN_PATH, config.DEPTH_PATH)
# but neither existed in config.py. These are my best guess based on the
# uploaded filenames (train.json, test.json, depth_cache.csv in a data folder).
# CONFIRM these match the real paths in your teammate's Drive config.py before
# using this file — if hers differs, use hers instead of mine.
# CHANGED — folder is flat (no data/ subfolder), so these are just filenames.
# If you later organise into subfolders, prefix with DATA_DIR again.
TRAIN_PATH = "train.json"
TEST_PATH = "test.json"
DEPTH_PATH = "depth_cache.csv"

# ADDED — used by data.prepare_crops() / data.get_dataloaders(). CHANGED from
# /content/crops/ (Colab-only path, doesn't exist on the cluster) to the
# project directory itself, since that's confirmed writable
# (/home-mscluster/lpitsi/cv-project). If your cluster gives you faster
# per-node scratch storage (e.g. /scratch/$USER/ or /local/$USER/), point
# this there instead for quicker crop reads during training.
CROP_DIR = "crops/"

# ── Data ─────────────────────────────────────────────────────────────
CROP_MARGIN = 0.2          # Fractional margin around bounding box
IMG_SIZE = (224, 224)       # ResNet-50 input size
DEPTH_SHALLOW_MAX = 1056    # SET — from teammate's threshold sweep (confirm min_n=5 vs 10 as a team, see note below)
MIN_IMAGES_PER_SPECIES = 10 # Minimum samples to include a species
# NOTE: teammate's notebook sweep used min_n=5 to get species counts, which
# gives 314 usable species. This constant (10) gives only 227. Pick one value
# as a team and use it consistently in restrict_to_common_species() calls.

# ── Model ────────────────────────────────────────────────────────────
BACKBONE = "resnet50"
PRETRAINED = True           # ImageNet initialisation
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 25
NUM_WORKERS = 2

# ── Attribution ──────────────────────────────────────────────────────
GRADCAM_TARGET_LAYER = "layer4"  # Final conv block of ResNet-50

# ── Evaluation ───────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1  # ADDED — used by data.get_dataloaders() for the train/val/test image-level split
