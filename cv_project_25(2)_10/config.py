"""
Shared configuration: paths, hyperparameters, depth thresholds.
"""

# ── Paths ────────────────────────────────────────────────────────────
CROP_DIR = "/content/crops/"   # Crops, extracted from CROP_ARCHIVE each Colab session (fast VM disk)
CROP_ARCHIVE = "crops.tar"    # On the Shared Drive, so the group shares one copy
DATA_DIR = "data/"
MASK_DIR = "/content/masks/"   # Crop-aligned masks, cached by get_crop_mask (fast VM disk, like CROP_DIR)
MASK_ARCHIVE = "masks.tar"    # On the Shared Drive, like CROP_ARCHIVE
MODEL_SAVE_PATH = "checkpoints/"
RESULTS_DIR = "results/"

TRAIN_PATH = DATA_DIR + "train.json"
TEST_PATH = DATA_DIR + "test.json"
DEPTH_PATH = DATA_DIR + "depth_cache.csv"

CROP_JPEG_QUALITY = 95

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
LEARNING_RATE = 2e-4        # CHANGED from 1e-4: sqrt-scaled for Adam alongside the 4x batch increase
BATCH_SIZE = 128            # CHANGED from 32: better GPU utilisation; fits a 16 GB T4 with mixed precision
EVAL_BATCH_SIZE = 256       # Val/test batch (no gradients, so a larger batch fits)
EPOCHS = 12                 # Cap: early stopping usually ends sooner (val acc peaked at epochs 4-8 in the 25-epoch cluster run)
PATIENCE = 3                # Stop after this many epochs without a val-accuracy improvement
USE_AMP = True              # Mixed-precision (fp16) training on CUDA, ~2-3x faster on T4/V100/A100
USE_COSINE = False          # Optional cosine LR decay over EPOCHS
NUM_WORKERS = None          # None = os.cpu_count() (Colab gives 2, the cluster job 16)

# ── Attribution ──────────────────────────────────────────────────────
GRADCAM_TARGET_LAYER = "layer4"  # Final conv block of ResNet-50

# ── Evaluation ───────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1  # ADDED — used by data.get_dataloaders() for the train/val/test image-level split
