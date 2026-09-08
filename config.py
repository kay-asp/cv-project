"""
Shared configuration: paths, hyperparameters, depth thresholds.
"""

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR = "data/"
MASK_DIR = "data/masks/"
MODEL_SAVE_PATH = "checkpoints/"
RESULTS_DIR = "results/"

# ── Data ─────────────────────────────────────────────────────────────
CROP_MARGIN = 0.2          # Fractional margin around bounding box
IMG_SIZE = (224, 224)       # ResNet-50 input size
DEPTH_SHALLOW_MAX = None    # Set after inspecting training depth range
MIN_IMAGES_PER_SPECIES = 10 # Minimum samples to include a species

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
