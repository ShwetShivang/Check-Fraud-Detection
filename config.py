"""
Central configuration for the Cheque Fraud Detection project.
All hyperparameters, paths, and settings are defined here.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
SAMPLE_RESULTS_DIR = os.path.join(OUTPUT_DIR, "sample_results")

# ─── Image Settings ──────────────────────────────────────────────────────
IMAGE_SIZE = (256, 512)         # (height, width) after resize
PATCH_SIZE = 64                 # Patch dimensions (64×64)
PATCH_STRIDE = 32               # Sliding window stride (50% overlap)
MIN_INK_RATIO = 0.05            # Minimum ink pixels in patch to keep it

# ─── Model Hyperparameters ───────────────────────────────────────────────
EMBEDDING_DIM = 64              # Siamese embedding vector size
NUM_CLASSES = 2                 # Binary: same ink vs different ink

# ─── Training Settings ───────────────────────────────────────────────────
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CONTRASTIVE_MARGIN = 2.0        # Margin for contrastive loss
EARLY_STOPPING_PATIENCE = 5
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ─── Fraud Detection Thresholds ──────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.5      # Below this → suspicious patch pair
FRAUD_REGION_THRESHOLD = 0.2    # If >20% patches suspicious → fraud

# ─── Augmentation Settings ───────────────────────────────────────────────
AUG_ROTATION_RANGE = 15         # ±15 degrees
AUG_NOISE_STD = 0.01            # Gaussian noise standard deviation
AUG_BRIGHTNESS_RANGE = 0.2      # ±20% brightness

# ─── Classical Model Settings ────────────────────────────────────────────
LBP_RADIUS = 1
LBP_N_POINTS = 8
GLCM_DISTANCES = [1, 3]
GLCM_ANGLES = [0, 0.785, 1.571, 2.356]  # 0°, 45°, 90°, 135°
COLOR_HIST_BINS = 32
SVM_C = 1.0
RF_N_ESTIMATORS = 100

# ─── Synthetic Data Settings ─────────────────────────────────────────────
SYNTHETIC_NUM_IMAGES = 50
SYNTHETIC_IMAGE_SIZE = (300, 600)

# ─── Device ──────────────────────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Ensure directories exist ────────────────────────────────────────────
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DIR, SYNTHETIC_DIR,
          OUTPUT_DIR, MODEL_DIR, FIGURES_DIR, SAMPLE_RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
