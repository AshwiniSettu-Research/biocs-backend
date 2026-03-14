"""Configuration constants for the MLPT model."""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_default_dataset = os.path.join(os.path.dirname(BASE_DIR), "Research objective2 dataset.xlsx")
DATASET_PATH = os.environ.get("BIOCS_DATASET_PATH", _default_dataset)
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "mlpt", "saved_models")

# Dataset
NUM_CLASSES = 6
CLASS_NAMES = [
    "Cancer Antigenic Peptides",
    "Inactive Peptides-Lung Breast",
    "Moderately Active-Lung Breast",
    "Natural Peptide",
    "Non-Natural Peptide",
    "Very Active-Lung Breast",
]

# Sequence Processing
MAX_SEQ_LEN = 64
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 0 = padding
VOCAB_SIZE = len(AMINO_ACIDS) + 1  # 21

# Feature Extraction
NUM_PHYSICOCHEMICAL_FEATURES = 39

# Model Architecture
PATCH_SIZE = 4
EMBEDDING_DIM = 512
SWIN_NUM_HEADS = 8
SWIN_WINDOW_SIZE = 4
SWIN_DEPTH = 2  # layers per block
ADMAM_OUT_CHANNELS = 64
MLP_RATIO = 4.0
DROP_RATE = 0.3

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 100
PATIENCE = 15
TRAIN_SPLIT_1 = 0.7
TRAIN_SPLIT_2 = 0.8

# SA-BWK Optimizer
SABWK_POP_SIZE = 30
SABWK_MAX_ITER = 50
