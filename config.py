"""
config.py  –  Central configuration for the EEG-AD pipeline.
Dataset : OpenNeuro ds004505  (AD + Normal subjects only)
Pipeline: MNE epochs -> 3-domain biomarkers -> MDAF-Net -> SHAP/Attention XAI
"""
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

# If BASE_DIR is read-only (like in Kaggle), save outputs to current working directory
if not os.access(BASE_DIR, os.W_OK):
    OUTPUT_BASE = os.getcwd()
else:
    OUTPUT_BASE = BASE_DIR

DATA_DIR    = os.path.join(BASE_DIR, "data")
FEATURE_DIR = os.path.join(OUTPUT_BASE, "features")
MODEL_DIR   = os.path.join(OUTPUT_BASE, "models")
PLOT_DIR    = os.path.join(OUTPUT_BASE, "plots")
XAI_DIR     = os.path.join(OUTPUT_BASE, "xai")

for d in [FEATURE_DIR, MODEL_DIR, PLOT_DIR, XAI_DIR]:
    os.makedirs(d, exist_ok=True)


# Dataset
OPENNEURO_DATASET = "ds004504"
LABEL_MAP   = {"AD": 1, "A": 1, "CN": 0, "HC": 0, "C": 0, "Normal": 0, "control": 0}
CLASSES     = [0, 1]
CLASS_NAMES = {0: "Normal", 1: "AD"}

# Preprocessing
SFREQ             = 256
L_FREQ            = 1.0
H_FREQ            = 40.0
NOTCH_FREQ        = 50.0
ICA_N_COMPONENTS  = 20
EPOCH_DURATION    = 4.0
EPOCH_OVERLAP     = 0.5
AMP_THRESHOLD     = 100e-6
REFERENCE         = "average"

# 16 key channels (Al-Nuaimi et al. 2021)
AD_CHANNELS = ["Fp1", "Fp2", "F7", "F3", "F4", "F8", "T3", "C3", "C4", "T4", "T5", "P3", "P4", "T6", "O1", "O2"]

# Frequency bands
BANDS = {
    "delta": (1.0,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

# 11 Al-Nuaimi band-power ratios
BAND_RATIOS = [
    ("theta","alpha"), ("alpha","theta"), ("alpha","delta"),
    ("beta","theta"),  ("theta","beta"),  ("delta","alpha"),
    ("delta","beta"),  ("theta","delta"), ("delta","theta"),
    ("beta","alpha"),  ("alpha","beta"),
]

# Complexity
TSALLIS_Q = 0.5
HFD_KMAX  = 10
HJORTH    = True

# Connectivity
COH_METHOD = "magnitude_squared"

# Feature selection
MANNWHITNEY_ALPHA = 0.001 / 645   # Bonferroni
TOP_FEATURE_PCT   = 0.80

# Model
EMBEDDING_DIM = 128
ATTN_HEADS    = 8
DROPOUT       = 0.3
GAT_LAYERS    = 2
NUM_CLASSES   = 2

# Training
BATCH_SIZE    = 64
EPOCHS        = 100
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
LAMBDA_MMSE   = 0.1
LABEL_SMOOTH  = 0.1
PATIENCE      = 20
OUTER_FOLDS   = 5
INNER_FOLDS   = 3
SEED          = 42

# XAI
SHAP_BACKGROUND = 100
SHAP_EXPLAIN    = 200
