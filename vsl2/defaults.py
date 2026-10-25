"""Self-contained defaults for the vsl2 module.

Copied from backend/config.py at split time. All paths point inside vsl2/
(data, weights, actions.json). Override any value per-call via VSL2.train(...)
kwargs or `python -m vsl2 train --epochs ...` — never edit backend files.
"""

import json
import os

import numpy as np

VSL2_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Paths (all inside vsl2/) ─────────────────────────────────────────────
DATA_DIR = os.path.join(VSL2_DIR, "data")
MODELS_DIR = os.path.join(VSL2_DIR, "weights")
BACKUP_DIR = os.path.join(VSL2_DIR, "weights", "backups")
LOGS_DIR = os.path.join(VSL2_DIR, "logs")
MAX_BACKUPS = 5  # keep only the N most recent backups

# ── Actions / Labels ─────────────────────────────────────────────────────
ACTIONS_FILE = os.path.join(VSL2_DIR, "actions.json")


def _load_actions():
    """Load actions from actions.json. Falls back to defaults if missing."""
    if os.path.exists(ACTIONS_FILE):
        with open(ACTIONS_FILE, "r", encoding="utf-8") as f:
            return np.array(json.load(f))
    return np.array(["hello", "thanks", "iloveyou"])


def reload_actions():
    """Re-read actions.json (call after classes add/delete)."""
    global ACTIONS, LABEL_MAP
    ACTIONS = _load_actions()
    LABEL_MAP = {label: idx for idx, label in enumerate(ACTIONS)}
    return ACTIONS


ACTIONS = _load_actions()
LABEL_MAP = {label: idx for idx, label in enumerate(ACTIONS)}

# ── Data collection ──────────────────────────────────────────────────────
N_SEQUENCES = 15  # number of video sequences per action
SEQUENCE_LEN = 12  # frames per sequence

# ── Frame normalisation (unified cross-device convention) ────────────────
FRAME_STD_SIZE = int(os.environ.get("FRAME_STD_SIZE", "720"))

# ── MediaPipe ────────────────────────────────────────────────────────────
MIN_DETECTION_CONFIDENCE = 0.01
MIN_TRACKING_CONFIDENCE = 0.01
MODEL_COMPLEXITY = 2
SMOOTH_LANDMARKS = True

# ── Features ─────────────────────────────────────────────────────────────
HAND_DROPOUT_PROB = 0.5
HAND_SWAP_PROB = 0.3

# ── Idle-arm spatial gate (canonical units, see landmarks.extract()) ─────
IDLE_WRIST_Y_MAX = 1.75
IDLE_WRIST_X_MAX = 0.95

# ── Arm-depth estimation ─────────────────────────────────────────────────
ARM_DEPTH_Z_GAIN = 2.0
ARM_DEPTH_EMA_ALPHA = 0.4
ARM_DEPTH_SMOOTH_WINDOW = 5
ARM_DEPTH_CAL_WARMUP = 20
ARM_DEPTH_RATIO_MIN = 0.3
ARM_DEPTH_RATIO_MAX = 2.0

# ── Inactive-side injection augmentation (train only) ────────────────────
INACTIVE_INJECT_PROB = 0.15
INACTIVE_WRIST_Y_RANGE = (1.1, 1.6)
INACTIVE_DRIFT_STD = 0.015

# ── Training ─────────────────────────────────────────────────────────────
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42
EPOCHS = 200
MODEL_FILENAME = "model.keras"
TRAIN_RECORD_FILENAME = "train_record.json"
BATCH_SIZE = 16

# ── Inference ────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60
SMOOTHING_WINDOW = 3
SWITCH_STABLE_FRAMES = 2
CLASS_THRESHOLDS = {}


def threshold_for(idx):
    """Confidence threshold for class index idx (per-class or global)."""
    try:
        name = str(ACTIONS[idx])
    except (IndexError, TypeError):
        return CONFIDENCE_THRESHOLD
    return float(CLASS_THRESHOLDS.get(name, CONFIDENCE_THRESHOLD))


# ── Shoulder-relative / scale normalization ──────────────────────────────
SHOULDER_VIS_THRESHOLD = 0.5

# ── Data Augmentation ────────────────────────────────────────────────────
AUGMENT_FACTOR = 8
NOISE_STD = 0.05
SCALE_JITTER_STD = 0.05
TIME_WARP_SIGMA = 0.2
SHIFT_RANGE = (-0.02, 0.02)
FRAME_DROP_PROB = 0.05

# ── Strong augmentation preset (train only, opt-in) ──────────────────────
AUGMENT_PRESET = os.environ.get("AUGMENT_PRESET", "base")  # "base" | "strong"
STRONG_NOISE_STD = 0.09
STRONG_SCALE_JITTER_STD = 0.08
STRONG_TIME_WARP_SIGMA = 0.35
STRONG_SHIFT_RANGE = (-0.04, 0.04)
STRONG_FRAME_DROP_PROB = 0.15
STRONG_BURST_PROB = 0.6
STRONG_BURST_MIN = 3
STRONG_BURST_MAX = 6
STRONG_SPEED_CUT_PROB = 0.5
STRONG_SPEED_CUT_MAX = 3

# ── Learning-rate schedule ───────────────────────────────────────────────
LR_INITIAL = 1e-3
LR_MIN = 1e-6
LR_WARMUP_EPOCHS = 10

# ── Loss ─────────────────────────────────────────────────────────────────
FOCAL_ALPHA = 1.0
FOCAL_GAMMA = 1.0
LABEL_SMOOTHING = 0.02

# ── Early stopping & overfitting control ─────────────────────────────────
ES_PATIENCE = 60
ES_MIN_DELTA = 1e-4
ES_START_EPOCH = 30
OVERFIT_GAP_THRESHOLD = 0.25
TARGET_VAL_ACC = 0.98
TARGET_VAL_LOSS = 0.05
ES_LOSS_SPIKE_PATIENCE = 15
ES_PLATEAU_PATIENCE = 30
ES_PLATEAU_MIN_DELTA = 1e-4
OVERFIT_PATIENCE = 30
