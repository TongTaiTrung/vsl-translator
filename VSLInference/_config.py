"""Vendored constants for VSLInference (inlined from backend/config.py).

Self-contained: no import from backend. Values must stay in sync with the
config used at training time.
"""

SEQUENCE_LEN = 12
FRAME_STD_SIZE = 720

# Fallback labels only — VSLInference resolves labels per model at runtime:
# explicit `labels` arg > actions.json beside model > train_record.json beside
# model > this list. Keep in sync with training-time defaults.
ACTIONS = [
    "diachi",
    "camon",
    "xinchao",
    "toi",
    "khoer",
    "khochiu",
    "tambiet",
]

CONFIDENCE_THRESHOLD = 0.70
SWITCH_STABLE_FRAMES = 2
CLASS_THRESHOLDS = {}


def threshold_for(idx):
    """Confidence threshold for class index idx (per-class or global)."""
    try:
        name = str(ACTIONS[idx])
    except (IndexError, TypeError):
        return CONFIDENCE_THRESHOLD
    return float(CLASS_THRESHOLDS.get(name, CONFIDENCE_THRESHOLD))

# ── MediaPipe ──
MODEL_COMPLEXITY = 2
SMOOTH_LANDMARKS = True
MIN_DETECTION_CONFIDENCE = 0.01
MIN_TRACKING_CONFIDENCE = 0.01

# ── Features ──
SHOULDER_VIS_THRESHOLD = 0.5
IDLE_WRIST_Y_MAX = 1.75
IDLE_WRIST_X_MAX = 0.95
ARM_DEPTH_Z_GAIN = 2.0
ARM_DEPTH_EMA_ALPHA = 0.4
ARM_DEPTH_SMOOTH_WINDOW = 5
ARM_DEPTH_CAL_WARMUP = 20
