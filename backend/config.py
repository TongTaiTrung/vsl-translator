"""
Central configuration for the VSL Translator project.
All paths, actions, and hyperparameters live here.
"""

import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv():
    """Load PROJECT_ROOT/.env so env-overridable settings below see it."""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    try:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


_load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BACKUP_DIR = os.path.join(PROJECT_ROOT, "models", "backups")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
MAX_BACKUPS = 5                # keep only the N most recent backups

# ── Actions / Labels ──────────────────────────────────────────────────────────
ACTIONS_FILE = os.path.join(PROJECT_ROOT, "actions.json")

def _load_actions():
    """Load actions from actions.json. Falls back to defaults if missing."""
    import json
    if os.path.exists(ACTIONS_FILE):
        with open(ACTIONS_FILE, "r") as f:
            return np.array(json.load(f))
    return np.array(["hello", "thanks", "iloveyou"])

ACTIONS = _load_actions()
LABEL_MAP = {label: idx for idx, label in enumerate(ACTIONS)}

# ── Data collection ──────────────────────────────────────────────────────────
N_SEQUENCES = 15          # number of video sequences per action
SEQUENCE_LEN = 12        # frames per sequence

# ── Frame normalisation (unified cross-device convention) ─────────────────
FRAME_STD_SIZE = int(os.environ.get("FRAME_STD_SIZE", "720"))  # square letterbox canvas (720 matches 720p height; keep square — rectangular canvas changes w/h and breaks canonical features)
INGEST_PORT = int(os.environ.get("INGEST_PORT", "8765"))       # WebSocket /ws/stream
SESSION_TTL = float(os.environ.get("SESSION_TTL", "60.0"))     # idle seconds before a stream session is dropped

# ── MediaPipe ────────────────────────────────────────────────────────────────
MIN_DETECTION_CONFIDENCE = 0.01  # hạ từ 0.5 → nhạy hơn, ít miss tay khi che khuất
MIN_TRACKING_CONFIDENCE = 0.01   # hạ từ 0.5 → giữ track tay tốt hơn khi chuyển động nhanh
MODEL_COMPLEXITY = 2            # 0/1/2: 2 = pose chính xác nhất (đổi lấy ~30% FPS)
SMOOTH_LANDMARKS = True         # lọc jitter liên-frame → tọa độ ổn định hơn

# ── Features ─────────────────────────────────────────────────────────────────
# Canonical-skeleton retargeted keypoints: 356-dim body vector, no motion
# expansion (motion channels removed — GRU learns temporal dynamics itself).
# pose 6×(x,y,vis)=18 + 2 hands 21×(x,y)=84 + arm_depths 2 + shoulder-offsets 252.
# See landmarks.extract().
HAND_DROPOUT_PROB = 0.5    # augmentation: ngẫu nhiên zero 1 bên tay → model không
                           # học shortcut "tay nào hiện/tay nào mất" mà phải học
                           # động tác thật (0.5: mỗi mẫu thấy cả bản đủ tay lẫn mất tay)
HAND_SWAP_PROB = 0.3       # augmentation: swap 2 khối tay lh↔rh, giữ label →
                           # model bất biến phía tay (chỉ dùng khi class không
                           # phân biệt bằng tay trái/phải thật)

# ── Idle-arm spatial gate (canonical units, see landmarks.extract()) ──────────
IDLE_WRIST_Y_MAX = 1.75  # wrist below this depth + near torso → idle → zero side
IDLE_WRIST_X_MAX = 0.95  # |x| bound for the idle check

# ── Arm-depth estimation (see landmarks._compute_arm_depth_ratio) ──────────────
ARM_DEPTH_Z_GAIN = 2.0    # gain of the auxiliary (wrist.z - shoulder.z) correction
ARM_DEPTH_EMA_ALPHA = 0.4  # smoothing factor for ArmDepthSmoother (realtime only)
ARM_DEPTH_SMOOTH_WINDOW = 5  # median window for ArmDepthSmoother
ARM_DEPTH_CAL_WARMUP = 20  # non-idle frames to calibrate neutral arm length
ARM_DEPTH_RATIO_MIN = 0.3  # clip bounds for the final ratio
ARM_DEPTH_RATIO_MAX = 2.0

# ── Inactive-side injection augmentation (train only) ─────────────────────────
INACTIVE_INJECT_PROB = 0.15      # probability of injecting a neutral second arm
INACTIVE_WRIST_Y_RANGE = (1.1, 1.6)  # canonical y range for synthetic rest wrist
INACTIVE_DRIFT_STD = 0.015       # per-frame Gaussian drift of synthetic arm

# ── Training ─────────────────────────────────────────────────────────────────
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15          # held-out test set, never touched during training
RANDOM_SEED = 42
EPOCHS = 200
MODEL_FILENAME = "model.keras"
TRAIN_RECORD_FILENAME = "train_record.json"  # hyperparameters + epoch history, beside the model
BATCH_SIZE = 16

# ── Inference ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60  # below this → "uncertain", no label emitted
SMOOTHING_WINDOW = 3         # majority-vote window for live prediction
SWITCH_STABLE_FRAMES = 2     # challenger must win N vote rounds in a row
                             # before replacing the current prediction
                             # (hysteresis against A↔B flicker on similar signs)
CLASS_THRESHOLDS = {}        # optional per-class override, e.g.
                             # {"diachi": 0.75}; falls back to
                             # CONFIDENCE_THRESHOLD when absent


def threshold_for(idx):
    """Confidence threshold for class index idx (per-class or global)."""
    try:
        name = str(ACTIONS[idx])
    except (IndexError, TypeError):
        return CONFIDENCE_THRESHOLD
    return float(CLASS_THRESHOLDS.get(name, CONFIDENCE_THRESHOLD))

# ── Smart sentence (Groq LLM cleanup, see scripts/groq_client.py) ────────────
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_TIMEOUT = float(os.environ.get("GROQ_TIMEOUT", "30.0"))
GROQ_RETRIES = int(os.environ.get("GROQ_RETRIES", "1"))
SMART_CALL_INTERVAL = float(os.environ.get("SMART_CALL_INTERVAL", "7.0"))
SMART_RETRY_INTERVAL = float(os.environ.get("SMART_RETRY_INTERVAL", "15.0"))
SMART_POLL_INTERVAL = float(os.environ.get("SMART_POLL_INTERVAL", "0.1"))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


def set_groq_api_key(key):
    """Persist Groq API key to PROJECT_ROOT/.env and update runtime value."""
    global GROQ_API_KEY
    key = (key or "").strip().strip("'\"")
    if not key:
        raise ValueError("empty API key")
    GROQ_API_KEY = key
    os.environ["GROQ_API_KEY"] = key
    env_path = os.path.join(PROJECT_ROOT, ".env")
    try:
        with open(env_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        lines = []
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith("GROQ_API_KEY="):
            lines[i] = f'GROQ_API_KEY="{key}"'
            found = True
    if not found:
        lines.append(f'GROQ_API_KEY="{key}"')
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return True

# ── Shoulder-relative / scale normalization ──────────────────────────────────
SHOULDER_VIS_THRESHOLD = 0.5  # shoulders below this visibility → frame zeros

# ── Data Augmentation ────────────────────────────────────────────────────────
AUGMENT_FACTOR = 8             # how many augmented copies per original sample
                               # (dataset nhỏ ~90 mẫu → tăng lên 8 để train set ~500 mẫu)
# Calibrated for canonical units (pose range ~±2, not image [0,1]).
NOISE_STD = 0.05               # Gaussian noise std on (x,y) coords
SCALE_JITTER_STD = 0.05        # per-frame ~N(1, std) multiplier (bone-direction jitter)
TIME_WARP_SIGMA = 0.2          # temporal warping strength
SHIFT_RANGE = (-0.02, 0.02)    # canonical-origin jitter (canonical units)
FRAME_DROP_PROB = 0.05         # probability of dropping (zeroing) a frame

# ── Strong augmentation preset (train only, opt-in) ─────────────────────────
# Same ops as base but wider magnitudes + burst tracking-loss and speed-cut,
# simulating cross-session live variation (different signer speed, camera
# tracking gaps, early/late sign boundaries). Off by default: set
# AUGMENT_PRESET="strong" to use. Verified via scripts/kfold_eval.py, not
# the saturated single held-out split.
AUGMENT_PRESET = os.environ.get("AUGMENT_PRESET", "base")  # "base" | "strong"
STRONG_NOISE_STD = 0.09
STRONG_SCALE_JITTER_STD = 0.08
STRONG_TIME_WARP_SIGMA = 0.35
STRONG_SHIFT_RANGE = (-0.04, 0.04)
STRONG_FRAME_DROP_PROB = 0.15
STRONG_BURST_PROB = 0.6        # zero one hand for 3-6 consecutive frames
STRONG_BURST_MIN = 3
STRONG_BURST_MAX = 6
STRONG_SPEED_CUT_PROB = 0.5    # truncate head/tail 1-3 frames, zero-pad
STRONG_SPEED_CUT_MAX = 3

# ── Learning-rate schedule (Warmup + Cosine Annealing) ───────────────────────
LR_INITIAL = 1e-3
LR_MIN = 1e-6
LR_WARMUP_EPOCHS = 10          # linear warmup from LR_MIN → LR_INITIAL

# ── Loss hyperparameters (CategoricalFocalCrossentropy) ──────────────────────
FOCAL_ALPHA = 1.0              # class đã cân bằng (30 mẫu/class) + đã có class_weight
                               # → alpha scalar 0.25 chỉ thu nhỏ loss, không cân bằng class
FOCAL_GAMMA = 1.0
LABEL_SMOOTHING = 0.02         # 0.1 đè confidence dưới ngưỡng inference 0.70

# ── Early stopping & overfitting control ─────────────────────────────────────
ES_PATIENCE = 60               # epochs without val_loss improvement → stop
ES_MIN_DELTA = 1e-4            # minimum improvement to count as progress
ES_START_EPOCH = 30            # only check early stopping after warmup + ổn định
OVERFIT_GAP_THRESHOLD = 0.25   # if (train_acc - val_acc) > this → warn/act
TARGET_VAL_ACC = 0.98          # stop early if val_acc reaches this
TARGET_VAL_LOSS = 0.05         # stop early if val_loss drops below this
# ── Timely stopping (dừng kịp lúc, tránh train thừa) ─────────────────────────
ES_LOSS_SPIKE_PATIENCE = 15    # val_loss tăng liên tiếp N epochs → dừng ngay
ES_PLATEAU_PATIENCE = 30       # val_acc + val_loss cùng giậm chân N epochs → dừng
ES_PLATEAU_MIN_DELTA = 1e-4    # biến thiên nhỏ hơn mức này coi như giậm chân
OVERFIT_PATIENCE = 30          # train-val gap vượt ngưỡng N epochs liên tiếp → dừng


