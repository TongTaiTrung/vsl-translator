# VSL Translator

Vietnamese Sign Language (VSL) translator using **MediaPipe Holistic** for landmark extraction and a **Bidirectional GRU** classifier (Keras/TensorFlow) for sign recognition.

## How it works

1. **Landmark extraction** (`backend/src/vsl_translator/landmarks.py`): MediaPipe Holistic detects pose + two hands per frame. `extract()` converts them to a canonical-skeleton feature vector — invariant to body proportions and camera distance.
2. **Feature vector (356-dim)** per frame:
   - pose 6×(x,y,vis) = 18 — shoulders pinned at (∓0.5, 0), elbows/wrists rebuilt by Forward Kinematics from unit bone vectors
   - left hand 21×(x,y) = 42 + right hand 42 — normalized by each hand's own palm size around its own wrist
   - arm_depths = 2 — per-arm foreshortening ratio (toward/away from camera), calibrated per session
   - shoulder-offsets 126×2 = 252 — per-point hand offsets to 3 raised-shoulder anchors (left raised shoulder, right raised shoulder, their midpoint; shoulder x lifted to face-center y), palm-normalized
3. **Classification** (`backend/src/vsl_translator/model.py`): per-group LayerNorm over the 4 feature groups → BiGRU → attention pooling → softmax over actions.
4. **Training** (`backend/scripts/train.py`): train-only augmentation (noise, scale jitter, time warp, shift, hand dropout/swap, inactive-side injection) with zero-mask preservation; data must be (re)collected with the current `extract()` — old `.npy`/`.keras`/stats artifacts are incompatible across dim changes.

## Project Structure

```
vsl-translator/
├── backend/
│   ├── config.py                  # Central configuration (paths, actions, hyperparams)
│   ├── src/vsl_translator/        # Core library
│   │   ├── landmarks.py           # Holistic detection, drawing, keypoint extraction
│   │   ├── dataset.py             # Sequence loading, augmentation, standardization
│   │   └── model.py               # GRU model definition
│   ├── scripts/                   # Runnable CLI scripts
│   │   ├── collect_data.py        # Collect keypoint data from webcam
│   │   ├── train.py               # Train the GRU classifier
│   │   ├── predict.py             # Evaluate on the held-out test set
│   │   ├── realtime_predict.py    # Live webcam inference with smoothing
│   │   ├── predict_live.py        # Alternate live inference entrypoint
│   │   ├── cam_test.py            # Live webcam landmark visualisation
│   │   ├── add_noise.py           # Offline noise/distance-scale copies (use sparingly, see CHANGELOG)
│   │   └── migrate_data.py        # Disabled: stored vectors can't be migrated across layouts
│   ├── tests/                     # Unit tests (test_far_bias, test_arm_depths, ...)
│   ├── data/                      # Collected keypoint sequences (.npy)
│   ├── models/                    # Trained model weights (.keras) + stats
│   ├── logs/                      # TensorBoard training logs
│   ├── CHANGELOG.md               # Feature/layout history (read before training)
│   └── TRAINING.md                # Training notes
├── frontend/                      # Web frontend
├── venv/                          # Python virtualenv
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
cd backend

# 1. Test webcam + MediaPipe
venv/bin/python scripts/cam_test.py

# 2. Collect training data (one run per action in config.ACTIONS)
venv/bin/python scripts/collect_data.py --action hello

# 3. Train the model (saves held-out test set to models/test_set.npz)
venv/bin/python scripts/train.py

# 4. Evaluate on the held-out test set
venv/bin/python scripts/predict.py

# 5. Live webcam inference with temporal smoothing
venv/bin/python scripts/realtime_predict.py

# 6. Run unit tests
venv/bin/python -m unittest tests.test_far_bias tests.test_arm_depths
```

> **Note:** `data/`, `models/` artifacts are tied to the current 356-dim layout. After any `landmarks.extract()` change, re-collect data and retrain from scratch — do not mix old/new vectors. See `backend/CHANGELOG.md` for the full history.
