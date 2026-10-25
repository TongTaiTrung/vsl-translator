"""Batch inference: model loading + per-frame pipeline + sequence prediction.

Mirrors backend/scripts/_common.py + convenient_predict.py predict_raw,
minus Groq cleanup (vsl2 is Core ML only).
"""

import os

import numpy as np

from . import defaults as cfg
from . import model as _model_mod  # registers custom layers for load_model
from .landmarks import (
    BODY_ONLY_DIM,
    ArmDepthCalibrator,
    ArmDepthSmoother,
    create_model,
    detect,
    extract,
    feed_calibrator,
)
from .dataset import standardize_features

_CUSTOM_OBJECTS = {
    "WeightedMean": _model_mod.WeightedMean,
    "FrameVelocity": _model_mod.FrameVelocity,
    "Slice": _model_mod.Slice,
}

_MODEL = None
_FEAT_MEAN = None
_FEAT_STD = None
_MODEL_PATH = None


def load_classifier(model_path=None, stats_path=None, strict_model=True):
    """Load model + feature stats. Returns (model, feat_mean, feat_std)."""
    from keras.models import load_model
    if model_path is None:
        model_path = os.path.join(cfg.MODELS_DIR, cfg.MODEL_FILENAME)
    if stats_path is None:
        stats_path = os.path.join(cfg.MODELS_DIR, "feature_stats.npz")
    if not os.path.exists(model_path):
        if strict_model:
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. Run vsl2 train first.")
        return None, None, None
    model = load_model(model_path, custom_objects=_CUSTOM_OBJECTS)
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        return model, stats["mean"], stats["std"]
    return model, None, None


def _ensure_model(model_path=None):
    global _MODEL, _FEAT_MEAN, _FEAT_STD, _MODEL_PATH
    if _MODEL is not None and _MODEL_PATH == model_path:
        return _MODEL, _FEAT_MEAN, _FEAT_STD
    _MODEL, _FEAT_MEAN, _FEAT_STD = load_classifier(model_path)
    _MODEL_PATH = model_path
    return _MODEL, _FEAT_MEAN, _FEAT_STD


def extract_frame_vector(results, frame_shape, calibrator, smoother):
    """One frame: feed calibrator → extract → smooth arm_depths [102:104]."""
    feed_calibrator(results, frame_shape, calibrator)
    vec = extract(results, frame_shape=frame_shape,
                  neutral_lens=[calibrator.get(0), calibrator.get(1)])
    vec[102:104] = [smoother.update(s, vec[102 + s]) for s in (0, 1)]
    return vec


def standardize(X, mean, std):
    if mean is None or std is None:
        return X
    X, _, _ = standardize_features(X, mean=mean, std=std)
    return X


def predict_raw(frames, model_path=None):
    """Run the model over SEQUENCE_LEN frames. Returns (word, probs).

    word is "" when uncertain (below CONFIDENCE_THRESHOLD) or when no hand
    is visible in the sequence. probs is the raw softmax vector.
    """
    if not isinstance(frames, (list, tuple)) or len(frames) != cfg.SEQUENCE_LEN:
        raise ValueError(
            f"frames must be a list of exactly {cfg.SEQUENCE_LEN} "
            f"OpenCV frames, got {len(frames) if isinstance(frames, (list, tuple)) else type(frames)}.")
    for i, f in enumerate(frames):
        if not isinstance(f, np.ndarray) or f.ndim != 3:
            raise ValueError(f"frames[{i}] is not a valid BGR image ndarray.")

    model, feat_mean, feat_std = _ensure_model(model_path)
    calibrator = ArmDepthCalibrator(warmup_frames=cfg.ARM_DEPTH_CAL_WARMUP)
    smoother = ArmDepthSmoother(alpha=cfg.ARM_DEPTH_EMA_ALPHA,
                                window=cfg.ARM_DEPTH_SMOOTH_WINDOW)

    seq = []
    any_hand = False
    with create_model() as holistic:
        for frame in frames:
            _, results = detect(frame, holistic)
            has_hand = (results.left_hand_landmarks is not None
                        or results.right_hand_landmarks is not None)
            any_hand = any_hand or has_hand
            seq.append(extract_frame_vector(results, frame.shape[:2],
                                            calibrator, smoother))

    if not any_hand:
        return "", np.zeros(len(cfg.ACTIONS), dtype=np.float32)

    X = np.array([seq], dtype="float32")
    X = standardize(X, feat_mean, feat_std)
    probs = np.asarray(model.predict(X, verbose=0)[0], dtype=np.float32)
    best_idx = int(np.argmax(probs))
    if float(probs[best_idx]) < float(cfg.CONFIDENCE_THRESHOLD):
        return "", probs
    return str(cfg.ACTIONS[best_idx]), probs


def _as_feature_matrix(vectors):
    """Normalize 12 featurized vectors to a (12, 356) float32 array.

    Accepts a (12, 356) ndarray or a list/tuple of 12 (356,) arrays.
    Raises ValueError on wrong count or wrong feature dim.
    """
    X = np.asarray(vectors, dtype="float32")
    if X.shape != (cfg.SEQUENCE_LEN, BODY_ONLY_DIM):
        raise ValueError(
            f"features must have shape ({cfg.SEQUENCE_LEN}, {BODY_ONLY_DIM}), "
            f"got {X.shape}.")
    return X


def _decode_probs(probs):
    """Softmax vector → (word, probs); "" when below confidence threshold."""
    probs = np.asarray(probs, dtype=np.float32).ravel()
    best_idx = int(np.argmax(probs))
    if float(probs[best_idx]) < float(cfg.CONFIDENCE_THRESHOLD):
        return "", probs
    return str(cfg.ACTIONS[best_idx]), probs


def predict_features(vectors, model_path=None):
    """Predict over 12 featurized vectors. Returns {"raw", "probs"}.

    Skips MediaPipe entirely: stacks the vectors, standardizes with train
    stats (zero-mask preserved), runs the model. Returns "" when every
    vector is zero (no hand) or confidence is below threshold.
    """
    X = _as_feature_matrix(vectors)
    if not np.any(X):
        return {"raw": "",
                "probs": [0.0] * len(cfg.ACTIONS)}
    model, feat_mean, feat_std = _ensure_model(model_path)
    Xs = standardize(X[None, :, :], feat_mean, feat_std)
    probs = np.asarray(model.predict(Xs, verbose=0)[0], dtype=np.float32)
    raw, probs = _decode_probs(probs)
    return {"raw": raw, "probs": [float(p) for p in probs]}


def _load_seq_features(seq_dir):
    """Load 0..11.npy from a sequence dir. Returns list of 12 arrays."""
    vecs = []
    for i in range(cfg.SEQUENCE_LEN):
        path = os.path.join(seq_dir, f"{i}.npy")
        if not os.path.isfile(path):
            return None
        vecs.append(np.load(path))
    return vecs


def _looks_like_features(source):
    """True if source is 12 feature vectors (not 12 BGR images)."""
    if isinstance(source, np.ndarray):
        return source.ndim == 2
    if isinstance(source, (list, tuple)) and len(source) == cfg.SEQUENCE_LEN:
        first = source[0]
        return isinstance(first, np.ndarray) and first.ndim == 1
    return False


def predict(source, model_path=None):
    """Predict over a video file, a folder of images, or 12 feature vectors.

    Feature input: list of 12 (356,) arrays, a (12, 356) ndarray, or a
    sequence dir holding 0..11.npy. Returns {"raw": word, "probs": [...]}.
    """
    import cv2
    import glob as _glob
    if _looks_like_features(source):
        return predict_features(source, model_path=model_path)
    if isinstance(source, str) and os.path.isdir(source):
        vecs = _load_seq_features(source)
        if vecs is not None:
            return predict_features(vecs, model_path=model_path)
        paths = sorted(_glob.glob(os.path.join(source, "*")))
        imgs = [cv2.imread(p) for p in paths[:cfg.SEQUENCE_LEN]]
        imgs = [im for im in imgs if im is not None]
    else:
        cap = cv2.VideoCapture(source)
        imgs = []
        while len(imgs) < cfg.SEQUENCE_LEN:
            ret, fr = cap.read()
            if not ret:
                break
            imgs.append(fr)
        cap.release()
    if len(imgs) != cfg.SEQUENCE_LEN:
        raise ValueError(
            f"Need exactly {cfg.SEQUENCE_LEN} frames, got {len(imgs)}.")
    raw, probs = predict_raw(imgs, model_path=model_path)
    return {"raw": raw, "probs": [float(p) for p in np.asarray(probs).ravel()]}
