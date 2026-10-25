"""Standalone VSL inference: 12 frames or 12 featurized vectors -> prediction.

Inherits everything from vsl2: model layers (vsl2.model), feature extraction
(vsl2.landmarks), standardization (vsl2.dataset) and config (vsl2.defaults).
No vendored copies — loading a checkpoint uses the exact classes and
pipeline it was trained with.
"""

import base64
import json
import os
import sys

import cv2
import numpy as np
from keras.models import load_model

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from vsl2 import model as _vsl2_model
from vsl2 import defaults as cfg
from vsl2.landmarks import (
    BODY_ONLY_DIM,
    ArmDepthCalibrator,
    ArmDepthSmoother,
    create_model,
    detect,
    extract,
    feed_calibrator,
)


def _resolve_labels(model_path, n_outputs, labels=None):
    """Resolve class labels for a model, in priority order.

    1. Explicit `labels` argument.
    2. `actions.json` next to the model file.
    3. `train_record.json` next to the model file (written by backend train).
    4. Vendored cfg.ACTIONS fallback (training-time defaults).
    """
    if labels is not None:
        resolved = list(labels)
    else:
        model_dir = os.path.dirname(model_path)
        resolved = None
        actions_path = os.path.join(model_dir, "actions.json")
        if os.path.exists(actions_path):
            with open(actions_path, encoding="utf-8") as f:
                resolved = list(json.load(f))
        else:
            record_path = os.path.join(model_dir, "train_record.json")
            if os.path.exists(record_path):
                with open(record_path, encoding="utf-8") as f:
                    resolved = list(json.load(f).get("actions", [])) or None
        if resolved is None:
            resolved = list(cfg.ACTIONS)
    if len(resolved) != n_outputs:
        raise ValueError(
            f"Model has {n_outputs} outputs but {len(resolved)} labels "
            f"resolved ({resolved}). Pass matching `labels` or place an "
            f"actions.json beside the model.")
    return resolved


_CUSTOM_OBJECTS = {"vsl>WeightedMean": _vsl2_model.WeightedMean,
                   "vsl>Slice": _vsl2_model.Slice,
                   "vsl>FrameVelocity": _vsl2_model.FrameVelocity}


from vsl2.dataset import standardize_features as _standardize_features


def _standardize(X, mean, std):
    """Per-feature standardization ignoring missing keypoints (vsl2.dataset)."""
    X_norm, _, _ = _standardize_features(
        np.asarray(X, dtype=np.float32), mean=mean, std=std)
    return np.asarray(X_norm, dtype=np.float32).reshape(X.shape)


class VSLInference:
    """Single-shot sign classifier over exactly SEQUENCE_LEN frames.

    Usage:
        with VSLInference("/path/to/model.keras") as vsl:
            out = vsl.predict(frames)  # {"label", "confidence", "probs"}
    """

    def __init__(self, model_path, stats_path=None, labels=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'.")
        if stats_path is None:
            stats_path = os.path.join(os.path.dirname(model_path),
                                      "feature_stats.npz")
            feat_mean, feat_std = None, None
            if os.path.exists(stats_path):
                stats = np.load(stats_path)
                feat_mean, feat_std = stats["mean"], stats["std"]
        else:
            if not os.path.exists(stats_path):
                raise FileNotFoundError(
                    f"Feature stats not found at '{stats_path}'.")
            stats = np.load(stats_path)
            feat_mean, feat_std = stats["mean"], stats["std"]

        model = load_model(model_path, custom_objects=_CUSTOM_OBJECTS)
        in_shape = model.input_shape
        if tuple(in_shape) != (None, cfg.SEQUENCE_LEN, BODY_ONLY_DIM):
            raise ValueError(
                f"Model expects input {in_shape}, "
                f"need (None, {cfg.SEQUENCE_LEN}, {BODY_ONLY_DIM}).")
        self._labels = _resolve_labels(
            model_path, model.output_shape[-1], labels)

        self._model = model
        self._feat_mean = feat_mean
        self._feat_std = feat_std
        self._closed = False
        self.holistic = create_model()

    def predict(self, frames):
        """Classify 12 BGR frames. Returns {"label", "confidence", "probs"}.

        label is "" when no hand is visible or confidence is below threshold.
        Input frames are never mutated.
        """
        if self.holistic is None:
            raise RuntimeError("VSLInference is closed.")
        if (not isinstance(frames, (list, tuple))
                or len(frames) != cfg.SEQUENCE_LEN):
            raise ValueError(
                f"frames must be a list of exactly {cfg.SEQUENCE_LEN} "
                f"BGR frames, got "
                f"{len(frames) if isinstance(frames, (list, tuple)) else type(frames)}.")
        for i, f in enumerate(frames):
            if not isinstance(f, np.ndarray) or f.ndim != 3:
                raise ValueError(f"frames[{i}] is not a valid BGR image ndarray.")

        calibrator = ArmDepthCalibrator(warmup_frames=cfg.ARM_DEPTH_CAL_WARMUP)
        smoother = ArmDepthSmoother(alpha=cfg.ARM_DEPTH_EMA_ALPHA,
                                    window=cfg.ARM_DEPTH_SMOOTH_WINDOW)
        seq = []
        any_hand = False
        for frame in frames:
            _, results = detect(frame, self.holistic)
            has_hand = (results.left_hand_landmarks is not None
                        or results.right_hand_landmarks is not None)
            any_hand = any_hand or has_hand
            feed_calibrator(results, frame.shape[:2], calibrator)
            vec = extract(results, frame_shape=frame.shape[:2],
                          neutral_lens=[calibrator.get(0), calibrator.get(1)])
            vec[102:104] = [smoother.update(s, vec[102 + s]) for s in (0, 1)]
            seq.append(vec)

        n = len(self._labels)
        if not any_hand:
            return {"label": "", "confidence": 0.0,
                    "probs": [0.0] * n}

        X = np.array([seq], dtype="float32")
        return self._run_model(X)

    def _run_model(self, X):
        """Standardize a (1, 12, 356) batch-of-one → model → label dict."""
        if self._feat_mean is not None and self._feat_std is not None:
            X = _standardize(X, self._feat_mean, self._feat_std)
        probs = np.asarray(self._model.predict(X, verbose=0)[0], dtype=np.float32)
        best_idx = int(np.argmax(probs))
        conf = float(probs[best_idx])
        if conf < float(cfg.threshold_for(best_idx)):
            return {"label": "", "confidence": conf,
                    "probs": [float(p) for p in probs]}
        return {"label": str(self._labels[best_idx]), "confidence": conf,
                "probs": [float(p) for p in probs]}

    def predict_features(self, vectors):
        """Classify 12 pre-extracted 356-dim feature vectors.

        Same vsl2 feature contract as predict() produces internally
        (pose [0:18] | lh [18:60] | rh [60:102] | arm_depths [102:104] |
        sh_off_lh [104:230] | sh_off_rh [230:356]). Skips MediaPipe
        entirely — for clients (e.g. the webapp) that extract landmarks
        themselves. Returns {"label", "confidence", "probs"}.
        """
        if self._closed:
            raise RuntimeError("VSLInference is closed.")
        X = np.asarray(vectors, dtype=np.float32)
        if X.shape != (cfg.SEQUENCE_LEN, BODY_ONLY_DIM):
            raise ValueError(
                f"features must have shape ({cfg.SEQUENCE_LEN}, "
                f"{BODY_ONLY_DIM}), got {X.shape}.")
        if not np.any(X):
            return {"label": "", "confidence": 0.0,
                    "probs": [0.0] * len(self._labels)}
        return self._run_model(X[None, :, :])

    @staticmethod
    def _decode_one(raw, index):
        """Decode one raw image (bytes or base64 str) to a BGR MatLike."""
        if isinstance(raw, str):
            if "," in raw and raw.startswith("data:"):
                raw = raw.split(",", 1)[1]
            try:
                raw = base64.b64decode(raw, validate=True)
            except Exception:
                raise ValueError(
                    f"frames[{index}] is not valid base64 image data.")
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise ValueError(
                f"frames[{index}] must be raw image bytes or a base64 "
                f"string, got {type(raw).__name__}.")
        buf = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(
                f"frames[{index}] could not be decoded as an image "
                f"(need JPEG/PNG bytes).")
        return frame

    def predict_bytes(self, frames):
        """Classify 12 raw image bytes. Returns {"label", "confidence", "probs"}.

        Accepts a list/tuple of exactly SEQUENCE_LEN items, each raw
        JPEG/PNG bytes (as sent by a webapp) or a base64 string of them.
        Decodes to BGR MatLike via cv2.imdecode, then runs predict().
        """
        if (not isinstance(frames, (list, tuple))
                or len(frames) != cfg.SEQUENCE_LEN):
            raise ValueError(
                f"frames must be a list of exactly {cfg.SEQUENCE_LEN} "
                f"raw images, got "
                f"{len(frames) if isinstance(frames, (list, tuple)) else type(frames)}.")
        decoded = [self._decode_one(raw, i) for i, raw in enumerate(frames)]
        return self.predict(decoded)

    def close(self):
        """Release the MediaPipe holistic object. Idempotent."""
        holistic, self.holistic = self.holistic, None
        if holistic is not None:
            try:
                holistic.close()
            except Exception:
                pass
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
