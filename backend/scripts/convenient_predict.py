#!/usr/bin/env python3
"""Batch (non-realtime) prediction over exactly SEQUENCE_LEN OpenCV frames.

Mirrors the inference path of scripts/realtime_predict.py:
  detect -> feed_calibrator -> extract -> ArmDepthSmoother ->
  standardize_features (0.0 zero-mask preserved) -> model.predict ->
  CONFIDENCE_THRESHOLD gating.

Plus a synchronous Groq cleanup (same prompt/model as realtime) that
turns the raw word into the "smart" sentence.

Typical use:
    from scripts.convenient_predict import convenient_predict
    out = convenient_predict(frames)  # frames: list of SEQUENCE_LEN BGR ndarrays
    out["frames"]  # the untouched input frames (same objects, never drawn on)
    out["raw"]     # predicted word or "" when uncertain / no hand
    out["smart"]   # Groq-cleaned text ("" when raw is empty or on error)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import config as cfg

_MODEL = None
_FEAT_MEAN = None
_FEAT_STD = None


def _ensure_model():
    """Lazy-load model + feature stats once per process."""
    global _MODEL, _FEAT_MEAN, _FEAT_STD
    if _MODEL is not None:
        return _MODEL, _FEAT_MEAN, _FEAT_STD
    from scripts._common import load_classifier
    _MODEL, _FEAT_MEAN, _FEAT_STD = load_classifier()
    return _MODEL, _FEAT_MEAN, _FEAT_STD


def predict_raw(frames):
    """Run the model over SEQUENCE_LEN frames. Returns (word, probs).

    word is "" when uncertain (below CONFIDENCE_THRESHOLD) or when no hand
    is visible in the sequence. probs is the raw softmax vector (or zeros
    when no prediction was attempted). Never raises on missing hands —
    only on bad input shape or missing model file.
    """
    if not isinstance(frames, (list, tuple)) or len(frames) != cfg.SEQUENCE_LEN:
        raise ValueError(
            f"frames must be a list of exactly {cfg.SEQUENCE_LEN} "
            f"OpenCV frames, got {len(frames) if isinstance(frames, (list, tuple)) else type(frames)}.")
    for i, f in enumerate(frames):
        if not isinstance(f, np.ndarray) or f.ndim != 3:
            raise ValueError(f"frames[{i}] is not a valid BGR image ndarray.")

    from scripts._common import extract_frame_vector, standardize
    from vsl_translator.landmarks import (
        create_model, detect, ArmDepthCalibrator, ArmDepthSmoother,
    )

    model, feat_mean, feat_std = _ensure_model()
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


def predict_smart(raw_word):
    """Clean one raw word via Groq. Returns cleaned text ("" on empty/error)."""
    from scripts.groq_client import predict_smart as _predict_smart
    return _predict_smart(raw_word)


def convenient_predict(frames):
    """Batch entry point: SEQUENCE_LEN frames -> intact frames + raw + smart.

    Args:
        frames: list of exactly cfg.SEQUENCE_LEN BGR OpenCV frames.

    Returns:
        dict with:
          "frames": the input list unchanged (same objects, never annotated),
          "raw": predicted word ("" when uncertain / no hand),
          "probs": list[float] softmax vector,
          "smart": Groq-cleaned text ("" when raw empty or Groq fails).
    """
    raw, probs = predict_raw(frames)
    smart = predict_smart(raw)
    return {
        "frames": list(frames),
        "raw": raw,
        "probs": [float(p) for p in np.asarray(probs).ravel()],
        "smart": smart,
    }


if __name__ == '__main__':
    import argparse
    import cv2
    ap = argparse.ArgumentParser(description="Batch predict over a video/image folder.")
    ap.add_argument("source", help="video file or folder of images")
    args = ap.parse_args()
    if os.path.isdir(args.source):
        import glob
        paths = sorted(glob.glob(os.path.join(args.source, "*")))
        imgs = [cv2.imread(p) for p in paths[:cfg.SEQUENCE_LEN]]
        imgs = [im for im in imgs if im is not None]
    else:
        cap = cv2.VideoCapture(args.source)
        imgs = []
        while len(imgs) < cfg.SEQUENCE_LEN:
            ret, fr = cap.read()
            if not ret:
                break
            imgs.append(fr)
        cap.release()
    if len(imgs) != cfg.SEQUENCE_LEN:
        print(f"Need exactly {cfg.SEQUENCE_LEN} frames, got {len(imgs)}.")
        sys.exit(1)
    out = convenient_predict(imgs)
    print(f"raw={out['raw']!r} smart={out['smart']!r}")
