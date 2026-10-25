"""Shared inference helpers: model loading + per-frame feature extraction.

Used by scripts/stream_session.py, predict_live.py, convenient_predict.py
and server.py so the calibrator → extract → smoother → standardize pipeline
stays identical everywhere. No behavior change — pure dedup.
"""

import os

import numpy as np

import config as cfg


def load_classifier(model_path=None, stats_path=None, strict_model=True):
    """Load model + feature stats. Returns (model, feat_mean, feat_std).

    strict_model=True raises FileNotFoundError when the model is missing
    (train.py must run first); feature stats stay optional (None, None).
    """
    from keras.models import load_model
    if model_path is None:
        model_path = os.path.join(cfg.MODELS_DIR, cfg.MODEL_FILENAME)
    if stats_path is None:
        stats_path = os.path.join(cfg.MODELS_DIR, "feature_stats.npz")
    if not os.path.exists(model_path):
        if strict_model:
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. Run scripts/train.py first.")
        return None, None, None
    model = load_model(model_path)
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        return model, stats["mean"], stats["std"]
    return model, None, None


def extract_frame_vector(results, frame_shape, calibrator, smoother):
    """One frame: feed calibrator → extract → smooth arm_depths [102:104]."""
    from vsl_translator.landmarks import extract, feed_calibrator
    feed_calibrator(results, frame_shape, calibrator)
    vec = extract(results, frame_shape=frame_shape,
                  neutral_lens=[calibrator.get(0), calibrator.get(1)])
    vec[102:104] = [smoother.update(s, vec[102 + s]) for s in (0, 1)]
    return vec


def standardize(X, mean, std):
    """standardize_features with zero-mask preservation (thin wrapper)."""
    from vsl_translator.dataset import standardize_features
    if mean is None or std is None:
        return X
    X, _, _ = standardize_features(X, mean=mean, std=std)
    return X
