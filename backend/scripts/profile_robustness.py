#!/usr/bin/env python3
"""Per-augmentation robustness profile.

Sweeps each perturbation in isolation (plus a combined row) over the
held-out test_set.npz and reports accuracy at each strength. Tells us
which distortion the model is actually fragile to before any retraining.

Run from backend/:  venv/bin/python scripts/profile_robustness.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from keras.models import load_model

import config as cfg
from vsl_translator.model import WeightedMean, Slice
from vsl_translator.dataset import (
    standardize_features,
    _add_noise, _scale_jitter, _time_warp, _random_shift,
    _random_frame_drop, _random_hand_dropout, _random_hand_swap,
    inject_inactive_side,
)
from vsl_translator.landmarks import (
    POSE_DIM, HAND_DIM, slice_bounds)

MODEL = None
STATS = None
LABELS = list(cfg.ACTIONS)


def predict_raw(X):
    Xn, _, _ = standardize_features(
        X.astype(np.float32), mean=STATS["mean"], std=STATS["std"])
    return MODEL.predict(Xn, verbose=0)


def acc_of(Xp, y_true):
    return float((predict_raw(Xp).argmax(axis=1) == y_true).mean())


def burst_hand_drop(seq, length=4, seed=0):
    """Zero one whole hand for `length` consecutive frames."""
    rng = np.random.RandomState(seed)
    out = seq.copy()
    _, _, _, sh_start, _ = slice_bounds()
    hand = rng.choice(['lh', 'rh'])
    start = rng.randint(0, 12 - length + 1)
    end = start + length
    if hand == 'lh':
        out[start:end, POSE_DIM:POSE_DIM + HAND_DIM] = 0.0
        out[start:end, sh_start:sh_start + 126] = 0.0
    else:
        out[start:end, POSE_DIM + HAND_DIM:POSE_DIM + 2 * HAND_DIM] = 0.0
        out[start:end, sh_start + 126:sh_start + 252] = 0.0
    return out


def speed_cut(seq, cut, seed=0):
    """Truncate head or tail by `cut` frames, pad with zeros."""
    rng = np.random.RandomState(seed)
    if rng.rand() < 0.5:
        return np.concatenate(
            [np.zeros_like(seq[:cut]), seq[:-cut]], axis=0)
    return np.concatenate(
        [seq[cut:], np.zeros_like(seq[:cut])], axis=0)


def main():
    global MODEL, STATS
    model_path = os.path.join(cfg.MODELS_DIR, cfg.MODEL_FILENAME)
    STATS = np.load(os.path.join(cfg.MODELS_DIR, "feature_stats.npz"))
    test = np.load(os.path.join(cfg.MODELS_DIR, "test_set.npz"))
    X_test, y_test = test["X"].astype(np.float32), test["y"]
    y_true = y_test.argmax(axis=1)
    MODEL = load_model(model_path, custom_objects={
        "vsl>WeightedMean": WeightedMean, "vsl>Slice": Slice})

    N = len(X_test)
    seeds = np.random.RandomState(7).randint(0, 10 ** 6, size=N)
    print(f"Clean acc: {acc_of(X_test, y_true):.4f} ({N} samples)\n")

    def sweep(name, fn, strengths):
        row = []
        for s in strengths:
            Xp = np.stack(
                [fn(X_test[i], s, int(seeds[i])) for i in range(N)])
            row.append(acc_of(Xp, y_true))
        cells = "  ".join(f"{s}: {a:.3f}" for s, a in zip(strengths, row))
        print(f"  {name:<22s} {cells}")

    print("  (each row = one perturbation in isolation, strength sweep)")
    sweep("noise_std", lambda x, s, sd: _add_noise(
        (np.random.seed(sd), x)[1], std=s), [0.05, 0.09, 0.15, 0.25])
    sweep("time_warp_sigma", lambda x, s, sd: _time_warp(
        (np.random.seed(sd), x)[1], sigma=s), [0.2, 0.35, 0.5, 0.7])
    sweep("frame_drop_p", lambda x, s, sd: _random_frame_drop(
        (np.random.seed(sd), x)[1], p=s), [0.05, 0.15, 0.25, 0.4])
    sweep("shift_range", lambda x, s, sd: _random_shift(
        (np.random.seed(sd), x)[1], lo=-s, hi=s),
        [0.02, 0.04, 0.08, 0.12])
    sweep("scale_jitter_std", lambda x, s, sd: _scale_jitter(
        (np.random.seed(sd), x)[1], std=s), [0.05, 0.08, 0.15, 0.25])
    sweep("burst_hand_frames", burst_hand_drop, [2, 4, 6, 8])
    sweep("speed_cut_frames", speed_cut, [1, 2, 3, 4])
    # hand_dropout / hand_swap / inactive_inject take prob, not strength —
    # evaluate at p=1.0 (always applied) to see worst-case effect
    for nm, fn in [("hand_dropout_p1.0", _random_hand_dropout),
                   ("hand_swap_p1.0", _random_hand_swap),
                   ("inactive_inject_p1.0", inject_inactive_side)]:
        Xp = np.stack([
            fn((np.random.seed(int(seeds[i])), X_test[i])[1], p=1.0)
            for i in range(N)])
        print(f"  {nm:<22s} p=1.0: {acc_of(Xp, y_true):.3f}")


if __name__ == "__main__":
    main()
