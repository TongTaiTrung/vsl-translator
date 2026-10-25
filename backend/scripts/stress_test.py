#!/usr/bin/env python3
"""Stress test: evaluate the trained model under live-like perturbations.

Uses the held-out test_set.npz (never seen during training) and applies
stronger-than-train perturbations to simulate live conditions: heavier
time-warp, frame drops, hand jitter, speed variation.

Reports: clean accuracy, stress accuracy, per-class accuracy + confidence
(mean/min) under stress, and top confused pairs.

Run from backend/:  venv/bin/python scripts/stress_test.py [--repeats N]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
from keras.models import load_model

import config as cfg
from vsl_translator.model import WeightedMean, Slice, FrameVelocity
from vsl_translator.dataset import (
    standardize_features,
    _add_noise, _scale_jitter, _time_warp, _random_shift,
    _random_frame_drop,
)
from vsl_translator.landmarks import coord_mask as _coord_mask


def stress_one(seq, rng_seed):
    """Apply one round of live-like perturbation (stronger than train)."""
    np.random.seed(rng_seed)
    out = seq.copy()
    # Heavier hand jitter than train (0.05 -> 0.09)
    out = _add_noise(out, std=0.09)
    # Stronger temporal distortion than train (0.2 -> 0.35)
    if np.random.rand() < 0.7:
        out = _time_warp(out, sigma=0.35)
    # Heavier frame drop than train (0.05 -> 0.15)
    if np.random.rand() < 0.5:
        out = _random_frame_drop(out, p=0.15)
    # Burst tracking loss: zero one whole hand for 3-6 consecutive frames
    # (MediaPipe loses a hand mid-sign on live camera).
    if np.random.rand() < 0.6:
        from vsl_translator.landmarks import (
            POSE_DIM, HAND_DIM, slice_bounds)
        _, _, _, sh_start, _ = slice_bounds()
        hand = np.random.choice(['lh', 'rh'])
        start = np.random.randint(0, 9)
        length = np.random.randint(3, 7)
        end = min(12, start + length)
        if hand == 'lh':
            out[start:end, POSE_DIM:POSE_DIM + HAND_DIM] = 0.0
            out[start:end, sh_start:sh_start + 126] = 0.0
        else:
            out[start:end, POSE_DIM + HAND_DIM:POSE_DIM + 2 * HAND_DIM] = 0.0
            out[start:end, sh_start + 126:sh_start + 252] = 0.0
    # Speed variation: randomly truncate head/tail then re-pad with zeros
    # (simulates starting/stopping the sign early or late on live camera)
    if np.random.rand() < 0.5:
        cut = np.random.randint(1, 4)
        if np.random.rand() < 0.5:
            out = np.concatenate([np.zeros_like(out[:cut]), out[:-cut]], axis=0)
        else:
            out = np.concatenate([out[cut:], np.zeros_like(out[:cut])], axis=0)
    # Coordinate jitter
    if np.random.rand() < 0.5:
        out = _scale_jitter(out, std=0.08)
    if np.random.rand() < 0.5:
        out = _random_shift(out, lo=-0.04, hi=0.04)
    return out


def partial_windows(seq, rng_seed):
    """Simulate live sliding windows: embed the 12-frame sign in a longer
    stream with rest (zero) frames on both sides, take all 12-frame windows
    whose center-of-mass overlap with the sign is >= 50%.

    These partial windows are what the live deque actually feeds the model
    at sign boundaries — the main OOD gap vs clean batch evaluation.
    Returns list of (window, overlap_fraction).
    """
    np.random.seed(rng_seed)
    lead = np.random.randint(0, 13)
    trail = 12 - lead
    stream = np.concatenate(
        [np.zeros((lead, seq.shape[1]), dtype=np.float32),
         seq,
         np.zeros((trail, seq.shape[1]), dtype=np.float32)], axis=0)
    out = []
    for start in range(lead + trail + 1):
        w = stream[start:start + 12]
        # overlap = fraction of the 12 window covered by real sign frames
        s0, s1 = start, start + 12
        overlap = max(0, min(s1, lead + 12) - max(s0, lead)) / 12.0
        if overlap >= 0.5:
            out.append((w.copy(), overlap))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10,
                    help="stress copies per test sample (default 10)")
    args = ap.parse_args()

    model_path = os.path.join(cfg.MODELS_DIR, cfg.MODEL_FILENAME)
    stats = np.load(os.path.join(cfg.MODELS_DIR, "feature_stats.npz"))
    test = np.load(os.path.join(cfg.MODELS_DIR, "test_set.npz"))
    X_test_raw, y_test = test["X"], test["y"]
    labels = list(cfg.ACTIONS)
    n_classes = len(labels)
    y_true = y_test.argmax(axis=1)

    model = load_model(model_path,
                       custom_objects={"vsl>WeightedMean": WeightedMean,
                                       "vsl>Slice": Slice,
                                       "vsl>FrameVelocity": FrameVelocity})

    def predict_raw(X):
        Xn, _, _ = standardize_features(
            X.astype(np.float32), mean=stats["mean"], std=stats["std"])
        return model.predict(Xn, verbose=0)

    # ── Clean baseline ──
    probs_clean = predict_raw(X_test_raw)
    pred_clean = probs_clean.argmax(axis=1)
    clean_acc = (pred_clean == y_true).mean()
    print(f"Clean test acc: {clean_acc:.4f} ({len(X_test_raw)} samples)")

    # ── Stress ──
    R = args.repeats
    N = len(X_test_raw)
    rng = np.random.RandomState(1234)
    seeds = rng.randint(0, 10 ** 6, size=(N, R))
    X_stress = np.stack(
        [stress_one(X_test_raw[i], int(seeds[i, r]))
         for i in range(N) for r in range(R)]).astype(np.float32)
    y_stress_true = np.repeat(y_true, R)
    probs = predict_raw(X_stress)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    stress_acc = (pred == y_stress_true).mean()
    print(f"Stress test acc: {stress_acc:.4f} ({N}x{R} = {len(X_stress)} perturbed copies)")
    print()

    # ── Per-class stress accuracy + confidence ──
    print(f"  {'class':<12s} {'acc':>7s} {'conf_mean':>10s} {'conf_min':>9s} {'n':>5s}")
    for c in range(n_classes):
        m = y_stress_true == c
        acc = (pred[m] == c).mean() if m.sum() else float("nan")
        print(f"  {labels[c]:<12s} {acc:>7.3f} {conf[m].mean():>10.3f} "
              f"{conf[m].min():>9.3f} {m.sum():>5d}")
    print()

    # ── Top confused pairs (true -> pred) ──
    from collections import Counter
    wrong = Counter()
    for t, p in zip(y_stress_true, pred):
        if t != p:
            wrong[(t, p)] += 1
    print("  Top confused pairs (true -> pred):")
    for (t, p), cnt in wrong.most_common(5):
        print(f"    {labels[t]} -> {labels[p]}: {cnt}x "
              f"({cnt / (N * R / n_classes):.1%} of stressed {labels[t]})")
    if not wrong:
        print("    (none — no confusion under this stress level, increase repeats or noise)")

    # ── Confidence margin: mean gap between top-1 and top-2 probs ──
    part = np.partition(probs, -2, axis=1)
    margin = (part[:, -1] - part[:, -2]).mean()
    print(f"\n  Mean top1-top2 margin under stress: {margin:.4f}")

    # ── Partial windows (sign boundaries — the live OOD gap) ──
    print("\n── Partial sliding windows (sign embedded in rest frames) ──")
    rng_pw = np.random.RandomState(99)
    pw_X, pw_true, pw_ov = [], [], []
    for i in range(N):
        for w, ov in partial_windows(X_test_raw[i], int(rng_pw.randint(0, 10 ** 6))):
            pw_X.append(w)
            pw_true.append(y_true[i])
            pw_ov.append(ov)
    pw_X = np.stack(pw_X).astype(np.float32)
    pw_true = np.array(pw_true)
    pw_probs = predict_raw(pw_X)
    pw_pred = pw_probs.argmax(axis=1)
    pw_acc = (pw_pred == pw_true).mean()
    print(f"  Partial-window acc: {pw_acc:.4f} ({len(pw_X)} windows)")
    print(f"  {'class':<12s} {'acc':>7s} {'n':>5s}")
    for c in range(n_classes):
        m = pw_true == c
        acc = (pw_pred[m] == c).mean() if m.sum() else float("nan")
        print(f"  {labels[c]:<12s} {acc:>7.3f} {m.sum():>5d}")
    from collections import Counter as _Counter2
    pw_wrong = _Counter2()
    for t, p in zip(pw_true, pw_pred):
        if t != p:
            pw_wrong[(t, p)] += 1
    print("  Top confused pairs (true -> pred) on partial windows:")
    for (t, p), cnt in pw_wrong.most_common(5):
        print(f"    {labels[t]} -> {labels[p]}: {cnt}x")
    if not pw_wrong:
        print("    (none)")


if __name__ == "__main__":
    main()
