#!/usr/bin/env python3
"""Evaluate the trained model on the held-out test set saved during training.

Uses models/test_set.npz written by scripts/train.py — never re-splits the
raw data, so results are comparable across runs and free of split mismatch.
Reports accuracy, per-class precision/recall/F1, confusion matrix, and
confidence statistics. Predictions below CONFIDENCE_THRESHOLD are counted
separately as abstentions.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import config as cfg
from vsl_translator.dataset import standardize_features


def main():
    model_path = os.path.join(cfg.MODELS_DIR, cfg.MODEL_FILENAME)
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Train first via scripts/train.py.")
        return

    test_path = os.path.join(cfg.MODELS_DIR, "test_set.npz")
    if not os.path.exists(test_path):
        print(f"Error: '{test_path}' not found. Re-train with the updated train.py first.")
        return

    stats_path = os.path.join(cfg.MODELS_DIR, "feature_stats.npz")
    if not os.path.exists(stats_path):
        print(f"Error: '{stats_path}' not found. Re-train with the updated train.py first.")
        return

    bundle = np.load(test_path)
    X_test_raw, y_test = bundle["X"], bundle["y"]
    stats = np.load(stats_path)
    if stats["mean"].shape[-1] != X_test_raw.shape[-1]:
        print(f"Error: feature_stats dim {stats['mean'].shape[-1]} != test data dim "
              f"{X_test_raw.shape[-1]}. Stats còn của bản train cũ — chạy lại scripts/train.py.")
        return

    model = load_model(model_path)
    expected_dim = model.input_shape[-1]
    if X_test_raw.shape[-1] != expected_dim:
        print(f"Error: test data dim {X_test_raw.shape[-1]} != model dim {expected_dim}. "
              "Re-train để đồng bộ.")
        return
    X_test, _, _ = standardize_features(X_test_raw, mean=stats["mean"], std=stats["std"])
    probs = model.predict(X_test, verbose=0)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)

    print(f"Test samples : {len(X_test)}")
    print(f"Accuracy     : {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred, target_names=list(cfg.ACTIONS), digits=4, zero_division=0
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nConfidence:")
    print(f"  mean={conf.mean():.4f}  min={conf.min():.4f}  max={conf.max():.4f}")
    abstained = int(np.sum(conf < cfg.CONFIDENCE_THRESHOLD))
    print(f"  Below threshold ({cfg.CONFIDENCE_THRESHOLD:.2f}): {abstained}/{len(conf)} abstentions")
    if abstained:
        kept = conf >= cfg.CONFIDENCE_THRESHOLD
        if kept.sum() > 0:
            print(f"  Accuracy on non-abstained: {accuracy_score(y_true[kept], y_pred[kept]):.4f}")


if __name__ == '__main__':
    main()
