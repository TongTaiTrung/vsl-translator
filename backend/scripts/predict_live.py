#!/usr/bin/env python3
"""Live webcam sign-language inference with temporal smoothing.

Sliding window of SEQUENCE_LEN frames → model prediction per window.
Raw predictions are smoothed by majority vote over SMOOTHING_WINDOW windows,
and labels below CONFIDENCE_THRESHOLD are reported as uncertain.
Press 'q' to quit.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from collections import deque

import cv2
import numpy as np
from keras.models import load_model

import config as cfg
from vsl_translator.landmarks import (
    create_model, detect, draw_landmarks_new, extract, feed_calibrator,
    ArmDepthCalibrator, ArmDepthSmoother,
)
from vsl_translator.dataset import standardize_features


def smooth_label(history):
    """Majority vote over recent (label, confidence) pairs."""
    if not history:
        return None, 0.0
    labels = [l for l, _ in history]
    winner = max(set(labels), key=labels.count)
    conf = float(np.mean([c for l, c in history if l == winner]))
    return winner, conf


def main():
    model_path = os.path.join(cfg.MODELS_DIR, cfg.MODEL_FILENAME)
    stats_path = os.path.join(cfg.MODELS_DIR, "feature_stats.npz")
    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        print("Error: model or feature_stats.npz missing. Train first via scripts/train.py.")
        return

    model = load_model(model_path)
    stats = np.load(stats_path)
    mean, std = stats["mean"], stats["std"]

    window = deque(maxlen=cfg.SEQUENCE_LEN)
    votes = deque(maxlen=cfg.SMOOTHING_WINDOW)
    calibrator = ArmDepthCalibrator(warmup_frames=cfg.ARM_DEPTH_CAL_WARMUP)
    smoother = ArmDepthSmoother(alpha=cfg.ARM_DEPTH_EMA_ALPHA,
                                window=cfg.ARM_DEPTH_SMOOTH_WINDOW)

    cap = cv2.VideoCapture(0)
    with create_model() as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            img, results = detect(frame, holistic)
            img = draw_landmarks_new(img, results)
            feed_calibrator(results, frame.shape[:2], calibrator)
            vec = extract(results, frame_shape=frame.shape[:2],
                          neutral_lens=[calibrator.get(0), calibrator.get(1)])
            vec[102:104] = [smoother.update(s, vec[102 + s]) for s in (0, 1)]
            window.append(vec)

            text, color = "Collecting frames...", (200, 200, 200)
            if len(window) == cfg.SEQUENCE_LEN:
                seq = np.expand_dims(np.array(window, dtype=np.float32), axis=0)
                seq, _, _ = standardize_features(seq, mean=mean, std=std)
                probs = model.predict(seq, verbose=0)[0]
                pred, conf = int(np.argmax(probs)), float(np.max(probs))
                votes.append((pred, conf))
                label, sconf = smooth_label(list(votes))
                if sconf >= cfg.CONFIDENCE_THRESHOLD:
                    text = f"{cfg.ACTIONS[label]} ({sconf:.0%})"
                    color = (0, 255, 0)
                else:
                    text = f"Uncertain (top: {cfg.ACTIONS[label]} {sconf:.0%})"
                    color = (0, 165, 255)

            cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2, cv2.LINE_AA)
            cv2.imshow("VSL Translator", img)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
