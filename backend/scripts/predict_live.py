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

import warnings
warnings.warn(
    "predict_live.py is deprecated; use realtime_predict.py (StreamSession) instead.",
    DeprecationWarning, stacklevel=2)

from collections import deque

import cv2
import numpy as np

import config as cfg
from scripts._common import extract_frame_vector, load_classifier, standardize
from vsl_translator.landmarks import (
    create_model, detect, draw_landmarks_new,
    ArmDepthCalibrator, ArmDepthSmoother,
)


def smooth_label(history):
    """Majority vote over recent (label, confidence) pairs."""
    if not history:
        return None, 0.0
    labels = [l for l, _ in history]
    winner = max(set(labels), key=labels.count)
    conf = float(np.mean([c for l, c in history if l == winner]))
    return winner, conf


def main():
    try:
        model, mean, std = load_classifier()
    except FileNotFoundError:
        print("Error: model or feature_stats.npz missing. Train first via scripts/train.py.")
        return

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
            vec = extract_frame_vector(results, frame.shape[:2],
                                       calibrator, smoother)
            window.append(vec)

            text, color = "Collecting frames...", (200, 200, 200)
            if len(window) == cfg.SEQUENCE_LEN:
                seq = np.expand_dims(np.array(window, dtype=np.float32), axis=0)
                seq = standardize(seq, mean, std)
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
