"""Minimal realtime webcam loop: sliding window + vote buffer + sentence builder.

Gọn nhẹ, không phụ thuộc stream_session/server của backend.
"""

import collections

import cv2
import numpy as np

from . import defaults as cfg
from .landmarks import (
    ArmDepthCalibrator,
    ArmDepthSmoother,
    create_model,
    detect,
    draw_landmarks_new,
)
from .predict import extract_frame_vector, standardize, _ensure_model


def run_realtime(model_path=None, on_word=None):
    """Open webcam, predict continuously. Returns on 'q' or KeyboardInterrupt.

    on_word: optional callback(word, probs) fired when the vote buffer
    commits a new word into the sentence.
    """
    model, feat_mean, feat_std = _ensure_model(model_path)
    calibrator = ArmDepthCalibrator(warmup_frames=cfg.ARM_DEPTH_CAL_WARMUP)
    smoother = ArmDepthSmoother(alpha=cfg.ARM_DEPTH_EMA_ALPHA,
                                window=cfg.ARM_DEPTH_SMOOTH_WINDOW)

    window = collections.deque(maxlen=cfg.SEQUENCE_LEN)
    votes = collections.deque(maxlen=cfg.SMOOTHING_WINDOW)
    sentence = []
    stable = None
    stable_count = 0

    cap = cv2.VideoCapture(0)
    try:
        with create_model() as holistic:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                img, results = detect(frame, holistic)
                img = draw_landmarks_new(img, results)
                vec = extract_frame_vector(results, frame.shape[:2],
                                           calibrator, smoother)
                window.append(vec)

                word = ""
                if len(window) == cfg.SEQUENCE_LEN:
                    X = standardize(
                        np.array([list(window)], dtype="float32"),
                        feat_mean, feat_std)
                    probs = np.asarray(model.predict(X, verbose=0)[0])
                    best = int(np.argmax(probs))
                    if float(probs[best]) >= cfg.threshold_for(best):
                        word = str(cfg.ACTIONS[best])
                votes.append(word)
                if votes:
                    top = max(set(votes), key=list(votes).count)
                    if top and top == stable:
                        stable_count += 1
                    elif top:
                        stable, stable_count = top, 1
                    if stable and stable_count >= cfg.SWITCH_STABLE_FRAMES:
                        if not sentence or sentence[-1] != stable:
                            sentence.append(stable)
                            if on_word:
                                on_word(stable, None)
                        stable, stable_count = None, 0

                cv2.putText(img, f"{word} | {' '.join(sentence)}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                            cv2.LINE_AA)
                cv2.imshow("vsl2 realtime (q to quit)", img)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return sentence
