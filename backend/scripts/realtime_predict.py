#!/usr/bin/env python3
"""Realtime prediction using webcam and the trained LSTM model.

The script captures frames from the webcam, extracts MediaPipe holistic keypoints,
stores the last `SEQUENCE_LEN` vectors in a sliding window, and runs the model
once the window is full.

Includes:
  - Strict 0.0 zero-mask preservation during feature standardization
  - Rest-pose and hand presence detection to prevent idle false positives
  - Confidence thresholding to avoid biased predictions on low-confidence frames
  - Buffer clearing on resting to avoid bleeding between consecutive gestures

Press 'q' to quit.

Sentence builder: whenever the voted prediction reaches CONFIDENCE_THRESHOLD
(70%), the word is committed to a sentence overlay at the bottom of the video
immediately — no stability waiting. A word is appended only once until the
winner changes or hands disappear, so holding one gesture does not spam it.
Keys (OpenCV window must have focus):
  Backspace: remove last word | C: clear sentence | Q: quit
"""

import sys, os
import threading
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from keras.models import load_model

import config as cfg
from scripts.overlay import (
    SENTENCE_MAX_LINES,
    draw_confidence_bars,
    draw_sentence,
    draw_sentence_smart,
    put_text_unicode,
    text_height,
    text_width,
)
from vsl_translator.landmarks import (
    create_model, detect, draw_landmarks_new, extract, feed_calibrator,
    ArmDepthCalibrator, ArmDepthSmoother,
)
from vsl_translator.dataset import standardize_features

# Overlay drawing lives in scripts/overlay.py (re-exported above for
# backward compatibility: `from scripts.realtime_predict import draw_sentence`).
from scripts import overlay as _overlay

_get_font = _overlay._get_font
FONT_PATH = _overlay.FONT_PATH
FONT_SIZE = _overlay.FONT_SIZE

# ── Smart sentence (Groq LLM cleanup) ─────────────────────────────────────────
# Settings live in config.py (env-overridable: GROQ_MODEL, GROQ_TIMEOUT,
# GROQ_RETRIES, SMART_CALL_INTERVAL, SMART_RETRY_INTERVAL, SMART_POLL_INTERVAL).
# Implementation lives in scripts/groq_client.py; re-exported here for
# backward compatibility (`from scripts.realtime_predict import ...`).
from scripts.groq_client import (
    PROMPT_PATH,
    SmartSentence,
    _call_groq,
    _load_prompt_template,
    call_groq,
    load_prompt_template,
    smart_worker,
)


def main():
    # ── Load model ────────────────────────────────────────────────────────────
    model_path = os.path.join(cfg.MODELS_DIR, cfg.MODEL_FILENAME)
    if not os.path.exists(model_path):
        print(f"Model not found at '{model_path}'. Run scripts/train.py first.")
        return
    model = load_model(model_path)

    # ── Load feature standardisation stats ────────────────────────────────────
    stats_path = os.path.join(cfg.MODELS_DIR, "feature_stats.npz")
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        feat_mean, feat_std = stats["mean"], stats["std"]
        print("Feature stats loaded for standardisation.")
    else:
        feat_mean, feat_std = None, None
        print("WARNING: feature_stats.npz not found – running without standardisation.")

    print("Model loaded. Press 'q' in the video window to quit.")
    print("Backspace: xoa tu cuoi | C: xoa ca cau | Q: thoat "
          "(cua so video phai co focus)")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera. Check camera permissions in "
              "System Settings → Privacy & Security → Camera.")
        return

    # Persistent prediction state
    seq = deque(maxlen=cfg.SEQUENCE_LEN)
    votes = deque(maxlen=cfg.SMOOTHING_WINDOW)
    calibrator = ArmDepthCalibrator(warmup_frames=cfg.ARM_DEPTH_CAL_WARMUP)
    smoother = ArmDepthSmoother(alpha=cfg.ARM_DEPTH_EMA_ALPHA,
                                window=cfg.ARM_DEPTH_SMOOTH_WINDOW)
    probs = np.zeros(len(cfg.ACTIONS))
    pred_idx = -1
    no_hand_frames = 0
    # Sentence builder state
    sentence = []
    last_committed = None

    # ── Smart sentence worker (Groq, riêng 1 luồng, throttle 7s) ────────────
    smart = SmartSentence()
    smart_stop = threading.Event()
    smart_thread = threading.Thread(
        target=smart_worker,
        args=(lambda: list(sentence), smart, smart_stop),
        kwargs={"is_active": lambda: True},
        daemon=True,
    )
    smart_thread.start()

    cv2.namedWindow("Realtime VSL Prediction", cv2.WINDOW_NORMAL)

    with create_model() as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # frame = cv2.flip(frame,1)
            # ── MediaPipe inference ──────────────────────────────────────────
            img, results = detect(frame, holistic)
            img = draw_landmarks_new(img, results)

            # ── Check hand presence (rest pose detection) ────────────────────
            has_hand = (results.left_hand_landmarks is not None) or (results.right_hand_landmarks is not None)

            if not has_hand:
                no_hand_frames += 1
                # If hands are absent for >= 5 consecutive frames, clear buffer to avoid gesture bleed
                if no_hand_frames >= 5:
                    if len(seq) > 0:
                        seq.clear()
                    votes.clear()
                    pred_idx = -1
                    probs = np.zeros(len(cfg.ACTIONS))
                    last_committed = None
            else:
                no_hand_frames = 0

            # ── Keypoint extraction & sliding window ─────────────────────────
            feed_calibrator(results, frame.shape[:2], calibrator)
            keypoints = extract(results, frame_shape=frame.shape[:2],
                                neutral_lens=[calibrator.get(0), calibrator.get(1)])
            keypoints[102:104] = [smoother.update(s, keypoints[102 + s])
                                  for s in (0, 1)]
            seq.append(keypoints)

            # ── LSTM prediction (once window is full) ─────────────────────────
            if len(seq) == cfg.SEQUENCE_LEN and has_hand:
                X = np.array([list(seq)], dtype="float32")
                # Apply same standardisation used during training with 0.0 zero-mask preservation
                if feat_mean is not None and feat_std is not None:
                    X, _, _ = standardize_features(X, mean=feat_mean, std=feat_std)

                yhat = model.predict(X, verbose=0)
                probs = yhat[0]
                best_idx = int(np.argmax(probs))
                votes.append((best_idx, float(probs[best_idx])))
                # Majority vote over recent windows to suppress flicker
                labels = [l for l, _ in votes]
                winner = max(set(labels), key=labels.count)
                sconf = float(np.mean([c for l, c in votes if l == winner]))
                if sconf >= cfg.CONFIDENCE_THRESHOLD:
                    pred_idx = winner
                    probs = np.array([sconf if i == winner else 0.0
                                      for i in range(len(cfg.ACTIONS))])
                else:
                    pred_idx = -1

                # ── Sentence builder: commit immediately on confident vote ──
                if pred_idx >= 0:
                    word = str(cfg.ACTIONS[winner])
                    if word != last_committed:
                        sentence.append(word)
                        last_committed = word

            # ── Overlay: status & predicted label ─────────────────────────────
            if not has_hand and len(seq) < cfg.SEQUENCE_LEN:
                status_txt = "Nghỉ / Chưa phát hiện tay"
                txt_color = (180, 180, 180)
            elif pred_idx >= 0:
                pred_action = cfg.ACTIONS[pred_idx]
                status_txt = f"Pred: {pred_action} ({probs[pred_idx]*100:.1f}%)"
                txt_color = (0, 255, 0)
            else:
                status_txt = "Đang nhận diện..."
                txt_color = (0, 200, 255)

            put_text_unicode(img, status_txt, (20, 25), txt_color, size=28)

            # ── Overlay: per-class confidence bars ────────────────────────────
            draw_confidence_bars(img, cfg.ACTIONS, probs, pred_idx)

            # ── Overlay: committed sentence ───────────────────────────────────
            draw_sentence(img, sentence)

            # ── Overlay: LLM-cleaned smart sentence ─────────────────────────
            if not sentence:
                smart.clear()
            smart_text, smart_pending, smart_error = smart.read()
            draw_sentence_smart(img, smart_text,
                                pending=smart_pending, error=smart_error)

            # ── Buffer fill indicator (top-right corner) ──────────────────────
            filled = len(seq)
            total = cfg.SEQUENCE_LEN
            h, w = img.shape[:2]
            bar_txt = f"Buffer: {filled}/{total}"
            put_text_unicode(img, bar_txt, (w - 200, 12), (200, 200, 200),
                             size=18)

            cv2.imshow("Realtime VSL Prediction", img)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key in (8, 127):  # Backspace (macOS sends 127)
                if sentence:
                    sentence.pop()
                    last_committed = sentence[-1] if sentence else None
            elif key in (ord('c'), ord('C')):
                sentence.clear()
                last_committed = None

    smart_stop.set()
    smart_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
