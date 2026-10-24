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
    create_model, detect, draw_landmarks_new, normalize_frame,
)

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

    # Persistent prediction state (shared with remote WebSocket server)
    from scripts.stream_session import StreamSession
    session = StreamSession()

    # ── Smart sentence worker (Groq, riêng 1 luồng, throttle 7s) ────────────
    smart = SmartSentence()
    smart_stop = threading.Event()
    smart_thread = threading.Thread(
        target=smart_worker,
        args=(lambda: list(session.sentence), smart, smart_stop),
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
            frame, _meta = normalize_frame(frame)
            # ── MediaPipe inference + shared session (extract/predict) ─────
            img, results = detect(frame, holistic)
            img = draw_landmarks_new(img, results)
            out = session.process_frame(results, frame.shape[:2], model,
                                        feat_mean, feat_std)
            has_hand = out["has_hand"]
            pred_idx = out["pred_idx"]
            probs = out["probs"]

            # ── Overlay: status & predicted label ─────────────────────────────
            if not has_hand and len(session.seq) < cfg.SEQUENCE_LEN:
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
            draw_sentence(img, session.sentence)

            # ── Overlay: LLM-cleaned smart sentence ─────────────────────────
            if not session.sentence:
                smart.clear()
            smart_text, smart_pending, smart_error = smart.read()
            draw_sentence_smart(img, smart_text,
                                pending=smart_pending, error=smart_error)

            # ── Buffer fill indicator (top-right corner) ──────────────────────
            filled = len(session.seq)
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
                if session.sentence:
                    session.sentence.pop()
                    session.last_committed = session.sentence[-1] if session.sentence else None
            elif key in (ord('c'), ord('C')):
                session.sentence.clear()
                session.last_committed = None

    smart_stop.set()
    smart_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
