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
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

import config as cfg
from vsl_translator.landmarks import (
    create_model, detect, draw_landmarks_new, extract, feed_calibrator,
    ArmDepthCalibrator, ArmDepthSmoother,
)
from vsl_translator.dataset import standardize_features

# Max overlay lines for the sentence box at the bottom of the video.
SENTENCE_MAX_LINES = 3

# ── Unicode text rendering (Pillow + TrueType) ──────────────────────────────
# cv2.putText only supports ASCII Hershey fonts, so Vietnamese diacritics
# break. Render text via Pillow instead; OpenCV still draws boxes/video.
FONT_PATH = "/Library/Fonts/Arial Unicode.ttf"
FONT_SIZE = 22
_FONT_CACHE = {}


def _get_font(size=FONT_SIZE):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(FONT_PATH, size)
    return _FONT_CACHE[size]


def put_text_unicode(img, text, org, color=(255, 255, 255), size=FONT_SIZE):
    """Draw UTF-8 text (BGR color) onto a BGR OpenCV frame. Returns img."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    b, g, r = color
    draw.text(org, text, font=_get_font(size), fill=(r, g, b))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def text_width(text, size=FONT_SIZE):
    """Pixel width of text with the overlay font (for greedy wrapping)."""
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = draw.textbbox((0, 0), text, font=_get_font(size))
    return bbox[2] - bbox[0]


def text_height(size=FONT_SIZE):
    """Pixel line height (ascender+descender) of the overlay font."""
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = draw.textbbox((0, 0), "Ag", font=_get_font(size))
    return bbox[3] - bbox[1]

# ── Smart sentence (Groq LLM cleanup) ─────────────────────────────────────────
# Settings live in config.py (env-overridable: GROQ_MODEL, GROQ_TIMEOUT,
# GROQ_RETRIES, SMART_CALL_INTERVAL, SMART_RETRY_INTERVAL, SMART_POLL_INTERVAL).
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..",
                           "TXT_NOISE_RMV_PROMPT.md")


# .env is loaded by config.py on import (PROJECT_ROOT/.env).


def _load_prompt_template():
    with open(PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


class SmartSentence:
    """Thread-safe holder for the LLM-cleaned sentence."""

    def __init__(self):
        self._lock = threading.Lock()
        self.text = ""
        self.pending = False  # a request is in flight
        self.error = None     # last error message, if any
        self._last_sent = ""
        self._last_err_time = 0.0

    def set_result(self, raw_sent, cleaned):
        with self._lock:
            self.text = cleaned
            self._last_sent = raw_sent
            self.pending = False
            self.error = None

    def set_error(self, msg):
        with self._lock:
            self.pending = False
            self.error = msg

    def mark_pending(self, raw_sent):
        with self._lock:
            if self.pending:
                return False
            self.pending = True
            self._last_sent = raw_sent
            return True

    def retry_now(self):
        """Force an immediate retry of the last sentence (manual retry button).

        Clears cooldown + last-sent so the worker picks it up on next poll.
        Returns False if a request is already in flight or nothing to retry.
        """
        with self._lock:
            if self.pending or not self._last_sent:
                return False
            self._last_sent = ""
            self._last_err_time = 0.0
            self.error = None
            return True

    def clear(self):
        with self._lock:
            self.text = ""
            self._last_sent = ""
            self.pending = False
            self.error = None

    def read(self):
        with self._lock:
            return self.text, self.pending, self.error


def _call_groq(raw_sentence, prompt_template, api_key):
    """Send raw words to Groq for cleanup, return cleaned text.

    Uses the OpenAI module against the Groq endpoint (base_url from config,
    model GROQ_MODEL, key GROQ_API_KEY from .env).
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=cfg.GROQ_BASE_URL,
        timeout=cfg.GROQ_TIMEOUT,
        max_retries=cfg.GROQ_RETRIES + 1,
    )
    prompt = prompt_template.replace("{{Predict}}", raw_sentence)
    resp = client.chat.completions.create(
        model=cfg.GROQ_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": raw_sentence},
        ],
        temperature=0,
        top_p=1.0,
        max_tokens=256,
    )
    text = (resp.choices[0].message.content or "").strip()
    text = text.strip().strip('"').strip("'")
    if text.lower().startswith("output:"):
        text = text[7:].strip()
    if not text:
        raise RuntimeError(f"Empty Groq response: {resp}")
    return text


def smart_worker(get_sentence, smart, stop_event, is_active=None):
    """Poll every 0.1s; call Groq only when the sentence changed + 7s throttle."""
    try:
        prompt_template = _load_prompt_template()
    except OSError as e:
        smart.set_error(f"Prompt file missing: {e}")
        return
    if is_active is None:
        is_active = lambda: True
    last_call = 0.0
    while not stop_event.is_set():
        raw = " ".join(get_sentence())
        if raw:
            api_key = cfg.GROQ_API_KEY
            if not api_key:
                smart.set_error("Missing GROQ_API_KEY — nhập key ở phần Cài đặt")
            else:
                with smart._lock:
                    already_sent = raw == getattr(smart, "_last_sent", "")
                    busy = smart.pending
                    last_err = getattr(smart, "_last_err_time", 0.0)
                cooling = (time.monotonic() - last_err) < cfg.SMART_RETRY_INTERVAL
                throttled = (time.monotonic() - last_call) < cfg.SMART_CALL_INTERVAL
                if not already_sent and not busy and not cooling and not throttled:
                    if not is_active():
                        pass  # not inferring: skip call, retry on next poll
                    elif smart.mark_pending(raw):
                        try:
                            cleaned = None
                            last_exc = None
                            for _ in range(cfg.GROQ_RETRIES + 1):
                                try:
                                    cleaned = _call_groq(
                                        raw, prompt_template, api_key)
                                    break
                                except Exception as e:
                                    last_exc = e
                            if cleaned is None:
                                raise last_exc
                            smart.set_result(raw, cleaned)
                            last_call = time.monotonic()
                        except Exception as e:  # keep old text, retry after interval
                            smart.set_error(
                                f"[{cfg.GROQ_MODEL}] {type(e).__name__}: {e}")
                            with smart._lock:
                                # allow retry of the SAME sentence after cooldown
                                # (not only on next change)
                                smart._last_sent = ""
                                smart._last_err_time = time.monotonic()
        time.sleep(cfg.SMART_POLL_INTERVAL)


def draw_sentence(img, sentence):
    """Draw the committed sentence in a box at the bottom of the frame.

    Wraps with text_width() to fit the frame width, up to
    SENTENCE_MAX_LINES lines. The newest word is drawn green, the rest white.
    """
    h, w = img.shape[:2]
    margin, pad = 20, 10

    words = list(sentence) if sentence else ["(trống)"]
    # Greedy wrap into lines that fit the frame width.
    lines = [""]
    max_w = w - 2 * (margin + pad)
    for wd in words:
        cand = (lines[-1] + " " + wd).strip()
        tw = text_width(cand)
        if tw <= max_w or not lines[-1]:
            lines[-1] = cand
        elif len(lines) < SENTENCE_MAX_LINES:
            lines.append(wd)
        else:
            break  # box full: older words stay in `sentence`, only display clips
    lines = lines[:SENTENCE_MAX_LINES]

    lh = text_height()
    line_h = lh + 8
    box_h = line_h * len(lines) + 2 * pad
    box_y0 = h - box_h - margin
    overlay = img.copy()
    cv2.rectangle(overlay, (margin, box_y0),
                  (w - margin, h - margin), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    y = box_y0 + pad
    newest = words[-1] if sentence else None
    for line in lines:
        if newest and line.endswith(newest) and line != newest:
            # Split line so the newest word renders green.
            head = line[:len(line) - len(newest)].rstrip()
            put_text_unicode(img, head, (margin + pad, y), (255, 255, 255))
            put_text_unicode(img, newest,
                             (margin + pad + text_width(head + " "), y),
                             (0, 255, 0))
        else:
            color = (0, 255, 0) if line == newest else (255, 255, 255)
            put_text_unicode(img, line, (margin + pad, y), color)
        y += line_h
    return img


def draw_sentence_smart(img, smart_text, pending=False, error=None, y_bottom=None):
    """Draw the LLM-cleaned sentence in a box above the raw sentence box.

    Same greedy-wrap layout as draw_sentence (up to SENTENCE_MAX_LINES lines).
    `y_bottom`: bottom edge of this box; defaults to sitting just above the
    raw sentence box. Shows a status line when pending/error/empty.
    """
    h, w = img.shape[:2]
    margin, pad = 20, 10

    if error:
        lines = [f"(smart lỗi: {error})"]
    elif not smart_text:
        lines = ["(smart: ...)" if pending else "(smart: trống)"]
    else:
        lines = [""]
        max_w = w - 2 * (margin + pad)
        for wd in smart_text.split():
            cand = (lines[-1] + " " + wd).strip()
            tw = text_width(cand)
            if tw <= max_w or not lines[-1]:
                lines[-1] = cand
            elif len(lines) < SENTENCE_MAX_LINES:
                lines.append(wd)
            else:
                break
        lines = lines[:SENTENCE_MAX_LINES]

    title = "Smart:"
    lh = text_height()
    line_h = lh + 8
    box_h = line_h * (len(lines) + 1) + 2 * pad
    if y_bottom is None:
        # Reserve space for the raw box: estimate its height the same way
        # draw_sentence does (assume 1 line) and stack above it.
        raw_box_h = line_h + 2 * pad
        y_bottom = h - raw_box_h - margin - 8
    box_y0 = y_bottom - box_h
    overlay = img.copy()
    cv2.rectangle(overlay, (margin, box_y0),
                  (w - margin, y_bottom), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    y = box_y0 + pad
    put_text_unicode(img, title, (margin + pad, y), (0, 200, 255))
    y += line_h
    for line in lines:
        put_text_unicode(img, line, (margin + pad, y), (255, 255, 255))
        y += line_h
    return img


def draw_confidence_bars(img, actions, probs, pred_idx, bar_x=20, bar_y=80,
                         bar_w=200, bar_h=22, gap=10):
    """Draw a labelled confidence bar for each class."""
    for i, (action, prob) in enumerate(zip(actions, probs)):
        y = bar_y + i * (bar_h + gap)

        # Background track
        cv2.rectangle(img, (bar_x, y), (bar_x + bar_w, y + bar_h),
                      (50, 50, 50), -1)

        # Filled bar — green for predicted class, blue for others
        fill = int(prob * bar_w)
        color = (0, 220, 0) if (pred_idx >= 0 and i == pred_idx) else (200, 120, 0)
        if fill > 0:
            cv2.rectangle(img, (bar_x, y), (bar_x + fill, y + bar_h), color, -1)

        # Label + percentage
        label = f"{action}: {prob * 100:.1f}%"
        put_text_unicode(img, label, (bar_x + bar_w + 8, y + 2),
                         (255, 255, 255), size=18)


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
