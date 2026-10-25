"""Predictor service: singleton holding model + sliding-window state.

Moved verbatim from frontend/caller.py (mirrors
backend/scripts/realtime_predict.py). No inference logic changed.
"""

import os
import sys
import threading
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_FRONTEND = os.path.normpath(os.path.join(_HERE, "..", ".."))
_BACKEND = os.path.normpath(os.path.join(_FRONTEND, "..", "backend"))
for _p in (_BACKEND,
           os.path.join(_BACKEND, "src"),
           os.path.join(_BACKEND, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _FRONTEND not in sys.path:
    sys.path.insert(0, _FRONTEND)

import cv2
import numpy as np

import config as cfg
import overlay as ov
from app.services.groq_facade import FrontendGroqCaller
from vsl_translator.landmarks import (
    create_model, detect, draw_landmarks_new, extract, feed_calibrator,
    ArmDepthCalibrator, ArmDepthSmoother,
)
from vsl_translator.dataset import standardize_features


class Predictor:
    """Singleton holding model + sliding-window state (mirrors main())."""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        from keras.models import load_model

        model_path = os.path.join(cfg.MODELS_DIR, cfg.MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. Run scripts/train.py first.")
        self.model = load_model(model_path)

        stats_path = os.path.join(cfg.MODELS_DIR, "feature_stats.npz")
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.feat_mean, self.feat_std = stats["mean"], stats["std"]
        else:
            self.feat_mean, self.feat_std = None, None

        self.holistic = create_model().__enter__()
        self.seq = deque(maxlen=cfg.SEQUENCE_LEN)
        self.votes = deque(maxlen=cfg.SMOOTHING_WINDOW)
        self.calibrator = ArmDepthCalibrator(
            warmup_frames=cfg.ARM_DEPTH_CAL_WARMUP)
        self.smoother = ArmDepthSmoother(
            alpha=cfg.ARM_DEPTH_EMA_ALPHA,
            window=cfg.ARM_DEPTH_SMOOTH_WINDOW)
        self.probs = np.zeros(len(cfg.ACTIONS))
        self.pred_idx = -1
        self.no_hand_frames = 0
        self.sentence = []
        self.last_committed = None
        self._challenger = -1
        self._challenger_runs = 0
        self.status = "Bấm Bắt đầu — camera của thiết bị sẽ gửi khung hình"
        self.inferring = False
        self.debug = False  # True: vẽ landmark lên frame (tốn CPU)
        self._lock = threading.Lock()

        # -- Smart sentence (Groq, riêng 1 luồng, xem groq_facade)
        self.groq = FrontendGroqCaller(
            get_sentence=self._sentence_copy,
            is_active=self.is_inferring,
        )

    @classmethod
    def get(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def peek(cls):
        with cls._instance_lock:
            return cls._instance

    def is_inferring(self):
        with self._lock:
            return self.inferring

    def _sentence_copy(self):
        with self._lock:
            return list(self.sentence)

    # -- inference toggle (used by /api/inference) ---------------------------
    def start_inference(self, debug=False):
        with self._lock:
            self.inferring = True
            self.debug = debug
            self.seq.clear()
            self.votes.clear()
            self.pred_idx = -1
            self.probs = np.zeros(len(cfg.ACTIONS))
            self.no_hand_frames = 0
            self._challenger = -1
            self._challenger_runs = 0
            self.status = "Đang nhận diện..."

    def stop_inference(self):
        with self._lock:
            self.inferring = False
            self.seq.clear()
            self.votes.clear()
            self.pred_idx = -1
            self.probs = np.zeros(len(cfg.ACTIONS))
            self._challenger = -1
            self._challenger_runs = 0
            self.status = "Đã dừng"

    # -- sentence controls (used by /api/sentences) -------------------------
    def backspace(self):
        with self._lock:
            if self.sentence:
                self.sentence.pop()
                self.last_committed = (
                    self.sentence[-1] if self.sentence else None)

    def clear_sentence(self):
        with self._lock:
            self.sentence.clear()
            self.last_committed = None
        self.groq.clear()

    def retry_smart(self, raw_sentence=None):
        """Retry Groq cleanup for the sentence the frontend sends.

        Delegates to FrontendGroqCaller (direct sync call, no worker wait).
        """
        with self._lock:
            fallback = list(self.sentence)
        res = self.groq.retry(raw_sentence, fallback_sentence=fallback)
        if not res.get("ok"):
            return res
        return {"ok": True, **self.snapshot()}

    def set_groq_key(self, key):
        """Save Groq API key to backend .env.

        Worker đọc cfg.GROQ_API_KEY động mỗi vòng poll nên key mới có
        hiệu lực ngay, không cần restart server/predictor.
        """
        cfg.set_groq_api_key(key)
        # Xoá lỗi "missing key" cũ để worker thử lại ngay vòng sau.
        self.groq.notify_key_changed()
        return {"ok": True, **self.snapshot(),
                "groq_configured": bool(cfg.GROQ_API_KEY)}

    def snapshot(self):
        with self._lock:
            pred = (str(cfg.ACTIONS[self.pred_idx])
                    if self.pred_idx >= 0 else None)
            conf = (float(self.probs[self.pred_idx])
                    if self.pred_idx >= 0 else 0.0)
            snapshot = {
                "status": self.status,
                "pred": pred,
                "conf": round(conf, 3),
                "sentence": list(self.sentence),
                "buffer": f"{len(self.seq)}/{cfg.SEQUENCE_LEN}",
                "inferring": self.inferring,
                "debug": self.debug,
                "groq_model": cfg.GROQ_MODEL,
                "groq_configured": bool(cfg.GROQ_API_KEY),
            }
        smart_text, smart_pending, smart_error = self.groq.read()
        snapshot["smart"] = smart_text
        snapshot["smart_pending"] = smart_pending
        snapshot["smart_error"] = smart_error
        return snapshot

    # -- per-frame inference (same logic as realtime_predict.main) ---------
    def process(self, frame):
        img, results = detect(frame, self.holistic)
        with self._lock:
            debug = self.debug
        if debug:
            img = draw_landmarks_new(img, results)

        has_hand = (results.left_hand_landmarks is not None
                    or results.right_hand_landmarks is not None)

        with self._lock:
            inferring = self.inferring
            if not has_hand:
                self.no_hand_frames += 1
                if self.no_hand_frames >= 5:
                    if len(self.seq) > 0:
                        self.seq.clear()
                    self.votes.clear()
                    self.pred_idx = -1
                    self.probs = np.zeros(len(cfg.ACTIONS))
                    self.last_committed = None
                    self._challenger = -1
                    self._challenger_runs = 0
            else:
                self.no_hand_frames = 0

            if inferring:
                feed_calibrator(results, frame.shape[:2], self.calibrator)
                keypoints = extract(
                    results, frame_shape=frame.shape[:2],
                    neutral_lens=[self.calibrator.get(0),
                                  self.calibrator.get(1)])
                keypoints[102:104] = [self.smoother.update(s, keypoints[102 + s])
                                      for s in (0, 1)]
                self.seq.append(keypoints)

            if len(self.seq) == cfg.SEQUENCE_LEN and has_hand:
                X = np.array([list(self.seq)], dtype="float32")
                if self.feat_mean is not None and self.feat_std is not None:
                    X, _, _ = standardize_features(
                        X, mean=self.feat_mean, std=self.feat_std)
                yhat = self.model.predict(X, verbose=0)
                self.probs = yhat[0]
                best_idx = int(np.argmax(self.probs))
                self.votes.append((best_idx, float(self.probs[best_idx])))
                labels = [lb for lb, _ in self.votes]
                winner = max(set(labels), key=labels.count)
                sconf = float(np.mean(
                    [c for lb, c in self.votes if lb == winner]))
                if sconf >= cfg.threshold_for(winner):
                    if self.pred_idx < 0 or winner == self.pred_idx:
                        self.pred_idx = winner
                        self._challenger = -1
                        self._challenger_runs = 0
                    else:
                        if winner == self._challenger:
                            self._challenger_runs += 1
                        else:
                            self._challenger = winner
                            self._challenger_runs = 1
                        if self._challenger_runs >= cfg.SWITCH_STABLE_FRAMES:
                            self.pred_idx = winner
                            self._challenger = -1
                            self._challenger_runs = 0
                    if self.pred_idx == winner:
                        self.probs = np.array(
                            [sconf if i == winner else 0.0
                             for i in range(len(cfg.ACTIONS))])
                else:
                    self.pred_idx = -1
                    self._challenger = -1
                    self._challenger_runs = 0

                if self.pred_idx >= 0:
                    word = str(cfg.ACTIONS[winner])
                    if word != self.last_committed:
                        self.sentence.append(word)
                        self.last_committed = word

            if not inferring:
                self.status = "Đã dừng (bấm Bắt đầu để nhận diện)"
                color = (180, 180, 180)
            elif not has_hand and len(self.seq) < cfg.SEQUENCE_LEN:
                self.status = "Nghỉ / Chưa phát hiện tay"
                color = (180, 180, 180)
            elif self.pred_idx >= 0:
                self.status = (f"Pred: {cfg.ACTIONS[self.pred_idx]} "
                               f"({self.probs[self.pred_idx]*100:.1f}%)")
                color = (0, 255, 0)
            else:
                self.status = "Đang nhận diện..."
                color = (0, 200, 255)

            sentence = list(self.sentence)
            probs = np.array(self.probs)
            pred_idx = self.pred_idx

        if not sentence:
            self.groq.clear_if_empty(sentence)

        ov.put_text_unicode(img, self.status, (20, 25), color, size=28)
        ov.draw_confidence_bars(img, cfg.ACTIONS, probs, pred_idx)
        smart_text, smart_pending, smart_error = self.groq.read()
        ov.draw_sentence_smart(img, smart_text,
                               pending=smart_pending, error=smart_error)
        ov.draw_sentence(img, sentence)
        filled, total = len(self.seq), cfg.SEQUENCE_LEN
        h, w = img.shape[:2]
        ov.put_text_unicode(img, f"Buffer: {filled}/{total}",
                            (w - 200, 12), (200, 200, 200), size=18)
        return img
