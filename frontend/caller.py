"""Bridge between frontend (server.py / index.html) and the backend predictor.

Reuses the inference pipeline of backend/scripts/realtime_predict.py, but
instead of cv2.imshow it yields annotated JPEG frames (MJPEG) and exposes
the latest prediction state for /api/state polling.
"""

import os
import sys
import threading
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.normpath(os.path.join(_HERE, "..", "backend"))
for _p in (_BACKEND,
           os.path.join(_BACKEND, "src"),
           os.path.join(_BACKEND, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np

import config as cfg
import realtime_predict as rp
from vsl_translator.landmarks import (
    create_model, detect, draw_landmarks_new, extract, feed_calibrator,
    ArmDepthCalibrator, ArmDepthSmoother,
)
from vsl_translator.dataset import standardize_features


def test(info):
    print(info)


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
        self.status = "Chưa mở camera (bấm Bắt đầu để nhận diện)"
        self.inferring = False
        self.debug = False  # True: vẽ landmark lên frame (tốn CPU)
        self._lock = threading.Lock()

        # -- Smart sentence worker (Groq, riêng 1 luồng, như realtime_predict)
        self.smart = rp.SmartSentence()
        self._smart_stop = threading.Event()
        self._smart_thread = threading.Thread(
            target=rp.smart_worker,
            args=(self._sentence_copy, self.smart, self._smart_stop),
            kwargs={"is_active": self.is_inferring},
            daemon=True,
        )
        self._smart_thread.start()

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
            self.status = "Đang nhận diện..."

    def stop_inference(self):
        with self._lock:
            self.inferring = False
            self.seq.clear()
            self.votes.clear()
            self.pred_idx = -1
            self.probs = np.zeros(len(cfg.ACTIONS))
            self.status = "Đã dừng"

    # -- sentence controls (used by /api/sentence) -------------------------
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

    def retry_smart(self):
        """Manual retry of the Groq noise-remover (used by /api/smart)."""
        if not self.smart.retry_now():
            return {"ok": False,
                    "error": "nothing to retry (request in flight or empty)"}
        return {"ok": True, **self.snapshot()}

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
            }
        smart_text, smart_pending, smart_error = self.smart.read()
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
                if sconf >= cfg.CONFIDENCE_THRESHOLD:
                    self.pred_idx = winner
                    self.probs = np.array(
                        [sconf if i == winner else 0.0
                         for i in range(len(cfg.ACTIONS))])
                else:
                    self.pred_idx = -1

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
            self.smart.clear()

        rp.put_text_unicode(img, self.status, (20, 25), color, size=28)
        rp.draw_confidence_bars(img, cfg.ACTIONS, probs, pred_idx)
        smart_text, smart_pending, smart_error = self.smart.read()
        rp.draw_sentence_smart(img, smart_text,
                               pending=smart_pending, error=smart_error)
        rp.draw_sentence(img, sentence)
        filled, total = len(self.seq), cfg.SEQUENCE_LEN
        h, w = img.shape[:2]
        rp.put_text_unicode(img, f"Buffer: {filled}/{total}",
                            (w - 200, 12), (200, 200, 200), size=18)
        return img


# Only one client may hold the camera at a time.
_stream_lock = threading.Lock()


def videoStream(camera_index=0):
    """Yield annotated JPEG frames (MJPEG), like realtime_predict.py.

    Camera mở khi và chỉ khi inference đang bật. Client ngắt stream
    (bấm Dừng) thì vòng lặp dừng và camera được release ngay.
    """
    if not _stream_lock.acquire(blocking=False):
        raise RuntimeError("camera busy: another client is streaming")
    try:
        predictor = Predictor.peek()
        if predictor is None or not predictor.is_inferring():
            raise RuntimeError("inference is off: press Start first")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera "
                               "(check Privacy & Security → Camera)")
        try:
            while cap.isOpened():
                if not predictor.is_inferring():
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                img = predictor.process(frame)
                ok, buf = cv2.imencode(".jpg", img)
                if not ok:
                    continue
                yield buf.tobytes()
        finally:
            cap.release()
    finally:
        _stream_lock.release()


def get_state():
    predictor = Predictor.peek()
    if predictor is None:
        return {"status": "Chưa mở camera (bấm Bắt đầu để nhận diện)",
                "pred": None, "conf": 0.0, "sentence": [], "buffer": "0/0",
                "inferring": False, "debug": False,
                "smart": "", "smart_pending": False, "smart_error": None}
    try:
        return predictor.snapshot()
    except Exception as e:  # model not loaded yet / load failed
        return {"status": f"unavailable: {e}", "pred": None,
                "conf": 0.0, "sentence": [], "buffer": "0/0",
                "inferring": False, "debug": False,
                "smart": "", "smart_pending": False, "smart_error": None}
