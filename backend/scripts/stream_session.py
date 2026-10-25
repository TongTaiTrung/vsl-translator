"""Shared per-stream prediction state for local and remote (WebSocket) inference.

One StreamSession holds the sliding window, vote buffer, calibrators and
sentence builder for a single video source. Both realtime_predict.py (local
webcam) and server.py (remote frames) drive it through process_frame().
"""

from collections import deque

import numpy as np

import config as cfg
from scripts._common import extract_frame_vector, standardize
from vsl_translator.landmarks import ArmDepthCalibrator, ArmDepthSmoother


class StreamSession:
    """Mutable inference state for one video stream."""

    def __init__(self):
        self.seq = deque(maxlen=cfg.SEQUENCE_LEN)
        self.votes = deque(maxlen=cfg.SMOOTHING_WINDOW)
        self.calibrator = ArmDepthCalibrator(warmup_frames=cfg.ARM_DEPTH_CAL_WARMUP)
        self.smoother = ArmDepthSmoother(alpha=cfg.ARM_DEPTH_EMA_ALPHA,
                                         window=cfg.ARM_DEPTH_SMOOTH_WINDOW)
        self.probs = np.zeros(len(cfg.ACTIONS))
        self.pred_idx = -1
        self.no_hand_frames = 0
        self.sentence = []
        self.last_committed = None

    def process_frame(self, results, frame_shape, model,
                      feat_mean=None, feat_std=None):
        """Run extract + predict for one frame's MediaPipe results.

        Returns dict with has_hand, pred_idx, probs, sentence (list copy).
        """
        has_hand = ((results.left_hand_landmarks is not None)
                    or (results.right_hand_landmarks is not None))

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

        keypoints = extract_frame_vector(results, frame_shape,
                                           self.calibrator, self.smoother)
        self.seq.append(keypoints)

        if len(self.seq) == cfg.SEQUENCE_LEN and has_hand:
            X = np.array([list(self.seq)], dtype="float32")
            X = standardize(X, feat_mean, feat_std)
            yhat = model.predict(X, verbose=0)
            self.probs = yhat[0]
            best_idx = int(np.argmax(self.probs))
            self.votes.append((best_idx, float(self.probs[best_idx])))
            labels = [l for l, _ in self.votes]
            winner = max(set(labels), key=labels.count)
            sconf = float(np.mean([c for l, c in self.votes if l == winner]))
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

        return {"has_hand": has_hand, "pred_idx": self.pred_idx,
                "probs": self.probs, "sentence": list(self.sentence)}
