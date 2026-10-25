"""Regression tests for live voting hysteresis + per-class thresholds.

Covers scripts/stream_session.py StreamSession:
- flip-flop A/B winners do NOT switch pred until SWITCH_STABLE_FRAMES
  consecutive challenger wins;
- same winner keeps prediction;
- per-class threshold (cfg.threshold_for) gates low-confidence winners.
"""

import sys
import os
import unittest
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config as cfg
from scripts import stream_session as S
from scripts.stream_session import StreamSession


class _Results:
    left_hand_landmarks = object()
    right_hand_landmarks = None


class _FakeModel:
    def __init__(self, winner, conf=0.9):
        self.winner = winner
        self.conf = conf

    def predict(self, X, verbose=0):
        p = np.full(len(cfg.ACTIONS), (1.0 - self.conf) / (len(cfg.ACTIONS) - 1))
        p[self.winner] = self.conf
        return np.array([p])


class TestHysteresis(unittest.TestCase):
    def setUp(self):
        self._orig_extract = S.extract_frame_vector
        S.extract_frame_vector = lambda *a: np.zeros(356, dtype=np.float32)

    def tearDown(self):
        S.extract_frame_vector = self._orig_extract

    def _primed(self, winner=0, conf=0.9, rounds=6):
        s = StreamSession()
        s.seq = deque([np.zeros(356)] * (cfg.SEQUENCE_LEN - 1),
                      maxlen=cfg.SEQUENCE_LEN)
        z = np.zeros(356)
        o = np.ones(356)
        for _ in range(rounds):
            s.process_frame(_Results(), (640, 640),
                            _FakeModel(winner, conf), z, o)
        return s

    def test_stable_winner_commits(self):
        s = self._primed(winner=0)
        self.assertEqual(s.pred_idx, 0)

    def test_single_challenger_frame_does_not_switch(self):
        s = self._primed(winner=0)
        z = np.zeros(356)
        o = np.ones(356)
        s.process_frame(_Results(), (640, 640), _FakeModel(1), z, o)
        self.assertEqual(
            s.pred_idx, 0,
            "one challenger frame must not flip the prediction")

    def test_challenger_switches_after_stable_frames(self):
        s = self._primed(winner=0)
        z = np.zeros(356)
        o = np.ones(356)
        for _ in range(cfg.SWITCH_STABLE_FRAMES + cfg.SMOOTHING_WINDOW):
            s.process_frame(_Results(), (640, 640), _FakeModel(1), z, o)
        self.assertEqual(s.pred_idx, 1)

    def test_per_class_threshold_blocks_low_confidence(self):
        prev = dict(cfg.CLASS_THRESHOLDS)
        try:
            name = str(cfg.ACTIONS[0])
            cfg.CLASS_THRESHOLDS[name] = 0.95
            s = self._primed(winner=0, conf=0.9)
            self.assertEqual(
                s.pred_idx, -1,
                "winner below its per-class threshold must stay uncertain")
        finally:
            cfg.CLASS_THRESHOLDS.clear()
            cfg.CLASS_THRESHOLDS.update(prev)


if __name__ == "__main__":
    unittest.main()
