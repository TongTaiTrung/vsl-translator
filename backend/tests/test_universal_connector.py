"""Unit tests for universal_connector."""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as cfg
from universal_connector import UniversalConnector, PredictionResult, predict, get_connector


class TestUniversalConnector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build a throwaway model with the current extractor dim — the
        # on-disk models/* artifacts are stale (pre-face-offset) and must
        # not be loaded here. Blank-frame tests only assert structure.
        import tempfile
        from keras.models import Sequential
        from keras.layers import Input, Flatten, Dense
        from vsl_translator.landmarks import n_features as _nf
        cls._tmp = tempfile.TemporaryDirectory()
        n_actions = len(cfg.ACTIONS)
        m = Sequential([
            Input(shape=(cfg.SEQUENCE_LEN, _nf())),
            Flatten(),
            Dense(n_actions, activation="softmax"),
        ])
        m.compile(optimizer="adam", loss="categorical_crossentropy")
        cls._model_path = os.path.join(cls._tmp.name, "model.keras")
        m.save(cls._model_path)
        cls.connector = UniversalConnector(
            model_path=cls._model_path,
            stats_path=os.path.join(cls._tmp.name, "none.npz"),
        )

    @classmethod
    def tearDownClass(cls):
        cls.connector.close()
        cls._tmp.cleanup()

    def test_initialization(self):
        self.assertIsNotNone(self.connector.model)
        self.assertEqual(self.connector.sequence_len, cfg.SEQUENCE_LEN)
        self.assertGreater(len(self.connector.actions), 0)
        self.assertEqual(self.connector.confidence_threshold, cfg.CONFIDENCE_THRESHOLD)

    def test_predict_blank_frames(self):
        frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(cfg.SEQUENCE_LEN)]
        result = self.connector.predict(frames)

        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.num_frames_received, cfg.SEQUENCE_LEN)
        self.assertEqual(result.num_frames_processed, cfg.SEQUENCE_LEN)
        self.assertFalse(result.has_hand)
        self.assertFalse(result.is_valid)
        self.assertIsNone(result.action)
        self.assertIn(result.top_action, self.connector.actions)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertEqual(len(result.probabilities), len(self.connector.actions))

    def test_dict_interface_and_serialization(self):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(cfg.SEQUENCE_LEN)]
        result = self.connector.predict(frames)

        # Test dictionary-like access
        self.assertEqual(result["confidence"], result.confidence)
        self.assertEqual(result.get("top_action"), result.top_action)
        self.assertIn("action", result)
        self.assertIn("latency_ms", result)

        # Test to_dict serialization
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("probabilities", d)
        self.assertIsInstance(d["raw_probabilities"], list)
        self.assertEqual(len(d["probabilities"]), len(self.connector.actions))

    def test_resampling_variable_length(self):
        # Third-party passes 15 frames instead of 24
        short_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(15)]
        res = self.connector.predict(short_frames)
        self.assertEqual(res.num_frames_received, 15)
        self.assertEqual(res.num_frames_processed, cfg.SEQUENCE_LEN)

        # Third-party passes 35 frames instead of 24
        long_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(35)]
        res_long = self.connector.predict(long_frames)
        self.assertEqual(res_long.num_frames_received, 35)
        self.assertEqual(res_long.num_frames_processed, cfg.SEQUENCE_LEN)

    def test_empty_frames_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.connector.predict([])

    def test_invalid_frame_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.connector.predict(["not_a_numpy_array"] * 24)

    def test_module_level_predict(self):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(cfg.SEQUENCE_LEN)]
        result = predict(frames, connector=self.connector)
        self.assertIsInstance(result, PredictionResult)

    def test_context_manager(self):
        with UniversalConnector() as conn:
            self.assertIsNotNone(conn.holistic)
        self.assertIsNone(conn.holistic)


if __name__ == "__main__":
    unittest.main()
