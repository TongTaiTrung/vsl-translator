"""Unit tests for the 188-dimensional feature vector layout and arm-depth propagation."""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vsl_translator import landmarks as lm
from vsl_translator import dataset as ds
from vsl_translator.model import build_model


class TestArmDepths(unittest.TestCase):
    def test_dimensions_constant(self):
        self.assertEqual(lm.ARM_DEPTH_DIM, 2)
        self.assertEqual(lm.BODY_ONLY_DIM, 188)
        self.assertEqual(lm.n_features(), 188)

    def test_model_input_dimension(self):
        model = build_model(n_actions=4, sequence_len=24)
        self.assertEqual(model.input_shape, (None, 24, 188))

    def test_coord_mask_excludes_depth_ratios(self):
        mask = ds._coord_mask(188)
        # Left and right arm depth ratios are at indices 102 and 103
        self.assertFalse(mask[102])
        self.assertFalse(mask[103])

    def test_augmentations_preserve_depth(self):
        np.random.seed(42)
        seq = np.random.rand(24, 188).astype(np.float32)
        seq[:, 102:104] = [0.85, 1.15]

        # 1. Noise
        noisy = ds._add_noise(seq.copy())
        np.testing.assert_array_equal(noisy[:, 102:104], seq[:, 102:104])

        # 2. Scale jitter
        scaled = ds._scale_jitter(seq.copy())
        np.testing.assert_array_equal(scaled[:, 102:104], seq[:, 102:104])

        # 3. Random shift
        shifted = ds._random_shift(seq.copy())
        np.testing.assert_array_equal(shifted[:, 102:104], seq[:, 102:104])

        # 4. Hand dropout
        dropped = ds._random_hand_dropout(seq.copy(), p=1.0)
        np.testing.assert_array_equal(dropped[:, 102:104], seq[:, 102:104])

    def test_hand_swap_swaps_arm_depths(self):
        np.random.seed(0)
        seq = np.random.rand(24, 188).astype(np.float32)
        seq[:, 102:104] = [0.75, 1.25]

        swapped = ds._random_hand_swap(seq.copy(), p=1.0)
        np.testing.assert_array_almost_equal(swapped[:, 102], seq[:, 103])
        np.testing.assert_array_almost_equal(swapped[:, 103], seq[:, 102])

    def test_pixel_aspect_invariance(self):
        # Same pose + same per-session neutral length (in matching pixel
        # units) on 4:3 vs 16:9 frames must give nearly equal ratios.
        import math as _math
        shoulder, elbow, wrist = (0.4, 0.2), (0.42, 0.45), (0.44, 0.7)
        ratios = []
        for shape in ((480, 640), (720, 1280)):
            h, w = shape
            arm_px = (_math.hypot((elbow[0]-shoulder[0])*w, (elbow[1]-shoulder[1])*h)
                      + _math.hypot((wrist[0]-elbow[0])*w, (wrist[1]-elbow[1])*h))
            ratios.append(lm._compute_arm_depth_ratio(
                shoulder, elbow, wrist, frame_shape=shape, neutral_len=arm_px))
        for r in ratios:
            self.assertAlmostEqual(r, 1.0, places=5)

    def test_session_calibration(self):
        # Arms at different absolute scales (near/far camera) calibrate to ~1.0.
        import math as _math
        for shape, scale in (((480, 640), 0.5), ((720, 1280), 1.5)):
            h, w = shape
            sh = (0.4, 0.2)
            el = (sh[0], sh[1] + 0.3 * scale)
            wr = (sh[0], sh[1] + 0.7 * scale)
            arm_px = (_math.hypot((el[0]-sh[0])*w, (el[1]-sh[1])*h)
                      + _math.hypot((wr[0]-el[0])*w, (wr[1]-el[1])*h))
            cal = lm.ArmDepthCalibrator(warmup_frames=10)
            for _ in range(10):
                cal.update(0, arm_px, shoulder_width_px=0.2 * w * scale)
            neutral = cal.get(0)
            r = lm._compute_arm_depth_ratio(
                sh, el, wr, frame_shape=shape, neutral_len=neutral)
            self.assertAlmostEqual(r, 1.0, places=5)

    def test_z_correction_direction(self):
        # Same 2D foreshortening, but wrist pushed toward camera (dz<0)
        # must give a smaller ratio than dz≈0.
        shoulder, elbow, wrist = (0.4, 0.2), (0.38, 0.6), (0.42, 1.1)
        plain = lm._compute_arm_depth_ratio(shoulder, elbow, wrist, dz=0.0)
        pushed = lm._compute_arm_depth_ratio(shoulder, elbow, wrist, dz=-0.1)
        self.assertLess(pushed, plain)

    def test_smoother_no_jump(self):
        smoother = lm.ArmDepthSmoother(alpha=0.4, window=5)
        outs = [smoother.update(0, 1.0) for _ in range(5)]
        spiked = smoother.update(0, 0.4)  # single-frame spike
        self.assertLess(abs(spiked - outs[-1]), 0.15,
                        f"smoother jumped on spike: {spiked}")


if __name__ == '__main__':
    unittest.main()
