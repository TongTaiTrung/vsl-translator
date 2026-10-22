"""Regression tests for the far-distance two-hand class bias fix.

Covers: local-hand independence, palm-size guard, writeability restore,
idle-arm gate isolation, one-hand detection, injection probability/shape,
label preservation, train-only augmentation, 356-dim layout.

Run: venv/bin/python -m unittest tests.test_far_bias -v
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vsl_translator import landmarks as lm
from vsl_translator import dataset as ds


class FakePt:
    def __init__(self, x, y, visibility=1.0, z=0.0):
        self.x, self.y, self.visibility, self.z = x, y, visibility, z


class FakeFace:
    def __init__(self, pts):
        self.landmark = pts


class FakeHand:
    def __init__(self, pts):
        self.landmark = [FakePt(x, y) for x, y in pts]


class FakePose:
    def __init__(self, pts):
        self.landmark = pts


class FakeResults:
    def __init__(self, pose_pts, left=None, right=None, face=None):
        self.pose_landmarks = FakePose(pose_pts)
        self.left_hand_landmarks = left
        self.right_hand_landmarks = right
        self.face_landmarks = face


def make_hand(x0=0.5, y0=0.5, spread=0.05):
    rng = np.random.RandomState(0)
    pts = np.array([[x0, y0]])
    pts = np.vstack([pts, pts[0] + (rng.rand(20, 2) - 0.5) * spread])
    return FakeHand(pts)


def make_pose(elbow_l, wrist_l, elbow_r, wrist_r):
    pts = [FakePt(0, 0, 0.0)] * 17
    pts = list(pts)
    pts[7] = FakePt(0.47, 0.05, 0.99)  # L ear (face-center anchor)
    pts[8] = FakePt(0.53, 0.05, 0.99)  # R ear
    pts[11] = FakePt(0.4, 0.2, 0.99)  # L shoulder
    pts[12] = FakePt(0.6, 0.2, 0.99)  # R shoulder
    pts[13] = FakePt(*elbow_l, 0.99)
    pts[15] = FakePt(*wrist_l, 0.99)
    pts[14] = FakePt(*elbow_r, 0.99)
    pts[16] = FakePt(*wrist_r, 0.99)
    return pts


class TestHandExtractor(unittest.TestCase):
    def test_local_hand_independent_of_global_wrist(self):
        """Hand output is local (near origin), not offset by canonical wrist."""
        hand = make_hand()
        out = lm._process_hand(hand)
        self.assertEqual(out.shape, (42,))
        # Local coords: wrist at origin, magnitude bounded by palm scale.
        self.assertAlmostEqual(out[0], 0.0)
        self.assertAlmostEqual(out[1], 0.0)
        # Even a far canonical wrist (~2.2) must not leak into hand slots.
        self.assertTrue(np.all(np.abs(out) < 1.0),
                        f"hand magnitude too large: {np.abs(out).max()}")

    def test_tiny_palm_returns_zeros(self):
        hand = make_hand(spread=1e-4)  # palm_size ~ 1e-4 < 0.012
        out = lm._process_hand(hand)
        np.testing.assert_array_equal(out, np.zeros(42))

    def test_missing_hand_returns_zeros(self):
        np.testing.assert_array_equal(lm._process_hand(None), np.zeros(42))

    def test_writeability_restored_on_exception(self):
        # cv2.cvtColor copies, so the check targets the array model.process
        # actually saw (captured), plus the returned image.
        seen = {}

        class Boom:
            def process(self, im):
                seen['arr'] = im
                raise RuntimeError("mp failure")

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            lm.detect(img, Boom())
        self.assertTrue(seen['arr'].flags.writeable)

    def test_writeability_restored_on_success(self):
        class Echo:
            def process(self, im):
                assert not im.flags.writeable
                return "ok"

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        out_img, res = lm.detect(img, Echo())
        self.assertEqual(res, "ok")
        self.assertTrue(out_img.flags.writeable)


class TestIdleGate(unittest.TestCase):
    def test_gate_zeroes_hanging_side_only(self):
        # Left arm hangs straight down → canonical wrist y ≈ 2.2 → gated.
        # Right arm raised → kept.
        pose = make_pose(elbow_l=(0.4, 0.5), wrist_l=(0.4, 0.9),
                         elbow_r=(0.62, 0.1), wrist_r=(0.62, 0.05))
        res = FakeResults(pose, left=make_hand(), right=make_hand())
        vec = lm.extract(res)
        self.assertEqual(vec.shape, (356,))
        np.testing.assert_array_equal(vec[0:9], np.zeros(9))     # left pose
        np.testing.assert_array_equal(vec[18:60], np.zeros(42))  # left hand
        self.assertTrue(np.any(vec[9:18] != 0.0))   # right pose kept
        self.assertTrue(np.any(vec[60:102] != 0.0))  # right hand kept
        # arm_depths: left arm gated → pose/hand zeroed, depth keeps measured
        # value (no longer forced to 1.0, so no jump when entering idle)
        self.assertTrue(np.isfinite(vec[102]) and vec[102] > 0.0)
        expected = lm._compute_arm_depth_ratio(
            (0.4, 0.2), (0.4, 0.5), (0.4, 0.9))
        self.assertAlmostEqual(vec[102], expected)

    def test_active_arms_pass_gate(self):
        pose = make_pose(elbow_l=(0.38, 0.1), wrist_l=(0.38, 0.05),
                         elbow_r=(0.62, 0.1), wrist_r=(0.62, 0.05))
        res = FakeResults(pose, left=make_hand(), right=make_hand())
        vec = lm.extract(res)
        self.assertTrue(np.any(vec[0:9] != 0.0))
        self.assertTrue(np.any(vec[9:18] != 0.0))


class TestInjection(unittest.TestCase):
    def _one_hand_seq(self, t=24):
        seq = np.zeros((t, 356), dtype=np.float32)
        rng = np.random.RandomState(1)
        seq[:, 0:9] = rng.rand(t, 9).astype(np.float32) + 0.1
        seq[:, 18:60] = rng.rand(t, 42).astype(np.float32) + 0.1
        seq[:, 102:104] = 1.0  # default arm depths
        return seq

    def test_one_hand_detection(self):
        seq = self._one_hand_seq()
        self.assertTrue(ds._side_active(seq, 0))
        self.assertFalse(ds._side_active(seq, 1))

    def test_two_hand_passes_through(self):
        seq = self._one_hand_seq()
        seq[:, 9:18] = 0.5
        seq[:, 60:102] = 0.5
        out = ds.inject_inactive_side(seq, p=1.0)
        np.testing.assert_array_equal(out, seq)

    def test_injection_probability(self):
        np.random.seed(7)
        n, hits = 2000, 0
        for _ in range(n):
            out = ds.inject_inactive_side(self._one_hand_seq())
            if np.any(out[:, 9:18] != 0.0):
                hits += 1
        rate = hits / n
        self.assertTrue(0.28 < rate < 0.42, f"injection rate {rate}")

    def test_injection_plausible_and_stationary(self):
        np.random.seed(3)
        out = None
        for _ in range(50):
            cand = ds.inject_inactive_side(self._one_hand_seq(), p=1.0)
            out = cand
            break
        wrist_y = out[:, 9 + 7]  # right wrist y
        self.assertTrue(np.all((wrist_y > 0.8) & (wrist_y < 2.0)),
                        f"wrist y out of rest range: {wrist_y.min()}-{wrist_y.max()}")
        step = np.abs(np.diff(out[:, 9:18], axis=0)).max()
        self.assertLess(step, 0.15, f"abrupt transition: {step}")
        # Local hand has no global offset: wrist landmark at local origin.
        hand = out[:, 60:102].reshape(-1, 21, 2)
        np.testing.assert_allclose(hand[:, 0, :], 0.0, atol=1e-6)

    def test_labels_unchanged(self):
        X = np.stack([self._one_hand_seq() for _ in range(4)])
        y = np.eye(4, dtype=np.float32)
        Xa, ya = ds.augment_train_data(X, y, augment_factor=2)
        self.assertEqual(Xa.shape[0], 12)
        # Each original + 2 copies keep the same one-hot label.
        for i in range(4):
            np.testing.assert_array_equal(ya[3 * i], y[i])
            np.testing.assert_array_equal(ya[3 * i + 1], y[i])
            np.testing.assert_array_equal(ya[3 * i + 2], y[i])

    def test_p_zero_leaves_unchanged(self):
        seq = self._one_hand_seq()
        out = ds.inject_inactive_side(seq, p=0.0)
        np.testing.assert_array_equal(out, seq)


class TestLayout(unittest.TestCase):
    def test_dim_356(self):
        self.assertEqual(lm.n_features(), 356)
        self.assertEqual(lm.BODY_ONLY_DIM, 356)
        pose = make_pose(elbow_l=(0.38, 0.1), wrist_l=(0.38, 0.05),
                         elbow_r=(0.62, 0.1), wrist_r=(0.62, 0.05))
        vec = lm.extract(FakeResults(pose, left=make_hand(), right=make_hand()))
        self.assertEqual(vec.shape, (356,))


class TestShoulderOffsets(unittest.TestCase):
    def test_offsets_match_formula(self):
        pose = make_pose(elbow_l=(0.38, 0.1), wrist_l=(0.38, 0.05),
                         elbow_r=(0.62, 0.1), wrist_r=(0.62, 0.05))
        left = make_hand(x0=0.3, y0=0.7)
        vec = lm.extract(FakeResults(pose, left=left, right=make_hand()))
        # Raised anchors: shoulder x, face-center y (ear midpoint y = 0.05).
        anchors = [(0.4, 0.05), (0.6, 0.05), (0.5, 0.05)]
        pts = np.array([[p.x, p.y] for p in left.landmark])
        palm = max(float(np.linalg.norm(pts[0] - pts[9])),
                   float(np.linalg.norm(pts[5] - pts[17])) * 1.25)
        expected = np.concatenate(
            [((pts - np.array(a)) / palm * lm.STD_PALM).flatten()
             for a in anchors])
        np.testing.assert_allclose(vec[104:230], expected, rtol=1e-6)

    def test_missing_anchors_gives_zeros(self):
        pose = make_pose(elbow_l=(0.38, 0.1), wrist_l=(0.38, 0.05),
                         elbow_r=(0.62, 0.1), wrist_r=(0.62, 0.05))
        # Wipe the ears: no pose ears and no face mesh → no raised anchors.
        pose[7] = FakePt(0, 0, 0.0)
        pose[8] = FakePt(0, 0, 0.0)
        vec = lm.extract(FakeResults(pose, left=make_hand(), right=make_hand()))
        self.assertEqual(vec.shape, (356,))
        np.testing.assert_array_equal(vec[104:356], np.zeros(252))
        # Hand slots still populated.
        self.assertTrue(np.any(vec[18:60] != 0.0))

    def test_idle_side_zeroes_sh_block(self):
        pose = make_pose(elbow_l=(0.4, 0.5), wrist_l=(0.4, 0.9),
                         elbow_r=(0.62, 0.1), wrist_r=(0.62, 0.05))
        vec = lm.extract(FakeResults(pose, left=make_hand(), right=make_hand()))
        np.testing.assert_array_equal(vec[104:230], np.zeros(126))
        self.assertTrue(np.any(vec[230:356] != 0.0))


if __name__ == '__main__':
    unittest.main()
