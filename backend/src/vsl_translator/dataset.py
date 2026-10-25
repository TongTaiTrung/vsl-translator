"""
Dataset loading and augmentation utilities for VSL keypoint sequences.
Includes zero-mask preservation to prevent corruption of missing keypoints.
"""

import os
import warnings
import numpy as np
from keras.utils import to_categorical
from scipy.interpolate import interp1d

import config as cfg
from vsl_translator.landmarks import (
    POSE_DIM, HAND_DIM, ARM_DEPTH_DIM, SH_OFF_REF_DIM, SH_OFF_DIM,
    BODY_ONLY_DIM, STD_PALM, STD_UPPER_ARM, STD_FOREARM,
    slice_bounds,
)
from vsl_translator.landmarks import coord_mask as _landmark_coord_mask
from vsl_translator.landmarks import shift_masks as _landmark_shift_masks


# ── Augmentation helpers (strictly preserve 0.0 missing keypoints) ───────────
# Layout: pose 6×(x,y,vis)=18 | lh | rh. Visibility (every 3rd pose channel) is
# a confidence score in [0,1], NOT a coordinate — additive augments must skip it.

def _coord_mask(n_features):
    """True for (dx,dy) coordinate channels, False for visibility and ratio channels."""
    return _landmark_coord_mask(n_features)


def _add_noise(sequence, std=None):
    if std is None:
        std = cfg.NOISE_STD
    """Add Gaussian noise to non-zero coordinates only (visibility untouched)."""
    noise = np.random.normal(0, std, size=sequence.shape)
    out = sequence.copy()
    coord = _coord_mask(sequence.shape[-1])
    if sequence.ndim == 3:
        mask = (sequence != 0.0) & coord[None, None, :]
    else:
        mask = (sequence != 0.0) & coord[None, :]
    out[mask] += noise[mask]
    return out


def _scale_jitter(sequence, std=None):
    if std is None:
        std = cfg.SCALE_JITTER_STD
    """Multiply all offsets of each frame by ~N(1, std).

    Simulates shoulder-width measurement jitter: when the reference length
    wobbles, every offset scales together. Per-frame factor, visibility untouched.
    """
    out = sequence.copy()
    coord = _coord_mask(sequence.shape[-1])
    if sequence.ndim == 3:
        factors = np.random.normal(1.0, std, size=(sequence.shape[0], 1, 1))
        out[:, :, coord] *= factors
    else:
        factors = np.random.normal(1.0, std, size=(sequence.shape[0], 1))
        out[:, coord] *= factors
    # A frame that was fully zero stays zero (0 * factor = 0).
    return out


def _time_warp(sequence, sigma=None):
    if sigma is None:
        sigma = cfg.TIME_WARP_SIGMA
    """Apply smooth temporal warping while strictly preserving zero-mask.

    Generates a random monotonically increasing warp path, then interpolates
    the original frames onto it. Concurrently interpolates a mask of non-zero
    elements; any value where warped_mask < 0.99 is reset to 0.0 to prevent
    phantom hand artifacts. Visibility channels are NOT interpolated (they
    are confidence scores, not coordinates) — they are sampled nearest-frame
    and then re-zeroed wherever the mask kills the frame.
    """
    seq_len = sequence.shape[0]
    orig_steps = np.arange(seq_len)
    warp = np.cumsum(np.abs(np.random.normal(1.0, sigma, size=seq_len)))
    warp = (warp - warp[0]) / (warp[-1] - warp[0]) * (seq_len - 1)

    interp_seq = interp1d(orig_steps, sequence, axis=0, kind='linear', fill_value='extrapolate')
    warped_seq = interp_seq(warp).astype(sequence.dtype)

    mask = (sequence != 0.0).astype(np.float32)
    interp_mask = interp1d(orig_steps, mask, axis=0, kind='linear', fill_value='extrapolate')
    warped_mask = interp_mask(warp)

    warped_seq[warped_mask < 0.99] = 0.0

    # Restore visibility from nearest source frame (no interpolation bleed).
    coord = _coord_mask(sequence.shape[-1])
    vis_idx = np.where(~coord)[0]
    if len(vis_idx) > 0:
        nearest = np.clip(np.rint(warp).astype(int), 0, seq_len - 1)
        if sequence.ndim == 3:
            warped_seq[:, :, vis_idx] = sequence[nearest][:, :, vis_idx]
            kill = warped_mask[:, :, vis_idx] < 0.99
            for k, j in enumerate(vis_idx):
                warped_seq[kill[:, :, k], j] = 0.0
        else:
            warped_seq[:, vis_idx] = sequence[nearest][:, vis_idx]
            kill = warped_mask[:, vis_idx] < 0.99
            for k, j in enumerate(vis_idx):
                warped_seq[kill[:, k], j] = 0.0
    return warped_seq


def _random_shift(sequence, lo=None, hi=None):
    if lo is None:
        lo = cfg.SHIFT_RANGE[0]
    if hi is None:
        hi = cfg.SHIFT_RANGE[1]
    """Small 2D jitter of (dx,dy) coordinates (visibility untouched).

    Simulates shoulder-midpoint jitter: the origin wobbles slightly per
    sequence. Drawn once per sequence as an (sx, sy) pair, not a diagonal scalar.
    """
    shift = np.random.uniform(lo, hi, size=(2,))
    out = sequence.copy()
    n_feat = sequence.shape[-1]
    coord = _coord_mask(n_feat)
    if sequence.ndim == 3:
        nz = (sequence != 0.0) & coord[None, None, :]
    else:
        nz = (sequence != 0.0) & coord[None,:]
    is_dx, is_dy = _landmark_shift_masks(n_feat)
    if sequence.ndim == 3:
        out[:, :, is_dx] += np.where(nz[:, :, is_dx], shift[0], 0.0)
        out[:, :, is_dy] += np.where(nz[:, :, is_dy], shift[1], 0.0)
    else:
        out[:, is_dx] += np.where(nz[:, is_dx], shift[0], 0.0)
        out[:, is_dy] += np.where(nz[:, is_dy], shift[1], 0.0)
    return out


def _random_frame_drop(sequence, p=None):
    if p is None:
        p = cfg.FRAME_DROP_PROB
    """Zero-out random frames to simulate tracking failures."""
    mask = np.random.rand(sequence.shape[0]) > p
    out = sequence.copy()
    out[~mask] = 0.0
    return out


def _random_hand_dropout(sequence, p=None):
    if p is None:
        p = cfg.HAND_DROPOUT_PROB
    """Zero out one entire hand (pose-relative block) with probability p.

    Counters the hand-presence shortcut: when one class systematically lacks
    a hand (occlusion / out of frame), the model learns presence instead of
    motion. Operates on the canonical layout pose [0:18] | lh [18:60] | rh [60:102]
    | arm_depths [102:104] | sh_off [104:356].

    Skipped for one-hand samples: wiping the single active hand would erase
    the hand-shape detail that distinguishes similar one-hand signs
    (e.g. khoingu vs khochiu) — and dropping the already-zero side is a no-op.
    """
    if _is_one_sided(sequence):
        return sequence.copy()  # one-hand sample: keep its only active hand
    out = sequence.copy()
    _, _, _, sh_start, _ = slice_bounds()
    if np.random.rand() < p:
        hand = np.random.choice(['lh', 'rh'])
        if hand == 'lh':
            out[:, POSE_DIM:POSE_DIM + HAND_DIM] = 0.0
            if out.shape[-1] >= sh_start + SH_OFF_REF_DIM:
                out[:, sh_start:sh_start + SH_OFF_REF_DIM] = 0.0
        else:
            out[:, POSE_DIM + HAND_DIM:POSE_DIM + HAND_DIM * 2] = 0.0
            if out.shape[-1] >= sh_start + SH_OFF_REF_DIM * 2:
                out[:, sh_start + SH_OFF_REF_DIM:sh_start + SH_OFF_REF_DIM * 2] = 0.0
    return out


def _random_hand_swap(sequence, p=None):
    if p is None:
        p = cfg.HAND_SWAP_PROB
    """Swap the two hand blocks (lh↔rh) with probability p, label unchanged.

    Forces side-invariance: the model must classify by motion trajectory, not
    by which hand slot is active. Pure block permutation — exact-zero blocks
    stay zero, no interpolation bleed. Only valid when classes are NOT
    lateralized (no class distinguished by left-vs-right hand alone).

    Skipped for one-hand samples: swapping would move the active hand into a
    slot that is always zero in real data (e.g. khoingu/khochiu are both
    one-sided), creating out-of-distribution copies that blur similar signs.
    """
    if _is_one_sided(sequence):
        return sequence.copy()  # one-hand sample: side is real signal, don't mirror
    out = sequence.copy()
    _, _, depths_end, sh_start, _ = slice_bounds()
    if np.random.rand() < p:
        lh = out[..., POSE_DIM:POSE_DIM + HAND_DIM].copy()
        rh = out[..., POSE_DIM + HAND_DIM:POSE_DIM + HAND_DIM * 2].copy()
        out[..., POSE_DIM:POSE_DIM + HAND_DIM] = rh
        out[..., POSE_DIM + HAND_DIM:POSE_DIM + HAND_DIM * 2] = lh
        # Also swap arm_depths (L↔R) if present
        if out.shape[-1] > POSE_DIM + HAND_DIM * 2:
            depth_slice = slice(POSE_DIM + HAND_DIM * 2, depths_end)
            d = out[..., depth_slice].copy()
            out[..., depth_slice] = d[..., ::-1]
        # Also swap shoulder-offset sides (L↔R) if present
        if out.shape[-1] >= sh_start + SH_OFF_REF_DIM * 2:
            sL = out[..., sh_start:sh_start + SH_OFF_REF_DIM].copy()
            sR = out[..., sh_start + SH_OFF_REF_DIM:sh_start + SH_OFF_REF_DIM * 2].copy()
            out[..., sh_start:sh_start + SH_OFF_REF_DIM] = sR
            out[..., sh_start + SH_OFF_REF_DIM:sh_start + SH_OFF_REF_DIM * 2] = sL
    return out


def _side_energies(seq_2d):
    """Total |pose + hand| energy per side (body channels only)."""
    n = min(seq_2d.shape[-1], BODY_ONLY_DIM)
    seq_2d = seq_2d[..., :n]
    l_pose = np.abs(seq_2d[..., 0:9]).sum()
    r_pose = np.abs(seq_2d[..., 9:18]).sum()
    l_hand = np.abs(seq_2d[..., POSE_DIM:POSE_DIM + HAND_DIM]).sum() if n > POSE_DIM else 0.0
    r_hand = np.abs(seq_2d[..., POSE_DIM + HAND_DIM:POSE_DIM + 2 * HAND_DIM]).sum() if n > POSE_DIM + HAND_DIM else 0.0
    return float(l_pose + l_hand), float(r_pose + r_hand)


def _is_one_sided(sequence, thresh=0.15):
    """True if one side holds ~all the energy (minor side < thresh of total).

    Zero-absolute checks miss near-zero leakage in the idle side; an energy
    ratio tolerates that while still catching truly one-hand signs
    (khoingu/khochiu/xinchao all sit below 0.08, two-hand signs above it —
    except a few strongly asymmetric samples, which are also better left alone).
    """
    seq_2d = sequence.reshape(-1, sequence.shape[-1]) if sequence.ndim == 3 else sequence
    l, r = _side_energies(seq_2d)
    total = l + r
    if total <= 0:
        return False
    return min(l, r) / total < thresh


def _side_active(seq_2d, side):
    """True if the side's pose, hand, or shoulder-offset block has any non-zero."""
    _, _, _, sh_start, sh_end_full = slice_bounds()
    pose_base = 0 if side == 0 else 9
    hand_base = POSE_DIM if side == 0 else POSE_DIM + HAND_DIM
    pose_nz = np.any(seq_2d[:, pose_base:pose_base + 9] != 0.0)
    hand_nz = np.any(seq_2d[:, hand_base:hand_base + HAND_DIM] != 0.0)
    sh_base = sh_start if side == 0 else sh_start + SH_OFF_REF_DIM
    sh_nz = (seq_2d.shape[1] >= sh_start + SH_OFF_REF_DIM * 2
             and np.any(seq_2d[:, sh_base:sh_base + SH_OFF_REF_DIM] != 0.0))
    return bool(pose_nz or hand_nz or sh_nz)


def inject_inactive_side(sequence, p=None):
    """Inject a plausible neutral second arm into one-hand samples (train only).

    Detects one-hand samples (one side's pose [0:9]/[9:18] + hand block all
    zero while the other side is active) and with probability p fills the
    inactive side's pose + hand + shoulder-offset slots with a quasi-stationary
    resting arm (canonical wrist y in INACTIVE_WRIST_Y_RANGE, drift
    INACTIVE_DRIFT_STD). Local hand shape is mostly static + small jitter,
    no global wrist added. Shoulder offsets are derived from the synthetic wrist
    trajectory against canonical raised-shoulder anchors (shoulder x at ±0.5,
    y at a face-level proxy, plus their midpoint) plus the same
    local hand shape, so the injected side matches the real joint distribution.
    Label-preserving. Two-hand and all-zero samples pass through unchanged.
    """
    _, _, _, sh_start, sh_end_full = slice_bounds()
    if p is None:
        p = cfg.INACTIVE_INJECT_PROB
    out = sequence.copy()
    seq_2d = out.reshape(-1, out.shape[-1]) if out.ndim == 3 else out
    left_on = _side_active(seq_2d, 0)
    right_on = _side_active(seq_2d, 1)
    if left_on == right_on:  # both active or both idle → nothing to do
        return out
    if np.random.rand() >= p:
        return out
    side = 1 if left_on else 0  # the inactive side to fill
    t = seq_2d.shape[0]
    y_lo, y_hi = cfg.INACTIVE_WRIST_Y_RANGE
    drift = cfg.INACTIVE_DRIFT_STD
    sx = -0.5 if side == 0 else 0.5
    wrist_y0 = float(np.random.uniform(y_lo, y_hi))
    wrist_x0 = sx * 0.6
    elbow_y0 = wrist_y0 * 0.55
    # Quasi-stationary: one base posture + small cumulative drift, no jumps.
    wrist_y = wrist_y0 + np.cumsum(np.random.normal(0, drift, size=t))
    wrist_x = wrist_x0 + np.cumsum(np.random.normal(0, drift, size=t))
    elbow_y = elbow_y0 + np.cumsum(np.random.normal(0, drift, size=t))
    elbow_x = (sx * 0.7) + np.cumsum(np.random.normal(0, drift, size=t))
    pose_base = 0 if side == 0 else 9
    hand_base = POSE_DIM if side == 0 else POSE_DIM + HAND_DIM
    vis = 0.9
    seq_2d[:, pose_base:pose_base + 3] = np.stack(
        [np.full(t, sx), np.zeros(t), np.full(t, vis)], axis=1)
    seq_2d[:, pose_base + 3:pose_base + 6] = np.stack(
        [elbow_x, elbow_y, np.full(t, vis)], axis=1)
    seq_2d[:, pose_base + 6:pose_base + 9] = np.stack(
        [wrist_x, wrist_y, np.full(t, vis)], axis=1)
    # Mostly-static local hand: fixed neutral shape + per-frame jitter.
    base_hand = (np.random.rand(21, 2) - 0.5) * STD_PALM * 0.8
    base_hand[0] = [0.0, 0.0]  # wrist at local origin
    # Injected palm scale: median-scale proxy so shoulder offsets are canonical.
    # Reuse the canonical neutral arm length as the scale reference.
    inject_palm = (STD_UPPER_ARM + STD_FOREARM) / 4.0
    # Canonical raised-shoulder anchors for the synthetic side: shoulder x at
    # ±0.5 lifted to a face-level y proxy, plus their midpoint. Mirrors the real
    # _get_raised_shoulders anchors (shoulder x, face y).
    _fc_y = -0.8
    inject_anchors = ((-0.5, _fc_y), (0.5, _fc_y), (0.0, _fc_y))
    sh_base = sh_start if side == 0 else sh_start + SH_OFF_REF_DIM
    has_sh = seq_2d.shape[1] >= sh_start + SH_OFF_REF_DIM * 2
    for i in range(t):
        jitter = np.random.normal(0, drift, size=(21, 2))
        jitter[0] = [0.0, 0.0]
        local_hand = base_hand + jitter
        seq_2d[i, hand_base:hand_base + HAND_DIM] = local_hand.flatten()
        if has_sh:
            wrist_pos = np.array([wrist_x[i], wrist_y[i]], dtype=np.float64)
            global_hand = local_hand / STD_PALM * inject_palm + wrist_pos
            blocks = [((global_hand - np.array(a)) / inject_palm * STD_PALM).flatten()
                      for a in inject_anchors]
            seq_2d[i, sh_base:sh_base + SH_OFF_REF_DIM] = np.concatenate(blocks)
    return out


def augment_sequence(sequence):
    """Apply a random combination of augmentations to a single sequence."""
    seq = sequence.copy()
    # 1. Add noise to non-zero coords (visibility untouched)
    seq = _add_noise(seq)
    # 2. 50% chance scale jitter (shoulder-width measurement noise)
    if np.random.rand() < 0.5:
        seq = _scale_jitter(seq)
    # 3. 50% chance time warp with zero-mask preservation
    if np.random.rand() < 0.5:
        seq = _time_warp(seq)
    # 4. 50% chance shoulder-midpoint jitter (2D shift, visibility untouched)
    if np.random.rand() < 0.5:
        seq = _random_shift(seq)
    # 5. 30% chance frame drop at the very end
    if np.random.rand() < 0.3:
        seq = _random_frame_drop(seq)
    # 6. Hand dropout: force learning motion, not hand presence
    seq = _random_hand_dropout(seq)
    # 7. Hand swap: force side-invariance (label unchanged)
    seq = _random_hand_swap(seq)
    # 8. Inactive-side injection: neutral second arm into one-hand samples
    # (train only — augment_train_data is never called on val/test).
    # Runs after the swap so the injected side is pose/hand consistent.
    seq = inject_inactive_side(seq)
    return seq


# ── Feature standardization (strictly preserve 0.0 missing keypoints) ────────

def standardize_features(X, mean=None, std=None):
    """Per-feature standardization ignoring missing keypoints (0.0).

    When computing mean/std:
      - 0.0 entries are replaced with NaN
      - nanmean and nanstd are calculated per feature across samples and frames
      - Constant/unobserved features get std = 1.0, mean = 0.0

    After (X - mean) / std:
      - All original 0.0 positions are explicitly restored to 0.0.

    Args:
        X: np.ndarray of shape (num_samples, seq_len, n_features) or (seq_len, n_features)
        mean: Optional precomputed mean array of shape (n_features,)
        std: Optional precomputed std array of shape (n_features,)

    Returns:
        X_norm: Standardized array with preserved 0.0 (same shape as X)
        mean: The mean used (n_features,)
        std: The std used (n_features,)
    """
    orig_shape = X.shape
    X_flat = X.reshape(-1, orig_shape[-1]).astype(np.float32)
    zero_mask = (X_flat == 0.0)

    if mean is None or std is None:
        X_nan = np.where(zero_mask, np.nan, X_flat)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mean = np.nanmean(X_nan, axis=0)
            std = np.nanstd(X_nan, axis=0)
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.nan_to_num(std, nan=1.0)
        std[std < 1e-6] = 1.0
    else:
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)

    X_norm = (X_flat - mean) / std
    X_norm[zero_mask] = 0.0
    return X_norm.reshape(orig_shape), mean, std


# ── Dataset loading & augmentation ───────────────────────────────────────────

def load_raw_dataset(data_dir=cfg.DATA_DIR, actions=cfg.ACTIONS, label_map=cfg.LABEL_MAP,
                     n_sequences=cfg.N_SEQUENCES, sequence_len=cfg.SEQUENCE_LEN,
                     return_devices=False):
    """Load raw keypoint sequences without any augmentation or normalization.

    Reads optional meta.json per sequence dir for the camera source label
    (written by collect_data.py --device / migrate_data.py, else "legacy").

    Returns:
        X (np.ndarray): Shape (num_samples, sequence_len, num_features), dtype float32
        y (np.ndarray): Shape (num_samples, num_classes), dtype float32 (one-hot)
        devices (list[str], only if return_devices): camera source per sample
    """
    sequences, labels, devices = [], [], []
    found_actions = []

    for action in actions:
        action_dir = os.path.join(data_dir, action)
        if not os.path.isdir(action_dir):
            continue

        seq_folders = sorted(
            [d for d in os.listdir(action_dir) if os.path.isdir(os.path.join(action_dir, d))],
            key=lambda x: int(x) if x.isdigit() else x
        )

        action_seq_count = 0
        for seq_folder in seq_folders:
            seq_dir = os.path.join(action_dir, seq_folder)
            window = []
            valid_seq = True
            for n_frame in range(sequence_len):
                frame_path = os.path.join(seq_dir, f"{n_frame}.npy")
                if not os.path.isfile(frame_path):
                    valid_seq = False
                    break
                frame_data = np.load(frame_path)
                if frame_data.shape == (104,):
                    # Backward compatibility: pad 104-dim frames to 356-dim.
                    # Zero-frames stay all zeros for Masking(0.0) compatibility;
                    # active frames get default arm depth ratio 1.0 and zero
                    # shoulder-offsets (no anchor info recoverable from stored data).
                    pad_val = 0.0 if not np.any(frame_data) else 1.0
                    sh_zeros = np.zeros(252, dtype=frame_data.dtype)
                    frame_data = np.concatenate([
                        frame_data[:102],
                        np.full(2, pad_val, dtype=frame_data.dtype),
                        sh_zeros,
                    ])
                window.append(frame_data)

            if valid_seq and len(window) == sequence_len:
                sequences.append(np.array(window, dtype=np.float32))
                labels.append(label_map[action])
                action_seq_count += 1
                try:
                    import json as _json
                    with open(os.path.join(seq_dir, "meta.json")) as _f:
                        devices.append(_json.load(_f).get("device", "legacy"))
                except (OSError, ValueError, KeyError):
                    devices.append("legacy")

        if action_seq_count > 0:
            found_actions.append(f"{action} ({action_seq_count} raw)")

    if not sequences:
        raise FileNotFoundError(
            f"No valid sequences found in '{data_dir}'. "
            "Please record keypoint data using scripts/collect_data.py first."
        )

    print(f"Loaded raw dataset: {len(sequences)} total sequences from {found_actions}")
    X = np.array(sequences, dtype=np.float32)
    y = to_categorical(labels, num_classes=len(actions)).astype(np.float32)
    if return_devices:
        return X, y, devices
    return X, y


def augment_train_data(X_train, y_train, augment_factor=cfg.AUGMENT_FACTOR):
    """Augment ONLY training sequences to prevent data leakage.

    For each sample in X_train, keeps the original sample and adds
    `augment_factor` perturbed copies.

    Args:
        X_train: (N, seq_len, n_features)
        y_train: (N, num_classes)
        augment_factor: number of augmented copies per sample

    Returns:
        X_aug: (N * (1 + augment_factor), seq_len, n_features)
        y_aug: (N * (1 + augment_factor), num_classes)
    """
    if augment_factor <= 0:
        return X_train, y_train

    aug_X, aug_y = [], []
    for seq, label in zip(X_train, y_train):
        aug_X.append(seq)
        aug_y.append(label)
        for _ in range(augment_factor):
            aug_X.append(augment_sequence(seq))
            aug_y.append(label)

    X_aug = np.array(aug_X, dtype=np.float32)
    y_aug = np.array(aug_y, dtype=np.float32)
    print(f"Augmented train data: {len(X_train)} → {len(X_aug)} samples ({augment_factor}x expansion)")
    return X_aug, y_aug


def load_dataset(data_dir=cfg.DATA_DIR, actions=cfg.ACTIONS, label_map=cfg.LABEL_MAP,
                 n_sequences=cfg.N_SEQUENCES, sequence_len=cfg.SEQUENCE_LEN,
                 augment=False, augment_factor=cfg.AUGMENT_FACTOR):
    """Backward-compatible loader.

    NOTE: For training without data leakage, use `load_raw_dataset()` then split,
    then `augment_train_data()`, and finally `standardize_features()`.
    """
    X, y = load_raw_dataset(data_dir=data_dir, actions=actions, label_map=label_map,
                            n_sequences=n_sequences, sequence_len=sequence_len)
    if augment:
        X, y = augment_train_data(X, y, augment_factor=augment_factor)
    X, mean, std = standardize_features(X)

    stats_path = os.path.join(cfg.MODELS_DIR, "feature_stats.npz")
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    np.savez(stats_path, mean=mean, std=std)
    return X, y
