#!/usr/bin/env python3
"""Add adjustable Gaussian noise / distance-scale to collected landmark files.

Takes existing sequences under data/<action>/<seq>/<frame>.npy and generates
new noisy copies as data/<action>/<new_seq>/<frame>.npy.

Noise is added ONLY to non-zero keypoints by default (0.0 = missing landmark
stays 0.0); pass --noise-zeros to also noise zero entries (breaks the mask).

Distance simulation (--scale-range): each copy's whole sequence is multiplied
by one uniform factor, mimicking the same gesture performed closer/farther
from the camera. This is a uniform scale, NOT the same as the per-frame
Gaussian _scale_jitter() used in training augmentation (dataset.py).

Usage:
    venv/bin/python scripts/add_noise.py --action hello --copies 1 --noise-std 0.02
    venv/bin/python scripts/add_noise.py --action hello --copies 2 --scale-range 0.7 1.4 --seed 42
    venv/bin/python scripts/add_noise.py --action hello --copies 1 --noise-zeros --noise-std 0.05 --seed 42
    venv/bin/python scripts/add_noise.py --all-actions --copies 2 --scale-range 0.7 1.4 --seed 42
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np

import config as cfg


def _coord_mask(n_features):
    """True for (dx,dy) coordinate channels, False for visibility and ratio channels."""
    from vsl_translator.landmarks import POSE_DIM, HAND_DIM, ARM_DEPTH_DIM
    coord = np.ones(n_features, dtype=bool)
    if n_features >= POSE_DIM:
        coord[np.arange(2, POSE_DIM, 3)] = False
    # Exclude arm_depth ratio channels (face-offset blocks stay augmentable)
    hand_end = POSE_DIM + HAND_DIM * 2
    if n_features > hand_end:
        coord[hand_end:hand_end + ARM_DEPTH_DIM] = False
    return coord


def add_noise(vec, std, noise_zeros=False):
    """Gaussian noise on (dx,dy) coords.

    Default: non-zero entries only, zeros (missing) preserved. With
    noise_zeros=True, zero entries are noised too — visibility channels
    stay untouched in both cases.
    """
    noise = np.random.normal(0.0, std, size=vec.shape)
    out = vec.copy()
    coord = _coord_mask(vec.shape[-1])
    if noise_zeros:
        mask = coord
    else:
        mask = (vec != 0.0) & coord
    out[mask] += noise[mask]
    return out


def apply_distance_scale(vec, factor):
    """Multiply all (dx,dy) offsets by one factor (uniform distance change).

    Visibility untouched; exact zeros stay zero (0 * factor = 0).
    """
    out = vec.copy()
    coord = _coord_mask(vec.shape[-1])
    out[coord] *= factor
    return out


def sorted_frame_files(seq_dir):
    files = [f for f in os.listdir(seq_dir) if f.endswith('.npy')]
    def key(f):
        stem = os.path.splitext(f)[0]
        return int(stem) if stem.isdigit() else stem
    return sorted(files, key=key)


def main():
    parser = argparse.ArgumentParser(description="Generate noisy copies of landmark sequences.")
    parser.add_argument("--action", type=str, default="hello",
                        help="Action folder under data/ (default: hello)")
    parser.add_argument("--all-actions", action="store_true", default=False,
                        help="Process every action in cfg.ACTIONS instead of --action "
                             "(--src-seqs is ignored in this mode).")
    parser.add_argument("--copies", type=int, default=1,
                        help="Noisy copies per source sequence (default: 1)")
    parser.add_argument("--noise-std", type=float, default=cfg.NOISE_STD,
                        help=f"Gaussian noise std (default: {cfg.NOISE_STD})")
    parser.add_argument("--noise-zeros", action="store_true", default=False,
                        help="Also add noise to zero entries (breaks the 0.0=missing "
                             "mask semantics — use sparingly, compare confusion matrix).")
    parser.add_argument("--scale-range", type=float, nargs=2, default=None,
                        metavar=("LOW", "HIGH"),
                        help="Per-copy uniform distance factor range, e.g. --scale-range 0.7 1.4 "
                             "(each copy scaled by one uniform(LOW, HIGH) factor before noise).")
    parser.add_argument("--src-seqs", type=int, nargs="*", default=None,
                        help="Source sequence ids (default: all numeric seqs)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: None)")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.noise_zeros:
        print("WARNING: --noise-zeros breaks the 0.0=missing mask (Masking layer "
              "can no longer skip those points). Use low --noise-std and compare metrics.")
    if args.scale_range is not None:
        lo, hi = args.scale_range
        if lo <= 0 or hi <= 0 or lo > hi:
            print(f"Error: invalid --scale-range {args.scale_range} (need 0 < LOW <= HIGH).")
            return
        if lo < 0.5 or hi > 2.0:
            print(f"WARNING: wide --scale-range {args.scale_range} may create "
                  f"unrealistic offsets; recommended 0.7 1.4.")

    actions = list(cfg.ACTIONS) if args.all_actions else [args.action]
    if args.all_actions and args.src_seqs is not None:
        print("Note: --src-seqs ignored with --all-actions (all seqs of each action).")

    total = 0
    for action in actions:
        total += _process_action(action, args)
    print(f"Done: {total} noisy sequences across {len(actions)} action(s) (std={args.noise_std}).")


def _process_action(action, args):
    action_root = os.path.join(cfg.DATA_DIR, action)
    if not os.path.isdir(action_root):
        print(f"Error: '{action_root}' not found.")
        return 0

    existing = [d for d in os.listdir(action_root)
                if os.path.isdir(os.path.join(action_root, d)) and d.isdigit()]
    if args.src_seqs is not None and not args.all_actions:
        src_ids = [str(s) for s in args.src_seqs
                   if os.path.isdir(os.path.join(action_root, str(s)))]
        missing = set(str(s) for s in args.src_seqs) - set(src_ids)
        if missing:
            print(f"  Warning: seq not found, skipped: {sorted(missing)}")
    else:
        src_ids = sorted(existing, key=int)
    if not src_ids:
        print(f"No source sequences found for '{action}'.")
        return 0

    next_idx = max([int(d) for d in existing], default=0) + 1
    created = 0
    for src in sorted(src_ids, key=int):
        src_dir = os.path.join(action_root, src)
        frames = sorted_frame_files(src_dir)
        if not frames:
            print(f"  Skip seq {src}: no .npy frames.")
            continue
        for _ in range(args.copies):
            dst_dir = os.path.join(action_root, str(next_idx))
            os.makedirs(dst_dir, exist_ok=False)
            factor = None
            if args.scale_range is not None:
                factor = float(np.random.uniform(args.scale_range[0], args.scale_range[1]))
            for f in frames:
                vec = np.load(os.path.join(src_dir, f))
                if factor is not None:
                    vec = apply_distance_scale(vec, factor)
                np.save(os.path.join(dst_dir, f), add_noise(vec, args.noise_std,
                                                            noise_zeros=args.noise_zeros))
            tag = f" [x{factor:.2f}]" if factor is not None else ""
            print(f"  {action}/{src} ({len(frames)} frames) → {action}/{next_idx}{tag}")
            next_idx += 1
            created += 1

    print(f"Done: {created} noisy sequences for '{action}' (std={args.noise_std}).")
    return created


if __name__ == '__main__':
    main()
