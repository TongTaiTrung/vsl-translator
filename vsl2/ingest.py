"""Video → featurized data: one video becomes 12 frames → 12 .npy files.

Pipeline:
  1. Read all frames with cv2.VideoCapture.
  2. MediaPipe detect per frame; has_hand = left or right hand present.
  3. Trim leading/trailing no-hand frames (keeps ≥ SEQUENCE_LEN or raises).
  4. Downsample to exactly SEQUENCE_LEN: repeatedly drop odd-indexed frames
     (i & 1 == 1); linspace fallback if a pass removes nothing.
  5. Feature-extract each kept frame (same calibrator → extract → smoother
     pipeline as predict.extract_frame_vector) and save 0..11.npy + meta.json
     in data/<action>/<next_seq>/ (same layout as collect.py).
"""

import json
import os

import cv2
import numpy as np

from . import defaults as cfg
from .landmarks import (
    ArmDepthCalibrator,
    ArmDepthSmoother,
    create_model,
    detect,
    normalize_frame,
)
from .predict import extract_frame_vector


def _read_frames(video_path):
    """Read every frame of a video file. Returns list of BGR ndarrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames read from: {video_path}")
    return frames


def _hand_flags(frames):
    """Per-frame bool: True if MediaPipe sees a left or right hand."""
    flags = []
    with create_model() as holistic:
        for frame in frames:
            std, _ = normalize_frame(frame)
            _, results = detect(std, holistic)
            flags.append(results.left_hand_landmarks is not None
                         or results.right_hand_landmarks is not None)
    return flags


def _trim_static_ends(frames, flags):
    """Drop leading/trailing no-hand frames; keeps ≥ SEQUENCE_LEN or raises."""
    n = len(frames)
    start, end = 0, n
    while start < n and not flags[start]:
        start += 1
    while end > start and not flags[end - 1]:
        end -= 1
    trimmed = frames[start:end]
    if len(trimmed) < cfg.SEQUENCE_LEN:
        raise ValueError(
            f"Only {len(trimmed)} active frames after trimming, "
            f"need at least {cfg.SEQUENCE_LEN}.")
    return trimmed


def _downsample_to_12(frames):
    """Reduce to exactly SEQUENCE_LEN frames by dropping odd indices.

    Each pass drops odd-indexed frames (i & 1 == 1), which removes
    floor(n/2) frames. On the final pass — when dropping all odds would
    overshoot below target — only the needed number of odds is dropped
    (evenly spread), so the result lands on exactly SEQUENCE_LEN.
    Always terminates: every pass strictly shrinks a list longer than target.
    """
    target = cfg.SEQUENCE_LEN
    frames = list(frames)
    while len(frames) > target:
        k = len(frames) - target
        last = len(frames) - 1
        # Odd positions, excluding the final frame (index 0 is even and
        # therefore never dropped — both endpoints of the sign are kept).
        odd_pos = [i for i in range(len(frames)) if (i & 1) and i != last]
        if k >= len(odd_pos):
            frames = [f for i, f in enumerate(frames)
                      if not (i & 1) or i == last]
        else:
            drop = set(np.linspace(0, len(odd_pos) - 1, k).round().astype(int))
            drop_idx = {odd_pos[j] for j in drop}
            frames = [f for i, f in enumerate(frames) if i not in drop_idx]
    return frames[:target]


def _next_seq_dir(action):
    """Next free data/<action>/<seq>/ dir (same numbering as collect.py)."""
    action_root = os.path.join(cfg.DATA_DIR, action)
    os.makedirs(action_root, exist_ok=True)
    existing = [d for d in os.listdir(action_root)
                if os.path.isdir(os.path.join(action_root, d)) and d.isdigit()]
    start = max([int(d) for d in existing], default=0) + 1
    seq_dir = os.path.join(action_root, str(start))
    os.makedirs(seq_dir, exist_ok=False)
    return seq_dir


def ingest_video(video_path, action, device="video"):
    """Convert one video into a featurized sequence dir. Returns seq dir path.

    Raises ValueError on unreadable video or too few active frames.
    """
    frames = _read_frames(video_path)
    flags = _hand_flags(frames)
    kept = _downsample_to_12(_trim_static_ends(frames, flags))
    assert len(kept) == cfg.SEQUENCE_LEN, (
        f"expected {cfg.SEQUENCE_LEN} frames, got {len(kept)}")

    calibrator = ArmDepthCalibrator(warmup_frames=cfg.ARM_DEPTH_CAL_WARMUP)
    smoother = ArmDepthSmoother(alpha=cfg.ARM_DEPTH_EMA_ALPHA,
                                window=cfg.ARM_DEPTH_SMOOTH_WINDOW)
    src_shape = None
    with create_model() as holistic:
        vecs = []
        for frame in kept:
            std, meta = normalize_frame(frame)
            if src_shape is None:
                src_shape = [meta["src_h"], meta["src_w"]]
            _, results = detect(std, holistic)
            vecs.append(extract_frame_vector(results, std.shape[:2],
                                             calibrator, smoother))

    seq_dir = _next_seq_dir(action)
    zero_frames = 0
    for i, vec in enumerate(vecs):
        if not np.any(vec):
            zero_frames += 1
        np.save(os.path.join(seq_dir, str(i)), vec)
    with open(os.path.join(seq_dir, "meta.json"), "w") as f:
        json.dump({"device": device,
                   "source": os.path.basename(video_path),
                   "src_shape": src_shape,
                   "std_size": cfg.FRAME_STD_SIZE}, f)
    print(f"  {action}/{os.path.basename(seq_dir)}: {zero_frames}/"
          f"{cfg.SEQUENCE_LEN} zero frames (from {os.path.basename(video_path)}).")
    return seq_dir


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert a video into featurized data (12 .npy frames).")
    parser.add_argument("video", help="input video file")
    parser.add_argument("--action", default="hello")
    parser.add_argument("--device", default="video")
    args = parser.parse_args()
    ingest_video(args.video, args.action, device=args.device)


if __name__ == "__main__":
    main()
