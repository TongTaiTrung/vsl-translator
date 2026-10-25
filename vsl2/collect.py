#!/usr/bin/env python3
"""Collect keypoint sequences from the webcam for a given sign action."""

import os

import argparse
import cv2
import json
import numpy as np
from .landmarks import (
    create_model, detect, draw_landmarks_new, extract, normalize_frame,
)
from . import defaults as cfg


def main():
    parser = argparse.ArgumentParser(description="Collect keypoint sequences from webcam.")
    parser.add_argument(
        "--action",
        type=str,
        default="hello",
        choices=list(cfg.ACTIONS),
        help=f"Action to collect data for (choices: {', '.join(cfg.ACTIONS)}, default: hello)",
    )
    parser.add_argument(
        "--sequences",
        type=int,
        default=cfg.N_SEQUENCES,
        help=f"Number of video sequences to capture (default: {cfg.N_SEQUENCES})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="macbook",
        help="Camera source label saved to meta.json per sequence (e.g. macbook, iphone)",
    )
    args = parser.parse_args()

    action = args.action
    total_sequences = args.sequences
    device = args.device

    print(f"Collecting data for action: '{action}' ({total_sequences} sequences of {cfg.SEQUENCE_LEN} frames)")
    action_root = os.path.join(cfg.DATA_DIR, action)
    os.makedirs(action_root, exist_ok=True)
    existing = [d for d in os.listdir(action_root)
                if os.path.isdir(os.path.join(action_root, d)) and d.isdigit()]
    start_idx = max([int(d) for d in existing], default=0) + 1
    if existing:
        print(f"  Found {len(existing)} existing sequences — new ones start at #{start_idx} (no overwrite).")
    cap = cv2.VideoCapture(0)
    with create_model() as model:
        for seq_no in range(start_idx, start_idx + total_sequences):
            cnt = 0
            zero_frames = 0
            src_shape = None
            while cnt < cfg.SEQUENCE_LEN:
                ret, frame = cap.read()
                if not ret:
                    break
                frame, meta = normalize_frame(frame)
                if src_shape is None:
                    src_shape = [meta["src_h"], meta["src_w"]]
                img, results = detect(frame, model)
                img = draw_landmarks_new(img, results)
                status_text = f"Action: {action} | Seq {seq_no}/{start_idx + total_sequences - 1}"
                if cnt == 0:
                    cv2.putText(
                        img,
                        f"STARTING {status_text}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("OpenCV Feed", img)
                    cv2.waitKey(2000 if seq_no == start_idx else 1000)
                else:
                    cv2.putText(
                        img,
                        f"{status_text} | Frame {cnt + 1}/{cfg.SEQUENCE_LEN}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("OpenCV Feed", img)

                keypoints = extract(results, frame_shape=frame.shape[:2])
                if not np.any(keypoints):
                    zero_frames += 1
                dir_path = os.path.join(cfg.DATA_DIR, action, str(seq_no))
                os.makedirs(dir_path, exist_ok=True)
                np.save(os.path.join(dir_path, str(cnt)), keypoints)
                cnt += 1

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            print(f"  Seq {seq_no}: {zero_frames}/{cfg.SEQUENCE_LEN} zero frames (missing/untrusted landmarks).")
            dir_path = os.path.join(cfg.DATA_DIR, action, str(seq_no))
            with open(os.path.join(dir_path, "meta.json"), "w") as f:
                json.dump({"device": device, "src_shape": src_shape,
                           "std_size": cfg.FRAME_STD_SIZE}, f)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished collecting {total_sequences} sequences for '{action}'.")


if __name__ == '__main__':
    main()
