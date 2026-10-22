#!/usr/bin/env python3
"""Inspect a video dataset: print frame counts, durations, and labels."""

import os
import time
import cv2
import csv

folder_path = "/Users/trungtran/Downloads/archive/Dataset/Videos"
label_path = "/Users/trungtran/Downloads/archive/Dataset/Labels/label.csv"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
output_file = "results.txt"

labels = {}
if os.path.exists(label_path):
    with open(label_path, mode="r", encoding="utf-8") as f_csv:
        reader = csv.reader(f_csv)
        next(reader, None)  # Bỏ qua dòng tiêu đề (ID, VIDEO, LABEL)
        for row in reader:
            if len(row) >= 3:
                video_name = row[1].strip()
                label_val = row[2].strip()
                labels[video_name] = label_val

start_time = time.time()
processed_count = 0

with open(output_file, "w") as f:
    for root, _, files in os.walk(folder_path):
        for filename in sorted(files):
            # 1. Skip hidden files and non-video formats
            ext = os.path.splitext(filename)[1].lower()
            if ext not in VIDEO_EXTENSIONS:
                continue

            filepath = os.path.join(root, filename)
            cap = cv2.VideoCapture(filepath)

            # 2. Verify video opened properly
            if not cap.isOpened():
                msg = f"Skipping (cannot open): {filename}"
                print(msg)
                f.write(msg + "\n")
                cap.release()
                continue

            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            label = labels.get(filename, labels.get(os.path.basename(filepath), "N/A"))

            # 3. Guard against invalid or 0/negative fps/frame values
            if fps and fps > 0 and frames and frames > 0:
                duration = frames / fps
                msg = f"{filename} [{label}]: {int(frames)} frames, {duration:.2f}s ({fps:.2f} fps)"
            else:
                msg = f"{filename} [{label}]: Unable to read valid frame metadata"

            print(msg)
            f.write(msg + "\n")
            processed_count += 1

    summary_msg = f"\nProcessed {processed_count} videos in {time.time() - start_time:.2f} seconds."
    print(summary_msg)
    f.write(summary_msg + "\n")
