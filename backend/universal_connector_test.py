#!/usr/bin/env python3
"""Web application testbed for UniversalConnector.

Captures live video from camera (or video file), automatically buffers ("charges")
N frames into a sliding sequence, feeds the sequence into UniversalConnector.predict(),
and displays live video, landmark overlays, buffer fill status, and prediction results
in an interactive web interface.

Usage:
    ./venv/bin/python universal_connector_test.py
    ./venv/bin/python universal_connector_test.py --camera 0 --port 8080
    ./venv/bin/python universal_connector_test.py --video sample.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

# Ensure project root and src/ are in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config as cfg
from universal_connector import PredictionResult, UniversalConnector


class VideoFeedManager:
    """Manages camera frame acquisition, frame charging buffer, and inference."""

    def __init__(
        self,
        source: Any = 0,
        confidence_threshold: float = cfg.CONFIDENCE_THRESHOLD,
        stride: int = 3,
        overlay_landmarks: bool = True,
    ):
        self.source = source
        self.confidence_threshold = confidence_threshold
        self.stride = stride
        self.overlay_landmarks = overlay_landmarks

        print("Initializing UniversalConnector...")
        self.connector = UniversalConnector(
            confidence_threshold=self.confidence_threshold,
            require_hand=True,
            resample=True,
        )
        self.sequence_len = self.connector.sequence_len
        print(f"UniversalConnector ready (sequence_len={self.sequence_len}, threshold={self.confidence_threshold:.2f}).")

        # Camera & State
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.lock = threading.Lock()

        # Frame buffers
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_display_frame: Optional[np.ndarray] = None
        self.charge_buffer: Deque[np.ndarray] = deque(maxlen=self.sequence_len)

        # Performance & Stats
        self.camera_fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.frame_interval_count = 0

        # Prediction state
        self.latest_result: Optional[PredictionResult] = None
        self.history: List[Dict[str, Any]] = []
        self.last_emitted_action: Optional[str] = None
        self.last_emitted_time: float = 0.0

        # Control flags
        self.is_paused = False
        self.is_predicting = False

        # Threads
        self.capture_thread: Optional[threading.Thread] = None
        self.predict_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start camera capture and inference background workers."""
        self.is_running = True
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"WARNING: Could not open video source {self.source}. Running in fallback dummy mode.")

        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.predict_thread = threading.Thread(target=self._predict_worker, daemon=True)
        self.capture_thread.start()
        self.predict_thread.start()

    def stop(self) -> None:
        """Stop all workers and release camera resources."""
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.connector.close()

    def _generate_fallback_frame(self, message: str = "Camera Not Available") -> np.ndarray:
        """Generate an informative dummy frame when camera is unavailable or disconnected."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (620, 460), (30, 30, 40), -1)
        cv2.rectangle(img, (20, 20), (620, 460), (70, 70, 90), 2)
        cv2.putText(img, "VSL Universal Connector", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(img, message, (50, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "Please check camera permissions in System Settings", (50, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(img, "or pass a video file via --video <file>", (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        t_str = time.strftime("%H:%M:%S")
        cv2.putText(img, f"Time: {t_str}", (50, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 255, 140), 1, cv2.LINE_AA)
        return img

    def _capture_worker(self) -> None:
        """Capture frames from OpenCV and push to charging buffer."""
        frames_in_interval = 0
        start_interval = time.time()

        while self.is_running:
            if self.is_paused:
                time.sleep(0.05)
                continue

            frame = None
            if self.cap and self.cap.isOpened():
                ret, captured = self.cap.read()
                if ret and captured is not None:
                    frame = captured
                elif isinstance(self.source, str):
                    # Loop video file if reached end
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, captured = self.cap.read()
                    if ret:
                        frame = captured

            if frame is None:
                frame = self._generate_fallback_frame()
                time.sleep(0.033)  # ~30 FPS dummy cadence

            # Update FPS calculation
            frames_in_interval += 1
            now = time.time()
            dt = now - start_interval
            if dt >= 1.0:
                self.camera_fps = round(frames_in_interval / dt, 1)
                frames_in_interval = 0
                start_interval = now

            with self.lock:
                self.latest_raw_frame = frame
                # Charge frame into sequence buffer
                self.charge_buffer.append(frame.copy())
                self.frame_count += 1

                # If no overlay is requested or before first inference, use raw frame
                if not self.overlay_landmarks or self.latest_display_frame is None:
                    self.latest_display_frame = frame.copy()

            time.sleep(0.005)

    def _predict_worker(self) -> None:
        """Charge frames and call connector.predict() on buffered sequences."""
        last_processed_frame = 0

        while self.is_running:
            if self.is_paused:
                time.sleep(0.05)
                continue

            frames_to_predict = None
            with self.lock:
                buffer_len = len(self.charge_buffer)
                current_frame = self.frame_count

                # Only predict when buffer is fully charged AND we've advanced by 'stride' frames
                if buffer_len == self.sequence_len and (current_frame - last_processed_frame >= self.stride):
                    frames_to_predict = list(self.charge_buffer)
                    last_processed_frame = current_frame
                    self.is_predicting = True

            if frames_to_predict is not None:
                try:
                    # Call UniversalConnector.predict() with full sequence of OpenCV frames
                    result = self.connector.predict(
                        frames_to_predict,
                        return_annotated=self.overlay_landmarks,
                    )

                    with self.lock:
                        self.latest_result = result
                        self.is_predicting = False

                        # If annotated frames were generated, update latest display frame with last landmark overlay
                        if self.overlay_landmarks and result.annotated_frames and len(result.annotated_frames) > 0:
                            self.latest_display_frame = result.annotated_frames[-1]

                        # Add confirmed gesture to history (with debounce)
                        now = time.time()
                        if result.is_valid and result.action:
                            is_new_action = (result.action != self.last_emitted_action)
                            time_since_last = now - self.last_emitted_time
                            if is_new_action or (time_since_last >= 2.0):
                                self.history.insert(0, {
                                    "action": result.action,
                                    "confidence": round(result.confidence, 3),
                                    "time": time.strftime("%H:%M:%S"),
                                    "timestamp": now,
                                })
                                if len(self.history) > 30:
                                    self.history.pop()
                                self.last_emitted_action = result.action
                                self.last_emitted_time = now

                except Exception as e:
                    print(f"Error during connector.predict(): {e}")
                    with self.lock:
                        self.is_predicting = False
            else:
                time.sleep(0.01)

    def get_display_frame_jpeg(self) -> bytes:
        """Encode latest frame with overlay HUD as JPEG."""
        with self.lock:
            if self.latest_display_frame is not None:
                img = self.latest_display_frame.copy()
            else:
                img = self._generate_fallback_frame("Loading Stream...")

            res = self.latest_result
            buf_count = len(self.charge_buffer)
            seq_len = self.sequence_len
            fps = self.camera_fps
            is_pred = self.is_predicting

        # Draw real-time HUD on image
        h, w = img.shape[:2]

        # Top banner: Buffer status & FPS
        buf_ratio = min(1.0, buf_count / float(seq_len))
        buf_color = (0, 220, 0) if buf_count == seq_len else (0, 165, 255)
        status_tag = "PREDICTING" if is_pred else ("CHARGED" if buf_count == seq_len else "CHARGING")

        cv2.rectangle(img, (0, 0), (w, 36), (20, 20, 24), -1)
        cv2.putText(
            img,
            f"BUFFER: {buf_count}/{seq_len} [{status_tag}] | FPS: {fps}",
            (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            buf_color,
            1,
            cv2.LINE_AA,
        )

        # Buffer progress mini bar
        bar_w = 120
        cv2.rectangle(img, (w - bar_w - 15, 12), (w - 15, 24), (50, 50, 50), -1)
        fill_w = int(buf_ratio * bar_w)
        if fill_w > 0:
            cv2.rectangle(img, (w - bar_w - 15, 12), (w - bar_w - 15 + fill_w, 24), buf_color, -1)

        # Bottom banner: Latest prediction
        if res is not None:
            if res.is_valid and res.action:
                txt = f"ACTION: {res.action.upper()} ({res.confidence:.1%})"
                color = (0, 255, 100)
            elif not res.has_hand:
                txt = "STATUS: Nghi / Chua phat hien tay"
                color = (180, 180, 180)
            else:
                txt = f"STATUS: Dang nhan dien... (Top: {res.top_action} {res.confidence:.0%})"
                color = (0, 200, 255)

            cv2.rectangle(img, (0, h - 36), (w, h), (20, 20, 24), -1)
            cv2.putText(img, txt, (14, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        ret, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes() if ret else b""

    def get_status_json(self) -> Dict[str, Any]:
        """Compile complete system status dictionary for web UI."""
        with self.lock:
            buf_count = len(self.charge_buffer)
            seq_len = self.sequence_len
            buf_pct = round((buf_count / float(seq_len)) * 100.0, 1)
            pred_data = self.latest_result.to_dict() if self.latest_result else None
            hist = list(self.history)
            fps = self.camera_fps
            is_pred = self.is_predicting
            paused = self.is_paused
            overlay = self.overlay_landmarks
            thresh = self.confidence_threshold
            actions = list(self.connector.actions)

        return {
            "fps": fps,
            "buffer_count": buf_count,
            "sequence_len": seq_len,
            "buffer_percent": buf_pct,
            "is_predicting": is_pred,
            "is_paused": paused,
            "overlay_landmarks": overlay,
            "confidence_threshold": thresh,
            "actions": actions,
            "prediction": pred_data,
            "history": hist,
        }

    def set_settings(self, overlay: Optional[bool] = None, threshold: Optional[float] = None) -> None:
        """Update runtime settings."""
        with self.lock:
            if overlay is not None:
                self.overlay_landmarks = bool(overlay)
            if threshold is not None:
                self.confidence_threshold = max(0.1, min(0.99, float(threshold)))
                self.connector.confidence_threshold = self.confidence_threshold

    def reset_buffer_and_history(self) -> None:
        """Clear charging buffer and history."""
        with self.lock:
            self.charge_buffer.clear()
            self.history.clear()
            self.latest_result = None
            self.last_emitted_action = None


# ── Web Application Frontend HTML/CSS/JS ─────────────────────────────────────

INDEX_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VSL Universal Connector - WebApp Testbed</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f17;
      --card-bg: rgba(22, 28, 42, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.35);
      --success: #10b981;
      --success-glow: rgba(16, 185, 129, 0.35);
      --warning: #f59e0b;
      --danger: #ef4444;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
      background-image: radial-gradient(circle at 10% 20%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
                        radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.06) 0%, transparent 40%);
    }

    header {
      padding: 16px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--card-border);
      backdrop-filter: blur(12px);
      background: rgba(11, 15, 23, 0.8);
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-logo {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: linear-gradient(135deg, #0284c7, #10b981);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 18px;
      box-shadow: 0 0 16px var(--accent-glow);
    }
    .brand-text h1 {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .brand-text p {
      font-size: 12px;
      color: var(--text-muted);
    }
    .header-badges {
      display: flex;
      gap: 12px;
    }
    .badge {
      font-size: 12px;
      font-weight: 600;
      padding: 6px 12px;
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--card-border);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .badge-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 8px var(--success);
      animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

    main {
      flex: 1;
      padding: 24px;
      max-width: 1440px;
      margin: 0 auto;
      width: 100%;
      display: grid;
      grid-template-columns: 1fr 420px;
      gap: 24px;
    }
    @media (max-width: 1024px) {
      main { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 20px;
      backdrop-filter: blur(12px);
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.25);
    }

    /* Video feed section */
    .video-section {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .video-container {
      position: relative;
      width: 100%;
      border-radius: 14px;
      overflow: hidden;
      background: #000;
      border: 1px solid var(--card-border);
      aspect-ratio: 4 / 3;
      max-height: 560px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .video-container img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .hud-overlay {
      position: absolute;
      top: 14px;
      left: 14px;
      right: 14px;
      display: flex;
      justify-content: space-between;
      pointer-events: none;
    }
    .hud-chip {
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(8px);
      padding: 6px 14px;
      border-radius: 30px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Frame charge meter */
    .charge-card {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .charge-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
    }
    .charge-title {
      font-weight: 600;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .charge-val {
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      color: var(--accent);
    }
    .progress-bar-bg {
      width: 100%;
      height: 10px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 6px;
      overflow: hidden;
      position: relative;
    }
    .progress-bar-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #0284c7, #38bdf8);
      border-radius: 6px;
      transition: width 0.15s ease, background 0.3s;
      box-shadow: 0 0 10px var(--accent-glow);
    }
    .progress-bar-fill.charged {
      background: linear-gradient(90deg, #059669, #10b981);
      box-shadow: 0 0 10px var(--success-glow);
    }

    /* Side panels */
    .side-panels {
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    /* Current prediction box */
    .pred-box {
      text-align: center;
      padding: 24px 20px;
      border: 1px solid var(--card-border);
      background: linear-gradient(180deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.7) 100%);
      border-radius: 16px;
      position: relative;
      overflow: hidden;
    }
    .pred-status {
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
      margin-bottom: 8px;
    }
    .pred-action {
      font-size: 38px;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: #fff;
      margin-bottom: 6px;
      text-shadow: 0 0 20px rgba(255, 255, 255, 0.15);
      min-height: 48px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .pred-action.valid {
      color: #34d399;
      text-shadow: 0 0 25px rgba(52, 211, 153, 0.4);
    }
    .pred-action.uncertain {
      color: #fbbf24;
      font-size: 26px;
    }
    .pred-confidence {
      font-family: 'JetBrains Mono', monospace;
      font-size: 14px;
      color: var(--text-muted);
    }
    .pred-badges {
      display: flex;
      justify-content: center;
      gap: 10px;
      margin-top: 14px;
    }
    .tag {
      font-size: 11px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.06);
    }
    .tag-valid { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .tag-hand { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }

    /* Probability bars */
    .bars-section h3 {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 14px;
      display: flex;
      justify-content: space-between;
    }
    .prob-row {
      margin-bottom: 12px;
    }
    .prob-info {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      margin-bottom: 4px;
    }
    .prob-name { font-weight: 600; }
    .prob-val { font-family: 'JetBrains Mono', monospace; color: var(--text-muted); font-size: 12px; }
    .prob-track {
      width: 100%;
      height: 8px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 4px;
      overflow: hidden;
    }
    .prob-fill {
      height: 100%;
      width: 0%;
      background: var(--accent);
      border-radius: 4px;
      transition: width 0.2s ease, background 0.2s;
    }
    .prob-fill.winner {
      background: linear-gradient(90deg, #10b981, #34d399);
      box-shadow: 0 0 10px var(--success-glow);
    }

    /* History Timeline */
    .history-section h3 {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 12px;
    }
    .history-list {
      max-height: 180px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding-right: 4px;
    }
    .history-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 12px;
      background: rgba(255, 255, 255, 0.04);
      border-radius: 8px;
      border: 1px solid var(--card-border);
      font-size: 13px;
      animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }
    .hist-action { font-weight: 700; color: #34d399; }
    .hist-meta { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); }

    /* Controls */
    .controls-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 10px;
    }
    .control-item {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .control-item label {
      font-size: 12px;
      color: var(--text-muted);
      font-weight: 500;
    }
    .slider-row {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    input[type="range"] {
      flex: 1;
      accent-color: var(--accent);
      cursor: pointer;
    }
    .btn {
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid var(--card-border);
      background: rgba(255, 255, 255, 0.06);
      color: var(--text);
      transition: all 0.2s;
    }
    .btn:hover {
      background: rgba(255, 255, 255, 0.12);
      border-color: rgba(255, 255, 255, 0.2);
    }
    .btn-danger { color: #f87171; }
    .btn-danger:hover { background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.3); }

    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 3px; }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-logo">V</div>
      <div class="brand-text">
        <h1>VSL Universal Connector</h1>
        <p>Real-Time Frame-Charging Testbed</p>
      </div>
    </div>
    <div class="header-badges">
      <div class="badge">
        <div class="badge-dot"></div>
        <span id="fps-badge">FPS: --</span>
      </div>
      <div class="badge">
        <span id="latency-badge">Latency: -- ms</span>
      </div>
    </div>
  </header>

  <main>
    <!-- Left Column: Video stream + Buffer status -->
    <div class="video-section">
      <div class="video-container">
        <img id="live-stream" src="/video_feed" alt="Video Feed">
        <div class="hud-overlay">
          <div class="hud-chip" id="hud-status">● Live Stream</div>
          <div class="hud-chip" id="hud-buffer">Buffer: 0/24</div>
        </div>
      </div>

      <!-- Frame Charging Progress -->
      <div class="card charge-card">
        <div class="charge-header">
          <div class="charge-title">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
            </svg>
            Auto Frame Charger (OpenCV → Connector)
          </div>
          <div class="charge-val" id="charge-text">0 / 24 frames (0%)</div>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" id="charge-fill"></div>
        </div>
      </div>

      <!-- Controls Card -->
      <div class="card">
        <h3 style="font-size: 14px; font-weight: 600; color: var(--text-muted); margin-bottom: 12px;">Controls & Settings</h3>
        <div class="controls-grid">
          <div class="control-item">
            <label for="thresh-slider">Confidence Threshold: <span id="thresh-val" style="color: var(--accent); font-weight:700;">70%</span></label>
            <div class="slider-row">
              <input type="range" id="thresh-slider" min="30" max="95" value="70">
            </div>
          </div>
          <div class="control-item" style="justify-content: flex-end;">
            <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
              <input type="checkbox" id="overlay-toggle" checked style="accent-color: var(--accent);">
              Overlay MediaPipe Landmarks
            </label>
          </div>
        </div>
        <div style="display: flex; gap: 10px; margin-top: 14px;">
          <button class="btn" id="btn-pause">Pause / Resume</button>
          <button class="btn btn-danger" id="btn-reset">Reset Buffer & History</button>
        </div>
      </div>
    </div>

    <!-- Right Column: Prediction + Probability bars + History -->
    <div class="side-panels">

      <!-- Current Prediction Result -->
      <div class="pred-box">
        <div class="pred-status" id="pred-state-label">Waiting for frames...</div>
        <div class="pred-action" id="pred-action-display">--</div>
        <div class="pred-confidence" id="pred-conf-display">Confidence: --</div>
        <div class="pred-badges">
          <span class="tag" id="tag-valid">Not Verified</span>
          <span class="tag" id="tag-hand">No Hands</span>
        </div>
      </div>

      <!-- Class Probability Distribution -->
      <div class="card bars-section">
        <h3>
          <span>Class Probabilities</span>
          <span style="font-size: 11px; font-family:'JetBrains Mono', monospace;" id="pred-top-tag"></span>
        </h3>
        <div id="prob-bars-container">
          <!-- Dynamically populated -->
        </div>
      </div>

      <!-- Confirmed History -->
      <div class="card history-section">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
          <h3 style="margin: 0;">Gesture Timeline</h3>
          <span style="font-size: 11px; color: var(--text-muted);" id="history-count">0 items</span>
        </div>
        <div class="history-list" id="history-container">
          <div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 20px;">
            Chưa có ký hiệu được xác nhận...
          </div>
        </div>
      </div>

    </div>
  </main>

  <script>
    let isPaused = false;

    // Polling API for real-time status
    async function updateStatus() {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const data = await res.json();

        // 1. FPS & Latency
        document.getElementById('fps-badge').textContent = `FPS: ${data.fps}`;
        if (data.prediction && data.prediction.latency_ms) {
          document.getElementById('latency-badge').textContent = `Latency: ${data.prediction.latency_ms} ms`;
        }

        // 2. Buffer Charging progress
        const count = data.buffer_count;
        const total = data.sequence_len;
        const pct = data.buffer_percent;
        document.getElementById('charge-text').textContent = `${count} / ${total} frames (${pct}%)`;
        document.getElementById('hud-buffer').textContent = `Buffer: ${count}/${total}`;

        const fill = document.getElementById('charge-fill');
        fill.style.width = `${pct}%`;
        if (count === total) {
          fill.classList.add('charged');
        } else {
          fill.classList.remove('charged');
        }

        // 3. Current Prediction
        const pAction = document.getElementById('pred-action-display');
        const pConf = document.getElementById('pred-conf-display');
        const pState = document.getElementById('pred-state-label');
        const tagValid = document.getElementById('tag-valid');
        const tagHand = document.getElementById('tag-hand');

        if (data.prediction) {
          const pred = data.prediction;
          pConf.textContent = `Confidence: ${(pred.confidence * 100).toFixed(1)}%`;

          if (pred.is_valid && pred.action) {
            pAction.textContent = pred.action.toUpperCase();
            pAction.className = 'pred-action valid';
            pState.textContent = 'RECOGNIZED SIGN';
            tagValid.className = 'tag tag-valid';
            tagValid.textContent = 'VALID SIGN';
          } else if (!pred.has_hand) {
            pAction.textContent = 'REST / IDLE';
            pAction.className = 'pred-action uncertain';
            pState.textContent = 'CHƯA PHÁT HIỆN TAY';
            tagValid.className = 'tag';
            tagValid.textContent = 'NO HAND';
          } else {
            pAction.textContent = pred.top_action.toUpperCase();
            pAction.className = 'pred-action uncertain';
            pState.textContent = 'DƯỚI NGƯỠNG TIN CẬY';
            tagValid.className = 'tag';
            tagValid.textContent = 'UNCERTAIN';
          }

          if (pred.has_hand) {
            tagHand.className = 'tag tag-hand';
            tagHand.textContent = 'HAND DETECTED';
          } else {
            tagHand.className = 'tag';
            tagHand.textContent = 'NO HAND';
          }

          // 4. Probability Bars
          renderProbBars(pred.probabilities, pred.top_action, pred.is_valid);
        }

        // 5. History Timeline
        renderHistory(data.history);

      } catch (err) {
        console.error('Update status error:', err);
      }
    }

    function renderProbBars(probs, topAction, isValid) {
      const container = document.getElementById('prob-bars-container');
      if (!probs) return;

      const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
      let html = '';

      for (const [action, prob] of entries) {
        const pct = (prob * 100).toFixed(1);
        const isWinner = (action === topAction && isValid);
        const barClass = isWinner ? 'prob-fill winner' : 'prob-fill';

        html += `
          <div class="prob-row">
            <div class="prob-info">
              <span class="prob-name" style="${isWinner ? 'color: #34d399;' : ''}">${action}</span>
              <span class="prob-val">${pct}%</span>
            </div>
            <div class="prob-track">
              <div class="${barClass}" style="width: ${pct}%;"></div>
            </div>
          </div>
        `;
      }
      container.innerHTML = html;
    }

    function renderHistory(history) {
      const container = document.getElementById('history-container');
      const countEl = document.getElementById('history-count');
      if (!history || history.length === 0) {
        container.innerHTML = '<div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 20px;">Chưa có ký hiệu được xác nhận...</div>';
        countEl.textContent = '0 items';
        return;
      }

      countEl.textContent = `${history.length} items`;
      let html = '';
      for (const item of history) {
        html += `
          <div class="history-item">
            <span class="hist-action">${item.action.toUpperCase()}</span>
            <span class="hist-meta">${(item.confidence * 100).toFixed(0)}% · ${item.time}</span>
          </div>
        `;
      }
      container.innerHTML = html;
    }

    // Controls listeners
    const slider = document.getElementById('thresh-slider');
    const threshVal = document.getElementById('thresh-val');
    slider.addEventListener('input', async (e) => {
      const val = parseFloat(e.target.value) / 100.0;
      threshVal.textContent = `${e.target.value}%`;
      await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ threshold: val })
      });
    });

    const overlayToggle = document.getElementById('overlay-toggle');
    overlayToggle.addEventListener('change', async (e) => {
      await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ overlay: e.target.checked })
      });
    });

    document.getElementById('btn-pause').addEventListener('click', async () => {
      isPaused = !isPaused;
      await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pause: isPaused })
      });
    });

    document.getElementById('btn-reset').addEventListener('click', async () => {
      await fetch('/api/reset', { method: 'POST' });
      updateStatus();
    });

    // Loop polling every 120ms
    setInterval(updateStatus, 120);
    updateStatus();
  </script>
</body>
</html>
"""


# ── HTTP Server Request Handler ──────────────────────────────────────────────

class WebAppHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP Request Handler serving video feed and REST endpoints."""

    manager: VideoFeedManager  # Set on server initialization

    def do_GET(self) -> None:
        url = urlparse(self.path)

        if url.path == "/" or url.path == "/index.html":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))

        elif url.path == "/video_feed":
            # Stream MJPEG video
            self.send_response(HTTPStatus.OK)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            try:
                while self.manager.is_running:
                    jpeg_bytes = self.manager.get_display_frame_jpeg()
                    if jpeg_bytes:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpeg_bytes)))
                        self.end_headers()
                        self.wfile.write(jpeg_bytes)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.033)  # ~30 FPS stream cap
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif url.path == "/api/status":
            status = self.manager.get_status_json()
            payload = json.dumps(status).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)

        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Path Not Found")

    def do_POST(self) -> None:
        url = urlparse(self.path)

        if url.path == "/api/settings":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            try:
                data = json.loads(body)
                if "overlay" in data:
                    self.manager.set_settings(overlay=data["overlay"])
                if "threshold" in data:
                    self.manager.set_settings(threshold=data["threshold"])
                if "pause" in data:
                    self.manager.is_paused = bool(data["pause"])
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            except Exception as e:
                self.send_error(HTTPStatus.BAD_REQUEST, str(e))

        elif url.path == "/api/reset":
            self.manager.reset_buffer_and_history()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Path Not Found")

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress routine GET log spam to keep terminal clean
        if "/api/status" not in args[0] and "/video_feed" not in args[0]:
            super().log_message(format, *args)


def run_webapp(
    source: Any = 0,
    port: int = 8080,
    host: str = "0.0.0.0",
    threshold: float = cfg.CONFIDENCE_THRESHOLD,
    stride: int = 3,
) -> None:
    """Initialize manager, start web server and listen for connections."""
    manager = VideoFeedManager(
        source=source,
        confidence_threshold=threshold,
        stride=stride,
        overlay_landmarks=True,
    )
    manager.start()

    # Pass manager instance to handler class
    WebAppHandler.manager = manager

    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, WebAppHandler)

    display_host = "localhost" if host == "0.0.0.0" else host
    print("\n" + "=" * 64)
    print(" VSL UNIVERSAL CONNECTOR WEBAPP READY")
    print("=" * 64)
    print(f" ▶ Open in browser: http://{display_host}:{port}")
    print(f" ▶ Camera source  : {source}")
    print(f" ▶ Sequence len   : {manager.sequence_len} frames")
    print(f" ▶ Stride         : {stride} frames")
    print(f" ▶ Actions        : {list(manager.connector.actions)}")
    print(" Press Ctrl+C in terminal to stop.")
    print("=" * 64 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web application...")
    finally:
        httpd.server_close()
        manager.stop()
        print("Application stopped cleanly.")


def main():
    parser = argparse.ArgumentParser(
        description="VSL Universal Connector WebApp Testbed (Auto Frame Charge & Predict)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional path to a video file instead of live camera.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Web server port (default: 8080).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Web server host address (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=cfg.CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {cfg.CONFIDENCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Frame step interval between consecutive sequence predictions (default: 3).",
    )
    args = parser.parse_args()

    source = args.video if args.video else args.camera
    run_webapp(
        source=source,
        port=args.port,
        host=args.host,
        threshold=args.threshold,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
