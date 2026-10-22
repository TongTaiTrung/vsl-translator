"""Universal Connector for VSL (Vietnamese Sign Language) Translator.

Provides a clean, standardized, high-performance interface for third-party
applications (web services, desktop applications, streaming pipelines, robotics)
to perform sign language inference.

Input:
    - Sequence of N OpenCV frames (BGR numpy arrays, e.g. from cv2.VideoCapture,
      video files, or camera streams).
Output:
    - PredictionResult object containing the predicted action, confidence score,
      per-class probability distribution, validation flags, and latency stats.
"""

from __future__ import annotations

import os
import sys
import time
import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

# Ensure project root and src/ are in sys.path even when imported from elsewhere
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config as cfg
from vsl_translator.landmarks import (
    create_model,
    detect,
    draw_landmarks_new,
    extract,
)
from vsl_translator.dataset import standardize_features


@dataclass
class PredictionResult:
    """Standardized prediction result for third-party consumption.

    Supports both attribute access (result.action) and dictionary access
    (result["action"]) for maximum compatibility.
    """

    action: Optional[str]
    confidence: float
    probabilities: Dict[str, float]
    is_valid: bool
    has_hand: bool
    top_action: str
    raw_probabilities: np.ndarray = field(repr=False)
    latency_ms: float = 0.0
    num_frames_received: int = 0
    num_frames_processed: int = 0
    hand_frame_ratio: float = 0.0
    annotated_frames: Optional[List[np.ndarray]] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a JSON-serializable dictionary."""
        return {
            "action": self.action,
            "confidence": float(self.confidence),
            "top_action": self.top_action,
            "is_valid": bool(self.is_valid),
            "has_hand": bool(self.has_hand),
            "probabilities": {k: float(v) for k, v in self.probabilities.items()},
            "raw_probabilities": [float(p) for p in self.raw_probabilities],
            "latency_ms": round(float(self.latency_ms), 2),
            "num_frames_received": int(self.num_frames_received),
            "num_frames_processed": int(self.num_frames_processed),
            "hand_frame_ratio": round(float(self.hand_frame_ratio), 4),
        }

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self):
        return self.to_dict().keys()

    def items(self):
        return self.to_dict().items()

    def values(self):
        return self.to_dict().values()

    def __str__(self) -> str:
        status = self.action if self.is_valid else f"Uncertain (top: {self.top_action})"
        return (
            f"PredictionResult(action={status}, confidence={self.confidence:.2%}, "
            f"has_hand={self.has_hand}, latency={self.latency_ms:.1f}ms)"
        )


class UniversalConnector:
    """Universal Connector wrapping MediaPipe extraction and the trained Keras model.

    Designed for reuse across multiple inference calls with minimal overhead.

    Args:
        model_path: Path to trained Keras model (.keras). Defaults to models/model.keras
                    or models/best_model/model.keras.
        stats_path: Path to feature standardization stats (.npz). Defaults to
                    models/feature_stats.npz.
        actions: List or array of action labels. Defaults to config.ACTIONS.
        confidence_threshold: Minimum confidence score [0.0 - 1.0] to accept a prediction.
                              Defaults to config.CONFIDENCE_THRESHOLD (0.70).
        sequence_len: Target sequence length in frames. Defaults to model input shape
                      or config.SEQUENCE_LEN (24).
        require_hand: If True, sequences without hand detections are marked invalid.
        min_hand_frames: Minimum number of frames with hand landmarks required when
                         require_hand is True. Defaults to 2.
        resample: If True, input sequences with frame counts != sequence_len are
                  smoothly resampled uniformly across time. Defaults to True.
        device: TensorFlow device specification (optional).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        stats_path: Optional[str] = None,
        actions: Optional[Union[Sequence[str], np.ndarray]] = None,
        confidence_threshold: Optional[float] = None,
        sequence_len: Optional[int] = None,
        require_hand: bool = True,
        min_hand_frames: int = 2,
        resample: bool = True,
    ):
        self.require_hand = require_hand
        self.min_hand_frames = min_hand_frames
        self.resample = resample

        # ── 1. Confidence threshold ──────────────────────────────────────────
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else getattr(cfg, "CONFIDENCE_THRESHOLD", 0.70)
        )

        # ── 2. Actions / Labels ──────────────────────────────────────────────
        if actions is not None:
            self.actions = np.array(actions)
        else:
            self.actions = np.array(getattr(cfg, "ACTIONS", []))

        # ── 3. Model path resolution ─────────────────────────────────────────
        if model_path is None:
            candidates = [
                os.path.join(cfg.MODELS_DIR, cfg.MODEL_FILENAME),
                os.path.join(cfg.MODELS_DIR, "best_model", cfg.MODEL_FILENAME),
                os.path.join(cfg.MODELS_DIR, "best_model", "model.keras"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break
            if model_path is None:
                model_path = os.path.join(cfg.MODELS_DIR, cfg.MODEL_FILENAME)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                "Please train the model first using scripts/train.py."
            )

        # Lazy load keras to avoid unnecessary startup time on simple imports
        from keras.models import load_model

        self.model = load_model(model_path)
        self.model_path = model_path

        # Determine expected sequence length and feature dimension from model
        expected_seq_len = self.model.input_shape[1]
        self.expected_dim = self.model.input_shape[2]
        self.sequence_len = sequence_len if sequence_len is not None else (
            expected_seq_len if expected_seq_len is not None else getattr(cfg, "SEQUENCE_LEN", 24)
        )

        # Check action count vs model output shape
        num_classes = self.model.output_shape[-1]
        if len(self.actions) != num_classes:
            warnings.warn(
                f"Actions count ({len(self.actions)}) does not match model output dim ({num_classes}). "
                "Falling back to generic action indices."
            )
            self.actions = np.array([f"class_{i}" for i in range(num_classes)])

        # ── 4. Feature stats ─────────────────────────────────────────────────
        if stats_path is None:
            candidates_stats = [
                os.path.join(cfg.MODELS_DIR, "feature_stats.npz"),
                os.path.join(cfg.MODELS_DIR, "best_model", "feature_stats.npz"),
            ]
            for cs in candidates_stats:
                if os.path.exists(cs):
                    stats_path = cs
                    break

        if stats_path and os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.feat_mean = stats["mean"]
            self.feat_std = stats["std"]
            if self.feat_mean.shape[-1] != self.expected_dim:
                warnings.warn(
                    f"Feature stats dimension {self.feat_mean.shape[-1]} != model dimension {self.expected_dim}."
                )
        else:
            self.feat_mean = None
            self.feat_std = None
            warnings.warn(
                "feature_stats.npz not found – running without feature standardization."
            )

        # ── 5. MediaPipe Holistic model instance ─────────────────────────────
        self.holistic = create_model()

    def __enter__(self) -> UniversalConnector:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "holistic") and self.holistic is not None:
            try:
                self.holistic.close()
            except Exception:
                pass
            self.holistic = None

    def warmup(self) -> None:
        """Warm up MediaPipe and Keras model with dummy data to eliminate first-call jitter."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_seq = [dummy_frame for _ in range(self.sequence_len)]
        _ = self.predict(dummy_seq)

    def _resample_frames(
        self, frames: Sequence[np.ndarray], target_len: int
    ) -> List[np.ndarray]:
        """Uniformly resample a sequence of frames to target_len."""
        n = len(frames)
        if n == target_len:
            return list(frames)
        if n == 0:
            raise ValueError("Cannot resample empty frame sequence.")
        indices = np.linspace(0, n - 1, target_len).round().astype(int)
        return [frames[idx] for idx in indices]

    def extract_frame_features(
        self, frame: np.ndarray, annotate: bool = False
    ) -> tuple[np.ndarray, bool, Optional[np.ndarray]]:
        """Extract landmarks from a single OpenCV BGR frame.

        Args:
            frame: BGR numpy uint8 image.
            annotate: If True, draws landmarks on the frame and returns it.

        Returns:
            Tuple of (feature_vector_188d, has_hand_bool, annotated_frame_or_None).
        """
        if self.holistic is None:
            self.holistic = create_model()

        img, results = detect(frame, self.holistic)
        keypoints = extract(results)
        has_hand = (results.left_hand_landmarks is not None) or (
            results.right_hand_landmarks is not None
        )

        annotated_img = None
        if annotate:
            annotated_img = draw_landmarks_new(img.copy(), results)

        return keypoints, has_hand, annotated_img

    def predict(
        self,
        frames: Sequence[np.ndarray],
        return_annotated: bool = False,
    ) -> PredictionResult:
        """Run sign language prediction on a sequence of OpenCV frames.

        Args:
            frames: Sequence or list of OpenCV BGR frames (numpy ndarrays).
                    Length should ideally match sequence_len (e.g. 24 frames),
                    or will be resampled if self.resample is True.
            return_annotated: If True, includes annotated frames with landmarks overlay
                              in the result.

        Returns:
            PredictionResult: Object containing prediction, confidence, class probabilities,
                              and status flags.
        """
        start_time = time.perf_counter()

        if frames is None or len(frames) == 0:
            raise ValueError("Input 'frames' sequence cannot be empty.")

        num_received = len(frames)

        # ── 1. Sequence length handling / Resampling ─────────────────────────
        if num_received != self.sequence_len:
            if self.resample:
                proc_frames = self._resample_frames(frames, self.sequence_len)
            else:
                raise ValueError(
                    f"Expected {self.sequence_len} frames, but got {num_received}. "
                    "Set resample=True or provide exact sequence length."
                )
        else:
            proc_frames = list(frames)

        # ── 2. Feature extraction via MediaPipe ──────────────────────────────
        keypoints_list: List[np.ndarray] = []
        annotated_list: Optional[List[np.ndarray]] = [] if return_annotated else None
        hand_frame_count = 0

        for frame in proc_frames:
            if not isinstance(frame, np.ndarray):
                raise TypeError(
                    f"Each frame must be a numpy.ndarray, got {type(frame)}."
                )

            kp, has_hand, ann_frame = self.extract_frame_features(
                frame, annotate=return_annotated
            )
            keypoints_list.append(kp)
            if has_hand:
                hand_frame_count += 1
            if return_annotated and ann_frame is not None:
                annotated_list.append(ann_frame)

        # ── 3. Array formatting and feature standardization ──────────────────
        X = np.array([keypoints_list], dtype=np.float32)  # Shape: (1, seq_len, 188)

        if X.shape[-1] != self.expected_dim:
            raise ValueError(
                f"Feature dimension {X.shape[-1]} != model dimension {self.expected_dim}. "
                "The extractor changed (face-offset block added) — retrain the model "
                "and regenerate feature_stats.npz."
            )
        if self.feat_mean is not None and self.feat_std is not None:
            if self.feat_mean.shape[-1] != X.shape[-1]:
                warnings.warn(
                    f"Feature stats dimension {self.feat_mean.shape[-1]} != extracted "
                    f"dimension {X.shape[-1]} — skipping standardization."
                )
            else:
                X, _, _ = standardize_features(X, mean=self.feat_mean, std=self.feat_std)

        # ── 4. Keras model inference ─────────────────────────────────────────
        yhat = self.model.predict(X, verbose=0)
        probs = yhat[0]

        best_idx = int(np.argmax(probs))
        top_confidence = float(probs[best_idx])
        top_action = str(self.actions[best_idx])

        # ── 5. Hand presence and confidence validation ───────────────────────
        has_hand = hand_frame_count >= self.min_hand_frames
        hand_ratio = hand_frame_count / float(self.sequence_len)

        is_confident = top_confidence >= self.confidence_threshold
        is_valid = is_confident and (has_hand if self.require_hand else True)

        action = top_action if is_valid else None

        prob_dict = {
            str(self.actions[i]): float(probs[i]) for i in range(len(self.actions))
        }

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return PredictionResult(
            action=action,
            confidence=top_confidence,
            probabilities=prob_dict,
            is_valid=is_valid,
            has_hand=has_hand,
            top_action=top_action,
            raw_probabilities=probs,
            latency_ms=latency_ms,
            num_frames_received=num_received,
            num_frames_processed=self.sequence_len,
            hand_frame_ratio=hand_ratio,
            annotated_frames=annotated_list,
        )

    def predict_batch(
        self,
        batch_sequences: Sequence[Sequence[np.ndarray]],
        return_annotated: bool = False,
    ) -> List[PredictionResult]:
        """Predict a batch of frame sequences.

        Args:
            batch_sequences: List of frame sequences.
            return_annotated: Whether to annotate frames.

        Returns:
            List of PredictionResult objects.
        """
        return [
            self.predict(seq, return_annotated=return_annotated)
            for seq in batch_sequences
        ]


# ── Global Singleton & Convenience Functions ─────────────────────────────────

_GLOBAL_CONNECTOR: Optional[UniversalConnector] = None


def get_connector(**kwargs) -> UniversalConnector:
    """Retrieve or initialize the global shared UniversalConnector instance.

    Reusing a single connector instance across calls avoids re-loading the Keras model
    and MediaPipe graph.
    """
    global _GLOBAL_CONNECTOR
    if _GLOBAL_CONNECTOR is None:
        _GLOBAL_CONNECTOR = UniversalConnector(**kwargs)
    return _GLOBAL_CONNECTOR


def predict(
    frames: Sequence[np.ndarray],
    connector: Optional[UniversalConnector] = None,
    return_annotated: bool = False,
    **connector_kwargs,
) -> PredictionResult:
    """Convenience functional interface for third parties to run sign prediction.

    Example:
        import universal_connector as uc
        result = uc.predict(frames)
        print(result.action, result.confidence)

    Args:
        frames: Sequence of OpenCV BGR frames.
        connector: Optional existing UniversalConnector instance. If None,
                   uses the shared global connector.
        return_annotated: If True, returns landmark annotated frames.
        **connector_kwargs: Passed to UniversalConnector if initializing.

    Returns:
        PredictionResult: Standardized prediction result.
    """
    if connector is None:
        connector = get_connector(**connector_kwargs)
    return connector.predict(frames, return_annotated=return_annotated)


# ── CLI & Self-Test ──────────────────────────────────────────────────────────

def _self_test():
    """Run a diagnostic self-test with dummy frames to verify pipeline integrity."""
    print("=" * 60)
    print("Running UniversalConnector self-test...")
    print("=" * 60)

    try:
        with UniversalConnector() as connector:
            print(f"Model path          : {connector.model_path}")
            print(f"Actions             : {list(connector.actions)}")
            print(f"Target sequence len : {connector.sequence_len}")
            print(f"Confidence threshold: {connector.confidence_threshold}")

            # 1. Test exact sequence len with blank frames
            blank_frames = [
                np.zeros((480, 640, 3), dtype=np.uint8)
                for _ in range(connector.sequence_len)
            ]
            res1 = connector.predict(blank_frames)
            print("\n[Test 1] Blank frames (exact length):")
            print(f"  Result str        : {res1}")
            print(f"  Action            : {res1.action} (Expected: None / uncertain)")
            print(f"  Top action        : {res1.top_action}")
            print(f"  Has hand          : {res1.has_hand} (Expected: False)")
            print(f"  Is valid          : {res1.is_valid} (Expected: False)")
            print(f"  Latency           : {res1.latency_ms:.1f}ms")

            # 2. Test resampling (e.g. 15 frames -> 24 frames)
            short_frames = [
                np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(15)
            ]
            res2 = connector.predict(short_frames)
            print("\n[Test 2] Resampling test (15 frames -> 24 frames):")
            print(f"  Received frames   : {res2.num_frames_received}")
            print(f"  Processed frames  : {res2.num_frames_processed}")
            print(f"  Dict output keys  : {list(res2.to_dict().keys())}")

            # 3. Test dict-like access
            assert res2["top_action"] == res2.top_action
            print("\n[Test 3] Dictionary-like access: PASSED")

        print("\nSelf-test PASSED successfully!")
    except Exception as e:
        print(f"\nSelf-test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Universal Connector for VSL sign prediction."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run diagnostic pipeline self-test with dummy frames.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Test live inference on camera index (e.g. 0).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Run inference on a video file path.",
    )
    args = parser.parse_args()

    if args.test or (args.camera is None and args.video is None):
        _self_test()
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error opening video file: {args.video}")
            sys.exit(1)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        print(f"Loaded {len(frames)} frames from video.")
        with UniversalConnector() as conn:
            result = conn.predict(frames)
            print(json.dumps(result.to_dict(), indent=2))
    elif args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Could not open camera {args.camera}")
            sys.exit(1)
        print("Camera opened. Capturing window of frames...")
        with UniversalConnector() as conn:
            buffer = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                buffer.append(frame)
                if len(buffer) == conn.sequence_len:
                    res = conn.predict(buffer)
                    print(f"Prediction: {res.action} ({res.confidence:.1%})")
                    buffer.clear()
        cap.release()
