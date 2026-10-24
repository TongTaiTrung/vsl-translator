#!/usr/bin/env python3
"""WebSocket ingest server: remote devices stream JPEG frames, backend predicts.

Protocol:
    Client -> server: binary JPEG frame bytes (any resolution; server
        letterboxes to FRAME_STD_SIZE via normalize_frame).
    Server -> client: JSON text per frame:
        {"pred": str|None, "conf": float, "sentence": [str], "has_hand": bool}

One StreamSession per connection; idle connections past SESSION_TTL are
closed. Run: python scripts/server.py (or ../venv/bin/python ...).
"""

import asyncio
import json
import sys, os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np

import config as cfg

_model = None
_feat_mean = None
_feat_std = None


def _load_artifacts():
    global _model, _feat_mean, _feat_std
    if _model is not None:
        return
    from keras.models import load_model
    model_path = os.path.join(cfg.MODELS_DIR, cfg.MODEL_FILENAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at '{model_path}'. Run scripts/train.py first.")
    _model = load_model(model_path)
    stats_path = os.path.join(cfg.MODELS_DIR, "feature_stats.npz")
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        _feat_mean, _feat_std = stats["mean"], stats["std"]
        print("Feature stats loaded.")
    else:
        _feat_mean, _feat_std = None, None
        print("WARNING: feature_stats.npz not found – no standardisation.")
    # Deferred imports (need TF/MediaPipe lazily at runtime)
    print("Model loaded.")


async def _handler(websocket):
    import websockets
    from vsl_translator.landmarks import create_model, detect, normalize_frame
    from scripts.stream_session import StreamSession

    _load_artifacts()
    session = StreamSession()
    last_seen = time.monotonic()
    print(f"Client connected: {websocket.remote_address}")
    try:
        with create_model() as holistic:
            async for message in websocket:
                if time.monotonic() - last_seen > cfg.SESSION_TTL:
                    break
                last_seen = time.monotonic()
                if isinstance(message, str):
                    continue  # ignore text control frames
                buf = np.frombuffer(message, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                frame, _meta = normalize_frame(frame)
                _img, results = detect(frame, holistic)
                out = session.process_frame(results, frame.shape[:2],
                                            _model, _feat_mean, _feat_std)
                pred = (str(cfg.ACTIONS[out["pred_idx"]])
                        if out["pred_idx"] >= 0 else None)
                conf = float(out["probs"][out["pred_idx"]]) if out["pred_idx"] >= 0 else 0.0
                await websocket.send(json.dumps({
                    "pred": pred, "conf": conf,
                    "sentence": out["sentence"],
                    "has_hand": bool(out["has_hand"]),
                }))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print(f"Client disconnected: {websocket.remote_address}")


async def _main():
    import websockets
    _load_artifacts()
    async with websockets.serve(_handler, "0.0.0.0", cfg.INGEST_PORT,
                                max_size=8 * 1024 * 1024):
        print(f"WebSocket ingest on ws://0.0.0.0:{cfg.INGEST_PORT}/ws/stream")
        await asyncio.get_running_loop().create_future()  # serve forever


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
