"""MJPEG video stream service. Moved from frontend/caller.py unchanged."""

import threading

import cv2


# Only one client may hold the camera at a time.
_stream_lock = threading.Lock()


def video_stream(predictor, camera_index=0):
    """Yield annotated JPEG frames (MJPEG), like realtime_predict.py.

    Camera mở khi và chỉ khi inference đang bật. Client ngắt stream
    (bấm Dừng) thì vòng lặp dừng và camera được release ngay.
    """
    if not _stream_lock.acquire(blocking=False):
        raise RuntimeError("camera busy: another client is streaming")
    try:
        if predictor is None or not predictor.is_inferring():
            raise RuntimeError("inference is off: press Start first")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera "
                               "(check Privacy & Security → Camera)")
        try:
            while cap.isOpened():
                if not predictor.is_inferring():
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                img = predictor.process(frame)
                ok, buf = cv2.imencode(".jpg", img)
                if not ok:
                    continue
                yield buf.tobytes()
        finally:
            cap.release()
    finally:
        _stream_lock.release()
