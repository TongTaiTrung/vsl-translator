"""Nhận JPEG do trình duyệt chụp và đẩy vào Predictor.

Không mở camera. Cửa sổ 12 frame vẫn nằm trong Predictor, giống
realtime_predict — client chỉ gửi từng khung, không gom batch.
"""

import base64
import threading

import cv2
import numpy as np

MAX_BYTES = 1_500_000

_lock = threading.Lock()
_paused = False


class FrameBusy(Exception):
    pass


class FrameDecodeError(Exception):
    pass


def reset():
    """Ngừng nhận khung mới và chờ khung đang chạy xong.

    start/stop xóa cửa sổ và phiếu bầu ngay sau lệnh này. Nếu process()
    còn đang chạy, kết quả của nó sẽ ghi vào cửa sổ vừa xóa.
    """
    global _paused
    _paused = True
    with _lock:
        pass


def resume():
    """Cho nhận khung trở lại. Chỉ start gọi; stop giữ trạng thái dừng."""
    global _paused
    _paused = False


def ingest_jpeg(data, predictor):
    """Giải mã một JPEG và chạy Predictor.process. Trả snapshot.

    MediaPipe không an toàn đa luồng (detect nằm ngoài lock của Predictor),
    nên chỉ một khung được xử lý tại một thời điểm. Khung đến khi đang bận
    thì bỏ — client gửi khung kế tiếp.
    """
    if not data or len(data) > MAX_BYTES:
        raise FrameDecodeError("khung hình quá lớn hoặc rỗng")
    if _paused:
        raise FrameBusy()
    if not _lock.acquire(blocking=False):
        raise FrameBusy()
    try:
        if _paused:
            raise FrameBusy()
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise FrameDecodeError("không đọc được JPEG")
        img = predictor.process(frame)
        snap = predictor.snapshot()
        if snap.get("debug"):
            ok, buf = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                snap["frame"] = base64.b64encode(buf.tobytes()).decode("ascii")
        return snap
    finally:
        _lock.release()
