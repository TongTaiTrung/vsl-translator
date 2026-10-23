"""POST /api/frames — body thô image/jpeg, không phải JSON."""

from app.services.client_frames import FrameBusy, FrameDecodeError, ingest_jpeg
from app.services.predictor import Predictor


def ingest(body):
    predictor = Predictor.peek()
    if predictor is None or not predictor.is_inferring():
        return 409, {"ok": False, "error": "chưa bấm Bắt đầu"}
    try:
        snap = ingest_jpeg(body, predictor)
    except FrameBusy:
        return 429, {"ok": False, "error": "máy đang bận"}
    except FrameDecodeError as exc:
        return 400, {"ok": False, "error": str(exc)}
    return 200, {"ok": True, "data": snap}
