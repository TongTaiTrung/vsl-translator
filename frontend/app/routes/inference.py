"""POST /api/inference/start | POST /api/inference/stop"""

from app.services.client_frames import reset as reset_frames, resume as resume_frames
from app.services.predictor import Predictor


def start(payload):
    try:
        predictor = Predictor.get()
    except Exception as e:
        return 500, {"ok": False, "error": str(e)}
    debug = bool((payload or {}).get("debug", False))
    reset_frames()
    predictor.start_inference(debug=debug)
    resume_frames()
    return 200, {"ok": True, "data": predictor.snapshot()}


def stop(payload):
    try:
        predictor = Predictor.get()
    except Exception as e:
        return 500, {"ok": False, "error": str(e)}
    reset_frames()
    predictor.stop_inference()
    return 200, {"ok": True, "data": predictor.snapshot()}
