"""State snapshot service. Moved from frontend/caller.get_state()."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_FRONTEND = os.path.normpath(os.path.join(_HERE, "..", ".."))
_BACKEND = os.path.normpath(os.path.join(_FRONTEND, "..", "backend"))
for _p in (_BACKEND,
           os.path.join(_BACKEND, "src"),
           os.path.join(_BACKEND, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as cfg

from app.services.predictor import Predictor


def get_state():
    predictor = Predictor.peek()
    if predictor is None:
        return {"status": "Bấm Bắt đầu — camera của thiết bị sẽ gửi khung hình",
                "pred": None, "conf": 0.0, "sentence": [], "buffer": "0/0",
                "inferring": False, "debug": False,
                "smart": "", "smart_pending": False, "smart_error": None,
                "groq_model": cfg.GROQ_MODEL,
                "groq_configured": bool(cfg.GROQ_API_KEY)}
    try:
        return predictor.snapshot()
    except Exception as e:  # model not loaded yet / load failed
        return {"status": f"unavailable: {e}", "pred": None,
                "conf": 0.0, "sentence": [], "buffer": "0/0",
                "inferring": False, "debug": False,
                "groq_model": cfg.GROQ_MODEL,
                "groq_configured": bool(cfg.GROQ_API_KEY),
                "smart": "", "smart_pending": False, "smart_error": None}
