"""PUT /api/config/groq-key {key}"""

from app.services.predictor import Predictor


def set_groq_key(payload):
    key = ((payload or {}).get("key", "") or "").strip()
    if not key:
        return 400, {"ok": False, "error": "empty key"}
    try:
        predictor = Predictor.peek() or Predictor.get()
    except Exception as e:
        return 500, {"ok": False, "error": str(e)}
    try:
        result = predictor.set_groq_key(key)
    except ValueError as e:
        return 400, {"ok": False, "error": str(e)}
    except Exception as e:
        return 500, {"ok": False, "error": str(e)}
    snap = {k: v for k, v in result.items() if k != "ok"}
    return 200, {"ok": True, "data": snap}
