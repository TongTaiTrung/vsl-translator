"""POST /api/smart/retries {sentence}"""

from app.services.predictor import Predictor


def retry(payload):
    predictor = Predictor.peek()
    if predictor is None:
        return 409, {"ok": False,
                     "error": "predictor not started — bấm Bắt đầu ít nhất 1 lần để nạp model"}
    raw = (payload or {}).get("sentence", "")
    if isinstance(raw, (list, tuple)):
        raw = " ".join(str(w) for w in raw)
    try:
        result = predictor.retry_smart(raw)
    except Exception as e:
        return 500, {"ok": False, "error": str(e)}
    if not result.get("ok"):
        return 409, {"ok": False, "error": result.get("error", "smart failed")}
    snap = {k: v for k, v in result.items() if k != "ok"}
    return 200, {"ok": True, "data": snap}
