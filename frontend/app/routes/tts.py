"""POST /api/tts {text} -> audio/mpeg bytes"""

from app.services import tts as tts_svc


def speak(payload):
    text = (payload or {}).get("text", "")
    if isinstance(text, (list, tuple)):
        text = " ".join(str(w) for w in text)
    try:
        data = tts_svc.synthesize(text)
    except ValueError:
        return 400, {"ok": False, "error": "văn bản rỗng"}
    except RuntimeError as e:
        return 502, {"ok": False, "error": str(e)}
    return 200, data, "audio/mpeg"
