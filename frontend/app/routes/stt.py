"""POST /api/stt/start | POST /api/stt/stop | GET /api/stt"""

from app.services.stt import SpeechToText


def start(payload):
    try:
        SpeechToText.get().start()
    except RuntimeError as e:
        return 500, {"ok": False, "error": str(e)}
    return 200, {"ok": True,
                 "data": SpeechToText.get().status()}


def stop(payload):
    SpeechToText.get().stop()
    return 200, {"ok": True,
                 "data": SpeechToText.get().status()}


def poll(payload):
    stt = SpeechToText.get()
    data = stt.status()
    data["texts"] = stt.drain()
    return 200, {"ok": True, "data": data}
