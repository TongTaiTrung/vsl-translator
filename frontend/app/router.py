"""Central dispatch: (method, path) -> handler. JSON parse + response helpers."""

import json

from app.routes import inference, sentences, smart, config, state, tts, stt

# Exact-match table first; legacy action-based paths handled separately.
ROUTES = {
    ("POST", "/api/inference/start"): inference.start,
    ("POST", "/api/inference/stop"): inference.stop,
    ("GET", "/api/sentences"): lambda payload: sentences.list_sentences(),
    ("DELETE", "/api/sentences"): lambda payload: sentences.clear(),
    ("DELETE", "/api/sentences/last"): lambda payload: sentences.backspace(),
    ("POST", "/api/smart/retries"): smart.retry,
    ("PUT", "/api/config/groq-key"): config.set_groq_key,
    ("GET", "/api/state"): lambda payload: state.get_state(),
    ("GET", "/api/health"): lambda payload: state.health(),
    ("POST", "/api/tts"): tts.speak,
    ("POST", "/api/stt/start"): stt.start,
    ("POST", "/api/stt/stop"): stt.stop,
    ("GET", "/api/stt"): lambda payload: stt.poll(payload),
}

# Legacy action-based endpoints: 410 Gone with pointer to the new path.
LEGACY = {
    "/api/inference": "/api/inference/start|/api/inference/stop",
    "/api/sentence": "/api/sentences, /api/sentences/last",
    "/api/smart": "/api/smart/retries",
    "/api/config": "/api/config/groq-key",
    "/api/test": None,  # removed entirely
}


def parse_json_body(rfile, headers):
    length = int(headers.get("Content-Length", 0))
    raw = rfile.read(length) if length else b"{}"
    try:
        return json.loads(raw or b"{}"), None
    except json.JSONDecodeError:
        return None, (400, {"ok": False, "error": "invalid JSON"})


def dispatch(method, path, payload):
    """Return (status, body_dict) or (None, None) if not an API route."""
    handler = ROUTES.get((method, path))
    if handler is None:
        return None, None
    return handler(payload)


def legacy_gone(path):
    if path in LEGACY:
        target = LEGACY[path]
        msg = ("removed; use %s" % target) if target else "removed"
        return 410, {"ok": False, "error": msg, "migrated_to": target}
    return None, None
