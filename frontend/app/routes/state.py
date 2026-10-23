"""GET /api/state | GET /api/health"""

from app.services import state as state_svc


def get_state():
    try:
        return 200, {"ok": True, "data": state_svc.get_state()}
    except Exception as e:
        return 500, {"ok": False, "error": str(e)}


def health():
    return 200, {"ok": True, "data": {"status": "up"}}
