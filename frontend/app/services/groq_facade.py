"""Frontend-side Groq caller: owns all backend groq_client usage for the UI.

caller.py (Predictor) delegates here so it never imports backend groq_client
directly:

  - FrontendGroqCaller(get_sentence, is_active): starts the SmartSentence
    worker thread (auto cleanup while inferring).
  - retry(raw_sentence, fallback_sentence): direct sync Groq call for the
    sentence the frontend sends; falls back to the backend-held sentence;
    finally falls back to nudging the worker thread.
  - read() / clear_if_empty(sentence): state accessors for snapshot/overlay.
  - notify_key_changed(): wake the worker after a new API key is saved.
"""

import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
# app/services -> frontend -> backend (+src, +scripts)
_FRONTEND = os.path.normpath(os.path.join(_HERE, "..", ".."))
_BACKEND = os.path.normpath(os.path.join(_FRONTEND, "..", "backend"))
for _p in (_BACKEND,
           os.path.join(_BACKEND, "src"),
           os.path.join(_BACKEND, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _FRONTEND not in sys.path:
    sys.path.insert(0, _FRONTEND)

import groq_client as gc


class FrontendGroqCaller:
    """Owns SmartSentence + worker thread for one Predictor."""

    def __init__(self, get_sentence, is_active):
        self._smart = gc.SmartSentence()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=gc.smart_worker,
            args=(get_sentence, self._smart, self._stop),
            kwargs={"is_active": is_active},
            daemon=True,
        )
        self._thread.start()

    def read(self):
        """Return (text, pending, error) of the cleaned sentence."""
        return self._smart.read()

    def clear(self):
        """Drop the cleaned sentence immediately (reset)."""
        self._smart.clear()

    def clear_if_empty(self, sentence):
        """Clear stale smart text when there is no raw sentence."""
        if not sentence:
            self._smart.clear()

    def retry(self, raw_sentence=None, fallback_sentence=None):
        """Retry Groq cleanup. Returns {"ok": True} or {"ok": False, "error"}.

        Prefers the sentence the frontend sends; falls back to the sentence
        the backend still holds (so retry works after Stop); finally nudges
        the worker thread when there is truly nothing new to send.
        """
        raw = (" ".join(raw_sentence)
               if isinstance(raw_sentence, (list, tuple))
               else (raw_sentence or ""))
        raw = raw.strip()
        if not raw:
            raw = (" ".join(fallback_sentence)
                   if isinstance(fallback_sentence, (list, tuple))
                   else (fallback_sentence or "")).strip()
        if raw:
            try:
                cleaned = gc.retry_smart_sync(raw)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self._smart.set_error(msg)
                return {"ok": False, "error": msg}
            self._smart.set_result(raw, cleaned)
            return {"ok": True}
        if not self._smart.retry_now():
            return {"ok": False,
                    "error": "nothing to retry (câu trống hoặc request đang chạy)"}
        return {"ok": True}

    def notify_key_changed(self):
        """Wake the worker after a new Groq API key is saved."""
        self._smart.retry_now()
