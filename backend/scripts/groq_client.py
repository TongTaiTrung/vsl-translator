"""Groq LLM cleanup client (shared module for scripts/).

Centralizes everything related to the Groq "smart sentence" feature so
realtime (threaded worker) and batch (synchronous) callers share one
implementation:

  - PROMPT_PATH / load_prompt_template(): read TXT_NOISE_RMV_PROMPT.md
  - call_groq(): single OpenAI-compatible chat call against Groq endpoint
  - predict_smart(): sync one-shot cleanup, "" on empty input / missing key / error
  - SmartSentence + smart_worker(): threaded poll loop used by realtime UI

Settings live in config.py (env-overridable: GROQ_MODEL, GROQ_BASE_URL,
GROQ_TIMEOUT, GROQ_RETRIES, SMART_CALL_INTERVAL, SMART_RETRY_INTERVAL,
SMART_POLL_INTERVAL, GROQ_API_KEY).
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config as cfg

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..",
                           "TXT_NOISE_RMV_PROMPT.md")


def load_prompt_template(path=None):
    """Read the noise-removal prompt file. Raises OSError if missing."""
    with open(path or PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def call_groq(raw_sentence, prompt_template=None, api_key=None):
    """Send raw words to Groq for cleanup, return cleaned text.

    Uses the OpenAI module against the Groq endpoint (base_url from config,
    model GROQ_MODEL, key GROQ_API_KEY from .env).
    """
    from openai import OpenAI

    api_key = api_key if api_key is not None else cfg.GROQ_API_KEY
    if prompt_template is None:
        prompt_template = load_prompt_template()
    client = OpenAI(
        api_key=api_key,
        base_url=cfg.GROQ_BASE_URL,
        timeout=cfg.GROQ_TIMEOUT,
        max_retries=cfg.GROQ_RETRIES + 1,
    )
    prompt = prompt_template.replace("{{Predict}}", raw_sentence)
    resp = client.chat.completions.create(
        model=cfg.GROQ_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": raw_sentence},
        ],
        temperature=0,
        top_p=1.0,
        max_tokens=256,
    )
    text = (resp.choices[0].message.content or "").strip()
    text = text.strip().strip('"').strip("'")
    if text.lower().startswith("output:"):
        text = text[7:].strip()
    if not text:
        raise RuntimeError(f"Empty Groq response: {resp}")
    return text


def cleanup_with_retries(raw_sentence, prompt_template, api_key):
    """Call Groq with GROQ_RETRIES retries. Raises last error if all fail."""
    last_exc = None
    for _ in range(cfg.GROQ_RETRIES + 1):
        try:
            return call_groq(raw_sentence, prompt_template, api_key)
        except Exception as e:
            last_exc = e
    raise last_exc


def retry_smart_sync(raw_sentence, api_key=None):
    """Synchronous one-shot cleanup. Raises on empty input/missing key/error.

    Used by the frontend retry path: caller passes the raw sentence shown in
    the UI, gets cleaned text back immediately (no worker polling).
    """
    raw = raw_sentence.strip() if isinstance(raw_sentence, str) else ""
    if not raw:
        raise ValueError("empty sentence: nothing to retry")
    key = api_key if api_key is not None else cfg.GROQ_API_KEY
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY — nhập key ở phần Cài đặt")
    return cleanup_with_retries(raw, load_prompt_template(), key)


def predict_smart(raw_word):
    """Clean one raw word via Groq. Returns cleaned text ("" on empty/error)."""
    if not raw_word:
        return ""
    if not os.environ.get("GROQ_API_KEY", ""):
        return ""
    try:
        return call_groq(raw_word, load_prompt_template()).strip()
    except Exception:
        return ""


class SmartSentence:
    """Thread-safe holder for the LLM-cleaned sentence."""

    def __init__(self):
        self._lock = threading.Lock()
        self.text = ""
        self.pending = False  # a request is in flight
        self.error = None     # last error message, if any
        self._last_sent = ""
        self._last_err_time = 0.0
        self._force_retry = False  # manual retry fires even when not inferring
        self._gen = 0
        self._inflight_gen = 0

    def set_result(self, raw_sent, cleaned):
        with self._lock:
            self.text = cleaned
            self._last_sent = raw_sent
            self.pending = False
            self.error = None
            self._force_retry = False

    def set_error(self, msg):
        with self._lock:
            self.pending = False
            self.error = msg

    def mark_pending(self, raw_sent):
        with self._lock:
            if self.pending:
                return False
            self.pending = True
            self._last_sent = raw_sent
            self._inflight_gen = self._gen
            return True

    def retry_now(self):
        """Force an immediate retry of the last sentence (manual retry button).

        Clears cooldown + last-sent so the worker picks it up on next poll.
        Sets _force_retry so the worker also fires when inference is stopped.
        Returns False if a request is already in flight or nothing to retry.
        """
        with self._lock:
            if self.pending or not self._last_sent:
                return False
            self._last_sent = ""
            self._last_err_time = 0.0
            self.error = None
            self._force_retry = True
            return True

    def clear(self):
        with self._lock:
            self.text = ""
            self._last_sent = ""
            self.pending = False
            self.error = None
            self._force_retry = False
            self._gen += 1

    def finish(self, raw_sent, cleaned):
        """Store a worker result. Drops it if clear() ran while the call was in flight."""
        with self._lock:
            if self._inflight_gen != self._gen:
                self.pending = False
                return False
            self.text = cleaned
            self._last_sent = raw_sent
            self.pending = False
            self.error = None
            self._force_retry = False
            return True

    def fail(self, msg):
        """Store a worker error. Drops it if clear() ran while the call was in flight."""
        with self._lock:
            if self._inflight_gen != self._gen:
                self.pending = False
                return False
            self.pending = False
            self.error = msg
            return True

    def read(self):
        with self._lock:
            return self.text, self.pending, self.error


def smart_worker(get_sentence, smart, stop_event, is_active=None):
    """Poll every 0.1s; call Groq only when the sentence changed + 7s throttle."""
    try:
        prompt_template = load_prompt_template()
    except OSError as e:
        smart.set_error(f"Prompt file missing: {e}")
        return
    if is_active is None:
        is_active = lambda: True
    last_call = 0.0
    while not stop_event.is_set():
        raw = " ".join(get_sentence())
        if raw:
            api_key = cfg.GROQ_API_KEY
            if not api_key:
                smart.set_error("Missing GROQ_API_KEY — nhập key ở phần Cài đặt")
            else:
                with smart._lock:
                    already_sent = raw == getattr(smart, "_last_sent", "")
                    busy = smart.pending
                    last_err = getattr(smart, "_last_err_time", 0.0)
                    forced = getattr(smart, "_force_retry", False)
                cooling = (time.monotonic() - last_err) < cfg.SMART_RETRY_INTERVAL
                throttled = (time.monotonic() - last_call) < cfg.SMART_CALL_INTERVAL
                if not already_sent and not busy and not cooling and not throttled:
                    if not is_active() and not forced:
                        pass  # not inferring: skip call, retry on next poll
                    elif smart.mark_pending(raw):
                        # Consume explicit retry flag (auto path leaves it False).
                        with smart._lock:
                            smart._force_retry = False
                        try:
                            cleaned = cleanup_with_retries(
                                raw, prompt_template, api_key)
                            if smart.finish(raw, cleaned):
                                last_call = time.monotonic()
                        except Exception as e:  # keep old text, retry after interval
                            if smart.fail(
                                    f"[{cfg.GROQ_MODEL}] {type(e).__name__}: {e}"):
                                with smart._lock:
                                    # allow retry of the SAME sentence after cooldown
                                    # (not only on next change)
                                    smart._last_sent = ""
                                    smart._last_err_time = time.monotonic()
        time.sleep(cfg.SMART_POLL_INTERVAL)


# ── Backward-compatible private aliases ─────────────────────────────────────
# Older callers did `from scripts.realtime_predict import _call_groq`.
_load_prompt_template = load_prompt_template
_call_groq = call_groq
