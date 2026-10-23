"""Vietnamese text-to-speech via gTTS. synthesize(text) -> MP3 bytes.

Single shared module: every TTS request goes through here, no caller
hardcodes language or gTTS params. Language and voice are defined once
in DEFAULT_LANG / TLD below.
"""

import hashlib
import io
import os
import re
import tempfile

from gtts import gTTS

DEFAULT_LANG = "vi"
# com.vn endpoint gives a native Vietnamese voice; fall back to com
# if it is unreachable.
TLD_PRIMARY = "com.vn"
TLD_FALLBACK = "com"
MAX_CHARS = 500
_CACHE_SIZE = 50

_cache = {}


def _cached_path(key):
    return os.path.join(tempfile.gettempdir(), f"vsl_tts_{key}.mp3")


def normalize(text):
    """Light cleanup so Groq/smart output reads naturally."""
    text = (text or "").strip()
    # strip wrapping quotes / markdown fences the LLM sometimes adds
    text = re.sub(r"^```\w*\s*|\s*```$", "", text).strip()
    text = text.strip("\"'“”").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _render(text):
    last_exc = None
    for tld in (TLD_PRIMARY, TLD_FALLBACK):
        try:
            buf = io.BytesIO()
            gTTS(text=text, lang=DEFAULT_LANG, tld=tld).write_to_fp(buf)
            data = buf.getvalue()
            if data:
                return data
        except Exception as e:  # noqa: BLE001 - try next endpoint
            last_exc = e
    raise RuntimeError(f"không tạo được âm thanh: {last_exc}")


def synthesize(text):
    text = normalize(text)
    if not text:
        raise ValueError("empty text")
    text = text[:MAX_CHARS]
    key = hashlib.sha1(f"{DEFAULT_LANG}:{text}".encode("utf-8")).hexdigest()

    if key in _cache:
        return _cache[key]
    path = _cached_path(key)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        _cache[key] = data
        return data

    data = _render(text)

    _cache[key] = data
    if len(_cache) > _CACHE_SIZE:
        _cache.pop(next(iter(_cache)))
    try:
        with open(path, "wb") as f:
            f.write(data)
    except OSError:
        pass
    return data
