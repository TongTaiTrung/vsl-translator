"""Realtime Vietnamese speech-to-text via SpeechRecognition.

Singleton (same pattern as Predictor): listen_in_background appends
recognized sentences to a thread-safe buffer; GET /api/stt drains it.
"""

import threading

LANG = "vi-VN"


class SpeechToText:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._texts = []
        self._error = None
        self._stop = None
        self._listening = False

    @classmethod
    def get(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _callback(self, recognizer, audio):
        try:
            text = recognizer.recognize_google(audio, language=LANG)
        except Exception as e:  # UnknownValueError -> skip; RequestError -> record
            import speech_recognition as sr
            if isinstance(e, sr.UnknownValueError):
                return
            with self._lock:
                self._error = "lỗi kết nối mạng"
            return
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._texts.append(text)
            self._error = None

    def start(self):
        import speech_recognition as sr

        with self._lock:
            if self._listening:
                return
        try:
            recognizer = sr.Recognizer()
            mic = sr.Microphone()
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
            self._stop = recognizer.listen_in_background(
                mic, self._callback)
        except Exception as e:
            with self._lock:
                self._error = f"không mở được micro: {e}"
            raise RuntimeError(self._error)
        with self._lock:
            self._listening = True
            self._error = None

    def stop(self):
        with self._lock:
            stop = self._stop
            self._stop = None
            self._listening = False
        if stop is not None:
            stop(wait_for_stop=False)

    def drain(self):
        with self._lock:
            texts = list(self._texts)
            self._texts.clear()
            return texts

    def status(self):
        with self._lock:
            return {"listening": self._listening, "error": self._error}
