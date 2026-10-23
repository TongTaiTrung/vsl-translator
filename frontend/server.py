import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import caller


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/inference":
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send_json({"error": "invalid JSON"}, 400)
            action = payload.get("action", "")
            try:
                predictor = caller.Predictor.get()
            except Exception as e:
                return self._send_json({"error": str(e)}, 500)
            if action == "start":
                predictor.start_inference(
                    debug=bool(payload.get("debug", False)))
            elif action == "stop":
                predictor.stop_inference()
            else:
                return self._send_json({"error": "unknown action"}, 400)
            return self._send_json({"ok": True, **predictor.snapshot()})
        if path == "/api/smart":
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send_json({"error": "invalid JSON"}, 400)
            if payload.get("action", "") != "retry":
                return self._send_json({"error": "unknown action"}, 400)
            predictor = caller.Predictor.peek()
            if predictor is None:
                return self._send_json({"error": "predictor not started"},
                                       409)
            try:
                result = predictor.retry_smart()
            except Exception as e:
                return self._send_json({"error": str(e)}, 500)
            return self._send_json(result,
                                   200 if result.get("ok") else 409)
        if path == "/api/sentence":
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send_json({"error": "invalid JSON"}, 400)
            action = payload.get("action", "")
            try:
                predictor = caller.Predictor.get()
            except Exception as e:
                return self._send_json({"error": str(e)}, 500)
            if action == "backspace":
                predictor.backspace()
            elif action == "clear":
                predictor.clear_sentence()
            else:
                return self._send_json({"error": "unknown action"}, 400)
            return self._send_json({"ok": True, **predictor.snapshot()})
        if path == "/api/config":
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send_json({"error": "invalid JSON"}, 400)
            if payload.get("action", "") != "set_groq_key":
                return self._send_json({"error": "unknown action"}, 400)
            key = (payload.get("key", "") or "").strip()
            if not key:
                return self._send_json(
                    {"error": "empty key"}, 400)
            try:
                predictor = caller.Predictor.peek() or caller.Predictor.get()
            except Exception as e:
                return self._send_json({"error": str(e)}, 500)
            try:
                result = predictor.set_groq_key(key)
            except ValueError as e:
                return self._send_json({"error": str(e)}, 400)
            except Exception as e:
                return self._send_json({"error": str(e)}, 500)
            return self._send_json(result)
        if path != "/api/test":
            return self._send_json({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send_json({"error": "invalid JSON"}, 400)
        result = caller.test(payload.get("info", ""))
        return self._send_json({"ok": True, "result": result})

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            try:
                return self._send_json({"ok": True, **caller.get_state()})
            except Exception as e:
                return self._send_json({"error": str(e)}, 500)
        if path == "/video":
            predictor = caller.Predictor.peek()
            if predictor is None or not predictor.is_inferring():
                self.send_response(409)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    "Inference is off — press Start first.".encode())
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                for chunk in caller.videoStream():
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n"
                    )
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open("index.html", "rb") as f:
                self.wfile.write(f.read())
        else:
            self._send_json({"error": "not found"}, 404)


if __name__ == "__main__":
    print("Serving at http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
