"""HTTP inference server for VSLInference (stdlib only, no extra deps).

Local webapps POST 12 raw image frames and get back the prediction:

    POST /predict   {"frames": ["<base64 jpeg/png>", ... x12]}
    -> 200 {"label": str, "confidence": float, "probs": [float, ...]}

    GET  /health    -> 200 {"ok": true, "labels": [...]}

Each item in "frames" is a base64 JPEG/PNG image; a data-URL prefix
("data:image/jpeg;base64,...", as produced by canvas.toDataURL) is accepted.
Errors return non-200 {"ok": false, "error": "..."}.

Run:  python -m VSLInference.server --model /path/to/model.keras [--port 8001]
"""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from VSLInference import VSLInference  # noqa: E402

MAX_BODY_BYTES = 32 * 1024 * 1024

_vsl = None


def _send_json(handler, data, status=200):
    body = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    server_version = "VSLInference/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("VSLInference.server: " + fmt % args + "\n")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            _send_json(self, {"ok": True, "labels": _vsl._labels})
        else:
            _send_json(self, {"ok": False,
                              "error": "unknown endpoint (GET /health, POST /predict)"},
                       status=404)

    def do_POST(self):
        if self.path != "/predict":
            _send_json(self, {"ok": False,
                              "error": "unknown endpoint (GET /health, POST /predict)"},
                       status=404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            _send_json(self, {"ok": False, "error": "invalid Content-Length"},
                       status=411)
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            _send_json(self, {"ok": False,
                              "error": f"body must be 1..{MAX_BODY_BYTES} bytes"},
                       status=413)
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            _send_json(self, {"ok": False, "error": "body is not valid JSON"},
                       status=400)
            return
        frames = payload.get("frames") if isinstance(payload, dict) else None
        if frames is None:
            _send_json(self, {"ok": False,
                              "error": 'missing "frames": list of 12 base64 images'},
                       status=400)
            return
        try:
            out = _vsl.predict_bytes(frames)
        except (ValueError, RuntimeError) as e:
            _send_json(self, {"ok": False, "error": str(e)}, status=400)
            return
        except Exception as e:  # noqa: BLE001 — never leak a traceback to clients
            _send_json(self, {"ok": False,
                              "error": f"inference failed: {e}"}, status=500)
            return
        _send_json(self, {"ok": True, **out})


def serve(model_path, host="127.0.0.1", port=8001, **vsl_kwargs):
    global _vsl
    _vsl = VSLInference(model_path, **vsl_kwargs)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"VSLInference serving {model_path} on http://{host}:{port} "
          f"(POST /predict, GET /health)")
    try:
        httpd.serve_forever()
    finally:
        _vsl.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="path to model.keras")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--stats", default=None,
                        help="path to feature_stats.npz (default: beside model)")
    args = parser.parse_args(argv)
    serve(args.model, host=args.host, port=args.port, stats_path=args.stats)


if __name__ == "__main__":
    main()
