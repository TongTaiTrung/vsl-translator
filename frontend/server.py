import json
import os
import socket
import ssl
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from app import router
from app.routes import frames as frames_route
from app.routes import static as static_route
from app.services.client_frames import MAX_BYTES


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- API ---------------------------------------------------------------
    def _send_bytes(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_api(self, method, path):
        result = router.dispatch(method, path, self._payload)
        status = result[0] if result else None
        if status is not None:
            if len(result) == 3:  # binary body: (status, bytes, mime)
                _, body, mime = result
                self._send_bytes(body, mime, status)
            else:
                _, body = result
                self._send_json(body, status)
            return True
        status, body = router.legacy_gone(path)
        if status is not None:
            self._send_json(body, status)
            return True
        return False

    def _read_payload(self):
        payload, err = router.parse_json_body(self.rfile, self.headers)
        if err:
            return None, err
        return payload, None

    def log_message(self, fmt, *args):
        if urlparse(self.path).path == "/api/frames":
            return
        super().log_message(fmt, *args)

    def _handle_frames(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._send_json({"ok": False, "error": "bad length"}, 400)
        if length <= 0:
            return self._send_json({"ok": False, "error": "khung hình rỗng"}, 400)
        if length > MAX_BYTES:
            return self._send_json({"ok": False, "error": "khung hình quá lớn"}, 413)
        body = self.rfile.read(length)
        status, payload = frames_route.ingest(body)
        self._send_json(payload, status)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/frames":
            return self._handle_frames()
        if path == "/video" or path.startswith("/static/") or path in ("/", "/index.html"):
            return self._send_json({"ok": False, "error": "not found"}, 404)
        payload, err = self._read_payload()
        if err:
            status, body = err
            return self._send_json(body, status)
        self._payload = payload
        if self._handle_api("POST", path):
            return
        self._send_json({"ok": False, "error": "not found"}, 404)

    def do_PUT(self):
        path = urlparse(self.path).path
        payload, err = self._read_payload()
        if err:
            status, body = err
            return self._send_json(body, status)
        self._payload = payload
        if self._handle_api("PUT", path):
            return
        self._send_json({"ok": False, "error": "not found"}, 404)

    def do_DELETE(self):
        path = urlparse(self.path).path
        # DELETE has no body in our contract
        self._payload = {}
        if self._handle_api("DELETE", path):
            return
        self._send_json({"ok": False, "error": "not found"}, 404)

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            self._payload = {}
            if self._handle_api("GET", path):
                return
            self._send_json({"ok": False, "error": "not found"}, 404)
            return
        if path == "/video":
            self._send_json(
                {"ok": False,
                 "error": "camera is on the device; POST /api/frames"},
                410)
            return
        fpath = static_route.resolve_static(path)
        if fpath is not None:
            self.send_response(200)
            self.send_header("Content-Type", static_route.mime_for(fpath))
            self.end_headers()
            with open(fpath, "rb") as f:
                self.wfile.write(f.read())
        else:
            self._send_json({"ok": False, "error": "not found"}, 404)


def _lan_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()


def _is_ip(host):
    try:
        socket.inet_aton(host)
    except OSError:
        return False
    return True


def _ensure_cert(directory, hosts):
    """Chứng chỉ tự ký, SAN gồm localhost và IP LAN. Tạo lại khi IP đổi."""
    os.makedirs(directory, exist_ok=True)
    cert = os.path.join(directory, "cert.pem")
    key = os.path.join(directory, "key.pem")
    stamp = os.path.join(directory, "hosts.txt")
    wanted = "\n".join(hosts) + "\n"
    if os.path.isfile(cert) and os.path.isfile(key) and os.path.isfile(stamp):
        with open(stamp) as f:
            if f.read() == wanted:
                return cert, key
    san = ",".join(
        ("IP:%s" % h) if _is_ip(h) else ("DNS:%s" % h) for h in hosts)
    cnf_path = os.path.join(directory, "san.cnf")
    with open(cnf_path, "w") as f:
        f.write(
            "[req]\n"
            "distinguished_name = dn\n"
            "x509_extensions = v3\n"
            "prompt = no\n"
            "[dn]\n"
            "CN = vsl-translator\n"
            "[v3]\n"
            "subjectAltName = %s\n"
            "basicConstraints = CA:FALSE\n"
            "keyUsage = digitalSignature, keyEncipherment\n"
            "extendedKeyUsage = serverAuth\n" % san)
    try:
        subprocess.check_call(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048",
             "-keyout", key, "-out", cert,
             "-days", "365", "-nodes", "-config", cnf_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("Không tạo được chứng chỉ HTTPS (cần openssl): %s" % exc)
    with open(stamp, "w") as f:
        f.write(wanted)
    return cert, key


def _send_https_hint(conn):
    """Plain HTTP onto the TLS port: Safari reports that as a lost connection."""
    conn.settimeout(2)
    host = ""
    try:
        data = conn.recv(2048)
    except OSError:
        data = b""
    for line in data.split(b"\r\n"):
        if line.lower().startswith(b"host:"):
            host = line.split(b":", 1)[1].strip().decode("latin1", "replace")
            break
    url = ("https://%s" % host) if host else "https://<ip-máy>:8000"
    body = ("Trang này chỉ mở bằng HTTPS.\nMở: %s\n" % url).encode("utf-8")
    resp = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n"
        b"Content-Length: %d\r\n"
        b"Connection: close\r\n\r\n" % len(body)
    ) + body
    try:
        conn.sendall(resp)
    except OSError:
        pass
    conn.close()


class _TLSServer(ThreadingHTTPServer):
    """Wrap each accepted socket. Do not wrap the listening socket.

    Cap at TLS 1.2: iPhone Safari drops a self-signed TLS 1.3 handshake and
    shows "network connection was lost" instead of the certificate sheet.
    """

    def __init__(self, addr, handler, context):
        super().__init__(addr, handler)
        self.ssl_context = context

    def get_request(self):
        conn, addr = super().get_request()
        conn.settimeout(5)
        try:
            peeked = conn.recv(1, socket.MSG_PEEK)
        except OSError:
            conn.close()
            raise
        conn.settimeout(None)
        if peeked[:1] != b"\x16":
            print("Kết nối thường (không TLS) từ %s — cần https://" % addr[0])
            _send_https_hint(conn)
            raise OSError("plain http")
        try:
            return self.ssl_context.wrap_socket(conn, server_side=True), addr
        except ssl.SSLError as exc:
            print("TLS từ %s thất bại: %s" % (addr[0], exc))
            conn.close()
            raise


def serve(port=8000):
    hosts = ["localhost", "127.0.0.1"]
    lan = _lan_ip()
    if lan and lan not in hosts:
        hosts.append(lan)
    cert, key = _ensure_cert(os.path.join(_HERE, "certs"), hosts)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(cert, key)
    httpd = _TLSServer(("0.0.0.0", port), Handler, ctx)
    print("Máy này: https://127.0.0.1:%d" % port)
    if lan:
        print("Điện thoại cùng mạng: https://%s:%d" % (lan, port))
    else:
        print("Không thấy IP mạng — điện thoại sẽ không vào được.")
    print("Gõ đủ https://. Lần đầu Safari hiện This Connection Is Not Private:")
    print("Show Details → visit this website → Visit.")
    httpd.serve_forever()


if __name__ == "__main__":
    serve()
