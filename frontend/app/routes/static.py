"""Static file handler. Camera frames arrive at POST /api/frames."""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "static"))
INDEX_FILE = os.path.join(STATIC_DIR, "index.html")

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
}


def resolve_static(path):
    """Return absolute file path under static/ or None."""
    if path == "/":
        return INDEX_FILE
    if not path.startswith("/static/"):
        if path == "/index.html":  # legacy direct path
            return INDEX_FILE
        return None
    rel = path[len("/static/"):]
    if ".." in rel or rel.startswith("/"):
        return None
    full = os.path.normpath(os.path.join(STATIC_DIR, rel))
    if not full.startswith(STATIC_DIR) or not os.path.isfile(full):
        return None
    return full


def mime_for(path):
    _, ext = os.path.splitext(path)
    return MIME.get(ext.lower(), "application/octet-stream")
