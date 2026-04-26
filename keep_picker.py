"""
keep_picker.py — review captured backgrounds in a browser, mark which to keep.

Serves a paginated grid of images from backgrounds/ on https://0.0.0.0:4444/.
Click any image to toggle "kept". Click 'save' to copy the selected ones to
backgrounds_kept/. Use that directory as the input to synthesize.py.
"""

from __future__ import annotations

import json
import os
import shutil
import ssl
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
BG_DIR = os.path.join(ROOT, "backgrounds")
KEEP_DIR = os.path.join(ROOT, "backgrounds_kept")
INDEX_HTML = os.path.join(ROOT, "keep_picker.html")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def _send_file(self, path: str, ctype: str) -> None:
        try:
            with open(path, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            self.send_error(404); return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_file(INDEX_HTML, "text/html"); return
        if path == "/list":
            exts = (".jpg", ".jpeg", ".png")
            files = sorted(f for f in os.listdir(BG_DIR) if f.lower().endswith(exts)) if os.path.isdir(BG_DIR) else []
            kept = set(os.listdir(KEEP_DIR)) if os.path.isdir(KEEP_DIR) else set()
            entries = [{"name": f, "kept": (f in kept)} for f in files]
            self._send_json(200, {"files": entries, "total": len(files), "already_kept": len(kept)})
            return
        if path.startswith("/img/"):
            name = path[len("/img/"):]
            if "/" in name or "\\" in name or ".." in name:
                self.send_error(403); return
            full = os.path.join(BG_DIR, name)
            if not os.path.isfile(full):
                self.send_error(404); return
            ctype = "image/png" if name.lower().endswith(".png") else "image/jpeg"
            self._send_file(full, ctype); return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n).decode("utf-8") if n else "{}"
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "bad JSON"}); return

        if path == "/save":
            names = data.get("names", [])
            if not isinstance(names, list):
                self._send_json(400, {"error": "names must be a list"}); return
            os.makedirs(KEEP_DIR, exist_ok=True)
            copied = 0
            for n in names:
                if not isinstance(n, str) or "/" in n or ".." in n:
                    continue
                src = os.path.join(BG_DIR, n)
                if not os.path.isfile(src):
                    continue
                shutil.copy(src, os.path.join(KEEP_DIR, n))
                copied += 1
            print(f"[save] copied {copied}/{len(names)} → {KEEP_DIR}", flush=True)
            self._send_json(200, {"ok": True, "kept": copied, "dir": KEEP_DIR})
            return

        if path == "/clear_kept":
            removed = 0
            if os.path.isdir(KEEP_DIR):
                for f in os.listdir(KEEP_DIR):
                    if f.endswith(".jpg"):
                        try:
                            os.remove(os.path.join(KEEP_DIR, f))
                            removed += 1
                        except OSError:
                            pass
            print(f"[clear_kept] removed {removed} from {KEEP_DIR}", flush=True)
            self._send_json(200, {"ok": True, "removed": removed})
            return

        self.send_error(404)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def log_message(self, *a, **kw) -> None:
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4444
    cert_dir = os.path.expanduser("~/.local/share/certbot/live/your-domain")
    use_https = os.path.isfile(f"{cert_dir}/fullchain.pem") and "--http" not in sys.argv
    os.makedirs(BG_DIR, exist_ok=True)
    os.makedirs(KEEP_DIR, exist_ok=True)
    server = HTTPServer(("0.0.0.0", port), Handler)
    proto = "http"
    if use_https:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(f"{cert_dir}/fullchain.pem", f"{cert_dir}/privkey.pem")
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        proto = "https"
    print(f"keep_picker on {proto}://0.0.0.0:{port}/", flush=True)
    print(f"  bg dir:   {BG_DIR}", flush=True)
    print(f"  keep dir: {KEEP_DIR}", flush=True)
    server.serve_forever()
