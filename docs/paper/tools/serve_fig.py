"""Serves fig.html, the avatar files and run poses for the paper's figures: python serve_fig.py [port]."""
import json, sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STATIC, OUT, HERE = ROOT / "scripts" / "asl" / "static", ROOT / "data" / "asl" / "out", Path(__file__).parent


class H(SimpleHTTPRequestHandler):
    def send(self, body, ctype):
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/fig.html":
            return self.send((HERE / "fig.html").read_bytes(), "text/html")
        if path.startswith("/static/"):
            f = STATIC / path[8:]
            ctype = "text/javascript" if f.suffix == ".js" else "application/octet-stream"
            return self.send(f.read_bytes(), ctype) if f.is_file() else self.send_error(404)
        if path.startswith("/pose/"):
            f = OUT / path[6:].replace(".json", "") / "pose.npy"
            if not f.is_file(): return self.send_error(404)
            return self.send(json.dumps({"fps": 25, "frames": np.nan_to_num(np.load(f)).astype(float).round(4).tolist()}).encode(), "application/json")
        self.send_error(404)

    def log_message(self, *a): pass


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8790), H).serve_forever()
