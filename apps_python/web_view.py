"""
Browser view and remote control of the tracker: http://<board>:<port>/

The H.264 RTP stream needs a GStreamer receiver or an SDP file on the viewing
side; this needs nothing but a browser. It serves the drawn frames (boxes,
HUD, shot effects) as MJPEG, downscaled and rate limited so it stays cheap:

  - publish() is called from the pipeline thread on every frame. It returns
    immediately unless a browser is connected and the frame interval has
    passed, and even then only downscales (~2 ms at 640x360). JPEG encoding
    happens in the HTTP thread, and cv2 releases the GIL while encoding.
  - With nobody watching it costs one time comparison per frame.

With a controls.Controls the page is also a remote for the rig, the same
controls as the touch panel:

  GET  /api/state            tracker status, score and control state (JSON)
  POST /api/fire             take a shot
  POST /api/effect  {"effect": "aim"}
  POST /api/mode    {"manual": true}
  POST /api/servo   {"on": false}                   servo stop / resume
  POST /api/move    {"right": -1..1, "up": -1..1}   resend while held; it
                             expires after controls.MOVE_TIMEOUT
  POST /api/tap     {"x": 0..1, "y": 0..1}          a tap on the picture,
                             as a fraction of its size: lock onto the face
                             under it, or drop the lock

There is no authentication: anyone who can reach the port can steer the
servos. Keep it on the board's own networks (USB, mesh, a trusted Wi-Fi).
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_ui.html")
MAX_BODY = 4096


class WebView:
    def __init__(self, port=8080, width=640, max_fps=15, quality=70, controls=None):
        self.width = width
        self.interval = 1.0 / max_fps
        self.quality = quality
        self.controls = controls
        self.clients = 0
        self._frame = None
        self._seq = 0
        self._last = 0.0
        self._cond = threading.Condition()
        with open(PAGE_FILE, "rb") as f:
            self.page = f.read()

        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body, ctype="application/json"):
                if isinstance(body, (dict, list)):
                    body = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/":
                    self._send(200, view.page, "text/html; charset=utf-8")
                elif path == "/stream":
                    view._stream(self)
                elif path == "/api/state" and view.controls is not None:
                    self._send(200, view.controls.snapshot())
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                if view.controls is None:
                    self._send(404, {"error": "no controls"})
                    return
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    if n > MAX_BODY:
                        raise ValueError("body too large")
                    data = json.loads(self.rfile.read(n) or b"{}") if n else {}
                    if not isinstance(data, dict):
                        raise ValueError("expected a JSON object")
                    view._command(self.path.split("?", 1)[0], data)
                except KeyError:
                    self._send(404, {"error": "not found"})
                except (ValueError, TypeError) as exc:
                    self._send(400, {"error": str(exc)})
                else:
                    self._send(200, view.controls.snapshot())

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print("[INFO] Web view on http://<board>:%d/" % port)

    def _command(self, path, data):
        c = self.controls
        if path == "/api/fire":
            c.request_fire()
        elif path == "/api/effect":
            c.set_effect(str(data["effect"]))
        elif path == "/api/mode":
            c.set_manual(bool(data["manual"]))
        elif path == "/api/servo":
            c.set_servo(bool(data["on"]))
        elif path == "/api/move":
            c.set_move(float(data.get("right", 0)), float(data.get("up", 0)))
        elif path == "/api/tap":
            x, y = float(data["x"]), float(data["y"])
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("x and y are fractions, 0..1")
            snap = c.snapshot()
            fw, fh = snap.get("frame_w"), snap.get("frame_h")
            if not fw:
                raise ValueError("no frame yet")
            c.request_tap((x * fw, y * fh))
        else:
            raise KeyError(path)

    def publish(self, frame):
        """Hand over an RGB frame. Cheap no-op when nobody is watching."""
        if not self.clients:
            return
        now = time.monotonic()
        if now - self._last < self.interval:
            return
        self._last = now
        h, w = frame.shape[:2]
        small = cv2.resize(
            frame, (self.width, h * self.width // w), interpolation=cv2.INTER_LINEAR
        )
        with self._cond:
            self._frame = small
            self._seq += 1
            self._cond.notify_all()

    def _stream(self, handler):
        handler.send_response(200)
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        handler.end_headers()
        with self._cond:
            self.clients += 1
            # Wait for a frame published after this client arrived: before
            # the first client there is none, and an old one would be stale.
            seen = self._seq
        try:
            while True:
                with self._cond:
                    if not self._cond.wait_for(lambda: self._seq != seen, timeout=5):
                        continue
                    frame, seen = self._frame, self._seq
                ok, jpg = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality],
                )
                if not ok:
                    continue
                handler.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                    % len(jpg)
                )
                handler.wfile.write(jpg.tobytes())
                handler.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with self._cond:
                self.clients -= 1
