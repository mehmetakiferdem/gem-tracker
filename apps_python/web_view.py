"""
Browser view of the tracker output: http://<board>:<port>/

The H.264 RTP stream needs a GStreamer receiver or an SDP file on the viewing
side; this needs nothing but a browser. It serves the drawn frames (boxes,
badge, trails) as MJPEG, downscaled and rate limited so it stays cheap:

  - publish() is called from the pipeline thread on every frame. It returns
    immediately unless a browser is connected and the frame interval has
    passed, and even then only downscales (~2 ms at 640x360). JPEG encoding
    happens in the HTTP thread, and cv2 releases the GIL while encoding.
  - With nobody watching it costs one time comparison per frame.
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Gemstone Tracker</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { margin: 0; background: #111; color: #ddd; font: 14px sans-serif; }
  img { display: block; width: 100%; max-width: 1280px; margin: 0 auto; }
  p { text-align: center; }
</style></head>
<body><img src="/stream" alt="camera"><p>Gemstone O1 tracker</p></body></html>
"""


class WebView:
    def __init__(self, port=8080, width=640, max_fps=15, quality=70):
        self.width = width
        self.interval = 1.0 / max_fps
        self.quality = quality
        self.clients = 0
        self._frame = None
        self._seq = 0
        self._last = 0.0
        self._cond = threading.Condition()

        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(PAGE)))
                    self.end_headers()
                    self.wfile.write(PAGE)
                elif self.path.startswith("/stream"):
                    view._stream(self)
                else:
                    self.send_error(404)

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print("[INFO] Web view on http://<board>:%d/" % port)

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
