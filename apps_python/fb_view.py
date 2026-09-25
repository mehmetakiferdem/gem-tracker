"""
Tracker output on a local panel through the Linux framebuffer.

The Waveshare 5" DSI panel is 720x1280, natively portrait, and the tracker
draws 1280x720 landscape frames. kmssink cannot rotate and the TIOVX output
path has no rotator, so the drawn frame is rotated and written to /dev/fb0
here instead, from a thread of its own:

  - publish() only stores a reference to the latest frame, so the pipeline
    thread never waits on the panel.
  - The writer converts RGB to the framebuffer's XRGB byte order, rotates and
    copies, ~10 ms per frame at 1280x720; cv2 releases the GIL meanwhile.
  - While it runs, the console on tty1 is switched to graphics mode so fbcon
    does not draw its text and cursor over the picture; it is switched back
    on close().
"""

import fcntl
import mmap
import os
import threading

import cv2

KDSETMODE = 0x4B3A
KD_TEXT, KD_GRAPHICS = 0, 1

ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _read(path):
    with open(path) as f:
        return f.read().strip()


class FbView:
    def __init__(self, device="/dev/fb0", rotate=90, tty="/dev/tty1"):
        if rotate not in ROTATIONS:
            raise ValueError("display_rotate must be one of 0, 90, 180, 270")
        name = os.path.basename(device)
        sysfs = "/sys/class/graphics/%s" % name
        self.width, self.height = (int(v) for v in _read(sysfs + "/virtual_size").split(","))
        bpp = int(_read(sysfs + "/bits_per_pixel"))
        if bpp != 32:
            raise RuntimeError("%s is %d bpp, only 32 bpp is supported" % (device, bpp))
        self.stride = int(_read(sysfs + "/stride"))
        self.rotate = ROTATIONS[rotate]
        self.rotate_deg = rotate
        # Where the last frame landed on the panel, for panel_to_frame():
        # (frame_w, frame_h, scale, scaled_w, scaled_h, x0, y0).
        self._geom = None
        # Optional on-screen controls, drawn in panel pixels; see panel_ui.py.
        self.ui = None
        # Optional picture filter applied to the scaled frame on this thread,
        # before the controls are drawn (hud.Hud.filter). Kept off the
        # pipeline thread: at 1280x720 it cost 12-20 ms per frame there.
        self.filter = None

        self.fd = os.open(device, os.O_RDWR)
        self.fb = mmap.mmap(self.fd, self.stride * self.height)

        # Target size before rotation, so that it fills the panel after it.
        if rotate in (90, 270):
            self.fit = (self.height, self.width)
        else:
            self.fit = (self.width, self.height)

        self.tty_fd = None
        try:
            self.tty_fd = os.open(tty, os.O_RDWR)
            fcntl.ioctl(self.tty_fd, KDSETMODE, KD_GRAPHICS)
        except OSError as exc:
            print("[WARNING] Could not switch %s to graphics mode: %s" % (tty, exc))

        self._frame = None
        self._cond = threading.Condition()
        self._stop = False
        threading.Thread(target=self._writer, daemon=True).start()
        # Give the console back on any normal exit, Ctrl+C / SIGINT included;
        # otherwise tty1 stays in graphics mode and the panel shows the last
        # frame forever.
        import atexit

        atexit.register(self.close)
        print(
            "[INFO] Panel view on %s (%dx%d, rotate %d)"
            % (device, self.width, self.height, rotate)
        )

    def publish(self, frame):
        with self._cond:
            self._frame = frame
            self._cond.notify()

    def _writer(self):
        import numpy as np

        # The frame is written into the mapping row by row through a numpy
        # view, which also handles a stride wider than width * 4.
        view = np.frombuffer(self.fb, np.uint8).reshape(self.height, self.stride)
        view = view[:, : self.width * 4].reshape(self.height, self.width, 4)
        canvas = np.zeros((self.height, self.width, 4), np.uint8)
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._frame is not None or self._stop)
                if self._stop:
                    return
                frame, self._frame = self._frame, None
            h, w = frame.shape[:2]
            fw, fh = self.fit
            scale = 1.0
            if (w, h) != (fw, fh):
                # Letterbox rather than stretch if the aspect ratios differ.
                scale = min(fw / w, fh / h)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            sh, sw = frame.shape[:2]
            if self.filter is not None:
                frame = self.filter(frame)
            out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
            if self.rotate is not None:
                out = cv2.rotate(out, self.rotate)
            oh, ow = out.shape[:2]
            y0, x0 = (self.height - oh) // 2, (self.width - ow) // 2
            if self.ui is not None:
                # Compose off screen so the controls never flicker.
                canvas[...] = 0
                canvas[y0 : y0 + oh, x0 : x0 + ow] = out
                self.ui.draw(canvas)
                view[...] = canvas
            elif (ow, oh) == (self.width, self.height):
                view[...] = out
            else:
                view[y0 : y0 + oh, x0 : x0 + ow] = out
            self._geom = (w, h, scale, sw, sh, x0, y0)

    def panel_to_frame(self, px, py):
        """
        Map a panel pixel (a touch) back to tracker frame pixels, undoing the
        placement, rotation and scaling of the last frame written. None if
        nothing was drawn yet or the point is in the letterbox bars.
        """
        if self._geom is None:
            return None
        w, h, scale, sw, sh, x0, y0 = self._geom
        x, y = px - x0, py - y0
        r = self.rotate_deg
        if r == 0:
            u, v = x, y
        elif r == 90:      # clockwise: (u, v) -> (sh - 1 - v, u)
            u, v = y, sh - 1 - x
        elif r == 180:
            u, v = sw - 1 - x, sh - 1 - y
        else:              # 270, counter clockwise: (u, v) -> (v, sw - 1 - u)
            u, v = sw - 1 - y, x
        if not (0 <= u < sw and 0 <= v < sh):
            return None
        return (u / scale, v / scale)

    def close(self):
        with self._cond:
            self._stop = True
            self._cond.notify()
        if self.tty_fd is not None:
            try:
                fcntl.ioctl(self.tty_fd, KDSETMODE, KD_TEXT)
            except OSError:
                pass
