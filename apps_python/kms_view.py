"""
Tracker output on every connected display through DRM/KMS.

fbdev gives all outputs a single framebuffer sized to the largest one, and
each CRTC scans it from (0, 0): with the 800x480 DSI panel and a 1366x768 TV
both connected, fb0 is 1366x768 and the panel shows only its top left corner.
Here each connected output gets a dumb buffer of its own at its own mode, so
the picture fills both.

Same interface as fb_view.FbView (publish, width/height, ui, filter,
panel_to_frame, close). The "panel" is the first DSI output, or the first
connected output if there is no DSI; width/height, rotation, the touch
controls (ui) and panel_to_frame() refer to it. The other outputs get the
filtered picture without the controls.

A display connected later (or plugged back in) is picked up within
POLL_INTERVAL seconds from the connector status the kernel keeps in sysfs.
On close the device is released and the kernel puts
the fbdev console back on the screens.
"""

import ctypes
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

# ------------------------------------------------------------ DRM uapi

u16, u32, u64 = ctypes.c_uint16, ctypes.c_uint32, ctypes.c_uint64


class ModeInfo(ctypes.Structure):
    _fields_ = [
        ("clock", u32),
        ("hdisplay", u16), ("hsync_start", u16), ("hsync_end", u16), ("htotal", u16), ("hskew", u16),
        ("vdisplay", u16), ("vsync_start", u16), ("vsync_end", u16), ("vtotal", u16), ("vscan", u16),
        ("vrefresh", u32), ("flags", u32), ("type", u32),
        ("name", ctypes.c_char * 32),
    ]


class CardRes(ctypes.Structure):
    _fields_ = [
        ("fb_id_ptr", u64), ("crtc_id_ptr", u64), ("connector_id_ptr", u64), ("encoder_id_ptr", u64),
        ("count_fbs", u32), ("count_crtcs", u32), ("count_connectors", u32), ("count_encoders", u32),
        ("min_width", u32), ("max_width", u32), ("min_height", u32), ("max_height", u32),
    ]


class GetConnector(ctypes.Structure):
    _fields_ = [
        ("encoders_ptr", u64), ("modes_ptr", u64), ("props_ptr", u64), ("prop_values_ptr", u64),
        ("count_modes", u32), ("count_props", u32), ("count_encoders", u32),
        ("encoder_id", u32), ("connector_id", u32), ("connector_type", u32), ("connector_type_id", u32),
        ("connection", u32), ("mm_width", u32), ("mm_height", u32), ("subpixel", u32),
        ("pad", u32),
    ]


class GetEncoder(ctypes.Structure):
    _fields_ = [
        ("encoder_id", u32), ("encoder_type", u32), ("crtc_id", u32),
        ("possible_crtcs", u32), ("possible_clones", u32),
    ]


class Crtc(ctypes.Structure):
    _fields_ = [
        ("set_connectors_ptr", u64), ("count_connectors", u32),
        ("crtc_id", u32), ("fb_id", u32), ("x", u32), ("y", u32),
        ("gamma_size", u32), ("mode_valid", u32), ("mode", ModeInfo),
    ]


class CreateDumb(ctypes.Structure):
    _fields_ = [
        ("height", u32), ("width", u32), ("bpp", u32), ("flags", u32),
        ("handle", u32), ("pitch", u32), ("size", u64),
    ]


class MapDumb(ctypes.Structure):
    _fields_ = [("handle", u32), ("pad", u32), ("offset", u64)]


class FbCmd(ctypes.Structure):
    _fields_ = [
        ("fb_id", u32), ("width", u32), ("height", u32), ("pitch", u32),
        ("bpp", u32), ("depth", u32), ("handle", u32),
    ]


def _iowr(nr, struct):
    return (3 << 30) | (ctypes.sizeof(struct) << 16) | (ord("d") << 8) | nr


DRM_IOCTL_SET_MASTER = (ord("d") << 8) | 0x1E
DRM_IOCTL_MODE_GETRESOURCES = _iowr(0xA0, CardRes)
DRM_IOCTL_MODE_GETCRTC = _iowr(0xA1, Crtc)
DRM_IOCTL_MODE_SETCRTC = _iowr(0xA2, Crtc)
DRM_IOCTL_MODE_GETENCODER = _iowr(0xA6, GetEncoder)
DRM_IOCTL_MODE_GETCONNECTOR = _iowr(0xA7, GetConnector)
DRM_IOCTL_MODE_ADDFB = _iowr(0xAE, FbCmd)
DRM_IOCTL_MODE_RMFB = _iowr(0xAF, u32)
DRM_IOCTL_MODE_CREATE_DUMB = _iowr(0xB2, CreateDumb)
DRM_IOCTL_MODE_MAP_DUMB = _iowr(0xB3, MapDumb)

DRM_MODE_CONNECTED = 1
DRM_MODE_TYPE_PREFERRED = 1 << 3
CONNECTOR_NAMES = {
    1: "VGA", 2: "DVI-I", 3: "DVI-D", 4: "DVI-A", 5: "Composite", 6: "SVIDEO",
    7: "LVDS", 8: "Component", 9: "DIN", 10: "DP", 11: "HDMI-A", 12: "HDMI-B",
    13: "TV", 14: "eDP", 15: "Virtual", 16: "DSI", 17: "DPI", 18: "Writeback",
}


def _array(ctype, n):
    arr = (ctype * max(n, 1))()
    return arr, ctypes.addressof(arr)


class _Output:
    """One connector driven at its mode from a dumb buffer of its own."""

    def __init__(self, name, width, height, pitch, fb, rotate):
        import numpy as np

        self.name = name
        self.width, self.height = width, height
        self.rotate_deg = rotate
        self.rotate = ROTATIONS[rotate]
        # Target size before rotation, so that it fills the output after it.
        self.fit = (height, width) if rotate in (90, 270) else (width, height)
        self.fb = fb
        view = np.frombuffer(fb, np.uint8).reshape(height, pitch)
        self.view = view[:, : width * 4].reshape(height, width, 4)
        self.canvas = np.zeros((height, width, 4), np.uint8)


class KmsView:
    # Seconds between hotplug checks, see _poll().
    POLL_INTERVAL = 2.0

    def __init__(self, device="/dev/dri/card0", rotate=0, tty="/dev/tty1"):
        if rotate not in ROTATIONS:
            raise ValueError("display_rotate must be one of 0, 90, 180, 270")
        self.rotate_deg = rotate
        # Same meaning as in FbView, for the panel output.
        self._geom = None
        self.ui = None
        self.filter = None
        # Optional drawing for the other outputs only (the HDMI TV), called
        # with the BGRA image in place after the filter; see perf_stats.py.
        self.overlay = None

        self._card = os.path.basename(os.path.realpath(device))
        self.fd = os.open(device, os.O_RDWR | os.O_CLOEXEC)
        try:
            self._setup()
        except Exception:
            os.close(self.fd)
            raise
        self.panel = self.outputs[0]
        self.width, self.height = self.panel.width, self.panel.height

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
        import atexit

        atexit.register(self.close)
        print(
            "[INFO] KMS view on %s: %s"
            % (
                device,
                ", ".join(
                    "%s %dx%d%s" % (o.name, o.width, o.height,
                                    " rotate %d" % o.rotate_deg if o.rotate_deg else "")
                    for o in self.outputs
                ),
            )
        )

    # ------------------------------------------------------------ setup

    def _ioctl(self, req, arg):
        fcntl.ioctl(self.fd, req, arg, True)
        return arg

    def _setup(self):
        try:
            fcntl.ioctl(self.fd, DRM_IOCTL_SET_MASTER)
        except OSError:
            # The first opener is master already; anything else shows up as
            # EACCES on SETCRTC below.
            pass

        res = self._ioctl(DRM_IOCTL_MODE_GETRESOURCES, CardRes())
        crtcs, res.crtc_id_ptr = _array(u32, res.count_crtcs)
        conns, res.connector_id_ptr = _array(u32, res.count_connectors)
        # Every array must stay referenced until the ioctl has filled it.
        fbs, res.fb_id_ptr = _array(u32, res.count_fbs)
        encs, res.encoder_id_ptr = _array(u32, res.count_encoders)
        self._ioctl(DRM_IOCTL_MODE_GETRESOURCES, res)
        self._crtc_ids = list(crtcs[: res.count_crtcs])
        self._conn_ids = list(conns[: res.count_connectors])
        # connector id -> _Output, and connector id -> sysfs status file,
        # read by _poll() for hotplug (it holds the status the kernel probed
        # last; reading it does not touch the display).
        self._driven = {}
        self._status = {}
        self._last = {}
        self.outputs = []

        found = []
        for cid in self._conn_ids:
            c = self._connector(cid)
            if c is not None:
                found.append(c)
        if not found:
            raise RuntimeError("no connected display")
        # The panel first: DSI, then the rest in kernel order.
        found.sort(key=lambda c: 0 if c[1].startswith("DSI") else 1)
        for c in found:
            self._drive(*c)
        if not self.outputs:
            raise RuntimeError("no display could be driven")
        for cid in self._conn_ids:
            self._last[cid] = self._read_status(cid)
        return self.outputs

    def _drive(self, conn, name, modes):
        """Put the connector on a free CRTC at its mode with a buffer of its own."""
        used = {o.crtc_id for o in self._driven.values()}
        crtc_id = self._pick_crtc(conn, self._crtc_ids, used)
        if crtc_id is None:
            print("[WARNING] No free CRTC for %s, not driven" % name)
            return None
        mode = self._pick_mode(crtc_id, modes)
        w, h = mode.hdisplay, mode.vdisplay

        dumb = self._ioctl(DRM_IOCTL_MODE_CREATE_DUMB, CreateDumb(height=h, width=w, bpp=32))
        fbc = self._ioctl(
            DRM_IOCTL_MODE_ADDFB,
            FbCmd(width=w, height=h, pitch=dumb.pitch, bpp=32, depth=24, handle=dumb.handle),
        )
        mp = self._ioctl(DRM_IOCTL_MODE_MAP_DUMB, MapDumb(handle=dumb.handle))
        fb = mmap.mmap(self.fd, dumb.size, mmap.MAP_SHARED,
                       mmap.PROT_READ | mmap.PROT_WRITE, offset=mp.offset)

        ids = (u32 * 1)(conn.connector_id)
        self._ioctl(
            DRM_IOCTL_MODE_SETCRTC,
            Crtc(set_connectors_ptr=ctypes.addressof(ids), count_connectors=1,
                 crtc_id=crtc_id, fb_id=fbc.fb_id, mode_valid=1, mode=mode),
        )
        # Rotation and the touch controls belong to the first output only.
        rotate = self.rotate_deg if not self.outputs else 0
        out = _Output(name, w, h, dumb.pitch, fb, rotate)
        out.crtc_id, out.fb_id = crtc_id, fbc.fb_id
        self.outputs.append(out)
        self._driven[conn.connector_id] = out
        return out

    def _read_status(self, cid):
        path = self._status.get(cid)
        if path is None:
            return None
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            return None

    def _poll(self):
        """
        Hotplug: a display connected after start is set up, one plugged back
        in is set up again (its mode may differ, or the sink needs the
        modeset again). The panel output is never replaced.
        """
        for cid in self._conn_ids:
            now = self._read_status(cid)
            was, self._last[cid] = self._last.get(cid), now
            if now != "connected" or was == "connected":
                continue
            old = self._driven.get(cid)
            if old is self.panel:
                continue
            c = self._connector(cid)
            if c is None:
                # Not ready yet (no modes); try again on the next poll.
                self._last[cid] = was
                continue
            if old is not None:
                self.outputs.remove(old)
                del self._driven[cid]
                try:
                    fcntl.ioctl(self.fd, DRM_IOCTL_MODE_RMFB, u32(old.fb_id))
                except OSError:
                    pass
            try:
                out = self._drive(*c)
            except OSError as exc:
                print("[WARNING] Could not drive %s: %s" % (c[1], exc), flush=True)
                continue
            if out is not None:
                print("[INFO] KMS view: %s connected, %dx%d"
                      % (out.name, out.width, out.height), flush=True)

    def _connector(self, cid):
        """(connector, name, modes) if connected with modes, else None. Probes."""
        c = self._ioctl(DRM_IOCTL_MODE_GETCONNECTOR, GetConnector(connector_id=cid))
        name = "%s-%d" % (CONNECTOR_NAMES.get(c.connector_type, "Unknown"), c.connector_type_id)
        self._status[cid] = "/sys/class/drm/%s-%s/status" % (self._card, name)
        if c.connection != DRM_MODE_CONNECTED or not c.count_modes:
            return None
        # The counts may change between the two calls on a hotplug; the
        # kernel then fills only what fits, which is fine here.
        modes, c.modes_ptr = _array(ModeInfo, c.count_modes)
        encs, c.encoders_ptr = _array(u32, c.count_encoders)
        props, c.props_ptr = _array(u32, c.count_props)
        values, c.prop_values_ptr = _array(u64, c.count_props)
        self._ioctl(DRM_IOCTL_MODE_GETCONNECTOR, c)
        c._encoders = list(encs[: c.count_encoders])
        return c, name, list(modes[: c.count_modes])

    def _pick_crtc(self, conn, crtc_ids, used):
        # Keep the CRTC fbdev already put this connector on, if any.
        if conn.encoder_id:
            enc = self._ioctl(DRM_IOCTL_MODE_GETENCODER, GetEncoder(encoder_id=conn.encoder_id))
            if enc.crtc_id and enc.crtc_id not in used:
                return enc.crtc_id
        for eid in conn._encoders:
            enc = self._ioctl(DRM_IOCTL_MODE_GETENCODER, GetEncoder(encoder_id=eid))
            for i, crtc_id in enumerate(crtc_ids):
                if enc.possible_crtcs & (1 << i) and crtc_id not in used:
                    return crtc_id
        return None

    def _pick_mode(self, crtc_id, modes):
        # The display's preferred mode. fbdev may have left a smaller one
        # running (it sizes one framebuffer for all outputs), so that one is
        # only the fallback, then the first mode offered.
        for m in modes:
            if m.type & DRM_MODE_TYPE_PREFERRED:
                return m
        cur = self._ioctl(DRM_IOCTL_MODE_GETCRTC, Crtc(crtc_id=crtc_id))
        if cur.mode_valid:
            for m in modes:
                if (m.hdisplay, m.vdisplay, m.clock) == (
                    cur.mode.hdisplay, cur.mode.vdisplay, cur.mode.clock):
                    return m
        return modes[0]

    # ------------------------------------------------------------ drawing

    def publish(self, frame):
        with self._cond:
            self._frame = frame
            self._cond.notify()

    def _writer(self):
        import time

        next_poll = time.monotonic() + self.POLL_INTERVAL
        frame = None
        while True:
            with self._cond:
                self._cond.wait_for(
                    lambda: self._frame is not None or self._stop,
                    timeout=max(0.0, next_poll - time.monotonic()),
                )
                if self._stop:
                    return
                if self._frame is not None:
                    frame, self._frame = self._frame, None
                else:
                    frame = None
            # Outputs are only added or replaced here, on this thread.
            if time.monotonic() >= next_poll:
                next_poll = time.monotonic() + self.POLL_INTERVAL
                self._poll()
            if frame is not None:
                for out in self.outputs:
                    self._draw(out, frame)

    def _draw(self, out, frame):
        h, w = frame.shape[:2]
        fw, fh = out.fit
        scale = 1.0
        if (w, h) != (fw, fh):
            # Letterbox rather than stretch if the aspect ratios differ.
            scale = min(fw / w, fh / h)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        sh, sw = frame.shape[:2]
        if self.filter is not None:
            frame = self.filter(frame)
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
        if out.rotate is not None:
            img = cv2.rotate(img, out.rotate)
        if out is not self.panel and self.overlay is not None:
            self.overlay(img)
        oh, ow = img.shape[:2]
        y0, x0 = (out.height - oh) // 2, (out.width - ow) // 2
        if out is self.panel and self.ui is not None:
            # Compose off screen so the controls never flicker.
            out.canvas[...] = 0
            out.canvas[y0 : y0 + oh, x0 : x0 + ow] = img
            self.ui.draw(out.canvas)
            out.view[...] = out.canvas
        elif (ow, oh) == (out.width, out.height):
            out.view[...] = img
        else:
            out.view[y0 : y0 + oh, x0 : x0 + ow] = img
        if out is self.panel:
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
            if self._stop:
                return
            self._stop = True
            self._cond.notify()
        if self.tty_fd is not None:
            try:
                fcntl.ioctl(self.tty_fd, KDSETMODE, KD_TEXT)
            except OSError:
                pass
        # The mappings keep the device open until the process exits; then
        # the kernel lets fbdev restore the console on the screens.
        try:
            os.close(self.fd)
        except OSError:
            pass
