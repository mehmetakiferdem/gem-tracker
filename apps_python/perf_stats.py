"""
Performance box for the HDMI output: the detector model, frame rate, the
load of each Cortex-A53 core and of the two C7x DSPs that run the detector.

  - A53 load comes from /proc/stat, per core, over the last second.
  - C7x load comes from the TI perf stats of libtivision_apps, the library
    TIDL already loaded and initialized (appInit) in this process; the load
    is the average since the previous read, which resets it. A C7x above
    BUSY_LOAD is marked as the one running inference, so the box shows which
    cores infer_cores really uses.

Sampled once a second on a thread of its own; draw() only reads the result.
"""

import ctypes
import threading
import time

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
# BGRA, the KMS output images are BGRA.
WHITE = (255, 255, 255, 255)
DIM = (160, 160, 160, 255)
GREEN = (80, 220, 80, 255)
AMBER = (0, 190, 255, 255)
RED = (60, 60, 240, 255)
BAR_BG = (70, 70, 70, 255)

# j722s CPU ids, app_ipc.h (SOC_J722S).
C7X_CPUS = ((3, "C7X-1"), (4, "C7X-2"))
BUSY_LOAD = 5.0


class _CpuLoad(ctypes.Structure):
    _fields_ = [("cpu_load", ctypes.c_uint32), ("hwi_load", ctypes.c_uint32),
                ("swi_load", ctypes.c_uint32)]


def _proc_stat():
    """{core index: (busy, total)} jiffies from /proc/stat."""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            name, *vals = line.split()
            vals = [int(v) for v in vals]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            out[int(name[3:])] = (sum(vals) - idle, sum(vals))
    return out


class PerfStats:
    INTERVAL = 1.0

    def __init__(self, model_name=""):
        self.model_name = model_name
        self.a53 = {}           # core -> %
        self.c7x = {}           # name -> %, only the cores that answered
        self._lib = None
        try:
            lib = ctypes.CDLL("libtivision_apps.so")
            lib.appIpcIsCpuEnabled.argtypes = [ctypes.c_uint32]
            lib.appPerfStatsCpuLoadGet.argtypes = [ctypes.c_uint32, ctypes.POINTER(_CpuLoad)]
            self._lib = lib
        except OSError as exc:
            print("[WARNING] C7x load not shown: %s" % exc)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        prev = _proc_stat()
        while True:
            time.sleep(self.INTERVAL)
            cur = _proc_stat()
            a53 = {}
            for core, (busy, total) in cur.items():
                pb, pt = prev.get(core, (busy, total))
                a53[core] = 100.0 * (busy - pb) / (total - pt) if total > pt else 0.0
            prev = cur
            self.a53 = a53
            if self._lib is not None:
                self.c7x = self._read_c7x()

    def _read_c7x(self):
        c7x = {}
        try:
            for cpu_id, name in C7X_CPUS:
                if not self._lib.appIpcIsCpuEnabled(cpu_id):
                    continue
                load = _CpuLoad()
                if self._lib.appPerfStatsCpuLoadGet(cpu_id, ctypes.byref(load)) == 0:
                    c7x[name] = load.cpu_load / 100.0
            self._lib.appPerfStatsResetAll()
        except Exception as exc:
            print("[WARNING] C7x load read failed, not shown any more: %s" % exc)
            self._lib = None
        return c7x

    def draw(self, img, fps):
        """Draw the box at the top right of a BGRA image, in place."""
        h, w = img.shape[:2]
        k = w / 1280.0
        line = int(26 * k)
        scale, th = 0.55 * k, max(1, int(round(1.5 * k)))
        name_scale = 0.45 * k
        rows = 1 + len(self.a53) + len(self.c7x) + (2 if self.model_name else 0)
        bw = int(290 * k)
        if self.model_name:
            # The box grows to fit the model name, which can be long.
            (nw, _), _ = cv2.getTextSize(self.model_name, FONT, name_scale, th)
            bw = min(w - int(48 * k), max(bw, nw + int(20 * k)))
        bh = int(14 * k) + rows * line
        x0, y0 = w - bw - int(24 * k), int(80 * k)
        roi = img[y0 : y0 + bh, x0 : x0 + bw]
        roi[...] = roi // 3            # darken, so the text reads on anything

        tx, bx, bar_w = x0 + int(10 * k), x0 + int(130 * k), int(100 * k)
        y = y0 + line

        if self.model_name:
            cv2.putText(img, "MODEL", (tx, y), FONT, scale, DIM, th)
            y += line
            cv2.putText(img, self.model_name, (tx, y), FONT, name_scale, WHITE, th)
            y += line

        cv2.putText(img, "FPS", (tx, y), FONT, scale, DIM, th)
        cv2.putText(img, "%.1f" % fps, (bx, y), FONT, scale * 1.1, WHITE, th + 1)

        def row(label, pct, note=""):
            nonlocal y
            y += line
            color = GREEN if pct < 60 else (AMBER if pct < 85 else RED)
            cv2.putText(img, label, (tx, y), FONT, scale, WHITE if note else DIM, th)
            top = y - int(13 * k)
            cv2.rectangle(img, (bx, top), (bx + bar_w, y), BAR_BG, -1)
            cv2.rectangle(img, (bx, top), (bx + int(bar_w * min(pct, 100) / 100), y), color, -1)
            cv2.putText(img, "%3.0f%%" % pct, (bx + bar_w + int(8 * k), y), FONT, scale, WHITE, th)
            if note:
                cv2.putText(img, note, (tx + int(62 * k), y), FONT, scale * 0.7, AMBER, th)

        for core in sorted(self.a53):
            row("A53-%d" % core, self.a53[core])
        for name, pct in self.c7x.items():
            row(name, pct, "AI" if pct >= BUSY_LOAD else "")
