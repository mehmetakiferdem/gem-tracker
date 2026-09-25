"""
Taps on the panel's touchscreen, read straight from evdev.

Reports on_down(x, y) where a finger lands and on_up() when it lifts, from
this module's own thread, so a button can act for as long as it is held. Only single touch is used:
the multi-touch position codes are read too, since some controllers (the
FT5x06 of the Waveshare 5" DSI LCD among them) report both.
"""

import glob
import os
import struct
import threading

EV_SYN, EV_KEY, EV_ABS = 0, 1, 3
BTN_TOUCH = 330
ABS_X, ABS_Y, ABS_MT_POSITION_X, ABS_MT_POSITION_Y = 0, 1, 53, 54
EVENT = struct.Struct("llHHi")


def find_touch_device(hint="ft5x06"):
    """/dev/input/eventN of the first input device whose name contains hint."""
    for name_file in sorted(glob.glob("/sys/class/input/event*/device/name")):
        with open(name_file) as f:
            if hint in f.read():
                return "/dev/input/" + name_file.split("/")[4]
    return None


class TouchInput:
    def __init__(self, on_down, on_up=None, device="auto"):
        if device == "auto":
            device = find_touch_device()
        if not device:
            raise RuntimeError("no touchscreen found")
        self.device = device
        self.on_down = on_down
        self.on_up = on_up or (lambda: None)
        self.fd = os.open(device, os.O_RDONLY)
        threading.Thread(target=self._reader, daemon=True).start()
        print("[INFO] Touch input on %s" % device)

    def _reader(self):
        x = y = None
        pending = False
        while True:
            data = os.read(self.fd, EVENT.size * 64)
            for off in range(0, len(data) - EVENT.size + 1, EVENT.size):
                _, _, typ, code, val = EVENT.unpack_from(data, off)
                if typ == EV_ABS and code in (ABS_X, ABS_MT_POSITION_X):
                    x = val
                elif typ == EV_ABS and code in (ABS_Y, ABS_MT_POSITION_Y):
                    y = val
                elif typ == EV_KEY and code == BTN_TOUCH:
                    if val == 1:
                        pending = True
                    else:
                        pending = False
                        self._call(self.on_up)
                elif typ == EV_SYN and pending and x is not None and y is not None:
                    pending = False
                    self._call(self.on_down, x, y)

    @staticmethod
    def _call(fn, *args):
        try:
            fn(*args)
        except Exception as exc:
            print("[WARNING] touch handler: %s" % exc)
