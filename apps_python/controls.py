"""
Operator controls shared by every front end: the touch panel (panel_ui.py)
and the browser (web_view.py) write here, the tracker reads once per frame.

Front ends run on their own threads (touch reader, HTTP handlers), so every
field goes through the lock. Requests that must happen exactly once (a shot,
a tap on the picture) are queued and taken by the tracker; states that last
(mode, held direction, picture style) are plain values.

A held direction from the browser expires on its own after MOVE_TIMEOUT: a
page that loses its connection mid press must not leave the camera turning.
The panel refreshes its direction on every touch event, the browser resends
while the button is held.
"""

import threading
import time

from hud import STYLES

MOVE_TIMEOUT = 0.6  # s, a held direction without refresh is dropped


class Controls:
    def __init__(self, effect="aim"):
        self._lock = threading.Lock()
        self._effect = effect if effect in STYLES else "aim"
        self._manual = False
        self._move = (0, 0)
        self._move_time = 0.0
        self._move_expires = True
        self._fire = False
        self._tap = None
        self._servo_on = True
        # Published by the tracker every frame for the front ends to show.
        self._status = {}

    # ----------------------------------------------------------- front ends

    def set_effect(self, name):
        if name not in STYLES:
            raise ValueError("unknown effect '%s'" % name)
        with self._lock:
            self._effect = name

    def set_manual(self, manual):
        with self._lock:
            self._manual = bool(manual)
            if not self._manual:
                self._move = (0, 0)

    def set_move(self, right, up, expires=True):
        """
        Held direction, right/up each in {-1, 0, 1}. Any direction also
        switches to MANUAL. expires=False for the panel, whose touch thread
        clears it on release itself.
        """
        right = max(-1, min(1, int(right)))
        up = max(-1, min(1, int(up)))
        with self._lock:
            self._move = (right, up)
            self._move_time = time.monotonic()
            self._move_expires = expires
            if right or up:
                self._manual = True

    def set_servo(self, on):
        """Servo stop: False cuts the pulses and the rig goes limp."""
        with self._lock:
            self._servo_on = bool(on)
            if not self._servo_on:
                self._move = (0, 0)

    def request_fire(self):
        with self._lock:
            self._fire = True

    def request_tap(self, point):
        """A tap on the picture, in tracker frame pixels."""
        with self._lock:
            self._tap = point

    # --------------------------------------------------------------- tracker

    @property
    def effect(self):
        with self._lock:
            return self._effect

    @property
    def manual(self):
        with self._lock:
            return self._manual

    @property
    def servo_on(self):
        with self._lock:
            return self._servo_on

    @property
    def move(self):
        with self._lock:
            if self._move_expires and time.monotonic() - self._move_time > MOVE_TIMEOUT:
                self._move = (0, 0)
            return self._move

    def take_fire(self):
        with self._lock:
            fire, self._fire = self._fire, False
            return fire

    def take_tap(self):
        with self._lock:
            tap, self._tap = self._tap, None
            return tap

    def publish_status(self, status):
        with self._lock:
            self._status = status

    def snapshot(self):
        """Everything a front end shows, one consistent copy."""
        with self._lock:
            s = dict(self._status)
            s.update(effect=self._effect, manual=self._manual, servo_on=self._servo_on,
                     effects=list(STYLES))
            return s
