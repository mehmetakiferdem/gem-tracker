"""
On-screen controls for the local panel, drawn in panel pixels by FbView on
top of the tracker picture, so they never reach the web or RTP streams.

  D-pad (bottom right)  Hold to move the camera; switches to MANUAL, where the
                        tracker keeps detecting and drawing but no longer
                        steers or searches.
  AUTO / MANUAL (bottom left)
                        Shows the mode; tap it to switch back to automatic
                        tracking.
  ATES (bottom, red)    Fire: a recoil kick on the servos plus a hit / miss
                        effect on the picture (fire.py, hud.py). A ring fills
                        up while the gun cools down.
  SERVO (above AUTO)    Servo stop: cuts the pulses, the rig goes limp and
                        stays so in every mode until it is tapped again.
  Round buttons (right edge)
                        Picture style, see hud.py: SADE, NISAN, GECE, TERMAL.
  Anywhere else         Goes to the tracker as a tap on the picture: a face
                        locks onto it (and returns to AUTO), empty space drops
                        the lock.

Sizes follow the panel height (480 px is the reference), so the same layout
fits the 5" and the 7" panels. All state lives in controls.Controls, shared
with the browser view; this class only maps touches to it and draws.
"""

import cv2
import numpy as np

EFFECTS = (("plain", "SADE"), ("aim", "NISAN"), ("night", "GECE"), ("thermal", "TERMAL"))
FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGRA, the panel canvas is BGRA.
WHITE = (255, 255, 255, 255)
IDLE = (60, 60, 60, 255)
ACTIVE = (0, 170, 255, 255)
PRESSED = (0, 220, 255, 255)
FIRE = (30, 30, 220, 255)
AUTO = (0, 160, 0, 255)
MANUAL = (0, 140, 255, 255)


class PanelUI:
    def __init__(self, width, height, controls, on_picture_tap):
        self.controls = controls
        self.on_picture_tap = on_picture_tap
        self.pressed = None

        s = height / 480.0
        self.s = s
        m = int(16 * s)
        btn, gap = int(64 * s), int(6 * s)
        self.btn = btn
        self.effect_r = int(27 * s)
        self.fire_r = int(40 * s)

        # Round effect buttons down the right edge, above the D-pad.
        er = self.effect_r
        self.effect_buttons = [
            (name, label, (width - m - er, m + er + i * (2 * er + int(8 * s))))
            for i, (name, label) in enumerate(EFFECTS)
        ]

        cx = width - m - btn - btn // 2 - gap
        cy = height - m - btn - btn // 2 - gap

        def box(x, y):
            return (x - btn // 2, y - btn // 2, x + btn // 2, y + btn // 2)

        step = btn + gap
        self.buttons = {
            "up": (box(cx, cy - step), (0, 1)),
            "down": (box(cx, cy + step), (0, -1)),
            "left": (box(cx - step, cy), (-1, 0)),
            "right": (box(cx + step, cy), (1, 0)),
        }
        self.mode_button = (m, height - m - int(56 * s), m + int(150 * s), height - m)
        y1 = self.mode_button[1] - int(10 * s)
        self.servo_button = (m, y1 - int(46 * s), m + int(150 * s), y1)
        self.fire_center = (self.mode_button[2] + int(24 * s) + self.fire_r,
                            height - m - self.fire_r)

    # ---------------------------------------------------------------- touch

    @staticmethod
    def _inside(rect, x, y):
        return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

    def down(self, x, y):
        c = self.controls
        for name, (rect, move) in self.buttons.items():
            if self._inside(rect, x, y):
                self.pressed = name
                c.set_move(*move, expires=False)
                return
        slack = int(6 * self.s)
        for name, _, (ex, ey) in self.effect_buttons:
            if (x - ex) ** 2 + (y - ey) ** 2 <= (self.effect_r + slack) ** 2:
                self.pressed = "fx:" + name
                c.set_effect(name)
                return
        fx, fy = self.fire_center
        if (x - fx) ** 2 + (y - fy) ** 2 <= (self.fire_r + 2 * slack) ** 2:
            self.pressed = "fire"
            c.request_fire()
            return
        if self._inside(self.servo_button, x, y):
            self.pressed = "servo"
            c.set_servo(not c.servo_on)
            return
        if self._inside(self.mode_button, x, y):
            self.pressed = "mode"
            c.set_manual(not c.manual)
            return
        self.on_picture_tap(x, y)

    def up(self):
        if self.pressed in self.buttons:
            self.controls.set_move(0, 0, expires=False)
        self.pressed = None

    # ----------------------------------------------------------------- draw

    def _label(self, canvas, text, center, scale, thickness):
        scale *= self.s
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
        cv2.putText(canvas, text, (center[0] - tw // 2, center[1] + th // 2),
                    FONT, scale, WHITE, thickness, cv2.LINE_AA)

    def draw(self, canvas):
        """Draw the controls onto a BGRA panel canvas, in place."""
        snap = self.controls.snapshot()
        effect, manual = snap["effect"], snap["manual"]
        servo_on = snap.get("servo_on", True)
        cooldown = snap.get("cooldown", 0.0)

        overlay = canvas.copy()
        for name, (rect, _) in self.buttons.items():
            x1, y1, x2, y2 = rect
            cv2.rectangle(overlay, (x1, y1), (x2, y2),
                          PRESSED if self.pressed == name else IDLE, -1)
        mx1, my1, mx2, my2 = self.mode_button
        cv2.rectangle(overlay, (mx1, my1), (mx2, my2), MANUAL if manual else AUTO, -1)
        sx1, sy1, sx2, sy2 = self.servo_button
        cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), FIRE if servo_on else IDLE, -1)
        for name, _, (ex, ey) in self.effect_buttons:
            cv2.circle(overlay, (ex, ey), self.effect_r,
                       ACTIVE if name == effect else IDLE, -1, cv2.LINE_AA)
        cv2.circle(overlay, self.fire_center, self.fire_r,
                   PRESSED if self.pressed == "fire" else FIRE, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, dst=canvas)

        # Fire button, with a ring that fills up while the gun cools down.
        fx, fy = self.fire_center
        lw = max(2, int(3 * self.s))
        cv2.circle(canvas, (fx, fy), self.fire_r, WHITE, lw, cv2.LINE_AA)
        if cooldown > 0.0:
            from fire import COOLDOWN
            end = int(360 * (1.0 - cooldown / COOLDOWN))
            r = self.fire_r + lw + int(3 * self.s)
            cv2.ellipse(canvas, (fx, fy), (r, r), -90, 0, end, ACTIVE, lw, cv2.LINE_AA)
        cv2.circle(canvas, (fx, fy), self.fire_r - int(12 * self.s), WHITE, 1, cv2.LINE_AA)
        self._label(canvas, "ATES", (fx, fy), 0.7, 2)
        shots = snap.get("shots")
        if shots:
            self._label(canvas, "%d/%d" % (snap.get("hits", 0), shots),
                        (fx, fy - self.fire_r - int(24 * self.s)), 0.5, 1)

        for name, label, (ex, ey) in self.effect_buttons:
            cv2.circle(canvas, (ex, ey), self.effect_r, WHITE,
                       3 if name == effect else 2, cv2.LINE_AA)
            self._label(canvas, label, (ex, ey), 0.36, 1)

        for name, (rect, _) in self.buttons.items():
            x1, y1, x2, y2 = rect
            cv2.rectangle(canvas, (x1, y1), (x2, y2), WHITE, 2)
            cx, cy, r = (x1 + x2) // 2, (y1 + y2) // 2, self.btn // 4
            tri = {
                "up": [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)],
                "down": [(cx, cy + r), (cx - r, cy - r), (cx + r, cy - r)],
                "left": [(cx - r, cy), (cx + r, cy - r), (cx + r, cy + r)],
                "right": [(cx + r, cy), (cx - r, cy - r), (cx - r, cy + r)],
            }[name]
            cv2.fillPoly(canvas, [np.array(tri, np.int32)], WHITE, cv2.LINE_AA)

        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), WHITE, 2)
        self._label(canvas, "SERVO DUR" if servo_on else "SERVO AC",
                    ((sx1 + sx2) // 2, (sy1 + sy2) // 2), 0.6, 2)
        cv2.rectangle(canvas, (mx1, my1), (mx2, my2), WHITE, 2)
        self._label(canvas, "MANUEL" if manual else "OTO",
                    ((mx1 + mx2) // 2, (my1 + my2) // 2), 0.9, 2)
