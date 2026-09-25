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
                        effect on the picture (tracker side, see hud.py).
  Round buttons (right edge)
                        Picture style, see hud.py: SADE, NISAN, GECE, TERMAL.
  Anywhere else         Goes to the tracker as a tap on the picture: a face
                        locks onto it (and returns to AUTO), empty space drops
                        the lock.

Touches arrive on the touch thread; this only records state (the mode, the
held direction) and hands picture taps to on_picture_tap. The tracker reads
the state once per frame.
"""

import cv2
import numpy as np

BTN = 64      # D-pad button size, px
GAP = 6
EFFECT_R = 27  # radius of the round effect buttons
FIRE_R = 40
EFFECTS = (("plain", "SADE"), ("aim", "NISAN"), ("night", "GECE"), ("thermal", "TERMAL"))


class PanelUI:
    def __init__(self, width, height, on_picture_tap, effect="aim"):
        self.on_picture_tap = on_picture_tap
        self.effect = effect
        # Round effect buttons down the right edge, above the D-pad.
        self.effect_buttons = [
            (name, label, (width - 16 - EFFECT_R, 16 + EFFECT_R + i * (2 * EFFECT_R + 8)))
            for i, (name, label) in enumerate(EFFECTS)
        ]
        self.manual = False
        # Held direction, (right, up) in {-1, 0, 1}.
        self.move = (0, 0)
        self.pressed = None

        cx, cy = width - 16 - BTN - BTN // 2 - GAP, height - 16 - BTN - BTN // 2 - GAP

        def box(x, y):
            return (x - BTN // 2, y - BTN // 2, x + BTN // 2, y + BTN // 2)

        step = BTN + GAP
        self.buttons = {
            "up": (box(cx, cy - step), (0, 1)),
            "down": (box(cx, cy + step), (0, -1)),
            "left": (box(cx - step, cy), (-1, 0)),
            "right": (box(cx + step, cy), (1, 0)),
        }
        self.mode_button = (16, height - 16 - 56, 16 + 150, height - 16)
        self.fire_center = (16 + 150 + 24 + FIRE_R, height - 16 - FIRE_R)
        # Set here on a press, consumed by the tracker on its next frame.
        self.fire_request = False

    # ---------------------------------------------------------------- touch

    @staticmethod
    def _inside(rect, x, y):
        return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

    def down(self, x, y):
        for name, (rect, move) in self.buttons.items():
            if self._inside(rect, x, y):
                self.pressed = name
                self.manual = True
                self.move = move
                return
        for name, _, (ex, ey) in self.effect_buttons:
            if (x - ex) ** 2 + (y - ey) ** 2 <= (EFFECT_R + 6) ** 2:
                self.pressed = "fx:" + name
                self.effect = name
                return
        fx, fy = self.fire_center
        if (x - fx) ** 2 + (y - fy) ** 2 <= (FIRE_R + 8) ** 2:
            self.pressed = "fire"
            self.fire_request = True
            return
        if self._inside(self.mode_button, x, y):
            self.pressed = "mode"
            self.manual = not self.manual
            self.move = (0, 0)
            return
        self.on_picture_tap(x, y)

    def up(self):
        self.pressed = None
        self.move = (0, 0)

    # ----------------------------------------------------------------- draw

    def draw(self, canvas):
        """Draw the controls onto a BGRA panel canvas, in place."""
        overlay = canvas.copy()
        for name, (rect, _) in self.buttons.items():
            x1, y1, x2, y2 = rect
            fill = (0, 200, 255, 255) if self.pressed == name else (60, 60, 60, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
        mx1, my1, mx2, my2 = self.mode_button
        mode_fill = (0, 140, 255, 255) if self.manual else (0, 160, 0, 255)
        cv2.rectangle(overlay, (mx1, my1), (mx2, my2), mode_fill, -1)
        for name, _, (ex, ey) in self.effect_buttons:
            fill = (0, 170, 255, 255) if name == self.effect else (60, 60, 60, 255)
            cv2.circle(overlay, (ex, ey), EFFECT_R, fill, -1)
        fire_fill = (0, 220, 255, 255) if self.pressed == "fire" else (30, 30, 220, 255)
        cv2.circle(overlay, self.fire_center, FIRE_R, fire_fill, -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, dst=canvas)

        fx, fy = self.fire_center
        cv2.circle(canvas, (fx, fy), FIRE_R, (255, 255, 255, 255), 3)
        cv2.circle(canvas, (fx, fy), FIRE_R - 12, (255, 255, 255, 255), 1)
        (tw, th), _ = cv2.getTextSize("ATES", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(canvas, "ATES", (fx - tw // 2, fy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255, 255), 2)

        for name, label, (ex, ey) in self.effect_buttons:
            ring = (255, 255, 255, 255)
            cv2.circle(canvas, (ex, ey), EFFECT_R, ring, 3 if name == self.effect else 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            cv2.putText(canvas, label, (ex - tw // 2, ey + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255, 255), 1)

        for name, (rect, _) in self.buttons.items():
            x1, y1, x2, y2 = rect
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255, 255), 2)
            cx, cy, r = (x1 + x2) // 2, (y1 + y2) // 2, BTN // 4
            tri = {
                "up": [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)],
                "down": [(cx, cy + r), (cx - r, cy - r), (cx + r, cy - r)],
                "left": [(cx - r, cy), (cx + r, cy - r), (cx + r, cy + r)],
                "right": [(cx + r, cy), (cx - r, cy - r), (cx - r, cy + r)],
            }[name]
            cv2.fillPoly(canvas, [np.array(tri, np.int32)], (255, 255, 255, 255))

        cv2.rectangle(canvas, (mx1, my1), (mx2, my2), (255, 255, 255, 255), 2)
        text = "MANUEL" if self.manual else "OTO"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(
            canvas,
            text,
            ((mx1 + mx2 - tw) // 2, (my1 + my2 + th) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255, 255),
            2,
        )
