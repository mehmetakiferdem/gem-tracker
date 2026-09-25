"""
Visual styles for the tracker picture, picked on the panel (PanelUI effect
buttons) or with the config key hud_style.

  plain    the original look: coloured box and id per target, status badge
  aim      game style HUD: centre crosshair, corner brackets that close in on
           a new lock, a rotating reticle around the target, a line from the
           crosshair to it, screen corner frame and a pan/tilt readout
  night    aim HUD, green night vision picture on the panel
  thermal  aim HUD, false colour "thermal" picture on the panel

The HUD is drawn in place on the RGB frame (~4 ms), so the web and RTP views
show it too. The picture filters are applied by the panel writer thread to
the frame already scaled to the panel (~5 ms at 800x450, in parallel with the
pipeline): at 1280x720 on the pipeline thread they measured 12-20 ms, a big
bite out of the face model's ~67 ms per frame.
"""

import math

import cv2
import numpy as np

STYLES = ("plain", "aim", "night", "thermal")
FONT = cv2.FONT_HERSHEY_SIMPLEX

# RGB, the frame is RGB.
RED = (255, 50, 50)
GREEN = (60, 255, 120)
CYAN = (0, 220, 255)
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
AMBER = (255, 170, 0)


def _norm(box):
    """
    Integer [x1, y1, x2, y2] with x1 <= x2 and y1 <= y2. A Kalman predicted
    box can come out inverted, and a negative size made cv2.ellipse assert
    and kill the pipeline thread.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _brackets(img, box, color, thickness=3, frac=0.28, margin=0):
    x1, y1, x2, y2 = _norm(box)
    x1 -= margin
    y1 -= margin
    x2 += margin
    y2 += margin
    lx = max(8, int((x2 - x1) * frac))
    ly = max(8, int((y2 - y1) * frac))
    for x, y, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (x, y), (x + dx * lx, y), color, thickness)
        cv2.line(img, (x, y), (x, y + dy * ly), color, thickness)


def _text(img, text, org, color, scale=0.6, thickness=2):
    # Dark outline first so the text reads on any background.
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 3)
    cv2.putText(img, text, org, FONT, scale, color, thickness)


class Hud:
    def __init__(self):
        self._lock_id = None
        self._lock_frame = 0

    # ------------------------------------------------------------- filters

    # Both picture filters are a grey conversion plus one colour lookup
    # table. The obvious versions (histogram equalisation + merge, blur +
    # applyColorMap) measured 33 and 74 ms at 1280x720.
    _LUTS = None

    @classmethod
    def _luts(cls):
        if cls._LUTS is None:
            g = np.arange(256, dtype=np.float32)
            # Night vision: lifted shadows, green phosphor. RGB order.
            v = np.clip(255.0 * (g / 255.0) ** 0.6, 0, 255)
            night = np.stack([v * 0.22, v, v * 0.22], axis=-1)
            # Thermal: OpenCV's INFERNO map, converted from BGR to RGB.
            inferno = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, 256), cv2.COLORMAP_INFERNO)
            thermal = inferno.reshape(256, 3)[:, ::-1]
            cls._LUTS = {
                "night": night.astype(np.uint8).reshape(1, 256, 3),
                "thermal": np.ascontiguousarray(thermal).reshape(1, 256, 3),
            }
        return cls._LUTS

    @classmethod
    def filter(cls, img, style):
        """Picture filter for the style; returns the (possibly new) frame."""
        lut = cls._luts().get(style)
        if lut is None:
            return img
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.LUT(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), lut)

    # ----------------------------------------------------------------- HUD

    def draw(self, img, style, frame, locked, others, state, pan, tilt, manual, searching):
        """
        locked  (box, id, score) of the locked target or None
        others  boxes of the other tracked objects
        """
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        main = GREEN if style == "night" else (WHITE if style == "thermal" else CYAN)
        hot = GREEN if style == "night" else RED

        # Screen corner frame.
        m, l = 18, 60
        for x, y, dx, dy in ((m, m, 1, 1), (w - m, m, -1, 1), (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
            cv2.line(img, (x, y), (x + dx * l, y), main, 2)
            cv2.line(img, (x, y), (x, y + dy * l), main, 2)

        # Centre crosshair with a gap in the middle.
        g, s = 10, 34
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(img, (cx + dx * g, cy + dy * g), (cx + dx * s, cy + dy * s), main, 2)
        cv2.circle(img, (cx, cy), 46, main, 1)

        for box in others:
            _brackets(img, box, GREY, thickness=2, frac=0.2)

        if locked is not None:
            box, tid, score = locked
            if tid != self._lock_id:
                self._lock_id, self._lock_frame = tid, frame
            age = frame - self._lock_frame
            # Brackets start 70 px out and close in over ~8 frames; white at
            # first, then the lock colour.
            k = min(1.0, age / 8.0)
            margin = int((1.0 - k) * 70) + 4
            color = WHITE if age < 3 else hot
            _brackets(img, box, color, thickness=3, margin=margin)

            x1, y1, x2, y2 = _norm(box)
            tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
            r = max(1, int(max(x2 - x1, y2 - y1) * 0.75))
            spin = (frame * 8) % 360
            for a in range(0, 360, 90):
                cv2.ellipse(img, (tx, ty), (r, r), 0, spin + a, spin + a + 50, hot, 2)
            cv2.circle(img, (tx, ty), 4, hot, -1)

            # Line from the crosshair to the target, while it is off centre.
            if abs(tx - cx) + abs(ty - cy) > 30:
                cv2.line(img, (cx, cy), (tx, ty), hot, 1)

            _text(img, "HEDEF KILITLI", (x1, max(24, y1 - margin - 14)), hot, 0.7, 2)
            label = "ID %d" % tid + ("  %%%d" % int(score * 100) if score is not None else "")
            _text(img, label, (x1, min(h - 10, y2 + margin + 24)), hot, 0.55, 2)
        elif manual:
            _text(img, "MANUEL KONTROL", (cx - 105, 60), AMBER, 0.8, 2)
        else:
            # Searching: blinking title and a scan line sweeping down.
            if (frame // 8) % 2 == 0:
                _text(img, "HEDEF ARANIYOR", (cx - 110, 60), main, 0.8, 2)
            sy = (frame * 14) % h
            cv2.line(img, (0, sy), (w, sy), main, 1)

        # Pan / tilt readout, bottom centre, with a tick for the pan position.
        bar_w = 260
        bx, by = cx - bar_w // 2, h - 40
        cv2.line(img, (bx, by), (bx + bar_w, by), main, 1)
        for i in range(0, 7):
            x = bx + i * bar_w // 6
            cv2.line(img, (x, by - 5), (x, by + 5), main, 1)
        px = bx + int((pan + 90.0) / 180.0 * bar_w)
        cv2.drawMarker(img, (px, by - 10), hot, cv2.MARKER_TRIANGLE_DOWN, 12, 2)
        _text(img, "PAN %+4.0f  TILT %+4.0f" % (pan, tilt), (cx - 110, h - 14), main, 0.55, 2)
        if manual:
            _text(img, "MANUEL", (w - 150, 60), AMBER, 0.7, 2)


    def draw_fire(self, img, age, hit, point, score):
        """
        Shot effect, age frames after the shot: a short white flash, rings
        out of the crosshair, then a burst and VURULDU! on the target or ISKA.
        The score stays in the corner afterwards. ~10 frames (~0.7 s at the
        face model's 15 fps); only the flash (2 frames) touches every pixel.
        """
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        if age <= 1:
            # Flash: img * 0.45 + 140 in one pass (addWeighted with a white
            # frame measured 25 ms).
            cv2.convertScaleAbs(img, dst=img, alpha=0.45, beta=140)
        if age <= 6:
            for k in range(3):
                r = 20 + (age + k) * 26
                cv2.circle(img, (cx, cy), r, AMBER, 3 - k if k < 3 else 1)
        if age <= 12:
            if hit and point is not None:
                px, py = point
                for a in range(0, 360, 30):
                    t = math.radians(a + age * 7)
                    r0, r1 = 12 + age * 4, 30 + age * 9
                    cv2.line(img, (int(px + r0 * math.cos(t)), int(py + r0 * math.sin(t))),
                             (int(px + r1 * math.cos(t)), int(py + r1 * math.sin(t))), RED, 3)
                cv2.circle(img, (px, py), 10 + age * 3, AMBER, -1 if age < 4 else 2)
                _text(img, "VURULDU!", (max(10, px - 90), max(40, py - 60 - age * 3)), RED, 1.2, 3)
            else:
                _text(img, "ISKA", (cx - 45, cy - 70), AMBER, 1.2, 3)
        # Top left, clear of the panel's D-pad and effect buttons.
        _text(img, "SKOR %d" % score, (34, 100), AMBER, 0.8, 2)
        return age <= 12
