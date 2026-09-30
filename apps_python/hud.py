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


    def draw_fire(self, img, shot, now, point, stats, target_half=None):
        """
        Shot effect, timed in seconds from the shot (fire.Shot) so it looks
        the same at any frame rate:

          0.00-0.10  muzzle flash at the bottom centre, where the "gun" is
          0.00-0.14  tracer from the muzzle to the aim point
          0.03-0.40  hit marker round the crosshair (hits only)
          0.00-0.55  hit: shockwave ring and sparks on the target, following
                     it (point is its current centre); miss: a puff of dust
                     at the crosshair
          0.00-1.10  VURULDU! / ISKA rising and fading, streak callout;
                     target_half (half the target box height) keeps the
                     callout clear of the box and its lock label

        Everything is drawn into small regions and blended there; the old
        full frame white flash hid the picture and cost a full frame pass.
        """
        h, w = img.shape[:2]
        k = w / 1280.0
        cx, cy = w // 2, h // 2
        t = shot.age(now)
        hit = shot.hit and point is not None
        aim = (int(point[0]), int(point[1])) if hit else (cx, cy)
        muzzle = (cx, h - 1)

        if t <= 0.10:
            a = 1.0 - t / 0.10
            r = int((70 + 50 * (1.0 - a)) * k)
            _blend(img, (muzzle[0] - r, h - r, muzzle[0] + r, h), a * 0.85,
                   lambda roi, ox, oy: _flash(roi, muzzle[0] - ox, muzzle[1] - oy, r))

        if t <= 0.14:
            a = 1.0 - t / 0.14
            # The tracer's head runs ahead, its tail follows.
            head = min(1.0, t / 0.05)
            tail = max(0.0, (t - 0.03) / 0.11)
            p0 = _lerp(muzzle, aim, tail)
            p1 = _lerp(muzzle, aim, head)
            cv2.line(img, p0, p1, AMBER, max(2, int(6 * k * a)), cv2.LINE_AA)
            cv2.line(img, p0, p1, WHITE, max(1, int(2 * k)), cv2.LINE_AA)

        if hit and 0.03 <= t <= 0.40:
            # FPS style hit marker: four short diagonals round the crosshair.
            a = 1.0 - (t - 0.03) / 0.37
            g, l = int(14 * k), int(34 * k)
            col = _scale(WHITE, 0.4 + 0.6 * a)
            for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                cv2.line(img, (cx + dx * g, cy + dy * g), (cx + dx * l, cy + dy * l),
                         col, max(2, int(4 * k)), cv2.LINE_AA)

        if t <= 0.55:
            f = t / 0.55
            px, py = aim
            if hit:
                if t <= 0.12:
                    cv2.circle(img, aim, int((16 + 90 * t) * k), AMBER, -1, cv2.LINE_AA)
                    cv2.circle(img, aim, int((8 + 40 * t) * k), WHITE, -1, cv2.LINE_AA)
                reach = int((20 + 420 * f) * k) + 4

                def burst(roi, ox, oy):
                    c = (px - ox, py - oy)
                    cv2.circle(roi, c, int((20 + 220 * f) * k), RED,
                               max(2, int(6 * k * (1.0 - f))), cv2.LINE_AA)
                    # Sparks: fixed pseudo random directions per shot,
                    # flying out.
                    for i in range(14):
                        ang = (shot.seq * 97 + i * 211) % 360
                        spd = 0.6 + ((shot.seq * 31 + i * 57) % 100) / 100.0
                        r1 = (18 + 260 * spd * f) * k
                        r0 = r1 - 40 * k * (1.0 - 0.5 * f)
                        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                        cv2.line(roi, (int(c[0] + r0 * ca), int(c[1] + r0 * sa)),
                                 (int(c[0] + r1 * ca), int(c[1] + r1 * sa)),
                                 AMBER if i % 3 else WHITE, max(2, int(3 * k)), cv2.LINE_AA)

                _blend(img, (px - reach, py - reach, px + reach, py + reach),
                       1.0 - f, burst)
            else:
                reach = int(90 * k)

                def puff(roi, ox, oy):
                    for i in range(6):
                        ang = math.radians((shot.seq * 53 + i * 60) % 360)
                        rr = (10 + 60 * f) * k
                        cv2.circle(roi, (int(px - ox + rr * math.cos(ang)),
                                         int(py - oy + rr * math.sin(ang))),
                                   max(1, int((9 - 7 * f) * k)), GREY, -1, cv2.LINE_AA)

                _blend(img, (px - reach, py - reach, px + reach, py + reach),
                       1.0 - f, puff)

        if t <= 1.1:
            a = 1.0 if t < 0.6 else 1.0 - (t - 0.6) / 0.5
            rise = int(70 * k * t)
            if hit:
                # Clear of the box and its HEDEF KILITLI label above it.
                top = aim[1] - (target_half or int(80 * k))
                text, col, scale = "VURULDU!", RED, 1.4 * k
                org = (aim[0], max(int(60 * k), top - int(100 * k) - rise))
            else:
                text, col, scale = "ISKA", AMBER, 1.2 * k
                org = (cx, cy + int(110 * k) - rise)
            _text_fade(img, text, org, col, scale, max(2, int(3 * k)), a)
            if hit and stats.get("streak", 0) >= 2:
                _text_fade(img, "SERI x%d" % stats["streak"],
                           (org[0], org[1] + int(42 * k)), AMBER,
                           0.8 * k, max(2, int(2 * k)), a)

    def draw_score(self, img, stats, cooldown):
        """
        Score panel, top left (clear of the panel's D-pad and effect
        buttons), shown once the first shot is fired.
        """
        if not stats.get("shots"):
            return
        h, w = img.shape[:2]
        k = w / 1280.0
        x0, y0 = int(30 * k), int(78 * k)
        pw, ph = int(250 * k), int(84 * k)
        _blend(img, (x0, y0, x0 + pw, y0 + ph), 0.55,
               lambda roi, ox, oy: roi.__setitem__(Ellipsis, 0))
        cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), AMBER, 1)
        _text(img, "SKOR %d" % stats["hits"], (x0 + int(12 * k), y0 + int(32 * k)),
              AMBER, 0.85 * k, 2)
        acc = stats.get("accuracy")
        _text(img, "ISABET %s  %d/%d" % ("-" if acc is None else "%%%d" % acc,
                                           stats["hits"], stats["shots"]),
              (x0 + int(12 * k), y0 + int(60 * k)), WHITE, 0.5 * k, 1)
        _text(img, "SERI %d  (EN IYI %d)" % (stats["streak"], stats["best_streak"]),
              (x0 + int(12 * k), y0 + int(78 * k)), GREY, 0.45 * k, 1)
        if cooldown > 0.0:
            # Reload bar along the bottom edge of the panel.
            from fire import COOLDOWN
            fill = int(pw * (1.0 - cooldown / COOLDOWN))
            cv2.line(img, (x0, y0 + ph), (x0 + fill, y0 + ph), AMBER, max(2, int(4 * k)))


def _lerp(p, q, f):
    return (int(p[0] + (q[0] - p[0]) * f), int(p[1] + (q[1] - p[1]) * f))


def _scale(color, f):
    f = max(0.0, min(1.0, f))
    return tuple(int(c * f) for c in color)


def _text_centered(img, text, org, color, scale, thickness):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    _text(img, text, (int(org[0] - tw / 2), int(org[1])), color, scale, thickness)


def _text_fade(img, text, org, color, scale, thickness, alpha):
    """Centred outlined text at org, blended in at alpha."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness + 3)
    x, y = int(org[0] - tw / 2), int(org[1])
    pad = thickness + 4
    _blend(img, (x - pad, y - th - pad, x + tw + pad, y + base + pad), alpha,
           lambda roi, ox, oy: _text(roi, text, (x - ox, y - oy), color, scale, thickness))


def _flash(roi, x, y, r):
    """Muzzle flash drawn into roi around (x, y): hot core and star rays."""
    cv2.circle(roi, (x, y), r, AMBER, -1, cv2.LINE_AA)
    cv2.circle(roi, (x, y), int(r * 0.55), (255, 240, 200), -1, cv2.LINE_AA)
    for ang in range(200, 341, 35):
        t = math.radians(ang)
        cv2.line(roi, (x, y), (int(x + 1.6 * r * math.cos(t)), int(y + 1.6 * r * math.sin(t))),
                 WHITE, max(2, r // 12), cv2.LINE_AA)


def _blend(img, rect, alpha, draw):
    """
    Draw into a copy of img[rect] with draw(roi, x0, y0) and blend it back at
    alpha: transparency for one small region instead of the whole frame.
    """
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in rect)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0 or alpha <= 0.0:
        return
    roi = img[y0:y1, x0:x1]
    layer = roi.copy()
    draw(layer, x0, y0)
    cv2.addWeighted(layer, min(1.0, alpha), roi, 1.0 - min(1.0, alpha), 0, dst=roi)
