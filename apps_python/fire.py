"""
Fire button game logic: cooldown, hit test, score keeping.

Everything is timed in seconds, not frames. The first version counted frames,
so the same shot lasted 0.85 s at 14 fps and 0.6 s at 20 fps, and a stalled
picture could skip the whole effect. The effect itself is drawn by
hud.Hud.draw_fire from the Shot kept here.

A shot hits when the locked target's centre is within HIT_RADIUS of the
crosshair (frame centre), or inside the target box if that is bigger, so a
face filling the screen always counts. The target id is kept with the shot,
so the impact follows the target while the effect plays.
"""

import time

COOLDOWN = 0.35      # s between shots
FX_TIME = 1.1        # s the shot effect plays
HIT_RADIUS = 60.0    # px at 1280x720


class Shot:
    __slots__ = ("seq", "t0", "hit", "target_id", "point")

    def __init__(self, seq, t0, hit, target_id, point):
        self.seq = seq
        self.t0 = t0
        self.hit = hit
        self.target_id = target_id
        self.point = point  # target centre at the shot, frame pixels

    def age(self, now):
        return now - self.t0


class FireControl:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.shots = 0
        self.hits = 0
        self.streak = 0
        self.best_streak = 0
        self.last = None

    def cooldown_left(self, now=None):
        if self.last is None:
            return 0.0
        now = self.clock() if now is None else now
        return max(0.0, COOLDOWN - (now - self.last.t0))

    def fire(self, frame_w, frame_h, target=None):
        """
        Take a shot. target is (box, id) of the locked target or None.
        Returns the Shot, or None while the gun is still cooling down.
        """
        now = self.clock()
        if self.cooldown_left(now) > 0.0:
            return None
        hit, tid, point = False, None, None
        if target is not None:
            (x1, y1, x2, y2), tid = target
            tx, ty = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            point = (int(tx), int(ty))
            half = max(abs(x2 - x1), abs(y2 - y1)) / 2.0
            dist = ((tx - frame_w / 2.0) ** 2 + (ty - frame_h / 2.0) ** 2) ** 0.5
            hit = dist <= max(HIT_RADIUS * frame_w / 1280.0, half)
        self.shots += 1
        if hit:
            self.hits += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
        else:
            self.streak = 0
        self.last = Shot(self.shots, now, hit, tid, point)
        return self.last

    def active(self, now=None):
        """The shot whose effect is still playing, or None."""
        if self.last is None:
            return None
        now = self.clock() if now is None else now
        return self.last if self.last.age(now) <= FX_TIME else None

    def stats(self):
        return {
            "shots": self.shots,
            "hits": self.hits,
            "accuracy": round(100.0 * self.hits / self.shots) if self.shots else None,
            "streak": self.streak,
            "best_streak": self.best_streak,
        }
