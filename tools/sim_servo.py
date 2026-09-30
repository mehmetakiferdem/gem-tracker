#!/usr/bin/env python3
"""
Offline check of the pan loop (PanTiltServoController.update) against a
simple rig model: the camera follows the commanded angle, the picture the
detector sees is LATENCY seconds old, the detected box jitters a little.

    python3 tools/sim_servo.py [other_post_process.py ...]

Runs the controller of every post_process.py given (default: the one in
apps_python) at 14 and 20 fps and prints, per run:

  overshoot   worst overshoot after a 12 degree target jump, degrees
  settle      time until the error stays within 1 degree, s
  track rms   error while the target sways, degrees
  twitches    pulse changes while a centred target stands still for 4 s
"""

import importlib.util
import math
import os
import sys
import time as _time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "apps_python"))

LATENCY = 0.13   # s, capture plus inference
HFOV = 40.0
NOISE = 0.003    # normalized dx noise (~2 px at 1280)


def load(path, name):
    # post_process imports the GStreamer side of the app at module level;
    # only the servo class is needed here.
    import types
    for mod in ("utils", "gi"):
        sys.modules.setdefault(mod, types.ModuleType(mod))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PanTiltServoController


def run(Ctrl, fps, gain, max_step, seed=0, lead=None):
    clock = [0.0]
    real_time = _time.time
    _time.time = lambda: clock[0]
    try:
        kw = {} if lead is None else {"lead": lead}
        c = Ctrl(backend="none", gain=gain, max_step=max_step, deadzone=0.03, hfov=HFOV,
                 search=False, **kw)
        c.enabled = True
        sends = []

        def send(pan, tilt):
            sends.append((clock[0], round(pan, 3)))
            return True

        c._send = send
        rng = np.random.default_rng(seed)
        dt = 1.0 / fps
        history = []           # (t, camera angle)
        errs = {"step": [], "sway": [], "still": []}
        twitches = 0
        last_cmd = None

        def target(t):
            if t < 0.5:
                return 0.0
            if t < 4.0:
                return 12.0
            if t < 10.0:
                return 12.0 + 8.0 * math.sin(2 * math.pi * 0.3 * (t - 4.0))
            return 12.0

        # Target velocity the way the tracker gets it: the world position
        # (seen camera angle + error) through an alpha-beta filter, a
        # stand-in for Norfair's Kalman filter. deg/s.
        wpos, wvel = None, 0.0
        t = 0.0
        while t < 14.0:
            clock[0] = t
            cam = -c.pan_angle           # pan_invert False: angle down = view right
            history.append((t, cam))
            seen = t - LATENCY
            cam_seen = next((a for tt, a in reversed(history) if tt <= seen), 0.0)
            dx = (target(seen) - cam_seen) / (HFOV / 2.0) + rng.normal(0, NOISE)
            w = cam_seen + dx * HFOV / 2.0
            if wpos is None:
                wpos = w
            else:
                pred = wpos + wvel * dt
                r = w - pred
                wpos = pred + 0.5 * r
                wvel = wvel + 0.3 * r / dt
            if lead is None:
                c.update(dx, 0.0)
            else:
                c.update(dx, 0.0, vel=(wvel, 0.0))
            err = target(t) - (-c.pan_angle)
            if 0.5 <= t < 4.0:
                errs["step"].append((t, err))
            elif 4.0 <= t < 10.0:
                errs["sway"].append(err)
            elif t >= 10.0:
                errs["still"].append(err)
                pw = c._angle_to_pw(c.pan_angle)
                if t >= 11.0 and last_cmd is not None and pw != last_cmd:
                    twitches += 1
                last_cmd = pw
            t += dt
    finally:
        _time.time = real_time
    over = max(0.0, -min(e for _, e in errs["step"]))
    settle = None
    for i, (tt, _) in enumerate(errs["step"]):
        if all(abs(e) <= 1.0 for _, e in errs["step"][i:]):
            settle = tt - 0.5
            break
    rms = math.sqrt(sum(e * e for e in errs["sway"]) / len(errs["sway"]))
    return over, settle, rms, twitches


def main():
    paths = sys.argv[1:] or [os.path.join(HERE, "..", "apps_python", "post_process.py")]
    print("%-28s %4s  %9s  %7s  %9s  %8s" % ("controller", "fps", "overshoot", "settle",
                                           "track rms", "twitches"))
    for n, path in enumerate(paths):
        Ctrl = load(path, "pp%d" % n)
        leads = [None] if "old" in path else [0.0, 0.08, 0.13]
        for fps, lead in [(f, l) for l in leads for f in (14, 20)]:
            res = [run(Ctrl, fps, 0.20, 2.5, seed=s, lead=lead) for s in range(5)]
            over = np.mean([r[0] for r in res])
            settle = [r[1] for r in res]
            settle = "never" if None in settle else "%.2f s" % np.mean(settle)
            rms = np.mean([r[2] for r in res])
            tw = np.mean([r[3] for r in res])
            name = os.path.basename(os.path.dirname(os.path.abspath(path)))
            name += "" if lead is None else " lead %.2f" % lead
            print("%-28s %4d  %7.2f°  %7s  %7.2f°  %8.1f" % (name, fps, over, settle, rms, tw))


if __name__ == "__main__":
    main()
