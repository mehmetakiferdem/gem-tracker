#!/usr/bin/env python3
"""
Offline benchmark for detection to track association: synthetic face boxes
through the same Norfair version as the board (2.3.0), plain IoU (the old
setting) against DIoU (tracking_math.diou_distance).

Counts, per scenario, how many track ids the main face went through (1 is
perfect; every extra id is a rename the lock has to survive) and, where two
faces cross, how often the main face's id ended up on the other face.

    python3 tools/sim_tracking.py
"""

import os
import sys

import numpy as np
from norfair import Detection, Tracker
from norfair.distances import VectorizedDistance

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "apps_python"))
from tracking_math import diou_distance  # noqa: E402

W, H = 1280, 720


def make_tracker(kind, thr):
    t = Tracker(distance_function="iou", distance_threshold=thr,
                initialization_delay=4, hit_counter_max=30)
    if kind == "diou":
        t.distance_function = VectorizedDistance(diou_distance)
    return t


def face(cx, cy, size):
    return np.array([[cx - size / 2, cy - size * 0.6], [cx + size / 2, cy + size * 0.6]])


def run(kind, thr, frames, rng):
    """frames: list of lists of (tag, box or None). Returns ids seen per tag."""
    t = make_tracker(kind, thr)
    ids = {}
    swaps = 0
    owner = {}
    for dets in frames:
        detections = []
        for tag, box in dets:
            if box is None:
                continue
            noisy = box + rng.normal(0, 2.5, box.shape)
            d = Detection(points=noisy, scores=np.array([0.9, 0.9]), label=0)
            d.tag = tag
            detections.append(d)
        for obj in t.update(detections=detections):
            ld = obj.last_detection
            if ld is None or not hasattr(ld, "tag") or ld not in detections:
                continue
            ids.setdefault(ld.tag, set()).add(obj.id)
            prev = owner.get(obj.id)
            if prev is not None and prev != ld.tag:
                swaps += 1
            owner[obj.id] = ld.tag
    return ids, swaps


def scenario_speed(speed, miss, n=240, size=90, seed=0):
    """One face sweeping left-right at `speed` px/frame, detections missed
    with probability `miss`, a still face on the side as a distractor."""
    rng = np.random.default_rng(seed)
    frames = []
    x, d = 300.0, 1.0
    for _ in range(n):
        x += d * speed
        if x > W - 200 or x < 200:
            d = -d
        main = None if rng.random() < miss else face(x, 330, size)
        frames.append([("main", main), ("other", face(1100, 180, 70))])
    return frames


def scenario_cross(speed, n=200, seed=0):
    """Two faces walking past each other (same row, opposite directions)."""
    frames = []
    for i in range(n):
        a = 250 + i * speed
        b = 1030 - i * speed
        if a > W - 100 or b < 100:
            break
        frames.append([("main", face(a, 360, 110)), ("other", face(b, 380, 100))])
    return frames


def main():
    configs = [("iou", 1.0), ("diou", 1.2), ("diou", 1.3), ("diou", 1.4)]
    print("speed/miss  " + "  ".join("%-9s" % ("%s %.1f" % c) for c in configs)
          + "   (ids used by the main face; 1 = no rename)")
    for speed in (10, 30, 60, 90, 120):
        for miss in (0.0, 0.15):
            row = []
            for kind, thr in configs:
                tot = 0
                for seed in range(5):
                    ids, _ = run(kind, thr, scenario_speed(speed, miss, seed=seed),
                                 np.random.default_rng(seed))
                    tot += len(ids.get("main", ()))
                row.append("%-9.1f" % (tot / 5.0))
            print("%3d px %3.0f%%  %s" % (speed, miss * 100, "  ".join(row)))
    print()
    print("crossing    " + "  ".join("%-9s" % ("%s %.1f" % c) for c in configs)
          + "   (id swaps between the two faces; 0 = none)")
    for speed in (6, 12, 20, 30):
        row = []
        for kind, thr in configs:
            tot = 0
            for seed in range(5):
                _, swaps = run(kind, thr, scenario_cross(speed, seed=seed),
                               np.random.default_rng(seed))
                tot += swaps
            row.append("%-9.1f" % (tot / 5.0))
        print("%3d px/f    %s" % (speed, "  ".join(row)))


if __name__ == "__main__":
    main()
