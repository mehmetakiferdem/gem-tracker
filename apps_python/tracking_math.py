"""
Pure geometry for the tracker: detection to track association and target
re-acquisition. No camera, servo or Norfair state here, so it can be tested
on a PC against recorded or synthetic boxes (see tools/sim_tracking.py).

Boxes are [x1, y1, x2, y2] in pixels.
"""

import math

import numpy as np


def diou_distance(candidates, objects):
    """
    Distance-IoU (Zheng et al., AAAI 2020) as a Norfair vectorized distance:

        d = 1 - IoU + rho^2 / c^2

    rho is the distance between the box centres and c the diagonal of the
    smallest box enclosing both. Same as 1 - IoU while boxes overlap, but it
    keeps ranking pairs once they no longer do: plain IoU puts every
    non-overlapping pair at 1.0, so a face that moved more than its own width
    between two frames (fast turn, 14 fps, a small face far away) could never
    be matched and its id died. Range [0, 2).

    candidates (N, 4), objects (K, 4) -> (N, K)
    """
    c = candidates[:, None, :]
    o = objects[None, :, :]
    area_c = (c[..., 2] - c[..., 0]) * (c[..., 3] - c[..., 1])
    area_o = (o[..., 2] - o[..., 0]) * (o[..., 3] - o[..., 1])
    iw = np.clip(np.minimum(c[..., 2], o[..., 2]) - np.maximum(c[..., 0], o[..., 0]), 0, None)
    ih = np.clip(np.minimum(c[..., 3], o[..., 3]) - np.maximum(c[..., 1], o[..., 1]), 0, None)
    inter = iw * ih
    iou = inter / np.maximum(area_c + area_o - inter, 1e-9)
    rho2 = ((c[..., 0] + c[..., 2]) - (o[..., 0] + o[..., 2])) ** 2 / 4.0 \
        + ((c[..., 1] + c[..., 3]) - (o[..., 1] + o[..., 3])) ** 2 / 4.0
    ew = np.maximum(c[..., 2], o[..., 2]) - np.minimum(c[..., 0], o[..., 0])
    eh = np.maximum(c[..., 3], o[..., 3]) - np.minimum(c[..., 1], o[..., 1])
    c2 = np.maximum(ew ** 2 + eh ** 2, 1e-9)
    return 1.0 - iou + rho2 / c2


def box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_size(box):
    """Geometric mean of width and height: one length for a box."""
    return math.sqrt(max(1.0, box[2] - box[0]) * max(1.0, box[3] - box[1]))


def reacquire_score(last_box, box, max_dist, max_scale):
    """
    How well box continues a target last seen at last_box, lower is better;
    None when it cannot be the same target.

    Distance is measured in target sizes, not frame pixels: 200 px is a small
    step for a face filling the frame and a jump across the room for a face
    in the distance. A size change counts too (log ratio), so a small face in
    the background near the old position does not inherit the lock of a
    close one.

      max_dist   centre distance limit, in sizes of the last box
      max_scale  size ratio limit, either way (2.0: half to double)
    """
    s0, s1 = box_size(last_box), box_size(box)
    scale = s1 / s0
    if scale > max_scale or scale < 1.0 / max_scale:
        return None
    (x0, y0), (x1, y1) = box_center(last_box), box_center(box)
    dist = math.hypot(x1 - x0, y1 - y0) / s0
    if dist > max_dist:
        return None
    return dist + abs(math.log(scale))


def best_reacquire(last_box, candidates, max_dist, max_scale):
    """
    candidates: (key, box) pairs. Returns (key, box, score) of the best
    continuation of last_box, or None.
    """
    best = None
    for key, box in candidates:
        s = reacquire_score(last_box, box, max_dist, max_scale)
        if s is not None and (best is None or s < best[2]):
            best = (key, box, s)
    return best


class DeadzoneHysteresis:
    """
    Per axis deadzone with hysteresis. Once the error has come inside
    `inner` the axis is "settled" and stays still until the error leaves
    `outer`. A single threshold made a centred target, whose box jitters a
    few pixels frame to frame, cross it back and forth: the servo kept
    twitching by one step while the target stood still.
    """

    def __init__(self, inner, outer):
        self.inner = inner
        self.outer = max(inner, outer)
        self.settled = False

    def active(self, error):
        """True when the axis should move for this error."""
        e = abs(error)
        if self.settled:
            if e > self.outer:
                self.settled = False
        elif e <= self.inner:
            self.settled = True
        return not self.settled

    def reset(self):
        self.settled = False
