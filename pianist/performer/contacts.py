"""Finger bodies as capsules (one per bone) for 3D checks of a frame.

Seen from above two fingers may overlap while one passes under the other; what must not happen is the
fingers' bodies running into each other. Each finger bone is a capsule around the segment from its head to
the next head (the last bone to its fingertip), with the radius its own skin has around it in the relaxed
hand. Per frame: the interpenetration of neighbouring fingers (1-2 .. 4-5), the clearance between the two
hands, and how high a free finger is held above two pressed neighbours (fingers are not independent: a
ring finger cannot rise far while the middle and little fingers press their keys).
"""
import numpy as np

from .kinematics import DIGITS
from .pose import RELAXED

PAIRS = ((1, 2), (2, 3), (3, 4), (4, 5))


def segment_distances(p0, p1, q0, q1):
    """Closest distances between segments p0-p1 and q0-q1 (arrays of shape (n, 3)), vectorised."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e = (d1 * d1).sum(1), (d2 * d2).sum(1)
    b, c, f = (d1 * d2).sum(1), (d1 * r).sum(1), (d2 * r).sum(1)
    denom = a * e - b * b
    s = np.where(denom > 1e-12, np.clip((b * f - c * e) / np.maximum(denom, 1e-12), 0., 1.), 0.)
    t = (b * s + f) / np.maximum(e, 1e-12)
    t_clip = np.clip(t, 0., 1.)
    s = np.where(t != t_clip, np.clip((b * t_clip - c) / np.maximum(a, 1e-12), 0., 1.), s)
    closest = (p0 + d1 * s[:, None]) - (q0 + d2 * t_clip[:, None])
    return np.linalg.norm(closest, axis=1)


def point_distances(points, a, b):
    """Distances of ``points`` (n, 3) from the segment a-b."""
    ab = b - a
    t = np.clip((points - a) @ ab / max(ab @ ab, 1e-12), 0., 1.)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


class HandCapsules:
    """Capsules of one hand's fingers: ``segments(values, armature)`` -> {digit: (heads (3, 3), tails (3, 3))}."""

    def __init__(self, rig, hand, armature):
        self.rig, self.hand = rig, hand
        self.fingers = rig.fingers[hand]
        relaxed = {d: RELAXED[d] for d in DIGITS}
        self.tip_length, self.radius = {}, {}
        heads = self._heads(relaxed, armature)
        for d in DIGITS:
            finger = self.fingers[d]
            skin = finger.clearance
            points = rig.clearance_points(hand, d, armature, relaxed)
            owner = np.full(len(points), -1)
            for k, name in enumerate(finger.names):
                if name in skin.bones:
                    mine = skin.weights[:, skin.bones.index(name)] >= .5
                    owner[mine] = k
            h, axis = heads[d]
            tip = points[owner == 2] if (owner == 2).any() else points
            self.tip_length[d] = float(np.max((tip - h[2]) @ axis)) if len(tip) else .02
            starts, ends = self._segments(d, h, axis)
            radius = []
            for k in range(3):
                own = points[owner == k]
                if not len(own):
                    radius.append(.006)
                    continue
                radius.append(float(np.median(point_distances(own, starts[k], ends[k]))))
            self.radius[d] = np.array(radius)

    def _heads(self, values, armature):
        world = self.rig.arm_world
        out = {}
        for d in DIGITS:
            mats = [world @ armature @ m for m in self.fingers[d].local_matrices(values[d])]
            heads = np.array([m[:3, 3] for m in mats])
            axis = mats[2][:3, 1] / np.linalg.norm(mats[2][:3, 1])      # a bone points along its local y
            out[d] = (heads, axis)
        return out

    def _segments(self, d, heads, axis):
        starts = heads
        ends = np.array([heads[1], heads[2], heads[2] + axis * self.tip_length[d]])
        return starts, ends

    def segments(self, values, armature):
        return {d: self._segments(d, *h) for d, h in self._heads(values, armature).items()}


def finger_pairs(caps, segs):
    """Interpenetration depth (m, > 0 when the bodies overlap) of each neighbouring finger pair."""
    out = []
    for a, b in PAIRS:
        (sa, ea), (sb, eb) = segs[a], segs[b]
        i, j = np.meshgrid(np.arange(3), np.arange(3), indexing='ij')
        i, j = i.ravel(), j.ravel()
        if a == 1:
            keep = ~((i == 0) & (j == 0))   # the thumb's and the index finger's roots meet in the palm by design
            i, j = i[keep], j[keep]
        dist = segment_distances(sa[i], ea[i], sb[j], eb[j])
        out.append(float(np.max(caps.radius[a][i] + caps.radius[b][j] - dist)))
    return np.array(out)


def hands_gap(caps, segs):
    """Clearance between the two hands' fingers (m, < 0 when they overlap): ``caps``/``segs`` per hand."""
    left = [(d, k) for d in DIGITS for k in range(3)]
    i, j = np.meshgrid(np.arange(len(left)), np.arange(len(left)), indexing='ij')
    i, j = i.ravel(), j.ravel()
    ls = np.array([segs['L'][d][0][k] for d, k in left]), np.array([segs['L'][d][1][k] for d, k in left])
    rs = np.array([segs['R'][d][0][k] for d, k in left]), np.array([segs['R'][d][1][k] for d, k in left])
    lr = np.array([caps['L'].radius[d][k] for d, k in left])
    rr = np.array([caps['R'].radius[d][k] for d, k in left])
    dist = segment_distances(ls[0][i], ls[1][i], rs[0][j], rs[1][j])
    return float(np.min(dist - lr[i] - rr[j]))


def pointing(segs):
    """Per digit: the finger's direction seen from above (knuckle to fingertip) against the keys' length, in
    degrees (+ toward the high keys). Wrist turn and the finger's own sideways swing together."""
    out = np.zeros(5)
    for d in DIGITS:
        heads, tails = segs[d]
        v = tails[2] - heads[0]
        out[d - 1] = np.degrees(np.arctan2(v[0], v[1]))
    return out


def raised_between(segs, riding):
    """Highest fingertip of a free finger 3 or 4 above its two pressing neighbours' fingertips (m; 0 if none)."""
    tip = {d: segs[d][1][2] for d in DIGITS}
    worst = 0.
    for d in (3, 4):
        if d not in riding and d - 1 in riding and d + 1 in riding:
            worst = max(worst, float(tip[d][2] - (tip[d - 1][2] + tip[d + 1][2]) / 2))
    return worst
