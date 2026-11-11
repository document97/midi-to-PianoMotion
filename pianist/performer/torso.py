"""Torso lean toward a hand playing far out (numpy version of lib/body_driver.BodyDriver.pose), on a constant forward lean;
the shoulders brought forward (the collarbone turned) when a hand reaches far."""
import math

import numpy as np

from .kinematics import FLOAT, axis_matrix, mat3_q

FORWARD_LEAN_DEGREES = 3.       # the straight back (rig_export.straight_back) leans in a little from the hips: sitting nearly upright
# The shoulder comes forward only for a far reach: from REACH_EASY of the arm's length (shoulder to wrist over upper arm
# plus forearm) it grows with the square of the excess, to its full SHOULDER_FORWARD / SHOULDER_UP (in arm lengths) at
# REACH_FULL; beyond that a note is not played (planner). Smoothed over time like the lean: a stretch of far notes, not
# one note, brings it forward.
REACH_EASY, REACH_FULL = .80, .95
LEAN_SHIFT, LEAN_LIMIT_DEGREES = .3, 30.    # m the lean follows a far hand at most; its angle's soft limit (reach: about 90 % of the keys)
SHOULDER_FORWARD, SHOULDER_UP = .10, .025


def lean_transform(shift, height, pivot):
    """World transform of the upper body: the forward lean and the side lean toward a hand ``shift`` m out (soft limit)."""
    limit = math.radians(LEAN_LIMIT_DEGREES)
    angle = limit * math.tanh(math.asin(float(np.clip(shift / max(.05, height), -1, 1))) / limit)
    transform = np.eye(4, dtype=FLOAT)
    transform[:3, :3] = axis_matrix((0., 1., 0.), angle) @ axis_matrix((1., 0., 0.), -math.radians(FORWARD_LEAN_DEGREES))
    transform[:3, 3] = pivot - transform[:3, :3] @ pivot
    return transform


def reach_window(solver, torso, relaxed):
    """(lowest, highest) key x the hand reaches within REACH_FULL of its arm: from the standard seat's shoulder
    (posture.standard_shoulder: the plan does not depend on the scene's seat), the upper body leaned fully toward
    the side and the shoulder fully forward, the wrist at its neutral depth and height; widened by how far the
    relaxed hand's outer pads lie beside the wrist. ``torso``: the rig's torso data (None: no lean, no shoulder)."""
    from .posture import standard_shoulder
    arm, wrist = solver.arm, np.asarray(solver.rig.neutral_wrist[solver.hand], FLOAT)
    length = arm.l1 + arm.l2
    shoulder = standard_shoulder(solver)
    ahead = wrist - shoulder
    ahead[2] = 0.
    ahead /= np.linalg.norm(ahead)
    forward = shoulder + length * (SHOULDER_FORWARD * ahead + SHOULDER_UP * np.array([0., 0., 1.], dtype=FLOAT))
    palm = solver.palm(wrist, 0., 0.)
    beside = [float(np.asarray(solver.pad_world(d, relaxed, palm)).reshape(-1, 3)[:, 0].mean() - wrist[0]) for d in relaxed]
    ends = []
    for side in (-1., 1.):
        moved = forward
        if torso is not None:
            reference = np.array(torso['reference'][solver.hand], dtype=FLOAT)[:3, 3]
            pivot = shoulder - (reference - np.array(torso['pivot'], dtype=FLOAT))
            height = np.mean([np.array(m, dtype=FLOAT)[2, 3] for m in torso['reference'].values()]) - float(torso['pivot'][2])
            moved = (lean_transform(side * LEAN_SHIFT, height, pivot) @ np.r_[forward, 1.])[:3]
        across = (REACH_FULL * length) ** 2 - (wrist[1] - moved[1]) ** 2 - (wrist[2] - moved[2]) ** 2
        reach = math.sqrt(max(across, 0.))
        ends.append(moved[0] + side * reach + (max(beside) if side > 0 else min(beside)))
    return tuple(ends)


class Torso:
    """Slow torso lean toward a hand that plays far out."""

    def __init__(self, data, arms):
        self.data = data
        self.bone = data['bone']
        self.pivot = np.array(data['pivot'], dtype=FLOAT)
        self.pre_inverse = np.linalg.inv(np.array(data['pre'], dtype=FLOAT))
        self.world = np.array(data['world'], dtype=FLOAT)
        self.reference = {h: np.array(m, dtype=FLOAT) for h, m in data['reference'].items()}
        self.height = np.mean([m[2, 3] for m in self.reference.values()]) - self.pivot[2]
        self.rows = np.zeros((0, 3))
        self.reach = {h: np.zeros((0, 3)) for h in 'LR'}
        self.first = self.last = 0.
        self.shoulders = {}
        for h, s in data.get('shoulders', {}).items():
            length = arms[h].l1 + arms[h].l2
            ahead = np.array(s['ahead'], dtype=FLOAT)
            self.shoulders[h] = dict(bone=s['bone'], head=np.array(s['head'], dtype=FLOAT),
                                     pre_inverse=np.linalg.inv(np.array(s['pre'], dtype=FLOAT)), world=np.array(s['world'], dtype=FLOAT),
                                     offset=length * (SHOULDER_FORWARD * ahead + SHOULDER_UP * np.array([0., 0., 1.], dtype=FLOAT)))

    def set_notes(self, rows, first, last, reach=None):
        """``rows``: (start, wrist x - shoulder x beyond .15 m, (velocity/127)^.6) per note; ``reach``: per hand
        (start, wrist distance from the shoulder over the arm's length, weight) per note."""
        self.rows, self.first, self.last = np.asarray(rows, float).reshape(-1, 3), first, last
        for h, r in (reach or {}).items():
            self.reach[h] = np.asarray(r, float).reshape(-1, 3)

    def _window(self, rows, t):
        """The rows near ``t`` and their Gaussian time weights (the lean's)."""
        rows = rows[np.abs(rows[:, 0] - t) < .8]
        return rows, np.exp(-.5 * ((rows[:, 0] - t) / .35) ** 2) * rows[:, 2] * self._smooth((.8 - np.abs(rows[:, 0] - t)) / .2)

    def protraction(self, h, t):
        """Share (0..1) of the full shoulder offset at ``t``."""
        rows, weights = self._window(self.reach[h], t)
        if not len(rows):
            return 0.
        need = np.clip((rows[:, 1] - REACH_EASY) / (REACH_FULL - REACH_EASY), 0., 1.) ** 2
        return float(np.sum(need * weights) / (.25 + np.sum(weights)))

    @staticmethod
    def _smooth(x):
        x = np.clip(x, 0, 1)
        return x * x * x * (10 + x * (-15 + 6 * x))

    def pose(self, t):
        """(torso bone local quaternion, {hand: shoulder pre-matrix}, {hand: collarbone local quaternion})."""
        rows, weights = self._window(self.rows, t)
        shift = 0.
        if len(rows):
            shift = float(np.sum(rows[:, 1] * weights) / (.25 + np.sum(weights)))
        shift = float(np.clip(shift, -LEAN_SHIFT, LEAN_SHIFT) * self._smooth((t - self.first + .85) / .65) * (1 - self._smooth((t - self.last) / .7)))
        transform = lean_transform(shift, self.height, self.pivot)
        pre, collar = {}, {}
        for h, m in self.reference.items():
            turn = np.eye(4, dtype=FLOAT)
            s = self.shoulders.get(h)
            if s is not None:       # the collarbone turns about its head so the shoulder joint moves by the offset
                arm = m[:3, 3] - s['head']
                moved = arm + self.protraction(h, t) * s['offset']
                axis = np.cross(arm, moved)
                if np.linalg.norm(axis) > 1e-12:
                    turn[:3, :3] = axis_matrix(axis / np.linalg.norm(axis), math.atan2(np.linalg.norm(axis), arm @ moved))
                    turn[:3, 3] = s['head'] - turn[:3, :3] @ s['head']
                collar[h] = mat3_q(s['pre_inverse'] @ turn @ s['world'])
            pre[h] = transform @ turn @ m
        return mat3_q(self.pre_inverse @ transform @ self.world), pre, collar
