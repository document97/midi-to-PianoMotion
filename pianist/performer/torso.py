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
LEAN_SHIFT, LEAN_LIMIT_DEGREES = .2, 25.    # m the lean follows a far hand at most; its angle's soft limit
# The seat slides along the bench when the music stays at one end: up to SLIDE_SHARE of the hip joints' spacing; the
# lean covers the rest. Lean and slide together reach about 90 % of the keys (reach_window). What the music asks is
# the far-out excess of the notes averaged over SLIDE_SECONDS (Gaussian sigma): one far note asks little, a passage
# out there asks it all. The seat gets there SLIDE_AHEAD s before the notes (quickly: the hands are playing) and comes
# back slowly, over SLIDE_BACK s after them, holding its place between the notes of a passage.
SLIDE_SHARE, SLIDE_SECONDS, SLIDE_AHEAD, SLIDE_BACK = .4, .6, .6, 3.
SHOULDER_FORWARD, SHOULDER_UP = .10, .025
# The side lean and the turn are shared up the spine, the upper bones more (shares 1 : 2 : 3 ... from the lowest): the
# chest leans and turns further than the belly. The forward lean stays a hinge at the hips (the back straight).
# The chest turns toward the side the seat has slid to, TURN_DEGREES at the full slide; the neck takes back
# HEAD_STEADY of the spine's side lean and turn, so the head moves less than the chest.
TURN_DEGREES, HEAD_STEADY = 8., .5


def _about(point, rotation):
    m = np.eye(4)
    m[:3, :3] = rotation
    m[:3, 3] = point - rotation @ point
    return m


def spine_transforms(shift, turn, heads, top, height, offset=(0., 0., 0.)):
    """Cumulative world transforms of the spine bones (lowest first; ``heads``: their heads, ``top``: where the neck
    starts): the seat slid by ``offset``, the forward lean at the lowest head, the side lean toward a hand ``shift`` m
    out (soft limit) and the ``turn`` (rad, about the vertical) shared up the spine. The shares are scaled so the top
    moves sideways as a rigid lean of the whole back about its lowest head would (what the reach is planned on)."""
    heads = np.asarray(heads, float)
    limit = math.radians(LEAN_LIMIT_DEGREES)
    angle = limit * math.tanh(math.asin(float(np.clip(shift / max(.05, height), -1, 1))) / limit)
    shares = np.arange(1., len(heads) + 1.)
    shares /= shares.sum()
    rise = top[2] - heads[:, 2]
    gain = rise[0] / max(float(np.sum(shares * rise)), 1e-9)
    t = np.eye(4)
    t[:3, 3] = offset
    t = t @ _about(heads[0], axis_matrix((1., 0., 0.), -math.radians(FORWARD_LEAN_DEGREES)).astype(float))
    out = []
    for head, share in zip(heads, shares):
        here = (t @ np.r_[head, 1.])[:3]
        r = axis_matrix((0., 1., 0.), angle * share * gain).astype(float) @ axis_matrix((0., 0., 1.), turn * share).astype(float)
        t = _about(here, r) @ t
        out.append(t)
    return out


def _half(rotation, share):
    """``share`` of a rotation (same axis)."""
    angle = math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    if angle < 1e-9:
        return np.eye(3)
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    return axis_matrix(axis / np.linalg.norm(axis), angle * share).astype(float)


def slide_max(torso):
    """m the seat may slide along the bench: SLIDE_SHARE of the hip joints' spacing (0 without them)."""
    hips = (torso or {}).get('hips')
    return SLIDE_SHARE * float(np.linalg.norm(np.subtract(hips[0], hips[1]))) if hips else 0.


def reach_window(solver, torso, relaxed):
    """(lowest, highest) key x the hand reaches within REACH_FULL of its arm: from the standard seat's shoulder
    (posture.standard_shoulder: the plan does not depend on the scene's seat), the upper body leaned fully toward
    the side, the seat slid fully along the bench and the shoulder fully forward, the wrist at its neutral depth and
    height; widened by how far the
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
            move = shoulder - np.array(torso['reference'][solver.hand], dtype=float)[:3, 3]   # the body to the standard seat
            height = np.mean([np.array(m, dtype=FLOAT)[2, 3] for m in torso['reference'].values()]) - float(torso['pivot'][2])
            heads = [np.array(b['head'], float) + move for b in torso['spine']]
            last = spine_transforms(side * LEAN_SHIFT, -side * math.radians(TURN_DEGREES), heads,
                                    np.array(torso['neck']['head'], float) + move, height, (side * slide_max(torso), 0., 0.))[-1]
            moved = (last @ np.r_[forward, 1.])[:3]
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
        self.slide_max = slide_max(data)
        self.first = self.last = 0.
        self.spine = [dict(bone=b['bone'], head=np.array(b['head'], float), pre_inverse=np.linalg.inv(np.array(b['pre'], float)),
                           world=np.array(b['world'], float)) for b in data['spine']]
        n = data['neck']
        self.neck = dict(bone=n['bone'], head=np.array(n['head'], float), pre_inverse=np.linalg.inv(np.array(n['pre'], float)),
                         world=np.array(n['world'], float))
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

    def _slide_need(self):
        """Per note (onset, m the music asks the seat to slide there), once."""
        if getattr(self, '_need_rows', None) is not self.rows:
            need = []
            for start in self.rows[:, 0]:
                near = self.rows[np.abs(self.rows[:, 0] - start) < 3. * SLIDE_SECONDS]
                weights = np.exp(-.5 * ((near[:, 0] - start) / SLIDE_SECONDS) ** 2) * near[:, 2]
                need.append(float(np.sum(near[:, 1] * weights) / (.25 + np.sum(weights))))
            self._need = np.c_[self.rows[:, 0], np.clip(need, -self.slide_max, self.slide_max)] if len(need) else np.zeros((0, 2))
            self._need_rows = self.rows
        return self._need

    def slide(self, t):
        """m the seat has slid along the bench (+x) at ``t``: toward each side, the most any note asks, reached
        SLIDE_AHEAD s before it and let go over SLIDE_BACK s after it."""
        if not self.slide_max:
            return 0.
        need = self._slide_need()
        need = need[(need[:, 0] > t - SLIDE_BACK) & (need[:, 0] < t + SLIDE_AHEAD)]
        if not len(need):
            return 0.
        lag = t - need[:, 0]                 # > 0: the note is past
        reach = np.where(lag < 0, self._smooth(1 + lag / SLIDE_AHEAD), 1 - self._smooth(lag / SLIDE_BACK))
        out = max(0., float(np.max(need[:, 1] * reach))) - max(0., float(np.max(-need[:, 1] * reach)))
        return out * (1 - self._smooth((t - self.last) / .7))

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
        """({bone: local quaternion} of the spine and the neck, {hand: shoulder pre-matrix}, {hand: collarbone local
        quaternion}, the seat's slide along the bench as a world offset)."""
        rows, weights = self._window(self.rows, t)
        shift = 0.
        if len(rows):
            shift = float(np.sum(rows[:, 1] * weights) / (.25 + np.sum(weights)))
        slide = self.slide(t)
        if shift * slide > 0:           # the slid seat takes its part of the reach; a seat still coming back adds no lean
            shift = math.copysign(max(0., abs(shift) - abs(slide)), shift)
        shift = float(np.clip(shift, -LEAN_SHIFT, LEAN_SHIFT) * self._smooth((t - self.first + .85) / .65) * (1 - self._smooth((t - self.last) / .7)))
        offset = np.array([slide, 0., 0.], dtype=FLOAT)
        turn = -math.radians(TURN_DEGREES) * (slide / self.slide_max if self.slide_max else 0.)
        heads = [b['head'] for b in self.spine]
        chain = spine_transforms(shift, turn, heads, self.neck['head'], self.height, offset)
        spine_q, below = {}, np.eye(4)
        below[:3, 3] = offset
        for b, moved in zip(self.spine, chain):
            spine_q[b['bone']] = mat3_q((b['pre_inverse'] @ np.linalg.inv(below) @ moved @ b['world']).astype(FLOAT))
            below = moved
        transform = chain[-1]
        # the neck takes back part of the side lean and turn (not the forward lean): the head steadier than the chest
        lean_only = spine_transforms(shift, turn, heads, self.neck['head'], self.height)[-1][:3, :3] \
            @ np.linalg.inv(spine_transforms(0., 0., heads, self.neck['head'], self.height)[-1][:3, :3])
        steady = np.eye(4)
        steady[:3, :3] = _half(lean_only, HEAD_STEADY).T
        n = self.neck
        spine_q[n['bone']] = mat3_q((n['pre_inverse'] @ np.linalg.inv(transform) @ steady @ transform @ n['world']).astype(FLOAT))
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
            pre[h] = (transform @ turn @ m).astype(FLOAT)
        return spine_q, pre, collar, offset
