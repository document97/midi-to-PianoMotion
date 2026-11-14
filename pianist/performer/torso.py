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
# The seat's own travel is quantized: the body answers the music's far ends, it does not fidget. Every note's excess
# (over LEAN_NEAR) asks the seat to move a little, and the positive and negative asks are subtracted, so a passage that
# alternates ends asks a few millimetres one way, then the other -- a hip that slides 3 mm every few seconds, which the
# eye reads as fidgeting. The committed place moves only when the ask leaves a DEADBAND around it (HYST keeps a value
# sitting on the edge from re-committing) and the place has been held DWELL s: a small ask is the LEAN's to cover.
# The quantized curve is then smoothed (TAU) so a commit is a slow shift, not a step.
SLIDE_DEADBAND, SLIDE_HYST, SLIDE_DWELL, SLIDE_TAU = .025, .010, 2., .5
SLIDE_GRID = .01                # m: the place commits on a 10 mm grid, so a 26 mm ask and a 30 mm ask land alike
SLIDE_STEP = .02
# The lean follows a hand from LEAN_NEAR out (m past the standard seat's wrist): inside that the hand is close
# enough that neither the seat nor the chest has to move. It used to be .15 m, which left every small ask to the
# seat alone -- the hips did the fidgeting the chest could have covered with two degrees of tilt.
LEAN_NEAR = .05
# The side lean and the turn are shared up the spine, the upper bones more (shares 1 : 2 : 3 ... from the lowest): the
# chest leans and turns further than the belly. The forward lean stays a hinge at the hips (the back straight).
# The chest turns toward the hand the lean is for (the lean's own shift, not the seat's: the seat's small travel
# used to swing the shoulders and the head with it, which magnified the fidgeting); the neck takes back
# HEAD_STEADY of the spine's side lean and turn, so the head moves less than the chest.
TURN_DEGREES, HEAD_STEADY = 8., .5


def _gauss_time(values, step, tau):
    """A zero-phase gaussian over a uniform series; the ends are extrapolated linearly, not copied (copying
    flat-lines a ramp)."""
    r = max(1, int(round(tau / step * 2.)))
    k = np.arange(-r, r + 1) * step
    w = np.exp(-.5 * (k / tau) ** 2)
    w /= w.sum()
    pad = np.concatenate([values[0] + (values[0] - values[1]) * np.arange(r, 0, -1), values,
                          values[-1] + (values[-1] - values[-2]) * np.arange(1, r + 1)])
    return np.convolve(pad, w, mode='valid')


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
        self.cap = None               # (times, shares 0..1): how much of the lean the body keeps (elbows.py)
        self.cap_slopes = None
        self.yield_rows = None        # (times, shifts m): the elbows solve's wrist-deviation ask, a lean toward
        self.yield_slopes = None      # a hand whose natural hang the outline buries past the wrist's floor
        self.slide_yield_rows = None  # ... and its seat part: a slide, which raises the palm angle where the
        self.slide_yield_slopes = None  # lean's tilt lowers it (measured +2 deg/cm against the lean's -1.4)
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

    def set_cap(self, times, shares):
        """How much of its lean the body keeps over time (elbows.py: the elbows out of the body, the body yields
        before an elbow is raised): interpolated at ``pose``; None keeps the full lean."""
        self.cap = (np.asarray(times, float), np.asarray(shares, float))
        from .motion import pchip_slopes
        self.cap_slopes = pchip_slopes(self.cap[0], self.cap[1][:, None])

    def _cap(self, t):
        if self.cap is None:
            return 1.
        from .motion import hermite_eval
        return float(hermite_eval(self.cap[0], self.cap[1][:, None], self.cap_slopes, t)[0])

    def set_yield(self, times, shifts):
        """The elbows solve's wrist-deviation ask: a lean toward a hand (m, + right) over time, where the swing
        keeping its elbow out of the outline would bend the rendered wrist past the comfort floor -- the body
        moves instead of the wrist. Interpolated at ``pose`` like the cap; None is no ask."""
        self.yield_rows = (np.asarray(times, float), np.asarray(shifts, float))
        from .motion import pchip_slopes
        self.yield_slopes = pchip_slopes(self.yield_rows[0], self.yield_rows[1][:, None])

    def clear_yield(self):
        self.yield_rows = self.yield_slopes = None

    def set_slide_yield(self, times, offsets):
        """The ask's seat part: a slide (m, + the seat's own slide direction) over time -- the slide raises the
        palm-vs-forearm angle where the lean's tilt lowers it (measured +2 deg/cm against the lean's -1.4), so
        the two are searched jointly. Interpolated like the lean yield; None is no ask."""
        self.slide_yield_rows = (np.asarray(times, float), np.asarray(offsets, float))
        from .motion import pchip_slopes
        self.slide_yield_slopes = pchip_slopes(self.slide_yield_rows[0], self.slide_yield_rows[1][:, None])

    def clear_slide_yield(self):
        self.slide_yield_rows = self.slide_yield_slopes = None

    def _slide_yield(self, t):
        if self.slide_yield_rows is None:
            return 0.
        from .motion import hermite_eval
        v = float(hermite_eval(self.slide_yield_rows[0], self.slide_yield_rows[1][:, None],
                               self.slide_yield_slopes, t)[0])
        env = self._smooth((t - self.first + .85) / .65) * (1 - self._smooth((t - self.last) / .7))
        return (float(np.clip(v, -self.slide_max, self.slide_max)) if self.slide_max else v) * env

    def _yield(self, t):
        if self.yield_rows is None:
            return 0.
        from .motion import hermite_eval
        v = float(hermite_eval(self.yield_rows[0], self.yield_rows[1][:, None], self.yield_slopes, t)[0])
        env = self._smooth((t - self.first + .85) / .65) * (1 - self._smooth((t - self.last) / .7))
        return float(np.clip(v, -LEAN_SHIFT, LEAN_SHIFT)) * env

    def _lean(self, t, cap=1.):
        """(side lean shift, turn, seat offset) at ``t``; ``cap`` (0..1) scales the lean: the body yields."""
        rows, weights = self._window(self.rows, t)
        shift = 0.
        if len(rows):
            shift = float(np.sum(rows[:, 1] * weights) / (.25 + np.sum(weights)))
        slide = self.slide(t) + self._slide_yield(t)
        if self.slide_max:              # the seat's travel: the plan's curve plus the ask's, clamped to it
            slide = float(np.clip(slide, -self.slide_max, self.slide_max))
        if shift * slide > 0:           # the slid seat takes its part of the reach; a seat still coming back adds no lean
            shift = math.copysign(max(0., abs(shift) - abs(slide)), shift)
        shift = float(np.clip(shift, -LEAN_SHIFT, LEAN_SHIFT) * self._smooth((t - self.first + .85) / .65)
                      * (1 - self._smooth((t - self.last) / .7))) * cap
        shift += self._yield(t)         # the wrist-deviation ask rides on the rows' lean, independent of the cap
        offset = np.array([slide, 0., 0.], dtype=FLOAT)
        # the chest turns toward the hand the body is leaning for; the seat's own travel does not turn it
        turn = -math.radians(TURN_DEGREES) * float(np.clip(shift / max(LEAN_SHIFT, 1e-9), -1., 1.))
        return shift, turn, offset

    def _shoulders(self, t, chain):
        """({hand: shoulder pre-matrix}, {hand: collarbone local quaternion}) under the spine's ``chain``."""
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
            pre[h] = (chain @ turn @ m).astype(FLOAT)
        return pre, collar

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
        SLIDE_AHEAD s before it and let go over SLIDE_BACK s after it -- then quantized (SLIDE_DEADBAND) and
        smoothed (SLIDE_TAU): a small ask is the lean's, the seat's place changes rarely and slowly."""
        if not self.slide_max:
            return 0.
        times, values = self._slide_curve()
        return float(np.interp(t, times, values))

    def _slide_raw(self, t):
        """The unquantized ask at ``t`` (the seat's curve before the deadband)."""
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

    def _slide_curve(self):
        """(times, m) of the seat's plan travel: the raw ask quantized to a committed place -- moved only when the
        ask leaves SLIDE_DEADBAND around it (plus SLIDE_HYST once the place is committed) and the place has been
        held SLIDE_DWELL s -- then smoothed over SLIDE_TAU and rate-limited. Built once per notes set."""
        key = (id(self.rows), self.first, self.last, self.slide_max)
        got = getattr(self, '_slide_cache', None)
        if got is not None and got[0] == key:
            return got[1], got[2]
        if not self.slide_max or not len(self.rows):
            self._slide_cache = (key, np.zeros(2), np.zeros(2))
            return self._slide_cache[1], self._slide_cache[2]
        times = np.arange(self.first - 1., self.last + SLIDE_BACK + 1., SLIDE_STEP)
        raw = np.array([self._slide_raw(float(t)) for t in times])
        place, held = 0., times[0] - SLIDE_DWELL
        committed = np.empty(len(times))
        for i, t in enumerate(times):
            away = abs(raw[i] - place)
            if away > SLIDE_DEADBAND and t - held >= SLIDE_DWELL:
                # the ask has to keep going the same way: a value dancing on the edge of the band settles nothing
                j = min(i + 1, len(times) - 1)
                if away > SLIDE_DEADBAND + SLIDE_HYST or (raw[i] - place) * (raw[j] - place) > 0:
                    place, held = round(float(raw[i]) / SLIDE_GRID) * SLIDE_GRID, t
            committed[i] = place
        values = _gauss_time(committed, SLIDE_STEP, SLIDE_TAU)
        self._slide_cache = (key, times, values)
        return times, values

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
        shift, turn, offset = self._lean(t, self._cap(t))
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
        pre, collar = self._shoulders(t, transform)
        return spine_q, pre, collar, offset

    def shoulder_world(self, h, t, cap=None):
        """The hand's shoulder pre-matrix at ``t`` (pose's pre[h]) for a lean ``cap``: the elbow-swing search."""
        shift, turn, offset = self._lean(t, self._cap(t) if cap is None else cap)
        chain = spine_transforms(shift, turn, [b['head'] for b in self.spine], self.neck['head'], self.height, offset)
        return self._shoulders(t, chain[-1])[0][h]

    def trunk_world(self, t, cap=None):
        """The trunk bone's world matrix at ``t`` for a lean ``cap``: the trunk outline hangs on it (elbows.py).
        The lowest spine bone takes its share of the lean about the neutral pose (spine_transforms) on top of its
        exported world matrix."""
        shift, turn, offset = self._lean(t, self._cap(t) if cap is None else cap)
        chain = spine_transforms(shift, turn, [b['head'] for b in self.spine], self.neck['head'], self.height, offset)
        return (chain[0] @ self.spine[0]['world']).astype(FLOAT)
