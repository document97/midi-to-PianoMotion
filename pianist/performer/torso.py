"""Torso lean toward a hand playing far out (numpy version of lib/body_driver.BodyDriver.pose), on a constant forward lean;
the shoulders brought forward (the collarbone turned) when a hand reaches far."""
import math
import os

import numpy as np

from .kinematics import FLOAT, axis_matrix, mat3_q

FAST_TORSO = os.environ.get('PIANOMOTION_FAST_TORSO', '1') != '0'   # set 0 to A/B the cached shoulder path

FORWARD_LEAN_DEGREES = 3.       # the straight back (rig_export.straight_back) leans in a little from the hips: sitting nearly upright
# The shoulder comes forward only for a far reach: from REACH_EASY of the arm's length (shoulder to wrist over upper arm
# plus forearm) it grows with the square of the excess, to its full SHOULDER_FORWARD / SHOULDER_UP (in arm lengths) at
# REACH_FULL; beyond that a note is not played (planner). Smoothed over time like the lean: a stretch of far notes, not
# one note, brings it forward.
REACH_EASY, REACH_FULL = .80, .95
LEAN_SHIFT, LEAN_LIMIT_DEGREES = .2, 30.    # m the lean follows a far hand at most; its angle's soft limit
# The shoulders stay put. The arm alone reaches every note on this rig: measured on canonpd, the farthest
# wrist-to-fixed-shoulder distance is 380 mm against a 411 mm arm (31 mm to spare), so leaning/turning/sliding
# the whole upper body to chase a note is unnecessary -- on screen it reads as the shoulders wandering for no
# reason. PIANOMOTION_BODY_STILL scales all three (lean shift, seat slide, chest turn) at once, for BOTH the
# solve and the render, so the IK solves against the same nearly-still body it draws. 0 restores the old full
# body travel; 1 (default) leaves only a small natural follow. The constants below are untouched so the
# planner's reach_window (which still assumes the full travel) keeps its headroom.
BODY_STILL = float(os.environ.get('PIANOMOTION_BODY_STILL', '1'))
BODY_STILL_SCALE = 1. - .85 * max(0., min(1., BODY_STILL))   # 1 -> 0.15, 0 -> 1.0
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
SLIDE_DEADBAND, SLIDE_HYST, SLIDE_DWELL, SLIDE_TAU = .040, .012, 3., .5
SLIDE_GRID = .01                # m: the place commits on a 10 mm grid, so a 26 mm ask and a 30 mm ask land alike
SLIDE_STEP = .02
# The elbows' seat ask (``_slide_yield``) is a hermite interp that answers the collision solve on its own exact
# grid, so its raw curve can jump 60 mm in under a second (measured on canonpd at 166.3 s: 0.003 -> 0.063 m in
# 0.87 s, 103 mm/s). The seat carries the whole upper body with it, so on screen that reads as the hips shuffling
# and the shoulders snapping sideways -- repeated at every ask across the piece. The solve keeps the raw ask (it
# needs the exact clearance), but the rendered pose reads a gaussian-smoothed, rate-limited copy: the seat still
# arrives at exactly the same place, just over a human time scale instead of a jump.
SEAT_YIELD_TAU, SEAT_YIELD_RATE = .40, .030     # s, m/s: 30 mm/s is a deliberate seated shift, not a fidget
SEAT_YIELD_STEP = .02
# The lean follows a hand from LEAN_NEAR out (m past the standard seat's wrist): inside that the hand is close
# enough that neither the seat nor the chest has to move. It used to be .15 m, which left every small ask to the
# seat alone -- the hips did the fidgeting the chest could have covered with two degrees of tilt.
# A hand this far off the standard seat's wrist asks NOTHING of the body: a passage in the middle of the keyboard
# is played with the arm alone. It was .05, then .035 for a day -- both let an ordinary 7 cm offset (the Canon's
# left hand on F#3/B3) push the chest 3.6 cm and the seat 2.4 cm, which the user read as the shoulder moving for
# no reason. The gain below buys back the far passages: past the dead zone the lean answers faster than 1:1.
LEAN_NEAR = .09
LEAN_GAIN = 1.8
# The lean's own commitment, in the seat's spirit: the rows' windowed average is smoothed (LEAN_TAU) and
# rate-limited (LEAN_RATE), so a figure that glances a side for a second glances -- the chest does not dart
# 7 cm toward it in .3 s and back (measured at the user's 860). A passage that holds a side still leans in,
# arriving over ~1 s: the hands travel further than that to get there, so the lean is not late.
LEAN_TAU, LEAN_RATE, LEAN_STEP = .35, .09, .05
# The side lean and the turn are shared up the spine, the LOWER bones more (shares n : n-1 : ... : 1 from the lowest):
# the lean comes from the waist and carries the whole upper body over as one -- the chest does not roll on top of the
# belly, so the shoulder line stays close to level while the head translates with it (the user's read: "腰发力，肩膀也
# 大概平"). The forward lean stays a hinge at the hips (the back straight).
# The chest turns toward the hand the lean is for (the lean's own shift, not the seat's: the seat's small travel
# used to swing the shoulders and the head with it, which magnified the fidgeting); the neck takes back
# HEAD_STEADY of the spine's side lean and turn, so the head moves less than the chest.
TURN_DEGREES, HEAD_STEADY = 8., .85    # HEAD_STEADY was .5: the head rode half the chest's side lean and the user
                                       # reads a tilted head on a body that only leans a little -- a real player
                                       # keeps the head nearly level and just translates it with the chest


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


def spine_transforms(shift, turn, heads, top, height, offset=(0., 0., 0.), sway=0.):
    """Cumulative world transforms of the spine bones (lowest first; ``heads``: their heads, ``top``: where the neck
    starts): the seat slid by ``offset``, the forward lean at the lowest head (plus the intro's ``sway`` deg,
    + backward / - forward), the side lean toward a hand ``shift`` m out (soft limit) and the ``turn`` (rad, about
    the vertical) shared up the spine. The shares are scaled so the top
    moves sideways as a rigid lean of the whole back about its lowest head would (what the reach is planned on)."""
    heads = np.asarray(heads, float)
    limit = math.radians(LEAN_LIMIT_DEGREES)
    angle = limit * math.tanh(math.asin(float(np.clip(shift / max(.05, height), -1, 1))) / limit)
    shares = np.arange(len(heads), 0., -1.)
    shares /= shares.sum()
    rise = top[2] - heads[:, 2]
    gain = rise[0] / max(float(np.sum(shares * rise)), 1e-9)
    t = np.eye(4)
    t[:3, 3] = offset
    t = t @ _about(heads[0], axis_matrix((1., 0., 0.),
                    math.radians(-FORWARD_LEAN_DEGREES + sway)).astype(float))
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
        self.slide_yield_slopes = None
        self.slide_yield_smooth = None  # rendered (gaussian + rate-limited) copy of the seat ask; see set_slide_yield
        self.sway_rows = None         # (times, deg +back/-forward): the intro's deliberate body sway
        self.sway_slopes = None  # lean's tilt lowers it (measured +2 deg/cm against the lean's -1.4)
        self.slide_return = None      # (s0, s1): the closing gesture eases the seat back to its opening place
        self.slide_max = slide_max(data)
        self.first = self.last = 0.
        self._world_cache = {}      # (h, t) -> shoulder pre-matrix; cleared whenever cap/yield changes
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
        self.rows = np.array(rows, dtype=float, copy=True).reshape(-1, 3)
        self.first, self.last = first, last
        self.reach = {h: np.zeros((0, 3)) for h in 'LR'}
        for h, r in (reach or {}).items():
            self.reach[h] = np.array(r, dtype=float, copy=True).reshape(-1, 3)
        self._rows_ver = getattr(self, '_rows_ver', 0) + 1
        self._invalidate_world_cache()

    def set_cap(self, times, shares):
        """How much of its lean the body keeps over time (elbows.py: the elbows out of the body, the body yields
        before an elbow is raised): interpolated at ``pose``; None keeps the full lean."""
        times = np.array(times, dtype=float, copy=True)
        shares = np.array(shares, dtype=float, copy=True)
        old = self.cap
        if (FAST_TORSO and old is not None and old[0].shape == times.shape
                and np.array_equal(old[0], times) and np.array_equal(old[1], shares)):
            return                      # unchanged cap: keep the cached shoulder/trunk matrices
        self.cap = (times, shares)
        from .motion import pchip_slopes
        self.cap_slopes = pchip_slopes(self.cap[0], self.cap[1][:, None])
        self._invalidate_world_cache()

    def _invalidate_world_cache(self):
        """Drop the per-time shoulder/trunk cache when any pose input changes (cap, the ask's lean or seat).
        The elbow solve asks the same thousands of grid times again on every scan/rescan; between two body
        changes the answer is fixed, so caching turns ~5000 spine-chain rebuilds per scan into one per time."""
        if getattr(self, '_world_cache', None) is not None:
            self._world_cache.clear()

    def _cap(self, t):
        if self.cap is None:
            return 1.
        from .motion import hermite_eval
        return float(hermite_eval(self.cap[0], self.cap[1][:, None], self.cap_slopes, t)[0])

    def set_yield(self, times, shifts):
        """The elbows solve's wrist-deviation ask: a lean toward a hand (m, + right) over time, where the swing
        keeping its elbow out of the outline would bend the rendered wrist past the comfort floor -- the body
        moves instead of the wrist. Interpolated at ``pose`` like the cap; None is no ask."""
        times = np.array(times, dtype=float, copy=True)
        shifts = np.array(shifts, dtype=float, copy=True)
        old = self.yield_rows
        if (FAST_TORSO and old is not None and old[0].shape == times.shape
                and np.array_equal(old[0], times) and np.array_equal(old[1], shifts)):
            return
        self.yield_rows = (times, shifts)
        from .motion import pchip_slopes
        self.yield_slopes = pchip_slopes(self.yield_rows[0], self.yield_rows[1][:, None])
        self._invalidate_world_cache()

    def clear_yield(self):
        self.yield_rows = self.yield_slopes = None
        self._invalidate_world_cache()

    def set_slide_yield(self, times, offsets):
        """The ask's seat part: a slide (m, + the seat's own slide direction) over time -- the slide raises the
        palm-vs-forearm angle where the lean's tilt lowers it (measured +2 deg/cm against the lean's -1.4), so
        the two are searched jointly. Interpolated like the lean yield; None is no ask."""
        times = np.array(times, dtype=float, copy=True)
        offsets = np.array(offsets, dtype=float, copy=True)
        old = self.slide_yield_rows
        if (FAST_TORSO and old is not None and old[0].shape == times.shape
                and np.array_equal(old[0], times) and np.array_equal(old[1], offsets)):
            return
        self.slide_yield_rows = (times, offsets)
        from .motion import pchip_slopes
        self.slide_yield_slopes = pchip_slopes(self.slide_yield_rows[0], self.slide_yield_rows[1][:, None])
        # Smoothed, rate-limited copy shared by solve and render so both see identical shoulders/trunk.
        # Scale the world speed cap back into pre-BODY_STILL_SCALE units so world seat speed matches SEAT_YIELD_RATE.
        if len(times) >= 2 and times[-1] - times[0] >= SEAT_YIELD_STEP:
            g = np.arange(times[0], times[-1] + SEAT_YIELD_STEP, SEAT_YIELD_STEP)
            if len(g) >= 2:
                raw = np.interp(g, times, offsets)
                smooth = _gauss_time(raw, SEAT_YIELD_STEP, SEAT_YIELD_TAU)
                max_step = (SEAT_YIELD_RATE / max(BODY_STILL_SCALE, 1e-3)) * SEAT_YIELD_STEP
                for i in range(1, len(smooth)):
                    d = smooth[i] - smooth[i - 1]
                    smooth[i] = smooth[i - 1] + math.copysign(min(abs(d), max_step), d)
                self.slide_yield_smooth = (g, smooth, pchip_slopes(g, smooth[:, None]))
            else:
                self.slide_yield_smooth = None
        else:
            self.slide_yield_smooth = None
        self._invalidate_world_cache()

    def clear_slide_yield(self):
        self.slide_yield_rows = self.slide_yield_slopes = None
        self.slide_yield_smooth = None
        self._invalidate_world_cache()

    def set_slide_return(self, s0, s1):
        """The closing gesture: the seat, slid toward the last notes, eases back to where it opened so the
        hands land on the thighs in the SAME seat as the intro's first frame (the user's spec). ``s0`` is the
        return's start, ``s1`` the moment the hands rest."""
        self.slide_return = (float(s0), float(s1))
        self._invalidate_world_cache()

    def set_sway(self, times, degrees):
        """The intro's body sway (deg, + backward / - forward) on top of the hips' forward lean: the drawing
        arms tip the chest back a hair, the reaching arms carry it a little forward (the user's spec)."""
        self.sway_rows = (np.array(times, dtype=float, copy=True), np.array(degrees, dtype=float, copy=True))
        from .motion import pchip_slopes
        self.sway_slopes = pchip_slopes(self.sway_rows[0], self.sway_rows[1][:, None])
        self._invalidate_world_cache()

    def _sway(self, t):
        if self.sway_rows is None:
            return 0.
        from .motion import hermite_eval
        return float(hermite_eval(self.sway_rows[0], self.sway_rows[1][:, None], self.sway_slopes, t)[0])

    def _slide_yield(self, t, smooth=False):
        if self.slide_yield_rows is None:
            return 0.
        from .motion import hermite_eval
        if smooth and self.slide_yield_smooth is not None:
            g, s, slopes = self.slide_yield_smooth
            v = float(hermite_eval(g, s[:, None], slopes, t)[0])
        else:
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

    def _shift_curve(self):
        """(times, m) the rows' windowed average, smoothed (LEAN_TAU) and rate-limited (LEAN_RATE): the
        lean's commitment, built once per notes set. The per-call gaussian window alone answered a 1.4 s
        figure with a .3 s dart (23 cm/s) -- faster than any deliberate human lean."""
        key = (getattr(self, '_rows_ver', 0), self.first, self.last)
        got = getattr(self, '_shift_cache', None)
        if got is not None and got[0] == key:
            return got[1], got[2]
        times = np.arange(self.first - 1., self.last + 1., LEAN_STEP)
        if len(self.rows):
            values = np.empty(len(times))
            for i, t in enumerate(times):
                near = self.rows[np.abs(self.rows[:, 0] - t) < .8]
                w = np.exp(-.5 * ((near[:, 0] - t) / .35) ** 2) * near[:, 2]
                values[i] = float(np.sum(near[:, 1] * w) / (.25 + np.sum(w)))
        else:
            values = np.zeros(len(times))
        values = _gauss_time(values, LEAN_STEP, LEAN_TAU)
        max_step = (LEAN_RATE / max(BODY_STILL_SCALE, 1e-3)) * LEAN_STEP
        for i in range(1, len(values)):          # rate limit, forward pass (the tau already made it smooth)
            d = values[i] - values[i - 1]
            values[i] = values[i - 1] + math.copysign(min(abs(d), max_step), d)
        self._shift_cache = (key, times, values)
        return times, values

    def _lean(self, t, cap=1., smooth=False):
        """(side lean shift, turn, seat offset) at ``t``; ``cap`` (0..1) scales the lean: the body yields.
        ``smooth``: read the rate-limited seat ask for the rendered pose; the solve keeps the raw ask."""
        st, sv = self._shift_curve()
        shift = float(np.interp(t, st, sv)) if len(st) else 0.
        slide = self.slide(t) + self._slide_yield(t, smooth)
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
        # The body stays nearly still; the arm covers the distance (see BODY_STILL above). Scale all three at
        # the one exit so shoulder_world/trunk_world (the solve) and pose (the render) agree on the body pose.
        k = BODY_STILL_SCALE
        return shift * k, turn * k, offset * k

    def _shoulders(self, t, chain, only=None, pre_only=False):
        """({hand: shoulder pre-matrix}, {hand: collarbone local quaternion}) under the spine's ``chain``.
        ``only`` computes one hand (the elbow search uses just one); ``pre_only`` skips the collarbone
        quaternion (mat3_q) that only ``pose`` needs -- the shoulder's world matrix does not depend on it."""
        pre, collar = {}, {}
        for h, m in self.reference.items():
            if only is not None and h != only:
                continue
            turn = np.eye(4, dtype=FLOAT)
            s = self.shoulders.get(h)
            if s is not None:       # the collarbone turns about its head so the shoulder joint moves by the offset
                arm = m[:3, 3] - s['head']
                moved = arm + self.protraction(h, t) * s['offset']
                axis = np.cross(arm, moved)
                if np.linalg.norm(axis) > 1e-12:
                    turn[:3, :3] = axis_matrix(axis / np.linalg.norm(axis), math.atan2(np.linalg.norm(axis), arm @ moved))
                    turn[:3, 3] = s['head'] - turn[:3, :3] @ s['head']
                if not pre_only:
                    collar[h] = mat3_q(s['pre_inverse'] @ turn @ s['world'])
            pre[h] = (chain @ turn @ m).astype(FLOAT)
        return pre, collar

    def _window(self, rows, t):
        """The rows near ``t`` and their Gaussian time weights (the lean's)."""
        rows = rows[np.abs(rows[:, 0] - t) < .8]
        return rows, np.exp(-.5 * ((rows[:, 0] - t) / .35) ** 2) * rows[:, 2] * self._smooth((.8 - np.abs(rows[:, 0] - t)) / .2)

    def _slide_need(self):
        """Per note (onset, m the music asks the seat to slide there), once."""
        need_key = (getattr(self, '_rows_ver', 0), self.slide_max)
        if getattr(self, '_need_key', None) != need_key:
            need = []
            for start in self.rows[:, 0]:
                near = self.rows[np.abs(self.rows[:, 0] - start) < 3. * SLIDE_SECONDS]
                weights = np.exp(-.5 * ((near[:, 0] - start) / SLIDE_SECONDS) ** 2) * near[:, 2]
                need.append(float(np.sum(near[:, 1] * weights) / (.25 + np.sum(weights))))
            self._need = np.c_[self.rows[:, 0], np.clip(need, -self.slide_max, self.slide_max)] if len(need) else np.zeros((0, 2))
            self._need_key = need_key
        return self._need

    def opening_slide(self):
        """World-space (+x, m) opening seat offset (scaled by BODY_STILL_SCALE, matching pose())."""
        if not self.slide_max:
            return 0.
        return float(self._slide_curve()[1][0]) * BODY_STILL_SCALE

    def slide(self, t):
        """m the seat has slid along the bench (+x) at ``t``: toward each side, the most any note asks, reached
        SLIDE_AHEAD s before it and let go over SLIDE_BACK s after it -- then quantized (SLIDE_DEADBAND) and
        smoothed (SLIDE_TAU): a small ask is the lean's, the seat's place changes rarely and slowly."""
        if not self.slide_max:
            return 0.
        times, values = self._slide_curve()
        v = float(np.interp(t, times, values))
        if self.slide_return is not None and t > self.slide_return[0]:
            # The seat's staged place is latched: it does not walk back on its own after the last note. During
            # the closing gesture ease it to the opening place so the resting hands meet the thighs exactly as
            # they did in the intro's first frame -- wrist, pitch AND the whole upper body share one seat.
            s0, s1 = self.slide_return
            w = self._smooth((t - s0) / max(s1 - s0, 1e-6))
            # the opening seat is the curve's first latched place, before any note asked it to travel
            v = v * (1. - w) + float(values[0]) * w
        return v

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
        key = (getattr(self, '_rows_ver', 0), self.first, self.last, self.slide_max)
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
                # the ask has to keep moving outward: a value bouncing back across the band edge settles nothing
                j = min(i + 1, len(times) - 1)
                if away > SLIDE_DEADBAND + SLIDE_HYST or (raw[j] - raw[i]) * (raw[i] - place) > 0:
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
        cap = self._cap(t)
        shift, turn, offset = self._lean(t, cap, smooth=True)
        if os.environ.get('PIANOMOTION_DEBUG_TORSO'):    # diagnosis only: the lean's components per frame
            import json as _json
            with open(os.environ['PIANOMOTION_DEBUG_TORSO'], 'a') as f:
                _json.dump([[float(t), float(shift), math.degrees(turn), float(cap), float(self._yield(t)),
                             float(self._slide_yield(t, True)), float(self.slide(t)), float(offset[0])]], f)
        heads = [b['head'] for b in self.spine]
        sway = self._sway(t)
        chain = spine_transforms(shift, turn, heads, self.neck['head'], self.height, offset, sway)
        spine_q, below = {}, np.eye(4)
        below[:3, 3] = offset
        for b, moved in zip(self.spine, chain):
            spine_q[b['bone']] = mat3_q((b['pre_inverse'] @ np.linalg.inv(below) @ moved @ b['world']).astype(FLOAT))
            below = moved
        transform = chain[-1]
        # the neck takes back part of the side lean and turn (not the forward lean): the head steadier than the chest
        lean_only = spine_transforms(shift, turn, heads, self.neck['head'], self.height, sway=sway)[-1][:3, :3] \
            @ np.linalg.inv(spine_transforms(0., 0., heads, self.neck['head'], self.height, sway=sway)[-1][:3, :3])
        steady = np.eye(4)
        steady[:3, :3] = _half(lean_only, HEAD_STEADY).T
        n = self.neck
        spine_q[n['bone']] = mat3_q((n['pre_inverse'] @ np.linalg.inv(transform) @ steady @ transform @ n['world']).astype(FLOAT))
        pre, collar = self._shoulders(t, transform)
        return spine_q, pre, collar, offset

    def shoulder_world(self, h, t, cap=None):
        """The hand's shoulder pre-matrix at ``t`` (pose's pre[h]) for a lean ``cap``: the elbow-swing search."""
        if not FAST_TORSO:
            shift, turn, offset = self._lean(t, self._cap(t) if cap is None else cap, smooth=True)
            chain = spine_transforms(shift, turn, [b['head'] for b in self.spine], self.neck['head'], self.height,
                                     offset, self._sway(t))
            return self._shoulders(t, chain[-1])[0][h]
        key = (0, h, float(t), None if cap is None else float(cap))
        hit = self._world_cache.get(key)
        if hit is not None:
            return hit
        shift, turn, offset = self._lean(t, self._cap(t) if cap is None else cap, smooth=True)
        chain = spine_transforms(shift, turn, [b['head'] for b in self.spine], self.neck['head'], self.height,
                                 offset, self._sway(t))
        out = self._shoulders(t, chain[-1], only=h, pre_only=True)[0][h]
        self._world_cache[key] = out
        return out

    def trunk_world(self, t, cap=None):
        """The trunk bone's world matrix at ``t`` for a lean ``cap``: the trunk outline hangs on it (elbows.py).
        The lowest spine bone takes its share of the lean about the neutral pose (spine_transforms) on top of its
        exported world matrix."""
        if not FAST_TORSO:
            shift, turn, offset = self._lean(t, self._cap(t) if cap is None else cap, smooth=True)
            chain = spine_transforms(shift, turn, [b['head'] for b in self.spine], self.neck['head'], self.height,
                                     offset, self._sway(t))
            return (chain[0] @ self.spine[0]['world']).astype(FLOAT)
        key = (1, float(t), None if cap is None else float(cap))
        hit = self._world_cache.get(key)
        if hit is not None:
            return hit
        shift, turn, offset = self._lean(t, self._cap(t) if cap is None else cap, smooth=True)
        chain = spine_transforms(shift, turn, [b['head'] for b in self.spine], self.neck['head'], self.height,
                                 offset, self._sway(t))
        out = (chain[0] @ self.spine[0]['world']).astype(FLOAT)
        self._world_cache[key] = out
        return out
