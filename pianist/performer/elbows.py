"""The elbows kept out of the body.

The trunk's outline (trunk.json, exported by rig_export.neutral_scene: its skin in the trunk bone's frame as
convex hulls of 1 cm slices, and the arms' skin radii about their bones) hangs on the trunk bone's world matrix
the lean produces. On the elbow's IK circle (shoulder and wrist kept, arm.py.elbow_circle) the arm's two
capsules are sampled at every swing angle; the free arcs are where they clear the outline. Per arm one smooth
swing-angle curve for the whole piece is solved like the wrist paths (bounded least squares over windows,
motion.chunked): inside the free arcs, pulled toward the elbow's natural hang. Where no swing clears the trunk
the upper body leans less instead (the torso's cap, torso.Torso.set_cap: the body yields before the elbow is
raised above its natural hang), searched per sample and smoothed over time. And where a free arc exists but
excludes the natural hang, the swing it forces rotates the forearm away from the settled palm axis -- past the
wrist's comfort floor (posture.PALM_FLOOR_DEG) the body leans toward the hand instead (torso.Torso.set_yield:
the ask searches the least lean that opens the hang, then the scans re-read the moved body on it).

Solved once in the main process between the finger keyposes and the frames; the frames stage interpolates the
curve and hands the angle to arm.Arm.pose.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .arm import closest_elbow_angle, elbow_circle

SLICE = .01                      # m of outline per hull slice, along the trunk bone's y
AXIS, OTHERS = 1, (0, 2)         # the slices run along the bone's y; each hull lies in its x-z
ANGLE_STEP = 3.                  # deg between the sampled swing angles on the elbow's circle
TRUNK_ANG_DEG = np.arange(-180., 180., ANGLE_STEP)      # the sampled circle, degrees (the verify's interp grid)
UP = np.linspace(.36, 1., 8)     # the upper arm's capsule: shares of the shoulder->elbow segment. Not the deltoid:
                                 # its skin rides 6-7 cm off the bone (sleeve), no swing clears it -- that region is
                                 # the body-yield's to help, and demanding it here only drags the elbow into big swings
FO = np.linspace(0., 1., 12)     # ... the forearm's: shares of the elbow->wrist segment
CLEARANCE = 0.                   # m the arm keeps outside the outline. Zero: the outline is sampled, the skin is not,
                                 # and the necessary-swing study (median 12 deg, p90 25) was measured at touch level --
                                 # a margin here sits at the bounds' edge and multiplies into visible swings
RISE_LIMIT = .04                 # m the elbow may rise above its natural hang (an absolute wrist-relative cap would
                                 # bar the hang itself whenever the wrist rides low: hands entering or leaving)
CURVE_STEP = .05                 # s between the swing curve's samples
DAMP_TAU = .06                   # s: the swing curve's damping, a gaussian over time (zero phase, the same hand as
                                 # the body ask's _ease). The curve is C1 on its .05 s grid and the frames' hermite is
                                 # C1 too, so the acceleration steps at every knot -- measured on dl34 at p95 576
                                 # deg/s^2 (L) / 219 (R), worst 4086 / 6939, which is what a stiff upper arm looks
                                 # like: the elbow turns, stops turning and turns again within two frames. Damping
                                 # costs 1.2-1.6 deg of angle and <=4 mm of elbow travel and takes those steps to
                                 # 88 / 30.
DAMP_SUB = 5                     # the damped curve's grid is CURVE_STEP/DAMP_SUB: fine enough that its own knots
                                 # sit under a frame's .0167 s, so the hermite's corners fall below what a frame
                                 # can show
CAP_LEVELS = 9                   # lean shares tried for a blocked sample, from no yield down to none left
CAP_ROUNDS = 3                   # blocked samples ask, the smoothed cap is re-checked, twice
CAP_PULL = .12                   # the cap's pull to no yield: one unit of cost this far off 1
JERK, ACC, VEL, PULL = 150., 25., 5., .3     # the swing curve's weights: one unit of cost at these
TRACK_RATE = .25                 # m/s of wrist motion at which the curve's pull doubles: a hop is tracked in
                                 # step with the hand (a rigid chain, the elbow rides along), a hold is smoothed
SMOOTH_TAU = .3                  # s the constraint signals (bounds, natural hang) are smoothed over where the
                                 # wrist sits still: the body's own millimetre drift (a lean ramp) is amplified
                                 # near tangency into visible elbow sweeps -- smoothed the elbow follows the body
                                 # at body speed; where the wrist moves the signals are tracked exactly
INTRUDE_OK = .001                # m of outline intrusion the smoothing may add at the bounds (about a noise unit)
RISE_OK = .003                   # m of rise-cap slack the smoothing may add
                                 # (rad/s^3, rad/s^2, rad/s, rad off the natural hang)
YIELD_STEPS = (.016, .032, .064)   # m the wrist-deviation ask leans, searched small first (the least motion wins
                                  # ties); the outline's response to the shift is a parabola whose good side
                                  # moves along the arm, so both signs are tried
YIELD_SLIDES = (.03, .05, .08, .11)   # m the ask slides the seat: the slide raises the palm-vs-forearm angle
                                      # where the lean's tilt lowers it (+2 deg/cm against the lean's -1.4,
                                      # measured), at the price of the seat's travel (both hands feel it)
YIELD_TAU = .12                  # s of gaussian on the ask's lean: the lean is a body motion, it ramps at body speed
YIELD_SLIDE_TAU = .4             # s of gaussian on the ask's slide: the seat ramps slower than the lean
SWING_COST = .25                 # per deg^2 of swing growth a candidate's cost takes: the ask scores both hands, and
                                 # a body move that heals one by shoving the other's elbow into a sweep is no heal
                                 # (.25/deg^2 makes a 10 deg sweep growth cost as much as a 32 deg angle breach)
SWING_ENV_DEG = 25               # deg of the necessary-swing study's envelope (p90 25 at touch level): a hand whose
                                 # baseline swing stays inside it must stay inside it under the ask -- the body's
                                 # yielding may not manufacture a sweep the outline never demanded
SWING_ENV_MAX_DEG = 30           # deg the same hand's worst single sample may reach under the ask: the p90 above
                                 # is the study's own measure, this bounds its tail (a 40 deg sweep was the wreck)
MAX_HONEST = 12                  # (lean, slide) pairs per span tried by a real rescan: the per-sample forecast
                                 # cannot see the arcs move under the curve, and the realized swing exceeds it
RENDER_RATE = 60                 # Hz the honest scorer reads the rendered angles at: the frames stage
                                 # interpolates the swing curve at the render's own rate (pipeline.FRAME_RATE),
                                 # and the grid's .05 s samples miss the dips between them
ASK_LOG = None                   # dev hook: a list to append the honest trials' (pair, cost, reject) rows to


def _ask_log(**row):
    if ASK_LOG is not None:
        ASK_LOG.append(row)


def hull2d(points):
    """The convex hull of 2 d points, counter-clockwise (elbow_room's)."""
    p = np.unique(np.round(points, 6), axis=0)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h
    return np.array(half(p)[:-1] + half(p[::-1])[:-1])


class Trunk:
    """The trunk's outline: hulls of 1 cm slices of its skin in the trunk bone's frame. A point's depth is how
    far in from the outline at its height it sits (positive inside, negative the clearance out). A slice too
    sparse for a hull is no hole: its band reads the nearer hulled slice's hull."""

    def __init__(self, local, radius):
        local = np.asarray(local, float)
        self.radius = np.asarray(radius, float).reshape(2, 2)       # per hand (upper arm, forearm)
        self.lo = float(local[:, AXIS].min())
        buckets = {}
        for k in range(int((local[:, AXIS].max() - self.lo) / SLICE) + 1):
            s = local[(local[:, AXIS] >= self.lo + k * SLICE) & (local[:, AXIS] < self.lo + (k + 1) * SLICE)][:, OTHERS]
            if len(s) >= 6:
                buckets[k] = hull2d(s)
        E = max(len(h) for h in buckets.values())
        self.keys = np.array(sorted(buckets))               # slices with a hull, lowest first
        self.origin, self.normal = np.zeros((len(self.keys), E, 2)), np.zeros((len(self.keys), E, 2))
        for i, (k, h) in enumerate(sorted(buckets.items())):
            e = np.roll(h, -1, axis=0) - h
            n = np.c_[e[:, 1], -e[:, 0]] / np.linalg.norm(e, axis=1)[:, None]
            if np.mean((h - h.mean(0)) @ n.T) < 0:
                n = -n
            m = len(h)
            self.origin[i, :m], self.normal[i, :m] = h, n
            self.origin[i, m:], self.normal[i, m:] = h[0], n[0]

    @classmethod
    def from_dict(cls, data):
        radius = data['radius']
        if isinstance(radius, dict):
            radius = [radius[h] for h in 'LR']
        return cls(data['local'], radius)

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    def depth(self, points, matrix):
        """Signed depth (m) of world ``points`` in the outline, ``matrix`` the trunk bone's world matrix: -1 off
        the outline's height range; a point between hulled slices reads the nearer slice's hull (a slice the
        skin's bones leave too sparse for a hull is no hole: the surface runs on between its neighbours)."""
        R, o = matrix[:3, :3], matrix[:3, 3]
        p = (np.asarray(points, float).reshape(-1, 3) - o) @ R
        k = np.floor((p[:, AXIS] - self.lo) / SLICE).astype(int)
        out = np.full(len(p), -1.)
        ok = (k >= self.keys[0]) & (k <= self.keys[-1])
        kk = k[ok]
        j = np.searchsorted(self.keys, kk)
        j = np.minimum(j, len(self.keys) - 1)
        left = self.keys[np.maximum(j - 1, 0)]
        near = np.where(np.abs(kk - left) <= np.abs(self.keys[j] - kk), np.maximum(j - 1, 0), j)
        out[ok] = np.min(((self.origin[near] - p[ok][:, None, OTHERS]) * self.normal[near]).sum(-1), axis=1)
        return out

    def swing(self, shoulder, wrist, l1, l2, side, matrix, hand):
        """The arm's capsules' depth per swing angle (shoulder and wrist kept): (angles, depth, the elbow's rise
        above the wrist). The free arcs are read off these by the caller (a clearance margin, or touching at all)."""
        center, radius, t, q = elbow_circle(shoulder, wrist, l1, l2, side)
        ang = np.radians(np.arange(-180., 180., ANGLE_STEP))
        e = center + radius * (np.cos(ang)[:, None] * t + np.sin(ang)[:, None] * q)
        s = np.asarray(shoulder, float)
        w = np.asarray(wrist, float)
        up = s + UP[:, None, None] * (e[None] - s)
        fo = e[None] + FO[:, None, None] * (w - e)[None]
        r_up, r_fo = self.radius[hand]
        d_up = self.depth(up, matrix).reshape(len(UP), -1).max(0) + r_up
        d_fo = self.depth(fo, matrix).reshape(len(FO), -1).max(0) + r_fo
        return ang, np.maximum(d_up, d_fo), e[:, 2] - w[2]


def _runs(free, ang):
    """The free mask's circular runs as (lo, hi) angle pairs, hi > lo (a span may cross +-pi)."""
    n = len(free)
    s = int(np.argmin(free))            # start at a gap: no run wraps in the rolled mask
    rolled = np.roll(free, -s)
    out, i = [], 0
    while i < n:
        if rolled[i]:
            j = i
            while j < n and rolled[j]:
                j += 1
            lo = float(ang[(i + s) % n])
            out.append((lo, lo + (j - i) * math.radians(ANGLE_STEP)))
            i = j
        else:
            i += 1
    return out


def _near(angle, anchor):
    """``angle`` shifted by whole turns to sit nearest ``anchor`` (the bounds are unwrapped along the curve)."""
    return angle + 2 * math.pi * round((anchor - angle) / (2 * math.pi))


def _choose(runs, target, prev):
    """The run this sample is bounded to: the previous one while it lasts (the arcs move slowly, the curve does
    not jump between them), else the one nearest the pull target; unwrapped around ``prev`` or the target."""
    anchor = (.5 * (prev[0] + prev[1])) if prev is not None else target
    if prev is not None:
        for lo, hi in runs:
            a = prev[0] + ((lo - prev[0]) % (2 * math.pi))       # the run starts inside the previous interval ...
            b = lo + ((prev[0] - lo) % (2 * math.pi))            # ... or the previous interval starts in the run
            if a <= prev[1] + 1e-9 or b <= hi + 1e-9:
                near = _near(lo, anchor)
                return near, near + (hi - lo)
    best, cost = None, None
    for lo, hi in runs:
        near = _near(lo, anchor)
        d = max(near - target, target - (near + hi - lo), 0.) if np.isfinite(target) else 0.
        if cost is None or d < cost:
            best, cost = (near, near + (hi - lo)), d
    return best


def _rows(n, step, jerk, acc, vel, pull, target):
    """The swing curve's sparse least squares: minimum jerk, damped acceleration, no needless speed, pulled to
    the natural hang where it is known (solve_path's rows, one dimension). What is damped is the deviation from
    the natural hang, not the curve: where the hang is known the smoothness rows apply to (x - target), so the
    curve carries the target's own motion -- the elbow rides each wrist hop like a rigid chain and its velocity
    dies as the wrist lands (no glide while the hand sits) -- and where it is unknown they apply to x itself.
    ``pull`` may be per sample: tracking tightens where the wrist itself is moving."""
    from scipy import sparse
    t = np.asarray(target, float)
    d3 = sparse.diags([-1., 3., -3., 1.], [0, 1, 2, 3], shape=(n - 3, n)) / step ** 3 / jerk
    d2 = sparse.diags([1., -2., 1.], [0, 1, 2], shape=(n - 2, n)) / step ** 2 / acc
    d1 = sparse.diags([-1., 1.], [0, 1], shape=(n - 1, n)) / step / vel
    has = np.isfinite(t)
    pull = np.broadcast_to(np.asarray(pull, float), (n,))
    pull_rows = sparse.diags(has.astype(float) / pull, 0, shape=(n, n)).tocsr()[has]
    A = sparse.vstack([d3, d2, d1, pull_rows]).tocsr()
    ok3 = np.convolve(has, np.ones(4, int), 'valid') == 4      # all four targets known: deviation rows
    ok2 = np.convolve(has, np.ones(3, int), 'valid') == 3
    ok1 = has[:-1] & has[1:]
    b = np.r_[np.where(ok3, d3 @ t, 0.),            # the row's rhs is the target's own derivative
              np.where(ok2, d2 @ t, 0.),
              np.where(ok1, d1 @ t, 0.),
              t[has] / pull[has]]
    return A, b


def _scan(torso, path, grid, arm, label, hand, trunk, clearance):
    """One hand's per-sample state over the grid: the natural hang (``target``), the free arcs the swing curve is
    bounded to (``lo``, ``hi``, unwrapped along the curve) and how much lean the body must give up for a free arc
    (``required``, 1 where none is needed or helps). The arcs are read off a ladder: a ``clearance`` margin out of
    the outline where a swing reaches it, else any swing that only keeps the capsules from running in, and where
    even that is beyond every swing the body leans less (the elbow is never raised out of the trunk's height
    range for this). A sample whose wrist is beyond the arm (the frames stage stretches it) has no bounds, no pull.
    The scan also returns each sample's ``necessary`` swing (rad): how far the natural hang sits outside the sample's
    free arc (0 where the hang is inside it, or where a pinned sample already takes its best possible point) -- the
    excess-swing metric subtracts it from the curve's deviation, for free.
    Where the wrist sits still the signals are smoothed (SMOOTH_TAU, shrinking to nothing where it moves): the
    body's own millimetre drift -- a lean ramp engaging as a passage starts -- is amplified near tangency into
    visible elbow sweeps that begin before the hand moves; smoothed, the elbow follows the body at body speed,
    and every smoothed bound is verified against the sample's own depth and rise (INTRUDE_OK / RISE_OK slack)."""
    n = len(grid)
    lo, hi = np.empty(n), np.empty(n)
    target, required, deep = np.full(n, np.nan), np.ones(n), np.full(n, np.nan)
    necessary, tight, prev = np.zeros(n), np.zeros(n, bool), None
    ws = np.full((n, 3), np.nan)        # the wrist per sample (the smoothing scales with its speed)
    fine = np.zeros(n, bool)            # the sample has real bounds (not stretched, not wholly free)
    depths = np.full((n, len(TRUNK_ANG_DEG)), np.nan)   # the capsule depth along the circle (the verify)
    rises = np.full((n, len(TRUNK_ANG_DEG)), np.nan)    # ... and the elbow's rise
    rise_caps = np.full(n, np.nan)
    for i, t in enumerate(grid):
        anchor = (.5 * (prev[0] + prev[1])) if prev is not None else 0.
        try:
            s = torso.shoulder_world(label, t)[:3, 3]
            w = np.asarray(path(t)[:3], float)
            m = torso.trunk_world(t)
            ws[i] = w
            target[i] = closest_elbow_angle(s, w, arm.l1, arm.l2, arm.side)
            ang, deep_i, rise = trunk.swing(s, w, arm.l1, arm.l2, arm.side, m, hand)
            depths[i], rises[i] = deep_i, rise
            rise_caps[i] = float(rise[int(np.argmin(np.abs(ang - target[i])))]) + RISE_LIMIT
        except ValueError:              # the wrist beyond the unstretched arm at this lean: the frames stage
            lo[i], hi[i] = anchor, anchor   # stretches it (angle unused there); the curve carries on unchanged
            prev = None
            continue
        deep[i] = float(deep_i.min())
        free = (deep_i <= -clearance) & (rise <= rise_caps[i])
        if free.all():                  # the whole circle is free (the arm far out): no bound at all
            lo[i], hi[i] = -np.inf, np.inf
            prev = None
            continue
        if not free.any():
            free = (deep_i <= 0.) & (rise <= rise_caps[i])      # the margin is beyond this pose: not running in is what there is
            tight[i] = free.any()
        if free.any():
            bound = _choose(_runs(free, ang), target[i], prev)
            lo[i], hi[i] = bound
            necessary[i] = max(0., bound[0] - target[i], target[i] - bound[1])
            prev = bound
            fine[i] = True
            continue
        # no swing keeps the capsules out of the outline: the body leans less until one does, the least yield
        now = torso._cap(t)
        for share in np.linspace(min(1., now), 0., CAP_LEVELS):
            try:
                s = torso.shoulder_world(label, t, cap=float(share))[:3, 3]
                m = torso.trunk_world(t, cap=float(share))
                target[i] = closest_elbow_angle(s, w, arm.l1, arm.l2, arm.side)
                ang, deep2, rise = trunk.swing(s, w, arm.l1, arm.l2, arm.side, m, hand)
            except ValueError:
                continue
            free = (deep2 <= 0.) & (rise <= rise_caps[i])
            if free.any():
                required[i] = float(share)
                bound = _choose(_runs(free, ang), target[i], prev)
                lo[i], hi[i] = bound
                necessary[i] = max(0., bound[0] - target[i], target[i] - bound[1])
                depths[i], rises[i] = deep2, rise
                prev = bound
                fine[i] = True
                break
        else:                           # no share opens an arc: the elbow takes the least deep swing there is
            best = float(ang[int(np.argmin(deep_i))])
            lo[i] = hi[i] = _near(best, anchor)
            depths[i], rises[i] = deep_i, rise
            prev = (lo[i], hi[i])
            fine[i] = True
    _relax(grid, lo, hi, target, fine, ws, depths, rises, rise_caps, clearance)
    for i in range(n):                  # the necessary swing of the (possibly smoothed) constraint
        if np.isfinite(target[i]) and np.isfinite(lo[i]) and np.isfinite(hi[i]):
            necessary[i] = max(0., lo[i] - target[i], target[i] - hi[i])
    return dict(lo=lo, hi=hi, target=target, required=required, deep=deep, necessary=necessary, tight=tight,
                depths=depths, rises=rises, rise_caps=rise_caps)


def _relax(grid, lo, hi, target, fine, ws, depths, rises, rise_caps, clearance):
    """Smooth the scan's constraint signals along the grid where the wrist sits still (the width shrinks to
    nothing with its speed; segments split where the signals restart). Each smoothed bound is verified against
    the sample's own depth and rise so the smoothing breaks the clearance no more than the slack."""
    n = len(grid)
    d = np.linalg.norm(np.diff(ws, axis=0), axis=1) / CURVE_STEP        # (n-1,) wrist speed, nan at gaps
    spd = np.full(n, np.inf)
    if n > 1:
        spd[0], spd[-1] = d[0], d[-1]
        spd[1:-1] = np.fmax(d[:-1], d[1:])      # either neighbour moving counts as moving
    tau = np.where(np.isfinite(spd), SMOOTH_TAU / (1. + spd / TRACK_RATE), SMOOTH_TAU)
    barrier = ~fine | ~np.isfinite(lo) | ~np.isfinite(hi) | ~np.isfinite(target)
    seg = np.zeros(n, int)
    seg[barrier] = -1
    for i in range(1, n):
        if seg[i] >= 0:
            seg[i] = seg[i - 1] + (1 if barrier[i - 1] else 0)
    W = int(np.ceil(SMOOTH_TAU / CURVE_STEP)) + 1
    for k, values in enumerate((lo, hi, target)):
        bounds = k < 2
        true = np.array(values, float)  # the unsmoothed constraint (the verify's reference)
        out = np.array(values, float)
        for i in range(n):
            if seg[i] < 0 or (bounds and not fine[i]) or not np.isfinite(values[i]):
                continue
            j0, j1 = max(0, i - W), min(n, i + W + 1)
            sl = slice(j0, j1)
            near = seg[sl] == seg[i]
            near &= fine[sl] if bounds else np.isfinite(true[sl])
            if not near[i - j0]:
                near[i - j0] = True
            wgt = np.exp(-.5 * ((grid[sl] - grid[i]) / tau[i]) ** 2) * near
            out[i] = float(np.sum(true[sl][near] * wgt[near]) / np.sum(wgt))
        if bounds:
            for i in range(n):          # verify each smoothed bound against the sample's own depth and rise
                if seg[i] < 0 or not fine[i] or out[i] == true[i]:
                    continue
                out[i] = _verified(out[i], true[i], depths[i], rises[i], rise_caps[i], clearance)
        values[:] = out


def _verified(smooth, true, depth, rise, rise_cap, clearance):
    """``smooth`` pulled back toward ``true`` until it breaks the sample's clearance no more than ``true`` plus
    the slack: near tangency the true edge is the deepest legal point, off it the geometry is shallow."""
    if not np.isfinite(depth).all() or not np.isfinite(rise).all() or not np.isfinite(rise_cap):
        return smooth
    xs = np.r_[TRUNK_ANG_DEG, 180.]     # the circle's sample grid closed up (the seam reads the first sample)
    depth, rise = np.r_[depth, depth[0]], np.r_[rise, rise[0]]
    span = true - smooth
    lim_d = float(np.interp(_wrap_deg(true), xs, depth)) + INTRUDE_OK
    lim_r = min(float(np.interp(_wrap_deg(true), xs, rise)) + RISE_OK, rise_cap)

    def breaks(a):
        return (float(np.interp(_wrap_deg(a), xs, depth)) > lim_d or
                float(np.interp(_wrap_deg(a), xs, rise)) > lim_r)
    if not breaks(smooth):
        return smooth
    a, b = 0., 1.                       # smooth + (true - smooth) * s: s = 1 is the true bound (legal by construction)
    for _ in range(8):
        m = (a + b) / 2
        if breaks(smooth + span * m):
            a = m
        else:
            b = m
    return smooth + span * b


def _wrap_deg(angle):
    """An unwrapped angle (rad) into the trunk sample grid's range, degrees in [-180, 180)."""
    return -180. + ((math.degrees(angle) + 180.) % 360.)


def _cap_curve(required, n):
    """The body's shared yield over time: pulled to no yield, never below what the samples ask, smooth over the
    spans it yields in (one body, both hands' asks minimized into ``required``)."""
    from .motion import chunked
    A, b = _rows(n, CURVE_STEP, JERK, ACC, VEL, CAP_PULL, np.ones(n))
    return chunked(A, b, required, np.ones(n))


def _ease(values, step, tau=YIELD_TAU):
    """A gaussian over time (``tau``): the ask's body motion ramps at body speed."""
    r = max(1, int(round(tau / step * 2.)))
    x = np.arange(-r, r + 1) * step
    w = np.exp(-.5 * (x / tau) ** 2)
    w /= w.sum()
    padded = np.concatenate([np.full(r, values[0]), values, np.full(r, values[-1])])
    return np.convolve(padded, w, mode='valid')


def _sample(curve, times):
    """The curve's swing angle (deg) at ``times``, read through the frames' own hermite: what this reports is what
    the render does."""
    from .motion import pchip_slopes, hermite_eval
    node = np.asarray(curve['times'], float)
    ang = np.asarray(curve['angles'], float)[:, None]
    sl = pchip_slopes(node, ang)
    return np.array([float(hermite_eval(node, ang, sl, t)[0]) for t in np.atleast_1d(np.asarray(times, float))])


def _damped(curve, tau=DAMP_TAU, sub=DAMP_SUB):
    """The swing curve resampled ``sub`` times finer and smoothed over time: the frames' hermite is C1, so the
    curve's .05 s knots step the acceleration (the elbow turns, stops turning and turns again within two frames, the
    stiff upper arm). A gaussian of ``tau`` -- zero phase, so the timing is kept -- removes those steps at the cost
    of ~1.5 deg of angle; the endpoint padding holds the clip's edges still."""
    node = np.asarray(curve['times'], float)
    if len(node) < 3:
        return curve
    step = float(np.median(np.diff(node)))
    h = step / sub
    fine = np.arange(node[0], node[-1] + h * .5, h)
    dense = _sample(curve, fine)
    r = max(1, int(round(tau / h * 2.)))
    x = np.arange(-r, r + 1) * h
    w = np.exp(-.5 * (x / tau) ** 2)
    w /= w.sum()
    # linear extrapolation, not endpoint copies: a copy pulls a ramp's level (measured .62 deg on a 1.5 deg/s ramp),
    # a line leaves it exactly as it was (the gaussian is symmetric and normalized)
    left = dense[0] + (dense[0] - dense[1]) * np.arange(r, 0, -1)
    right = dense[-1] + (dense[-1] - dense[-2]) * np.arange(1, r + 1)
    return dict(times=fine, angles=np.convolve(np.concatenate([left, dense, right]), w, mode='valid'))


def _rendered(torso, curve, times, path, h, arm, solver):
    """The rendered palm-vs-forearm angle at ``times`` (deg): the curve's phi (hermite-interpolated, as the
    frames stage does) swings the elbow while the path's own wrist and yaw/roll aim the palm -- motion
    .frame_hand hands the path's 5 dims to arm.pose verbatim, so this is what the user sees. NaN where the
    hand is beyond its arm."""
    from .kinematics import q_mat3
    from .arm import closest_elbow
    from .motion import pchip_slopes, hermite_eval
    sl = pchip_slopes(curve['times'], curve['angles'][:, None])
    out = np.full(len(times), np.nan)
    for k, t in enumerate(times):
        try:
            v = np.asarray(path(float(t)), float)
            s = torso.shoulder_world(h, float(t))[:3, 3]
            phi = float(hermite_eval(curve['times'], curve['angles'][:, None], sl, float(t))[0])
            e = closest_elbow(s, v[:3], arm.l1, arm.l2, arm.side, angle=phi)
            f = v[:3] - e
            pq = q_mat3(arm.palm_q(v[3], v[4])) @ np.array([0., 1., 0.])
            out[k] = 180. - math.degrees(math.acos(np.clip(float(pq @ f) / np.linalg.norm(f), -1., 1.)))
        except ValueError:
            continue
    return out


def _swing_of(data, curve):
    """The realized swing per sample: the curve's phi off the natural hang (0 where the hand is beyond its arm)."""
    tgt = np.asarray(data['target'], float)
    phi = np.asarray(curve['angles'], float)
    return np.where(np.isfinite(tgt), phi - tgt, 0.)


def _build(data, paths, grid):
    """The swing curves from one scan's bounds: the least-jerk phi inside them (solve's)."""
    from .motion import chunked
    n = len(grid)
    out = {}
    for h in 'LR':
        d = data[h]
        if n >= 8:
            pts = np.array([np.asarray(paths[h](t)[:3], float) for t in grid])
            speed = np.r_[np.linalg.norm(np.diff(pts, axis=0), axis=1)[0],
                          np.linalg.norm(np.diff(pts, axis=0), axis=1)] / CURVE_STEP
            A, b = _rows(n, CURVE_STEP, JERK, ACC, VEL, PULL / (1. + speed / TRACK_RATE), d['target'])
            phi = chunked(A, b, d['lo'], np.maximum(d['hi'], d['lo'] + 1e-12))
        else:
            phi = np.where(np.isfinite(d['target']), d['target'], 0.)
        out[h] = dict(times=grid, angles=phi)
    return out


def _wrist_ask(torso, data, curves, grid, paths, rig, trunk, clearance, solvers, rescan):
    """The wrist-deviation ask (torso.Torso.set_yield / set_slide_yield): where the swing keeping the elbow out
    of the outline would bend the rendered wrist past the user's floor (posture.PALM_FLOOR_DEG) -- the path's
    own yaw/roll are what the render stores (motion.frame_hand hands them to arm.pose verbatim, and across
    holds they sag off the corridor's local optimum) -- the body leans and slides instead of the wrist bending
    further. The search is per span, two passes: a cheap per-sample pass (greedy over a lean and a slide, both
    signs, YIELD_STEPS / YIELD_SLIDES, forecast by the moved body's own free arc, the scan's criterion) narrows
    the candidates; then each contiguous span takes ONE (lean, slide) pair -- one coherent body move, the arcs
    follow at body speed -- scored by an HONEST rescan (one scan+curve round at the frozen cap, the rendered
    angles and swings read off the rescanned curves, for the per-sample forecast cannot see the arcs move under
    the curve). A pair is admissible only where it never takes a hand the baseline kept clear below the floor,
    nor past the study's swing envelope (SWING_ENV_DEG) a hand whose baseline stayed inside it, and it must
    beat doing nothing on the span's summed cost (square breach below the floor, plus SWING_COST per deg^2 of
    swing growth). The least motion wins ties. Returns the per-sample raw (lean shift, slide offset), still
    to be eased."""
    from . import posture
    from .arm import closest_elbow, closest_elbow_angle
    n = len(grid)
    floor = posture.PALM_FLOOR_DEG
    ft = np.arange(float(grid[0]), float(grid[-1]) + 1. / RENDER_RATE * .5, 1. / RENDER_RATE)   # the render's axis
    base_ang_g = {h: _rendered(torso, curves[h], grid, paths[h], h, rig.arms[h], solvers[h]) for h in 'LR'}
    base_ang = {h: _rendered(torso, curves[h], ft, paths[h], h, rig.arms[h], solvers[h]) for h in 'LR'}
    base_sw = {h: _swing_of(data[h], curves[h]) for h in 'LR'}
    shield = {h: not (base_ang[h] < floor).any() for h in 'LR'}        # never breached: the ask keeps it so
    env = {h: np.abs(base_sw[h]).max() <= math.radians(SWING_ENV_DEG) for h in 'LR'}
    plan_slide = torso.slide
    room = np.zeros(n)
    for k in range(n):
        t = float(grid[k])
        here = float(plan_slide(t)) if callable(plan_slide) else float(plan_slide)
        room[k] = max(0., torso.slide_max - abs(here)) if torso.slide_max else 0.
    leans = [0.] + [sgn * m for m in YIELD_STEPS for sgn in (1., -1.)]
    slides = [0.] + ([sgn * m for m in YIELD_SLIDES for sgn in (1., -1.)] if torso.slide_max else [])

    def cheap(k, ln, sl):
        """The per-sample forecast at one body move: (breach cost, swings) or None (not a candidate)."""
        if abs(sl) > room[k]:
            return None             # beyond the seat's travel from where the plan has it
        t = float(grid[k])
        torso.set_yield(np.array([t - 1., t + 1.]), np.full(2, ln))
        torso.set_slide_yield(np.array([t - 1., t + 1.]), np.full(2, sl))
        cost, swings = 0., []
        for i, h in enumerate('LR'):
            arm, solver = rig.arms[h], solvers[h]
            try:
                v = np.asarray(paths[h](t), float)
                w = v[:3]
                s2 = torso.shoulder_world(h, t)[:3, 3]
                tgth = closest_elbow_angle(s2, w, arm.l1, arm.l2, arm.side)
                angb, deep2, rise2 = trunk.swing(s2, w, arm.l1, arm.l2, arm.side, torso.trunk_world(t), i)
                cap2 = float(rise2[int(np.argmin(np.abs(angb - tgth)))]) + RISE_LIMIT
                free2 = (deep2 <= -clearance) & (rise2 <= cap2)
                if not free2.any():
                    free2 = (deep2 <= 0.) & (rise2 <= cap2)   # not running in is what there is
                if not free2.any():
                    return None          # this move leaves no swing at all: not a candidate
                lo2, hi2 = _choose(_runs(free2, angb), tgth, None)
                realized2 = min(max(tgth, lo2), hi2)
                e2 = closest_elbow(s2, w, arm.l1, arm.l2, arm.side, angle=realized2)
                a = posture.palm_forearm_deg(solver, w, v[3], v[4], s2, elbow=e2)
            except ValueError:
                return None              # a hand beyond its arm at this body: not a candidate
            cost += max(0., floor - a) ** 2
            swings.append(realized2 - tgth)
        return cost, swings

    def grown(g, k):
        """The cheap forecast's full cost: the breach plus the swing growth past the baseline's."""
        return g[0] + SWING_COST * sum(max(0., abs(sw) - abs(base_sw[h][k])) ** 2
                                       for h, sw in zip('LR', g[1]))

    picks = [None] * n
    for k in range(n):
        if not ((base_ang_g['L'][k] < floor) or (base_ang_g['R'][k] < floor)):
            continue
        base_cost = sum(max(0., floor - base_ang_g[h][k]) ** 2 for h in 'LR')
        best = (0., 0., base_cost)
        for ln in leans:                            # the lean alone
            g = cheap(k, ln, 0.)
            if g is not None and grown(g, k) < best[2] - 1e-9:
                best = (ln, 0., grown(g, k))
        for sl in slides:                           # the slide alone
            g = cheap(k, 0., sl)
            if g is not None and grown(g, k) < best[2] - 1e-9:
                best = (0., sl, grown(g, k))
        if best[0] and best[1]:                     # the pair the two alone found
            g = cheap(k, best[0], best[1])
            if g is not None and grown(g, k) < best[2] - 1e-9:
                best = (best[0], best[1], grown(g, k))
        if best[2] < base_cost - 1e-9 and (best[0] or best[1]):
            picks[k] = (best[0], best[1])
    torso.clear_yield()
    torso.clear_slide_yield()

    def honest(p, run):
        """The pair's run-scoped cost by a real rescan, scored at the render's own frame rate (the grid's
        .05 s samples miss the dips between them), or None where it breaks a guard."""
        ln, sl = p
        rows = np.zeros(n)
        rows[run] = ln
        srows = np.zeros(n)
        srows[run] = sl
        torso.set_yield(grid, _ease(rows, CURVE_STEP))
        torso.set_slide_yield(grid, _ease(srows, CURVE_STEP, YIELD_SLIDE_TAU))
        try:
            cdata, ccurves, _ = rescan()
            kf = (ft >= float(grid[run[0]]) - .5 / RENDER_RATE) & (ft <= float(grid[run[-1]]) + .5 / RENDER_RATE)
            cost = 0.
            ks = np.asarray(run)
            for h in 'LR':
                ang = _rendered(torso, ccurves[h], ft, paths[h], h, rig.arms[h], solvers[h])
                sw = _swing_of(cdata[h], ccurves[h])
                if shield[h] and ((ang[kf] < floor - 1e-9).any() or (ang[~kf] < floor - 1e-9).any()
                                  or (np.isfinite(base_ang[h]) & ~np.isfinite(ang)).any()):
                    _ask_log(pair=p, span=(run[0], run[-1]), hand=h, cost=None,
                             why='breach', min_ang=round(float(np.nanmin(ang)), 1))
                    return None         # a hand the baseline kept clear stays clear, tails included
                if env[h] and (np.percentile(np.abs(sw[ks]), 90) > math.radians(SWING_ENV_DEG) + 1e-12
                               or np.abs(sw[ks]).max() > math.radians(SWING_ENV_MAX_DEG) + 1e-12):
                    _ask_log(pair=p, span=(run[0], run[-1]), hand=h, cost=None, why='envelope',
                             max_swing=round(float(np.degrees(np.abs(sw[ks]).max())), 1),
                             p90_swing=round(float(np.degrees(np.percentile(np.abs(sw[ks]), 90))), 1))
                    return None         # a hand inside the study's envelope stays inside it
                fin = np.isfinite(ang[kf])
                cost += float(np.sum(np.maximum(0., floor - ang[kf][fin]) ** 2))
                cost += SWING_COST * float(np.sum(np.maximum(0., np.abs(sw[ks]) - np.abs(base_sw[h][ks])) ** 2))
            _ask_log(pair=p, span=(run[0], run[-1]), cost=round(cost, 1),
                     breach_frames=int((ang[kf] < floor).sum()))
            return cost
        finally:
            torso.clear_yield()
            torso.clear_slide_yield()

    ask = np.zeros(n)
    sask = np.zeros(n)
    trig = [k for k in range(n) if picks[k] is not None]
    runs = []
    for k in trig:                              # one span: a gap of a few samples is still one body move
        if runs and k - runs[-1][-1] <= 4:
            runs[-1].append(k)
        else:
            runs.append([k])
    ladder = [(0., .05), (0., .08), (0., .11), (-.016, .05), (-.016, .08), (-.032, .05), (-.032, .08)]
    for run in runs:
        seen = []
        for k in run:
            if picks[k] not in seen:
                seen.append(picks[k])
        cands = seen + [p for p in ladder if p not in seen]
        if len(cands) > MAX_HONEST:             # the cheap forecast only prunes a pathological pool
            worst = sorted(run, key=lambda k: min(base_ang_g['L'][k], base_ang_g['R'][k]))[:3]
            cands.sort(key=lambda p: sum(1e9 if (g := cheap(k, *p)) is None else grown(g, k) for k in worst))
            cands = cands[:MAX_HONEST]
        cands.sort(key=lambda p: abs(p[0]) + abs(p[1]))     # least motion first: it wins ties
        kf = (ft >= float(grid[run[0]]) - .5 / RENDER_RATE) & (ft <= float(grid[run[-1]]) + .5 / RENDER_RATE)
        base_span = float(sum(float(np.sum(np.maximum(0., floor - base_ang[h][kf][np.isfinite(base_ang[h][kf])]) ** 2))
                              for h in 'LR'))
        best_pair, best_cost = None, base_span
        for p in cands:
            c = honest(p, run)
            if c is not None and c < best_cost - 1e-9:
                best_cost, best_pair = c, p
        if best_pair is not None:
            _ask_log(chosen=best_pair, span=(run[0], run[-1]), base=round(base_span, 1), cost=round(best_cost, 1))
            ask[run] = best_pair[0]
            sask[run] = best_pair[1]
    return ask, sask


def solve(torso, paths, t0, t1, rig, trunk, clearance=CLEARANCE, solvers=None):
    """Both arms' swing curves over the piece (times t0..t1): ({hand: dict(times, angles)}, report). The torso's
    cap is set on ``torso`` (the body yields where no swing clears), and where the forced swing would bend the
    rendered wrist past the comfort floor the body leans and slides instead (the wrist-deviation ask,
    torso.set_yield / set_slide_yield, ``solvers`` given); the scans then re-read the moved body. The ask is
    searched against the settled ask-free solution (the honest baseline) and accepted only by a real rescan.
    The frames stage interpolates ``angles`` and hands them to arm.Arm.pose."""
    from .motion import chunked
    grid = np.arange(t0, t1 + CURVE_STEP * .5, CURVE_STEP)
    n = len(grid)
    caps = np.ones(n)
    data, curves, report = {}, {}, {}
    asked = 0

    def scan_build():
        torso.set_cap(grid, caps)       # the scans read the shared cap from the torso
        required = np.ones(n)
        d = {}
        for i, h in enumerate('LR'):
            d[h] = _scan(torso, paths[h], grid, rig.arms[h], h, i, trunk, clearance)
            required = np.minimum(required, d[h]['required'])
        return d, _build(d, paths, grid), required

    # 1. the ask-free solution settles the body-yield (the honest baseline the ask is searched against)
    for round_ in range(CAP_ROUNDS + 2):
        data, curves, required = scan_build()
        if required.min() >= 1. or round_ == CAP_ROUNDS + 1:
            break
        caps = _cap_curve(required, n)  # the cap smooths over spans; the next round re-checks both hands on it
    if solvers is not None:
        ys, ss = _wrist_ask(torso, data, curves, grid, paths, rig, trunk, clearance, solvers, scan_build)
        if ys.any() or ss.any():
            ys = _ease(ys, CURVE_STEP)
            torso.set_yield(grid, ys)       # the ask rides on the re-check: the next scans read it
            if ss.any():
                torso.set_slide_yield(grid, _ease(ss, CURVE_STEP, YIELD_SLIDE_TAU))
            asked = int(((ys != 0.) | (ss != 0.)).sum())
            # 2. the cap re-checks on the moved body
            for round_ in range(CAP_ROUNDS + 2):
                data, curves, required = scan_build()
                if required.min() >= 1. or round_ == CAP_ROUNDS + 1:
                    break
                caps = _cap_curve(required, n)
    damped = {h: _damped(curves[h]) for h in 'LR'}   # what the frames will read; the report below follows it
    ym = float(np.abs(torso.yield_rows[1]).max() * 1000.) if asked else 0.
    sm = float(np.abs(torso.slide_yield_rows[1]).max() * 1000.) if asked and torso.slide_yield_rows is not None else 0.
    for i, h in enumerate('LR'):
        d = data[h]
        off = np.abs(_sample(damped[h], grid) - d['target'])[np.isfinite(d['target'])]
        ex = np.degrees(np.maximum(off - d['necessary'][np.isfinite(d['target'])], 0.))
        report[h] = dict(samples=n, blocked=int((d['required'] < 1.).sum()), tight=int(d['tight'].sum()),
                         stretch=int(np.isnan(d['target']).sum()),
                         yield_min=round(float(caps.min()), 2), asked=asked, yield_max_mm=round(ym, 1),
                         slide_max_mm=round(sm, 1),
                         swing_p95_deg=round(float(np.degrees(np.percentile(off, 95))), 1) if len(off) else 0.,
                         excess_p50_deg=round(float(np.percentile(ex, 50)), 1) if len(ex) else 0.,
                         excess_p95_deg=round(float(np.percentile(ex, 95)), 1) if len(ex) else 0.,
                         excess_max_deg=round(float(ex.max()), 1) if len(ex) else 0.,
                         clear_p10_mm=round(float(np.nanpercentile(d['deep'], 10) * 1000), 1) if np.isfinite(d['deep']).any() else 0.)
    return damped, report


def load(rig_path):
    """The trunk outline beside a rig.json (rig_export writes it there), or None without one -- or with one traced
    from another rig: an outline from a different seat, character or export clears a body that is not there, and
    clearing the wrong body is worse than not clearing at all (no outline, no swing, as before the export existed).
    Pairs are checked on the rig's sha256 stamp; outlines that predate it on the trunk bone's world matrix, which
    both files already carry."""
    path = Path(rig_path).with_name('trunk.json')
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    stale = _stale_reason(data, Path(rig_path))
    if stale:
        print(f'TRUNK_STALE {path}: {stale}', flush=True)
        return None
    return Trunk.from_dict(data)


def _stale_reason(data, rig_path):
    """Why the outline and the rig do not belong together, or None when they do (or when the older outline carries
    nothing to check against: a stale-looking pair is no proof of a stale one)."""
    want = data.get('rig_sha256')
    if want:
        try:
            have = hashlib.sha256(rig_path.read_bytes()).hexdigest()
        except OSError:
            return f'{rig_path} is unreadable'
        return None if have == want else f'traced from rig {want[:12]}, this rig is {have[:12]}'
    rig = json.loads(rig_path.read_text(encoding='utf-8'))
    traced, bone = data.get('matrix'), data.get('bone')
    neutral, world = rig.get('neutral_bones', {}).get(bone), rig.get('arm_world')
    if traced is None or neutral is None or world is None:
        return None
    seat = np.asarray(world, float) @ np.asarray(neutral, float)
    gap = float(np.abs(np.asarray(traced, float) - seat).max())
    return None if gap < 1e-3 else f'traced at another seat (trunk bone {gap * 1000:.0f} mm off)'
