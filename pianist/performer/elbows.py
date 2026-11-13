"""The elbows kept out of the body.

The trunk's outline (trunk.json, exported by rig_export.neutral_scene: its skin in the trunk bone's frame as
convex hulls of 1 cm slices, and the arms' skin radii about their bones) hangs on the trunk bone's world matrix
the lean produces. On the elbow's IK circle (shoulder and wrist kept, arm.py.elbow_circle) the arm's two
capsules are sampled at every swing angle; the free arcs are where they clear the outline. Per arm one smooth
swing-angle curve for the whole piece is solved like the wrist paths (bounded least squares over windows,
motion.chunked): inside the free arcs, pulled toward the elbow's natural hang. Where no swing clears the trunk
the upper body leans less instead (the torso's cap, torso.Torso.set_cap: the body yields before the elbow is
raised above its natural hang), searched per sample and smoothed over time.

Solved once in the main process between the finger keyposes and the frames; the frames stage interpolates the
curve and hands the angle to arm.Arm.pose.
"""
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
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        radius = data['radius']
        if isinstance(radius, dict):
            radius = [radius[h] for h in 'LR']
        return cls(data['local'], radius)

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


def solve(torso, paths, t0, t1, rig, trunk, clearance=CLEARANCE):
    """Both arms' swing curves over the piece (times t0..t1): ({hand: dict(times, angles)}, report). The torso's
    cap is set on ``torso`` (the body yields where no swing clears); the frames stage interpolates ``angles`` and
    hands them to arm.Arm.pose."""
    from .motion import chunked
    grid = np.arange(t0, t1 + CURVE_STEP * .5, CURVE_STEP)
    n = len(grid)
    caps = np.ones(n)
    data, curves, report = {}, {}, {}
    for round_ in range(CAP_ROUNDS + 1):
        torso.set_cap(grid, caps)       # the scans read the shared yield from the torso
        required = np.ones(n)
        for i, h in enumerate('LR'):
            data[h] = _scan(torso, paths[h], grid, rig.arms[h], h, i, trunk, clearance)
            required = np.minimum(required, data[h]['required'])
        if required.min() >= 1. or round_ == CAP_ROUNDS:
            break
        caps = _cap_curve(required, n)  # the yield smooths over spans; the next round re-checks both hands on it
    for i, h in enumerate('LR'):
        d = data[h]
        if n >= 8:
            pts = np.array([np.asarray(paths[h](t)[:3], float) for t in grid])
            speed = np.r_[np.linalg.norm(np.diff(pts, axis=0), axis=1)[0],
                          np.linalg.norm(np.diff(pts, axis=0), axis=1)] / CURVE_STEP
            A, b = _rows(n, CURVE_STEP, JERK, ACC, VEL, PULL / (1. + speed / TRACK_RATE), d['target'])
            phi = chunked(A, b, d['lo'], np.maximum(d['hi'], d['lo'] + 1e-12))
        else:
            phi = np.where(np.isfinite(d['target']), d['target'], 0.)
        curves[h] = dict(times=grid, angles=phi)
        off = np.abs(phi - d['target'])[np.isfinite(d['target'])]
        ex = np.degrees(np.maximum(off - d['necessary'][np.isfinite(d['target'])], 0.))
        report[h] = dict(samples=n, blocked=int((d['required'] < 1.).sum()), tight=int(d['tight'].sum()),
                         stretch=int(np.isnan(d['target']).sum()),
                         yield_min=round(float(caps.min()), 2),
                         swing_p95_deg=round(float(np.degrees(np.percentile(off, 95))), 1) if len(off) else 0.,
                         excess_p50_deg=round(float(np.percentile(ex, 50)), 1) if len(ex) else 0.,
                         excess_p95_deg=round(float(np.percentile(ex, 95)), 1) if len(ex) else 0.,
                         excess_max_deg=round(float(ex.max()), 1) if len(ex) else 0.,
                         clear_p10_mm=round(float(np.nanpercentile(d['deep'], 10) * 1000), 1) if np.isfinite(d['deep']).any() else 0.)
    return curves, report


def load(rig_path):
    """The trunk outline beside a rig.json (rig_export writes it there), or None without one."""
    path = Path(rig_path).with_name('trunk.json')
    return Trunk.from_json(path) if path.exists() else None
