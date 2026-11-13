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
    the natural hang where it is known (solve_path's rows, one dimension)."""
    from scipy import sparse
    d3 = sparse.diags([-1., 3., -3., 1.], [0, 1, 2, 3], shape=(n - 3, n)) / step ** 3 / jerk
    d2 = sparse.diags([1., -2., 1.], [0, 1, 2], shape=(n - 2, n)) / step ** 2 / acc
    d1 = sparse.diags([-1., 1.], [0, 1], shape=(n - 1, n)) / step / vel
    has = np.isfinite(target)
    pull_rows = sparse.diags(has.astype(float) / pull, 0, shape=(n, n)).tocsr()[has]
    A = sparse.vstack([d3, d2, d1, pull_rows]).tocsr()
    b = np.r_[np.zeros(n - 3), np.zeros(n - 2), np.zeros(n - 1), target[has] / pull]
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
    excess-swing metric subtracts it from the curve's deviation, for free."""
    n = len(grid)
    lo, hi = np.empty(n), np.empty(n)
    target, required, deep = np.full(n, np.nan), np.ones(n), np.full(n, np.nan)
    necessary, tight, prev = np.zeros(n), np.zeros(n, bool), None
    for i, t in enumerate(grid):
        anchor = (.5 * (prev[0] + prev[1])) if prev is not None else 0.
        try:
            s = torso.shoulder_world(label, t)[:3, 3]
            w = np.asarray(path(t)[:3], float)
            m = torso.trunk_world(t)
            target[i] = closest_elbow_angle(s, w, arm.l1, arm.l2, arm.side)
            ang, deep_i, rise = trunk.swing(s, w, arm.l1, arm.l2, arm.side, m, hand)
            rise_cap = float(rise[int(np.argmin(np.abs(ang - target[i])))]) + RISE_LIMIT
        except ValueError:              # the wrist beyond the unstretched arm at this lean: the frames stage
            lo[i], hi[i] = anchor, anchor   # stretches it (angle unused there); the curve carries on unchanged
            prev = None
            continue
        deep[i] = float(deep_i.min())
        free = (deep_i <= -clearance) & (rise <= rise_cap)
        if free.all():                  # the whole circle is free (the arm far out): no bound at all
            lo[i], hi[i] = -np.inf, np.inf
            prev = None
            continue
        if not free.any():
            free = (deep_i <= 0.) & (rise <= rise_cap)      # the margin is beyond this pose: not running in is what there is
            tight[i] = free.any()
        if free.any():
            bound = _choose(_runs(free, ang), target[i], prev)
            lo[i], hi[i] = bound
            necessary[i] = max(0., bound[0] - target[i], target[i] - bound[1])
            prev = bound
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
            free = (deep2 <= 0.) & (rise <= rise_cap)
            if free.any():
                required[i] = float(share)
                bound = _choose(_runs(free, ang), target[i], prev)
                lo[i], hi[i] = bound
                necessary[i] = max(0., bound[0] - target[i], target[i] - bound[1])
                prev = bound
                break
        else:                           # no share opens an arc: the elbow takes the least deep swing there is
            best = float(ang[int(np.argmin(deep_i))])
            lo[i] = hi[i] = _near(best, anchor)
            prev = (lo[i], hi[i])
    return dict(lo=lo, hi=hi, target=target, required=required, deep=deep, necessary=necessary, tight=tight)


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
            A, b = _rows(n, CURVE_STEP, JERK, ACC, VEL, PULL, d['target'])
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
