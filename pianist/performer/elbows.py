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
import os
import time
from pathlib import Path

import numpy as np

from . import elbows_fast
from .arm import closest_elbow_angle, elbow_circle
from .torso import REACH_FULL

SLICE = .01                      # m of outline per hull slice, along the trunk bone's y
AXIS, OTHERS = 1, (0, 2)         # the slices run along the bone's y; each hull lies in its x-z
ANGLE_STEP = 3.                  # deg between the sampled swing angles on the elbow's circle
TRUNK_ANG_DEG = np.arange(-180., 180., ANGLE_STEP)      # the sampled circle, degrees (the verify's interp grid)
TRUNK_ANG = np.radians(TRUNK_ANG_DEG)                   # ... and radians (the scan's, once instead of per sample)
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
DWELL_SAMPLES = 5                # bounds samples a constraint must hold for (~0.2 s) before the solve believes it
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

# Where the scan's time goes, for the full-piece runs (PIANOMOTION_ELBOW_PROFILE=1): a scan is two hands of hull
# tests, a build is one least-squares over the piece, and the wrist ask runs a full scan+build per candidate pair
# per run -- the counts say which of the two the piece is paying for.
PROFILE = os.environ.get('PIANOMOTION_ELBOW_PROFILE') == '1'
_prof = {}


def _tick(key, dt=0.):
    if PROFILE:
        hit = _prof.get(key)
        _prof[key] = [hit[0] + 1, hit[1] + dt] if hit else [1, dt]


def _profile_line():
    if not (PROFILE and _prof):
        return None
    return 'ELBOW_PROFILE ' + json.dumps(profile_snapshot())


def profile_snapshot():
    """Diagnostics-only read of the per-substep call/seconds counters, or None if profiling was off.
    Exposed so bench/dev tools can drop structured hotspot data into JSON instead of scraping stdout."""
    if not (PROFILE and _prof):
        return None
    return {k: dict(calls=v[0], seconds=round(v[1], 2)) for k, v in sorted(_prof.items())}


def _ask_log(**row):
    if ASK_LOG is not None:
        ASK_LOG.append(row)


def _cross2(a, b):
    """z of the 2 d cross product (numpy 2 dropped ``np.cross`` for 2 d vectors)."""
    return float(a[0] * b[1] - a[1] * b[0])


def hull2d(points):
    """The convex hull of 2 d points, counter-clockwise (elbow_room's)."""
    p = np.unique(np.round(points, 6), axis=0)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and _cross2(h[-1] - h[-2], q - h[-2]) <= 0:
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
        self.step = SLICE
        self._fast = None                                           # the compiled kernel's arrays, built on first use
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

    def _packed(self):
        """The kernel's copies of the hull tables, built once (None without numba: the numpy path runs)."""
        hit = self._fast
        if hit is None:
            hit = elbows_fast.pack(self) if elbows_fast.available() else False
            self._fast = hit
        return hit or None

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
        above the wrist). The free arcs are read off these by the caller (a clearance margin, or touching at all).
        One sample: the compiled kernel takes it as a block of one (elbows_fast), the numpy body below is the
        fallback where numba is missing and the reference the kernel is checked against."""
        packed = self._packed()
        if packed is not None:
            m = np.asarray(matrix, float)
            depth, rise, ok = self.swing_block(np.asarray(shoulder, float)[None], np.asarray(wrist, float)[None],
                                               m[None, :3, :3], m[None, :3, 3], l1, l2, side, hand)
            if not ok[0]:
                raise ValueError('Arm target outside unstretched reach')
            return TRUNK_ANG, depth[0], rise[0]
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

    def swing_block(self, S, W, R, O, l1, l2, side, hand):
        """``swing`` for a whole scan at once: (depth, rise, ok), each (n, angles) / (n,) -- the compiled kernel,
        one sample per thread. ``S``/``W`` the shoulders and wrists (n, 3), ``R``/``O`` the trunk matrices'
        rotation and origin (n, 3, 3) / (n, 3). ``ok`` is False where the wrist is beyond the unstretched arm
        (swing raises there; a scan wants the whole grid's answer and its own way of handling the stretched
        sample -- the frames stage stretches it and the angle is unused)."""
        r_up, r_fo = self.radius[hand]
        return elbows_fast.swing_block(self._packed(), S, W, R, O, l1, l2, side, r_up, r_fo,
                                       np.cos(TRUNK_ANG), np.sin(TRUNK_ANG), UP, FO)


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
    not jump between them), else the one nearest the pull target; unwrapped around ``prev`` or the target.
    Where the previous interval reaches into MORE than one run the continuity argument says nothing -- and taking
    the first one in angle order is arbitrary: canon 200.20-200.35 had the natural hang's own arc free (distance
    0 deg) beside a 6-12 deg sliver 60-73 deg below it, and the curve dove into the sliver, swung 70 deg in .2 s
    and sprang back (the user's 6145-6190). Nearest the hang wins there; the single-overlap case is unchanged."""
    anchor = (.5 * (prev[0] + prev[1])) if prev is not None else target
    if prev is not None:
        cands = []
        for lo, hi in runs:
            a = prev[0] + ((lo - prev[0]) % (2 * math.pi))       # the run starts inside the previous interval ...
            b = lo + ((prev[0] - lo) % (2 * math.pi))            # ... or the previous interval starts in the run
            if a <= prev[1] + 1e-9 or b <= hi + 1e-9:
                near = _near(lo, anchor)
                cands.append((near, near + (hi - lo)))
        if cands:
            if len(cands) == 1 or not np.isfinite(target):
                return cands[0]
            return min(cands, key=lambda c: max(c[0] - target, target - c[1], 0.))
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


def _trace(torso, path, grid, label, kin, moved):
    """The scan's per-sample kinematics -- the wrist, the shoulder, the trunk's world matrix, and whether the arm
    reaches at all -- traced once per paths and cap round (``kin``, keyed by hand) and thereafter only over
    ``moved``: the wrist-deviation ask shifts the body inside its own span, so every rescan re-traces a few dozen
    samples instead of the piece's thousands. This pass is the single most expensive thing in the scan: each sample
    builds the spine's chain twice (shoulder_world, trunk_world)."""
    n = len(grid)
    got = None if kin is None else kin.get(label)
    if got is None or got['n'] != n or got['path'] is not path or got['grid'] is not grid:
        got = dict(n=n, path=path, grid=grid, S=np.full((n, 3), np.nan), W=np.full((n, 3), np.nan),
                   M=np.zeros((n, 4, 4)), have=np.zeros(n, bool))
        if kin is not None:
            kin[label] = got
        todo = range(n)
    else:
        todo = range(n) if moved is None else (int(i) for i in np.flatnonzero(moved))
    for i in todo:
        t = float(grid[i])
        try:
            S_i = torso.shoulder_world(label, t)[:3, 3]
            W_i = np.asarray(path(t)[:3], float)
            M_i = torso.trunk_world(t)
        except ValueError:              # the wrist beyond the unstretched arm at this lean: the frames stage
            got['have'][i] = False      # stretches it (angle unused there); the curve carries on unchanged
            continue
        got['S'][i], got['W'][i], got['M'][i] = S_i, W_i, M_i
        got['have'][i] = True
    return got['S'], got['W'], got['M'], got['have']


def _least_lean(torso, t, wrist, label, hand, arm, trunk, prev, levels=CAP_LEVELS):
    """The least lean (share of the cap, 0..1) at which ``wrist`` is inside the arm's length AND some swing of the
    elbow clears the trunk: (share, bound, depth, rise, target), or None when even the upright body cannot reach
    it. The elbow solver's rule is that the body yields before the arm breaks (elbows.solve); a hand parked on the
    lap through the other hand's far passage is the case that rule was missing -- there the body had leaned until
    this arm was dead straight, and a straight arm's elbow is solved through the chest."""
    for share in np.linspace(min(1., torso._cap(t)), 0., levels):
        try:
            s = torso.shoulder_world(label, t, cap=float(share))[:3, 3]
            m = torso.trunk_world(t, cap=float(share))
            ang, deep, rise = trunk.swing(s, wrist, arm.l1, arm.l2, arm.side, m, hand)
        except ValueError:                  # still beyond the arm at this lean: lean less
            continue
        free = deep <= 0.
        if not free.any():
            continue
        target = closest_elbow_angle(s, wrist, arm.l1, arm.l2, arm.side)
        return float(share), _choose(_runs(free, ang), target, prev), deep, rise, target
    return None


def _scan(torso, path, grid, arm, label, hand, trunk, clearance, kin=None, moved=None):
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
    st = None if kin is None else kin.get(('state', label))
    fresh = st is None or st['n'] != n
    if fresh:
        st = dict(n=n, have=np.zeros(n, bool), fine=np.zeros(n, bool), tight=np.zeros(n, bool),
                  raw_lo=np.zeros(n), raw_hi=np.zeros(n), raw_target=np.full(n, np.nan),
                  lo=np.zeros(n), hi=np.zeros(n), target=np.full(n, np.nan),
                  deep=np.full(n, np.nan), necessary=np.zeros(n), required=np.ones(n),
                  ws=np.full((n, 3), np.nan), depths=np.full((n, len(TRUNK_ANG_DEG)), np.nan),
                  rises=np.full((n, len(TRUNK_ANG_DEG)), np.nan), rise_caps=np.full(n, np.nan))
        if kin is not None:
            kin[('state', label)] = st
    # The scan's own answer is kept (``st``): a rescan of the same paths and cap only re-derives the samples the
    # body moved on -- the ask's span -- and smooths (_relax) only around them.
    lo, hi, target = st['lo'], st['hi'], st['target']
    raw_lo, raw_hi, raw_target = st['raw_lo'], st['raw_hi'], st['raw_target']
    fine, tight, required, necessary = st['fine'], st['tight'], st['required'], st['necessary']
    ws, depths, rises, rise_caps, deep = st['ws'], st['depths'], st['rises'], st['rise_caps'], st['deep']
    todo = np.arange(n) if (moved is None or fresh) else np.flatnonzero(moved)
    mask = np.zeros(n, bool)
    mask[todo] = True
    # One pass of kinematics (shoulder, wrist, trunk matrix -- cheap), then the whole grid's hull tests in one
    # block: the samples are independent, so the compiled kernel spreads them over the cores instead of the
    # numpy path's per-sample temporaries. Without numba the block is skipped and the loop below swings one
    # sample at a time, exactly as before.
    S, W, M, have = _trace(torso, path, grid, label, kin, None if fresh else moved)
    st['have'][todo] = have[todo]
    reach = st['have'].copy()           # the traced reachability; the swing's verdict on top of it is this scan's
    for i in todo[reach[todo]]:
        ws[i] = W[i]
    ang = TRUNK_ANG
    if trunk._packed() is not None and (reach & mask).any():
        idx = np.flatnonzero(reach & mask)
        blk, ris, ok = trunk.swing_block(S[idx], W[idx], M[idx, :3, :3], M[idx, :3, 3],
                                         arm.l1, arm.l2, arm.side, hand)
        depths[idx], rises[idx] = blk, ris
        reach[idx[~ok]] = False         # swing would have raised there: the scan's stretched sample, below
    prev = None
    if not fresh and len(todo) and todo[0] > 0:     # the bound the sample before the span carried, unwrapped
        j = int(todo[0]) - 1
        if fine[j] and reach[j]:
            prev = (raw_lo[j], raw_hi[j])
    for i in todo:
        i = int(i)
        t = float(grid[i])
        anchor = (.5 * (prev[0] + prev[1])) if prev is not None else 0.
        required[i] = 1.
        if not reach[i]:
            raw_lo[i], raw_hi[i] = anchor, anchor
            fine[i] = False
            prev = None
            continue
        s, w = S[i], W[i]
        # The lean may not carry a hand away from its own wrist past a comfortable reach. A hand parked on the
        # lap while the other plays far out is the case: the lean takes its shoulder across the body, the arm
        # goes dead straight, and its elbow has no freedom left -- it is solved into the chest. (The hand the
        # lean is FOR is the other case: there the shoulder comes toward its wrist, nothing to cap.)
        limit = REACH_FULL * (arm.l1 + arm.l2)
        if np.linalg.norm(w - s) > limit:
            still = torso.shoulder_world(label, t, cap=0.)[:3, 3]
            if float(np.dot(s - still, w - s)) < 0.:
                for share in np.linspace(torso._cap(t), 0., CAP_LEVELS):
                    moved = torso.shoulder_world(label, t, cap=float(share))[:3, 3]
                    if np.linalg.norm(w - moved) <= limit:
                        required[i] = float(share)
                        s, M[i] = moved, torso.trunk_world(t, cap=float(share))
                        break
        if trunk._packed() is None:     # no numba: the one-sample swing, as the scan always did
            try:
                _, deep_i, rise = trunk.swing(s, w, arm.l1, arm.l2, arm.side, M[i], hand)
            except ValueError:
                # The body leaned away from this hand until its wrist sits beyond the arm's length (the other
                # hand's lean carried this shoulder off while the wrist stayed put -- a hand parked on the lap
                # through the other hand's high passage). The frames stage would stretch the arm, and a
                # stretched arm has no elbow freedom: the elbow is solved straight through the chest. The body
                # yields instead -- the least lean at which this wrist is inside the arm AND a swing clears the
                # trunk (a passage where the other hand is far out has the slack for it; where it has not, the
                # stretch stands as before).
                got = _least_lean(torso, t, w, label, hand, arm, trunk, prev)
                if got is None:
                    raw_lo[i], raw_hi[i] = anchor, anchor
                    fine[i] = False
                    prev = None
                    continue
                share, bound, deep_i, rise, raw_target[i] = got
                required[i] = share
                raw_lo[i], raw_hi[i] = bound
                depths[i], rises[i] = deep_i, rise
                ws[i] = w
                prev = bound
                fine[i] = True
                continue
            depths[i], rises[i] = deep_i, rise
        else:
            deep_i, rise = depths[i], rises[i]
        raw_target[i] = closest_elbow_angle(s, w, arm.l1, arm.l2, arm.side)
        rise_caps[i] = float(rise[int(np.argmin(np.abs(ang - raw_target[i])))]) + RISE_LIMIT
        deep[i] = float(deep_i.min())
        free = (deep_i <= -clearance) & (rise <= rise_caps[i])
        if free.all():                  # the whole circle is free (the arm far out): no bound at all
            raw_lo[i], raw_hi[i] = -np.inf, np.inf
            fine[i] = False
            prev = None
            continue
        if not free.any():
            free = (deep_i <= 0.) & (rise <= rise_caps[i])      # the margin is beyond this pose: not running in is what there is
            tight[i] = free.any()
        if free.any():
            bound = _choose(_runs(free, ang), raw_target[i], prev)
            raw_lo[i], raw_hi[i] = bound
            prev = bound
            fine[i] = True
            continue
        # no swing keeps the capsules out of the outline: the body leans less until one does, the least yield
        now = torso._cap(t)
        for share in np.linspace(min(1., now), 0., CAP_LEVELS):
            try:
                s = torso.shoulder_world(label, t, cap=float(share))[:3, 3]
                m = torso.trunk_world(t, cap=float(share))
                raw_target[i] = closest_elbow_angle(s, w, arm.l1, arm.l2, arm.side)
                ang, deep2, rise = trunk.swing(s, w, arm.l1, arm.l2, arm.side, m, hand)
            except ValueError:
                continue
            free = (deep2 <= 0.) & (rise <= rise_caps[i])
            if free.any():
                required[i] = float(share)
                bound = _choose(_runs(free, ang), raw_target[i], prev)
                raw_lo[i], raw_hi[i] = bound
                depths[i], rises[i] = deep2, rise
                prev = bound
                fine[i] = True
                break
        else:                           # no share opens an arc: the elbow takes the least deep swing there is
            best = float(ang[int(np.argmin(deep_i))])
            raw_lo[i] = raw_hi[i] = _near(best, anchor)
            depths[i], rises[i] = deep_i, rise
            prev = (raw_lo[i], raw_hi[i])
            fine[i] = True
    for b0, b1 in _blocks(todo):        # each block of the change, not the gap between two of them
        _relax(grid, lo, hi, target, fine, ws, depths, rises, rise_caps, clearance,
               raw=(raw_lo, raw_hi, raw_target), span=(b0, b1))
        for i in range(max(0, b0 - 1), min(n, b1 + 1)):
            if np.isfinite(target[i]) and np.isfinite(lo[i]) and np.isfinite(hi[i]):
                necessary[i] = max(0., lo[i] - target[i], target[i] - hi[i])
    return dict(lo=lo, hi=hi, target=target, required=required, deep=deep, necessary=necessary, tight=tight,
                depths=depths, rises=rises, rise_caps=rise_caps)


def _blocks(idx):
    """The contiguous runs of ``idx`` as (start, stop) pairs: the rescan's smoothing runs per run, not over the
    gap between two spans of the piece it did not touch."""
    out = []
    for i in idx:
        i = int(i)
        if out and i == out[-1][1]:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return out


def _relax(grid, lo, hi, target, fine, ws, depths, rises, rise_caps, clearance, raw=None, span=None):
    """Smooth the scan's constraint signals along the grid where the wrist sits still (the width shrinks to
    nothing with its speed; segments split where the signals restart). Each smoothed bound is verified against
    the sample's own depth and rise so the smoothing breaks the clearance no more than the slack.
    ``raw`` (the unsmoothed lo, hi, target) and ``span`` (the samples the last scan re-derived): a sample's window
    is +-W wide, so only the samples within W of the change can come out differently -- the rest keeps the answer
    it already had, which is why a rescan of one span costs a few dozen samples, not the piece."""
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
    r0, r1 = (0, n) if span is None else (max(0, span[0] - W - 1), min(n, span[1] + W + 2))
    for k, values in enumerate((lo, hi, target)):
        bounds = k < 2
        true = values if raw is None else raw[k]     # the unsmoothed constraint (the verify's reference)
        out = values if raw is not None else np.array(values, float)
        if raw is not None:                 # the samples the smoothing leaves alone keep their raw value
            out[r0:r1] = true[r0:r1]
        for i in range(r0, r1):
            if seg[i] < 0 or (bounds and not fine[i]) or not np.isfinite(true[i]):
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
            for i in range(r0, r1):          # verify each smoothed bound against the sample's own depth and rise
                if seg[i] < 0 or not fine[i] or out[i] == true[i]:
                    continue
                out[i] = _verified(out[i], true[i], depths[i], rises[i], rise_caps[i], clearance)
        if raw is None:
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
    from . import elbows_render_fast as RF
    tq = np.atleast_1d(np.asarray(times, float))
    # The curve's hermite is evaluated for every query frame. In the wrist ask this is called hundreds of
    # times over the render-rate grid; the compiled batch does the pchip slopes and all frames in one kernel
    # (identical formulas, ~0.8 ms instead of ~330 ms of Python loops). Blender has no numba: keep the
    # original per-frame path there, byte-for-byte.
    if RF.available():
        phi_all = RF.eval_many(curve['times'], curve['angles'][:, None], tq)
    else:
        from .motion import pchip_slopes, hermite_eval
        sl = pchip_slopes(curve['times'], curve['angles'][:, None])
        phi_all = np.array([float(hermite_eval(curve['times'], curve['angles'][:, None], sl, float(x))[0])
                            for x in tq])
    out = np.full(len(tq), np.nan)
    for k, t in enumerate(tq):
        try:
            v = np.asarray(path(float(t)), float)
            s = torso.shoulder_world(h, float(t))[:3, 3]
            phi = float(phi_all[k])
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


def _dwell(bound, width=DWELL_SAMPLES):
    """The scan's per-sample bounds median-filtered over ``width`` samples: a constraint that the geometry
    asserts for under ~0.2 s is a transient of the moving body (the forearm skims the outline for two
    samples and the arc snaps shut), not a place the elbow can stay -- left in, the solve dives to meet it
    and springs past the target when it lifts (a 33 deg S-swing around a 16 deg target, measured). The
    median keeps every constraint the geometry holds for keeps."""
    from scipy.ndimage import median_filter
    return median_filter(np.asarray(bound, float), size=width, mode='nearest')


def _build(data, paths, grid, cache=None, prev=None, dirty=None):
    """The swing curves from one scan's bounds: the least-jerk phi inside them (solve's). ``cache`` keeps each
    hand's path samples (they do not move with the body), and ``prev`` (the last curves) with ``dirty`` (the samples
    whose bounds the last scan moved) hands the solve its own answer to continue from -- chunked's restart."""
    from .motion import chunked
    n = len(grid)
    out = {}
    for h in 'LR':
        d = data[h]
        if n >= 8:
            got = None if cache is None else cache.get(('pts', h))
            if got is None or got[0] is not grid:
                pts = np.array([np.asarray(paths[h](t)[:3], float) for t in grid])
                got = (grid, pts, np.r_[np.linalg.norm(np.diff(pts, axis=0), axis=1)[0],
                                        np.linalg.norm(np.diff(pts, axis=0), axis=1)] / CURVE_STEP)
                if cache is not None:
                    cache[('pts', h)] = got
            speed = got[2]
            A, b = _rows(n, CURVE_STEP, JERK, ACC, VEL, PULL / (1. + speed / TRACK_RATE), d['target'])
            lo, hi = _dwell(d['lo']), _dwell(d['hi'])
            phi = chunked(A, b, lo, np.maximum(hi, lo + 1e-12),
                          x0=None if prev is None else prev[h]['angles'], dirty=dirty)
        else:
            phi = np.where(np.isfinite(d['target']), d['target'], 0.)
        out[h] = dict(times=grid, angles=phi)
    return out


def _wrist_ask(torso, data, curves, grid, paths, rig, trunk, clearance, solvers, rescan, kin=None, traced=None):
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
    t_part = time.perf_counter()
    base_ang_g = {h: _rendered(torso, curves[h], grid, paths[h], h, rig.arms[h], solvers[h]) for h in 'LR'}
    base_ang = {h: _rendered(torso, curves[h], ft, paths[h], h, rig.arms[h], solvers[h]) for h in 'LR'}
    base_sw = {h: _swing_of(data[h], curves[h]) for h in 'LR'}
    print(f'[elbows]   baseline {time.perf_counter() - t_part:.1f}s ({len(ft)} frames x 2 hands)', flush=True)
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
    t_pick = time.perf_counter()
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
    _tick('picks', time.perf_counter() - t_pick)
    print(f'[elbows]   picks {time.perf_counter() - t_pick:.1f}s, {sum(p is not None for p in picks)} of {n} samples '
          f'want a body move', flush=True)
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
        ky = _ease(rows, CURVE_STEP)
        ks_ = _ease(srows, CURVE_STEP, YIELD_SLIDE_TAU)
        torso.set_yield(grid, ky)
        torso.set_slide_yield(grid, ks_)
        moved = None
        if traced is not None:                  # the body moved on this span and (back) on the last one's: re-trace those
            moved = (ky != traced.get('y', 0.)) | (ks_ != traced.get('s', 0.))
            traced['y'], traced['s'] = ky, ks_
        try:
            t_ = time.perf_counter()
            count[0] += 1
            if count[0] % 20 == 0:
                print(f'[elbows]   rescan {count[0]} {time.perf_counter() - t_runs:.1f}s', flush=True)
            cdata, ccurves, _ = rescan(moved)
            _tick('honest_rescan', time.perf_counter() - t_)
            if PROFILE and count[0] <= 5:
                print(f'[elbows]   rescan {count[0]} {time.perf_counter() - t_:.2f}s '
                      f'moved {0 if moved is None else int(moved.sum())}', flush=True)
            kf = (ft >= float(grid[run[0]]) - .5 / RENDER_RATE) & (ft <= float(grid[run[-1]]) + .5 / RENDER_RATE)
            cost = 0.
            ks = np.asarray(run)
            inside = np.zeros(len(grid), bool)
            inside[run] = True      # the samples the ask moves the body on; the rest is the baseline's, sample for sample
            for h in 'LR':
                t_ = time.perf_counter()
                ang = base_ang[h].copy()            # outside the span the curve did not move: the rendered angles hold
                moved_t = np.abs(ccurves[h]['angles'] - curves[h]['angles'])
                moved_t[inside] = 0.
                moved_t[~np.isfinite(moved_t)] = np.inf
                idx = np.flatnonzero(moved_t > 1e-5)     # 1e-5 rad is a thousandth of a degree: no guard turns on it
                if len(idx):                        # the carry-over did move the curve out there: render those frames
                    extra = ((ft >= float(grid[idx[0]]) - .5 / RENDER_RATE)
                             & (ft <= float(grid[idx[-1]]) + .5 / RENDER_RATE) & ~kf)
                    if extra.any():
                        ang[extra] = _rendered(torso, ccurves[h], ft[extra], paths[h], h, rig.arms[h], solvers[h])
                ang[kf] = _rendered(torso, ccurves[h], ft[kf], paths[h], h, rig.arms[h], solvers[h])
                _tick('rendered', time.perf_counter() - t_)
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
    count = [0]
    trig = [k for k in range(n) if picks[k] is not None]
    runs = []
    for k in trig:                              # one span: a gap of a few samples is still one body move
        if runs and k - runs[-1][-1] <= 4:
            runs[-1].append(k)
        else:
            runs.append([k])
    ladder = [(0., .05), (0., .08), (0., .11), (-.016, .05), (-.016, .08), (-.032, .05), (-.032, .08)]
    t_runs = time.perf_counter()
    for done, run in enumerate(runs, 1):
        if len(runs) >= 20 and done % max(1, len(runs) // 10) == 0:
            print(f'[elbows]   ask {done}/{len(runs)} runs {time.perf_counter() - t_runs:.1f}s', flush=True)
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

    kin = {}        # the traced kinematics per hand, valid while the cap and the body's own move hold
    traced = {}     # the ask arrays the cache was traced on (what a rescan's ``moved`` is measured against)

    def scan_build(moved=None):
        torso.set_cap(grid, caps)       # the scans read the shared cap from the torso
        required = np.ones(n)
        d = {}
        t_ = time.perf_counter()
        for i, h in enumerate('LR'):    # measured -- threads lost to GIL contention (scan 77.6s -> 100.7s)
            d[h] = _scan(torso, paths[h], grid, rig.arms[h], h, i, trunk, clearance, kin=kin, moved=moved)
            required = np.minimum(required, d[h]['required'])
        _tick('scan', time.perf_counter() - t_)
        t_ = time.perf_counter()
        span = None if moved is None or not moved.any() else (int(np.argmax(moved)), int(len(moved) - 1 - np.argmax(moved[::-1])))
        curve = _build(d, paths, grid, cache=kin, prev=curves if span is not None else None, dirty=span)
        _tick('build', time.perf_counter() - t_)
        _tick('scan_build')
        return d, curve, required

    # 1. the ask-free solution settles the body-yield (the honest baseline the ask is searched against)
    t_phase = time.perf_counter()

    def phase(name):
        nonlocal t_phase
        now = time.perf_counter()
        print(f'[elbows] {name} {now - t_phase:.1f}s', flush=True)
        t_phase = now

    for round_ in range(CAP_ROUNDS + 2):
        data, curves, required = scan_build()
        if required.min() >= 1. or round_ == CAP_ROUNDS + 1:
            break
        caps = _cap_curve(required, n)  # the cap smooths over spans; the next round re-checks both hands on it
        kin.clear(), traced.clear()     # a new cap leans the whole body: every sample is re-traced
    phase(f'cap ({round_ + 1} scans, min {float(required.min()):.2f})')
    if solvers is not None:
        t_ask = time.perf_counter()
        ys, ss = _wrist_ask(torso, data, curves, grid, paths, rig, trunk, clearance, solvers, scan_build,
                            kin=kin, traced=traced)
        _tick('wrist_ask', time.perf_counter() - t_ask)
        phase('wrist ask')
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
                kin.clear(), traced.clear()
            phase(f'recheck ({round_ + 1} scans, min {float(required.min()):.2f})')
    damped = {h: _damped(curves[h]) for h in 'LR'}   # what the frames will read; the report below follows it
    phase('damped')
    if os.environ.get('PIANOMOTION_DEBUG_ELBOWS'):       # diagnosis only: the swing curve against its bounds
        import json as _json
        if os.environ.get('PIANOMOTION_DEBUG_ELBOWS_ALL'):   # the whole grid (locating a report's excess peaks)
            sel = np.ones(n, bool)
        else:
            raw = [float(x) for x in (os.environ.get('PIANOMOTION_DEBUG_AT') or '').split(',') if x.strip()]
            sel = np.zeros(n, bool)
            for win in (raw or [0.]):
                sel |= (grid > win - 1.2) & (grid < win + 1.2)
        rows = {}
        for h in 'LR':
            d = data[h]
            ang = _sample(damped[h], grid[sel])
            rows[h] = [[float(t), math.degrees(lo), math.degrees(hi), math.degrees(tg), math.degrees(nc), float(rq), float(a)]
                       for t, lo, hi, tg, nc, rq, a in zip(grid[sel], d['lo'][sel], d['hi'][sel], d['target'][sel],
                                                           d['necessary'][sel], d['required'][sel], ang)]
        with open(os.environ['PIANOMOTION_DEBUG_ELBOWS'], 'a') as f:
            _json.dump(rows, f)
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
    line = _profile_line()
    if line:
        print(line, flush=True)
    return damped, report


def load(rig_path):
    """The trunk outline beside a rig.json (rig_export writes it there), or None without one -- or with one traced
    from another rig: an outline from a different seat, character or export clears a body that is not there, and
    clearing the wrong body is worse than not clearing at all (no outline, no swing, as before the export existed).
    Pairs are checked on the rig's sha256 stamp; outlines that predate it on the trunk bone's world matrix, which
    both files already carry."""
    rig_path = Path(rig_path)
    # <rig stem>.trunk.json first (one outline per rig, so seats sharing a directory cannot shadow each other),
    # then the bare trunk.json that exports before this naming and rig.json itself still use.
    for path in (rig_path.with_name(rig_path.stem + '.trunk.json'), rig_path.with_name('trunk.json')):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        stale = _stale_reason(data, rig_path)
        if stale:
            print(f'TRUNK_STALE {path}: {stale}', flush=True)
            continue
        return Trunk.from_dict(data)
    return None


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
