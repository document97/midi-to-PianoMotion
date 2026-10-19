"""Whole-hand pose solve: wrist (x, y, z, yaw, roll) and the four values of every contact finger.

Contacts are hard targets: the lowest point of the finger pad (a soft minimum over
the calibrated pad patch) lies on the key surface at the requested depth, inside
the key's playable region (the wide front part, or the narrow part between black
keys). Everything else is a soft preference toward the calibrated relaxed hand.
Bounded Levenberg-Marquardt in numpy; the wrist block of the Jacobian is exact
because pad points are rigid in the wrist frame for fixed finger values.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from .checks import Heightfield, stretched
from .kinematics import DIGITS, SkinSet, q_mat3

PAD_SOFTMIN_METRES = .0005      # temperature of the pad's lowest-point soft minimum
X_MARGIN = .001                  # pad centre kept this far inside a key's side edges (neighbour contact is checked separately)
Y_FRONT_MARGIN = .006            # ... and behind the key's front edge
NARROW_TRANSITION = .002
WHITE_BACK_MARGIN = .004          # white-key pads stay this far in front of the black keys         # width of the front-to-narrow blend behind the black-key front
BLACK_DEPTH_REACH = .07          # how far into a key (from its front) a pad may land
CONTACT_TOLERANCE = .0002        # accepted pad-to-surface error (m)
CLEARANCE_SCALE = .0005          # skin penetration into other keys, metres per residual unit

# Relaxed hand (contact_solver.HandModel.idle): thumb and the long fingers.
RELAXED = {1: np.array([0., 8., 8., 16.]), 2: np.array([0., 9., 34., 18.])}
for _d in (3, 4, 5):
    RELAXED[_d] = RELAXED[2]
VALUE_SCALE = np.array([18., 18., 30., 25.])   # degrees per unit of preference residual
WRIST_SCALE = np.array([.04, .03, .025, 12., 10.])   # x, y, z (m), yaw, roll (deg)
WRIST_TRAVEL = np.array([.08, .07, .05])
FOLLOW_WRIST_SCALE = np.array([.004, .004, .004, 2., 2.])   # frame correction: m and deg per unit
FOLLOW_VALUE_SCALE = 40.   # metres the wrist may leave its hint / neutral y, z
YAW_BOUNDS = (-25., 25.)   # wrist side bend; real hands ~-25 (thumb side) .. +35. +-15 stopped black-key octaves
SEEDS = ((0., 0., 0.), (0., 0., -.03), (-12., -.02, 0.), (12., .02, 0.))   # (yaw deg, wrist dx, dy m) starts
SEED_ITERATIONS = 12
WIDE_SEEDS = tuple((y, dx, dy) for y in (-12., 0., 12.) for dx in (-.03, 0., .03) for dy in (0., -.03))   # offline tables
ROLL_BOUNDS = (-20., 20.)


@dataclass(frozen=True)
class Contact:
    digit: int
    pitch: int
    depth: float = 0.           # key travel below its top (0 .. anchors.KEY_TRAVEL m)


@dataclass
class HandPose:
    wrist: np.ndarray            # world position (m)
    yaw: float
    roll: float
    values: dict                 # digit -> 4 values (deg)
    contact_error: float = 0.    # worst pad-to-target error (m)
    ok: bool = True
    iterations: int = 0
    targets: dict = field(default_factory=dict)   # digit -> world pad point
    clearance: float = 0.        # worst skin penetration into another key (m)

    def vector(self):
        return np.r_[self.wrist, self.yaw, self.roll]


class KeyRegions:
    """Playable pad region of each key: wide front part, narrow part between black keys."""

    def __init__(self, keyboard):
        self.keys = {int(p): k for p, k in keyboard.items()}
        self.black_front = min(k['bounds_min'][1] for k in self.keys.values() if self._black(k))
        self.rows = {p: self._region(p) for p in self.keys}

    @staticmethod
    def _black(k):
        return k['bounds_min'][1] > -.025

    def is_black(self, pitch):
        return self._black(self.keys[pitch])

    def _region(self, p):
        k = self.keys[p]
        lo, hi = k['bounds_min'], k['bounds_max']
        if self._black(k):
            wide = narrow = (lo[0] + X_MARGIN, hi[0] - X_MARGIN)
            return dict(top=hi[2], wide=wide, narrow=narrow, split=lo[1], y=(lo[1] + Y_FRONT_MARGIN, lo[1] + BLACK_DEPTH_REACH))
        # The rig's fingertips are wider than a white key's part between two black keys, so white
        # keys are played on their wide front part only (in front of the black keys).
        wide = (lo[0] + X_MARGIN, hi[0] - X_MARGIN)
        return dict(top=hi[2], wide=wide, narrow=wide, split=self.black_front,
                    y=(lo[1] + Y_FRONT_MARGIN, self.black_front - WHITE_BACK_MARGIN))

    def residual(self, pitch, depth, point):
        """(x, y, z) metres outside the region / off the surface."""
        r = self.rows[pitch]
        x, y, z = point
        s = 1. / (1. + math.exp(-max(-50., min(50., (y - r['split']) / NARROW_TRANSITION))))
        lo = (1 - s) * r['wide'][0] + s * r['narrow'][0]
        hi = (1 - s) * r['wide'][1] + s * r['narrow'][1]
        rx = max(0., lo - x) - max(0., x - hi)
        ry = max(0., r['y'][0] - y) - max(0., y - r['y'][1])
        return np.array([rx, ry, z - (r['top'] - depth)])

    def centre(self, pitch):
        r = self.rows[pitch]
        return np.array([(r['wide'][0] + r['wide'][1]) / 2, (r['y'][0] + r['split']) / 2, r['top']])


class PoseSolver:
    """One hand. ``rig``: kinematics.Rig."""

    def __init__(self, rig, hand, regions=None, heightfield=None):
        self.rig, self.hand = rig, hand
        self.heightfield = heightfield     # checks.Heightfield: contact fingers keep clear of other keys
        self.arm = rig.arms[hand]
        self.regions = regions or KeyRegions(rig.keyboard)
        self.identity = np.eye(4)
        self.neutral = rig.neutral_wrist[hand]
        self._pad_cache = {}
        owner = rig.bone_owner[hand]
        self.pad_digits = {d: sorted({owner[b] for b in rig.fingers[hand][d].pad.bones if b in owner}) for d in DIGITS}
        self.clearance_skin = {}
        for d in DIGITS:
            skin = rig.fingers[hand][d].clearance
            own = [k for k, b in enumerate(skin.bones) if owner.get(b) == d]
            mask = skin.weights[:, own].sum(axis=1) >= .5      # the finger itself, not palm skin
            self.clearance_skin[d] = SkinSet(skin.bones, skin.weights[mask], skin.local[:, mask])
        self.clearance_digits = {d: sorted({owner[b] for b in self.clearance_skin[d].bones if b in owner})
                                 for d in DIGITS}

    # ----- geometry -----
    def palm(self, wrist, yaw, roll):
        m = np.eye(4)
        m[:3, :3] = q_mat3(self.arm.palm_q(yaw, roll)) * self.arm.scale['wrist']
        m[:3, 3] = wrist
        return m

    def local_pad(self, digit, values):
        """Pad points (homogeneous) in the wrist frame; world = palm @ local."""
        return self._local('pad', digit, values)

    def local_clearance(self, digit, values):
        return self._local('clearance', digit, values)

    def _local(self, kind, digit, values):
        depends = self.pad_digits if kind == 'pad' else self.clearance_digits
        key = (kind, digit) + tuple(v for d in depends[digit] for v in np.round(values[d], 6))
        hit = self._pad_cache.get(key)
        if hit is not None:
            return hit
        finger = self.rig.fingers[self.hand][digit]
        skin = finger.pad if kind == 'pad' else self.clearance_skin[digit]
        mats = self.rig.bone_matrices(self.hand, self.identity, values, skin.bones)
        acc = np.zeros((skin.weights.shape[0], 4))
        for k, name in enumerate(skin.bones):
            acc += skin.weights[:, k:k + 1] * (skin.local[k] @ mats[name].T)
        local = acc.T     # wrist-bone frame: world = arm_world @ W @ p = palm @ p, since palm = arm_world @ W
        if len(self._pad_cache) > 50000:
            self._pad_cache.clear()
        self._pad_cache[key] = local
        return local

    def clearance_residual(self, digit, values, palm, depths, own_pitch, brush=False):
        """Penetration of the finger's skin into keys other than its own (scaled), fixed length."""
        if self.heightfield is None:
            return np.zeros(0)
        points = (palm @ self.local_clearance(digit, values))[:3].T
        own = None if own_pitch is None else (own_pitch, np.ones(len(points), bool))
        return self.heightfield.penetration(points, depths, own, brush=brush) / CLEARANCE_SCALE

    @staticmethod
    def contact_point(points):
        """Soft lowest point of the pad: z by soft-min, x/y as its weighted centroid."""
        z = points[:, 2]
        w = np.exp(-(z - z.min()) / PAD_SOFTMIN_METRES)
        w /= w.sum()
        return w @ points

    def pad_world(self, digit, values, palm):
        return (palm @ self.local_pad(digit, values))[:3].T

    # ----- solve -----
    def solve(self, contacts, values=None, wrist=None, yaw=0., roll=0., x_hint=None, iterations=40, seeds=SEEDS,
              clearance=True):
        """Pose with every contact on its key; ``values`` seeds / fixes free fingers.

        Feasibility (``ok``) is the contacts alone. Key clearance of the contact fingers is minimised
        (``clearance=True``) and reported, not a gate: the key tops are piecewise flat and as a hard
        screen they make the search call reachable chords impossible."""
        values = {d: np.array(v, float) for d, v in (values or {d: RELAXED[d] for d in DIGITS}).items()}
        for d in DIGITS:
            values.setdefault(d, RELAXED[d].copy())
        contacts = sorted(contacts, key=lambda c: c.digit)
        digits = [c.digit for c in contacts]
        if wrist is None:
            wrist = self.neutral.copy()
            if contacts:
                wrist[0] = np.mean([self.regions.centre(c.pitch)[0] for c in contacts]) \
                    - np.mean([self._reach_x(c.digit) for c in contacts])
        if x_hint is None:
            x_hint = wrist[0]
        fingers = self.rig.fingers[self.hand]
        centre = np.array([x_hint, self.neutral[1], self.neutral[2]])
        lower = np.r_[centre - WRIST_TRAVEL, YAW_BOUNDS[0], ROLL_BOUNDS[0],
                      np.concatenate([fingers[d].lower for d in digits]) if digits else []]
        upper = np.r_[centre + WRIST_TRAVEL, YAW_BOUNDS[1], ROLL_BOUNDS[1],
                      np.concatenate([fingers[d].upper for d in digits]) if digits else []]
        x = np.r_[wrist, yaw, roll, np.concatenate([values[d] for d in digits]) if digits else []]
        x = np.clip(x, lower, upper)
        prior_wrist = np.r_[x_hint, self.neutral[1], self.neutral[2], 0., 0.]

        depths = {c.pitch: c.depth for c in contacts}

        def unpack(v):
            vals = dict(values)
            for i, d in enumerate(digits):
                vals[d] = v[5 + 4 * i: 9 + 4 * i]
            return v[:3], v[3], v[4], vals

        stretch = stretched(c.pitch for c in contacts)

        def residuals(v, prior, clear):
            w, ya, ro, vals = unpack(v)
            palm = self.palm(w, ya, ro)
            out = []
            for c in contacts:
                point = self.contact_point(self.pad_world(c.digit, vals, palm))
                out.append(self.regions.residual(c.pitch, c.depth, point) / CONTACT_TOLERANCE)
                if clear:
                    out.append(self.clearance_residual(c.digit, vals, palm, depths, c.pitch, stretch))
            out.append(prior * (v[:5] - prior_wrist) / WRIST_SCALE)
            for d in digits:
                out.append(prior * (vals[d] - RELAXED[d]) / VALUE_SCALE)
            return np.concatenate(out)

        steps = np.r_[1e-5, 1e-5, 1e-5, 1e-2, 1e-2, np.full(4 * len(digits), 1e-2)]

        def touch(v):
            return residuals(v, .03, False)

        def reach(v):
            return residuals(v, .03, clearance)

        # The key surfaces are piecewise flat, so their clearance has poor gradients and stretched
        # chords have local minima: each wrist seed first reaches the keys, then adds key clearance;
        # the best seed is finished, and a last phase relaxes the hand while keeping both.
        it, best = 0, None
        for yaw_seed, dx, dy in seeds:
            start = x.copy()
            start[0] += dx
            start[1] += dy
            start[3] = yaw_seed
            start = np.clip(start, lower, upper)
            trial, used = self._levenberg_marquardt(touch, start, lower, upper, steps, SEED_ITERATIONS)
            trial, more = self._levenberg_marquardt(reach, trial, lower, upper, steps, SEED_ITERATIONS) if clearance else (trial, 0)
            it += used + more
            r = reach(trial)
            if best is None or r @ r < best[0]:
                best = (r @ r, trial)
            if r @ r < 1e-4:
                break
        x, used = self._levenberg_marquardt(reach, best[1], lower, upper, steps, iterations)
        it += used
        x, used = self._levenberg_marquardt(lambda v: residuals(v, 1., clearance), x, lower, upper, steps, iterations // 2)
        it += used
        w, ya, ro, vals = unpack(x)
        palm = self.palm(w, ya, ro)
        error, targets = 0., {}
        worst_clearance = 0.
        for c in contacts:
            point = self.contact_point(self.pad_world(c.digit, vals, palm))
            targets[c.digit] = point
            error = max(error, float(np.max(np.abs(self.regions.residual(c.pitch, c.depth, point)))))
            pen = self.clearance_residual(c.digit, vals, palm, depths, c.pitch)
            worst_clearance = max(worst_clearance, float(pen.max()) * CLEARANCE_SCALE if len(pen) else 0.)
        return HandPose(np.array(w), float(ya), float(ro), {d: np.array(v) for d, v in vals.items()},
                        error, error <= CONTACT_TOLERANCE * 2.5, it, targets, worst_clearance)

    def follow(self, points, vector, values, iterations=20, depths=None, pitches=None):
        """Hold every pad on its fixed world point, moving the wrist as little as possible from ``vector``.

        Used frame by frame when finger IK alone cannot keep a contact (touch lift, wrist travel
        while keys are held). ``points``: {digit: world point}; returns (wrist vector, values, error).
        """
        digits = sorted(points)
        fingers = self.rig.fingers[self.hand]
        base = np.asarray(vector, float)
        seed = {d: np.asarray(values[d], float) for d in digits}
        lower = np.r_[base[:3] - WRIST_TRAVEL, YAW_BOUNDS[0], ROLL_BOUNDS[0], np.concatenate([fingers[d].lower for d in digits])]
        upper = np.r_[base[:3] + WRIST_TRAVEL, YAW_BOUNDS[1], ROLL_BOUNDS[1], np.concatenate([fingers[d].upper for d in digits])]
        x = np.clip(np.r_[base, np.concatenate([seed[d] for d in digits])], lower, upper)
        vals = dict(values)

        def residuals(v):
            palm = self.palm(v[:3], v[3], v[4])
            out = [(v[:5] - base) / FOLLOW_WRIST_SCALE]
            for i, d in enumerate(digits):
                vals[d] = v[5 + 4 * i: 9 + 4 * i]
                out.append((vals[d] - seed[d]) / FOLLOW_VALUE_SCALE)
            for d in digits:
                out.append((self.contact_point(self.pad_world(d, vals, palm)) - points[d]) / CONTACT_TOLERANCE)
                if depths is not None:
                    out.append(self.clearance_residual(d, vals, palm, depths, pitches[d]))
            return np.concatenate(out)

        steps = np.r_[1e-5, 1e-5, 1e-5, 1e-2, 1e-2, np.full(4 * len(digits), 1e-2)]
        x, _ = self._levenberg_marquardt(residuals, x, lower, upper, steps, iterations)
        residuals(x)
        palm = self.palm(x[:3], x[3], x[4])
        error = max(float(np.linalg.norm(self.contact_point(self.pad_world(d, vals, palm)) - points[d])) for d in digits)
        return x[:5], {d: np.array(v) for d, v in vals.items()}, error

    @staticmethod
    def _levenberg_marquardt(residuals, x, lower, upper, steps, iterations):
        lam, r = 1e-2, residuals(x)
        cost = r @ r
        for it in range(1, iterations + 1):
            jac = np.empty((len(r), len(x)))
            for j in range(len(x)):
                trial = x.copy()
                trial[j] += steps[j]
                jac[:, j] = (residuals(trial) - r) / steps[j]
            g, h = jac.T @ r, jac.T @ jac
            improved = False
            for _ in range(10):
                delta = np.linalg.solve(h + lam * np.diag(np.diag(h) + 1e-9), -g)
                trial = np.clip(x + delta, lower, upper)
                tr = residuals(trial)
                if tr @ tr < cost:
                    x, r, cost, lam, improved = trial, tr, tr @ tr, max(lam / 3, 1e-7), True
                    break
                lam *= 4
            if not improved or np.max(np.abs(trial - x if not improved else delta)) < 1e-7 or cost < 1e-6:
                return x, it
        return x, iterations

    def _reach_x(self, digit):
        """Pad x offset from the wrist in the relaxed neutral hand (for the initial wrist)."""
        key = ('reach', digit)
        if key not in self._pad_cache:
            palm = self.palm(self.neutral, 0., 0.)
            point = self.contact_point(self.pad_world(digit, {d: RELAXED[d] for d in DIGITS}, palm))
            self._pad_cache[key] = point[0] - self.neutral[0]
        return self._pad_cache[key]


def hand_solvers(rig):
    """Both hands' solvers sharing one key height field."""
    heightfield = Heightfield(rig.data['key_meshes'])
    return {h: PoseSolver(rig, h, heightfield=heightfield) for h in 'LR'}
