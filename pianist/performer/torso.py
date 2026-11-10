"""Torso lean toward a hand playing far out (numpy version of lib/body_driver.BodyDriver.pose), on a constant forward lean."""
import math

import numpy as np

from .kinematics import FLOAT, axis_matrix, mat3_q

FORWARD_LEAN_DEGREES = 10.      # sitting back (thighs clear of the keybed), the player leans in from the hips
# The trunk turns (about the vertical through the hips, which stay where they sit) toward where the hands play: the
# angle whose tangent is the notes' mean distance sideways from the body over TURN_DEPTH, times TURN_GAIN, eased into
# TURN_LIMIT; the mean is taken over TURN_SECONDS (gaussian width), slower than the lean, a neutral weight pulling it
# home where few notes play.
TURN_DEPTH = .35                # m: about the hands' distance in front of the hips
TURN_GAIN = .5
TURN_LIMIT = 12.                # deg
TURN_SECONDS = .6
TURN_NEUTRAL = .5


class Trunk:
    """The trunk's skin as convex horizontal slices at rest (rig_export.trunk_shape): how deep points are inside it."""

    def __init__(self, data):
        self.z0, self.step = data['z0'], data['slice']
        self.radii = {h: tuple(r) for h, r in data['radii'].items()}
        keys = sorted(int(k) for k in data['hulls'])
        self.first, n = keys[0], keys[-1] - keys[0] + 1
        edges = max(len(v) for v in data['hulls'].values())
        self.origin = np.zeros((n, edges, 2))
        self.normal = np.zeros((n, edges, 2))
        self.bias = np.full((n, edges), -1.)      # an empty slice: every point 1 m outside
        for k in keys:
            h = np.array(data['hulls'][str(k)], dtype=float)
            e = np.roll(h, -1, axis=0) - h
            nrm = np.c_[e[:, 1], -e[:, 0]] / np.maximum(np.linalg.norm(e, axis=1), 1e-12)[:, None]
            if np.mean(((h - h.mean(0)) * nrm).sum(1)) < 0:
                nrm = -nrm
            i, m = k - self.first, len(h)
            self.origin[i, :m], self.normal[i, :m], self.bias[i, :m] = h, nrm, 0.
            self.origin[i, m:], self.normal[i, m:], self.bias[i, m:] = h[0], nrm[0], 0.

    def depth(self, points, inverse=None):
        """Per point (rows, world m): how far inside the trunk it lies (negative outside); ``inverse``: world -> rest."""
        p = np.asarray(points, float)
        if inverse is not None:
            p = p @ inverse[:3, :3].T + inverse[:3, 3]
        k = np.floor((p[:, 2] - self.z0) / self.step).astype(int) - self.first
        ok = (k >= 0) & (k < len(self.origin))
        out = np.full(len(p), -1.)
        kk = k[ok]
        out[ok] = np.min(((self.origin[kk] - p[ok, None, :2]) * self.normal[kk]).sum(-1) + self.bias[kk], axis=1)
        return out


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
        self.rows = np.zeros((0, 4))
        self.trunk = Trunk(data['trunk']) if data.get('trunk') else None
        self.first = self.last = 0.

    def set_notes(self, rows, first, last):
        """``rows``: (start, wrist x - shoulder x beyond .15 m, (velocity/127)^.6, wrist x - hips x) per note."""
        self.rows, self.first, self.last = np.asarray(rows, float).reshape(-1, 4), first, last

    @staticmethod
    def _smooth(x):
        x = np.clip(x, 0, 1)
        return x * x * x * (10 + x * (-15 + 6 * x))

    def pose(self, t):
        """(torso bone local quaternion, {hand: shoulder pre-matrix})."""
        transform = self.transform(t)
        pre = {h: transform @ m for h, m in self.reference.items()}
        return mat3_q(self.pre_inverse @ transform @ self.world), pre

    def transform(self, t):
        """World transform of the trunk at ``t``: the forward lean, the lean toward a far hand, the turn."""
        rows = self.rows[np.abs(self.rows[:, 0] - t) < .8]
        shift = 0.
        if len(rows):
            weights = np.exp(-.5 * ((rows[:, 0] - t) / .35) ** 2) * rows[:, 2] * self._smooth((.8 - np.abs(rows[:, 0] - t)) / .2)
            shift = float(np.sum(rows[:, 1] * weights) / (.25 + np.sum(weights)))
        shift = float(np.clip(shift, -.11, .11) * self._smooth((t - self.first + .85) / .65) * (1 - self._smooth((t - self.last) / .7)))
        limit = math.radians(16)
        raw = math.asin(float(np.clip(shift / max(.05, self.height), -1, 1)))
        angle = limit * math.tanh(raw / limit)
        near = self.rows[np.abs(self.rows[:, 0] - t) < 3 * TURN_SECONDS]
        side = 0.
        if len(near):
            w = np.exp(-.5 * ((near[:, 0] - t) / TURN_SECONDS) ** 2) * near[:, 2]
            side = float(np.sum(near[:, 3] * w) / (TURN_NEUTRAL + np.sum(w)))
        fade = self._smooth((t - self.first + .85) / .65) * (1 - self._smooth((t - self.last) / .7))
        limit = math.radians(TURN_LIMIT)
        turn = limit * math.tanh(TURN_GAIN * math.atan2(side, TURN_DEPTH) / limit) * fade
        transform = np.eye(4, dtype=FLOAT)
        # toward +x (the treble) the front turns right: about z, negative
        transform[:3, :3] = (axis_matrix((0., 0., 1.), -turn) @ axis_matrix((0., 1., 0.), angle)
                             @ axis_matrix((1., 0., 0.), -math.radians(FORWARD_LEAN_DEGREES)))
        transform[:3, 3] = self.pivot - transform[:3, :3] @ self.pivot
        return transform
