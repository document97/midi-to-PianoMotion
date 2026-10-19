"""Torso lean toward a hand playing far out (numpy version of lib/body_driver.BodyDriver.pose), on a constant forward lean."""
import math

import numpy as np

from .kinematics import axis_matrix, mat3_q

FORWARD_LEAN_DEGREES = 10.      # sitting back (thighs clear of the keybed), the player leans in from the hips


class Torso:
    """Slow torso lean toward a hand that plays far out."""

    def __init__(self, data, arms):
        self.data = data
        self.bone = data['bone']
        self.pivot = np.array(data['pivot'])
        self.pre_inverse = np.linalg.inv(np.array(data['pre']))
        self.world = np.array(data['world'])
        self.reference = {h: np.array(m) for h, m in data['reference'].items()}
        self.height = np.mean([m[2, 3] for m in self.reference.values()]) - self.pivot[2]
        self.rows = np.zeros((0, 3))
        self.first = self.last = 0.

    def set_notes(self, rows, first, last):
        """``rows``: (start, wrist x - shoulder x beyond .15 m, (velocity/127)^.6) per note."""
        self.rows, self.first, self.last = np.asarray(rows, float).reshape(-1, 3), first, last

    @staticmethod
    def _smooth(x):
        x = np.clip(x, 0, 1)
        return x * x * x * (10 + x * (-15 + 6 * x))

    def pose(self, t):
        """(torso bone local quaternion, {hand: shoulder pre-matrix})."""
        rows = self.rows[np.abs(self.rows[:, 0] - t) < .8]
        shift = 0.
        if len(rows):
            weights = np.exp(-.5 * ((rows[:, 0] - t) / .35) ** 2) * rows[:, 2] * self._smooth((.8 - np.abs(rows[:, 0] - t)) / .2)
            shift = float(np.sum(rows[:, 1] * weights) / (.25 + np.sum(weights)))
        shift = float(np.clip(shift, -.11, .11) * self._smooth((t - self.first + .85) / .65) * (1 - self._smooth((t - self.last) / .7)))
        limit = math.radians(16)
        raw = math.asin(float(np.clip(shift / max(.05, self.height), -1, 1)))
        angle = limit * math.tanh(raw / limit)
        transform = np.eye(4)
        transform[:3, :3] = axis_matrix((0., 1., 0.), angle) @ axis_matrix((1., 0., 0.), -math.radians(FORWARD_LEAN_DEGREES))
        transform[:3, 3] = self.pivot - transform[:3, :3] @ self.pivot
        pre = {h: transform @ m for h, m in self.reference.items()}
        return mat3_q(self.pre_inverse @ transform @ self.world), pre
