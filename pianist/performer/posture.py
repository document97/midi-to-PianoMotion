"""The hand's natural orientation on the arm: it continues the forearm part of the way, and the forearm
is not fully pronated.

A reference pianist (FurElise dataset, performance 060) keeps the palm nearly square to the keys,
turned slightly inward, with the little-finger side ~7 deg lower. With the elbows hanging a little wide
(arm.ELBOW_OUT) the forearm points inward in front of the body; the hand follows HAND_FOLLOW of that
angle and the wrist (ulnar deviation) takes the rest.
"""
import math

import numpy as np

from .arm import closest_elbow

HAND_FOLLOW = .3                # share of the forearm's inward angle the palm follows
LITTLE_SIDE_DOWN = 6.           # deg: the little-finger side of the palm lower than the thumb side


def _palm_angles(solver, yaw, roll):
    """(palm yaw: + fingers toward the high keys, little-finger side tilt: + higher), deg."""
    from .motion import Spacing
    if not hasattr(solver, '_spacing_lateral'):
        solver._spacing_lateral = Spacing(solver).lateral
    palm = solver.palm(np.zeros(3), yaw, roll)[:3, :3]
    forward, across = palm @ np.array([0., 1., 0.]), palm @ solver._spacing_lateral
    return (math.degrees(math.atan2(forward[0], forward[1])),
            math.degrees(math.asin(across[2] / np.linalg.norm(across))))


def _calibration(solver):
    """Linear map of the (yaw, roll) parameters onto palm yaw and tilt (exact for yaw; roll near linear)."""
    if not hasattr(solver, '_orientation_map'):
        y0, t0 = _palm_angles(solver, 0., 0.)
        y1, _ = _palm_angles(solver, 1., 0.)
        _, t1 = _palm_angles(solver, 0., 1.)
        solver._orientation_map = (y0, y1 - y0, t0, t1 - t0)
    return solver._orientation_map


def natural_orientation(solver, wrist):
    """(yaw, roll) parameters of the relaxed hand with the wrist at ``wrist`` (world)."""
    y0, dy, t0, dt = _calibration(solver)
    arm = solver.arm
    shoulder = arm.pre_upper_world[:3, 3]
    try:
        elbow = closest_elbow(shoulder, np.asarray(wrist, float), arm.l1, arm.l2, arm.side)
        forearm = np.asarray(wrist, float) - elbow
        forearm_yaw = math.degrees(math.atan2(forearm[0], forearm[1]))
    except ValueError:
        forearm_yaw = 0.
    yaw = (HAND_FOLLOW * forearm_yaw - y0) / dy
    roll = (-LITTLE_SIDE_DOWN - t0) / dt
    return np.array([yaw, roll])
