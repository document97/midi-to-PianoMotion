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
from .kinematics import FLOAT

HAND_FOLLOW = .3                # share of the forearm's inward angle the palm follows
# The forearm the hand's orientation follows is taken from a standard seat, not the scene's: the hand's plan (fingering,
# poses, wrist path) is the same wherever the performer sits within reach, the arm reaching to it from the actual seat.
# The standard shoulder relative to the hand's neutral wrist over the keys, in arm lengths (upper + forearm): x toward
# the hand's own side, y toward the player, z up (a seated pianist's, measured on the development scene).
STANDARD_SHOULDER = np.array([-.006, -.766, .427])
LITTLE_SIDE_DOWN = 6.           # deg: the little-finger side of the palm lower than the thumb side


def _palm_angles(solver, yaw, roll):
    """(palm yaw: + fingers toward the high keys, little-finger side tilt: + higher), deg."""
    from .motion import Spacing
    if not hasattr(solver, '_spacing_lateral'):
        solver._spacing_lateral = Spacing(solver).lateral
    palm = solver.palm(np.zeros(3, dtype=FLOAT), yaw, roll)[:3, :3]
    forward, across = palm @ np.array([0., 1., 0.], dtype=FLOAT), palm @ solver._spacing_lateral
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


def standard_shoulder(solver):
    """The shoulder of the standard seat (``STANDARD_SHOULDER``) for this hand."""
    arm = solver.arm
    offset = STANDARD_SHOULDER * (arm.l1 + arm.l2)
    offset[0] *= arm.side                  # toward the hand's own side: left hand -x, right hand +x
    return (np.asarray(solver.rig.neutral_wrist[solver.hand], FLOAT) + offset).astype(FLOAT)


def natural_orientation(solver, wrist, shoulder=None):
    """(yaw, roll) parameters of the relaxed hand with the wrist at ``wrist`` (world), its forearm from ``shoulder``:
    the standard seat's (``standard_shoulder``) unless given (the hand on the lap: the actual body's)."""
    y0, dy, t0, dt = _calibration(solver)
    arm = solver.arm
    shoulder = standard_shoulder(solver) if shoulder is None else np.asarray(shoulder, FLOAT)
    try:
        elbow = closest_elbow(shoulder, np.asarray(wrist, FLOAT), arm.l1, arm.l2, arm.side)
        forearm = np.asarray(wrist, FLOAT) - elbow
        forearm_yaw = math.degrees(math.atan2(forearm[0], forearm[1]))
    except ValueError:
        forearm_yaw = 0.
    yaw = (HAND_FOLLOW * forearm_yaw - y0) / dy
    roll = (-LITTLE_SIDE_DOWN - t0) / dt
    return np.array([yaw, roll], dtype=FLOAT)
