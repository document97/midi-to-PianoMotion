"""The hand's natural orientation on the arm: it continues the forearm part of the way, and the forearm
is not fully pronated.

A reference pianist (FurElise dataset, performance 060) keeps the palm nearly square to the keys,
turned slightly inward, with the little-finger side ~7 deg lower. With the elbows hanging a little wide
(arm.ELBOW_OUT) the forearm points inward in front of the body; the hand follows HAND_FOLLOW of that
angle and the wrist (ulnar deviation) takes the rest.
"""
import math
import os

import numpy as np

from .arm import closest_elbow, standard_shoulder as _standard_shoulder
from .kinematics import FLOAT

HAND_FOLLOW = .3                # share of the forearm's inward angle the palm follows
# The forearm the hand's orientation follows is taken from a standard seat (arm.STANDARD_SHOULDER), not the scene's:
# the hand's plan (fingering, poses, wrist path) is the same wherever the performer sits within reach, the arm
# reaching to it from the actual seat.
LITTLE_SIDE_DOWN = 6.           # deg: the little-finger side of the palm lower than the thumb side
PALM_SOFT_DEG = 160.            # sustained comfort line (pianists keep ~10 deg of ulnar deviation; past 20 the
# carpal tunnel loads): where the relaxed hand would sit below it, the palm follows the forearm further -- the
# wrist unbends, the arm line, not the hand's bend, carries the reach
PALM_FLOOR_DEG = 135.           # the user's transient acceptance floor: below it the body yields (torso.py), the
# wrist never bends more; what is left after the palm's full follow is the body's share to cover
# The corridor is an A/B switch too: PIANOMOTION_PALM_CORRIDOR=0 gives the hand the 11.11 behaviour (HAND_FOLLOW
# only), for comparing a run against an older one without editing code
PALM_CORRIDOR = os.environ.get('PIANOMOTION_PALM_CORRIDOR', '1').lower() not in ('0', 'off', 'false', 'no')


def palm_forearm_deg(solver, wrist, yaw, roll, shoulder, elbow=None):
    """The palm's long axis vs the forearm's, interior angle in degrees (180 straight): exactly what arm.pose
    stores per frame ({h}_palm_deg), predicted here from the plan (the same palm_q and elbow geometry).
    ``elbow``: a placed elbow (the swing curve's) instead of the natural hang -- the prediction at the forearm
    the rendered elbow actually takes."""
    from .kinematics import q_mat3
    arm = solver.arm
    palm = q_mat3(arm.palm_q(yaw, roll)) @ np.array([0., 1., 0.], dtype=FLOAT)
    if elbow is None:
        try:
            elbow = closest_elbow(shoulder, np.asarray(wrist, FLOAT), arm.l1, arm.l2, arm.side)
        except ValueError:              # beyond the unstretched arm: the forearm is the wrist's own direction
            return 180.
    forearm = np.asarray(wrist, FLOAT) - elbow
    return 180. - math.degrees(math.acos(np.clip(float(palm @ forearm) / max(np.linalg.norm(forearm), 1e-9), -1., 1.)))


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
    return _standard_shoulder(solver.arm, solver.rig.neutral_wrist[solver.hand])


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
    if PALM_CORRIDOR and abs(dy) > 1e-9:    # the comfort corridor: where the relaxed hand would sit below the
        now = palm_forearm_deg(solver, wrist, yaw, roll, shoulder)   # sustained line, the palm follows the
        if now < PALM_SOFT_DEG:         # forearm further -- the wrist unbends (the arm line, not the hand's
            full = (forearm_yaw - y0) / dy   # bend, carries the reach); never past the forearm itself
            far = palm_forearm_deg(solver, wrist, full, roll, shoulder)
            if far > now:
                share = 1. if far <= PALM_SOFT_DEG else min(1., HAND_FOLLOW + (1. - HAND_FOLLOW) *
                                                            (PALM_SOFT_DEG - now) / (far - now))
                yaw = (share * forearm_yaw - y0) / dy
    return np.array([yaw, roll], dtype=FLOAT)
