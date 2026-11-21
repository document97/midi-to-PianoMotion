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
from .pose import ROLL_BOUNDS

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
# The palm's yaw above continues the forearm, and the forearm says nothing about the keyboard: measured on the
# daylight piece the fingers then sit p50 13 deg / p95 27 deg off the keys' length while the palm-vs-forearm angle
# is a comfortable 160 -- the hand square to its own arm and crooked on the keys, which is the angle the eye reads.
# So the planned yaw turns to the keys first, and the corridor is a floor the turn may not fold past, not the aim.
KEY_AIM = os.environ.get('PIANOMOTION_KEY_AIM', '1').lower() not in ('0', 'off', 'false', 'no')
KEY_TOL = 4.                    # deg the fingers may sit off the keys' length and still count as down the keys
KEY_FLOOR = 145.                # deg on the palm-vs-forearm angle: the turn never folds the wrist past it
KEY_STEP = 1.                   # deg per yaw step while walking out from the relaxed yaw
KEY_TRAVEL = 35.                # deg the wrist may deflect either way (radial/ulnar, the hand's own travel)
KEY_RADIAL_MAX = 12.            # deg the palm may lean toward the thumb side (radial deviation) while a key is
# played: a modest thumb-side lean reads fine -- the user finds ten-odd degrees barely noticeable -- but past it
# the hand reads wrenched, and where the keys would ask for that the player moves the forearm instead (the elbow,
# the seat). The render sat radial on 15% of the right hand's played frames (p50 8, p90 13, worst 33 deg) before
# the cap; the search stops here and falls back to the most upright feasible pose (the fingers may then sit off
KEY_ULNAR_MAX = 15.             # deg the palm may lean toward the little-finger side (ulnar deviation) while a key
# is played: the side real playing bends toward freely, yet the user's screenshot shows a deep ulnar kink (26 deg,
# the palm 151) as by far the ugliest read -- the forearm must move instead, which is what a real player does
# (measured before the cap: 70% of the left hand's played frames past 15 deg, worst 44)


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


def palm_vs(arm, yaw, roll, forearm):
    """The palm's long axis vs a forearm vector, interior angle in degrees (180 straight)."""
    from .kinematics import q_mat3
    palm = q_mat3(arm.palm_q(yaw, roll)) @ np.array([0., 1., 0.], dtype=FLOAT)
    forearm = np.asarray(forearm, FLOAT)
    return 180. - math.degrees(math.acos(np.clip(float(palm @ forearm) / max(np.linalg.norm(forearm), 1e-9), -1., 1.)))


def finger_deg(arm, yaw, roll):
    """Where the fingers point, in deg off the keys' length: 0 is straight down the keyboard, +-90 across the keys,
    and past 90 the hand points back at the player (which a keyboard never wants). The palm's world orientation is
    q(yaw, roll) alone -- the elbow enters the wrist's comfort, not where the fingers point."""
    from .kinematics import q_mat3
    axis = q_mat3(arm.palm_q(yaw, roll)) @ np.array([0., 1., 0.], dtype=FLOAT)
    return math.degrees(math.atan2(float(axis[0]), float(axis[1])))


def radial_dev(arm, yaw, roll, forearm, side):
    """Deg the palm's long axis leans toward the thumb side (+) or the little-finger side (-) of the forearm:
    the wrist's lateral bend, measured on the palm axis the way the render has it. +side*0 is never right: the
    thumb sits toward the keys' far end on each hand, so the thumb direction is -side on the world x."""
    from .kinematics import q_mat3
    p = q_mat3(arm.palm_q(yaw, roll)) @ np.array([0., 1., 0.], dtype=FLOAT)
    f = np.asarray(forearm, FLOAT)
    p = p - (p @ f) * f
    th = np.array([-side, 0., 0.], dtype=FLOAT)
    th = th - (th @ f) * f
    n = np.linalg.norm(th)
    if n < 1e-9:
        return 0.
    return math.degrees(math.asin(max(-1., min(1., float(p @ th) / n))))


def turn_to_keys(arm, wrist, yaw, roll, elbow, tol=KEY_TOL, floor=KEY_FLOOR,
                 travel=KEY_TRAVEL, step=KEY_STEP, rolls=(0.,), strict=False,
                 side=0., radial_max=KEY_RADIAL_MAX, ulnar_max=KEY_ULNAR_MAX):
    """((yaw, roll), palm-vs-forearm deg) for a hand at ``wrist`` on the forearm from ``elbow``: the least turn of
    the relaxed yaw that puts the fingers down the keys, walked outward one degree at a time (nearest first, so the
    first hit is the least turn). ``rolls``: the roll offsets tried, each one before the next -- the yaw is searched
    to the end of its travel within the first; ``(0.,)`` keeps the planned roll (the little-finger side down), the
    wide set lets a hand whose plane cannot turn roll the forearm to buy the keys. ``side`` (+1 right, -1 left)
    engages the lean caps: a lean toward the thumb side past ``radial_max`` or toward the little-finger side past
    ``ulnar_max`` is refused (KEY_RADIAL_MAX / KEY_ULNAR_MAX) -- the player moves the forearm there, not the wrist,
    so the fingers may end up off the keys with the hand upright.

    Two searches meet here, and the floor settles them: a pose that folds the wrist past ``floor`` is not taken even
    where it straightens the hand on the keys. Where the travel runs out first (the forearm arrives from too far
    sideways) the least crooked feasible pose is returned, unless ``strict`` -- then the relaxed yaw stands, for a
    caller that would rather not move at all than move to somewhere it did not want."""
    forearm = np.asarray(wrist, FLOAT)[:3] - np.asarray(elbow, FLOAT)
    f = forearm / max(np.linalg.norm(forearm), 1e-9)
    # A roll that arrives outside the bounds (the attitude curve can carry one past them: the canon's opening right
    # wrist sits at -22.4 deg) used to fail the bounds test on every candidate below, so the search never ran and the
    # hand kept its crooked yaw -- 14 deg off the keys for the first second, until the roll walked back in range on
    # its own. Clamping it in makes an out-of-range input searchable instead of silently unalignable.
    roll = float(np.clip(roll, ROLL_BOUNDS[0], ROLL_BOUNDS[1]))
    here = abs(finger_deg(arm, yaw, roll))
    if here <= tol:
        return (float(yaw), float(roll)), palm_vs(arm, yaw, roll, forearm)
    best, best_palm = (here, float(yaw), float(roll)), palm_vs(arm, yaw, roll, forearm)
    for droll in rolls:
        r = roll + droll
        if not (ROLL_BOUNDS[0] <= r <= ROLL_BOUNDS[1]):
            continue
        for k in range(1, int(travel / step) + 1):
            for sgn in (-1., 1.):
                y = yaw + sgn * k * step
                d = palm_vs(arm, y, r, forearm)
                if d < floor:           # the wrist's floor: this way is closed, the other side may still be open
                    continue
                dev = radial_dev(arm, y, r, f, side) if side else 0.   # + toward the thumb, - toward the little finger
                if side and ((sgn == -side and dev > radial_max) or (sgn == side and -dev > ulnar_max)):
                    continue            # a lean past either cap: the hand reads wrenched -- the
                f_deg = abs(finger_deg(arm, y, r))   # forearm should have moved, so this pose is not taken
                if f_deg <= tol:
                    return (y, r), d    # nearest first in this order, so the least roll and the least turn
                if f_deg < best[0]:
                    best, best_palm = (f_deg, y, r), d
    if strict:
        return (float(yaw), float(roll)), palm_vs(arm, yaw, roll, forearm)
    return (best[1], best[2]), best_palm


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


def natural_orientation(solver, wrist, shoulder=None, elbow=None, keys=True):
    """(yaw, roll) parameters of the relaxed hand with the wrist at ``wrist`` (world), its forearm from ``shoulder``:
    the standard seat's (``standard_shoulder``) unless given (the hand on the lap: the actual body's).
    ``elbow`` (world) replaces the elbow the standard seat's natural hang predicts: the second orientation pass
    hands it the elbow the swing curve actually places, so the comfort corridor sees the arm the render will have
    (the trunk holds that elbow wide, and the forearm's own yaw grows from 4 deg to 11-39 -- the palm was left bent
    against it while the corridor, reading the natural hang, thought the wrist was straight).
    ``keys``: the hand is over the keyboard, so the yaw turns to the keys (``KEY_AIM``) instead of only continuing
    the forearm -- see the constant. False for the hand on the lap, where there is no keyboard to point down."""
    y0, dy, t0, dt = _calibration(solver)
    arm = solver.arm
    shoulder = standard_shoulder(solver) if shoulder is None else np.asarray(shoulder, FLOAT)
    if elbow is None:
        try:
            elbow = closest_elbow(shoulder, np.asarray(wrist, FLOAT), arm.l1, arm.l2, arm.side)
        except ValueError:
            elbow = None
    if elbow is None:
        forearm_yaw = 0.
    else:
        forearm = np.asarray(wrist, FLOAT) - np.asarray(elbow, FLOAT)
        forearm_yaw = math.degrees(math.atan2(forearm[0], forearm[1]))
    yaw = (HAND_FOLLOW * forearm_yaw - y0) / dy
    roll = (-LITTLE_SIDE_DOWN - t0) / dt
    if KEY_AIM and keys and elbow is not None:   # the keys are the aim; the corridor's line is this turn's floor
        (yaw, roll), _ = turn_to_keys(arm, wrist, yaw, roll, elbow, strict=True, side=arm.side)
    elif PALM_CORRIDOR and abs(dy) > 1e-9:  # the comfort corridor: where the relaxed hand would sit below the
        now = palm_forearm_deg(solver, wrist, yaw, roll, shoulder, elbow=elbow)   # sustained line, the palm follows
        if now < PALM_SOFT_DEG:         # the forearm further -- the wrist unbends (the arm line, not the hand's
            full = (forearm_yaw - y0) / dy   # bend, carries the reach); never past the forearm itself
            far = palm_forearm_deg(solver, wrist, full, roll, shoulder, elbow=elbow)
            if far > now:
                share = 1. if far <= PALM_SOFT_DEG else min(1., HAND_FOLLOW + (1. - HAND_FOLLOW) *
                                                            (PALM_SOFT_DEG - now) / (far - now))
                yaw = (share * forearm_yaw - y0) / dy
    return np.array([yaw, roll], dtype=FLOAT)
