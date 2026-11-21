"""The avatar's arms: analytic shoulder-elbow-wrist chain (numpy version of lib/arm_driver.ArmDriver.local_pose).

Kept apart from kinematics.py: the hand's feasibility (the planner's pair table, whose cache key digests
kinematics.py, pose.py and reach_table.py) never depends on the arm, so arm posture can change freely.
"""
import math

import numpy as np

from .kinematics import FLOAT, basis, loc_rot_scale, mat3_q, q_axis, q_mat3, q_mul

# deg the playing palm is tilted up from the calibrated (bake) hand, about the axis across the hand: the back of the hand
# rises from the wrist to the knuckles (pose.ARCHED bends the knuckles down past it, so they are its highest point)
PALM_PITCH = 15.
# On the lap the hand lies flat: the playing pitch would arch the knuckles 15 deg off the thigh. The pitch eases
# from this to PALM_PITCH across the intro lift, so the hand turns onto the keys as it comes up.
LAP_PITCH = -8.
# The intro's droop stages, the user's spec (2026-11-17): the flat draw keeps the lap's plane (fingertips
# still sliding on the thigh); the lifted hand hangs from the wrist; the poke's latter half flattens the
# palm; a brief wrist tension tilts the palm axis 30 deg off vertical (TILT_PITCH, reduced from the first
# 45-deg take -- the user trimmed the hover's amplitude); the wrist then lets go and the palm falls toward
# flat (FALL_PITCH) while the hand creeps forward; the reach crosses the key fronts still slightly drooped
# (EDGE_DROOP) and flattens onto the playing pitch above the keys.
HANG_PITCH = -33.
FLAT_PITCH = 0.
TILT_PITCH = 30.
FALL_PITCH = 4.
TOP_DROOP = -22.
EDGE_DROOP = -18.
# Closing gesture (the user's 2026-11-18 spec): the wrist lifts 2-3 cm off the last chord, then lets go --
# the palm hangs 20-25 deg while the curled fingers cross the white-key fronts, and flattens only onto the lap.
# PALM_PITCH is +15 (the playing plane), so a 25 deg droop from the keys is -10, not 25 deg below flat: deeper
# than this and the still-tucked fingertips pass through the key tops while the hand crosses the fronts.
OUTRO_DROOP = -10.

ELBOW_OUT = .035                # elbow rest: this far outside the shoulder (elbows a little wider than the shoulders)
ELBOW_FOLLOW = .4               # share of the hand's sideways position the elbow follows
# The elbow rests AGAINST the ribs, never inside them: the torso's side surface (hip joints + this shell) is a
# wall the elbow may not cross. Without it a chest that leans far toward the playing hand (LEAN_SHIFT is 20 cm)
# drags the other shoulder over the lap hand and the upper arm is solved straight through the ribcage.
BODY_SHELL = .030               # m the body's surface lies outside the hip joints (measured on the mesh: 30-40 mm)
ELBOW_BODY = 20000.             # weight of the wall in the elbow's cost (its other terms are ~1-8 per m^2)


def elbow_circle(shoulder, wrist, l1, l2, side):
    """The elbow's circle for a wrist the arm can reach: (center, radius, t, q), the elbow at angle a (rad) is
    center + radius * (t * cos(a) + q * sin(a)); angle 0 hangs the elbow down (t is the down direction projected
    accross the shoulder-wrist line), q = u x t. Raises ValueError when the wrist is outside the unstretched arm."""
    s, w = np.asarray(shoulder, FLOAT), np.asarray(wrist, FLOAT)
    d = w - s
    distance = float(np.linalg.norm(d))
    if distance >= l1 + l2 - .0001 or distance <= abs(l1 - l2) + .0001:
        raise ValueError(f'Arm target outside unstretched reach: {distance:.6f}')
    u = d / distance
    a = (l1 * l1 - l2 * l2 + distance * distance) / (2 * distance)
    radius = math.sqrt(max(0, l1 * l1 - a * a))
    center = s + a * u
    downward = np.array([0., 0., -1.], dtype=FLOAT)
    t = downward - u * np.dot(downward, u)
    if np.linalg.norm(t) < .01:
        t = np.array([side, 0., 0.], dtype=FLOAT) - u * side * u[0]
    t /= np.linalg.norm(t)
    return center, radius, t, np.cross(u, t)


def closest_elbow(shoulder, wrist, l1, l2, side, reference=None, angle=None, wall=None):
    """arm_driver.closest_elbow; ``angle`` (rad on elbow_circle, the swing curve's convention): the elbow placed
    there instead of optimized to the preferred rest. ``wall``: x the elbow may not cross (the torso's side
    surface; the hand's own side, so the left elbow stays left of it and the right elbow right of it)."""
    center, radius, t, q = elbow_circle(shoulder, wrist, l1, l2, side)
    s, w = np.asarray(shoulder, FLOAT), np.asarray(wrist, FLOAT)
    if angle is not None:
        return center + radius * (t * math.cos(angle) + q * math.sin(angle))
    # The upper arm hangs: the elbow rests a little outside the shoulder and follows the hand only partly,
    # so a hand in front of the body is reached with the forearm angled in (not straight ahead).
    rest_x = s[0] + side * ELBOW_OUT
    preferred = np.array([rest_x + ELBOW_FOLLOW * (w[0] - rest_x), s[1] + .055, w[2] - .035], dtype=FLOAT)

    def cost(angle):
        point = center + radius * (t * math.cos(angle) + q * math.sin(angle))
        error = point - preferred
        value = 8 * error[0] ** 2 + error[1] ** 2 + 2 * error[2] ** 2
        value += 20 * max(0, point[2] - w[2] - .025) ** 2
        if wall is not None:
            value += ELBOW_BODY * max(0., side * (wall - point[0])) ** 2
        if reference is not None:
            value += .4 * float(np.sum((point - np.asarray(reference)) ** 2))
        return value, point
    grid = np.linspace(-math.pi, math.pi, 81, dtype=FLOAT)
    candidates = center + radius * (np.cos(grid)[:, None] * t + np.sin(grid)[:, None] * q)
    errors = candidates - preferred
    costs = (errors * errors) @ np.array([8., 1., 2.], dtype=FLOAT) + 20 * np.maximum(0, candidates[:, 2] - w[2] - .025) ** 2
    if wall is not None:
        costs += ELBOW_BODY * np.maximum(0., side * (wall - candidates[:, 0])) ** 2
    if reference is not None:
        costs += .4 * np.sum((candidates - np.asarray(reference)) ** 2, axis=1)
    angle = grid[int(np.argmin(costs))]
    step = math.pi / 40
    for _ in range(12):
        angle = min([angle - step, angle, angle + step], key=lambda theta: cost(theta)[0])
        step *= .5
    return cost(angle)[1]


def closest_elbow_angle(shoulder, wrist, l1, l2, side):
    """The angle on the elbow's circle of the preferred rest placement (elbows.py pulls the swing curve toward
    it): the arm's natural hang for this wrist."""
    elbow = closest_elbow(shoulder, wrist, l1, l2, side)
    center, radius, t, q = elbow_circle(shoulder, wrist, l1, l2, side)
    e = np.asarray(elbow, FLOAT) - center
    return math.atan2(float(e @ q), float(e @ t))


# The standard seat's shoulder relative to a hand's neutral wrist over the keys, in arm lengths (upper arm + forearm):
# x toward the hand's own side, y toward the player, z up (a seated pianist's, measured on the development scene).
# Whatever depends on the forearm's direction in the hand's plan takes it from here, not from the scene's seat.
STANDARD_SHOULDER = np.array([-.006, -.766, .427])


def standard_shoulder(arm, neutral_wrist):
    """World position of the standard seat's shoulder for the hand whose neutral wrist is ``neutral_wrist``."""
    offset = STANDARD_SHOULDER * (arm.l1 + arm.l2)
    offset[0] *= arm.side                  # toward the hand's own side: left hand -x, right hand +x
    return (np.asarray(neutral_wrist, FLOAT) + offset).astype(FLOAT)


class Arm:
    def __init__(self, data, arm_world):
        self.data = data
        self.arm_world = arm_world
        self.arm_inverse = np.linalg.inv(arm_world)
        self.pre_upper_world = np.array(data['pre_upper_world'], dtype=FLOAT)
        self.l1, self.l2, self.side = data['l1'], data['l2'], data['sign']
        self.rest_upper_q, self.rest_elbow_q = np.array(data['rest_upper_q'], dtype=FLOAT), np.array(data['rest_elbow_q'], dtype=FLOAT)
        self.rest_upper_basis = np.array(data['rest_upper_basis'], dtype=FLOAT)
        self.rest_elbow_basis = np.array(data['rest_elbow_basis'], dtype=FLOAT)
        self.polarity = data['hinge_polarity']
        names = data['bones']
        self.names = names
        self.scale = {k: np.array(data['scales'][names[k]], dtype=FLOAT) for k in ('upper', 'elbow', 'wrist')}
        self.upper_elbow_link = np.array(data['upper_elbow_link'], dtype=FLOAT)
        self.elbow_twist_link = np.array(data['elbow_twist_link'], dtype=FLOAT)
        self.twist_wrist_link = np.array(data['twist_wrist_link'], dtype=FLOAT)
        self.palm_forward = np.array(data['palm_forward'], dtype=FLOAT)
        across = np.cross(self.palm_forward, (0., 0., 1.))
        self.palm_axis = across / np.linalg.norm(across)
        self.palm_bake = np.array(data['palm'], dtype=FLOAT)
        self.palm = q_mul(q_axis(self.palm_axis, PALM_PITCH), self.palm_bake).astype(FLOAT)
        self.wall = None                    # x the elbow may not cross (set from the rig's hips)

    def set_body_wall(self, center_x, half):
        """Keep the elbow outside the torso: ``center_x`` is the body's midline at the elbow's height and
        ``half`` its side surface's distance from it (see BODY_SHELL)."""
        self.wall = float(center_x + self.side * half)

    def palm_q(self, yaw, roll, pitch=None):
        palm = self.palm if pitch is None else q_mul(q_axis(self.palm_axis, pitch), self.palm_bake)
        return q_mul(q_axis((0, 0, 1), yaw), q_mul(q_axis(self.palm_forward, roll), palm))

    def pose(self, wrist, yaw=0., roll=0., pre_upper_world=None, elbow_angle=None, pitch=None):
        """World wrist matrix, armature-space wrist matrix and the bones' local quaternions; ``elbow_angle``
        (rad on elbow_circle): the elbow placed there (the swing curve's angle) instead of optimized."""
        pre_upper = self.pre_upper_world if pre_upper_world is None else pre_upper_world
        shoulder = pre_upper[:3, 3]
        wrist = np.asarray(wrist, FLOAT)
        elbow = closest_elbow(shoulder, wrist, self.l1, self.l2, self.side, angle=elbow_angle, wall=self.wall)
        upper_dir = (elbow - shoulder) / np.linalg.norm(elbow - shoulder)
        forearm_dir = (wrist - elbow) / np.linalg.norm(wrist - elbow)
        hinge = np.cross(upper_dir, forearm_dir)
        hinge = hinge / np.linalg.norm(hinge) * self.polarity
        upper_q = q_mul(mat3_q(basis(hinge, upper_dir) @ self.rest_upper_basis.T), self.rest_upper_q)
        elbow_q = q_mul(mat3_q(basis(hinge, forearm_dir) @ self.rest_elbow_basis.T), self.rest_elbow_q)
        upper = loc_rot_scale(shoulder, upper_q, self.scale['upper'])
        forearm = loc_rot_scale(elbow, elbow_q, self.scale['elbow'])
        twist_q = q_axis((0, 1, 0), roll)
        palm_q = self.palm_q(yaw, roll, pitch)
        palm = loc_rot_scale(wrist, palm_q, self.scale['wrist'])
        palm_axis = q_mat3(palm_q) @ np.array([0., 1., 0.], dtype=FLOAT)   # the palm bone's +Y: toward the fingers
        palm_deg = 180. - math.degrees(math.acos(np.clip(float(palm_axis @ forearm_dir), -1., 1.)))
        # the interior palm-vs-forearm angle (180 straight; the comfort line the user speaks in)
        twist4 = np.eye(4, dtype=FLOAT)
        twist4[:3, :3] = q_mat3(twist_q)
        wrist_pre = forearm @ self.elbow_twist_link @ twist4 @ self.twist_wrist_link
        local = {self.names['upper']: mat3_q(np.linalg.inv(pre_upper) @ upper),
                 self.names['elbow']: mat3_q(np.linalg.inv(upper @ self.upper_elbow_link) @ forearm),
                 self.names['wrist']: mat3_q(np.linalg.inv(wrist_pre) @ palm),
                 self.names['twist']: twist_q}
        return dict(world=palm, armature=self.arm_inverse @ palm, local=local, elbow=elbow, shoulder=shoulder,
                    forearm=forearm, palm_forearm_deg=palm_deg)
