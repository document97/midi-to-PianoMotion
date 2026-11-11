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

ELBOW_OUT = .035                # elbow rest: this far outside the shoulder (elbows a little wider than the shoulders)
ELBOW_FOLLOW = .4               # share of the hand's sideways position the elbow follows


def closest_elbow(shoulder, wrist, l1, l2, side, reference=None):
    """arm_driver.closest_elbow."""
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
    q = np.cross(u, t)
    # The upper arm hangs: the elbow rests a little outside the shoulder and follows the hand only partly,
    # so a hand in front of the body is reached with the forearm angled in (not straight ahead).
    rest_x = s[0] + side * ELBOW_OUT
    preferred = np.array([rest_x + ELBOW_FOLLOW * (w[0] - rest_x), s[1] + .055, w[2] - .035], dtype=FLOAT)

    def cost(angle):
        point = center + radius * (t * math.cos(angle) + q * math.sin(angle))
        error = point - preferred
        value = 8 * error[0] ** 2 + error[1] ** 2 + 2 * error[2] ** 2
        value += 20 * max(0, point[2] - w[2] - .025) ** 2
        if reference is not None:
            value += .4 * float(np.sum((point - np.asarray(reference)) ** 2))
        return value, point
    grid = np.linspace(-math.pi, math.pi, 81, dtype=FLOAT)
    candidates = center + radius * (np.cos(grid)[:, None] * t + np.sin(grid)[:, None] * q)
    errors = candidates - preferred
    costs = (errors * errors) @ np.array([8., 1., 2.], dtype=FLOAT) + 20 * np.maximum(0, candidates[:, 2] - w[2] - .025) ** 2
    if reference is not None:
        costs += .4 * np.sum((candidates - np.asarray(reference)) ** 2, axis=1)
    angle = grid[int(np.argmin(costs))]
    step = math.pi / 40
    for _ in range(12):
        angle = min([angle - step, angle, angle + step], key=lambda theta: cost(theta)[0])
        step *= .5
    return cost(angle)[1]


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
        self.palm = q_mul(q_axis(across / np.linalg.norm(across), PALM_PITCH), np.array(data['palm'], dtype=FLOAT)).astype(FLOAT)

    def palm_q(self, yaw, roll):
        return q_mul(q_axis((0, 0, 1), yaw), q_mul(q_axis(self.palm_forward, roll), self.palm))

    def pose(self, wrist, yaw=0., roll=0., pre_upper_world=None):
        """World wrist matrix, armature-space wrist matrix and the bones' local quaternions."""
        pre_upper = self.pre_upper_world if pre_upper_world is None else pre_upper_world
        shoulder = pre_upper[:3, 3]
        wrist = np.asarray(wrist, FLOAT)
        elbow = closest_elbow(shoulder, wrist, self.l1, self.l2, self.side)
        upper_dir = (elbow - shoulder) / np.linalg.norm(elbow - shoulder)
        forearm_dir = (wrist - elbow) / np.linalg.norm(wrist - elbow)
        hinge = np.cross(upper_dir, forearm_dir)
        hinge = hinge / np.linalg.norm(hinge) * self.polarity
        upper_q = q_mul(mat3_q(basis(hinge, upper_dir) @ self.rest_upper_basis.T), self.rest_upper_q)
        elbow_q = q_mul(mat3_q(basis(hinge, forearm_dir) @ self.rest_elbow_basis.T), self.rest_elbow_q)
        upper = loc_rot_scale(shoulder, upper_q, self.scale['upper'])
        forearm = loc_rot_scale(elbow, elbow_q, self.scale['elbow'])
        twist_q = q_axis((0, 1, 0), roll)
        palm = loc_rot_scale(wrist, self.palm_q(yaw, roll), self.scale['wrist'])
        twist4 = np.eye(4, dtype=FLOAT)
        twist4[:3, :3] = q_mat3(twist_q)
        wrist_pre = forearm @ self.elbow_twist_link @ twist4 @ self.twist_wrist_link
        local = {self.names['upper']: mat3_q(np.linalg.inv(pre_upper) @ upper),
                 self.names['elbow']: mat3_q(np.linalg.inv(upper @ self.upper_elbow_link) @ forearm),
                 self.names['wrist']: mat3_q(np.linalg.inv(wrist_pre) @ palm),
                 self.names['twist']: twist_q}
        return dict(world=palm, armature=self.arm_inverse @ palm, local=local, elbow=elbow, shoulder=shoulder, forearm=forearm)
