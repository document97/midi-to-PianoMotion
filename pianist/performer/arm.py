"""The avatar's arms: analytic shoulder-elbow-wrist chain (numpy version of lib/arm_driver.ArmDriver.local_pose).

Kept apart from kinematics.py: the hand's feasibility (the planner's pair table, whose cache key digests
kinematics.py, pose.py and reach_table.py) never depends on the arm, so arm posture can change freely.
"""
import math

import numpy as np

from .kinematics import basis, loc_rot_scale, mat3_q, q_axis, q_mat3, q_mul

ELBOW_OUT = .035                # elbow rest: this far outside the shoulder (elbows a little wider than the shoulders)
ELBOW_FOLLOW = .4               # share of the hand's sideways position the elbow follows


def closest_elbow(shoulder, wrist, l1, l2, side, reference=None):
    """arm_driver.closest_elbow."""
    s, w = np.asarray(shoulder, float), np.asarray(wrist, float)
    d = w - s
    distance = float(np.linalg.norm(d))
    if distance >= l1 + l2 - .0001 or distance <= abs(l1 - l2) + .0001:
        raise ValueError(f'Arm target outside unstretched reach: {distance:.6f}')
    u = d / distance
    a = (l1 * l1 - l2 * l2 + distance * distance) / (2 * distance)
    radius = math.sqrt(max(0, l1 * l1 - a * a))
    center = s + a * u
    downward = np.array([0., 0., -1.])
    t = downward - u * np.dot(downward, u)
    if np.linalg.norm(t) < .01:
        t = np.array([side, 0., 0.]) - u * side * u[0]
    t /= np.linalg.norm(t)
    q = np.cross(u, t)
    # The upper arm hangs: the elbow rests a little outside the shoulder and follows the hand only partly,
    # so a hand in front of the body is reached with the forearm angled in (not straight ahead).
    rest_x = s[0] + side * ELBOW_OUT
    preferred = np.array([rest_x + ELBOW_FOLLOW * (w[0] - rest_x), s[1] + .055, w[2] - .035])

    def cost(angle):
        point = center + radius * (t * math.cos(angle) + q * math.sin(angle))
        error = point - preferred
        value = 8 * error[0] ** 2 + error[1] ** 2 + 2 * error[2] ** 2
        value += 20 * max(0, point[2] - w[2] - .025) ** 2
        if reference is not None:
            value += .4 * float(np.sum((point - np.asarray(reference)) ** 2))
        return value, point
    grid = np.linspace(-math.pi, math.pi, 81)
    candidates = center + radius * (np.cos(grid)[:, None] * t + np.sin(grid)[:, None] * q)
    errors = candidates - preferred
    costs = (errors * errors) @ np.array([8., 1., 2.]) + 20 * np.maximum(0, candidates[:, 2] - w[2] - .025) ** 2
    if reference is not None:
        costs += .4 * np.sum((candidates - np.asarray(reference)) ** 2, axis=1)
    angle = grid[int(np.argmin(costs))]
    step = math.pi / 40
    for _ in range(12):
        angle = min([angle - step, angle, angle + step], key=lambda theta: cost(theta)[0])
        step *= .5
    return cost(angle)[1]


class Arm:
    def __init__(self, data, arm_world):
        self.data = data
        self.arm_world = arm_world
        self.arm_inverse = np.linalg.inv(arm_world)
        self.pre_upper_world = np.array(data['pre_upper_world'])
        self.l1, self.l2, self.side = data['l1'], data['l2'], data['sign']
        self.rest_upper_q, self.rest_elbow_q = np.array(data['rest_upper_q']), np.array(data['rest_elbow_q'])
        self.rest_upper_basis = np.array(data['rest_upper_basis'])
        self.rest_elbow_basis = np.array(data['rest_elbow_basis'])
        self.polarity = data['hinge_polarity']
        names = data['bones']
        self.names = names
        self.scale = {k: np.array(data['scales'][names[k]]) for k in ('upper', 'elbow', 'wrist')}
        self.upper_elbow_link = np.array(data['upper_elbow_link'])
        self.elbow_twist_link = np.array(data['elbow_twist_link'])
        self.twist_wrist_link = np.array(data['twist_wrist_link'])
        self.palm = np.array(data['palm'])
        self.palm_forward = np.array(data['palm_forward'])

    def palm_q(self, yaw, roll):
        return q_mul(q_axis((0, 0, 1), yaw), q_mul(q_axis(self.palm_forward, roll), self.palm))

    def pose(self, wrist, yaw=0., roll=0., pre_upper_world=None):
        """World wrist matrix, armature-space wrist matrix and the bones' local quaternions."""
        pre_upper = self.pre_upper_world if pre_upper_world is None else pre_upper_world
        shoulder = pre_upper[:3, 3]
        wrist = np.asarray(wrist, float)
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
        twist4 = np.eye(4)
        twist4[:3, :3] = q_mat3(twist_q)
        wrist_pre = forearm @ self.elbow_twist_link @ twist4 @ self.twist_wrist_link
        local = {self.names['upper']: mat3_q(np.linalg.inv(pre_upper) @ upper),
                 self.names['elbow']: mat3_q(np.linalg.inv(upper @ self.upper_elbow_link) @ forearm),
                 self.names['wrist']: mat3_q(np.linalg.inv(wrist_pre) @ palm),
                 self.names['twist']: twist_q}
        return dict(world=palm, armature=self.arm_inverse @ palm, local=local, elbow=elbow, shoulder=shoulder)
