"""Numpy forward kinematics of the avatar's arms, hands and skin points.

A re-implementation of the bake's own drivers (arm_driver.ArmDriver.local_pose,
thumb_driver.ThumbFrame.matrices, thumb_articulation.child_euler and linear
blend skinning), checked against poses evaluated in Blender (tests). Quaternions
are (w, x, y, z); matrices are 4x4 numpy arrays. No Blender import.

Finger parameters per digit are (swing_lateral, swing_lift, mcp, ip) in degrees,
as in the calibration's working bounds. A finger's local bone rotations depend on
those four values only, never on the wrist, so finger and wrist solve separately.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

CALIBRATION = Path(__file__).resolve().parents[1] / 'hand_calibration.json'
DIGITS = (1, 2, 3, 4, 5)
VALUE_NAMES = ('swing_lateral', 'swing_lift', 'mcp_delta', 'ip_delta')
# Sideways swing (deg, +-) widened over the calibration's working bounds (+-24/25) so the hand spans an
# octave, a small (160 cm) adult hand's limit: the thumb opens at its base joint (real abduction ~50 deg),
# the little finger ~32. Thumb-little minor seventh before, octave (all keys) after; tenths are not one hand.
SWING_LATERAL = {1: 50., 5: 32.}


# ---------- quaternion / matrix helpers ----------
def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def q_axis(axis, degrees):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    half = math.radians(degrees) / 2
    return np.r_[math.cos(half), math.sin(half) * axis]


def q_mat3(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat3_q(m):
    """Rotation of a (possibly scaled) 3x3 as a unit quaternion with w >= 0."""
    m = np.asarray(m, float)[:3, :3]
    m = m / np.linalg.norm(m, axis=0)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.) * 2
        q = np.array([.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1. + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, .25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1. + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, .25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1. + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, .25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def axis_matrix(axis, radians):
    """Rotation matrix about a unit axis (Rodrigues), without quaternion round trips."""
    x, y, z = axis
    c, s = math.cos(radians), math.sin(radians)
    t = 1 - c
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


def euler_xyz_matrix(euler):
    x, y, z = euler
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    return np.array([[cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
                     [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
                     [-sy, sx * cy, cx * cy]])


def compose(loc, rotation, scale):
    m = np.eye(4)
    m[:3, :3] = rotation * scale
    m[:3, 3] = loc
    return m


def loc_rot_scale(loc, q, scale):
    m = np.eye(4)
    m[:3, :3] = q_mat3(q) * np.asarray(scale, float)
    m[:3, 3] = loc
    return m


def child_euler(row, index, bend_degrees):
    """thumb_articulation.child_euler."""
    spec = row.get('joint_articulation', {}).get(row.get('child_joint_names', ('MCP', 'IP'))[index])
    if spec is None:
        result = list(row['base_child_euler'][index])
        result[0] += row['sign'] * math.radians(bend_degrees)
        return result
    result = list(spec['zero_euler'])
    result[spec['axis']] += spec['flexion_sign'] * math.radians(bend_degrees)
    return result


def basis(axis, forward):
    y = np.asarray(forward, float) / np.linalg.norm(forward)
    x = np.asarray(axis, float) - y * (np.asarray(axis, float) @ y)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    z /= np.linalg.norm(z)
    return np.column_stack([x, y, z])


# ---------- skin ----------
@dataclass
class SkinSet:
    """Linear blend skinning of a vertex subset over the bones that deform it."""
    bones: tuple           # bone names
    weights: np.ndarray    # (n, B)
    local: np.ndarray      # (B, n, 4) homogeneous rest point in each bone's rest frame

    @classmethod
    def from_rows(cls, rows):
        bones = tuple(sorted({b for r in rows for b in r['bones']}))
        weights = np.zeros((len(rows), len(bones)))
        local = np.zeros((len(bones), len(rows), 4))
        local[:, :, 3] = 1.
        for i, r in enumerate(rows):
            for name, (w, *point) in r['bones'].items():
                k = bones.index(name)
                weights[i, k] = w
                local[k, i, :3] = point
        return cls(bones, weights, local)

    def points(self, matrices, arm_world):
        """World positions given {bone: pose matrix (armature space)}."""
        acc = np.zeros((self.weights.shape[0], 4))
        for k, name in enumerate(self.bones):
            acc += self.weights[:, k:k + 1] * (self.local[k] @ matrices[name].T)
        return (acc @ arm_world.T)[:, :3]


# ---------- rig ----------
class Finger:
    def __init__(self, data, row):
        self.row = row
        self.names = data['names']
        self.pre = np.array(data['pre_in_wrist'])
        self.root_loc, self.root_scale = np.array(data['root_loc']), np.array(data['root_scale'])
        self.child_loc = [np.array(v) for v in data['child_loc']]
        self.child_scale = [np.array(v) for v in data['child_scale']]
        self.links = [np.array(m) for m in data['links']]
        self.neutral_q = np.array(data['neutral_root_q'])
        self.normal = np.array(data['palm_normal_local'])
        self.neutral_matrix = q_mat3(self.neutral_q)
        self.base = self.neutral_matrix @ np.array([0., 1., 0.])
        lift = np.cross(self.base, self.normal)
        self.lift = lift / np.linalg.norm(lift)
        self.pad = SkinSet.from_rows(data['pad'])
        self.clearance = SkinSet.from_rows(data['clearance'])
        self.pad_ids = [r['id'] for r in data['pad']]
        self.clearance_ids = [r['id'] for r in data['clearance']]
        b = data['bounds']
        self.lower = np.array([b[k][0] for k in VALUE_NAMES], float)
        self.upper = np.array([b[k][1] for k in VALUE_NAMES], float)

    def root_matrix(self, lateral, lift):
        """Root bone rotation (3x3) for the swing values: arc(base -> swung axis) @ neutral."""
        axis = axis_matrix(self.normal, math.radians(lateral)) @ (axis_matrix(self.lift, math.radians(lift)) @ self.base)
        k = np.cross(self.base, axis)
        sine = math.sqrt(k @ k)
        if sine < 1e-12:
            return self.neutral_matrix
        return axis_matrix(k / sine, math.atan2(sine, float(self.base @ axis))) @ self.neutral_matrix

    def local_rotations(self, values):
        """(root quaternion, mcp euler, ip euler) written to the bones."""
        lateral, lift, mcp, ip = values
        return mat3_q(self.root_matrix(lateral, lift)), child_euler(self.row, 0, mcp), child_euler(self.row, 1, ip)

    def local_matrices(self, values):
        """Bone matrices relative to the wrist bone's pose matrix."""
        lateral, lift, mcp, ip = values
        m0 = self.pre @ compose(self.root_loc, self.root_matrix(lateral, lift), self.root_scale)
        m1 = m0 @ self.links[0] @ compose(self.child_loc[0], euler_xyz_matrix(child_euler(self.row, 0, mcp)),
                                          self.child_scale[0])
        m2 = m1 @ self.links[1] @ compose(self.child_loc[1], euler_xyz_matrix(child_euler(self.row, 1, ip)),
                                          self.child_scale[1])
        return m0, m1, m2


class Rig:
    def __init__(self, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        self.data = data
        cal = json.loads(CALIBRATION.read_text(encoding='utf-8'))
        self.arm_world = np.array(data['arm_world'])
        from .arm import Arm        # arm posture lives apart: the pair table's code digest does not cover it
        self.arms = {h: Arm(data['hands'][h], self.arm_world) for h in 'LR'}
        self.fingers = {h: {int(d): Finger(f, cal['hands'][h]['fingers'][d]) for d, f in data['hands'][h]['fingers'].items()}
                        for h in 'LR'}
        self.neutral_wrist = {h: np.array(data['hands'][h]['neutral_wrist']) for h in 'LR'}
        self.keyboard = data['keyboard']
        self.fps = data['fps']
        self.neutral_bones = {k: np.array(m) for k, m in data['neutral_bones'].items()}
        self.wrist_bone = {h: data['hands'][h]['bones']['wrist'] for h in 'LR'}
        self.neutral_wrist_inverse = {h: np.linalg.inv(np.array(data['hands'][h]['neutral_wrist_matrix'])) for h in 'LR'}
        self.bone_owner = {h: {name: d for d, f in self.fingers[h].items() for name in f.names} for h in 'LR'}
        for h in 'LR':
            for d, swing in SWING_LATERAL.items():
                self.fingers[h][d].lower[0], self.fingers[h][d].upper[0] = -swing, swing

    def bone_matrices(self, hand, wrist_armature, values, bones):
        """Pose matrices (armature space) of ``bones`` for a hand pose; ``values`` {digit: 4 values}."""
        out, solved = {}, set()
        rigid = wrist_armature @ self.neutral_wrist_inverse[hand]   # bones that move with the wrist
        for name in bones:
            digit = self.bone_owner[hand].get(name)
            if digit is None:
                out[name] = wrist_armature if name == self.wrist_bone[hand] else rigid @ self.neutral_bones[name]
            elif digit not in solved:
                solved.add(digit)
                finger = self.fingers[hand][digit]
                for bone, m in zip(finger.names, finger.local_matrices(values[digit])):
                    out[bone] = wrist_armature @ m
        return out

    def pad_points(self, hand, digit, wrist_armature, values):
        skin = self.fingers[hand][digit].pad
        return skin.points(self.bone_matrices(hand, wrist_armature, values, skin.bones), self.arm_world)

    def clearance_points(self, hand, digit, wrist_armature, values):
        skin = self.fingers[hand][digit].clearance
        return skin.points(self.bone_matrices(hand, wrist_armature, values, skin.bones), self.arm_world)

