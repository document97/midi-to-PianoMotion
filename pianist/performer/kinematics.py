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

# Every array of the solve is float32 (Blender stores poses and keys as float32 too). Steps, tolerances
# and the Jacobians are chosen for it (pose.PoseSolver._levenberg_marquardt, fast_settle).
FLOAT = np.float32
CALIBRATION = Path(__file__).resolve().parents[1] / 'hand_calibration.json'

# 针对当前模型与观感调的参数；换模型需微调（髋宽倍数）。
LAP_FROM_SURFACE = dict(inward=.10574, forward=.05151, down=.13620)
# The reference this correction was fitted on (development model, standard seat). Documentation and tests
# only -- no hand is placed from this constant: lap_measured() -> + LAP_FROM_SURFACE -> STANDARD_LAP.
STANDARD_LAP = {'L': (-.104, -.19, .672), 'R': (.104, -.19, .672)}
LAP_POINT = STANDARD_LAP       # backwards-compatible name for the reference point


def lap_from_surface(surface, hand, hips, neutral_wrist):
    """World resting wrist for one hand, from this model's collision-measured thigh point.

    ``surface`` is rig_export.lap_measured()'s ray-cast hit at export time, ``hips`` the two leg-bone heads
    in world space (``torso.hips`` in a rig), and ``neutral_wrist`` this hand's playing wrist. The frame is
    taken from the body itself -- hip width sets the scale, the hip-to-keyboard direction sets forward --
    so the fitted offset follows a different model's size and facing instead of the development scene's.
    """
    p = np.asarray(surface, dtype=float)
    flat = [float(v) for v in p]
    hip_l, hip_r = (np.asarray(h, dtype=float) for h in hips)
    width = float(np.linalg.norm(hip_r - hip_l))
    if width < 1e-6:
        return flat
    mid = (hip_l + hip_r) * .5
    lateral = (hip_r - hip_l) / width                        # the body's own left-to-right axis
    forward = np.asarray(neutral_wrist, dtype=float) - mid
    forward[2] = 0.
    # The hand sits off to one side, so hip-to-keyboard is not perpendicular to the hip line. Take only
    # its forward part: otherwise the sideways share tilts the frame and drags the resting point off.
    forward -= lateral * float(np.dot(forward, lateral))
    length = float(np.linalg.norm(forward))
    if length < 1e-6:
        return flat
    forward /= length
    # Read outward from the surface rather than from bone names, so a mirrored rig still moves inward.
    outward = lateral if float(np.dot(p - mid, lateral)) > 0 else -lateral
    c = LAP_FROM_SURFACE
    result = p + width * (-outward * c['inward'] + forward * c['forward']
                          - np.array([0., 0., c['down']]))
    # A ray cast in float32 lands within a micron of the fitted point; the solver's plan is sensitive to
    # much less than that (see seat_free_hands), so keep the point on a millimetre grid. This reproduces
    # the approved pose exactly and stops mesh-evaluation noise from changing the whole hand plan.
    return [round(float(v), 3) for v in result]

DIGITS = (1, 2, 3, 4, 5)
VALUE_NAMES = ('swing_lateral', 'swing_lift', 'mcp_delta', 'ip_delta')
# Sideways swing (deg, +-) widened over the calibration's working bounds (+-24/25) so the hand spans an
# octave, a small (160 cm) adult hand's limit: the thumb opens at its base joint (real abduction ~50 deg),
# the little finger ~32. Thumb-little minor seventh before, octave (all keys) after; tenths are not one hand.
SWING_LATERAL = {1: 50., 5: 32.}
# The long fingers' last joint never straightens fully: a relaxed or playing finger keeps a small bend there (the
# calibration's working bound allows 5 deg, which reads as a stiff, straight tip).
LAST_JOINT_MIN = 14.
# The web between neighbouring fingers: how far apart two neighbours may open (deg, the angle of their first phalanges
# seen on the palm), each swinging sideways on its own up to SWING_LATERAL / the rig's bounds. From a pianist's right
# hand (references/analysis/for_elise_060: index-middle p95 17 / p99 20, middle-ring 13 / 20, ring-little 52 / 66,
# measured from a hand whose little finger rests ~22 deg further out than this rig's), the opening beyond which a
# shape strains (free) and its limit (hard). Thumb to index: not measured; an estimate of the web's reach.
WEB_FREE = {1: 55., 2: 15., 3: 11., 4: 30.}      # gap after digit d (d, d + 1)
WEB_HARD = {1: 65., 2: 18., 3: 18., 4: 44.}


# ---------- quaternion / matrix helpers ----------
def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dtype=FLOAT)


def q_axis(axis, degrees):
    axis = np.asarray(axis, FLOAT)
    axis = axis / np.linalg.norm(axis)
    half = math.radians(degrees) / 2
    return np.r_[np.float32(math.cos(half)), np.float32(math.sin(half)) * axis]


def q_mat3(q):
    w, x, y, z = np.asarray(q, FLOAT) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=FLOAT)


def mat3_q(m):
    """Rotation of a (possibly scaled) 3x3 as a unit quaternion with w >= 0."""
    m = np.asarray(m, FLOAT)[:3, :3]
    m = m / np.linalg.norm(m, axis=0)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.) * 2
        q = np.array([.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s], dtype=FLOAT)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1. + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, .25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s], dtype=FLOAT)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1. + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, .25 * s, (m[1, 2] + m[2, 1]) / s], dtype=FLOAT)
    else:
        s = math.sqrt(1. + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, .25 * s], dtype=FLOAT)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def axis_matrix(axis, radians):
    """Rotation matrix about a unit axis (Rodrigues), without quaternion round trips."""
    x, y, z = axis
    c, s = math.cos(radians), math.sin(radians)
    t = 1 - c
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c]], dtype=FLOAT)


def euler_xyz_matrix(euler):
    x, y, z = euler
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    return np.array([[cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
                     [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
                     [-sy, sx * cy, cx * cy]], dtype=FLOAT)


def compose(loc, rotation, scale):
    m = np.eye(4, dtype=FLOAT)
    m[:3, :3] = rotation * scale
    m[:3, 3] = loc
    return m


def loc_rot_scale(loc, q, scale):
    m = np.eye(4, dtype=FLOAT)
    m[:3, :3] = q_mat3(q) * np.asarray(scale, FLOAT)
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
    y = np.asarray(forward, FLOAT) / np.linalg.norm(forward)
    x = np.asarray(axis, FLOAT) - y * (np.asarray(axis, FLOAT) @ y)
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
        weights = np.zeros((len(rows), len(bones)), dtype=FLOAT)
        local = np.zeros((len(bones), len(rows), 4), dtype=FLOAT)
        local[:, :, 3] = 1.
        for i, r in enumerate(rows):
            for name, (w, *point) in r['bones'].items():
                k = bones.index(name)
                weights[i, k] = w
                local[k, i, :3] = point
        return cls(bones, weights, local)

    def points(self, matrices, arm_world):
        """World positions given {bone: pose matrix (armature space)}."""
        acc = np.zeros((self.weights.shape[0], 4), dtype=FLOAT)
        for k, name in enumerate(self.bones):
            acc += self.weights[:, k:k + 1] * (self.local[k] @ matrices[name].T)
        return (acc @ arm_world.T)[:, :3]


# ---------- rig ----------
class Finger:
    def __init__(self, data, row):
        self.row = row
        self.names = data['names']
        self.pre = np.array(data['pre_in_wrist'], dtype=FLOAT)
        self.root_loc, self.root_scale = np.array(data['root_loc'], dtype=FLOAT), np.array(data['root_scale'], dtype=FLOAT)
        self.child_loc = [np.array(v, dtype=FLOAT) for v in data['child_loc']]
        self.child_scale = [np.array(v, dtype=FLOAT) for v in data['child_scale']]
        self.links = [np.array(m, dtype=FLOAT) for m in data['links']]
        self.neutral_q = np.array(data['neutral_root_q'], dtype=FLOAT)
        self.normal = np.array(data['palm_normal_local'], dtype=FLOAT)
        self.neutral_matrix = q_mat3(self.neutral_q)
        self.base = self.neutral_matrix @ np.array([0., 1., 0.], dtype=FLOAT)
        lift = np.cross(self.base, self.normal)
        self.lift = lift / np.linalg.norm(lift)
        self.pad = SkinSet.from_rows(data['pad'])
        self.clearance = SkinSet.from_rows(data['clearance'])
        self.pad_ids = [r['id'] for r in data['pad']]
        self.clearance_ids = [r['id'] for r in data['clearance']]
        b = data['bounds']
        self.lower = np.array([b[k][0] for k in VALUE_NAMES], FLOAT)
        self.upper = np.array([b[k][1] for k in VALUE_NAMES], FLOAT)
        self.twist = 0.
        self._matrices = {}

    def root_matrix(self, lateral, lift):
        """Root bone rotation (3x3) for the swing values: arc(base -> swung axis) @ neutral, then for the thumb a
        turn about the swung axis that follows its sideways spread (``twist`` deg per deg)."""
        axis = axis_matrix(self.normal, math.radians(lateral)) @ (axis_matrix(self.lift, math.radians(lift)) @ self.base)
        k = np.cross(self.base, axis)
        sine = math.sqrt(k @ k)
        swing = self.neutral_matrix if sine < 1e-12 else             axis_matrix(k / sine, math.atan2(sine, float(self.base @ axis))) @ self.neutral_matrix
        if self.twist:
            swing = axis_matrix(axis / np.linalg.norm(axis), math.radians(self.twist * lateral)) @ swing
        return swing

    def local_rotations(self, values):
        """(root quaternion, mcp euler, ip euler) written to the bones."""
        lateral, lift, mcp, ip = values
        return mat3_q(self.root_matrix(lateral, lift)), child_euler(self.row, 0, mcp), child_euler(self.row, 1, ip)

    def local_matrices(self, values):
        """Bone matrices relative to the wrist bone's pose matrix (kept per exact ``values``: a solver's difference
        steps move one finger at a time and ask for its neighbours' unchanged ones again). Read-only."""
        array = np.asarray(values)
        key = (array.dtype.str, array.tobytes(), self.twist)
        hit = self._matrices.get(key)
        if hit is None:
            if len(self._matrices) > 20000:
                self._matrices.clear()
            hit = self._matrices[key] = self._local_matrices(values)
        return hit

    def _local_matrices(self, values):
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
        # Each hand rests on THIS model's thigh: rig_export writes the ray-cast surface in `lap_surface`,
        # and the same fitted relative correction turns it into the resting wrist. Rigs exported before
        # that field existed already carry the approved point in `lap`; keep it for backwards compatibility.
        hips = (data.get('torso') or {}).get('hips')
        for h in 'LR':
            hand = data['hands'][h]
            surface = hand.get('lap_surface')
            if surface is not None and hips:
                hand['lap'] = list(lap_from_surface(surface, h, hips, hand['neutral_wrist']))
        self.data = data
        cal = json.loads(CALIBRATION.read_text(encoding='utf-8'))
        self.arm_world = np.array(data['arm_world'], dtype=FLOAT)
        from .arm import Arm        # arm posture lives apart: the pair table's code digest does not cover it
        self.arms = {h: Arm(data['hands'][h], self.arm_world) for h in 'LR'}
        self.fingers = {h: {int(d): Finger(f, cal['hands'][h]['fingers'][d]) for d, f in data['hands'][h]['fingers'].items()}
                        for h in 'LR'}
        self.neutral_wrist = {h: np.array(data['hands'][h]['neutral_wrist'], dtype=FLOAT) for h in 'LR'}
        self.keyboard = data['keyboard']
        self.fps = data['fps']
        self.neutral_bones = {k: np.array(m, dtype=FLOAT) for k, m in data['neutral_bones'].items()}
        self.wrist_bone = {h: data['hands'][h]['bones']['wrist'] for h in 'LR'}
        self.neutral_wrist_inverse = {h: np.linalg.inv(np.array(data['hands'][h]['neutral_wrist_matrix'], dtype=FLOAT)) for h in 'LR'}
        self.bone_owner = {h: {name: d for d, f in self.fingers[h].items() for name in f.names} for h in 'LR'}
        self._web = {}
        for h in 'LR':
            for d, swing in SWING_LATERAL.items():
                self.fingers[h][d].lower[0], self.fingers[h][d].upper[0] = -swing, swing
            for d in (2, 3, 4, 5):
                self.fingers[h][d].lower[3] = max(self.fingers[h][d].lower[3], LAST_JOINT_MIN)

    def web(self, hand):
        """(rest, slope, free, hard), each (6, 6) by digit: two fingers a < b are open by
        ``rest[a, b] + slope[a, b] * (lateral_b - lateral_a)`` deg (fitted to the first phalanges on the palm, within
        half a degree), at most ``hard[a, b]`` (the gaps' WEB_HARD between them summed) and without strain up to
        ``free[a, b]``."""
        if hand not in self._web:
            fingers = self.fingers[hand]
            normal = np.mean([f.normal for f in fingers.values()], axis=0)
            normal /= np.linalg.norm(normal)

            def direction(d, lateral):
                v = fingers[d].local_matrices(np.array([lateral, 0., 0., 0.], FLOAT))[0][:3, 1]
                v = v - normal * (v @ normal)
                return v / np.linalg.norm(v)

            def angle(a, la, b, lb):
                u, v = direction(a, la), direction(b, lb)
                return math.degrees(math.atan2(float(np.cross(u, v) @ normal), float(u @ v)))

            gap_rest, gap_slope = {}, {}
            for d in (1, 2, 3, 4):
                rest = angle(d, 0., d + 1, 0.)
                slope = (angle(d, -10., d + 1, 10.) - angle(d, 10., d + 1, -10.)) / 40.
                # opening: the direction that takes the two fingertips apart
                tip = lambda e, lat: fingers[e].local_matrices(np.array([lat, 0., 0., 0.], FLOAT))[2][:3, 3]
                apart = np.linalg.norm(tip(d + 1, 10.) - tip(d, -10.)) > np.linalg.norm(tip(d + 1, -10.) - tip(d, 10.))
                sign = 1. if (slope > 0) == apart else -1.
                gap_rest[d], gap_slope[d] = sign * rest, sign * slope
            rest, slope, free, hard = (np.zeros((6, 6)) for _ in range(4))
            for a in DIGITS:
                for b in DIGITS:
                    if a < b:
                        gaps = range(a, b)
                        rest[a, b] = sum(gap_rest[g] for g in gaps)
                        slope[a, b] = float(np.mean([gap_slope[g] for g in gaps]))
                        free[a, b] = sum(WEB_FREE[g] for g in gaps)
                        hard[a, b] = sum(WEB_HARD[g] for g in gaps)
            self._web[hand] = (rest, slope, free, hard)
        return self._web[hand]

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

