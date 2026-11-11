"""Below the hands (run inside Blender): the bench under the character, the feet on the floor and the right foot on
the piano's damper pedal.

Everything is fitted to the posed skin, not to fixed numbers:
- the feet are put down by their leg IK targets: the left boot flat on the floor, the right boot's heel on the floor
  and its ball on the damper pedal, a little behind the pedal's tip;
- the bench top is lowered to the lowest point of the seated body (hips, thighs; clothes without cloth
  simulation may still dip into the seat);
- pressing turns the right boot about its heel tip and the piano's damper pedal about its back end, by the same
  depth at the contact.
"""
import bpy
import numpy as np
from mathutils import Matrix, Vector

from thumb_driver import mesh_world

PEDAL_DEPTH = .008             # m the damper pedal goes down where the foot presses
PIANO_DAMPER = 'piano_part_034'   # the grand's right pedal
FLOOR = 'floor'                # the floor object (its top is the floor height; 0 without one)
FOOT_TARGET = '右足ＩＫ'        # the right leg's IK target; the toe target is its child
FOOT_TARGETS = {'L': '左足ＩＫ', 'R': FOOT_TARGET}
ANKLES = {'L': '左足首', 'R': '右足首'}
BALL_ON_PEDAL = .04            # m from the damper pedal's tip back to under the ball of the foot
OLD_STAGING = ('pedal_extender', 'extender_pedal_L', 'extender_pedal_R')   # earlier versions' footrest objects
SIDE = {'L': -1., 'R': 1.}
SEATED_BONES = ('下半身', '左足D', '右足D', '左足', '右足')   # pelvis and thighs: what rests on the bench
HIP_JOINTS = ('左足', '右足')


def fit_bench(skin):
    """Lower the bench top to the lowest seated-body point over it behind the hip joints (the buttocks carry the
    weight; the thighs in front slope down to the knees and press into the front edge); the legs get shorter.
    Returns the drop (m)."""
    seat = bpy.data.objects['bench_seat']
    coords, _ = mesh_world(skin)
    seated = {g.index for g in skin.vertex_groups if g.name in SEATED_BONES}
    bones = skin.find_armature().data.bones
    deform = {g.index for g in skin.vertex_groups if g.name in bones and bones[g.name].use_deform}
    body = np.array([bool(v.groups) and max(((g.weight, g.group) for g in v.groups if g.group in deform), default=(0, -1))[1] in seated
                     for v in skin.data.vertices])
    corners = np.array([list(seat.matrix_world @ Vector(c)) for c in seat.bound_box])
    lo, hi = corners.min(0), corners.max(0)
    arm = skin.find_armature()
    hips = max((arm.matrix_world @ arm.pose.bones[n].head).y for n in HIP_JOINTS)
    over = body & (coords[:, 1] < hips) & (coords[:, 0] > lo[0]) & (coords[:, 0] < hi[0]) & (coords[:, 1] > lo[1]) & (coords[:, 1] < hi[1]) & (coords[:, 2] > lo[2])
    drop = max(0., hi[2] - coords[over, 2].min()) if over.any() else 0.
    if drop:
        world = seat.matrix_world.copy()
        world.translation.z -= drop
        seat.matrix_world = world
        for leg in [o for o in bpy.data.objects if o.name.startswith('bench_leg') and o.type == 'MESH']:
            leg.data = leg.data.copy()
            world = leg.matrix_world
            points = [world @ v.co for v in leg.data.vertices]
            low, high = min(p.z for p in points), max(p.z for p in points)
            for vertex, point in zip(leg.data.vertices, points):
                point.z = low + (point.z - low) * (1 - drop / (high - low))
                vertex.co = world.inverted() @ point
    return drop


def boot_contacts(arm, skin):
    """Per boot: heel tip (lowest point of its back 40 %) and ball (lowest point of its front half), world xyz."""
    coords, _ = mesh_world(skin)
    out = {}
    for h, sx in SIDE.items():
        ankle = arm.matrix_world @ arm.pose.bones[ANKLES[h]].head
        boot = coords[(coords[:, 0] * sx > .02) & (np.abs(coords[:, 0] - ankle.x) < .12) & (coords[:, 2] < ankle.z + .05)
                      & (np.abs(coords[:, 1] - ankle.y) < .3)]
        back, front = boot[:, 1].min(), boot[:, 1].max()
        rear = boot[boot[:, 1] < back + .4 * (front - back)]
        fore = boot[boot[:, 1] > back + .5 * (front - back)]
        out[h] = dict(heel=rear[np.argmin(rear[:, 2])], ball=fore[np.argmin(fore[:, 2])], lo=boot.min(0), hi=boot.max(0))
    return out


def _bounds(obj):
    corners = np.array([list(obj.matrix_world @ Vector(c)) for c in obj.bound_box])
    return corners.min(0), corners.max(0)


def _place_foot(arm, skin, h, target, steps=8):
    """Move and pitch the foot's IK target until the boot's heel and ball are at ``target(heel, ball)`` -> (heel, ball)
    targets (world). Returns the remaining miss (m)."""
    bone = arm.pose.bones[FOOT_TARGETS[h]]
    bone.rotation_mode = 'QUATERNION'
    miss = np.inf
    for _ in range(steps):
        c = boot_contacts(arm, skin)[h]
        heel, ball = c['heel'], c['ball']
        heel_to, ball_to = target(heel, ball)
        miss = max(np.linalg.norm(heel_to - heel), np.linalg.norm(ball_to - ball))
        if miss < 5e-4:
            break
        pitch = (np.arctan2(ball_to[2] - heel_to[2], ball_to[1] - heel_to[1]) - np.arctan2(ball[2] - heel[2], ball[1] - heel[1]))
        world = Matrix.Translation(Vector(heel_to)) @ Matrix.Rotation(pitch, 4, 'X') @ Matrix.Translation(-Vector(heel))
        bone.matrix = arm.matrix_world.inverted() @ world @ arm.matrix_world @ bone.matrix
        bpy.context.view_layer.update()
    return float(miss)


def place_feet(arm, skin):
    """Left boot flat on the floor, right boot's heel on the floor and ball on the damper pedal (by the leg IK targets).
    Removes earlier versions' footrest. Returns the contact geometry and the misses."""
    for name in OLD_STAGING:
        if bpy.data.objects.get(name) is not None:
            bpy.data.objects.remove(bpy.data.objects[name])
    floor = _bounds(bpy.data.objects[FLOOR])[1][2] if bpy.data.objects.get(FLOOR) else 0.
    lo, hi = _bounds(bpy.data.objects[PIANO_DAMPER])
    on_pedal = np.array([(lo[0] + hi[0]) / 2, lo[1] + BALL_ON_PEDAL, hi[2]])

    def right(heel, ball):
        length = np.linalg.norm(ball - heel)
        heel_to = on_pedal - [ball[0] - heel[0], np.sqrt(max(length ** 2 - (on_pedal[2] - floor) ** 2, 0.)), on_pedal[2] - floor]
        return heel_to, on_pedal
    misses = dict(R=_place_foot(arm, skin, 'R', right))
    right_heel = boot_contacts(arm, skin)['R']['heel']

    def left(heel, ball):
        heel_to = np.array([heel[0], right_heel[1], floor])
        return heel_to, heel_to + [ball[0] - heel[0], np.hypot(ball[1] - heel[1], ball[2] - heel[2]), 0.]
    misses['L'] = _place_foot(arm, skin, 'L', left)
    contacts = boot_contacts(arm, skin)
    contacts['floor'], contacts['miss_mm'] = floor, {h: m * 1000 for h, m in misses.items()}
    return contacts


def _pivot_origin(obj, pivot):
    """Move ``obj``'s origin to world ``pivot`` without moving its mesh (own mesh copy)."""
    obj.data = obj.data.copy()
    local = obj.matrix_world.inverted() @ Vector(pivot)
    obj.data.transform(Matrix.Translation(-local))
    obj.matrix_world = obj.matrix_world @ Matrix.Translation(local)


def damper_pose(arm, skin, contacts):
    """Rest and fully pressed (location, quaternion) of the right foot's IK target, turned about the heel tip.

    The leg IK does not turn the boot by exactly the target's angle, so the angle is solved on the posed skin:
    the ball of the boot goes down PEDAL_DEPTH, with the pressed damper pedal."""
    heel, ball = contacts['R']['heel'], contacts['R']['ball']
    bone = arm.pose.bones[FOOT_TARGET]
    bone.rotation_mode = 'QUATERNION'
    rest = (bone.location.copy(), bone.rotation_quaternion.copy())
    rest_matrix = bone.matrix.copy()
    coords, _ = mesh_world(skin)
    near = np.flatnonzero(np.linalg.norm(coords[:, :2] - ball[:2], axis=1) < .015)

    def drop(angle):
        turn = Matrix.Translation(Vector(heel)) @ Matrix.Rotation(-angle, 4, 'X') @ Matrix.Translation(-Vector(heel))
        bone.matrix = arm.matrix_world.inverted() @ turn @ arm.matrix_world @ rest_matrix
        bpy.context.view_layer.update()
        return ball[2] - mesh_world(skin)[0][near, 2].min()

    angle = PEDAL_DEPTH / (ball[1] - heel[1])
    for _ in range(4):                     # the drop is nearly proportional to the angle: rescale
        got = drop(angle)
        angle *= PEDAL_DEPTH / max(got, 1e-5)
    reached = drop(angle)
    pressed = (bone.location.copy(), bone.rotation_quaternion.copy())
    bone.location, bone.rotation_quaternion = rest
    bpy.context.view_layer.update()
    return rest, pressed, angle, reached


def animate_damper(arm, skin, contacts, frames, depth, intervals, fps, write_curve):
    """Foot and piano pedal follow ``depth``; the piano pedal also carries the exact pedal state
    as the custom property ``damper`` (0/1, constant between source events) for a sound engine."""
    depth = np.asarray(depth, float)[:, None]
    (rest_loc, rest_q), (pressed_loc, pressed_q), angle, reached = damper_pose(arm, skin, contacts)
    bone = arm.pose.bones[FOOT_TARGET]
    write_curve(bone, 'location', frames, np.array(rest_loc) + depth * (np.array(pressed_loc) - np.array(rest_loc)))
    q0, q1 = np.array(rest_q), np.array(pressed_q)
    q1 = q1 if q0 @ q1 >= 0 else -q1
    q = q0 + depth * (q1 - q0)
    write_curve(bone, 'rotation_quaternion', frames, q / np.linalg.norm(q, axis=1, keepdims=True))
    piano = bpy.data.objects[PIANO_DAMPER]
    corners = np.array([list(piano.matrix_world @ Vector(c)) for c in piano.bound_box])
    lo, hi = corners.min(0), corners.max(0)
    _pivot_origin(piano, ((lo[0] + hi[0]) / 2, hi[1], (lo[2] + hi[2]) / 2))
    piano.rotation_mode = 'XYZ'
    rest_x = piano.rotation_euler.x
    # its tip points at the player (-y), so the same press is the opposite turn
    write_curve(piano, 'rotation_euler', frames, rest_x + depth[:, 0] * PEDAL_DEPTH / (hi[1] - lo[1]), index=0)
    piano['damper'] = 0.
    piano.keyframe_insert(data_path='["damper"]', frame=1)
    for begin, end in intervals:
        piano['damper'] = 1.
        piano.keyframe_insert(data_path='["damper"]', frame=1 + begin * fps)
        if np.isfinite(end):
            piano['damper'] = 0.
            piano.keyframe_insert(data_path='["damper"]', frame=1 + end * fps)
    from thumb_driver import animation_curves
    for curve in animation_curves(piano):
        if curve.data_path == '["damper"]':
            for point in curve.keyframe_points:
                point.interpolation = 'CONSTANT'
    return dict(press_angle_deg=float(np.degrees(angle)), ball_drop_mm=float(reached * 1000))
