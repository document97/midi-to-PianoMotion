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
import os

import bpy
import numpy as np
from mathutils import Matrix, Vector

from thumb_driver import mesh_world

PEDAL_DEPTH = .008             # m the damper pedal goes down where the foot presses
# Scene object names are configuration, not constants: the instrument and room may come from another file.
PIANO_DAMPER = os.environ.get('PIANOMOTION_DAMPER') or 'piano_part_034'   # the grand's right pedal
FLOOR = os.environ.get('PIANOMOTION_FLOOR') or 'floor'                     # top of the floor; 0 without one
BENCH_SEAT = os.environ.get('PIANOMOTION_BENCH') or 'bench_seat'
# The seating script drops the leg IK constraints, so the IK target bones drive nothing. The legs are placed here
# by an analytic two-bone solve that writes FK rotations straight onto the thigh/shin/foot bones; the mesh is
# skinned to the matching 足D/ひざD/足首D bones, which follow those rotations through their own constraint chain.
LEGS = {'L': ('左足', '左ひざ', '左足首'), 'R': ('右足', '右ひざ', '右足首')}
FOOT_TARGET = '右足首'          # the bone the pedal press turns (about the heel)
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
    if BENCH_SEAT not in bpy.data.objects:
        return 0.
    seat = bpy.data.objects[BENCH_SEAT]
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
            if high - low <= drop + 1e-4:
                continue
            for vertex, point in zip(leg.data.vertices, points):
                point.z = low + (point.z - low) * (1 - drop / (high - low))
                vertex.co = world.inverted() @ point
    return drop


def boot_contacts(arm, skin):
    """Per boot: heel tip (lowest point of its back 40 %) and ball (lowest point of its front half), world xyz."""
    coords, _ = mesh_world(skin)
    mid_x = 0.5 * (float(_world_head(arm, HIP_JOINTS[0]).x) + float(_world_head(arm, HIP_JOINTS[1]).x))
    out = {}
    for h, sx in SIDE.items():
        ankle = arm.matrix_world @ arm.pose.bones[ANKLES[h]].head
        boot = coords[((coords[:, 0] - mid_x) * sx > .02) & (np.abs(coords[:, 0] - ankle.x) < .12) & (coords[:, 2] < ankle.z + .05)
                      & (np.abs(coords[:, 1] - ankle.y) < .3)]
        back, front = boot[:, 1].min(), boot[:, 1].max()
        rear = boot[boot[:, 1] < back + .4 * (front - back)]
        fore = boot[boot[:, 1] > back + .5 * (front - back)]
        out[h] = dict(heel=rear[np.argmin(rear[:, 2])], ball=fore[np.argmin(fore[:, 2])], lo=boot.min(0), hi=boot.max(0))
    return out


def _bounds(obj):
    corners = np.array([list(obj.matrix_world @ Vector(c)) for c in obj.bound_box])
    return corners.min(0), corners.max(0)


def _world_head(arm, name):
    return arm.matrix_world @ arm.pose.bones[name].head


def _aim(arm, pb, head, direction):
    """Turn a pose bone about ``head`` so its Y axis (the bone) points along ``direction``; world space."""
    m = arm.matrix_world @ pb.matrix
    cur = (m.to_3x3() @ Vector((0., 1., 0.))).normalized()
    q = cur.rotation_difference(direction.normalized())
    world = Matrix.Translation(head) @ q.to_matrix().to_4x4() @ Matrix.Translation(-head) @ m
    pb.matrix = arm.matrix_world.inverted() @ world
    bpy.context.view_layer.update()


def _solve_leg(arm, h, ankle, pole):
    """Two-bone IK from the hip to ``ankle`` with the knee pushed towards ``pole``; writes FK rotations."""
    thigh_name, shin_name, foot_name = LEGS[h]
    thigh, shin = arm.pose.bones[thigh_name], arm.pose.bones[shin_name]
    H = _world_head(arm, thigh_name)
    L1 = (_world_head(arm, shin_name) - H).length
    L2 = (_world_head(arm, foot_name) - _world_head(arm, shin_name)).length
    v = Vector(ankle) - H
    d = min(max(v.length, abs(L1 - L2) + 1e-4), L1 + L2 - 1e-4)
    axis = v.normalized()
    p = Vector(pole)
    p = p - axis * p.dot(axis)
    if p.length < 1e-6:
        p = Vector((0., 1., 0.))
    p.normalize()
    ca = (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d)
    ang = float(np.arccos(min(max(ca, -1.), 1.)))
    K = H + axis * (L1 * float(np.cos(ang))) + p * (L1 * float(np.sin(ang)))
    _aim(arm, thigh, H, K - H)
    K = _world_head(arm, shin_name)
    _aim(arm, shin, K, Vector(ankle) - K)


def _pitch_foot(arm, skin, h, target_pitch_deg, iters=5):
    """Turn the boot about its ankle until heel->ball reaches ``target_pitch_deg`` (degrees, + = toes up)."""
    foot = arm.pose.bones[LEGS[h][2]]
    for _ in range(iters):
        c = boot_contacts(arm, skin)[h]
        heel, ball = c['heel'], c['ball']
        cur = np.degrees(np.arctan2(ball[2] - heel[2], ball[1] - heel[1]))
        err = target_pitch_deg - float(cur)
        if abs(err) < .3:
            break
        head = _world_head(arm, LEGS[h][2])
        m = arm.matrix_world @ foot.matrix
        world = Matrix.Translation(head) @ Matrix.Rotation(np.radians(err), 4, 'X') @ Matrix.Translation(-head) @ m
        foot.matrix = arm.matrix_world.inverted() @ world
        bpy.context.view_layer.update()
    return boot_contacts(arm, skin)[h]


def place_feet(arm, skin):
    """Left boot flat on the floor, right boot's heel on the floor and its ball on the damper pedal.

    The seating script removes the leg IK constraints, so this drives the bones directly: a two-bone solve puts
    the ankle where the foot has to be, then the boot is pitched until heel and ball sit at the right heights.
    Removes earlier versions' footrest. Returns the contact geometry and the residual miss in mm.
    """
    for name in OLD_STAGING:
        if bpy.data.objects.get(name) is not None:
            bpy.data.objects.remove(bpy.data.objects[name])
    floor = _bounds(bpy.data.objects[FLOOR])[1][2] if bpy.data.objects.get(FLOOR) else 0.
    lo, hi = _bounds(bpy.data.objects[PIANO_DAMPER])
    pedal_top = float(hi[2])
    pedal_x = float((lo[0] + hi[0]) / 2)
    pedal_y = float(lo[1] + BALL_ON_PEDAL)
    hip_y = float(_world_head(arm, '右足').y)
    left_hip_x = float(_world_head(arm, '左足').x)

    # ankle targets: x from the hip width, y from the hip (shin near vertical), z solved against the floor.
    # The ball sits ~4 cm outboard and ~9.5 cm in front of the ankle, so aim the ankle back from the pedal.
    spec = {
        'L': dict(x=left_hip_x - .015, y=hip_y + .424, pitch=0.),
        'R': dict(x=pedal_x - .042, y=pedal_y - .095, pitch=float(np.degrees(np.arctan2(pedal_top - floor, .095)))),
    }
    misses = {}
    target_ball_z = {'L': float(floor), 'R': pedal_top}
    for h in ('L', 'R'):
        sx = SIDE[h]
        t = np.array([spec[h]['x'], spec[h]['y'], floor + .14], float)
        pitch = spec[h]['pitch']
        for _ in range(6):
            _solve_leg(arm, h, t, np.array([sx * .12, 1., .12]))
            c = _pitch_foot(arm, skin, h, pitch)
            dz = float(c['heel'][2]) - floor
            t[2] -= dz                                                # heel onto the floor
            foot_len = max(float(np.linalg.norm(c['ball'] - c['heel'])), 1e-3)
            # Account for the dz shift already applied to t[2] so heel error isn't double-counted into pitch.
            err = target_ball_z[h] - (float(c['ball'][2]) - dz)
            pitch += float(np.degrees(np.arcsin(max(-.9, min(.9, err / foot_len)))))
            if h == 'R':                                              # ball over the pedal, not just at its height
                t[1] += pedal_y - float(c['ball'][1])
                t[0] += pedal_x - float(c['ball'][0])
        _solve_leg(arm, h, t, np.array([sx * .12, 1., .12]))
        c = _pitch_foot(arm, skin, h, pitch)
        heel, ball = c['heel'], c['ball']
        ball_target = np.array([pedal_x, pedal_y, target_ball_z[h]]) if h == 'R' else np.array([ball[0], ball[1], target_ball_z[h]])
        misses[h] = float(max(np.linalg.norm(heel - np.array([heel[0], heel[1], floor])),
                              np.linalg.norm(ball - ball_target)))
    contacts = boot_contacts(arm, skin)
    contacts['floor'], contacts['miss_mm'] = floor, {h: m * 1000 for h, m in misses.items()}
    contacts['pedal_top'] = pedal_top
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
    near = np.flatnonzero((np.linalg.norm(coords[:, :2] - ball[:2], axis=1) < .015)
                          & (np.abs(coords[:, 2] - ball[2]) < .03))
    if not len(near):
        near = np.flatnonzero(np.linalg.norm(coords[:, :2] - ball[:2], axis=1) < .015)

    def drop(angle):
        turn = Matrix.Translation(Vector(heel)) @ Matrix.Rotation(-angle, 4, 'X') @ Matrix.Translation(-Vector(heel))
        bone.matrix = arm.matrix_world.inverted() @ turn @ arm.matrix_world @ rest_matrix
        bpy.context.view_layer.update()
        return ball[2] - mesh_world(skin)[0][near, 2].min()

    angle = PEDAL_DEPTH / max(ball[1] - heel[1], 1e-3)
    for _ in range(4):                     # the drop is nearly proportional to the angle: rescale
        got = drop(angle)
        if got < 1e-5:                     # the foot did not move: rescaling would blow the angle up
            print('damper: foot does not follow, press skipped', flush=True)
            angle = 0.
            break
        angle = min(angle * PEDAL_DEPTH / got, float(np.radians(20.)))
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
    # its tip points at the player (-y), so the same press is the opposite turn; divide by the real contact lever arm
    lever = max(float(hi[1] - (lo[1] + BALL_ON_PEDAL)), 0.02)
    write_curve(piano, 'rotation_euler', frames, rest_x + depth[:, 0] * PEDAL_DEPTH / lever, index=0)
    from thumb_driver import animation_curves
    if piano.animation_data and piano.animation_data.action:
        for curve in list(animation_curves(piano)):
            if curve.data_path == '["damper"]':
                curve.keyframe_points.clear()
    piano['damper'] = 0.
    piano.keyframe_insert(data_path='["damper"]', frame=1)
    for begin, end in intervals:
        piano['damper'] = 1.
        piano.keyframe_insert(data_path='["damper"]', frame=1 + begin * fps)
        if np.isfinite(end):
            piano['damper'] = 0.
            piano.keyframe_insert(data_path='["damper"]', frame=1 + end * fps)
    for curve in animation_curves(piano):
        if curve.data_path == '["damper"]':
            for point in curve.keyframe_points:
                point.interpolation = 'CONSTANT'
    return dict(press_angle_deg=float(np.degrees(angle)), ball_drop_mm=float(reached * 1000))
