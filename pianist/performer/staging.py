"""Below the hands (run inside Blender): the bench under the character, a pedal extender under the feet, the right
foot on the damper pedal.

A short character's shins do not reach the floor when seated at the keyboard (the development model's knees are
level with the bench), so, like a short player, it uses a pedal extender: a box over the piano's pedals with its
own pedals on top, linked to the piano's. Everything is fitted to the posed skin, not to fixed numbers:
- the bench top is lowered to the lowest point of the seated body (hips, thighs; clothes without cloth
  simulation may still dip into the seat);
- the box top carries the heels, each extender pedal lies under the ball of a foot;
- pressing turns the right boot about its heel tip, the extender pedal about its hinge and the piano's damper
  pedal about its back end, all by the same depth at the contact.
"""
import bpy
import numpy as np
from mathutils import Matrix, Vector

from thumb_driver import mesh_world

PEDAL_DEPTH = .008             # m the damper pedal goes down where the foot presses
PIANO_DAMPER = 'piano_part_034'   # the grand's right pedal (under the box)
BOX = 'pedal_extender'         # the extender body (apply_seating's footrest, refitted here)
LEVERS = {'L': 'extender_pedal_L', 'R': 'extender_pedal_R'}
FOOT_TARGET = '右足ＩＫ'        # the right leg's IK target; the toe target is its child
BOX_MARGIN = (.03, .04, .03)   # m around the boots: sides, behind the heels, in front of the toes
LEVER_WIDTH, LEVER_BACK, LEVER_FRONT, LEVER_THICK = .06, .08, .03, .005
SIDE = {'L': -1., 'R': 1.}
SEATED_BONES = ('下半身', '左足D', '右足D', '左足', '右足')   # pelvis and thighs: what rests on the bench


def _box_mesh(name, lo, hi):
    """An axis-aligned box mesh from lo to hi (object space)."""
    mesh = bpy.data.meshes.new(name)
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh.from_pydata(verts, [], faces)
    return mesh


def _material(name, colour):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = colour
    return material


def fit_bench(skin):
    """Lower the bench top to the lowest seated-body point over it; the legs get shorter. Returns the drop (m)."""
    seat = bpy.data.objects['bench_seat']
    coords, _ = mesh_world(skin)
    seated = {g.index for g in skin.vertex_groups if g.name in SEATED_BONES}
    bones = skin.find_armature().data.bones
    deform = {g.index for g in skin.vertex_groups if g.name in bones and bones[g.name].use_deform}
    body = np.array([bool(v.groups) and max(((g.weight, g.group) for g in v.groups if g.group in deform), default=(0, -1))[1] in seated
                     for v in skin.data.vertices])
    corners = np.array([list(seat.matrix_world @ Vector(c)) for c in seat.bound_box])
    lo, hi = corners.min(0), corners.max(0)
    over = body & (coords[:, 0] > lo[0]) & (coords[:, 0] < hi[0]) & (coords[:, 1] > lo[1]) & (coords[:, 1] < hi[1]) & (coords[:, 2] > lo[2])
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


def boot_contacts(skin):
    """Per boot: heel tip (lowest point of its back 40 %) and ball (lowest point of its front half), world xyz."""
    coords, _ = mesh_world(skin)
    feet = bpy.data.objects[BOX].matrix_world.translation
    out = {}
    for h, sx in SIDE.items():
        boot = coords[(coords[:, 0] * sx > .02) & (coords[:, 2] < feet.z + .25) & (np.abs(coords[:, 1] - feet.y) < .3)]
        back, front = boot[:, 1].min(), boot[:, 1].max()
        rear = boot[boot[:, 1] < back + .4 * (front - back)]
        fore = boot[boot[:, 1] > back + .5 * (front - back)]
        out[h] = dict(heel=rear[np.argmin(rear[:, 2])], ball=fore[np.argmin(fore[:, 2])], lo=boot.min(0), hi=boot.max(0))
    return out


def pedal_extender(skin):
    """Refit the box under the heels and put an extender pedal under each ball. Returns the contact geometry."""
    contacts = boot_contacts(skin)
    lo = np.min([c['lo'] for c in contacts.values()], axis=0)
    hi = np.max([c['hi'] for c in contacts.values()], axis=0)
    top = min(c['heel'][2] for c in contacts.values())
    floor = 0.
    box = bpy.data.objects[BOX]
    old = box.data
    box.data = _box_mesh(BOX, (lo[0] - BOX_MARGIN[0], lo[1] - BOX_MARGIN[1], floor), (hi[0] + BOX_MARGIN[0], hi[1] + BOX_MARGIN[2], top))
    box.matrix_world = Matrix.Identity(4)
    for material in old.materials:
        box.data.materials.append(material)
    brass = _material('extender_pedal_brass', (.55, .42, .18, 1.))
    for h, c in contacts.items():
        ball = c['ball']
        hinge = Vector((ball[0], ball[1] - LEVER_BACK, ball[2]))
        lever = bpy.data.objects.get(LEVERS[h])
        if lever is None:
            lever = bpy.data.objects.new(LEVERS[h], None)
            bpy.context.scene.collection.objects.link(lever)
        mesh = _box_mesh(LEVERS[h], (-LEVER_WIDTH / 2, 0., -LEVER_THICK), (LEVER_WIDTH / 2, LEVER_BACK + LEVER_FRONT, 0.))
        block = _box_mesh('hinge', (-LEVER_WIDTH / 2, 0., top - ball[2]), (LEVER_WIDTH / 2, .015, -LEVER_THICK))
        verts = [v.co.copy() for v in mesh.vertices] + [v.co.copy() for v in block.vertices]
        faces = [tuple(p.vertices) for p in mesh.polygons] + [tuple(i + 8 for i in p.vertices) for p in block.polygons]
        mesh.clear_geometry()
        mesh.from_pydata(verts, [], faces)
        bpy.data.meshes.remove(block)
        if lever.type != 'MESH':
            bpy.data.objects.remove(lever)
            lever = bpy.data.objects.new(LEVERS[h], mesh)
            bpy.context.scene.collection.objects.link(lever)
        else:
            lever.data = mesh
        lever.data.materials.append(brass)
        lever.matrix_world = Matrix.Translation(hinge)
        c['hinge'] = np.array(hinge)
    contacts['top'] = top
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
    the ball of the boot goes down PEDAL_DEPTH, onto the pressed extender pedal."""
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
    """Foot, extender pedal and piano pedal follow ``depth``; the piano pedal also carries the exact pedal state
    as the custom property ``damper`` (0/1, constant between source events) for a sound engine."""
    depth = np.asarray(depth, float)[:, None]
    (rest_loc, rest_q), (pressed_loc, pressed_q), angle, reached = damper_pose(arm, skin, contacts)
    bone = arm.pose.bones[FOOT_TARGET]
    write_curve(bone, 'location', frames, np.array(rest_loc) + depth * (np.array(pressed_loc) - np.array(rest_loc)))
    q0, q1 = np.array(rest_q), np.array(pressed_q)
    q1 = q1 if q0 @ q1 >= 0 else -q1
    q = q0 + depth * (q1 - q0)
    write_curve(bone, 'rotation_quaternion', frames, q / np.linalg.norm(q, axis=1, keepdims=True))
    lever = bpy.data.objects[LEVERS['R']]
    lever.rotation_mode = 'XYZ'
    write_curve(lever, 'rotation_euler', frames, -depth[:, 0] * PEDAL_DEPTH / LEVER_BACK, index=0)
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
