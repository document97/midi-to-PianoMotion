"""Export the rig, keyboard and validation poses for the numpy performer (run inside Blender).

blender -b <scene.blend> --python performer/rig_export.py -- --output performer/rig_cache/rig.json

The scene is put in its neutral state: the performer's seat and size, neutral fingers, analytic arms at the
calibrated neutral wrists. Everything a pose needs is exported in the wrist frame, so the numpy model never
needs Blender. Random validation poses are applied through the Blender-side drivers (lib/arm_driver.ArmDriver,
lib/thumb_driver.ThumbFrame) and the evaluated skin is recorded, so the numpy kinematics can be checked
against the real deformed mesh.
"""
import argparse
import hashlib
import json
import os
import math
from pathlib import Path
import random
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lib'))

from performer.kinematics import lap_from_surface, STANDARD_LAP  # noqa: E402  the fitted lap correction
from arm_driver import ArmDriver                       # noqa: E402
from body_driver import BodyDriver, apply_seating      # noqa: E402
from key_geometry import measured_keyboard             # noqa: E402
from probe_driver import remove_curves                 # noqa: E402
from thumb_articulation import child_euler             # noqa: E402
from thumb_driver import ThumbFrame, mesh_world, set_time   # noqa: E402

ARM_CHANNELS = ('左肩', '右肩', '左腕', '右腕', '左ひじ', '右ひじ', '左手', '右手', '左親指', '右親指',
                '左人指', '右人指', '左中指', '右中指', '左薬指', '右薬指', '左小指', '右小指', '上半身')
SIDE = {'L': '左', 'R': '右'}


def mat(m):
    return [list(map(float, row)) for row in m]


def vec(v):
    return [float(x) for x in v]


# The performer's seat: world offset of the character and bench from where the scene has them. Tuned for the
# development character: (0, .06, .17) put the thighs 13.5 mm into the keybed with the elbows bent tight; 9 cm
# further back and 2.5 cm lower leaves 16 mm under the keybed, a more open elbow and a flatter forearm.
DEFAULT_SEATING = (0., -.05, .145)       # fixed: performer and bench 2 cm further back than the original -.03
PERFORMER_SEATING = DEFAULT_SEATING
if os.environ.get('PIANOMOTION_SEAT'):  # another seat (x, y, z m) for a scene of its own
    PERFORMER_SEATING = tuple(float(v) for v in os.environ['PIANOMOTION_SEAT'].split(','))
# Uniform scale of the character. Set it for your model: the hand's size decides what it can reach. The development
# model (barefoot 1.52 m) missed black-key octaves (thumb to little finger) by 3.4 mm; at 1.05 (about 1.60 m) its
# hand holds them. The character is scaled about the middle of the bench top so it stays seated; the feet move with
# the scale (blender_main then refits the bench and puts the feet on the floor and the pedal: performer.staging).
CHARACTER_SCALE = 1.05
# Uniform scale of both hands about the wrist joints, on top of CHARACTER_SCALE: the model's hands are small for its
# height (wrist to middle fingertip 15.0 cm at 1.60 m, a real hand about 17 cm) and its white-key octaves landed on the
# key edges. Set on the wrist pose bones, so the fingers follow and the skin the wrist shares with the forearm blends
# from 1 to this over the wrist.
HAND_SCALE = 1.12


def scaled_seat(arm, cal, offset):
    """Scale the armature about the seated bench top; returns the calibration without a footrest (the feet stand on
    the floor and the piano's own pedal: performer.staging)."""
    root = arm.parent or arm
    cal = dict(cal, ergonomics={k: v for k, v in cal.get('ergonomics', {}).items() if k != 'footrest'})
    if 'performer_character_scale' in root:
        if abs(root['performer_character_scale'] - CHARACTER_SCALE) > 1e-9:
            raise ValueError('Scene already has a different character scale')
        return cal
    seat = bpy.data.objects['bench_seat']
    corners = [seat.matrix_world @ Vector(c) for c in seat.bound_box]
    pivot = Vector((sum(c.x for c in corners) / 8, sum(c.y for c in corners) / 8, max(c.z for c in corners)))
    s = CHARACTER_SCALE
    world = arm.matrix_world.copy()        # before seating: apply_seating then moves the character and the bench together
    arm.matrix_world = Matrix.Translation(pivot) @ Matrix.Scale(s, 4) @ Matrix.Translation(-pivot) @ world
    root['performer_character_scale'] = s
    return cal


def character():
    """The performer: the one armature with MMD-named hand bones, and the largest mesh it deforms."""
    arms = [o for o in bpy.data.objects if o.type == 'ARMATURE' and all(s + '手首' in o.pose.bones for s in SIDE.values())]
    if len(arms) != 1:
        raise RuntimeError(f'need exactly one armature with MMD hand bones (左手首, 右手首), found {[o.name for o in arms]}')
    skins = [o for o in bpy.data.objects if o.type == 'MESH'
             and any(m.type == 'ARMATURE' and m.object is arms[0] for m in o.modifiers)]
    if not skins:
        raise RuntimeError(f'no mesh is deformed by {arms[0].name}')
    return arms[0], max(skins, key=lambda o: len(o.data.vertices))


def scaled_hands(arm):
    """Scale both wrist pose bones by HAND_SCALE (before the arm drivers read the wrist's world scale)."""
    root = arm.parent or arm
    if 'performer_hand_scale' in root:
        if abs(root['performer_hand_scale'] - HAND_SCALE) > 1e-9:
            raise ValueError('Scene already has a different hand scale')
        return
    for s in SIDE.values():
        arm.pose.bones[s + '手首'].scale = (HAND_SCALE,) * 3
    root['performer_hand_scale'] = HAND_SCALE


def spine_chain(arm, torso='上半身', neck='首'):
    """The pose bones from the torso bone up to the neck's parent (lowest first), or [] if the neck is not above it."""
    bones = arm.pose.bones
    if torso not in bones or neck not in bones:
        return []
    chain, b = [], bones[neck].parent
    while b is not None and b.name != torso:
        chain.append(b)
        b = b.parent
    return [] if b is None else [b] + chain[::-1]


def _bone_frames(arm, bone):
    pre = arm.matrix_world @ bone.parent.matrix @ bone.parent.bone.matrix_local.inverted() @ bone.bone.matrix_local
    return dict(bone=bone.name, head=vec(arm.matrix_world @ bone.head), pre=mat(pre), world=mat(arm.matrix_world @ bone.matrix))


def spine_data(arm, torso, neck='首'):
    """The spine (torso bone up to the neck's parent) and the neck: names, heads, parent frames and posed matrices."""
    chain = spine_chain(arm, torso.name, neck)
    if not chain:
        return {}
    return dict(spine=[_bone_frames(arm, b) for b in chain], neck=_bone_frames(arm, arm.pose.bones[neck]))


def slide_bone(arm, torso, foot_target='右足ＩＫ'):
    """The bone that slides the seated body along the bench: the highest ancestor of the torso bone that is not an
    ancestor of the foot IK target (so the feet stay planted), e.g. MMD センター."""
    keep = set()
    b = arm.pose.bones.get(foot_target)
    while b is not None:
        keep.add(b.name)
        b = b.parent
    chain, b = [], torso.parent
    while b is not None and b.name not in keep:
        chain.append(b.name)
        b = b.parent
    return chain[-1] if chain else None


def collarbones(arm, drivers, cal):
    """Per hand the collarbone: the first unconstrained bone above the upper arm (MMD cancel bones such as 肩C carry
    constraints), with its parent frame, posed world matrix and head (world), and the level direction from the
    shoulder joint to the hand's neutral wrist (where the shoulder goes when it reaches)."""
    out = {}
    for h, driver in drivers.items():
        bone = arm.pose.bones[driver.upper.name].parent
        while bone is not None and bone.constraints:
            bone = bone.parent
        if bone is None or bone.parent is None:
            continue
        pre = arm.matrix_world @ bone.parent.matrix @ bone.parent.bone.matrix_local.inverted() @ bone.bone.matrix_local
        ahead = Vector(cal['hands'][h]['neutral_wrist']) - driver.pre_upper_world.translation
        ahead.z = 0.
        out[h] = dict(bone=bone.name, pre=mat(pre), world=mat(arm.matrix_world @ bone.matrix),
                      head=vec(arm.matrix_world @ bone.head), ahead=vec(ahead.normalized()))
    return out


def straight_back(arm, torso='上半身', neck='首'):
    """The torso bone and the spine bones above it up to the neck turned onto one upright line (a seated pianist's
    back is straight, not arched: the model's rest curve and any pose left in the scene are taken out); the neck
    keeps its direction in the armature. The torso's small forward lean (torso.py) turns it all."""
    bones = arm.pose.bones
    if torso not in bones or neck not in bones:
        return
    chain, b = [], bones[neck].parent
    while b is not None and b.name != torso:
        chain.append(b)
        b = b.parent
    if b is None or not chain:
        return
    chain = [b] + chain[::-1]
    remove_curves(arm, lambda c: any('"' + n.name + '"' in c.data_path for n in chain[1:] + [bones[neck]]))
    bpy.context.view_layer.update()
    neck_before = bones[neck].matrix.to_3x3()
    direction = arm.matrix_world.inverted().to_3x3() @ Vector((0., 0., 1.))
    for bone in chain:
        bone.rotation_mode = 'QUATERNION'
        bone.rotation_quaternion = (1., 0., 0., 0.)
        bpy.context.view_layer.update()
        turn = (bone.tail - bone.head).rotation_difference(direction).to_matrix().to_4x4()
        bone.matrix = Matrix.Translation(bone.head) @ turn @ Matrix.Translation(-bone.head) @ bone.matrix
        bpy.context.view_layer.update()
    head = bones[neck].head.copy()
    bones[neck].matrix = Matrix.Translation(head) @ neck_before.to_4x4()
    bpy.context.view_layer.update()


def neutral_scene(cal, seating=None):
    """The bake's neutral state (setup()), without HandModel / mesh clearance objects, on the performer's seat."""
    (arm, skin), scene = character(), bpy.context.scene
    set_time(scene, 0)
    keyboard = measured_keyboard()
    seating = PERFORMER_SEATING if seating is None else seating
    cal = dict(cal, ergonomics=dict(cal.get('ergonomics', {}), world_root_offset=list(seating)))
    cal = scaled_seat(arm, cal, seating)
    scaled_hands(arm)
    apply_seating(arm, cal)
    straight_back(arm)
    remove_curves(arm, lambda c: any('"' + s in c.data_path for s in ARM_CHANNELS))
    for h, s in SIDE.items():
        for c in arm.pose.bones[s + '手首'].constraints:
            c.mute = True
        for row in cal['hands'][h]['fingers'].values():
            root = arm.pose.bones[row['bone_names'][0]]
            root.rotation_mode = 'QUATERNION'
            root.rotation_quaternion = row['neutral_root_quaternion']
            for i, n in enumerate(row['bone_names'][1:]):
                arm.pose.bones[n].rotation_mode = 'XYZ'
                arm.pose.bones[n].rotation_euler = child_euler(row, i, 0)
    drivers = {h: ArmDriver(arm, h, r['neutral_palm_quaternion'], r.get('neutral_palm_forward'))
               for h, r in cal['hands'].items()}
    for h, driver in drivers.items():
        driver.apply(cal['hands'][h]['neutral_wrist'])
    body = BodyDriver(arm, drivers, cal.get('ergonomics', {}))
    bpy.context.view_layer.update()
    return arm, skin, drivers, body, keyboard


def skin_rows(arm, skin, vertex_ids):
    """Per vertex: normalised weights over every deforming bone and the rest point in each bone's rest frame."""
    transform = arm.matrix_world.inverted() @ skin.matrix_world
    names = {g.index: g.name for g in skin.vertex_groups}
    rows = []
    for i in vertex_ids:
        v = skin.data.vertices[int(i)]
        rest = transform @ v.co
        weights = {}
        for g in v.groups:
            name = names[g.group]
            if g.weight > 0 and name in arm.data.bones:
                weights[name] = weights.get(name, 0.) + g.weight
        total = sum(weights.values()) or 1.
        rows.append(dict(id=int(i), bones={n: [w / total] + vec(arm.data.bones[n].matrix_local.inverted() @ rest)
                                           for n, w in weights.items()}))
    return rows


def export_hand(arm, skin, driver, row, coords, normals):
    world = arm.matrix_world
    wrist = arm.pose.bones[driver.wrist.name]
    W = wrist.matrix.copy()                         # armature space, neutral
    hand = dict(
        bones=dict(upper=driver.upper.name, elbow=driver.elbow.name, wrist=driver.wrist.name,
                   twist=driver.forearm_twist.name),
        sign=driver.sign, l1=driver.l1, l2=driver.l2,
        pre_upper_world=mat(driver.pre_upper_world),
        rest_upper_q=vec(driver.rest_upper_q), rest_elbow_q=vec(driver.rest_elbow_q),
        hinge=vec(driver.hinge), hinge_polarity=driver.hinge_polarity,
        rest_upper_basis=mat(driver.rest_upper_basis), rest_elbow_basis=mat(driver.rest_elbow_basis),
        scales={k: vec(v) for k, v in driver.scales.items()},
        upper_elbow_link=mat(driver.upper_elbow_link), elbow_wrist_link=mat(driver.elbow_wrist_link),
        elbow_twist_link=mat(driver.elbow_twist_link), twist_wrist_link=mat(driver.twist_wrist_link),
        palm=vec(driver.palm), palm_forward=vec(driver.palm_forward),
        neutral_wrist=vec(row['neutral_wrist']), wrist_rest_local=mat(arm.data.bones[driver.wrist.name].matrix_local),
        fingers={},
        neutral_wrist_matrix=mat(W))
    for d, r in row['fingers'].items():
        frame = ThumbFrame(arm, skin, r, coords, normals)
        frame.set_root_reference(r['neutral_root_quaternion'])
        bones = [arm.pose.bones[n] for n in r['bone_names']]
        P = (world @ frame.pre).to_quaternion()
        loc0, _, scale0 = frame.root_basis.decompose()
        children = [b.decompose() for b in frame.child_basis]
        clearance = sorted(set(r['terminal_vertices']) | set(r['affected_vertices'][::3]))
        hand['fingers'][d] = dict(
            names=r['bone_names'],
            pre_in_wrist=mat(W.inverted() @ frame.pre),
            root_loc=vec(loc0), root_scale=vec(scale0),
            child_loc=[vec(c[0]) for c in children], child_scale=[vec(c[2]) for c in children],
            links=[mat(m) for m in frame.links],
            neutral_root_q=vec(r['neutral_root_quaternion']),
            # swing frame in the root's parent-free frame (world @ pre): constant under wrist motion
            palm_normal_local=vec(P.inverted() @ frame.palm_normal),
            pad=skin_rows(arm, skin, r['contact_patch']),
            clearance=skin_rows(arm, skin, clearance),
            bounds=r['working_bounds_degrees'])
        del bones
    return hand


def key_meshes():
    rows = {}
    for obj in bpy.data.objects:
        if obj.name.startswith('key_') and obj.type == 'MESH':
            obj.data.calc_loop_triangles()
            points = [obj.matrix_world @ v.co for v in obj.data.vertices]
            rows[obj.name[4:]] = dict(triangles=[[vec(points[i]) for i in t.vertices] for t in obj.data.loop_triangles])
    return rows


def validation_poses(arm, skin, drivers, cal, hands, count, seed):
    """Apply random poses through the Blender-side drivers; record evaluated pad and clearance points."""
    rng = random.Random(seed)
    coords, normals = mesh_world(skin)
    frames = {(h, d): ThumbFrame(arm, skin, r, coords, normals) for h in 'LR'
              for d, r in cal['hands'][h]['fingers'].items()}
    for (h, d), frame in frames.items():
        frame.set_root_reference(cal['hands'][h]['fingers'][d]['neutral_root_quaternion'])
    samples = []
    for _ in range(count):
        sample = dict(hands={})
        for h, driver in drivers.items():
            n = Vector(cal['hands'][h]['neutral_wrist'])
            for _ in range(100):            # only poses the arm can reach from this seat
                wrist = n + Vector((rng.uniform(-.25, .25) * (1 if h == 'R' else -1), rng.uniform(-.04, .05),
                                    rng.uniform(-.03, .04)))
                yaw, roll = rng.uniform(-10, 10), rng.uniform(-10, 10)
                try:
                    pose = driver.local_pose(wrist, yaw, roll)
                    break
                except ValueError:
                    continue
            for name, q in pose.items():
                b = arm.pose.bones[name]
                b.rotation_mode = 'QUATERNION'
                b.rotation_quaternion = q
            fingers = {}
            for d, r in cal['hands'][h]['fingers'].items():
                bounds = r['working_bounds_degrees']
                values = [rng.uniform(*bounds[k]) for k in ('swing_lateral', 'swing_lift', 'mcp_delta', 'ip_delta')]
                local_q, _ = frames[(h, d)].matrices(values)
                root = arm.pose.bones[r['bone_names'][0]]
                root.rotation_mode = 'QUATERNION'
                root.rotation_quaternion = local_q
                for i, name in enumerate(r['bone_names'][1:]):
                    arm.pose.bones[name].rotation_mode = 'XYZ'
                    arm.pose.bones[name].rotation_euler = child_euler(r, i, values[2 + i])
                fingers[d] = values
            sample['hands'][h] = dict(wrist=vec(wrist), yaw=yaw, roll=roll, fingers=fingers,
                                      bone_quaternions={k: vec(v) for k, v in pose.items()})
        bpy.context.view_layer.update()
        evaluated, _ = mesh_world(skin)
        for h in 'LR':
            out = sample['hands'][h]
            out['wrist_head'] = vec(arm.matrix_world @ arm.pose.bones[drivers[h].wrist.name].head)
            out['points'] = {d: {str(row['id']): vec(evaluated[row['id']]) for row in
                                 hands[h]['fingers'][d]['pad'] + hands[h]['fingers'][d]['clearance'][::7]}
                             for d in hands[h]['fingers']}
            out['root_quaternions'] = {d: vec(arm.pose.bones[cal['hands'][h]['fingers'][d]['bone_names'][0]]
                                              .rotation_quaternion) for d in hands[h]['fingers']}
        samples.append(sample)
    return samples


LAP_ALONG = .35                 # the hands rest on the thighs this share of the way from hip to knee ...
LAP_OUT = .18844                # ... and this share of the hip width OUTSIDE the thigh bone (m on the standard
                                # seat: .03 m at .1592 m hip width), on the clothes covering the thigh. A
                                # share, not metres, so a smaller MMD model does not push its hands too far out.

# The resting point is fitted from this model's own thigh surface: lap_measured() ray-casts against the
# mesh on this seat, and lap_from_surface() applies the portable correction (fractions of hip width) that
# reproduces the approved v9 look. Neither the surface nor the final wrist is a world-space constant.


def lap(arm, skin):
    """{hand: (x, y, z)}: the collision-measured top of each thigh on THIS model and seat.

    Stored as `lap_surface` in the rig; `lap_from_surface()` turns it into the resting wrist, so a new MMD
    model keeps the approved look without borrowing another model's coordinates.
    """
    return lap_measured(arm, skin)


def lap_measured(arm, skin):
    """The ray-cast measurement LAP_POINT was taken from: the top of each thigh (the clothes on it), the first
    skin from above not carried by an arm. Kept to re-measure the standard seat if the body or seat ever changes."""
    from mathutils import Vector
    world = arm.matrix_world
    evaluated = skin.evaluated_get(bpy.context.evaluated_depsgraph_get())
    inverse = skin.matrix_world.inverted()
    names = {g.index: g.name for g in skin.vertex_groups}
    carried = {b.name for side in SIDE.values() for b in [arm.data.bones[side + '腕']] + list(arm.data.bones[side + '腕'].children_recursive)}

    def arm_skin(polygon):
        weights = {}
        for v in skin.data.polygons[polygon].vertices:
            for g in skin.data.vertices[v].groups:
                name = names.get(g.group, '')
                if name in arm.data.bones:          # bone groups only (MMD adds others)
                    key = 'arm' if name in carried else 'other'
                    weights[key] = weights.get(key, 0.) + g.weight
        return weights.get('arm', 0.) > weights.get('other', 0.)

    out = {}
    for h, side in SIDE.items():
        bone = arm.pose.bones[side + '足']
        hip, knee = world @ bone.head, world @ bone.tail
        other = world @ arm.pose.bones[SIDE['L' if h == 'R' else 'R'] + '足'].head
        point = hip + (knee - hip) * LAP_ALONG
        width = (hip - other).length
        point.x += LAP_OUT * width * (1 if hip.x > other.x else -1)   # outward: away from the other hip
        origin, down = inverse @ Vector((point.x, point.y, point.z + 1.)), inverse @ Vector((point.x, point.y, point.z))
        direction = (down - origin).normalized()
        while True:
            ok, loc, _, polygon = evaluated.ray_cast(origin, direction, distance=(down - origin).length)
            if not ok:
                raise RuntimeError(f'no thigh surface found above {side}足')
            if not arm_skin(polygon):
                break
            origin = loc + direction * 1e-4
        out[h] = vec(skin.matrix_world @ loc)
    return out


def hand_bones(arm, drivers):
    """Names of the bones the hands' kinematics read (rig.neutral_bones relative to the wrist): each wrist, what it
    carries, and the forearm bones whose skin the hand shares."""
    names = set()
    for driver in drivers.values():
        names |= {driver.wrist.name, driver.elbow.name} | {b.name for b in driver.wrist.children_recursive}
        names |= {b.name for b in driver.twists}
    return names


def seat_free_hands(cal):
    """The hands' geometry exported at DEFAULT_SEATING (fingers, neutral wrist, the bones they read): the scene's own
    seat changes only the arms and body. The armature's world matrix comes along: the solves pass the wrist through
    the armature frame and back, and that round trip too must not depend on the seat. In float32 the same hand exported at another seat differs by ~0.1 um, and
    that is enough to send a pose solve down another valley: the hand's plan would change with the seat."""
    arm, skin, drivers, _, _ = neutral_scene(cal, DEFAULT_SEATING)
    coords, normals = mesh_world(skin)
    hands = {h: export_hand(arm, skin, drivers[h], cal['hands'][h], coords, normals) for h in 'LR'}
    bones = {b.name: mat(b.matrix) for b in arm.pose.bones if b.name in hand_bones(arm, drivers)}
    world = mat(arm.matrix_world)
    bpy.ops.wm.open_mainfile(filepath=bpy.data.filepath)        # back to the scene as saved, for the actual seat
    return hands, bones, world


TRUNK_BONES = ('上半身', '上半身2', '上半身3')       # the skin the trunk outline (performer.elbows) covers
ARM_BONES = {s: dict(upper=(s + '腕',) + tuple(s + '腕捩' + n for n in ('', '1', '2', '3')),
                     fore=(s + 'ひじ',) + tuple(s + '手捩' + n for n in ('', '1', '2', '3')))
             for s in '左右'}


def trunk_data(arm, skin):
    """The trunk outline the elbow-swing clearance reads (performer.elbows.Trunk): the skin whose dominant bone is
    a trunk bone, in that bone's frame, and per arm the median skin radius about its upper arm / forearm segment,
    at the neutral pose. None without the MMD bones the model reads."""
    names = [g.name for g in skin.vertex_groups]
    bones = set(arm.data.bones.keys())
    if not all(('上半身' in bones, *(b in bones for s in '左右' for group in ARM_BONES[s].values() for b in group))):
        return None
    dominant = np.full(len(skin.data.vertices), '', dtype=object)
    for v in skin.data.vertices:
        groups = [g for g in v.groups if names[g.group] in bones and g.weight > 0]
        if groups:
            dominant[v.index] = names[max(groups, key=lambda g: g.weight).group]
    evaluated = skin.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get('co', co)
    evaluated.to_mesh_clear()
    world = np.array(skin.matrix_world)
    co = co.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]
    trunk_bone = arm.pose.bones['上半身']
    trunk_m = np.array(arm.matrix_world @ trunk_bone.matrix)
    local = (co[np.isin(dominant, TRUNK_BONES)] - trunk_m[:3, 3]) @ trunk_m[:3, :3]

    def radius(a, b, mask):
        p, d = co[mask], b - a
        t = np.clip((p - a) @ d / (d @ d), 0, 1)
        return float(np.median(np.linalg.norm(p - (a + t[:, None] * d), axis=1)))

    heads = {s + n: np.array(arm.matrix_world @ arm.pose.bones[s + n].head) for s in '左右' for n in ('腕', 'ひじ', '手首')}
    radii = {h: [radius(heads[s + '腕'], heads[s + 'ひじ'], np.isin(dominant, ARM_BONES[s]['upper'])),
                 radius(heads[s + 'ひじ'], heads[s + '手首'], np.isin(dominant, ARM_BONES[s]['fore']))]
             for s, h in (('左', 'L'), ('右', 'R'))}
    return dict(schema='performer-trunk', bone='上半身', matrix=[list(map(float, row)) for row in trunk_m],
                local=[[float(x) for x in p] for p in local], radius=radii)


def write_trunk(out, trunk):
    """Write trunk.json beside the rig it was traced from, stamped with that rig.json's sha256 (the way rig.json
    stamps its scene and its calibration). performer.elbows.load rejects a pair whose stamps disagree, which is the
    only way a stale outline -- a re-exported rig, another seat's outline -- is caught before it clears a body that
    is not the one on screen."""
    if not out.exists():
        raise SystemExit(f'trunk.json needs the rig it describes: {out} is missing (export the rig first)')
    trunk = dict(trunk, rig_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    path = out.parent / 'trunk.json'
    path.write_text(json.dumps(trunk), encoding='utf-8')
    print('TRUNK_EXPORTED', path, len(trunk['local']), 'verts', 'rig', trunk['rig_sha256'][:12], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--trunk-only', action='store_true',
                        help='write trunk.json beside --output only (an existing rig.json is not touched: the '
                             'pair-table cache keys on it)')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    cal_path = ROOT / 'hand_calibration.json'
    cal = json.loads(cal_path.read_text(encoding='utf-8'))
    seat_free = seat_free_hands(cal) if tuple(PERFORMER_SEATING) != tuple(DEFAULT_SEATING) else None
    arm, skin, drivers, body, keyboard = neutral_scene(cal)
    coords, normals = mesh_world(skin)
    hands = {h: export_hand(arm, skin, drivers[h], cal['hands'][h], coords, normals) for h in 'LR'}
    surface = lap(arm, skin)
    hips = [vec(arm.matrix_world @ arm.pose.bones[s + '足'].head) for s in ('左', '右')]
    for h in 'LR':
        point = lap_from_surface(surface[h], h, hips, hands[h]['neutral_wrist'])
        hands[h]['lap_surface'], hands[h]['lap'] = surface[h], point
        print('[lap] %s surface %s -> wrist %s (reference %s)' % (
            h, [round(v * 1000., 1) for v in surface[h]], [round(v * 1000., 1) for v in point],
            [round(v * 1000., 1) for v in STANDARD_LAP[h]]), flush=True)
    out = Path(args.output)
    trunk = trunk_data(arm, skin)
    if args.trunk_only:
        if trunk is not None:
            write_trunk(out, trunk)
        return
    torso = None
    if body.bone is not None:
        torso = dict(bone=body.bone.name, pivot=vec(body.pivot), pre=mat(body.pre), world=mat(body.world),
                     reference={h: mat(m) for h, m in body.reference.items()}, shoulders=collarbones(arm, drivers, cal),
                     hips=[vec(arm.matrix_world @ arm.pose.bones[s + '足'].head) for s in ('左', '右')],
                     **spine_data(arm, body.bone),
                     slide_bone=slide_bone(arm, body.bone))
    neutral_bones = {b.name: mat(b.matrix) for b in arm.pose.bones}
    if seat_free is not None:
        free_hands, free_bones, free_world = seat_free
        neutral_bones.update(free_bones)
        for h in 'LR':
            # the arm's lengths too: the plan's natural orientation (posture) reads them, and float32 noise of 1e-8 m
            # from another seat is enough to change its solves
            for key in ('fingers', 'neutral_wrist', 'neutral_wrist_matrix', 'l1', 'l2'):
                hands[h][key] = free_hands[h][key]
    rig = dict(schema='performer-rig', neutral_bones=neutral_bones, scene=bpy.data.filepath,
               scene_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
               calibration_sha256=hashlib.sha256(cal_path.read_bytes()).hexdigest(),
               fps=bpy.context.scene.render.fps / bpy.context.scene.render.fps_base,
               arm_world=mat(arm.matrix_world) if seat_free is None else free_world, hands=hands, torso=torso, keyboard=keyboard,
               key_meshes=key_meshes(),
               key_depth_gain={p: k['local_z_world_gain'] for p, k in keyboard.items()})
    rig['validation'] = validation_poses(arm, skin, drivers, cal, hands, args.samples, 7)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rig), encoding='utf-8')
    print('RIG_EXPORTED', out, 'validation', len(rig['validation']), flush=True)
    if trunk is not None:                 # after the rig: the outline is stamped with the bytes written above
        write_trunk(out, trunk)


if __name__ == '__main__':
    main()
