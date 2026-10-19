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
import math
from pathlib import Path
import random
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lib'))

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
PERFORMER_SEATING = (0., -.03, .145)
# Uniform scale of the character. Set it for your model: the hand's size decides what it can reach. The development
# model (barefoot 1.52 m) missed black-key octaves (thumb to little finger) by 3.4 mm; at 1.05 (about 1.60 m) its
# hand holds them. The character is scaled about the middle of the bench top so it stays seated; the feet move with
# the scale, and so does the footrest (blender_main then refits bench and footrest: performer.staging).
CHARACTER_SCALE = 1.05


def scaled_seat(arm, cal, offset):
    """Scale the armature about the seated bench top; returns the calibration with the footrest moved to match."""
    root = arm.parent or arm
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
    pivot += Vector(offset)                # the footrest is placed in the seated world
    support = cal.get('ergonomics', {}).get('footrest')
    if support:
        x0, y0, x1, y1 = support['bounds_xy']
        support = dict(support, bounds_xy=[pivot.x + (x0 - pivot.x) * s, pivot.y + (y0 - pivot.y) * s,
                                           pivot.x + (x1 - pivot.x) * s, pivot.y + (y1 - pivot.y) * s],
                       top_z=pivot.z - (pivot.z - support['top_z']) * s)
        cal = dict(cal, ergonomics=dict(cal['ergonomics'], footrest=support))
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


def neutral_scene(cal):
    """The bake's neutral state (setup()), without HandModel / mesh clearance objects, on the performer's seat."""
    (arm, skin), scene = character(), bpy.context.scene
    set_time(scene, 0)
    keyboard = measured_keyboard()
    cal = dict(cal, ergonomics=dict(cal.get('ergonomics', {}), world_root_offset=list(PERFORMER_SEATING)))
    cal = scaled_seat(arm, cal, PERFORMER_SEATING)
    apply_seating(arm, cal)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=12)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    cal_path = ROOT / 'hand_calibration.json'
    cal = json.loads(cal_path.read_text(encoding='utf-8'))
    arm, skin, drivers, body, keyboard = neutral_scene(cal)
    coords, normals = mesh_world(skin)
    hands = {h: export_hand(arm, skin, drivers[h], cal['hands'][h], coords, normals) for h in 'LR'}
    torso = None
    if body.bone is not None:
        torso = dict(bone=body.bone.name, pivot=vec(body.pivot), pre=mat(body.pre), world=mat(body.world),
                     reference={h: mat(m) for h, m in body.reference.items()})
    neutral_bones = {b.name: mat(b.matrix) for b in arm.pose.bones}
    rig = dict(schema='performer-rig', neutral_bones=neutral_bones, scene=bpy.data.filepath,
               scene_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
               calibration_sha256=hashlib.sha256(cal_path.read_bytes()).hexdigest(),
               fps=bpy.context.scene.render.fps / bpy.context.scene.render.fps_base,
               arm_world=mat(arm.matrix_world), hands=hands, torso=torso, keyboard=keyboard,
               key_meshes=key_meshes(),
               key_depth_gain={p: k['local_z_world_gain'] for p, k in keyboard.items()})
    rig['validation'] = validation_poses(arm, skin, drivers, cal, hands, args.samples, 7)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rig), encoding='utf-8')
    print('RIG_EXPORTED', out, 'validation', len(rig['validation']), flush=True)


if __name__ == '__main__':
    main()
