"""Blender entry of the merged performer: solve (multi-process, outside Blender), write, save, check.

blender -b <scene with character and piano>.blend --python-exit-code 1 \
    --python pianist/performer/blender_main.py -- --inputs inputs/fur_elise --output output/fur_elise_vNN --workers 16

This is the only Blender process of a run. It prepares the bake's neutral scene,
starts the solve (performer.pipeline) as an ordinary Python process with its own
worker pool and waits for it, fits the bench and pedal extender (performer.staging),
then writes every curve (arms, torso, fingers, right foot and pedals,
keys) and saves ``<output>/performer.blend``. No rendering. Finally a sample
of frames is evaluated on the real deformed mesh to confirm the written pads
match the solved ones.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lib'))

from recorded_curve_io import replace_linear_curve   # noqa: E402
from thumb_driver import animation_curves, mesh_world   # noqa: E402

from performer.rig_export import neutral_scene       # noqa: E402
from performer.staging import animate_damper, fit_bench, pedal_extender   # noqa: E402

# the solve runs in an ordinary Python (numpy, scipy), not Blender's: PIANOMOTION_PYTHON, else the first python on PATH
PYTHON = os.environ.get('PIANOMOTION_PYTHON') or shutil.which('python') or shutil.which('python3')
CHECK_SECONDS = 2.          # native check: one frame every this many seconds


def write_curve(target, data_path, frames, values, index=None):
    """Replace the linear curves of ``target.data_path`` with ``values`` (n,) or (n, k)."""
    owner = target.id_data
    values = np.asarray(values, float)
    columns = values.reshape(len(frames), -1)
    if index is None:
        target.keyframe_insert(data_path=data_path, frame=1)
    else:
        target.keyframe_insert(data_path=data_path, index=index, frame=1)
    path = target.path_from_id(data_path) if target is not owner else data_path
    curves = {c.array_index: c for c in animation_curves(owner) if c.data_path == path}
    for k in range(columns.shape[1]):
        replace_linear_curve(curves[k if index is None else index], frames, columns[:, k])


def continuous_quaternions(q):
    q = np.array(q, float)
    for i in range(1, len(q)):
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    return q


def write_animation(arm, rig, motion):
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    frames = 1 + motion['times'] * fps
    for h in 'LR':
        names = rig['hands'][h]['bones']
        for k, role in enumerate(('upper', 'elbow', 'wrist', 'twist')):
            bone = arm.pose.bones[names[role]]
            bone.rotation_mode = 'QUATERNION'
            write_curve(bone, 'rotation_quaternion', frames, continuous_quaternions(motion[f'{h}_arm_q'][:, k]))
        for d, finger in rig['hands'][h]['fingers'].items():
            root = arm.pose.bones[finger['names'][0]]
            root.rotation_mode = 'QUATERNION'
            write_curve(root, 'rotation_quaternion', frames, continuous_quaternions(motion[f'{h}_root_q'][:, int(d) - 1]))
            for j, name in enumerate(finger['names'][1:]):
                bone = arm.pose.bones[name]
                bone.rotation_mode = 'XYZ'
                write_curve(bone, 'rotation_euler', frames, motion[f'{h}_child_euler'][:, int(d) - 1, j])
    if rig.get('torso') and np.any(motion['torso']):
        bone = arm.pose.bones[rig['torso']['bone']]
        bone.rotation_mode = 'QUATERNION'
        write_curve(bone, 'rotation_quaternion', frames, continuous_quaternions(motion['torso']))
    # Every key starts from rest with no animation: the scene may carry another performance's key curves
    # (the base file's), and a key this run never plays would replay them on its own.
    for key in [ob for ob in bpy.data.objects if ob.name.startswith('key_')]:
        key.animation_data_clear()
        key.delta_location = (0., 0., 0.)
    # Animated helper empties left in the scene by other tools would replay their motion too; nothing here animates one.
    for helper in [ob for ob in bpy.data.objects if ob.type == 'EMPTY' and ob.animation_data]:
        helper.animation_data_clear()
    for j, pitch in enumerate(motion['key_pitches']):
        key = bpy.data.objects[f'key_{int(pitch)}']
        gain = rig['key_depth_gain'][str(int(pitch))]
        write_curve(key, 'delta_location', frames, -motion['key_depth'][:, j] / gain, index=2)
    scene.frame_start = 1
    scene.frame_end = math.ceil(frames[-1])
    return frames


def add_audio(wav):
    """The plan's own audio preview as a packed sound strip from frame 1 (scene audio, no rendering)."""
    scene = bpy.context.scene
    if not scene.sequence_editor:
        scene.sequence_editor_create()
    editor = scene.sequence_editor
    strips = editor.strips if hasattr(editor, 'strips') else editor.sequences
    for s in [s for s in strips if s.type == 'SOUND']:
        strips.remove(s)
    strip = strips.new_sound('performance', str(wav), 1, 1)
    strip.sound.pack()
    # The base scene plays every frame ('NONE'): a viewport slower than the frame rate then falls behind the
    # sound, which seems early. Sync to audio drops frames instead, so keys and sound stay together.
    scene.sync_mode = 'AUDIO_SYNC'
    return strip.frame_final_duration


def native_check(skin, rig_path, motion, frames):
    """Evaluate the saved animation on the real mesh at sampled frames; compare pads with the solve."""
    from performer.kinematics import Rig
    from performer.pose import hand_solvers
    rig = Rig(rig_path)
    solvers = hand_solvers(rig)
    scene = bpy.context.scene
    step = max(1, int(round(CHECK_SECONDS * motion['rate'])))
    worst = 0.
    for i in range(0, len(frames), step):
        frame = frames[i]
        scene.frame_set(int(math.floor(frame)), subframe=float(frame - math.floor(frame)))
        coords, _ = mesh_world(skin)
        for h in 'LR':
            vector = motion[f'{h}_wrist'][i]
            # the wrist bone in armature space follows from the wrist vector alone (no arm or torso needed)
            arm_pose = dict(armature=rig.arms[h].arm_inverse @ solvers[h].palm(vector[:3], vector[3], vector[4]))
            values = {d: motion[f'{h}_values'][i, d - 1] for d in range(1, 6)}
            for d in range(1, 6):
                predicted = rig.pad_points(h, d, arm_pose['armature'], values)
                actual = coords[rig.fingers[h][d].pad_ids]
                worst = max(worst, float(np.abs(predicted - actual).max()))
    return dict(frames_checked=len(range(0, len(frames), step)), worst_pad_difference_mm=worst * 1000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--inputs', help='the piece folder (pipeline --inputs); not needed with --skip-solve')
    parser.add_argument('--workers', type=int, default=16, choices=(8, 16))
    parser.add_argument('--end', type=float)
    parser.add_argument('--skip-solve', action='store_true', help='write an existing motion.npz')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    started = time.perf_counter()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cal = json.loads((ROOT / 'hand_calibration.json').read_text(encoding='utf-8'))
    arm, skin, _, _, _ = neutral_scene(cal)
    bench_drop = fit_bench(skin)
    contacts = pedal_extender(skin)
    if not args.skip_solve:
        if not args.inputs:
            parser.error('--inputs is required unless --skip-solve')
        if not PYTHON:
            parser.error('no python on PATH for the solve: set PIANOMOTION_PYTHON to a Python with numpy and scipy')
        command = [PYTHON, '-m', 'performer.pipeline', '--inputs', args.inputs, '--output', str(out), '--workers', str(args.workers)]
        if args.end is not None:
            command += ['--end', str(args.end)]
        subprocess.run(command, cwd=ROOT, check=True)
    motion = dict(np.load(out / 'motion.npz'))
    from performer.pipeline import RIG as rig_path
    rig = json.loads(rig_path.read_text(encoding='utf-8'))
    frames = write_animation(arm, rig, motion)
    staging = dict(bench_drop_mm=bench_drop * 1000, extender_top_m=contacts['top'])
    if 'damper' in motion:
        fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
        staging.update(animate_damper(arm, skin, contacts, frames, motion['damper'], motion['damper_intervals'], fps, write_curve))
    audio_frames = add_audio(out / 'preview.wav') if (out / 'preview.wav').exists() else 0
    blend = out / 'performer.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), relative_remap=True)
    check = native_check(skin, rig_path, motion, frames)
    report_path = out / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    report['blender'] = dict(blend=str(blend), frames=int(len(frames)), frame_end=bpy.context.scene.frame_end, audio_frames=audio_frames,
                             native_check=check, staging=staging, write_seconds=time.perf_counter() - started)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PERFORMER_SAVED', blend, json.dumps(check), flush=True)


if __name__ == '__main__':
    main()
