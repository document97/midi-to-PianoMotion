"""Read-only: every key's animation in the .blend against the run's motion.npz; the damper pedal (foot, extender pedal,
piano pedal, exact pedal state); stale animation elsewhere; scene playback settings."""
import bpy, json, sys, numpy as np
from pathlib import Path
run = Path(sys.argv[sys.argv.index('--') + 1])
z = np.load(run / 'motion.npz'); t = z['times']; P = [int(p) for p in z['key_pitches']]
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'lib')]
from performer import staging as ST
rig = json.load(open(ROOT / 'performer' / 'rig_cache' / 'rig.json', encoding='utf-8'))
fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
frames = 1 + t * fps
def curves(ob):
    ad = ob.animation_data
    if not ad or not ad.action: return []
    out = []
    for layer in ad.action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                out += list(bag.fcurves)
    return out
stale, mismatch = [], []
for ob in bpy.data.objects:
    if not ob.name.startswith('key_'): continue
    p = int(ob.name.split('_')[1])
    fz = [fc for fc in curves(ob) if fc.data_path == 'delta_location' and fc.array_index == 2]
    vals = np.array([fz[0].evaluate(f) for f in frames]) if fz else np.zeros(len(frames))
    if p in P:
        want = -z['key_depth'][:, P.index(p)] / rig['key_depth_gain'][str(p)]
        err = np.abs(vals - want)
        if err.max() > 1e-5: mismatch.append((p, float(err.max()), int((err > 1e-5).sum())))
    else:
        moving = np.abs(vals) > 1e-6
        if moving.any():
            stale.append((p, ob.animation_data.action.name, int(moving.sum()), round(float(t[moving][0]), 2), round(float(np.abs(vals).max()) * 1000, 2)))
print('AUDIT', run.name, 'keys in run', len(P), 'stale keys (not played, still animated):', len(stale), 'keys not matching data:', len(mismatch))
for s in stale: print('STALE key', s[0], 'action', s[1], 'frames moving', s[2], 'first at s', s[3], 'max delta (scene units x1000)', s[4])
for m in mismatch[:10]: print('MISMATCH key', m)
mine = set()
for h in 'LR':
    hb = rig['hands'][h]['bones']; mine |= {hb[r] for r in ('upper', 'elbow', 'wrist', 'twist')}
    for d, f in rig['hands'][h]['fingers'].items(): mine |= set(f['names'])
if rig.get('torso'): mine.add(rig['torso']['bone'])
# The damper pedal: the extender pedal turns with the depth curve; the piano pedal's 'damper' state switches exactly
# at the source pedal events (what a sound engine reads).
pedal = []
if 'damper' in z.files:
    mine.add(ST.FOOT_TARGET)
    lever = [fc for fc in curves(bpy.data.objects[ST.LEVERS['R']]) if fc.data_path == 'rotation_euler' and fc.array_index == 0]
    got = np.array([lever[0].evaluate(f) for f in frames]) if lever else np.zeros(len(frames))
    err = np.abs(got + z['damper'] * ST.PEDAL_DEPTH / ST.LEVER_BACK).max()
    if err > 1e-5: pedal.append(f'extender pedal off the depth curve by {err:.2e} rad')
    state = [fc for fc in curves(bpy.data.objects[ST.PIANO_DAMPER]) if fc.data_path == '["damper"]']
    events = sorted([(b, 1.) for b, e in z['damper_intervals']] + [(e, 0.) for b, e in z['damper_intervals'] if np.isfinite(e)])
    if not state: pedal.append('no damper state on the piano pedal')
    else:
        for sec, value in events:
            f = 1 + sec * fps
            if abs(state[0].evaluate(f + .01) - value) > 1e-6 or (f > 1.02 and abs(state[0].evaluate(f - .01) - (1 - value)) > 1e-6):
                pedal.append(f'damper state wrong at {sec:.3f} s'); break
    foot = [fc for fc in curves(arm_ob) if fc.data_path.startswith(f'pose.bones["{ST.FOOT_TARGET}"]')] if (arm_ob := next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)) else []
    if len(foot) != 7: pedal.append(f'{len(foot)} foot curves (want location + quaternion)')
    print('PEDAL intervals', len(z['damper_intervals']), 'problems:', len(pedal))
    for p in pedal: print('PEDAL problem', p)
others = []
for ob in bpy.data.objects:
    if ob.name.startswith('key_') or ('damper' in z.files and ob.name in (ST.LEVERS['R'], ST.PIANO_DAMPER)): continue
    fcs = curves(ob)
    if not fcs: continue
    paths = {fc.data_path for fc in fcs}
    bones = {fc.data_path.split('"')[1] for fc in fcs if fc.data_path.startswith('pose.bones[')}
    stale_b = sorted(b for b in bones if b not in mine)
    obj_level = sorted(p for p in paths if not p.startswith('pose.bones['))
    if stale_b or obj_level:
        others.append((ob.name, ob.animation_data.action.name, len(stale_b), stale_b[:8], obj_level[:5]))
print('OTHER animated objects / bones not written by this run:', len(others))
for o in others: print('OTHER', o)
# Scene settings inherited from the base scene: none of them is trusted (its 'play every frame' made sound seem early).
scene, problems = bpy.context.scene, []
if scene.sync_mode != 'AUDIO_SYNC':
    problems.append(f'playback sync {scene.sync_mode} (sound runs ahead of a slow viewport; want AUDIO_SYNC)')
if scene.frame_end < frames[-1]:
    problems.append(f'scene ends at frame {scene.frame_end}, motion at {frames[-1]:.0f}')
ed = scene.sequence_editor
sounds = [s for s in ((ed.strips_all if hasattr(ed, 'strips_all') else ed.sequences_all) if ed else []) if s.type == 'SOUND']
if len(sounds) != 1:
    problems.append(f'{len(sounds)} sound strips (want the run preview only)')
for s in sounds:
    if s.frame_final_start != 1 or s.frame_offset_start != 0:
        problems.append(f'sound starts at frame {s.frame_final_start} (offset {s.frame_offset_start}), motion t=0 is frame 1')
    if not s.sound.packed_file:
        problems.append('sound not packed')
    wav = run / 'preview.wav'
    if wav.exists() and s.sound.packed_file and s.sound.packed_file.size != wav.stat().st_size:
        problems.append(f'packed sound ({s.sound.packed_file.size} B) is not this run preview.wav ({wav.stat().st_size} B)')
print('SCENE fps', fps, 'sync', scene.sync_mode, 'frames', scene.frame_start, scene.frame_end, 'problems:', len(problems))
for p in problems: print('SCENE problem', p)
