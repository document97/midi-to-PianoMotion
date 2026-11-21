"""One pass from the source MIDI inventory to animation arrays: plan, anchor poses, motion, checks.

python -m performer.pipeline --inputs inputs/fur_elise --output ../output/fur_elise_vNN --workers 16   (cwd: pianist)

Every stage runs on ``workers`` processes (8 or 16 equal cores): planning
segments (planner.parallel), anchor poses, wrist-path contact checks and
intermediate poses, finger keyposes and frame segments. The planner screens
chords with the rig's own two-finger table, and the motion is generated from
the poses on the chosen path, so nothing is solved twice. Output: motion.npz
(all curves for the Blender writer), plan.json, report.json and an audio
preview of what the hands actually play.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from planner.audio import render
from planner.fidelity import compare
from planner.parallel import plan_parallel
from planner.problem import PIECE_FILES, build, load_raw, plan_notes, register_report
from planner.search import Weights
from planner.validate import validate

from .anchors import KEY_RETURN_SECONDS, KEY_TRAVEL, REACH_TOLERANCE, REPAIR_FAR_CUT_SECONDS, Anchor, build_anchors, note_motions, solve_anchors, summary
from .checks import BRUSH_DEPTH, Heightfield, stretched
from .coherence import (COHERENCE_ROUNDS, MOTION_CUT_SECONDS, ORIENTATION_PULLS, free_releases, least_sunk, neighbour_targets,
                        settle, sunk, vector_of, wide_settle)
from .kinematics import DIGITS, SWING_LATERAL, Rig
from .torso import LEAN_NEAR, LEAN_GAIN, Torso, reach_window
from . import elbows
from .contacts import HandCapsules, finger_pairs, hands_gap, pointing, raised_between
from .motion import (CLEAR_STEP, REFINE_STEP, FingerIK, room_corrections, contact_error, contact_waypoints, finger_track, frame_hand,
                     placed, refine_points, refine_score, strokes, waypoint, wrist_path, SIDEWAYS_FREE, YieldingPath,
                     yield_offsets, hermite_eval, pchip_slopes, _split, REFINE_FREE, REFINE_LATERAL, REFINE_TOLERANCE,
                     OUTRO_T, OUTRO_HOLD, OUTRO_WAIT, OUTRO_START_GAP, lap_arrival, PATH_STEP)
from .arm import PALM_PITCH, LAP_PITCH, HANG_PITCH, FLAT_PITCH, TILT_PITCH, FALL_PITCH, OUTRO_DROOP

# The palm's pendulum: how far it turns up at the top of a lift, before the descent gives it back (deg). A few
# degrees is what the user reads as the wrist putting a little of itself into the rise; more looks like a
# flourish. PIANOMOTION_PALM_PENDULUM overrides it.
PITCH_PENDULUM = float(os.environ.get('PIANOMOTION_PALM_PENDULUM', 4.))
from .kinematics import FLOAT
from .pedal import PEDAL_LEAD, damper_depth, damper_intervals, pedal_union
from .pose import RELAXED, hand_solvers
from .progress import progress_map
from .reach_table import MAX_INTERVAL, REFERENCE_OCTAVE, cache_path, load_or_compute
from .regions import performer_solvers
from .posture import natural_orientation

HERE = Path(__file__).resolve().parent
RIG = Path(os.environ['PIANOMOTION_RIG']) if os.environ.get('PIANOMOTION_RIG') else HERE / 'rig_cache' / 'rig.json'     # performer seat and size (rig_export.PERFORMER_SEATING, CHARACTER_SCALE)
LEGACY_RIG = 'rig_unscaled.json'   # runs whose report does not name its rig were solved with this one


def run_rig(run):
    """The rig a run directory's motion was solved with (checks must measure a run on its own rig)."""
    report = json.loads((Path(run) / 'report.json').read_text(encoding='utf-8'))
    return HERE / 'rig_cache' / report.get('rig', LEGACY_RIG)
FRAME_RATE = 60.                  # samples per second (written as fractional scene frames)
# The second orientation pass (section 4.6: the hands re-aim at the forearms the elbows placed) is an A/B switch:
# PIANOMOTION_REAIM=0 gives the single-pass behaviour, for comparing against an older run without editing code.
REAIM_OFF = os.environ.get('PIANOMOTION_REAIM', '1').lower() in ('0', 'off', 'false', 'no')
# The attitude refine (section 4.65) follows the same switch: the hand is turned to point down the keys (attitude),
# so anchors are inserted wherever the path's yaw sits ATT_GATE degrees off that, ATT_STEP s apart, and settled with
# the aligned yaw as the target -- the wrist moves with it, by the pull, so the keys are kept.
ATT_STEP, ATT_GATE, ATT_ROUNDS, ATT_TURN = .3, 8., 2, 12.   # ATT_TURN: deg the target moves per round
# The attitude anchors' own pull: the wrist's place weighs more than in the orientation pulls (a hand that drifts
# across its keys to straighten its wrist trades one fault for another), the palm's yaw a little less.
ATT_PULLS = (np.array([.08, .03, .015, 1., 1.], dtype=FLOAT),)
APPLY_IDLE = os.environ.get('PIANOMOTION_IDLE_ATT', '1').lower() not in ('0', 'off', 'false', 'no')
TAIL_SECONDS = 2.4                # after the source end: the hands lift off and come to rest (silent)
# A held note keeps its lean row repeated this often (s): the lean's kernel reaches .8 s, so a note held for
# seconds would otherwise let the shoulder travel back while the hand is still on its key (the arm absorbs it).
TORSO_HOLD_STEP = .4
REST_OFFSET = np.array([0., 0., 0.])      # the rig's lap point: the wrist sits at the mid-thigh with the fingertips
                                          # reaching the part of the thigh just under the front of the keyboard.
REST_YAW = 0.                     # deg: the resting hand keeps its natural hang, fingers along the splayed thigh --
                                  # the inward turn to the keys happens on the lift, past the aim height (idle_attitude)
# Leg timings live in motion.py (REST_DRAW_T .. REST_SEAT_T); the palm's droop stages are HANG_PITCH /
# TOP_DROOP / EDGE_DROOP in arm.py, driven by the path's trip_* markers in the per-frame writer below.
PEEL_SKIN = .666                 # thigh-top skin height at the lap rest (measured): the peel keeps the trailing
PEEL_REACH = .15                 # finger chain (the droop's deepest point, fitted) from skin+4mm while the
                                 # wrist is low -- a real hand peels off the thigh, it does not stab into it
LAP_CONTACT = os.environ.get('PIANOMOTION_LAP_CONTACT', '1').lower() not in ('0', 'off', 'false', 'no')
                                  # put the resting hands on the leg where they actually touch it (lap_contact)
LAP_SLIDE_FOLLOW = float(os.environ.get('PIANOMOTION_LAP_SLIDE_FOLLOW', '0.91'))
                                  # share of the seat's lateral slide a resting hand rides: measured against the
                                  # real leg surface (the other ~9% is the body's lean). Tune per body/model.
LAP_SLIDE_FADE = .5               # s: fade the follow in around the landing so the return arc is untouched
INTRO_SECONDS = 5.0               # one breath of stillness, then a quick lift, a short thoughtful pause, a slow
                                  # forward-and-down reach and a slow press onto the keys: about 150 frames at 30 fps
WARM_FRAMES = 12                 # frames solved before a chunk (discarded) so the warm start carries over
CHUNKS_PER_WORKER = 4
REFINE_ROUNDS = 2                 # wrist path -> contact check -> intermediate poses (Daylight: after 1 round L 18 checks >8 mm, after 2 10; the 2nd round ~30 s)
SPIKE_ERROR = .005                # pad error above which a short run of frames is replaced by its neighbours
SPIKE_FRAMES = 4
SPIKE_LEAVE = .08                 # rad of finger travel into a frame that marks a candidate flicker run
SPIKE_JUMP = .07                  # rad: how far off the trajectory the run must go before it is worth bridging

_ctx = {}


def _init(context):
    rig = Rig(context['rig_path'])
    solvers = performer_solvers(rig)
    _ctx.update(context, rig=rig, solvers=solvers, ik={h: FingerIK(s) for h, s in solvers.items()},
                heightfield=Heightfield(rig.data['key_meshes']) if context.get('checks') else None,
                hand_notes={h: {k: n for k, n in context['notes'].items() if n.hand == h} for h in 'LR'})
    torso = None
    if context.get('torso_rows') and rig.data.get('torso'):
        torso = Torso(rig.data['torso'], rig.arms)
        torso.set_notes(*context['torso_rows'])
        if context.get('torso_cap') is not None:        # the elbows solve's ask: the body leans less here
            torso.set_cap(*context['torso_cap'])
        if context.get('torso_yield') is not None:      # ... and leans toward a hand for the wrist's floor
            torso.set_yield(*context['torso_yield'])
        if context.get('torso_slide_yield') is not None:    # ... and slides the seat where the lean's tilt
            torso.set_slide_yield(*context['torso_slide_yield'])   # would lower the palm angle itself
        if context.get('torso_sway') is not None:       # the intro's deliberate body sway (+back/-forward)
            torso.set_sway(*context['torso_sway'])
        if context.get('torso_slide_return') is not None:   # closing: seat back to its opening place
            torso.set_slide_return(*context['torso_slide_return'])
    _ctx['torso'] = torso
    _ctx['capsules'] = {h: HandCapsules(rig, h, rig.arms[h].arm_inverse @ solvers[h].palm(rig.neutral_wrist[h], 0., 0.))
                        for h in 'LR'}
    if context.get('tracks'):
        _configure_closing()


def _closing_palm(path, t):
    """Gentle release of the playing plane; compact FINGERS, not a wrist snap, provide clearance."""
    from .motion import OUTRO_TUCK_SECONDS, ease
    start = float(path.trip_outro) + OUTRO_TUCK_SECONDS
    end = getattr(path, 'trip_outro_touch', None) or path.trip_outro_land
    w = ease(float(np.clip((t - start) / max(end - start, 1e-6), 0., 1.)))
    return PALM_PITCH * (1. - w) + LAP_PITCH * w


def _configure_closing():
    """Configure the compact return on BOTH hands, using rig bounds and the real keybed geometry.

    The intro's tuck and the wrist/elbow paths are untouched. Called both for fresh runs and cached
    frame replays, so loading an old pkl cannot silently retain the failed shallow-tuck gesture.
    A hand is configured only if it actually closes (has trip_outro); a hand that never plays keeps
    its lap pose throughout.
    """
    from .motion import OUTRO_TUCK_SECONDS, KeyFloor
    from .checks import CELL
    bed = None
    for h in 'LR':
        path, tracks, rig = _ctx['paths'][h], _ctx['tracks'][h], _ctx['rig']
        start = getattr(path, 'trip_outro', None)
        if start is None:
            continue
        end = getattr(path, 'trip_outro_touch', None) or path.trip_outro_land
        tuck_end = start + OUTRO_TUCK_SECONDS
        if end <= tuck_end:
            continue
        # Capture the real lift-off articulation before enabling a closing override.
        for track in tracks.values():
            track.outro_gathered = None
            track.outro_from = None
            track.outro_unfold = None
        initial = frame_hand(start, path, tracks, _ctx['notes'], _ctx['waypoints'],
                             _ctx['solvers'][h], _ctx['ik'][h])[1]
        gathered = {}
        for d in DIGITS:
            finger, track = rig.fingers[h][d], tracks[d]
            # Fold the actual PIP/DIP hinges, not merely droop the palm. Stay within the model's existing
            # calibrated working bounds (no new joint limits; the opener continues to use LAP_TUCK).
            v = np.clip(np.asarray(track.rest, float), finger.lower, finger.upper)
            v[2:] = finger.upper[2:]
            if d != 1:
                # swing_lift is the actual MCP/root flexion on this rig; leaving it near the lap value
                # keeps the first phalanx straight even when PIP/DIP curl. Fold all three phalanges.
                v[1] = finger.lower[1]
            gathered[d] = v
            track.outro_gathered = v.copy()
            track.outro_from = initial[d].copy()
        if bed is None:
            bed = rig.data.get('piano_clearance')
            if bed is None:
                # Older rig caches predate the keybed envelope. The sidecar is measured from the SAME
                # scene; never substitute the movable key front, which is 28 mm farther away here.
                sidecar = Path(_ctx['rig_path']).with_name('piano_clearance.json')
                if not sidecar.exists():
                    raise ValueError('Export piano_clearance for this rig before solving a compact closing')
                bed = json.loads(sidecar.read_text(encoding='utf-8'))
        front = float(bed['front_y'])
        # Begin preparing to open once the folded skin has left the key-front footprint. Use the
        # existing key-clearance lead, not a scene-time constant. Final unfolding ends ABOVE the thigh.
        times = np.linspace(tuck_end, end, max(2, int(np.ceil((end - tuck_end) * FRAME_RATE)) + 1))
        unfold = end - 1. / FRAME_RATE
        for t in times:
            v = path(t)
            wrist = rig.arms[h].arm_inverse @ _ctx['solvers'][h].palm(v[:3], v[3], v[4], _closing_palm(path, t))
            points = np.concatenate([rig.clearance_points(h, d, wrist, gathered) for d in DIGITS])
            # Reserve 4 raster cells: the real fractional-frame mesh consumed the first two during
            # unfolding (1.4 mm residual in the diagnostic). This moves only the departure clock.
            if points[:, 1].max() <= front - KeyFloor.FRONT_LEAD - 4. * CELL:
                unfold = float(t)
                break
        for track in tracks.values():
            track.outro_unfold = unfold


# An anchor's pose is the most comfortable place for its own keys, but the keys can be played from other depths into
# them too: a hand playing a figure stays at one depth, its white keys pressed further in or out, and it turns rather
# than moving in and out (the thumb on a black key, the little finger on a white one beside another black key). The
# depth family is the whole pose settled at depths around the anchor's, everything but the depth free (the coherence's
# settle); the wrist path chooses the depth over time (motion.solve_path) and takes the rest of the hand from it.
FAMILY_STEP = .008               # m between the depths tried ...
FAMILY_REACH = .048              # ... up to this far each way
FAMILY_PULL = np.array([.08, .0005, .015, 4., 4.])    # the depth held where it is tried, the rest settled as the coherence does
FAMILY_HOLD = .002               # m: a settle that could not keep the depth this close is not that depth
FAMILY_ITERATIONS = 8           # each depth settled from the one before (a few mm off): a short solve
FAMILY_GIVE = .5                 # awkwardness units a depth may add to the pose's own (sink beside, swing, web, miss) ...
FAMILY_OK = 1.                   # ... and up to this a depth is never too awkward: one standard for every chord, so a
                                 # chord played where a fuller one around it is (as strained as that, at most) is not refused;
FAMILY_SINK = .002               # but no finger sinks into the keys beside its own further than in the pose (or this, m)
FAMILY_SINK_GIVE = .001          # ... plus this: a place is never bought with a finger in the keys
# ... and at a depth the hand may turn: a figure is played at one depth and one turn where its chords allow, the
# thumb on a black key reached by turning the hand rather than by moving it in and out. Turns are tried at every
# third depth (and the ends), held like the depth.
TURN_STEP = 8.                   # deg between the turns tried ...
TURN_STEPS = 2                   # ... this many each way
TURN_PULL = np.array([.08, .0005, .015, .2, 4.])
TURN_HOLD = 1.                   # deg
# ... and at those depths the hand may also stand to either side: a figure's keys played from one place, each finger
# reaching for its key (the thumb on its key from a hand standing over the figure's middle) rather than the hand
# moving over every key in turn. Held sideways only: standing to one side, the hand goes in or out as its keys need.
SIDE_STEP = .008                 # m between the side places tried ...
SIDE_STEPS = 4                   # ... this many each way
SIDE_PULL = np.array([.0005, .08, .015, 4., 4.])
SIDE_ITERATIONS = 25             # a side step moves the fingers on their keys more than a depth step: a full settle


def depth_family(solver, anchor, notes):
    """The anchor's hand at other depths, turns and sides: (wrist vectors, poses, awkwardness) of every place it was
    settled at and holds its keys from, the pose's own first. The depths: settled at depths into the keys around its
    own, as far each way as its keys are held without more strain (``awkwardness``, the measure the planner prices);
    the turns and sides: at every third of those depths, the hand turned or moved sideways each way as far as that
    holds. The wrist path chooses
    among them (motion.choose_places)."""
    pose = anchor.pose
    contacts = anchor.contacts(notes)
    base = vector_of(pose)
    if not contacts:
        return base[None], [pose], np.zeros(1)
    own = awkwardness(solver, contacts, pose)
    sink = max(FAMILY_SINK, max(sunk(solver, contacts, pose)) / 1000.) + FAMILY_SINK_GIVE

    def ok(p, target, axis):
        return (p.contact_error <= REACH_TOLERANCE / 2
                and all(abs(float(p.wrist[a]) - target[a]) <= FAMILY_HOLD for a in ({0} if axis == 0 else {1, axis}) if a < 3)
                and (axis != 3 or abs(float(p.yaw) - target[3]) <= TURN_HOLD)
                and max(sunk(solver, contacts, p)) / 1000. <= sink
                and awkwardness(solver, contacts, p) <= max(own + FAMILY_GIVE, FAMILY_OK))

    def walk(v0, p0, axis, step, steps, pull, iterations=FAMILY_ITERATIONS):
        """Settles from (v0, p0) along ``axis`` by ``step`` each way while they hold: [(vector, pose)] with v0's."""
        out = [(v0, p0)]
        for sign in (-1., 1.):
            last_v, last_values = v0, p0.values
            for i in range(1, steps + 1):
                target = v0.copy()
                target[axis] += sign * i * step
                start = last_v.copy()
                start[axis] = target[axis]
                p = settle(solver, contacts, start, {d: last_values.get(d, RELAXED[d]) for d in DIGITS}, target, pull,
                           iterations)
                if not ok(p, target, axis):
                    break
                last_v, last_values = vector_of(p), p.values
                out.append((last_v, p))
        return sorted(out, key=lambda e: float(e[0][axis]))

    line = walk(base, pose, 1, FAMILY_STEP, int(round(FAMILY_REACH / FAMILY_STEP)), FAMILY_PULL)
    centre = next(i for i, (v, _) in enumerate(line) if v is base)
    picks = sorted({i for i in range(len(line)) if (i - centre) % 3 == 0} | {0, len(line) - 1})
    places = list(line)
    for i in picks:
        for axis, step, steps, pull, iterations in ((3, TURN_STEP, TURN_STEPS, TURN_PULL, FAMILY_ITERATIONS),
                                                     (0, SIDE_STEP, SIDE_STEPS, SIDE_PULL, SIDE_ITERATIONS)):
            places += [e for e in walk(*line[i], axis, step, steps, pull, iterations) if e[1] is not line[i][1]]
    places.sort(key=lambda e: e[1] is not pose)          # the pose's own first
    return (np.array([e[0] for e in places]), [e[1] for e in places],
            np.array([awkwardness(solver, contacts, e[1]) for e in places]))


def _family(job):
    return [depth_family(_ctx['solvers'][a.hand], a, _ctx['notes']) for a in job]



def _check(job):
    hand, times = job
    return np.array([contact_error(t, _ctx['paths'][hand], _ctx['hand_notes'][hand], _ctx['waypoints'],
                                   _ctx['solvers'][hand], _ctx['ik'][hand]) for t in times])


def _held_errors(job):
    """Pad error (m) of each key the hand holds at ``t`` on its wrist path: {note id: error}."""
    hand, t = job
    path, solver, ik = _ctx['paths'][hand], _ctx['solvers'][hand], _ctx['ik'][hand]
    palm = solver.palm(*_split(path(t)))
    out = {}
    for n in _ctx['hand_notes'][hand].values():
        if n.touch <= t <= n.release and n.id in _ctx['waypoints']:
            top, seed = waypoint(_ctx['waypoints'][n.id], t, path)
            out[n.id] = float(ik.solve(n.digit, palm, top - np.array([0., 0., n.depth(t)], dtype=FLOAT), seed)[1])
    return out


def leave_lifts(drops, errors, samples, notes):
    """Held keys to let go early where the hand cannot hold them on its way (an intermediate pose there would snap
    the wrist off its path, ``gentle``): ``drops`` [(note id, time)], ``errors`` the hand's pad errors on ``samples``.
    Each key is lifted so it is up where its error run (pad error above REACH_TOLERANCE) begins, if that is legal and
    loses at most REPAIR_FAR_CUT_SECONDS of sound against the planned release (as an anchor repair may). -> {note id: lift time}."""
    lifts = {}
    bad = errors > REACH_TOLERANCE
    for k, t in drops:
        n = notes[k]
        i = int(np.argmin(np.abs(samples - t)))
        while i > 0 and bad[i - 1] and samples[i - 1] >= n.touch:
            i -= 1
        lift = max(n.earliest_release, float(samples[i]) - KEY_RETURN_SECONDS)
        if lift >= n.release - 1e-6 or max(0., min(n.pedal_release, n.release) - lift) > REPAIR_FAR_CUT_SECONDS:
            continue
        lifts[k] = min(lift, lifts.get(k, lift))
    return lifts


NEAR_PATH = np.array([.01, .01, .01, 3., 3.])     # intermediate poses stay this close to the wrist path
INTERMEDIATE_TURN = 120.          # deg/s: an intermediate pose turning the wrist off its path faster than this
INTERMEDIATE_SPEED = .3           # m/s: ... or moving it off faster than this (deviation / time to the nearest
                                  # anchor) is dropped: a snap of the hand is worse than a pad off its key centre
ARM_STRETCH = .998                # an out-of-reach wrist: the arm stretches to this share of its full length
RELAX_DELAY = .15                 # s after a hand lifts some of its keys it settles around those it still holds ...
RELAX_HOLD = .5                   # ... when it holds them at least this much longer before its next anchor (a pose
                                  # taken for all of them is no pose to hold the rest in)
HANDS_APART = .16                 # m between the wrists: an idle hand moves out of the playing one's way (about
                                  # 20 mm between the hands' bodies)
YIELD_SPEED = .4                  # m/s: how fast the idle hand moves aside and back
YIELD_BUSY = (.35, .15)           # s before a note's onset / after its release the hand counts as playing
# Awkward hand shapes: every anchor of two or more keys is settled as the coherence will settle it, and its
# awkwardness (squared, summed over the fingers) priced for the planner, which then plays the shape anyway, takes
# other fingers, omits a note or lets a held key go early, whichever costs least; the piece is planned again.
AWKWARD_SINK = .006               # m a finger sinks into the keys beside its own: one unit
AWKWARD_OUTER_SINK = .008         # ... beyond this for the thumb or little finger on the shape's outer key: at this
                                  # hand's size every octave has them lying 6-8 mm on the next keys, at their limit
AWKWARD_SIDEWAYS = .25            # units for a finger 2..4 swung sideways to its limit (beyond SIDEWAYS_FREE)
AWKWARD_MISS = (.002, .003)       # contact error free up to (an octave's usual miss), then m per unit
AWKWARD_FREE = .5                 # units a shape may have without cost
AWKWARD_COST = 9.                 # per unit beyond: one unit ~ an inner finger 6 mm into the next key ~ omitting a soft
                                  # inner note (Weights.omission x importance 8..16; melody and bass 12..24)
SHAPE_ROUNDS = 4                  # planning rounds at most (each prices the shapes the previous plan chose)


def relax_anchors(anchors, notes):
    """Anchors (no new key) where a hand, having lifted some keys, holds the rest ``RELAX_HOLD`` or more before its
    next anchor: there it settles around the keys it still holds (the coherence turns it to its natural orientation)
    instead of keeping, through the hold, the shape it took for all of them."""
    out = []
    for h in 'LR':
        mine = sorted((a for a in anchors if a.hand == h), key=lambda a: a.time)
        for a, b in zip(mine, mine[1:]):
            keys = a.incoming + a.held
            lifts = sorted({notes[k].release for k in keys if a.time < notes[k].release < b.time})
            for r, after in zip(lifts, lifts[1:] + [None]):
                t = r + RELAX_DELAY
                if after is not None and after <= t:
                    continue                      # settle once the lifts close together are done
                held = tuple(k for k in keys if notes[k].held_at(t))
                if held and b.time - t >= RELAX_HOLD:
                    out.append(Anchor(h, -1, t, a.x_hint, (), held))
    return out


def apply_yield(paths, notes, duration):
    """The idle hand's sideways give-way, applied to bare paths and exactly once.

    Two stages rebuild ``paths`` from the anchors (the re-aim and the attitude round), so the yield is asked for
    again after them -- on a path that may already carry one when no anchor was inserted in between. Wrapping a
    ``YieldingPath`` in another adds a second offset measured off the already-offset path (a fixed-point pass
    that converges to ~0 but re-reads both wrists at 60 Hz and can leave a millimetre), so the base is unwrapped
    first: the offsets are always computed from the path the anchors define, and the result carries exactly one.
    Returns ``({hand: YieldingPath}, offsets)``."""
    base = {h: (paths[h].base if isinstance(paths[h], YieldingPath) else paths[h]) for h in 'LR'}
    offsets = yield_offsets(base, notes, duration, HANDS_APART, YIELD_SPEED, YIELD_BUSY,
                            closing={h: getattr(base[h], 'trip_outro', None) for h in 'LR'},
                            opening={h: getattr(base[h], 'trip_leave', None) for h in 'LR'})
    return {h: YieldingPath(base[h], *offsets[h]) for h in 'LR'}, offsets


def gentle(anchor, pose, path, others):
    """An intermediate ``pose`` that leaves the wrist ``path`` slowly enough next to the hand's ``others`` anchors.
    The wrist's clock is not the anchors' spacing but the path's own knot grid: ``solve_path`` samples at PATH_STEP
    and every sample is a knot of the curve the frames read, so a pose that deviates by ``d`` puts the whole ``d``
    inside one 25 ms knot whatever the nearest anchor is. Measured (canon, frame 350): an anchor inserted 7 ms
    after a strike, 28 mm off the path, read 16 mm/frame -- ``gentle`` passed it because the nearest anchor was
    95 ms away. The palm's turn keeps the anchor spacing (the yaw is rate-limited downstream); the position does
    not."""
    dev = vector_of(pose) - path(anchor.time)
    dt = max(min((abs(o.time - anchor.time) for o in others if o is not anchor), default=1.), 1e-3)
    step = min(dt, PATH_STEP)
    return (float(np.abs(dev[3:]).max()) / dt <= INTERMEDIATE_TURN
            and float(np.linalg.norm(dev[:3])) / step <= INTERMEDIATE_SPEED)


def awkwardness(solver, contacts, pose):
    """Units (``AWKWARD_*``): squared sink of every finger into the keys beside its own (the thumb and little finger on
    the shape's outer keys beyond an octave's usual), squared sideways swing of fingers 2..4 beyond their free range,
    squared opening of two contact fingers beyond the web's free opening (one unit at its limit, Rig.web), squared
    contact miss."""
    fingers = solver.rig.fingers[solver.hand]
    pitches = [c.pitch for c in contacts]
    units = 0.
    for c, mm in zip(contacts, sunk(solver, contacts, pose)):
        outer = c.digit in (1, 5) and c.pitch in (min(pitches), max(pitches))
        units += (max(0., mm / 1000 - (AWKWARD_OUTER_SINK if outer else 0.)) / AWKWARD_SINK) ** 2
        if not outer:
            free, limit = SIDEWAYS_FREE[c.digit], float(fingers[c.digit].upper[0])
            units += AWKWARD_SIDEWAYS * (max(0., abs(float(pose.values[c.digit][0])) - free) / max(limit - free, 1e-6)) ** 2
    rest, slope, free, hard = solver.rig.web(solver.hand)
    digits = sorted({c.digit for c in contacts})
    for i, a in enumerate(digits):
        for b in digits[i + 1:]:
            opening = rest[a, b] + slope[a, b] * (float(pose.values[b][0]) - float(pose.values[a][0]))
            units += (max(0., opening - free[a, b]) / (hard[a, b] - free[a, b])) ** 2
    return units + (max(0., pose.contact_error - AWKWARD_MISS[0]) / AWKWARD_MISS[1]) ** 2


def free_start(solver, contacts):
    """(vector, values) of ``contacts`` solved from the hand's own wide seeds, not from where the planner put the wrist:
    a shape settled only from its anchor's pose stays in the basin it starts in, often a strained one."""
    from .pose import WIDE_SEEDS
    pose = solver.solve(contacts, seeds=WIDE_SEEDS)
    return vector_of(pose), pose.values


def _free_start(shape):
    hand, contacts = shape
    return free_start(_ctx['solvers'][hand], list(contacts))


def free_starts(pool, shapes, starts):
    """Adds to ``starts`` the ``free_start`` of every (hand, contacts) of two or more keys in ``shapes`` it lacks: it
    depends on nothing else, so each shape is solved once for every planning round and the coherence."""
    todo = sorted({(h, tuple(c)) for h, c in shapes if len(c) > 1} - set(starts), key=repr)
    starts.update(zip(todo, progress_map(pool, _free_start, todo, 'free starts')))


def _awkward(job):
    """``awkwardness`` of an anchor settled from its pose (over the key surface, as the coherence settles it), or for
    a shape of two or more keys from a free start (``free_start``) where that is less awkward, as the coherence settles
    it too; None without a heightfield."""
    hand, contacts, vector, values, error, free = job
    solver = _ctx['solvers'][hand]
    if solver.heightfield is None:
        return None
    starts = [(vector, values)] + ([free] if free is not None else [])
    return min(awkwardness(solver, contacts, settle(solver, contacts, v, {d: vals.get(d, RELAXED[d]) for d in DIGITS}, v,
                                                     ORIENTATION_PULLS[0])) for v, vals in starts)


def _awkward_key(job):
    hand, contacts, vector, values, error, free = job
    return hand, tuple(contacts), vector.tobytes(), tuple((d, np.asarray(v).tobytes()) for d, v in sorted(values.items())), error


def _pair_awkward(case):
    """``awkwardness`` of one finger pair on its keys (the pair table's pose, settled as an anchor is)."""
    from .pose import WIDE_SEEDS, Contact
    hand, f1, p1, f2, p2 = case
    solver = _ctx['solvers'][hand]
    contacts = [Contact(f1, p1, KEY_TRAVEL), Contact(f2, p2, KEY_TRAVEL)]
    pose = solver.solve(contacts, seeds=WIDE_SEEDS, clearance=False)
    if not pose.ok:
        # the performer's wrist sits further back than the pair table's (regions.PLAYING_WRIST_OFFSET), so its seeds
        # can all fall short of a wide reach the table found (black-key octaves): start from the table's own pose
        if 'table_solvers' not in _ctx:
            from .pose import hand_solvers
            _ctx['table_solvers'] = hand_solvers(_ctx['rig'])
        seed = _ctx['table_solvers'][hand].solve(contacts, seeds=WIDE_SEEDS, clearance=False)
        if seed.ok:
            pose = solver.solve(contacts, values=seed.values, wrist=seed.wrist.copy(), yaw=seed.yaw, roll=seed.roll,
                                seeds=WIDE_SEEDS, clearance=False)
    if not pose.ok:
        return case, float('inf')
    pose = settle(solver, contacts, vector_of(pose), {d: pose.values.get(d, RELAXED[d]) for d in DIGITS}, vector_of(pose),
                  ORIENTATION_PULLS[0])
    return case, awkwardness(solver, contacts, pose)


def pair_awkwardness(table, workers):
    """The shape awkwardness (``awkwardness``) of every feasible finger pair of the pair table, shaped like it (inf
    where infeasible): the planner prices a shape's strain with the measure the performer judges it by. Cached next
    to the pair table, for its geometry and the code that measures it: only the code the measure can reach (codekey),
    so an edit to another stage (the wrist path, the motion) leaves it valid."""
    import hashlib
    import inspect
    from . import arm, checks, kinematics, pose, regions
    from .codekey import code_key
    classes = [v for m in (kinematics, arm, pose, checks, regions) for v in vars(m).values()
               if inspect.isclass(v) and v.__module__ == m.__name__]       # the solvers, rig and keys it runs on
    code = code_key(_pair_awkward, awkwardness, performer_solvers, *classes,
                    extra=hashlib.sha256(np.ascontiguousarray(table).tobytes()).digest())
    path = cache_path(RIG).with_name(f'pair_awkward_{cache_path(RIG).stem[6:]}_{code[:8]}.npz')
    if path.exists():
        return np.load(path)['units']
    units = np.full(table.shape, np.inf)
    jobs = []
    for h, hand in enumerate('LR'):
        for f1 in DIGITS:
            for f2 in DIGITS:
                for pc in range(12):
                    for interval in range(1, MAX_INTERVAL + 1):
                        if f1 != f2 and table[h, f1 - 1, f2 - 1, pc, interval + MAX_INTERVAL]:
                            p1 = REFERENCE_OCTAVE[hand] + pc
                            jobs.append((hand, f1, p1, f2, p1 + interval))
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(dict(rig_path=str(RIG), notes={}, checks=True),)) as pool:
        for (hand, f1, p1, f2, p2), u in progress_map(pool, _pair_awkward, jobs, 'pair awkwardness', 8):
            h = 'LR'.index(hand)
            units[h, f1 - 1, f2 - 1, p1 % 12, p2 - p1 + MAX_INTERVAL] = u
            units[h, f2 - 1, f1 - 1, p2 % 12, p1 - p2 + MAX_INTERVAL] = u
    np.savez_compressed(path, units=units)
    return units


def shape_key(anchor, notes):
    """The planner's name for an anchor's shape (``Planner.shape_penalty``): hand and its (note, finger) on the keys."""
    return anchor.hand, frozenset((notes[i].index, notes[i].digit) for i in anchor.incoming + anchor.held)


def awkward_shapes(anchors, notes, base, workers, cache=None, starts=None):
    """{shape_key: awkwardness units} of every anchor holding two or more keys. ``starts``: the free starts
    (``free_starts``) found so far, added to."""
    chords = [a for a in anchors if len(a.incoming) + len(a.held) >= 2]
    starts = {} if starts is None else starts
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(base,)) as pool:
        shapes = [(a.hand, a.contacts(notes, hover=False)) for a in chords]    # the planner prices the keys it plays
        free_starts(pool, shapes, starts)
        jobs = [(h, c, vector_of(a.pose), a.pose.values, float(a.pose.contact_error), starts.get((h, tuple(c))))
                for a, (h, c) in zip(chords, shapes)]
        found = progress_map(pool, _awkward, jobs, 'chord shapes', 2, cache, _awkward_key)
    out = {}
    for a, units in zip(chords, found):
        if units is not None:
            key = shape_key(a, notes)
            out[key] = max(out.get(key, 0.), units)
    return out


def shape_report(chosen):
    """The plan's shapes beyond ``AWKWARD_FREE`` (played anyway: no cheaper alternative was found)."""
    units = sorted((u for u in chosen.values() if u > AWKWARD_FREE), reverse=True)
    return dict(shapes=len(chosen), awkward=len(units), worst_units=[round(u, 2) for u in units[:10]],
                cost=round(sum(AWKWARD_COST * (u - AWKWARD_FREE) for u in units), 1))


def _settle(job):
    """First pose (over the given pulls) whose contact error is within ``limit``; None if none is. ``wide``: a
    stretched hand, searched from several wrist starts (``wide_settle``) with the first pull. ``starts``: other
    (vector, values) to settle from as well; of the poses within ``limit`` the least awkward (``awkwardness``)."""
    hand, contacts, vector, values, target, pulls, limit, *extra = job
    wide, starts = (list(extra) + [False, ()][len(extra):])[:2]
    solver = _ctx['solvers'][hand]
    if wide:
        pose = wide_settle(solver, contacts, vector, {d: values.get(d, RELAXED[d]) for d in DIGITS}, target, pulls[0], limit)
        if pose is not None and pose.contact_error <= limit:
            return pose
    found = []
    for v, vals in ((vector, values), *starts):
        vals = {d: vals.get(d, RELAXED[d]) for d in DIGITS}
        for pull in pulls:
            pose = settle(solver, contacts, v, vals, target, pull)
            if pose.contact_error <= limit:
                found.append(pose)
                break
    if len(found) > 1 and solver.heightfield is not None:
        return min(found, key=lambda p: awkwardness(solver, contacts, p))
    return found[0] if found else None


def _track(job):
    hand, digit = job
    return job, finger_track(hand, digit, _ctx['notes'], _ctx['strokes'], _ctx['waypoints'], _ctx['paths'][hand],
                             _ctx['solvers'][hand], _ctx['ik'][hand])


def _room(job):
    h, times = job
    return room_corrections(times, _ctx['paths'][h], _ctx['tracks'][h], _ctx['notes'], _ctx['waypoints'],
                            _ctx['solvers'][h], _ctx['ik'][h])


def _frames(times):
    rig, notes, tracks = _ctx['rig'], _ctx['notes'], _ctx['tracks']
    # The lap rest is the one hand pose measured against the body (lap_contact: every pad on the leg within
    # 2 mm, nothing more than a millimetre inside anything). Record how low each finger's skin sits there, so
    # a frame can be asked whether it has driven a finger below the leg's surface. This is the check the
    # closing gesture needed: the fingers tuck 40 deg to clear the keybed and used to arrive still tucked.
    lap_ref = {'L': None, 'R': None}
    if _ctx.get('checks'):
        swing = _ctx.get('elbow_swing')
        for h in 'LR':
            path_h = _ctx['paths'][h]
            t_rest = float(path_h.times[-1])
            rest_v = np.asarray(path_h(t_rest), dtype=float)
            rest_values = {d: np.clip(np.asarray(tracks[h][d].rest, dtype=float),
                                      rig.fingers[h][d].lower, rig.fingers[h][d].upper) for d in DIGITS}
            # the SAME arm the frame stage builds: the elbow swing and the torso's shoulder frame move the
            # hand's own frame a millimetre or two, and the reference must not carry that as penetration
            pre_r = None if _ctx['torso'] is None else _ctx['torso'].pose(t_rest)[1].get(h)
            phi_r = (float(hermite_eval(swing[h]['times'], swing[h]['angles'][:, None],
                                        _ctx['elbow_swing_slopes'][h], t_rest)[0])
                     if swing is not None and h in swing else None)
            try:
                pose_r = rig.arms[h].pose(rest_v[:3], rest_v[3], rest_v[4], pre_r, phi_r,
                                          pitch=LAP_PITCH)
            except ValueError:
                continue
            lap_ref[h] = {d: float(rig.clearance_points(h, d, pose_r['armature'], rest_values)[:, 2].min())
                          for d in DIGITS}
    by_pitch = _ctx['by_pitch']
    pitches = sorted(by_pitch)
    n = len(times)
    # A resting hand must ride the leg as the seat slides under the body. The wrap below follows the seat's
    # lateral travel, scaled to the leg surface's measured travel (0.91 of the seat's: the rest is the lean).
    # The opening seat is the reference -- lap_contact's fit was measured there, so at delta=0 the pads match.
    torso_f = _ctx['torso']
    if torso_f is None or not torso_f.slide_max:
        slide_base = 0.
    else:
        # The opening seat = first value of the slide curve (what the closing eases back to). Independent of
        # which chunk of ``times`` this worker got -- using times[0] here would differ per parallel chunk.
        slide_base = float(torso_f._slide_curve()[1][0])
    out = dict(times=np.asarray(times), torso=np.zeros((n, 4)), key_depth=np.zeros((n, len(pitches))))
    out.update({f'{h}_collar_q': np.zeros((n, 4)) for h in 'LR'}, slide=np.zeros((n, 3)))
    spine_names = [] if _ctx['torso'] is None else [b['bone'] for b in _ctx['torso'].spine] + [_ctx['torso'].neck['bone']]
    out['spine'] = np.zeros((n, len(spine_names), 4))
    for h in 'LR':
        out[f'{h}_wrist'] = np.zeros((n, 5))
        out[f'{h}_values'] = np.zeros((n, 5, 4))
        out[f'{h}_root_q'] = np.zeros((n, 5, 4))
        out[f'{h}_child_euler'] = np.zeros((n, 5, 2, 3))
        out[f'{h}_arm_q'] = np.zeros((n, 4, 4))     # upper, elbow, wrist, twist
        out[f'{h}_pitch'] = np.full(n, PALM_PITCH)  # palm pitch actually written into the wrist bone (deg)
        out[f'{h}_arm_world'] = np.zeros((n, 2, 3))  # shoulder, elbow world (self-describing metrics)
        out[f'{h}_palm_deg'] = np.zeros(n)           # palm long axis vs forearm axis (wrist-deviation comfort)
        out[f'{h}_contact_error'] = np.zeros(n)
        out[f'{h}_penetration'] = np.zeros(n)
        out[f'{h}_lap_penetration'] = np.zeros(n)   # skin into the thigh (checks.LapField)
        out[f'{h}_arm_ok'] = np.ones(n, dtype=bool)
        out[f'{h}_brush'] = np.zeros((n, KEYS))     # push each unplayed key wants from a finger resting on it
        out[f'{h}_finger_gap'] = np.zeros((n, 3))   # neighbour fingers 2-3, 3-4, 4-5 seen from above (crossings)
        out[f'{h}_riding'] = np.zeros((n, 5), dtype=bool)   # finger on its key
        out[f'{h}_finger_overlap'] = np.zeros((n, 4))       # neighbour finger bodies 1-2 .. 4-5 running into each other (m)
        out[f'{h}_raised'] = np.zeros(n)                    # free finger 3/4 held above its two pressing neighbours (m)
        out[f'{h}_pointing'] = np.zeros((n, 5))             # finger direction from above against the keys (deg)
    out['hands_gap'] = np.zeros(n)                          # clearance between the two hands' fingers (m)
    # The intro palm is one continuous angular curve, not a chain of ease-to-zero stages. Only the true
    # endpoints and genuine reversals have zero angular speed; interior marks never dwell. Monotone cubic
    # slopes keep the hang and tension peaks inside their specified angles (no cubic overshoot past +30).
    intro_pitch = {}
    for h in 'LR':
        path_h = _ctx['paths'][h]
        leave_h = getattr(path_h, 'trip_leave', None)
        if leave_h is None:
            continue
        de = getattr(path_h, 'trip_draw_end', None) or leave_h + .30
        top = getattr(path_h, 'trip_lift_end', None) or de + .45
        tlt = getattr(path_h, 'trip_tilt_end', None) or top + .33
        red = getattr(path_h, 'trip_reach_end', None) or tlt + 1.5
        mid = (de + top) * .5
        quiet = tlt <= top + 1e-6
        kt = [de, mid, top] + ([] if quiet else [tlt]) + [red]
        kv = [LAP_PITCH, HANG_PITCH, FLAT_PITCH] + ([] if quiet else [TILT_PITCH]) + [PALM_PITCH]
        kt, kv = np.asarray(kt), np.asarray(kv)[:, None]
        slopes = pchip_slopes(kt, kv)
        grid = np.linspace(de, red, max(121, int(np.ceil((red - de) * 120))))
        intro_pitch[h] = (de, top, grid, np.asarray([hermite_eval(kt, kv, slopes, t)[0] for t in grid]))
    # Preserve the previous right-hand closing curve. The LEFT return is now articulated separately by
    # _configure_left_closing / _closing_palm: genuinely folded MCP/PIP/DIP joints, then a long unfolding
    # into the lap pose. Its clearance must not be purchased with the rapid -33 deg wrist snap below.
    outro_pitch = {}
    for h in 'LR':
        out_h = getattr(_ctx['paths'][h], 'trip_outro', None)
        if out_h is None:
            continue
        land_h = getattr(_ctx['paths'][h], 'trip_outro_land', None) or out_h + OUTRO_T - OUTRO_HOLD
        # The palm must be flat on the lap by the time the hand TOUCHES it, not by the gesture's nominal end:
        # the last stretch of the return is a slow slide along the leg, so a palm still drooping there puts
        # the fingertips a couple of millimetres into the thigh for a few tenths of a second.
        touch_h = getattr(_ctx['paths'][h], 'trip_outro_touch', None) or land_h
        # Hold the playing plane while the gathered fingers back off from the keys, then reach the deep
        # HANG_PITCH BEFORE the wrist sinks into the key-front band. The palm must cross the dangerous
        # -14..-19 deg band while the wrist is still high (z~0.78, which is clip-free), NOT a moment later
        # at z~0.768 where the gathered middle/ring finger hits piano_part_000 by 5-8 mm. Measured shares
        # of the return (v50, 0 mm): hold at 0.49, full hang at 0.58, flatten at 0.91. Re-timed to the
        # gesture's own length so any piece keeps the same clearance; never dwell at the mid angles.
        hold_at = out_h + .49 * (touch_h - out_h)
        droop_at = out_h + .58 * (touch_h - out_h)
        flat_at = out_h + .91 * (touch_h - out_h)
        tt = np.maximum.accumulate(np.asarray([out_h, hold_at, droop_at, max(droop_at + .04, flat_at),
                                               max(flat_at + .04, touch_h)], float) + np.arange(5) * 1e-3)
        vv = np.asarray([[PALM_PITCH], [PALM_PITCH], [HANG_PITCH], [HANG_PITCH], [LAP_PITCH]], float)
        mm = pchip_slopes(tt, vv)
        grid = np.linspace(tt[0], tt[-1], max(121, int(np.ceil((tt[-1] - tt[0]) * 120))))
        outro_pitch[h] = (grid[0], grid[-1], grid, np.asarray([hermite_eval(tt, vv, mm, t)[0] for t in grid]))
    # Riding fingers start from the frame before; a chunk is entered WARM_FRAMES early so its first frames
    # continue the solution its neighbour chunk ends on.
    previous = {h: {} for h in 'LR'}
    for t in np.asarray(times)[0] - np.arange(WARM_FRAMES, 0, -1) / FRAME_RATE:
        if t >= 0:
            warm = {p: max(note.depth(t) for note in by_pitch[p]) for p in pitches}
            for h in 'LR':
                frame_hand(t, _ctx['paths'][h], tracks[h], notes, _ctx['waypoints'], _ctx['solvers'][h], _ctx['ik'][h],
                           {p: d for p, d in warm.items() if d > 0}, previous[h])
    for i, t in enumerate(times):
        depths = {}
        for j, p in enumerate(pitches):
            d = max(note.depth(t) for note in by_pitch[p])
            out['key_depth'][i, j] = d
            if d > 0:
                depths[p] = d
        spine_q, pre, collar, slide = (None, None, {}, None) if _ctx['torso'] is None else _ctx['torso'].pose(t)
        segments = {}
        if spine_q is not None:
            out['spine'][i] = [spine_q[name] for name in spine_names]
            out['torso'][i] = out['spine'][i, 0]
            out['slide'][i] = slide
        for h, q in collar.items():
            out[f'{h}_collar_q'][i] = q
        for h in 'LR':
            # A hand resting on the lap rides the leg. The seat's slide (world +x) moves the thigh; IK used
            # to cancel it by pinning the wrist in world space, so the leg slid out from under a still hand
            # and the pads sank in / floated by up to 3 cm. Follow a fraction of the slide (the leg's measured
            # share) only while this hand is on the lap, fading in over LAP_SLIDE_FADE around the landing so
            # the return itself is untouched. A hand playing the keys keeps its world-fixed wrist.
            rest_slide = None
            if slide is not None:
                t1 = tracks[h][DIGITS[0]]
                before = min(1., max(0., (getattr(t1, 'rest_start', None) - t) / LAP_SLIDE_FADE + 1.)) \
                    if getattr(t1, 'rest_start', None) is not None else 0.
                after = min(1., max(0., (t - t1.rest_end) / LAP_SLIDE_FADE + 1.)) \
                    if getattr(t1, 'rest_end', None) is not None else 0.
                follow = max(before, after)
                if follow > 0.:
                    rest_slide = np.array([LAP_SLIDE_FOLLOW * (float(slide[0]) - slide_base) * follow, 0., 0.])
            arm = rig.arms[h]
            swing = _ctx.get('elbow_swing')
            phi = float(hermite_eval(swing[h]['times'], swing[h]['angles'][:, None],
                                     _ctx['elbow_swing_slopes'][h], t)[0]) if swing is not None and h in swing else None
            solver = _ctx['solvers'][h]
            # One continuous palm curve: the draw stays on the lap plane; then hang, flatten, the optional
            # tension peak and the playing plane are blended without ease-to-zero stops at each mark.
            pitch = None
            path_h = _ctx['paths'][h]
            leave_h = getattr(path_h, 'trip_leave', None)
            if h in outro_pitch and t >= outro_pitch[h][0]:
                # closing gesture: the palm eases from the playing plane back to the lap, in step with
                # the wrist's single continuous return (no palm still pointing at the keys on the lap).
                _, _, o_t, o_v = outro_pitch[h]
                if getattr(tracks[h][DIGITS[0]], 'outro_gathered', None) is not None:
                    pitch = _closing_palm(path_h, t)
                else:
                    pitch = float(np.interp(t, o_t, o_v))
            elif leave_h is not None:
                if t <= leave_h or h not in intro_pitch:
                    pitch = LAP_PITCH
                else:
                    de, top, pitch_t, pitch_v = intro_pitch[h]
                    pitch = LAP_PITCH if t < de else float(np.interp(t, pitch_t, pitch_v))
                    if de < t < top:             # low-hand peel: clearance has priority over the ideal curve
                        z = float(path_h(t)[2])
                        free = math.degrees(math.asin(min(1., max(0., (z - PEEL_SKIN - .004) / PEEL_REACH))))
                        pitch = max(pitch, -free)
            # A hand that never plays keeps its resting palm (no intro and no outro curve: neither leave_h
            # nor outro_pitch exists for it). Set the SAME value used below by arm.pose, so the recorded
            # pitch and the mesh agree (native_check's pad difference stays at its normal baseline).
            if pitch is None:
                pitch = LAP_PITCH
            # The palm's pendulum. In free time the hand turns up a few degrees on the rise and gives them back
            # on the way down (the wrist's slight effort, the hand's own inertia) -- keyed to the same up-and-back
            # the wrist does, not a vibration, and never while a finger is down: a palm turning under a held
            # pad drags it along its key.
            pend = 0.
            rise = getattr(path_h, 'pitch_rise', None)
            if rise is not None and leave_h is not None and t > leave_h:
                if not any(s is not None and s.touch <= t <= s.release
                           for s in (tracks[h][d].stroke(t) for d in DIGITS)):
                    pend = PITCH_PENDULUM * float(np.interp(t, rise[0], rise[1]))
            # ``pitch`` is only handed to the fingers when the pendulum is on: everywhere else the palm the IK
            # has always been solved with (the playing plane) is kept, so the intro and closing gestures do
            # not change under this.
            vector, values, error = frame_hand(t, _ctx['paths'][h], tracks[h], notes, _ctx['waypoints'],
                                               _ctx['solvers'][h], _ctx['ik'][h], depths, previous[h],
                                               rest_slide=rest_slide,
                                               pitch=(pitch + pend) if pend else None)
            out[f'{h}_wrist'][i] = vector
            out[f'{h}_contact_error'][i] = error
            # The key goes down as far as its finger takes it: a stretched hand that cannot reach the keybed
            # (an octave at the hand's limit) shows a key pressed a little less, not a finger above its key.
            palm = solver.palm(vector[:3], vector[3], vector[4], (pitch + pend) if pend else None)
            pitch += pend
            out[f'{h}_pitch'][i] = pitch
            for d in DIGITS:
                s = tracks[h][d].stroke(t)
                if s is None or not s.touch <= t <= s.release:
                    continue
                out[f'{h}_riding'][i, d - 1] = True
                p = notes[s.note].pitch
                reached = solver.regions.rows[p]['top'] - solver.contact_point(solver.pad_world(d, values, palm))[2]
                j = pitches.index(p)
                out['key_depth'][i, j] = min(out['key_depth'][i, j], max(reached, 0.))
            try:
                pose = arm.pose(vector[:3], vector[3], vector[4], None if pre is None else pre[h], phi, pitch=pitch)
                out[f'{h}_arm_q'][i] = [pose['local'][arm.names[k]] for k in ('upper', 'elbow', 'wrist', 'twist')]
                out[f'{h}_arm_world'][i] = [pose['shoulder'], pose['elbow']]
                out[f'{h}_palm_deg'][i] = pose['palm_forearm_deg']
            except ValueError:          # wrist beyond the unstretched arm
                # Never leave the arm's rotations at zero (Blender collapses such a bone and the arm vanishes):
                # stretch the arm toward the wrist instead; the hand stops just short of where it was sent.
                out[f'{h}_arm_ok'][i] = False
                shoulder_pre = arm.pre_upper_world if pre is None else pre[h]
                reach = shoulder_pre[:3, 3]
                aim = np.asarray(vector[:3], float) - reach
                stretched = reach + aim * (ARM_STRETCH * (arm.l1 + arm.l2) / np.linalg.norm(aim))
                pose = arm.pose(stretched, vector[3], vector[4], shoulder_pre, pitch=pitch)
                out[f'{h}_arm_q'][i] = [pose['local'][arm.names[k]] for k in ('upper', 'elbow', 'wrist', 'twist')]
                out[f'{h}_arm_world'][i] = [pose['shoulder'], pose['elbow']]
                out[f'{h}_palm_deg'][i] = pose['palm_forearm_deg']
            caps = _ctx['capsules'][h]
            segments[h] = caps.segments(values, pose['armature'])
            out[f'{h}_finger_overlap'][i] = finger_pairs(caps, segments[h])
            out[f'{h}_pointing'][i] = pointing(segments[h])
            out[f'{h}_raised'][i] = raised_between(segments[h], {d for d in DIGITS if out[f'{h}_riding'][i, d - 1]})
            spacing = _ctx['ik'][h].spacing
            whole = {d: spacing.positions(d, values) for d in (2, 3, 4, 5)}
            out[f'{h}_finger_gap'][i] = [float((whole[d + 1] - whole[d]).min()) for d in (2, 3, 4)]
            for d in DIGITS:
                v = values[d]
                out[f'{h}_values'][i, d - 1] = v
                root_q, e1, e2 = rig.fingers[h][d].local_rotations(v)
                out[f'{h}_root_q'][i, d - 1] = root_q
                out[f'{h}_child_euler'][i, d - 1] = [e1, e2]
            lap_worst = 0.
            if _ctx['heightfield'] is not None or lap_ref[h] is not None:
                worst = 0.
                for d in DIGITS:
                    points = rig.clearance_points(h, d, pose['armature'], values)
                    if lap_ref[h] is not None:
                        # How far this finger's skin has sunk below where the fitted lap pose puts it: the
                        # rest shape is measured ON the leg (lap_contact), so any lower is inside it. A
                        # height field of the leg cannot answer this -- the skirt overhangs the resting
                        # hand, so a column's top surface is not the surface the palm is under.
                        lap_worst = max(lap_worst, float(lap_ref[h][d] - points[:, 2].min()))
                    if _ctx['heightfield'] is None:
                        continue
                    s = tracks[h][d].stroke(t)
                    if s is None or not s.touch <= t <= s.release:
                        worst = max(worst, float(_ctx['heightfield'].penetration(points, depths).max()))
                        continue
                    hard, wanted = _ctx['heightfield'].split(points, depths, notes[s.note].pitch)
                    worst = max(worst, float(hard.max()))
                    for p, depth in wanted.items():
                        k = p - LOWEST_KEY
                        out[f'{h}_brush'][i, k] = max(out[f'{h}_brush'][i, k], depth)
                out[f'{h}_penetration'][i] = worst
            out[f'{h}_lap_penetration'][i] = max(lap_worst, 0.)
        out['hands_gap'][i] = hands_gap(_ctx['capsules'], segments)
    return out


LOWEST_KEY, KEYS = 21, 88
BRUSH_HOLD = .04                 # a pushed key stays down this long around the finger's touch (s): no flicker
BRUSH_SPEED = .03                # fastest a pushed key moves (m/s): 3 mm in .1 s; the softest sounding stroke is ~.09


def brushed_keys(frames, rate):
    """Unplayed keys a resting finger pushes down (checks.BRUSH_DEPTH): per key the deepest push either hand
    wants, capped, then held over BRUSH_HOLD each side and slew-limited to BRUSH_SPEED (both only ever deepen it).
    Writes the pushes into the key animation and counts what they do not absorb as penetration."""
    wanted = {h: frames.pop(f'{h}_brush') for h in 'LR'}
    # capped first: a deeper want is penetration, and slewed down from its own depth it would hold the key for
    # up to (want - BRUSH_DEPTH) / BRUSH_SPEED around a single frame
    want = np.minimum(np.maximum(wanted['L'], wanted['R']), BRUSH_DEPTH)
    push = want.copy()
    for k in range(1, int(round(BRUSH_HOLD * rate)) + 1):
        push[k:] = np.maximum(push[k:], want[:-k])
        push[:-k] = np.maximum(push[:-k], want[k:])
    step = BRUSH_SPEED / rate
    for i in range(1, len(push)):                       # slew limit as an upper envelope: never shallower
        push[i] = np.maximum(push[i], push[i - 1] - step)
    for i in range(len(push) - 2, -1, -1):
        push[i] = np.maximum(push[i], push[i + 1] - step)
    for h in 'LR':
        excess = np.maximum(wanted[h] - push, 0.).max(axis=1)
        frames[f'{h}_penetration'] = np.maximum(frames[f'{h}_penetration'], excess)
    used = np.where(push.max(axis=0) > 1e-5)[0]
    old = list(frames['key_pitches'])
    pitches = sorted(set(old) | {int(k) + LOWEST_KEY for k in used})
    depth = np.zeros((len(push), len(pitches)))
    for j, p in enumerate(pitches):
        if p in old:
            depth[:, j] = frames['key_depth'][:, old.index(p)]
        depth[:, j] = np.maximum(depth[:, j], push[:, p - LOWEST_KEY])
    frames['key_pitches'], frames['key_depth'] = np.array(pitches), depth
    moving = np.abs(np.diff(push, axis=0)).max() * rate if len(push) > 1 else 0.
    events = int(((push[1:] > 1e-5) & (push[:-1] <= 1e-5)).sum())
    return dict(keys=len(used), pushes=events, seconds_pushed=round(float((push > 1e-5).any(axis=1).sum() / rate), 2),
                deepest_mm=round(float(push.max() * 1000), 2), fastest_m_s=round(float(moving), 3),
                silent=bool(push.max() <= BRUSH_DEPTH + 1e-9))


def despike(frames):
    """Single-instant solver misses (a few frames above ``SPIKE_ERROR``) are bridged from the frames around
    them. Then the same bridging for a second family the error does not catch: a short run whose values jump
    far off and land back where the run started (the per-frame IK flickering between two solutions of its
    warm start -- measured: a finger flashing PIP -23 -> -34 -> -25 at its joint-limit corner while both
    neighbours nearly coincide). Real motion never comes back: its two neighbours differ by the flight."""
    fixed = {}
    for h in 'LR':
        bad = frames[f'{h}_contact_error'] > SPIKE_ERROR
        n = len(bad)
        count = 0
        i = 0
        while i < n:
            if not bad[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and bad[j + 1]:
                j += 1
            if j - i + 1 <= SPIKE_FRAMES and i > 0 and j < n - 1:
                w = ((np.arange(i, j + 1) - (i - 1)) / (j - i + 2))
                for key in ('wrist', 'values', 'root_q', 'child_euler', 'arm_q'):
                    a, b = frames[f'{h}_{key}'][i - 1], frames[f'{h}_{key}'][j + 1]
                    if key in ('root_q', 'arm_q'):
                        b = np.where((a * b).sum(axis=-1, keepdims=True) < 0, -b, b)
                    shape = (-1,) + (1,) * a.ndim
                    mixed = a[None] * (1 - w.reshape(shape)) + b[None] * w.reshape(shape)
                    if key in ('root_q', 'arm_q'):
                        mixed /= np.linalg.norm(mixed, axis=-1, keepdims=True)
                    frames[f'{h}_{key}'][i:j + 1] = mixed
                frames[f'{h}_contact_error'][i:j + 1] = 0.
                count += j - i + 1
            i = j + 1
        # the flicker family: a short run that jumps off the trajectory and lands back on it. The reference is
        # the trajectory extrapolated from the two frames before the run (the trajectory itself moves, so an
        # absolute "back where it started" is false on a slope): a real flight keeps moving away, a flicker's
        # landing misses the forecast by a fraction of how far it jumped (measured: PIP -23 -> -34 -> -25 in
        # three frames). A landing only counts when the motion settles afterwards (a trill lands too, and
        # keeps going).
        values = frames[f'{h}_values']
        i = 5
        while i < n - 2:
            slope = values[i - 1] - values[i - 2]
            offs = np.arange(-4, SPIKE_FRAMES + 2)[:, None, None]
            pred = values[i - 1] + offs * slope[None]          # the trajectory's forecast, before and after i
            back = float(np.abs(values[i - 5:i - 1] - pred[:4]).max())   # was the trajectory quiet before?
            dev0 = float(np.abs(values[i] - pred[4]).max())
            if dev0 < SPIKE_LEAVE or back >= SPIKE_LEAVE:
                i += 1
                continue
            cap = min(i + SPIKE_FRAMES, n - 2)
            j = i
            while j < cap:
                land = float(np.abs(values[j + 1] - pred[4 + (j + 1 - i)]).max())
                quiet = float(np.abs(values[j + 2] - values[j + 1]).max())
                if dev0 > SPIKE_JUMP and land < .55 * dev0 + .04 and quiet < SPIKE_LEAVE:
                    break           # landed back on a quiet trajectory: a flicker, not a trill or a flight
                j += 1
                dev0 = max(dev0, float(np.abs(values[j] - pred[4 + (j - i)]).max()))
            if j < cap:
                land = float(np.abs(values[j + 1] - pred[4 + (j + 1 - i)]).max())
                quiet = float(np.abs(values[j + 2] - values[j + 1]).max())
            else:
                land, quiet = 1., 1.
            if j < cap and dev0 > SPIKE_JUMP and land < .55 * dev0 + .04 and quiet < SPIKE_LEAVE:
                w = ((np.arange(i, j + 1) - (i - 1)) / (j - i + 2))
                for key in ('values', 'root_q', 'child_euler'):
                    aa, bb = frames[f'{h}_{key}'][i - 1], frames[f'{h}_{key}'][j + 1]
                    if key == 'root_q':
                        bb = np.where((aa * bb).sum(axis=-1, keepdims=True) < 0, -bb, bb)
                    shape = (-1,) + (1,) * aa.ndim
                    mixed = aa[None] * (1 - w.reshape(shape)) + bb[None] * w.reshape(shape)
                    if key == 'root_q':
                        mixed /= np.linalg.norm(mixed, axis=-1, keepdims=True)
                    frames[f'{h}_{key}'][i:j + 1] = mixed
                count += j - i + 1
                i = j + 2
            else:
                i += 1
        fixed[h] = count
    return fixed


def smoothness(frames, rate):
    """What the eye reads as jitter and snapping: finger reversals, wrist speed and turn rate."""
    out = {}
    for h in 'LR':
        vel = np.diff(frames[f'{h}_values'], axis=0)
        reversal = ((vel[1:] * vel[:-1]) < 0) & (np.abs(vel[1:]) > .3) & (np.abs(vel[:-1]) > .3)
        zigzag = reversal[:-1] & reversal[1:]          # back and forth on consecutive frames: jitter
        w = frames[f'{h}_wrist']
        speed = np.linalg.norm(np.diff(w[:, :3], axis=0), axis=1) * rate
        turn = np.abs(np.diff(w[:, 3:], axis=0)).max(axis=1) * rate
        out[h] = dict(finger_jitter_frames=round(float(zigzag.any(axis=(1, 2)).mean()), 4),
                      finger_reversal_frames=round(float(reversal.any(axis=(1, 2)).mean()), 4),
                      wrist_speed_p99=round(float(np.percentile(speed, 99)), 3), wrist_speed_max=round(float(speed.max()), 3),
                      turn_rate_p99_deg_s=round(float(np.percentile(turn, 99)), 1),
                      turn_rate_max_deg_s=round(float(turn.max()), 1),
                      still_fraction=round(float((speed < .02).mean()), 3))
    return out


def rig_finger_spacing(rig_path):
    """The rig hand's relaxed fingertip spacing per finger (index to little finger), for the planner's stretch."""
    solvers = hand_solvers(Rig(rig_path))
    from .motion import Spacing
    gaps = [sum(Spacing(s).gaps.values()) / .85 / 3 for s in solvers.values()]
    return float(np.mean(gaps))


def sideways(frames, notes, rate):
    """Share of presses where the playing finger swings sideways (the hand is not over its key)."""
    swing = np.array([abs(frames[f'{n.hand}_values'][min(int(round((n.onset - frames['times'][0]) * rate)), len(frames['times']) - 1), n.digit - 1, 0])
                      for n in notes.values()])
    limit = np.array([SWING_LATERAL.get(n.digit, 24.) - 1. for n in notes.values()])
    return dict(over_15_deg=round(float((swing > 15).mean()), 3), at_limit=round(float((swing >= limit).mean()), 3),
                median_deg=round(float(np.median(swing)), 1))


def riding_sideways(frames):
    """Sideways swing of fingers 2..5 over every frame they ride their key (thumb excluded: it reaches by
    technique): share above 10 and 15 deg, and at the joint limit. The masks stay in ``frames`` (saved for
    quality_report's sustained-frame split)."""
    out = {}
    for h in 'LR':
        riding = frames[f'{h}_riding']
        direction = np.abs(frames[f'{h}_pointing'])
        thumb = np.abs(frames[f'{h}_values'][:, 0, 0])[riding[:, 0]]
        swing = np.abs(frames[f'{h}_values'][:, 1:, 0])[riding[:, 1:]]
        limit = np.array([SWING_LATERAL.get(d, 25.) - 1. for d in (2, 3, 4, 5)])
        at_limit = (np.abs(frames[f'{h}_values'][:, 1:, 0]) >= limit)[riding[:, 1:]]
        pointed = direction[:, 1:][riding[:, 1:]]
        p99 = (lambda a: round(float(np.percentile(a, 99)), 1) if len(a) else 0.)
        out[h] = dict(over_10_deg=round(float((swing > 10).mean()), 3) if len(swing) else 0.,
                      over_15_deg=round(float((swing > 15).mean()), 3) if len(swing) else 0.,
                      at_limit=round(float(at_limit.mean()), 3) if len(at_limit) else 0.,
                      p99_deg=p99(swing),
                      pointing_over_25_deg=round(float((pointed > 25).mean()), 3) if len(pointed) else 0.,
                      pointing_p99_deg=p99(pointed),
                      thumb_over_35_deg=round(float((thumb > 35).mean()), 3) if len(thumb) else 0.,
                      thumb_p99_deg=round(float(np.percentile(thumb, 99)), 1) if len(thumb) else 0.)
    return out


OVERLAP_REPORTED = .002          # m: neighbour fingers resting side by side touch (0 .. -3 mm); deeper is reported
RAISED_REPORTED = .02            # m: a free finger this far above both pressing neighbours


def finger_bodies(frames, rate):
    """3D checks (contacts.py): neighbour finger bodies running into each other, the hands touching, a finger
    raised between two pressing neighbours. Removes the per-frame arrays from ``frames``."""
    times = frames['times']
    out = {}
    for h in 'LR':
        overlap = frames.pop(f'{h}_finger_overlap')
        raised = frames.pop(f'{h}_raised')
        deep = overlap.max(axis=1) > OVERLAP_REPORTED
        worst = np.argsort(-overlap.max(axis=1))[:200]
        moments = []
        for i in worst:
            if len(moments) < 6 and all(abs(times[i] - m[0]) > .5 for m in moments):
                moments.append((round(float(times[i]), 2), ('1-2', '2-3', '3-4', '4-5')[int(overlap[i].argmax())],
                                round(float(overlap[i].max()) * 1000, 1)))
        out[h] = dict(overlap_frames={pair: int((overlap[:, k] > OVERLAP_REPORTED).sum())
                                      for k, pair in enumerate(('1-2', '2-3', '3-4', '4-5'))},
                      overlap_share=round(float(deep.mean()), 4), worst=moments,
                      raised_frames=int((raised > RAISED_REPORTED).sum()), raised_max_mm=round(float(raised.max()) * 1000, 1))
    gap = frames.pop('hands_gap')
    out['hands'] = dict(touching_frames=int((gap < 0).sum()), closest_mm=round(float(gap.min()) * 1000, 1),
                        closest_at=round(float(times[int(gap.argmin())]), 2))
    return out


def crossings(frames, rate):
    """Neighbour fingers crossed seen from above (whole finger, motion.Spacing.positions): share of frames
    per pair, and the longest episodes (start s, seconds, deepest mm) per hand. Removes the gaps from ``frames``."""
    out = {}
    times = frames['times']
    for h in 'LR':
        gap = frames.pop(f'{h}_finger_gap')
        crossed = gap.min(axis=1) < 0
        episodes, start = [], None
        for i, c in enumerate(np.r_[crossed, False]):
            if c and start is None:
                start = i
            elif not c and start is not None:
                episodes.append((round(float(times[start]), 2), round((i - start) / rate, 2),
                                 round(float(gap[start:i].min()) * 1000, 1)))
                start = None
        out[h] = dict(pairs={pair: round(float((gap[:, k] < 0).mean()), 4) for k, pair in enumerate(('2-3', '3-4', '4-5'))},
                      frames=int(crossed.sum()), episodes=len(episodes),
                      longest=sorted(episodes, key=lambda e: -e[1])[:6])
    return out


def piece_inputs(folder):
    """Planner inputs of a piece: a folder with timeline.json and demands.json (inputs/build_piece.py)."""
    folder = Path(folder).resolve()
    return {k: str(folder / f'{k}.json') for k in PIECE_FILES}


def score_hands(source, rows):
    """Where the source names each note's hand (a hand-split score): how often the plan agrees."""
    known = {n['id']: n['hand'] for n in source if n.get('hand') in ('L', 'R')}
    if not known:
        return None
    played = [r for r in rows if r['id'] in known]
    same = sum(r['hand'] == known[r['id']] for r in played)
    return dict(notes_with_score_hand=len(known), played=len(played), same_hand=same,
                agreement=round(same / max(len(played), 1), 4),
                omitted_by_score_hand={h: sum(1 for i, k in known.items() if k == h) -
                                          sum(1 for r in played if known[r['id']] == h) for h in 'LR'})


def chunks(times, count):
    return [c for c in np.array_split(times, count) if len(c)]


def _mark(name, t0, stages):
    """The stage's wall clock, recorded and printed as it ends: a long run says where it spent the time, and a run
    that dies mid-piece still says what it finished before that."""
    dt = time.perf_counter() - t0
    stages[name] = dt
    print(f'[stage] {name} {dt:.1f}s', flush=True)
    return dt


def run(args):
    started = time.perf_counter()
    stages = {}
    workers = args.workers
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    table, _ = load_or_compute(RIG, workers)
    _mark('rig_pair_table', t0, stages)

    # 1. note planning on the rig's own feasibility
    t0 = time.perf_counter()
    inputs = piece_inputs(args.inputs)
    raw = load_raw(inputs)
    # the piece's own motion settings (optional): {"lift": share of the strike lifts, 1 = default}
    settings = Path(args.inputs) / 'motion.json'
    if settings.exists():
        from . import motion
        motion.LIFT_SCALE = float(json.loads(settings.read_text(encoding='utf-8')).get('lift', 1.))
        print(f'[motion] {settings.name}: lift x{motion.LIFT_SCALE}', flush=True)
    reach_x = {h: reach_window(s, Rig(RIG).data.get('torso'), RELAXED) for h, s in hand_solvers(Rig(RIG)).items()}
    print('[planning] reach (key x, m): ' + ', '.join(f'{h} {lo:.3f}..{hi:.3f}' for h, (lo, hi) in reach_x.items()), flush=True)
    planner_kwargs = dict(beam_width=args.beam, outer_voice_bonus=1.5, pair_table=table, pair_strain=pair_awkwardness(table, workers),
                          rest_x={h: float(Rig(RIG).neutral_wrist[h][0]) for h in 'LR'}, reach_x=reach_x,
                          weights=Weights(**({} if args.finger_spacing is None else dict(finger_spacing=args.finger_spacing)),
                                          **({} if args.lift_speed is None else dict(lift_speed=args.lift_speed))))
    stages['planning'] = stages['anchor_poses'] = stages['chord_shapes'] = 0.
    shapes, anchor_cache, shape_cache, starts = {}, {}, {}, {}
    for shape_round in range(SHAPE_ROUNDS):
        print(f'[planning] round {shape_round + 1}', flush=True)
        penalty = {k: AWKWARD_COST * (u - AWKWARD_FREE) for k, u in shapes.items() if u > AWKWARD_FREE}
        planned = plan_parallel(build, (inputs, args.end), dict(planner_kwargs, shape_penalty=penalty), workers=workers)
        print(f'[stage] plan_round {shape_round + 1} {time.perf_counter() - t0:.1f}s', flush=True)
        planner, score = planned['planner'], planned['planner'].score
        rows = plan_notes(planned['state'], score)
        problems = validate(rows, score, raw['keyboard'], raw['envelopes'], raw['radii'], planner.w.hard_speed)
        wrist_x = {(d.group, h): p.x for d in planned['state'].decisions() for h, p in d.poses}
        stages['planning'] += time.perf_counter() - t0

        # 2. anchor poses (keyframes) on the chosen path
        t0 = time.perf_counter()
        notes = note_motions(rows, score)
        anchors = solve_anchors(build_anchors(notes, wrist_x), notes, RIG, workers, cache=anchor_cache)
        stages['anchor_poses'] += time.perf_counter() - t0
        # 2a. the awkwardness of the shapes this plan chose; shapes not priced yet send the piece back to planning
        t0 = time.perf_counter()
        chosen = awkward_shapes(anchors, notes, dict(rig_path=str(RIG), notes=notes, checks=False), workers, shape_cache, starts)
        stages['chord_shapes'] += time.perf_counter() - t0
        unpriced = {k: u for k, u in chosen.items() if k not in shapes}
        shapes.update(unpriced)
        print(f'chord shapes round {shape_round + 1}: {len(chosen)} shapes, {len(unpriced)} new, '
              f'{sum(u > AWKWARD_FREE for u in unpriced.values())} of them awkward', flush=True)
        if not any(u > AWKWARD_FREE for u in unpriced.values()):
            break
        t0 = time.perf_counter()

    # 2b. free early releases (identical sound), then orientation coherence between neighbouring anchors
    t0 = time.perf_counter()
    stroke_of = strokes(notes)
    for n in notes.values():
        n.sound_release = n.release
    score_pedal = pedal_union(damper_intervals(raw['timeline']), [])
    pedal_changes = score_pedal[np.isfinite(score_pedal)]          # every damper down and up of the score
    lifts = [free_releases([a for a in anchors if a.hand == h], notes, stroke_of, args.motion_cut, pedal_changes=pedal_changes)
             for h in 'LR']
    freed = dict(sound_identical=sum(l[0] for l in lifts), audible=sum(l[1] for l in lifts), pedalled=sum(l[2] for l in lifts))
    relaxed = relax_anchors(anchors, notes)
    if relaxed:
        anchors = anchors + solve_anchors(relaxed, notes, RIG, workers, cache=anchor_cache)
        for a in anchors:   # a key a repair lifted earlier is no longer held
            a.held = tuple(k for k in a.held if notes[k].held_at(a.time))
    stroke_of = strokes(notes)
    base = dict(rig_path=str(RIG), notes=notes, strokes=stroke_of, checks=not args.no_checks)
    # The anchors were solved square to the keys; the first settle turns each toward the hand's natural
    # orientation on its forearm (plus what its keys ask of it), smoothed over the neighbours.
    posture_solvers = performer_solvers(Rig(RIG))
    natural = {id(a): natural_orientation(posture_solvers[a.hand], a.pose.wrist) for a in anchors}

    def cohere(anchors, natural, rounds, pool):
        """Settle each anchor toward ``natural`` (its hand's relaxed orientation), smoothed over the neighbours;
        an anchor whose settle succeeded keeps its own orientation and only takes the neighbour's wrist pull."""
        turned = set()
        for round_ in range(rounds):
            order, jobs = [], []
            for h in 'LR':
                mine = sorted((a for a in anchors if a.hand == h), key=lambda a: a.time)
                offsets = [np.zeros(2) if id(a) in turned else natural.get(id(a), np.zeros(2)) for a in mine]
                for a, target in zip(mine, neighbour_targets(mine, offsets)):
                    order.append(a)
                    contacts = a.contacts(notes)
                    # a stretched hand searched once, in the first round; the next rounds settle from what it found
                    # a shape of two or more keys is also settled from a free start in the first round: the least
                    # awkward is kept (its anchor's pose may sit in a strained basin)
                    jobs.append((h, contacts, vector_of(a.pose), a.pose.values, target, ORIENTATION_PULLS,
                                 max(a.pose.contact_error, .0005), round_ == 0 and stretched(c.pitch for c in contacts),
                                 (starts[h, tuple(contacts)],) if round_ == 0 and len(contacts) > 1 else ()))
            for a, pose in zip(order, progress_map(pool, _settle, jobs, f'coherence {round_ + 1}/{rounds}')):
                if pose is not None:
                    a.pose = pose
                    turned.add(id(a))

    with ProcessPoolExecutor(workers, initializer=_init, initargs=(base,)) as pool:
        free_starts(pool, [(a.hand, a.contacts(notes)) for a in anchors], starts)
        cohere(anchors, natural, COHERENCE_ROUNDS, pool)
    _mark('coherence', t0, stages)

    # 3. wrist paths, refined until the held keys stay under the fingers
    t0 = time.perf_counter()
    rig = Rig(RIG)
    duration = (args.end if args.end is not None else float(raw['timeline']['duration'])) + TAIL_SECONDS
    # The closing gesture lives AFTER the last key release and needs its own window. The instrument's tail is
    # generous (TAIL_SECONDS), but --end or a long final release can still push the last note up against
    # duration; guarantee room for the shared close (a possible wait, one full return, a still beat).
    music_end = max([n.release for n in notes.values()] + [0.])
    duration = max(duration, music_end + .12 + OUTRO_T + .15)
    intro = max(0., INTRO_SECONDS - min(n['start'] for n in raw['timeline']['notes']))
    rest_solvers = performer_solvers(rig)
    rest = {}
    for h in 'LR':      # on the lap: the palm on the thigh (rig_export.lap), the hand turned in a little
        wrist = np.array(rig.data['hands'][h]['lap'], dtype=float) + REST_OFFSET
        orientation = np.array(natural_orientation(rest_solvers[h], wrist, rest_solvers[h].arm.pre_upper_world[:3, 3],
                                                   keys=False), dtype=float)
        orientation[0] += REST_YAW * (1. if h == 'R' else -1.)
        rest[h] = np.r_[wrist, orientation]
    if LAP_CONTACT:
        # The lap point is a point on the thigh, and the hand laid flat on it neither fits the slope nor stays out
        # of it: the little finger sinks, the index finger hangs. The fitted place and finger shape (lap_contact,
        # measured on this seat) put every pad on the leg.
        from .lap_contact import place
        for h in 'LR':
            place(rest, h)
        print('[lap contact] wrist %s roll %s' % (
            {h: [round(float(v) * 1000., 1) for v in rest[h][:3] - np.array(rig.data['hands'][h]['lap'], dtype=float)]
             for h in 'LR'}, {h: round(float(rest[h][4]), 1) for h in 'LR'}), flush=True)
    hand_notes = {h: {k: n for k, n in notes.items() if n.hand == h} for h in 'LR'}

    def closing_starts():
        """When each hand leaves its last key. Two hands whose last notes are within ``OUTRO_WAIT`` close
        together: the earlier one waits in a slow 1-2 cm rise while the other finishes, then both start a
        hair apart (0.05-0.08 s) so the pair never looks like a mirror. A wider gap closes each on its own."""
        ends = {h: max((n.release for n in hand_notes[h].values()), default=None) for h in 'LR'}
        live = [h for h in 'LR' if ends[h] is not None]
        starts, variants = {}, {}
        if not live:
            return starts, variants
        together = len(live) == 2 and abs(ends['L'] - ends['R']) <= OUTRO_WAIT
        if together:
            # The hand that finishes LAST leaves on its own beat; the earlier one has been waiting and
            # starts OUTRO_START_GAP before it, so the two returns overlap without looking mirrored. The
            # wait itself is built by wrist_path from this hand's own release to this start.
            late = max(live, key=lambda h: ends[h])
            early = 'L' if late == 'R' else 'R'
            starts[late] = ends[late] + .09
            starts[early] = max(ends[early] + .01, starts[late] - OUTRO_START_GAP)
        else:
            for h in live:
                starts[h] = ends[h] + .04              # one hand alone: leave as soon as its music ends
        for h in live:                                 # never a 100% mirrored pair
            variants[h] = 1. if h == 'R' else -1.
        return starts, variants

    # The hand(s) that open the piece share one intro: one breath of rest, then the lift / pause / reach /
    # press, landing on the piece's first note. A hand whose first note comes later (the user's spec,
    # 2026-11-17: "如果起手是单手那么也该是单手抽出另一手在要弹之前的一百多帧再考虑动") stays ON THE LAP and
    # only stirs ~105 frames (30 fps) before its OWN first note, then makes its own quiet approach (one
    # calm arc, no ceremony -- the user's pick) into its first stroke -- it finds its keys by its own
    # notes, never seated early over music it has not reached.
    first_all = min(n['start'] for n in raw['timeline']['notes'])
    intro_leave_path = 1.2 - intro
    intro_land_path = first_all
    LATE_NOTICE = 3.5                # the resting hand's whole window: ~105 frames at 30 fps

    def hand_overrides(h):
        own = min((n.touch for n in hand_notes[h].values()), default=first_all)
        if own <= first_all + .05:   # this hand plays the opening: the shared intro
            return intro_leave_path, intro_land_path, False
        # late: still on the lap until ~105 frames before its own first note, then its own quiet approach
        return own - LATE_NOTICE, own, True

    def paths_for(anchors):
        t_p = time.perf_counter()
        starts, variants = closing_starts()
        out = {}
        for h in 'LR':      # timing probe: measured -- a 2-proc parallel solve loses to pickling the anchors (13.1s vs 9.0s)
            t_h = time.perf_counter()
            lv, ld, quiet = hand_overrides(h)
            out[h] = wrist_path([a for a in anchors if a.hand == h], hand_notes[h], stroke_of, rest[h], duration, -intro,
                                leave_override=lv, land_override=ld, quiet=quiet,
                                outro_start=starts.get(h), outro_variant=variants.get(h, 0.))
        print('[closing] starts %s' % {h: round(starts[h], 2) for h in starts}, flush=True)
        print(f'[wrist solve] serial {time.perf_counter() - t_p:.1f}s ({len(anchors)} anchors)', flush=True)
        return out

    def rebuild(anchors):
        """The paths and the contacts they place, rebuilt together -- the one way back from a changed anchor set.

        Three stages change ``anchors`` after the first solve: the refine's intermediate poses, the re-aim, and
        the attitude round. Each must read the paths its own anchors define and the waypoints those paths place,
        never one of the two from the round before (the waypoints feed every later settle, and a stale set aims
        the fingers at keys the wrist no longer reaches). The pair is returned together so no caller can rebuild
        one without the other."""
        p = paths_for(anchors)
        return p, contact_waypoints(placed(anchors, p, posture_solvers, notes), notes)

    with ProcessPoolExecutor(workers, initializer=_init, initargs=(base,)) as pool:
        played = [a for a in anchors if a.pose is not None]
        parts = progress_map(pool, _family, [list(c) for c in np.array_split(np.array(played, dtype=object), max(1, len(played) // 2)) if len(c)],
                             'depth families')
        for a, s in zip(played, [s for part in parts for s in part]):
            a.family = s
    # a small pool kept for the wrist-path solves: the L/R solves are independent and each takes seconds,
    # so they run side by side wherever no other pool is already warm (the solves pass all state per job)
    paths, waypoints = rebuild(anchors)
    samples = np.arange(0., duration, REFINE_STEP)
    intermediate = dropped_snaps = left_early = 0
    # Each round's check measures the paths the rounds before left; one more check after the last measures the
    # result (refine_history: what every further round bought, from a single run).
    refine_history = []
    for refine_round in range(args.refine_rounds + 1):
        last = refine_round == args.refine_rounds
        print(f'[wrist paths] ' + ('final check' if last else f'refine round {refine_round + 1}/{args.refine_rounds}'), flush=True)
        t_r = time.perf_counter()
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(dict(base, paths=paths, waypoints=waypoints),)) as pool:
            jobs = [(h, c) for h in 'LR' for c in chunks(samples, workers * 2)]
            errors = {h: [] for h in 'LR'}
            t_c = time.perf_counter()
            for (h, _), e in zip(jobs, progress_map(pool, _check, jobs, 'wrist path check')):
                errors[h].append(e)
            print(f'[wp probe] round {refine_round} check {time.perf_counter() - t_c:.1f}s', flush=True)
            entry = dict(after_rounds=refine_round, seconds=round(time.perf_counter() - t0, 1), intermediate=intermediate,
                         left_early=left_early)
            for h in 'LR':
                e = np.concatenate(errors[h])
                entry[h] = dict(off_over_tolerance=int((e[:, 0] > REFINE_TOLERANCE).sum()), off_over_8mm=int((e[:, 0] > .008).sum()),
                                off_max_mm=round(float(e[:, 0].max()) * 1000, 1), swung=int((e[:, 1] > REFINE_LATERAL).sum()),
                                sunk=int((e[:, 2] > REFINE_FREE).sum()), to_refine=int((refine_score(e) > 1).sum()))
            refine_history.append(entry)
            print('[wrist paths] after %d rounds: ' % refine_round + ', '.join(
                f"{h} off>8mm {entry[h]['off_over_8mm']} max {entry[h]['off_max_mm']} mm, to refine {entry[h]['to_refine']}" for h in 'LR'),
                flush=True)
            if last:
                break
            new = []
            for h in 'LR':
                # the intro lift's free legs are not errors: the refiner must not anchor inside the trip window
                guard = getattr(paths[h], 'trip_end', 0.)
                for t in refine_points(samples, np.where(samples < guard, 0., refine_score(np.concatenate(errors[h]))),
                                       [a for a in anchors if a.hand == h]):
                    # an idle hand too: its resting fingers lifted clear of the keys
                    held = tuple(n.id for n in hand_notes[h].values() if n.touch <= t <= n.release)
                    new.append(Anchor(h, -1, float(t), float(paths[h](t)[0]), (), held))
            if not new:
                break
            jobs = []
            for a in new:
                v = paths[a.hand](a.time)
                values = {notes[k].digit: waypoint(waypoints[k], a.time)[1] for k in a.held if k in waypoints}
                # also from the pose that took these keys: the path between anchors can carry a stretched hand
                # into another basin (the thumb's joint across the next key)
                took = max((o for o in anchors if o.hand == a.hand and o.time <= a.time and a.held
                            and set(a.held) <= set(o.incoming) | set(o.held)), key=lambda o: o.time, default=None)
                seeds = ((vector_of(took.pose), took.pose.values),) if took else ()
                jobs.append((a.hand, a.contacts(notes), v, values, v, (NEAR_PATH,), REACH_TOLERANCE, False, seeds))
            t_s = time.perf_counter()
            poses = progress_map(pool, _settle, jobs, 'intermediate poses', 2)
            print(f'[wp probe] round {refine_round} settle {time.perf_counter() - t_s:.1f}s', flush=True)
            kept = [replace(a, pose=p) for a, p in zip(new, poses)
                    if p is not None and gentle(a, p, paths[a.hand], [o for o in anchors + new if o.hand == a.hand])]
            # an intermediate pose is a place the keys can be held from, not a nail: it has its depths too
            if kept:
                parts = progress_map(pool, _family, [list(c) for c in np.array_split(np.array(kept, dtype=object), max(1, len(kept) // 2))
                                                      if len(c)], 'depth families')
                for a, sl in zip(kept, [sl for part in parts for sl in part]):
                    a.family = sl
            # where no gentle pose holds the keys, the keys the hand cannot hold on its way are let go early
            missed = [a for a, p in zip(new, poses) if a.held and not any(k is a for k in kept)]
            lifted = {}
            if missed:
                held = progress_map(pool, _held_errors, [(a.hand, a.time) for a in missed], 'held keys off', 4)
                for h in 'LR':
                    drops = [(k, a.time) for a, e in zip(missed, held) if a.hand == h for k, v in e.items() if v > REACH_TOLERANCE]
                    lifted.update(leave_lifts(drops, np.concatenate(errors[h])[:, 0], samples, notes))
            for k, lift in lifted.items():
                notes[k].release = lift
            if lifted:
                for a in anchors + kept:
                    a.held = tuple(k for k in a.held if notes[k].held_at(a.time))
                stroke_of = strokes(notes)
                base['strokes'] = stroke_of
                left_early += len(lifted)
        dropped_snaps += sum(p is not None for p in poses) - len(kept)
        if not kept and not lifted:
            break
        anchors = anchors + kept
        intermediate += len(kept)
        paths, waypoints = rebuild(anchors)
    # An idle hand moves out of the playing hand's way (the planner only keeps recently played keys apart).
    paths, offsets = apply_yield(paths, notes, duration)
    yielded = {h: round(float(np.abs(offsets[h][1]).max()) * 1000, 1) for h in 'LR'}
    _mark('wrist_paths', t0, stages)


    # 4. the torso's rows and the context the arm and finger stages read
    t0 = time.perf_counter()
    torso_rows, reach_rows = [], {'L': [], 'R': []}
    if rig.data.get('torso'):
        reference = {h: np.array(m) for h, m in rig.data['torso']['reference'].items()}
        for n in notes.values():
            wrist = np.asarray(paths[n.hand](n.onset)[:3], float)
            delta = wrist[0] - reference[n.hand][0, 3]
            weight = (n.velocity / 127) ** .6
            lean = math.copysign(LEAN_GAIN * max(0., abs(delta) - LEAN_NEAR), delta)
            arm = rig.arms[n.hand]
            reach = np.linalg.norm(wrist - reference[n.hand][:3, 3]) / (arm.l1 + arm.l2)
            torso_rows.append((n.onset, lean, weight))
            reach_rows[n.hand].append((n.onset, reach, weight))
            # Keep the body where it is while the key is still down. One row at the onset lets both the lean
            # and the shoulder protraction start back after their .8 s kernel; on a long note the arm then has
            # to undo that travel frame by frame -- the hand stays put on the key while the arm shakes, and
            # the lean/protraction recovering at different moments reads as a shoulder twisting into place.
            for t in np.arange(n.onset + TORSO_HOLD_STEP, n.release, TORSO_HOLD_STEP):
                torso_rows.append((float(t), lean, weight))
                reach_rows[n.hand].append((float(t), reach, weight))
    first = min(n.onset for n in notes.values())
    last = max(n.release for n in notes.values())
    by_pitch = {}
    for n in notes.values():
        by_pitch.setdefault(n.pitch, []).append(n)
    context = dict(base, paths=paths, waypoints=waypoints, torso_rows=(torso_rows, first, last, reach_rows), by_pitch=by_pitch)

    # The closing gesture returns the seat to its opening place so the resting hands meet the thighs in the same
    # seat as the intro's first frame (wrist, pitch, torso AND slide aligned). Set unconditionally: the frame
    # workers rebuild their own Torso in _init and read this there, whether or not the elbow solve loaded a trunk
    # (a short tail clip often has no trunk model). Carried through the dump for the diagnostic scripts too.
    if rig.data.get('torso'):
        _out_start = [getattr(paths[h], 'trip_outro', None) for h in 'LR']
        _out_land = [getattr(paths[h], 'trip_outro_land', None) for h in 'LR']
        if any(t is not None for t in _out_start) and any(t is not None for t in _out_land):
            context['torso_slide_return'] = (min(t for t in _out_start if t is not None),
                                             max(t for t in _out_land if t is not None))
            print('[closing] seat returns to opening place at %.2f..%.2f' % context['torso_slide_return'], flush=True)

    # 4.5 elbows: one smooth swing-angle curve per arm that keeps them out of the trunk's outline (elbows.solve,
    # the trunk.json export beside the rig), the body leaning less where no swing clears; the frames stage
    # interpolates the curve and hands the angle to arm.pose
    t0 = time.perf_counter()
    context['elbow_swing'], context['torso_cap'], context['torso_yield'] = None, None, None
    context['torso_slide_yield'] = None
    elbows_report = {}
    if rig.data.get('torso') and context.get('torso_rows'):
        trunk_model = elbows.load(RIG)
        if trunk_model is not None:
            torso = Torso(rig.data['torso'], rig.arms)
            torso.set_notes(*context['torso_rows'])
            dump = os.environ.get('PIANOMOTION_ELBOW_DUMP')   # dev hook: the elbow solve's own inputs, for the bench
            if dump:
                import pickle
                with open(dump, 'wb') as fh:
                    pickle.dump(dict(paths=paths, torso=torso, rig=rig, trunk=trunk_model, t0=-intro, t1=duration),
                                fh, protocol=4)
                print(f'[elbows] dumped {dump}', flush=True)
            context['elbow_swing'], elbows_report = elbows.solve(torso, paths, -intro, duration, rig, trunk_model,
                                                                 solvers=performer_solvers(rig))
            context['elbow_swing_slopes'] = {h: pchip_slopes(c['times'], c['angles'][:, None])
                                             for h, c in context['elbow_swing'].items()}
            context['torso_cap'] = torso.cap           # the yield the solve asked (the workers' torso reads it)
            context['torso_yield'] = torso.yield_rows  # the wrist-deviation ask (the workers' torso reads it)
            context['torso_slide_yield'] = torso.slide_yield_rows   # the ask's seat part
    _mark('elbows', t0, stages)

    # 4.6 The hands re-aim at the forearms the elbows actually placed. The first settle aimed each hand at the
    # forearm of the *standard seat's* elbow (the hand's plan does not depend on where the performer sits); the swing
    # curve then moves that elbow out of the trunk's way, and on the daylight piece the rendered forearm's yaw comes
    # out at p50 11 / p95 39 deg where the corridor predicted 4 -- so the palm is left bent 25-40 deg against the arm
    # it has (60 % of the left hand's frames below 160). With the curve known, the anchors settle again on the
    # forearm that is really there (posture.natural_orientation's forearm_yaw), and the wrist paths are rebuilt
    # through the new poses: the settle moves the wrist as well as the palm, so the fingers keep their keys.
    if context['elbow_swing'] is not None and not REAIM_OFF:
        t0 = time.perf_counter()
        from . import attitude
        natural2 = attitude.aim(posture_solvers, anchors, context['elbow_swing'], torso)
        for a in anchors:       # an anchor the curve could not place an elbow for keeps its first-pass target
            if id(a) not in natural2:
                natural2[id(a)] = natural.get(id(a), natural_orientation(posture_solvers[a.hand], a.pose.wrist))
        moved = float(np.median([np.linalg.norm(natural2[id(a)] - natural.get(id(a), natural2[id(a)]))
                                 for a in anchors])) if anchors else 0.
        print(f'[reaim] {len(natural2)} anchors, orientation target moved p50 {moved:.1f} deg', flush=True)
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(base,)) as pool:
            free_starts(pool, [(a.hand, a.contacts(notes)) for a in anchors], starts)   # the refine's own anchors
            cohere(anchors, natural2, 1, pool)
        paths, waypoints = rebuild(anchors)
        _mark('reaim', t0, stages)

        # 4.65 the attitude refine: across a held chord the settle lets the palm sag onto the forearm the trunk
        # forces (the contacts win, measured at 131-158 deg against the corridor's 160-162), and the anchors' own
        # palms are square only at the strike. Wherever the path's yaw sits far from what the placed elbow asks, an
        # anchor goes in and is settled with that yaw as the target: the settle moves the wrist with the palm, so the
        # keys are kept, which turning the yaw alone does not (measured: 17 deg at a fixed wrist, fingers 49 mm off).
        t0 = time.perf_counter()
        inserted = 0
        for att_round in range(ATT_ROUNDS):
            want = attitude.targets(paths, context['elbow_swing'], torso, posture_solvers, -intro, duration, ATT_STEP)
            new = []
            for h in 'LR':
                times_h, yaw_h, roll_h = want[h]
                for t, y, r in zip(times_h, yaw_h, roll_h):
                    if not (np.isfinite(y) and np.isfinite(r)):
                        continue
                    v = np.asarray(paths[h](float(t)), float)
                    if abs(y - v[3]) < ATT_GATE and abs(r - v[4]) < ATT_GATE:
                        continue
                    y = v[3] + float(np.clip(y - v[3], -ATT_TURN, ATT_TURN))
                    r = v[4] + float(np.clip(r - v[4], -ATT_TURN, ATT_TURN))
                    held = tuple(k for k, n in hand_notes[h].items() if n.touch <= t <= n.release)
                    if not held:            # an idle hand keeps no key; its own wrist loses nothing
                        continue
                    new.append((Anchor(h, -1, float(t), float(v[0]), (), held), (float(y), float(r))))
            if not new:
                break
            with ProcessPoolExecutor(workers, initializer=_init, initargs=(context,)) as pool:
                jobs = []
                for a, (y, r) in new:
                    v = np.asarray(paths[a.hand](a.time), float)
                    values = {notes[k].digit: waypoint(waypoints[k], a.time)[1] for k in a.held if k in waypoints}
                    jobs.append((a.hand, a.contacts(notes), v, values, np.r_[v[:3], y, r], ATT_PULLS,
                                 REACH_TOLERANCE, False, ()))
                poses = progress_map(pool, _settle, jobs, f'attitude {att_round + 1}/{ATT_ROUNDS}')
                near = {h: [o for o in anchors if o.hand == h] + [a for a, _ in new if a.hand == h] for h in 'LR'}
                kept = []
                for (a, _), p in zip(new, poses):
                    if p is None or p.contact_error > REACH_TOLERANCE:
                        continue
                    if not gentle(a, p, paths[a.hand], near[a.hand]):
                        continue
                    # the guard: the pose has to straighten the hand on the keyboard (the fingers closer to the keys'
                    # length) and still sit above the wrist's floor -- a settle that keeps the keys by folding the
                    # wrist is the trade this pass exists to refuse
                    before_v = np.asarray(paths[a.hand](a.time), float)
                    after = attitude.palm_at(posture_solvers[a.hand], torso, context['elbow_swing'], a.hand,
                                             vector_of(p), a.time)
                    if after < attitude.MIN_DEG - 1.:
                        continue
                    if attitude.finger_at(posture_solvers[a.hand], vector_of(p)) >= \
                            attitude.finger_at(posture_solvers[a.hand], before_v) - 1.:
                        continue
                    kept.append(replace(a, pose=p))
            if not kept:
                break
            context['paths'] = paths
            with ProcessPoolExecutor(workers, initializer=_init, initargs=(context,)) as pool:
                parts = progress_map(pool, _family, [list(c) for c in
                                                     np.array_split(np.array(kept, dtype=object), max(1, len(kept) // 2))
                                                     if len(c)], 'attitude families')
                for a, sl in zip(kept, [sl for part in parts for sl in part]):
                    a.family = sl
            anchors = anchors + kept
            inserted += len(kept)
            paths, waypoints = rebuild(anchors)
            print(f'[attitude] round {att_round + 1}: {len(kept)} of {len(new)} anchors settle on the placed elbow',
                  flush=True)
        if inserted:
            print(f'[attitude] {inserted} anchors inserted, wrist floor {attitude.MIN_DEG:.0f} deg', flush=True)
        # 4.66 an idle hand keeps no key, so nothing is bought by its own yaw: the palm turns onto the forearm there
        # too (a folded wrist in the air reads as a stiff hand), smoothed and rate-limited, the playing frames
        # untouched.
        if APPLY_IDLE:
            want = attitude.targets(paths, context['elbow_swing'], torso, posture_solvers, -intro, duration, ATT_STEP)
            from .motion import KeyFloor
            # the wrist rides ~4 cm above the white keys' top when playing and sits below it on the lap, so the
            # line between (the key tops themselves) splits resting from aiming on any seat height
            aim_z = KeyFloor(posture_solvers['R'].regions).white_top + .01
            for h in 'LR':
                tm = want[h][0]
                busy = np.zeros(len(tm), bool)
                for n in hand_notes[h].values():
                    busy |= (tm >= n.touch - .15) & (tm <= n.release + .15)
                paths[h].set_attitude(*attitude.idle_attitude(paths[h], want[h], busy, aim_z=aim_z))
            print('[attitude] idle stretches turned onto the forearm', flush=True)
        paths, _ = apply_yield(paths, notes, duration)
        # When the closing wrist actually reaches the thigh: the return curve's last stretch is a slow slide
        # along the leg, so this is well before the gesture's nominal end. Everything that must be settled by
        # the CONTACT -- the fingers' unfold, the palm's pitch -- reads this, not the end.
        for h in 'LR':
            if getattr(paths[h], 'trip_outro_land', None) is not None:
                paths[h].trip_outro_touch = lap_arrival(paths[h], paths[h](float(paths[h].times[-1])))
        context['paths'], context['waypoints'] = paths, waypoints
        # the intro's body sway (the user's spec): the drawing arms tip the chest a hair back while the hands
        # pull in and the forearm pokes up; the reaching arms carry it a little forward, easing back to neutral
        # as the playing settles. Positive = backward, on top of the hips' constant forward lean. The sway
        # rides the OPENING hand's gesture (a late hand's quiet approach does not re-tip the chest).
        opener = min('LR', key=lambda h: abs(getattr(paths[h], 'trip_land', 1e9) - intro_land_path))
        lv = getattr(paths[opener], 'trip_leave', None)
        if lv is not None and rig.data.get('torso') and context.get('torso_rows'):
            de = getattr(paths[opener], 'trip_draw_end', None) or lv + .30
            top = getattr(paths[opener], 'trip_lift_end', None) or de + .45
            tlt = getattr(paths[opener], 'trip_tilt_end', None) or top + .30
            red = getattr(paths[opener], 'trip_reach_end', None) or tlt + 1.15
            # strictly rising rows (trip markers can coincide after a scale collapse; two rows at one time
            # poison the pchip slopes and the sway then wobbles the whole fall's world wrist)
            sw_t = np.array([lv - 1., lv, de, top, tlt, red, red + 1.2, red + 3.2], dtype=float)
            sw_t = np.maximum.accumulate(sw_t + np.arange(len(sw_t)) * 1e-3)
            context['torso_sway'] = (sw_t,
                                     np.array([0., 0., 1.2, 1.8, .9, -2.5, -1.2, 0.], dtype=float))
            print('[torso sway] back 1.8 deg at the poke, forward 2.5 deg at the reach', flush=True)
        # The closing gesture straightens the chest 1-2 deg as the upper arm withdraws, back to the piece's
        # own lean by the time the hands rest. A second, much later row set; when a very short piece overlaps
        # the intro's sway the two contributions are summed on one strictly rising timeline.
        if rig.data.get('torso') and context.get('torso_rows'):
            out_start = [getattr(paths[h], 'trip_outro', None) for h in 'LR']
            out_land = [getattr(paths[h], 'trip_outro_land', None) for h in 'LR']
            if any(t is not None for t in out_start) and any(t is not None for t in out_land):
                s0 = min(t for t in out_start if t is not None)
                s1 = max(t for t in out_land if t is not None)
                out_t = np.asarray([s0 - .35, s0 + .30, s1], float)
                out_v = np.asarray([0., 1.4, 0.], float)
                if context.get('torso_sway') is not None:
                    old_t, old_v = context['torso_sway']
                    if old_t[-1] >= out_t[0]:
                        grid = np.unique(np.r_[old_t, out_t])
                        context['torso_sway'] = (grid, np.interp(grid, old_t, old_v) + np.interp(grid, out_t, out_v))
                    else:
                        context['torso_sway'] = (np.r_[old_t, out_t], np.r_[old_v, out_v])
                else:
                    context['torso_sway'] = (out_t, out_v)
                print('[torso sway] closing straightens 1.4 deg', flush=True)
        _mark('attitude', t0, stages)

    # 4.7 finger keyposes on the settled wrists
    t0 = time.perf_counter()
    with ProcessPoolExecutor(min(workers, 10), initializer=_init, initargs=(context,)) as pool:
        tracks = {'L': {}, 'R': {}}
        for (h, d), track in progress_map(pool, _track, [(h, d) for h in 'LR' for d in DIGITS], 'finger tracks'):
            tracks[h][d] = track
    # the lap rest's relaxed shape: held from the first frame to the hover's end, eased into the first note's
    # shape across the approach, and eased back onto after the piece's last keypose (the fingers on the return)
    from .motion import LAP_FINGERS, LAP_TUCK, LAP_TUCK_MIN_LIFT
    from .lap_contact import shape as lap_shape
    for h in 'LR':
        leave = getattr(paths[h], 'trip_leave', None)
        lap = lap_shape(h) if LAP_CONTACT else {d: LAP_FINGERS[d] for d in DIGITS}
        for d in DIGITS:
            track = tracks[h][d]
            track.rest = np.array(lap[d], dtype=float)
            track.tuck = LAP_TUCK[d].copy()
            # The curl starts only once the wrist has risen clear of the thigh. Starting it at the draw's end
            # (the old ``trip_draw_end``) curled the fingers while the palm still lay on the leg, driving the
            # pads into the thigh (canonpd: 7.4 mm at a 5 mm lift). Find the first time the path's z clears the
            # lap by LAP_TUCK_MIN_LIFT -- a geometric gate, the same for any piece. Falls back to the draw's
            # end when the path is already clear there or carries no usable rise.
            rest_tuck = getattr(paths[h], 'trip_draw_end', None) or leave
            path_h = paths[h]
            z0 = float(np.asarray(path_h(leave if leave is not None else rest_tuck), FLOAT)[2]) \
                if (leave is not None or rest_tuck is not None) else None
            if z0 is not None:
                t_scan, t_end = float(leave if leave is not None else rest_tuck), getattr(paths[h], 'trip_lift_end', None)
                t_end = t_end if t_end is not None else t_scan + 2.
                while t_scan < t_end:
                    if float(np.asarray(path_h(t_scan), FLOAT)[2]) - z0 >= LAP_TUCK_MIN_LIFT:
                        break
                    t_scan += 1. / 240.
                rest_tuck = max(rest_tuck, t_scan)
            track.rest_tuck = rest_tuck
            end = track.strokes[0].hover if track.strokes else (track.keyposes[0].time if len(track.keyposes) else None)
            if end is not None:
                # the fingers keep the tuck until just before their OWN first stroke: the keyposes scheduled
                # during the poise would have them shaping mid-air (spreading 30-45 deg and back, the user's
                # "不明不白地划了几下"); past the seat the hand is still, so the settle reads as placement
                track.rest_start = min(getattr(paths[h], 'trip_land', end) - .1, end - .45)
            if len(track.keyposes):
                # the fingers curl into the gathered lap shape as the wrist's closing gesture starts, and
                # open onto the thigh exactly when it arrives (one clock with the single return curve).
                out_at = getattr(paths[h], 'trip_outro', None)
                track.rest_back = out_at if out_at is not None else track.keyposes[-1].time + .4
                track.outro_end = getattr(paths[h], 'trip_outro_land', None)
                # the fingers must be open when the hand CONTACTS the thigh, which is earlier than the
                # gesture's nominal end (the last stretch is a slide along the leg); the path carries that
                # moment from the wrist_paths stage, so the palm's pitch and the fingers read one clock
                if track.outro_end is not None:
                    track.outro_touch = getattr(paths[h], 'trip_outro_touch', None)
                    print('[lap touch] %s outro=%.3f touch=%s open_by=%.3f' % (
                          h, track.outro_end,
                          None if track.outro_touch is None else round(track.outro_touch, 3),
                          track.outro_touch if track.outro_touch is not None else track.outro_end), flush=True)
                # Past the landing the hand is at rest again: skip the spacing just as at the intro's start,
                # so the fingers land in exactly the shape they opened in. A hand with no strokes at all never
                # leaves the lap, so it skips the spacing throughout.
                if track.outro_end is not None:
                    track.rest_end = track.outro_end
                    if track.outro_touch is not None:
                        track.rest_end = min(track.rest_end, track.outro_touch)
                elif not track.strokes:
                    track.rest_end = float('-inf')
        print('[lap rest] %s leave=%s rest_start=%s strokes0=%s keyposes0=%s' % (
              h, None if leave is None else round(leave, 3),
              None if tracks[h][2].rest_start is None else round(tracks[h][2].rest_start, 3),
              round(tracks[h][2].strokes[0].hover, 3) if tracks[h][2].strokes else None,
              round(tracks[h][2].keyposes[0].time, 3) if len(tracks[h][2].keyposes) else None), flush=True)
    # free fingers after the spacing: a smooth lift where the swing sank them into the keys
    grid = np.arange(-intro, duration + CLEAR_STEP, CLEAR_STEP)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(dict(context, tracks=tracks),)) as pool:
        jobs = [(h, part) for h in 'LR' for part in chunks(grid, workers * 2)]
        parts = progress_map(pool, _room, jobs, 'finger room')
    for h in 'LR':
        mine = [p for (hh, _), p in zip(jobs, parts) if hh == h]
        for d in DIGITS:
            tracks[h][d].set_room(grid, np.concatenate([p[d] for p in mine]))
    _mark('finger_keyposes', t0, stages)

    # 5. frames
    t0 = time.perf_counter()
    times = np.arange(-intro, duration, 1. / FRAME_RATE)
    context['tracks'] = tracks
    if args.dump:
        import pickle
        pickle.dump(dict(context, anchors=anchors, rest=rest, duration=duration), open(args.dump, 'wb'))
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(context,)) as pool:
        parts = progress_map(pool, _frames, chunks(times, workers * CHUNKS_PER_WORKER), 'frames')
    frames = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    frames['key_pitches'] = np.array(sorted(by_pitch))
    bridged = despike(frames)
    brushed = brushed_keys(frames, FRAME_RATE)
    diagnostics = {k: frames[k] for k in frames if k.endswith(('_finger_gap', '_riding', '_finger_overlap', '_raised', '_pointing', 'hands_gap'))}
    np.savez_compressed(out / 'diagnostics.npz', times=frames['times'], **diagnostics)
    crossed = crossings(frames, FRAME_RATE)
    ride_swing = riding_sideways(frames)
    bodies = finger_bodies(frames, FRAME_RATE)
    _mark('frames', t0, stages)

    # 6. outputs
    # a key lifted early is held by an extra pedal to its sound's end, or to the score's next pedal-up (which clears it)
    ups = np.sort(score_pedal[:, 1][np.isfinite(score_pedal[:, 1])])

    def held_to(begin, end):
        i = np.searchsorted(ups, begin, side='right')
        return min(end, ups[i]) if i < len(ups) else end
    frames['damper_intervals'] = pedal_union(score_pedal,
                                             [(n.release - PEDAL_LEAD, held_to(n.release, n.sound_release)) for n in notes.values()
                                              if n.sound_release is not None and n.release < n.sound_release - 1e-3])
    frames['damper'] = damper_depth(frames['damper_intervals'], frames['times'])
    frames['times'] = frames['times'] + intro          # outputs start at the lead-in
    frames['damper_intervals'] = frames['damper_intervals'] + intro
    np.savez_compressed(out / 'motion.npz', fps_source=rig.fps, rate=FRAME_RATE, **frames)
    played = {n.id for n in notes.values()}
    source = [n for n in raw['timeline']['notes'] if args.end is None or n['start'] <= args.end + 1e-9]
    sounding = {n.id: n.release if n.sound_release is None else n.sound_release for n in notes.values()}
    cuts = sounding
    final_rows = [dict(r, physical_release=sounding[r['id']], key_release=notes[r['id']].release) for r in rows]
    shortened = [(r, score.audible_shortening(r['index'], r['physical_release'])) for r in final_rows]
    shortened = [(r, s) for r, s in shortened if s > 1e-9]
    fidelity, _ = compare(raw['timeline'], sorted(played), cuts, source_ids=[n['id'] for n in source])
    report = dict(
        rig=RIG.name,
        listening=dict(source_notes=len(source), played=len(played), omitted=len(source) - len(played),
                       omitted_ratio=1 - len(played) / len(source), registers=register_report(source, played),
                       audibly_shortened_notes=len(shortened), band_deficit_db=fidelity,
                       plan_validation_problems=len(problems), score_hands=score_hands(source, final_rows)),
        motion=dict(frames=len(times), rate=FRAME_RATE, anchors=summary([a for a in anchors if a.incoming]),
                    awkward_shapes=shape_report(chosen), intermediate_poses=intermediate, intermediate_dropped_snaps=dropped_snaps, refine_rounds=args.refine_rounds, refine_history=refine_history, held_keys_left_early=left_early, idle_hand_yield_mm=yielded, early_releases_for_wrist=freed, bridged_frames=bridged,
                    brushed_keys=brushed,
                    smoothness=smoothness(frames, FRAME_RATE),
                    finger_sideways_at_press=sideways(frames, notes, FRAME_RATE),
                    finger_crossings=crossed,
                    riding_sideways=ride_swing,
                    finger_bodies=bodies,
                    contact_error_mm={h: dict(p99=float(np.percentile(frames[f'{h}_contact_error'], 99) * 1000),
                                              max=float(frames[f'{h}_contact_error'].max() * 1000)) for h in 'LR'},
                    penetration_frames_over_1mm={h: int((frames[f'{h}_penetration'] > .001).sum()) for h in 'LR'},
                    penetration_frames_over_3mm={h: int((frames[f'{h}_penetration'] > .003).sum()) for h in 'LR'},
                    worst_penetration_mm={h: float(frames[f'{h}_penetration'].max() * 1000) for h in 'LR'},
                    # the thighs, not the keys: the lap rest is fitted against this surface, so anything below
                    # it is a hand driven into the leg (the closing gesture's tuck used to land like that)
                    lap_penetration=dict(
                        frames_over_1mm={h: int((frames[f'{h}_lap_penetration'] > .001).sum())
                                         if f'{h}_lap_penetration' in frames else None for h in 'LR'},
                        worst_mm={h: round(float(frames[f'{h}_lap_penetration'].max() * 1000), 2)
                                  if f'{h}_lap_penetration' in frames else None for h in 'LR'}),
                    arm_out_of_reach_frames={h: int((~frames[f'{h}_arm_ok']).sum()) for h in 'LR'},
                    palm_vs_forearm_deg={h: dict(p50=round(float(np.median(frames[f'{h}_palm_deg'])), 1),
                                                 p5=round(float(np.percentile(frames[f'{h}_palm_deg'], 5)), 1),
                                                 min=round(float(frames[f'{h}_palm_deg'].min()), 1),
                                                 below_160=round(float((frames[f'{h}_palm_deg'] < 160).mean()), 3))
                                        for h in 'LR'},
                    # the hand on the keyboard, not on its own arm: the middle finger's direction seen from above
                    # against the keys' length (deg), the number the eye reads as "the hand is crooked on the keys"
                    fingers_vs_keys_deg={h: dict(p50=round(float(np.median(np.abs(frames[f'{h}_pointing'][:, 2]))), 1),
                                                 p95=round(float(np.percentile(np.abs(frames[f'{h}_pointing'][:, 2]), 95)), 1),
                                                 max=round(float(np.abs(frames[f'{h}_pointing'][:, 2]).max()), 1),
                                                 over_20=round(float((np.abs(frames[f'{h}_pointing'][:, 2]) > 20).mean()), 3))
                                         for h in 'LR'},
                    elbow_swing=elbows_report),
        seconds=dict(stages, total=time.perf_counter() - started), workers=workers,
        music_seconds=duration - TAIL_SECONDS, intro_seconds=intro)
    plan = dict(schema='performer-plan', notes=final_rows, omitted_ids=sorted(n['id'] for n in source if n['id'] not in played),
                validation_problems=problems,
                anchors=[dict(hand=a.hand, group=a.group, time=a.time, incoming=a.incoming, held=a.held, lifted=a.lifted,
                              ok=a.pose.ok, contact_error_mm=a.pose.contact_error * 1000) for a in anchors])
    (out / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding='utf-8')
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    render(raw['timeline'], played, out / 'preview.wav',
           cut_releases={r['id']: r['physical_release'] for r, _ in shortened}, lead=intro)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, help='worker processes (default: every thread of the big cores)')
    parser.add_argument('--beam', type=int, default=48)
    parser.add_argument('--end', type=float)
    parser.add_argument('--inputs', required=True, help='the piece: a folder with timeline.json and demands.json')
    parser.add_argument('--finger-spacing', type=float, default=None,
                        help="the planner's relaxed spacing per finger (m); default: an adult hand's. The rig's own "
                             "(rig_finger_spacing, ~.0154) avoids stretches by cutting held notes: -0.6 dB in the first 30 s")
    parser.add_argument('--lift-speed', type=float, help='wrist speed (m/s) the planner times releases for')
    parser.add_argument('--refine-rounds', type=int, default=REFINE_ROUNDS,
                        help=f'wrist path refine rounds: check, intermediate poses, early lifts (default {REFINE_ROUNDS})')
    parser.add_argument('--motion-cut', type=float, default=MOTION_CUT_SECONDS,
                        help='longest audible early lift that lets the wrist travel at a natural pace (s); 0: none')
    parser.add_argument('--dump', help='pickle the frame context (debugging)')
    parser.add_argument('--no-checks', action='store_true')
    args = parser.parse_args()
    from .cores import use_big_cores
    big = use_big_cores()
    args.workers = args.workers or big
    report = run(args)
    print(json.dumps(report, ensure_ascii=True, indent=2))      # any console encoding


if __name__ == '__main__':
    main()
