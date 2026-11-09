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
from pathlib import Path
import time

import numpy as np

from planner.audio import render
from planner.fidelity import compare
from planner.parallel import SUPPORTED_WORKERS, plan_parallel
from planner.problem import PIECE_FILES, build, load_raw, plan_notes, register_report
from planner.search import Weights
from planner.validate import validate

from .anchors import KEY_TRAVEL, REACH_TOLERANCE, Anchor, build_anchors, note_motions, solve_anchors, summary
from .checks import BRUSH_DEPTH, Heightfield, stretched
from .coherence import (COHERENCE_ROUNDS, MOTION_CUT_SECONDS, ORIENTATION_PULLS, free_releases, least_sunk, neighbour_targets,
                        settle, sunk, vector_of, wide_settle)
from .kinematics import DIGITS, SWING_LATERAL, Rig
from .torso import Torso
from .contacts import HandCapsules, finger_pairs, hands_gap, pointing, raised_between
from .motion import (CLEAR_STEP, REFINE_STEP, FingerIK, room_corrections, contact_error, contact_waypoints, finger_track, frame_hand,
                     placed, refine_points, refine_score, strokes, waypoint, wrist_path, SIDEWAYS_FREE, YieldingPath,
                     yield_offsets)
from .pedal import PEDAL_LEAD, damper_depth, damper_intervals, pedal_union
from .pose import RELAXED, hand_solvers
from .progress import progress_map
from .reach_table import MAX_INTERVAL, REFERENCE_OCTAVE, cache_path, load_or_compute
from .regions import performer_solvers
from .posture import natural_orientation

HERE = Path(__file__).resolve().parent
RIG = HERE / 'rig_cache' / 'rig.json'     # performer seat and size (rig_export.PERFORMER_SEATING, CHARACTER_SCALE)
LEGACY_RIG = 'rig_unscaled.json'   # runs whose report does not name its rig were solved with this one


def run_rig(run):
    """The rig a run directory's motion was solved with (checks must measure a run on its own rig)."""
    report = json.loads((Path(run) / 'report.json').read_text(encoding='utf-8'))
    return HERE / 'rig_cache' / report.get('rig', LEGACY_RIG)
FRAME_RATE = 60.                  # samples per second (written as fractional scene frames)
TAIL_SECONDS = 2.4                # after the source end: the hands lift off and come to rest (silent)
REST_OFFSET = np.array([0., -.04, .03])    # before and after the piece the hands rest on the lap: the wrist this far
                                  # from the thigh's top (rig_export.lap) where the palm lies
REST_YAW = 30.                    # deg the resting hands are turned in toward each other
INTRO_SECONDS = 2.                # the hands rest at least this long before the first note, then come onto the keys:
                                  # a piece starting sooner is preceded by silence (motion and audio start that much later)
WARM_FRAMES = 12                 # frames solved before a chunk (discarded) so the warm start carries over
CHUNKS_PER_WORKER = 4
REFINE_ROUNDS = 3                 # wrist path -> contact check -> intermediate poses, repeated
SPIKE_ERROR = .005                # pad error above which a short run of frames is replaced by its neighbours
SPIKE_FRAMES = 4

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
    _ctx['torso'] = torso
    _ctx['capsules'] = {h: HandCapsules(rig, h, rig.arms[h].arm_inverse @ solvers[h].palm(rig.neutral_wrist[h], 0., 0.))
                        for h in 'LR'}


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


def depth_family(solver, anchor, notes):
    """The anchor's hand at other depths and turns: (wrist vectors, poses, awkwardness) of every place it was settled
    at and holds its keys from, the pose's own first. The depths: settled at depths into the keys around its own, as
    far each way as its keys are held without more strain (``awkwardness``, the measure the planner prices); the
    turns: at every other of those depths, the hand turned each way as far as that holds. The wrist path chooses
    among them (motion.choose_places)."""
    pose = anchor.pose
    contacts = anchor.contacts(notes)
    base = vector_of(pose)
    if not contacts:
        return base[None], [pose], np.zeros(1)
    own = awkwardness(solver, contacts, pose)
    sink = max(FAMILY_SINK, max(sunk(solver, contacts, pose)) / 1000.) + FAMILY_SINK_GIVE

    def ok(p, target, hold_yaw):
        return (p.contact_error <= REACH_TOLERANCE / 2 and abs(float(p.wrist[1]) - target[1]) <= FAMILY_HOLD
                and (not hold_yaw or abs(float(p.yaw) - target[3]) <= TURN_HOLD)
                and max(sunk(solver, contacts, p)) / 1000. <= sink
                and awkwardness(solver, contacts, p) <= max(own + FAMILY_GIVE, FAMILY_OK))

    def walk(v0, p0, axis, step, steps, pull, hold_yaw):
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
                           FAMILY_ITERATIONS)
                if not ok(p, target, hold_yaw):
                    break
                last_v, last_values = vector_of(p), p.values
                out.append((last_v, p))
        return sorted(out, key=lambda e: float(e[0][axis]))

    line = walk(base, pose, 1, FAMILY_STEP, int(round(FAMILY_REACH / FAMILY_STEP)), FAMILY_PULL, False)
    centre = next(i for i, (v, _) in enumerate(line) if v is base)
    picks = sorted({i for i in range(len(line)) if (i - centre) % 3 == 0} | {0, len(line) - 1})
    places = list(line)
    for i in picks:
        places += [e for e in walk(*line[i], 3, TURN_STEP, TURN_STEPS, TURN_PULL, True) if e[1] is not line[i][1]]
    places.sort(key=lambda e: e[1] is not pose)          # the pose's own first
    return (np.array([e[0] for e in places]), [e[1] for e in places],
            np.array([awkwardness(solver, contacts, e[1]) for e in places]))


def _family(job):
    return [depth_family(_ctx['solvers'][a.hand], a, _ctx['notes']) for a in job]


def _check(job):
    hand, times = job
    return np.array([contact_error(t, _ctx['paths'][hand], _ctx['hand_notes'][hand], _ctx['waypoints'],
                                   _ctx['solvers'][hand], _ctx['ik'][hand]) for t in times])


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


def gentle(anchor, pose, path, others):
    """An intermediate ``pose`` that leaves the wrist ``path`` slowly enough next to the hand's ``others`` anchors."""
    dev = vector_of(pose) - path(anchor.time)
    dt = max(min((abs(o.time - anchor.time) for o in others if o is not anchor), default=1.), 1e-3)
    return float(np.abs(dev[3:]).max()) / dt <= INTERMEDIATE_TURN and float(np.linalg.norm(dev[:3])) / dt <= INTERMEDIATE_SPEED


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
        shapes = [(a.hand, a.contacts(notes)) for a in chords]
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
    by_pitch = _ctx['by_pitch']
    pitches = sorted(by_pitch)
    n = len(times)
    out = dict(times=np.asarray(times), torso=np.zeros((n, 4)), key_depth=np.zeros((n, len(pitches))))
    for h in 'LR':
        out[f'{h}_wrist'] = np.zeros((n, 5))
        out[f'{h}_values'] = np.zeros((n, 5, 4))
        out[f'{h}_root_q'] = np.zeros((n, 5, 4))
        out[f'{h}_child_euler'] = np.zeros((n, 5, 2, 3))
        out[f'{h}_arm_q'] = np.zeros((n, 4, 4))     # upper, elbow, wrist, twist
        out[f'{h}_contact_error'] = np.zeros(n)
        out[f'{h}_penetration'] = np.zeros(n)
        out[f'{h}_arm_ok'] = np.ones(n, dtype=bool)
        out[f'{h}_brush'] = np.zeros((n, KEYS))     # push each unplayed key wants from a finger resting on it
        out[f'{h}_finger_gap'] = np.zeros((n, 3))   # neighbour fingers 2-3, 3-4, 4-5 seen from above (crossings)
        out[f'{h}_riding'] = np.zeros((n, 5), dtype=bool)   # finger on its key
        out[f'{h}_finger_overlap'] = np.zeros((n, 4))       # neighbour finger bodies 1-2 .. 4-5 running into each other (m)
        out[f'{h}_raised'] = np.zeros(n)                    # free finger 3/4 held above its two pressing neighbours (m)
        out[f'{h}_pointing'] = np.zeros((n, 5))             # finger direction from above against the keys (deg)
    out['hands_gap'] = np.zeros(n)                          # clearance between the two hands' fingers (m)
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
        torso_q, pre = (None, None) if _ctx['torso'] is None else _ctx['torso'].pose(t)
        segments = {}
        if torso_q is not None:
            out['torso'][i] = torso_q
        for h in 'LR':
            vector, values, error = frame_hand(t, _ctx['paths'][h], tracks[h], notes, _ctx['waypoints'],
                                               _ctx['solvers'][h], _ctx['ik'][h], depths, previous[h])
            arm = rig.arms[h]
            out[f'{h}_wrist'][i] = vector
            out[f'{h}_contact_error'][i] = error
            # The key goes down as far as its finger takes it: a stretched hand that cannot reach the keybed
            # (an octave at the hand's limit) shows a key pressed a little less, not a finger above its key.
            solver = _ctx['solvers'][h]
            palm = solver.palm(vector[:3], vector[3], vector[4])
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
                pose = arm.pose(vector[:3], vector[3], vector[4], None if pre is None else pre[h])
                out[f'{h}_arm_q'][i] = [pose['local'][arm.names[k]] for k in ('upper', 'elbow', 'wrist', 'twist')]
            except ValueError:          # wrist beyond the unstretched arm
                # Never leave the arm's rotations at zero (Blender collapses such a bone and the arm vanishes):
                # stretch the arm toward the wrist instead; the hand stops just short of where it was sent.
                out[f'{h}_arm_ok'][i] = False
                shoulder_pre = arm.pre_upper_world if pre is None else pre[h]
                reach = shoulder_pre[:3, 3]
                aim = np.asarray(vector[:3], float) - reach
                stretched = reach + aim * (ARM_STRETCH * (arm.l1 + arm.l2) / np.linalg.norm(aim))
                pose = arm.pose(stretched, vector[3], vector[4], shoulder_pre)
                out[f'{h}_arm_q'][i] = [pose['local'][arm.names[k]] for k in ('upper', 'elbow', 'wrist', 'twist')]
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
            if _ctx['heightfield'] is not None:
                worst = 0.
                for d in DIGITS:
                    points = rig.clearance_points(h, d, pose['armature'], values)
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
    """Single-instant solver misses (a few frames above ``SPIKE_ERROR``) are bridged from the frames around them."""
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
    technique): share above 10 and 15 deg, and at the joint limit. Removes the riding masks from ``frames``."""
    out = {}
    for h in 'LR':
        riding = frames.pop(f'{h}_riding')
        direction = np.abs(frames.pop(f'{h}_pointing'))
        thumb = np.abs(frames[f'{h}_values'][:, 0, 0])[riding[:, 0]]
        swing = np.abs(frames[f'{h}_values'][:, 1:, 0])[riding[:, 1:]]
        limit = np.array([SWING_LATERAL.get(d, 25.) - 1. for d in (2, 3, 4, 5)])
        at_limit = (np.abs(frames[f'{h}_values'][:, 1:, 0]) >= limit)[riding[:, 1:]]
        out[h] = dict(over_10_deg=round(float((swing > 10).mean()), 3), over_15_deg=round(float((swing > 15).mean()), 3),
                      at_limit=round(float(at_limit.mean()), 3), p99_deg=round(float(np.percentile(swing, 99)), 1),
                      pointing_over_25_deg=round(float((direction[:, 1:][riding[:, 1:]] > 25).mean()), 3),
                      pointing_p99_deg=round(float(np.percentile(direction[:, 1:][riding[:, 1:]], 99)), 1),
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


def run(args):
    started = time.perf_counter()
    stages = {}
    workers = args.workers
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    table, _ = load_or_compute(RIG, workers)
    stages['rig_pair_table'] = time.perf_counter() - t0

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
    planner_kwargs = dict(beam_width=args.beam, outer_voice_bonus=1.5, pair_table=table, pair_strain=pair_awkwardness(table, workers),
                          rest_x={h: float(Rig(RIG).neutral_wrist[h][0]) for h in 'LR'},
                          weights=Weights(**({} if args.finger_spacing is None else dict(finger_spacing=args.finger_spacing)),
                                          **({} if args.lift_speed is None else dict(lift_speed=args.lift_speed))))
    stages['planning'] = stages['anchor_poses'] = stages['chord_shapes'] = 0.
    shapes, anchor_cache, shape_cache, starts = {}, {}, {}, {}
    for shape_round in range(SHAPE_ROUNDS):
        print(f'[planning] round {shape_round + 1}', flush=True)
        penalty = {k: AWKWARD_COST * (u - AWKWARD_FREE) for k, u in shapes.items() if u > AWKWARD_FREE}
        planned = plan_parallel(build, (inputs, args.end), dict(planner_kwargs, shape_penalty=penalty), workers=workers)
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
    lifts = [free_releases([a for a in anchors if a.hand == h], notes, stroke_of, args.motion_cut) for h in 'LR']
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
    turned = set()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(base,)) as pool:
        free_starts(pool, [(a.hand, a.contacts(notes)) for a in anchors], starts)
        for round_ in range(COHERENCE_ROUNDS):
            order, jobs = [], []
            for h in 'LR':
                mine = sorted((a for a in anchors if a.hand == h), key=lambda a: a.time)
                offsets = [np.zeros(2) if id(a) in turned else natural[id(a)] for a in mine]
                for a, target in zip(mine, neighbour_targets(mine, offsets)):
                    order.append(a)
                    contacts = a.contacts(notes)
                    # a stretched hand searched once, in the first round; the next rounds settle from what it found
                    # a shape of two or more keys is also settled from a free start in the first round: the least
                    # awkward is kept (its anchor's pose may sit in a strained basin)
                    jobs.append((h, contacts, vector_of(a.pose), a.pose.values, target, ORIENTATION_PULLS,
                                 max(a.pose.contact_error, .0005), round_ == 0 and stretched(c.pitch for c in contacts),
                                 (starts[h, tuple(contacts)],) if round_ == 0 and len(contacts) > 1 else ()))
            for a, pose in zip(order, progress_map(pool, _settle, jobs, f'coherence {round_ + 1}/{COHERENCE_ROUNDS}')):
                if pose is not None:
                    a.pose = pose
                    turned.add(id(a))
    stages['coherence'] = time.perf_counter() - t0

    # 3. wrist paths, refined until the held keys stay under the fingers
    t0 = time.perf_counter()
    rig = Rig(RIG)
    duration = (args.end if args.end is not None else float(raw['timeline']['duration'])) + TAIL_SECONDS
    intro = max(0., INTRO_SECONDS - min(n['start'] for n in raw['timeline']['notes']))
    rest_solvers = performer_solvers(rig)
    rest = {}
    for h in 'LR':      # on the lap: the palm on the thigh (rig_export.lap), the hand turned in a little
        wrist = np.array(rig.data['hands'][h]['lap'], dtype=float) + REST_OFFSET
        orientation = np.array(natural_orientation(rest_solvers[h], wrist), dtype=float)
        orientation[0] += REST_YAW * (1. if h == 'R' else -1.)
        rest[h] = np.r_[wrist, orientation]
    hand_notes = {h: {k: n for k, n in notes.items() if n.hand == h} for h in 'LR'}

    def paths_for(anchors):
        return {h: wrist_path([a for a in anchors if a.hand == h], hand_notes[h], stroke_of, rest[h], duration)
                for h in 'LR'}

    with ProcessPoolExecutor(workers, initializer=_init, initargs=(base,)) as pool:
        played = [a for a in anchors if a.pose is not None]
        parts = progress_map(pool, _family, [list(c) for c in np.array_split(np.array(played, dtype=object), workers * 4) if len(c)],
                             'depth families')
        for a, s in zip(played, [s for part in parts for s in part]):
            a.family = s
    paths = paths_for(anchors)
    waypoints = contact_waypoints(placed(anchors, paths, posture_solvers, notes), notes)
    samples = np.arange(0., duration, REFINE_STEP)
    intermediate = dropped_snaps = 0
    for refine_round in range(REFINE_ROUNDS):
        print(f'[wrist paths] refine round {refine_round + 1}/{REFINE_ROUNDS}', flush=True)
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(dict(base, paths=paths, waypoints=waypoints),)) as pool:
            jobs = [(h, c) for h in 'LR' for c in chunks(samples, workers * 2)]
            errors = {h: [] for h in 'LR'}
            for (h, _), e in zip(jobs, progress_map(pool, _check, jobs, 'wrist path check')):
                errors[h].append(e)
            new = []
            for h in 'LR':
                for t in refine_points(samples, refine_score(np.concatenate(errors[h])), [a for a in anchors if a.hand == h]):
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
                starts = ((vector_of(took.pose), took.pose.values),) if took else ()
                jobs.append((a.hand, a.contacts(notes), v, values, v, (NEAR_PATH,), REACH_TOLERANCE, False, starts))
            poses = progress_map(pool, _settle, jobs, 'intermediate poses', 2)
            kept = [replace(a, pose=p) for a, p in zip(new, poses)
                    if p is not None and gentle(a, p, paths[a.hand], [o for o in anchors + new if o.hand == a.hand])]
            # an intermediate pose is a place the keys can be held from, not a nail: it has its depths too
            if kept:
                parts = progress_map(pool, _family, [list(c) for c in np.array_split(np.array(kept, dtype=object), workers * 4)
                                                      if len(c)], 'depth families')
                for a, sl in zip(kept, [sl for part in parts for sl in part]):
                    a.family = sl
        dropped_snaps += sum(p is not None for p in poses) - len(kept)
        if not kept:
            break
        anchors = anchors + kept
        intermediate += len(kept)
        paths = paths_for(anchors)
        waypoints = contact_waypoints(placed(anchors, paths, posture_solvers, notes), notes)
    # An idle hand moves out of the playing hand's way (the planner only keeps recently played keys apart).
    offsets = yield_offsets(paths, notes, duration, HANDS_APART, YIELD_SPEED, YIELD_BUSY)
    yielded = {h: round(float(np.abs(offsets[h][1]).max()) * 1000, 1) for h in 'LR'}
    paths = {h: YieldingPath(paths[h], *offsets[h]) for h in 'LR'}
    stages['wrist_paths'] = time.perf_counter() - t0


    # 4. torso, finger keyposes
    t0 = time.perf_counter()
    torso_rows = []
    if rig.data.get('torso'):
        reference = {h: np.array(m) for h, m in rig.data['torso']['reference'].items()}
        for n in notes.values():
            delta = paths[n.hand](n.onset)[0] - reference[n.hand][0, 3]
            torso_rows.append((n.onset, math.copysign(max(0., abs(delta) - .15), delta), (n.velocity / 127) ** .6))
    first = min(n.onset for n in notes.values())
    last = max(n.release for n in notes.values())
    by_pitch = {}
    for n in notes.values():
        by_pitch.setdefault(n.pitch, []).append(n)
    context = dict(base, paths=paths, waypoints=waypoints, torso_rows=(torso_rows, first, last), by_pitch=by_pitch)
    with ProcessPoolExecutor(min(workers, 10), initializer=_init, initargs=(context,)) as pool:
        tracks = {'L': {}, 'R': {}}
        for (h, d), track in progress_map(pool, _track, [(h, d) for h in 'LR' for d in DIGITS], 'finger tracks'):
            tracks[h][d] = track
    # free fingers after the spacing: a smooth lift where the swing sank them into the keys
    grid = np.arange(-intro, duration + CLEAR_STEP, CLEAR_STEP)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(dict(context, tracks=tracks),)) as pool:
        jobs = [(h, part) for h in 'LR' for part in chunks(grid, workers * 2)]
        parts = progress_map(pool, _room, jobs, 'finger room')
    for h in 'LR':
        mine = [p for (hh, _), p in zip(jobs, parts) if hh == h]
        for d in DIGITS:
            tracks[h][d].set_room(grid, np.concatenate([p[d] for p in mine]))
    stages['finger_keyposes'] = time.perf_counter() - t0

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
    stages['frames'] = time.perf_counter() - t0

    # 6. outputs
    frames['damper_intervals'] = pedal_union(damper_intervals(raw['timeline']),
                                             [(n.release - PEDAL_LEAD, n.sound_release) for n in notes.values()
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
                    awkward_shapes=shape_report(chosen), intermediate_poses=intermediate, intermediate_dropped_snaps=dropped_snaps, idle_hand_yield_mm=yielded, early_releases_for_wrist=freed, bridged_frames=bridged,
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
                    arm_out_of_reach_frames={h: int((~frames[f'{h}_arm_ok']).sum()) for h in 'LR'}),
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
    parser.add_argument('--workers', type=int, default=8, choices=SUPPORTED_WORKERS)
    parser.add_argument('--beam', type=int, default=48)
    parser.add_argument('--end', type=float)
    parser.add_argument('--inputs', required=True, help='the piece: a folder with timeline.json and demands.json')
    parser.add_argument('--finger-spacing', type=float, default=None,
                        help="the planner's relaxed spacing per finger (m); default: an adult hand's. The rig's own "
                             "(rig_finger_spacing, ~.0154) avoids stretches by cutting held notes: -0.6 dB in the first 30 s")
    parser.add_argument('--lift-speed', type=float, help='wrist speed (m/s) the planner times releases for')
    parser.add_argument('--motion-cut', type=float, default=MOTION_CUT_SECONDS,
                        help='longest audible early lift that lets the wrist travel at a natural pace (s); 0: none')
    parser.add_argument('--dump', help='pickle the frame context (debugging)')
    parser.add_argument('--no-checks', action='store_true')
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=True, indent=2))      # any console encoding


if __name__ == '__main__':
    main()
