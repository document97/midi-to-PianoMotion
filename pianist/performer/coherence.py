"""Hand poses that read as one movement: free early releases and orientation coherence.

Anchor poses are solved one touch at a time. A single-finger touch leaves the
wrist's yaw and roll almost free, so neighbouring anchors can turn the hand in
opposite directions and the path between them twists. ``settle`` re-solves a pose
from where it is, keeping every contact on its key, while pulling the wrist
toward a target vector (the neighbours' orientation, or the wrist path for an
intermediate pose). A few Jacobi rounds over all anchors run in parallel.

``free_releases`` lets a key go earlier only where the pedal keeps the sound
identical (``pedal_release``), when the wrist would otherwise have to rush to its
next position. The audio does not change.
"""
import math

import numpy as np

from .checks import stretched
from .anchors import KEY_TRAVEL
from .kinematics import DIGITS, FLOAT
from .posture import natural_orientation
from .motion import SIDEWAYS_FREE, pad_residual, DEPART_DELAY, FLOOR_SCALE, MOVE_BASE_SECONDS, MOVE_SPEED, UNHURRIED, KeyFloor, KeySurface
from .pose import CONTACT_TOLERANCE, RELAXED, hand_shape, ROLL_BOUNDS, VALUE_SCALE, WEB_SCALE, YAW_BOUNDS, HandPose, PoseSolver, wrist_steps

NEIGHBOUR_SECONDS = .6          # gaussian width of the orientation smoothing over anchors
COHERENCE_ROUNDS = 2
ORIENTATION_PULLS = (np.array([.08, .03, .015, 1., 1.], dtype=FLOAT),   # x, y, z (m), yaw, roll (deg) per unit, toward the
                     np.array([.08, .03, .015, 4., 4.], dtype=FLOAT))   # target: tight first, looser if a key would be lost
LATERAL_SCALE = 6.              # deg per unit: a playing finger stays square to the hand; the hand moves over its key
THUMB_LATERAL_SCALE = 25.       # ... the thumb reaches sideways under the hand by technique
LATERAL_FREE = SIDEWAYS_FREE    # deg a finger swings sideways on its key without strain (motion.SIDEWAYS_FREE) ...
WEB_STRAIN = 4.                 # deg per unit two contact fingers open beyond the web's free opening (Rig.web)
LATERAL_HINGE = 4.              # ... beyond it, deg per unit: the hand moves over the key instead. Soft against the
                                # contact (1 mm off the key = 5 units), and a settle that loses a key is not taken
NEUTRAL_ORIENTATION = np.array([15., 12.], dtype=FLOAT)               # weak pull of yaw, roll toward the hand's natural
                                                         # orientation on its forearm (posture.natural_orientation)
POSITION_TRAVEL = .05
PALM_HEIGHT = .035              # the wrist at least this far above the white key tops while playing (m; the palm tilted up, arm.PALM_PITCH): the palm
PALM_SCALE = .004               # stays off the keys and the fingers reach down arched (m per unit, one-sided)
CENTRE_SCALE = .001             # pad pulled toward its key's centre line (m per unit): the finger's sides stay off the neighbours


def settle(solver, contacts, vector, values, target, pull, iterations=25, front=None):
    """Pose keeping ``contacts`` with the wrist pulled toward ``target`` (x, y, z, yaw, roll); ``front``: white-key
    pads preferably within this of their key's front edge (m) instead of ``WHITE_FRONT_CLEAR`` before the black keys.

    The contact fingers' skin stays above the key tops, seen as a continuous surface (``KeySurface``,
    bilinear), so a finger pressing a white key rises over the black keys behind it, raising the
    wrist when it must; the flat-topped cell lookup of the anchor solve would stall this local solve."""
    if not hasattr(solver, 'key_surface'):
        solver.key_surface = KeySurface(solver.heightfield) if solver.heightfield is not None else None
    surface = solver.key_surface
    if not hasattr(solver, 'white_top'):
        solver.white_top = KeyFloor(solver.regions).white_top
    vector = np.asarray(vector, FLOAT)
    target = np.asarray(target, FLOAT)
    values = {d: np.array(v, FLOAT) for d, v in values.items()}
    digits = sorted({c.digit for c in contacts})
    fingers = solver.rig.fingers[solver.hand]
    lower = np.r_[vector[:3] - POSITION_TRAVEL, YAW_BOUNDS[0], ROLL_BOUNDS[0],
                  np.concatenate([fingers[d].lower for d in digits]) if digits else []]
    upper = np.r_[vector[:3] + POSITION_TRAVEL, YAW_BOUNDS[1], ROLL_BOUNDS[1],
                  np.concatenate([fingers[d].upper for d in digits]) if digits else []]
    # start just inside the joint limits: a joint starting on its limit stalls the bounded solve (every step the
    # clip bends back fails), and the pose stays where it began
    margin = np.r_[np.zeros(3), np.full(len(lower) - 3, START_MARGIN)]
    x = np.clip(np.r_[vector, np.concatenate([values[d] for d in digits]) if digits else []], lower + margin, upper - margin)
    depths = {c.pitch: c.depth for c in contacts}
    stretch = stretched(depths)
    vals = dict(values)
    natural = natural_orientation(solver, vector[:3])
    free_digits = [d for d in DIGITS if d not in digits]
    shape = hand_shape(solver.span(depths))
    # free fingers preferably clear the keys keeping their curve (the knuckle opened FREE_CURVED_LIFT), so the wrist
    # stays high enough for them where the chord allows it; never at a cost to the contacts (re-settled without)
    curved_free = {d: shape[d] + np.array([0., FREE_CURVED_LIFT, 0., 0.], dtype=FLOAT) for d in free_digits}
    curved = [True]
    web_hard = solver.web_limit(digits)
    web_free = solver.web_limit(digits, solver.rig.web(solver.hand)[2])

    def unpack(v):
        for i, d in enumerate(digits):
            vals[d] = v[5 + 4 * i: 9 + 4 * i]
        return solver.palm(v[:3], v[3], v[4])

    def residuals(v):
        palm = unpack(v)
        out = []
        for c in contacts:
            point = solver.contact_point(solver.pad_world(c.digit, vals, palm))
            out.append(solver.regions.residual(c.pitch, c.depth, point) / CONTACT_TOLERANCE)
            out.append([(point[0] - solver.regions.centre(c.pitch)[0]) / CENTRE_SCALE])
            if not solver.regions.is_black(c.pitch):
                # a white key played close to the black keys catches their corners with the fingertip's sides
                edge = (solver.regions.black_front - WHITE_FRONT_CLEAR if front is None
                        else solver.regions.keys[c.pitch]['bounds_min'][1] + front)
                out.append([max(0., point[1] - edge) / WHITE_FRONT_SCALE])
            else:
                # a black key pressed at its very front edge: the pad slips off it, and its short travel is lost
                edge = solver.regions.keys[c.pitch]['bounds_min'][1] + BLACK_FRONT_IN
                out.append([max(0., edge - point[1]) / BLACK_FRONT_SCALE])
            out.append(pad_residual(solver, c.digit, vals[c.digit], palm))
            if c.digit == 1:
                # the thumb's last phalanx runs along its key (it plays with the side of its tip at the key's front);
                # lying across, its joint rides over the next key and sinks into it as the key goes down
                out.append([thumb_tip_angle(solver, vals, palm) / THUMB_TIP_SCALE])
            if surface is not None:
                skin = (palm @ solver.local_clearance(c.digit, vals))[:3].T
                out.append(surface.clearance(skin, depths, c.pitch, brush=stretch) / FLOOR_SCALE)
        if surface is not None:
            # Room for the fingers not on a key: lifted (as far as a free finger lifts) they clear the keys,
            # so a hand rolled or lowered onto its little-finger side comes back up.
            for d in free_digits:
                skin = (palm @ solver.local_clearance(d, {**vals, d: FREE_LIFTED[d != 1]}))[:3].T
                out.append(np.maximum(0., surface(skin, depths, None) - skin[:, 2]) / FREE_CLEAR_SCALE)
                if curved[0]:
                    skin = (palm @ solver.local_clearance(d, {**vals, d: curved_free[d]}))[:3].T
                    out.append(np.maximum(0., surface(skin, depths, None) - skin[:, 2]) / FREE_CURVED_SCALE)
        out.append(_order(solver, contacts, vals))
        out.append(np.array([vals[c.digit][0] / (THUMB_LATERAL_SCALE if c.digit == 1 else LATERAL_SCALE) for c in contacts], dtype=FLOAT))
        out.append(lateral_hinge(digits, vals))
        # the web between the contact fingers: a wall at its limit, strain beyond its free opening
        out.append(web_hard(vals))
        out.append(web_free(vals) * (WEB_SCALE / WEB_STRAIN))
        out.append((v[:5] - target) / pull)
        out.append([max(0., solver.white_top + PALM_HEIGHT - v[2]) / PALM_SCALE])
        out.append((v[3:5] - natural) / NEUTRAL_ORIENTATION)
        for d in digits:
            out.append((vals[d] - shape[d]) / VALUE_SCALE)
        return np.concatenate(out)

    def contact_error(v):
        palm = unpack(v)
        return max((float(np.max(np.abs(solver.regions.residual(c.pitch, c.depth,
                                                                solver.contact_point(solver.pad_world(c.digit, vals, palm))))))
                    for c in contacts), default=0.)

    steps = wrist_steps(len(digits))
    start = x.copy()
    x, used = PoseSolver._levenberg_marquardt(residuals, start, lower, upper, steps, iterations)
    if free_digits and surface is not None and contact_error(x) > CONTACT_TOLERANCE * 2.5:
        curved[0] = False      # the curved free fingers held the hand off its keys (an octave's middle fingers)
        x, more = PoseSolver._levenberg_marquardt(residuals, start, lower, upper, steps, iterations)
        used += more
    palm = unpack(x)
    error, targets, clearance = 0., {}, 0.
    for c in contacts:
        point = solver.contact_point(solver.pad_world(c.digit, vals, palm))
        targets[c.digit] = point
        error = max(error, float(np.max(np.abs(solver.regions.residual(c.pitch, c.depth, point)))))
        pen = solver.clearance_residual(c.digit, vals, palm, depths, c.pitch)
        clearance = max(clearance, float(pen.max()) * .0005 if len(pen) else 0.)
    return HandPose(np.array(x[:3], dtype=FLOAT), float(x[3]), float(x[4]), {d: np.array(v, dtype=FLOAT) for d, v in vals.items()},
                    error, error <= CONTACT_TOLERANCE * 2.5, used, targets, clearance)


FREE_LIFTED = (np.array([0., 20., 8., 8.], dtype=FLOAT),     # a free thumb and finger lifted clear of the keys
               np.array([0., 20., 20., 10.], dtype=FLOAT))   # (lateral, lift, PIP, DIP)
FREE_CLEAR_SCALE = .002         # m per unit: its skin below the key tops
FREE_CURVED_LIFT = 10.          # deg a free finger keeping its curve opens its knuckle to clear the keys ...
FREE_CURVED_SCALE = .01         # ... m per unit (a preference: far below a contact, 1 mm off its key = 5 units)
WHITE_FRONT_CLEAR = .012        # white-key pads preferably this far in front of the black keys ...
WHITE_FRONT_SCALE = .002        # ... m per unit (soft: a chord may still need them further in)
BLACK_FRONT_IN = .012           # black-key pads preferably this far behind the black key's front ...
BLACK_FRONT_SCALE = .002        # ... m per unit (soft, like WHITE_FRONT_CLEAR)
THUMB_TIP_SCALE = 10.          # deg per unit: the thumb's last phalanx off its key's long axis
ORDER_SCALE = .002              # m per unit: chord fingers 2..5 keep their order across the palm


WIDE_STARTS = [np.array([0., dy, dz, dyaw, 0.], dtype=FLOAT)    # wrist starts tried for a stretched hand: back from the
               for dy in (0., -.02, -.04) for dz in (0., -.015, -.03) for dyaw in (-10., 0., 10.)]   # keys, lower, turned
WIDE_START_PULL = np.array([.005, .005, .005, 3., 3.], dtype=FLOAT)   # m, deg per unit: the wrist taken to a start
WIDE_FRONT = .012               # ... each also with its white-key pads near the key's front edge (an octave's thumb and
                                # little finger play the front of their keys: the hand's widest span is at the fingertips)
WIDE_COARSE_ITERATIONS = 15     # every start is taken this far and scored (its rank barely changes after) ...
WIDE_KEEP = 3                   # ... and the best few settled fully
WIDE_ITERATIONS = 60
THUMB_SINK_WEIGHT = 2.          # the thumb's side lying on the next key reads worst
START_MARGIN = .1               # deg: a settle starts its angles this far inside their limits (``settle``)
WIDE_CLEAN = 1.                 # mm: a stretched hand whose plain settle sinks no deeper than this is not searched
WIDE_CLEAN_CONTACT = .0005      # m: ... when it is this close to its keys (off them, it is searched anyway)


def sunk(solver, contacts, pose):
    """Per contact: mm its finger's skin sinks into keys other than its own."""
    palm = solver.palm(pose.wrist, pose.yaw, pose.roll)
    depths = {c.pitch: c.depth for c in contacts}
    out = []
    for c in contacts:
        skin = (palm @ solver.local_clearance(c.digit, pose.values))[:3].T
        out.append(float(solver.heightfield.penetration(skin, depths, (c.pitch, np.ones(len(skin), bool))).max(initial=0.)) * 1000)
    return out


def sink(solver, contacts, pose):
    """Score of a pose: mm the contact fingers' skin sinks into keys other than their own (the thumb's weighted),
    plus a tenth of the thumb's last phalanx angle off its key (deg)."""
    score = sum(mm * (THUMB_SINK_WEIGHT if c.digit == 1 else 1.) for c, mm in zip(contacts, sunk(solver, contacts, pose)))
    if any(c.digit == 1 for c in contacts):
        score += abs(thumb_tip_angle(solver, pose.values, solver.palm(pose.wrist, pose.yaw, pose.roll))) / 10
    return score


def inner(contacts):
    """Indices of the contacts between the chord's lowest and highest key: the fingers the hand's span leaves free
    to choose (the outer two set the span)."""
    if len(contacts) < 3:
        return []
    pitches = [c.pitch for c in contacts]
    return [i for i, p in enumerate(pitches) if min(pitches) < p < max(pitches)]


def wide_settle(solver, contacts, vector, values, target, pull, limit):
    """``settle`` for a stretched hand (``stretched``: an octave). The local solve stays in the basin it starts in, and
    a wide hand has several: the thumb's joint lying across the next key, or the wrist back and low with the thumb
    along its key. Settle from each of ``WIDE_STARTS`` (and near the keys' front edge), then toward ``target``; keep
    the pose within ``limit`` whose fingers sink least (``sink``), else the closest. Every start is scored after
    ``WIDE_COARSE_ITERATIONS``; the ``WIDE_KEEP`` best go on. A plain settle on its keys (``WIDE_CLEAN_CONTACT``) that sinks
    nowhere (``WIDE_CLEAN``) is kept without the search; one off its keys is no reason to stop looking. None without a heightfield."""
    if solver.heightfield is None:
        return None
    vector = np.asarray(vector, FLOAT)
    plain = settle(solver, contacts, vector, values, target, pull)
    if plain.contact_error <= min(limit, WIDE_CLEAN_CONTACT) and max(sunk(solver, contacts, plain)) <= WIDE_CLEAN:
        return plain                     # on its keys and nothing sinks where it starts: no other basin to look for
    coarse = []
    for front in (None, WIDE_FRONT):
        for start in WIDE_STARTS:
            p = settle(solver, contacts, vector, values, vector + start, WIDE_START_PULL, WIDE_COARSE_ITERATIONS, front)
            coarse.append((sink(solver, contacts, p), front, p))
    coarse.sort(key=lambda c: c[0])
    return least_sunk(solver, contacts, [settle(solver, contacts, vector_of(p), p.values, target, pull, WIDE_ITERATIONS, front)
                                         for _, front, p in coarse[:WIDE_KEEP]], limit)


def least_sunk(solver, contacts, poses, limit):
    """Of ``poses``, the one within ``limit`` whose fingers sink least (``sink``), else the closest."""
    return min(poses, key=lambda p: (0, sink(solver, contacts, p)) if p.contact_error <= limit else (1, p.contact_error))


def thumb_tip_angle(solver, values, palm):
    """Top-view angle (deg) between the thumb's last bone and the keys' long axis."""
    name = solver.rig.fingers[solver.hand][1].names[-1]
    y = (palm @ solver.rig.bone_matrices(solver.hand, solver.identity, values, [name])[name])[:3, 1]
    return math.degrees(math.atan2(y[0], y[1]))


def lateral_hinge(digits, values):
    """Sideways swing of the contact fingers beyond ``LATERAL_FREE`` (scaled): an octave's spread is shared by the
    thumb and the little finger instead of one of them going to its limit."""
    return np.array([max(0., abs(values[d][0]) - LATERAL_FREE[d]) / LATERAL_HINGE for d in digits], dtype=FLOAT)


def _order(solver, contacts, vals):
    """Crossed or squeezed chord fingers (not the thumb): lateral gap of either finger half (proximal, distal)
    below 60% of relaxed (motion.Spacing.positions)."""
    if not hasattr(solver, 'spacing'):
        from .motion import Spacing
        solver.spacing = Spacing(solver)
    sp = solver.spacing
    digits = sorted(c.digit for c in contacts if c.digit >= 2)
    out = []
    for a, b in zip(digits, digits[1:]):
        need = sum(sp.whole_gaps[d] for d in range(a, b)) / .85 * .6
        out.append(np.maximum(0., need - (sp.positions(b, vals) - sp.positions(a, vals))) / ORDER_SCALE)
    return np.concatenate(out) if out else np.zeros(0, dtype=FLOAT)


def vector_of(pose):
    return np.r_[pose.wrist, pose.yaw, pose.roll]


def neighbour_targets(anchors, offsets=None):
    """Per anchor (same hand, time order): own position, orientation smoothed over its neighbours in time.
    ``offsets`` (yaw, roll per anchor) are added first: the hand's natural orientation on its forearm, for
    anchors still solved square to the keys."""
    out = []
    times = np.array([a.time for a in anchors])
    vectors = np.array([vector_of(a.pose) for a in anchors], dtype=FLOAT)
    if offsets is not None:
        vectors[:, 3:] += np.asarray(offsets, FLOAT)
    for i, a in enumerate(anchors):
        w = np.exp(-.5 * ((times - a.time) / NEIGHBOUR_SECONDS) ** 2)
        target = vectors[i].copy()
        target[3:] = (w[:, None] * vectors[:, 3:]).sum(axis=0) / w.sum()
        out.append(target)
    return out


MOTION_CUT_SECONDS = 0.         # a key may sound at most this much shorter so the wrist is not forced to jump


def free_releases(anchors, notes, strokes, max_cut=MOTION_CUT_SECONDS):
    """Release keys earlier so the wrist can travel to its next position at a natural pace.

    Lifts that keep the sound identical (down to ``pedal_release``) are taken for an unhurried move.
    Where the wrist still could not make even its natural duration, the key is lifted earlier with
    an audible shortening of at most ``max_cut``, as a pianist does before a leap.
    ``anchors``: one hand's anchors with poses; ``strokes``: stroke timing by note.
    Returns (sound-identical lifts, audible lifts)."""
    anchors = sorted(anchors, key=lambda a: a.time)
    identical = audible = 0
    for a, b in zip(anchors, anchors[1:]):
        keep = set(b.held)
        ready = min((strokes[k].hover for k in b.incoming), default=b.time)
        natural = MOVE_BASE_SECONDS + float(np.linalg.norm(b.pose.wrist - a.pose.wrist)) / MOVE_SPEED
        unhurried_leave = ready - UNHURRIED * natural - DEPART_DELAY
        needed_leave = ready - natural - DEPART_DELAY
        for k in a.incoming + a.held:
            n = notes[k]
            if k in keep or n.release <= unhurried_leave:
                continue
            lowest = max(n.earliest_release, n.onset + .05)
            new = min(n.release, max(n.pedal_release, unhurried_leave, lowest))
            if new > needed_leave + 1e-6 and max_cut > 0:
                new = min(new, max(needed_leave, n.release - max_cut, lowest))
            if new < n.release - 1e-6:
                if new >= n.pedal_release - 1e-9:
                    identical += 1
                else:
                    audible += 1
                n.release = new
    for a in anchors:
        a.held = tuple(k for k in a.held if notes[k].held_at(a.time))
    return identical, audible
