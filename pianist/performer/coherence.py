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
import numpy as np

from .checks import stretched
from .anchors import KEY_TRAVEL
from .kinematics import DIGITS
from .posture import natural_orientation
from .motion import SIDEWAYS_FREE, PLAYING_SHAPE, pad_residual, DEPART_DELAY, FLOOR_SCALE, MOVE_BASE_SECONDS, MOVE_SPEED, UNHURRIED, KeyFloor, KeySurface
from .pose import CONTACT_TOLERANCE, RELAXED, ROLL_BOUNDS, VALUE_SCALE, YAW_BOUNDS, HandPose, PoseSolver

NEIGHBOUR_SECONDS = .6          # gaussian width of the orientation smoothing over anchors
COHERENCE_ROUNDS = 2
ORIENTATION_PULLS = (np.array([.08, .03, .015, 1., 1.]),   # x, y, z (m), yaw, roll (deg) per unit, toward the
                     np.array([.08, .03, .015, 4., 4.]))   # target: tight first, looser if a key would be lost
LATERAL_SCALE = 6.              # deg per unit: a playing finger stays square to the hand; the hand moves over its key
THUMB_LATERAL_SCALE = 25.       # ... the thumb reaches sideways under the hand by technique
LATERAL_FREE = SIDEWAYS_FREE    # deg a finger swings sideways on its key without strain (motion.SIDEWAYS_FREE) ...
LATERAL_HINGE = 4.              # ... beyond it, deg per unit: the hand moves over the key instead. Soft against the
                                # contact (1 mm off the key = 5 units), and a settle that loses a key is not taken
NEUTRAL_ORIENTATION = np.array([15., 12.])               # weak pull of yaw, roll toward the hand's natural
                                                         # orientation on its forearm (posture.natural_orientation)
POSITION_TRAVEL = .05
PALM_HEIGHT = .045              # the wrist at least this far above the white key tops while playing (m): the palm
PALM_SCALE = .004               # stays off the keys and the fingers reach down arched (m per unit, one-sided)
CENTRE_SCALE = .002             # pad pulled toward its key's centre line (m per unit): the finger's sides stay off the neighbours


def settle(solver, contacts, vector, values, target, pull, iterations=25):
    """Pose keeping ``contacts`` with the wrist pulled toward ``target`` (x, y, z, yaw, roll).

    The contact fingers' skin stays above the key tops, seen as a continuous surface (``KeySurface``,
    bilinear), so a finger pressing a white key rises over the black keys behind it, raising the
    wrist when it must; the flat-topped cell lookup of the anchor solve would stall this local solve."""
    if not hasattr(solver, 'key_surface'):
        solver.key_surface = KeySurface(solver.heightfield) if solver.heightfield is not None else None
    surface = solver.key_surface
    if not hasattr(solver, 'white_top'):
        solver.white_top = KeyFloor(solver.regions).white_top
    vector = np.asarray(vector, float)
    target = np.asarray(target, float)
    values = {d: np.array(v, float) for d, v in values.items()}
    digits = sorted({c.digit for c in contacts})
    fingers = solver.rig.fingers[solver.hand]
    lower = np.r_[vector[:3] - POSITION_TRAVEL, YAW_BOUNDS[0], ROLL_BOUNDS[0],
                  np.concatenate([fingers[d].lower for d in digits]) if digits else []]
    upper = np.r_[vector[:3] + POSITION_TRAVEL, YAW_BOUNDS[1], ROLL_BOUNDS[1],
                  np.concatenate([fingers[d].upper for d in digits]) if digits else []]
    x = np.clip(np.r_[vector, np.concatenate([values[d] for d in digits]) if digits else []], lower, upper)
    depths = {c.pitch: c.depth for c in contacts}
    stretch = stretched(depths)
    vals = dict(values)
    natural = natural_orientation(solver, vector[:3])
    free_digits = [d for d in DIGITS if d != 1 and d not in digits]     # the thumb hangs below by design

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
                edge = solver.regions.black_front - WHITE_FRONT_CLEAR
                out.append([max(0., point[1] - edge) / WHITE_FRONT_SCALE])
            out.append(pad_residual(solver, c.digit, vals[c.digit], palm))
            if surface is not None:
                skin = (palm @ solver.local_clearance(c.digit, vals))[:3].T
                out.append(surface.clearance(skin, depths, c.pitch, brush=stretch) / FLOOR_SCALE)
        if surface is not None:
            # Room for the fingers not on a key: lifted (as far as a free finger lifts) they clear the keys,
            # so a hand rolled or lowered onto its little-finger side comes back up.
            for d in free_digits:
                skin = (palm @ solver.local_clearance(d, {**vals, d: FREE_LIFTED}))[:3].T
                out.append(np.maximum(0., surface(skin, depths, None) - skin[:, 2]) / FREE_CLEAR_SCALE)
        out.append(_order(solver, contacts, vals))
        out.append(np.array([vals[c.digit][0] / (THUMB_LATERAL_SCALE if c.digit == 1 else LATERAL_SCALE) for c in contacts]))
        out.append(lateral_hinge(digits, vals))
        out.append((v[:5] - target) / pull)
        out.append([max(0., solver.white_top + PALM_HEIGHT - v[2]) / PALM_SCALE])
        out.append((v[3:5] - natural) / NEUTRAL_ORIENTATION)
        for d in digits:
            out.append((vals[d] - PLAYING_SHAPE[d]) / VALUE_SCALE)
        return np.concatenate(out)

    steps = np.r_[1e-5, 1e-5, 1e-5, 1e-2, 1e-2, np.full(4 * len(digits), 1e-2)]
    x, used = PoseSolver._levenberg_marquardt(residuals, x, lower, upper, steps, iterations)
    palm = unpack(x)
    error, targets, clearance = 0., {}, 0.
    for c in contacts:
        point = solver.contact_point(solver.pad_world(c.digit, vals, palm))
        targets[c.digit] = point
        error = max(error, float(np.max(np.abs(solver.regions.residual(c.pitch, c.depth, point)))))
        pen = solver.clearance_residual(c.digit, vals, palm, depths, c.pitch)
        clearance = max(clearance, float(pen.max()) * .0005 if len(pen) else 0.)
    return HandPose(np.array(x[:3]), float(x[3]), float(x[4]), {d: np.array(v) for d, v in vals.items()},
                    error, error <= CONTACT_TOLERANCE * 2.5, used, targets, clearance)


FREE_LIFTED = np.array([0., 20., 20., 10.])   # a free finger lifted clear of the keys (lateral, lift, PIP, DIP)
FREE_CLEAR_SCALE = .002         # m per unit: its skin below the key tops
WHITE_FRONT_CLEAR = .012        # white-key pads preferably this far in front of the black keys ...
WHITE_FRONT_SCALE = .002        # ... m per unit (soft: a chord may still need them further in)
ORDER_SCALE = .002              # m per unit: chord fingers 2..5 keep their order across the palm


def lateral_hinge(digits, values):
    """Sideways swing of the contact fingers beyond ``LATERAL_FREE`` (scaled): an octave's spread is shared by the
    thumb and the little finger instead of one of them going to its limit."""
    return np.array([max(0., abs(values[d][0]) - LATERAL_FREE[d]) / LATERAL_HINGE for d in digits])


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
    return np.concatenate(out) if out else np.zeros(0)


def vector_of(pose):
    return np.r_[pose.wrist, pose.yaw, pose.roll]


def neighbour_targets(anchors, offsets=None):
    """Per anchor (same hand, time order): own position, orientation smoothed over its neighbours in time.
    ``offsets`` (yaw, roll per anchor) are added first: the hand's natural orientation on its forearm, for
    anchors still solved square to the keys."""
    out = []
    times = np.array([a.time for a in anchors])
    vectors = np.array([vector_of(a.pose) for a in anchors])
    if offsets is not None:
        vectors[:, 3:] += np.asarray(offsets, float)
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
