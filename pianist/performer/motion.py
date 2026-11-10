"""Continuous motion from the anchor poses: wrist path, finger strokes and per-frame finger IK.

Wrist: the hand rests while it holds keys it still needs, leaves when they are
released and travels unhurried (minimum-jerk, with a small arc over the keys) to
the next position, where it is settled before the finger begins its stroke.
Knots are joined by a monotone cubic; a segment between two resting knots has
the quintic minimum-jerk shape. Where a key held through a wrist move would be
lost, an intermediate whole-hand pose is solved and becomes a knot
(``refine_points``): the wrist path itself carries the contacts, the wrist is
never corrected frame by frame.

Fingers, per note: the finger waits above its key, strikes with an accelerating
stroke whose speed at the key top equals the key's own speed into the keybed,
rides the key down and holds it, and at the release leaves with the key's
spring-back and rises past the key top. Strike, ride and lift are pad paths
solved by IK every frame (smooth targets, deterministic solve from a smooth
seed); between notes the finger moves in joint space with a minimum-jerk
profile, relaxing in long gaps. Every frame is computed from fixed data, never
from the previous frame, so time segments are independent and the parallel
result equals the serial one.
"""
from dataclasses import dataclass, replace

import numpy as np

from planner.music import KEY_RETURN_SECONDS

from .anchors import KEY_TRAVEL, ease, lift_profile
from .checks import CELL, OWN_KEY_TOLERANCE, Heightfield, brushed, depth_table, stretched
from .kinematics import DIGITS, FLOAT
from .pose import ARCHED, RELAXED, PoseSolver, hand_shape

MOVE_BASE_SECONDS = .10          # natural wrist move: base duration ...
MOVE_SPEED = .9                  # ... plus distance / this speed (m/s)
UNHURRIED = 2.                   # a move with time to spare takes this many natural durations
MIN_TRAVEL_SECONDS = .25
MIN_MOVE_SECONDS = .03           # shortest wrist move the planner's own timing can force
STROKE_JOIN_SHARE = .4           # a stroke's clearance eases in over this share of its approach (and out of its lift)
ACCENT_VELOCITY = (65, 95)       # loudness from which the wrist sinks into a chord, and where it sinks fully
ACCENT_DEPTH = .004              # deepest sink (m), a full chord at forte; the arm's weight, not a bounce
ACCENT_SINGLE = .5               # a single loud note sinks this share
ACCENT_LEAD = .03                # the sink starts this long before the touch (s) ...
ACCENT_RECOVER = .2              # ... and the wrist is back this long after the onset
RIDE_STEADY = 3.                 # deg per unit: a riding finger prefers the previous frame's shape (no flicker
                                 # between near-equal solutions on the piecewise-flat key tops)
RIDE_SWITCH = .0005              # the fresh solution replaces the warm one only when this much better (m)
CORRECTION_FADE = .04            # key-clearance and room corrections fade out over this as a stroke reaches its key (s)
ANTICIPATE_SECONDS = .06         # a hurried leap sets off this long before the key is released ...
ARRIVE_LATE_SECONDS = .05        # ... and lands this long after the next touch (within the press)
DEPART_DELAY = .02               # after the last needed key is released
ARC_RATIO, ARC_MAX = .15, .04    # wrist rises by this fraction of the travel (m), at most this
LIFT_RATE = .3                   # m/s: a hurried leap lifts only between the release and the next touch, at most
                                 # this times that time (a hop, not a slide over the keys)
ARC_MIN_DISTANCE = .02
# Strike or crawl, per chord of a hand (measured on the reference pianist, references/data/for_elise_060):
# the hand crawls (fingers walk, the wrist stays level) while a key stays held or the notes join legato;
# it lifts and drops into the next chord when the same finger plays again, on a chord, on a position change
# or after a rest. How high is set by the time the hand is free (release .. next touch), not the reason.
LEGATO_MARGIN = .03              # a release this close before the next touch (or later) joins the notes
STRIKE_REST = .3                 # a free hand this long before its next touch lifts even for a single note
STRIKE_SHIFT = .025              # wrist travel (m) that counts as a position change
LIFT_WINDOWS = (.12, .25, .4, .7, 1.)          # free time (s) ...
LIFT_HEIGHTS = (0., .003, .01, .04, .065)      # ... and the wrist's rise (m) at a middle loudness: about a third above the
                                               # reference's knuckle lift, which read as too timid on the character
LIFT_LOUDNESS = ((40., 110.), (.75, 1.35))     # velocity of the chord struck -> share of that rise (a loud chord from higher)
LIFT_SCALE = 1.                  # the piece's own share (inputs/<piece>/motion.json "lift"): its character, lyrical to spirited
LEAD_IN_SECONDS = 1.2            # from the rest pose to the first anchor
LIFT_HEIGHT = .008               # pad above the key top at the end of a lift (m)
HOVER_HEIGHT = (.002, .025)      # pad above the key before the stroke: soft starts on the key, loud from above
RELAX_GAP_SECONDS = .6           # free time after which a finger returns to the relaxed hand
PREPARE_SECONDS = .22            # a rested finger takes its shape for the next key this long before the stroke
RETURN_SECONDS = .3              # ... and returns to the relaxed hand within this after a lift
RIDE_RETRY = .001                # a warm-started ride further than this off its key is solved again from the anchor
RIDE_SLIP = .004                 # m per unit: a riding pad may slip toward its key's front (soft) rather than wedge its
                                 # finger between the black keys or across the next one; never back, never sideways
RIDE_FRONT_KEEP = .008           # ... staying this far behind the key's front edge
IK_ITERATIONS = 20
ROOM_WORSE = .001                # m: a room correction sinking a finger this much deeper than without it is cut back ...
ROOM_SHARES = (0., .25, .5, .75)  # ... to the share of it that sinks least
STROKE_CLEAR_STEP = .01          # grid of the strike / lift clearance (s)
STROKE_CLEAR_FADE = .02          # ... faded out this close to the touch and the release
STROKE_CLEAR_TOLERANCE = .0005   # skin this far into another key is left alone (m)
STROKE_CLEAR_SMOOTH = .025       # gaussian width spreading a correction over neighbouring grid points (s)
FLOOR_SCALE = .0005              # skin below the key surface, metres per residual unit
FREE_VALUE_SCALE = 20.           # a free finger leaves its keyposed values by as little as possible (deg per unit)
IK_VALUE_SCALE = 15.             # prior toward the seed: fixes the finger's free direction (3 targets, 4 values)
# How each finger rises off its key by itself (anatomy, and the reference mocap's free fingers): the index lifts high
# from its knuckle keeping its curve; the middle and little fingers lift less and open the knuckle and middle joint
# together (a flatter finger); the ring finger hardly lifts on its own. FINGER_REACH: share of the hover / lift height
# a finger takes on its own (the rest of a loud stroke comes from the hand, strike_lift). FINGER_JOINTS: how freely
# each value (sideways, knuckle, middle, last) leaves its shape when the finger moves off the key (x the scales).
FINGER_REACH = {1: 1., 2: 1., 3: .6, 4: .35, 5: .6}
# A free finger that would dip into the keys rises along its own path (deg per unit, from the shape it waits in):
# the index from its knuckle keeping its curve, the others opening knuckle and middle joint together (flatter, the
# middle joint staying below the knuckle; curling up instead cannot clear the keys). The smallest step that clears
# is taken; a finger the path cannot clear falls back to the free solve.
LIFT_PATH = {2: np.array([0., 50., 0., 0.], dtype=FLOAT), 3: np.array([0., 35., -30., -10.], dtype=FLOAT),
             4: np.array([0., 35., -30., -10.], dtype=FLOAT), 5: np.array([0., 35., -30., -10.], dtype=FLOAT)}
LIFT_PATH_STEPS = 12
FINGER_JOINTS = {1: np.ones(4, dtype=FLOAT),
                 2: np.array([1., 1.6, .5, .5], dtype=FLOAT),
                 3: np.array([1., 1.2, 1.2, .8], dtype=FLOAT),
                 4: np.array([1., 1., .7, .7], dtype=FLOAT),
                 5: np.array([1., 1.2, 1.2, .8], dtype=FLOAT)}
FLOOR_RAMP = .015                # the key floor phases in over a stroke: lowered this much at the key (m)
SPACING_SHARE = .85              # free fingers keep this share of the relaxed fingertip spacing from their neighbours
SPACING_SOFTNESS = .002          # smooth maximum width (m)
SPREAD_COUPLING = .5             # share of a neighbour's stretch beyond the relaxed gap a free finger follows (tendons)
SPREAD_ROUNDS = 4                # ... passed on along the hand in this many rounds
WEB_ROUNDS = 24                  # rounds sharing out openings beyond the web (Spacing.web_fan)
SPREAD_SOFTNESS = .0005          # its onset eased over this (m): no pull at the relaxed gap
CLEAR_STEP = .05                 # free-finger key clearance is solved at this spacing and interpolated (s)
HOLD_SAMPLES = 3                 # ... each sample then holds the largest correction within this many samples (0.15 s)
CLEAR_RETRY = .001               # a free finger still this far into the keys after clearing is retried from its relaxed curl (m)
REFINE_STEP = .04                # contact check spacing along the wrist path (s)
REFINE_TOLERANCE = .0015         # held-key pad error that calls for an intermediate pose (m) ...
SIDEWAYS_FREE = {1: 30., 2: 8., 3: 8., 4: 10., 5: 20.}   # deg a finger swings sideways on its key without strain
                                                 # (thumb and little finger spread for an octave, sharing it); coherence
                                                 # prefers less; the thumb does not call for intermediate poses
REFINE_LATERAL = 7.              # ... or a finger 2..5 swung sideways this far beyond SIDEWAYS_FREE on its key (deg) ...
REFINE_FREE = .001               # ... or a free finger this far into the keys, lifted as far as it goes (m): the hand must rise
REFINE_FREE_NEAR = .15           # a finger counts as free this long after its release and before its touch (s)
LIFT_LATERAL_KEEP = .35          # a lifted finger lets its sideways swing go to this share, over the free time after
LIFT_LATERAL_SOON = (.15, .45)   # its lift, when its next stroke is further than this away (s; none when sooner) ...
LIFT_LATERAL_SECONDS = .2        # ... within at most this (the lift itself is too quick for it)
APPROACH_SECONDS = .15          # a finger takes its key's sideways swing over this before its stroke ...
APPROACH_MIN_TRAVEL = .005      # ... when the wrist travels further than this (m) to its touch
TRANSITION_SPEED = 400.          # deg/s: a finger's lifted shape and its next hover shape (solved apart) meet at
                                 # most this fast; when one finger's notes follow closely, both shapes meet halfway


# The playing hand's shape (arched, flattened by a wide span) is pose.hand_shape, shared with the planner.
PAD_PITCH_MAX = 50.              # deg: a last phalanx steeper than this meets the key with the fingertip (slips);
                                 # measured on the rig: the pad's lowest point is mid-pad below ~30, the tip above ~40;
                                 # an arched hand plays on the front of the pad, between the two
PAD_PITCH_SCALE = 5.             # deg per unit of the pad-contact residual


def _split(vector):
    return vector[:3], vector[3], vector[4]


# ---------- wrist ----------
def held(deltas, k=HOLD_SAMPLES):
    """Free-time corrections are solved per sample, and a finger skimming a key edge (the wrist drifting slowly)
    is lifted at one sample and not the next: a curve through them trembles. Each sample takes
    the largest correction within +-k samples, then a gaussian of k/2: the finger lifts once and stays up."""
    deltas = np.asarray(deltas, FLOAT)
    n = len(deltas)
    if n < 3 or k < 1:
        return deltas
    size = np.linalg.norm(deltas, axis=1)
    pick = [lo + int(np.argmax(size[lo:hi])) for lo, hi in ((max(0, i - k), min(n, i + k + 1)) for i in range(n))]
    out = deltas[pick]
    w = np.exp(-.5 * (np.arange(-k, k + 1) / (k / 2.)) ** 2)
    w /= w.sum()
    padded = np.concatenate([out[:1].repeat(k, 0), out, out[-1:].repeat(k, 0)])
    return np.stack([np.convolve(padded[:, j], w, mode='valid') for j in range(out.shape[1])], axis=1)


def pchip_slopes(times, values):
    """Fritsch-Carlson monotone cubic slopes; ``values`` (n, k)."""
    h = np.diff(times)
    delta = np.diff(values, axis=0) / h[:, None]
    slopes = np.zeros_like(values)
    for i in range(1, len(times) - 1):
        a, b = delta[i - 1], delta[i]
        w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
        same = a * b > 0
        slopes[i, same] = (w1 + w2) / (w1 / a[same] + w2 / b[same])
    return slopes


class WristPath:
    """Wrist vector (x, y, z, yaw, roll) over time from knots (time, vector, arc of the following segment);
    an arc is a height, or (height, from, to): the rise only within those times of the segment."""

    def __init__(self, knots):
        times, values, arcs = [], [], []
        for t, v, arc in sorted(knots, key=lambda k: k[0]):
            arc = tuple(arc) if isinstance(arc, tuple) else (arc, np.nan, np.nan)
            if times and t <= times[-1] + 1e-6:
                values[-1], arcs[-1] = np.asarray(v, FLOAT), arc
                continue
            times.append(t)
            values.append(np.asarray(v, FLOAT))
            arcs.append(arc)
        self.times, self.values, self.arcs = np.array(times, float), np.array(values, dtype=FLOAT), np.array(arcs, dtype=FLOAT)
        self.slopes = pchip_slopes(self.times, self.values) if len(times) > 1 else np.zeros_like(self.values)
        self.resting = np.all(np.abs(self.slopes) < 1e-12, axis=1)
        self.accents = np.zeros((0, 3))          # (start, onset, depth): the wrist sinks into loud chords

    def set_accents(self, notes):
        """Arm weight in loud chords: the wrist sinks from shortly before the touch to the onset and
        springs back after (deepest of overlapping accents). ``notes``: this hand's NoteMotions."""
        chords = {}
        for n in notes:
            chords.setdefault(round(n.onset, 3), []).append(n)
        rows = []
        for onset, ns in sorted(chords.items()):
            loud = min(max((max(n.velocity for n in ns) - ACCENT_VELOCITY[0]) /
                           (ACCENT_VELOCITY[1] - ACCENT_VELOCITY[0]), 0.), 1.)
            if loud > 0:
                depth = ACCENT_DEPTH * loud * (1. if len(ns) > 1 else ACCENT_SINGLE)
                rows.append((min(n.touch for n in ns) - ACCENT_LEAD, onset, depth))
        self.accents = np.array(rows, float) if rows else np.zeros((0, 3))

    def dip(self, t):
        if not len(self.accents):
            return 0.
        a = self.accents
        near = (a[:, 0] <= t) & (t <= a[:, 1] + ACCENT_RECOVER)
        out = 0.
        for start, onset, depth in a[near]:
            x = (t - start) / max(onset - start, 1e-6) if t < onset else 1. - (t - onset) / ACCENT_RECOVER
            out = max(out, depth * ease(x))
        return out

    def __call__(self, t):
        if len(self.times) == 1:
            return self.values[0].copy()
        i = int(np.clip(np.searchsorted(self.times, t, side='right') - 1, 0, len(self.times) - 2))
        h = self.times[i + 1] - self.times[i]
        s = min(max((t - self.times[i]) / h, 0.), 1.)
        a, b = self.values[i], self.values[i + 1]
        if self.resting[i] and self.resting[i + 1]:
            out = a + (b - a) * ease(s)
        else:
            h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
            out = h00 * a + h10 * h * self.slopes[i] + h01 * b + h11 * h * self.slopes[i + 1]
        arc, lo, hi = self.arcs[i]
        if not np.isnan(lo):
            s = min(max((t - lo) / max(hi - lo, 1e-6), 0.), 1.)
        out[2] += arc * 16 * s * s * (1 - s) * (1 - s) - self.dip(t)
        return out


def _vector(anchor):
    return np.r_[anchor.pose.wrist, anchor.pose.yaw, anchor.pose.roll]


def _arc(va, vb, free):
    d = float(np.linalg.norm(vb[:2] - va[:2]))
    return min(ARC_MAX, ARC_RATIO * d) if free and d > ARC_MIN_DISTANCE else 0.


def strike_lift(a, b, notes, free, keep, travel):
    """Wrist rise (m) before anchor ``b``'s chord when the hand strikes it; 0 when it crawls. ``free``: when
    the hand let go of its last key not kept; ``keep``: keys held through; ``travel``: wrist distance (m)."""
    if keep or not b.incoming:
        return 0.
    touch = min(notes[k].touch for k in b.incoming)
    window = touch - free
    legato = window < LEGATO_MARGIN
    shift = travel > STRIKE_SHIFT
    if legato and not shift:
        return 0.
    same = {notes[k].digit for k in b.incoming} & {notes[k].digit for k in a.incoming + a.held}
    chord = len(b.incoming) > 1
    if not (same or chord or shift or window >= STRIKE_REST):
        return 0.
    loud = np.interp(max(notes[k].velocity for k in b.incoming), *LIFT_LOUDNESS)
    return float(np.interp(window, LIFT_WINDOWS, LIFT_HEIGHTS) * loud * LIFT_SCALE)


# ---------- wrist path: one smoothest movement within the hand's constraints ----------
# The wrist path is the least-jerk movement (damped by acceleration and speed costs) through the anchors' poses.
# While keys are down the wrist stays within reach of them: at the place chosen for their anchor (choose_places, among
# the places its keys are held from, pipeline.depth_family) while they go down, within those places while they are
# held. With every key up it is free and rises over its free
# time by the strike / arc rules (strike_lift, _arc: the lowest height at the middle of the free time; the shape of
# the rise follows from smoothness). It lands softly (LAND_*). Departure, speed and arrival follow from the distance
# and the time the music leaves (keys lifted early under the pedal: coherence.free_releases), not from written curves.
# Solved as bounded least squares on a PATH_STEP grid, per dimension x, y, z (m), yaw, roll (deg).
PATH_STEP = 1 / 40.
# per dimension x, y, z (m), yaw, roll (deg)
JERK_SCALE = np.array([200., 200., 200., 12000., 12000.])          # per s^3: one unit of cost
ACC_SCALE = np.array([1., 1., 1., 70., 70.])                # per s^2: damps the ringing of a least-jerk path
VEL_SCALE = np.array([.3, .3, .03, 30., 30.])                             # per s: no needless motion
PULL_SCALE = np.array([.006, .06, .008, 6., 6.])               # the held pose's pull: one unit this far off it
FREE_DROP = .004                                                 # a free wrist sinks at most this below its poses
LIFT_SPAN = .25                                                  # share of the free time around its middle held at the lift
# Landing: the hand slows down onto the keys, it is set down rather than dropped like a hammer. Over LAND_SECONDS
# before a touch the wrist's speed is held near a landing speed: downward by the loudness (a soft note is placed,
# a loud one carries some arm weight), sideways near still (no skid into the keys). Soft: one unit at that speed.
LAND_SECONDS = .08
LAND_DOWN = (.03, .25)           # m/s at velocity 30 .. 110
LAND_SIDEWAYS = .04              # m/s
SETTLE_SECONDS = .08             # ... and the hand stays put sideways this long after the touch (no skid on the keys)


def moved_targets(solver, pose, vector, keys):
    """{digit: key point} of a pose's keys ``[(digit, pitch)]`` with its wrist at ``vector`` instead: each pad keeps
    its place on the hand (the fingers keep their shape) and presses its own key's playable part nearest to that. A
    key is long: a hand further in presses its white keys further in, rather than the keys pulling the hand out."""
    p0 = solver.palm(np.asarray(pose.wrist, FLOAT), float(pose.yaw), float(pose.roll))
    p1 = solver.palm(np.asarray(vector[:3], FLOAT), float(vector[3]), float(vector[4]))
    m = p1 @ np.linalg.inv(p0)
    out = {}
    for d, pitch in keys:
        t = np.asarray(pose.targets[d], FLOAT)
        out[d] = solver.regions.clamp(pitch, m[:3, :3] @ t + m[:3, 3])
        out[d][2] = t[2]
    return out


# ---------- where each anchor is played from (pipeline.depth_family) ----------
# An anchor's pose is the most comfortable place for its own keys; its family is every place its keys are held from
# without strain (other depths into the keys, other turns). Over the anchors of a hand in time the places are chosen
# together, for the least movement between consecutive anchors (for the time between them) and the least strain: a
# hand playing a figure stays where all its chords can be played, no one chord pulls it to its own best place.
PLACE_SCALE = np.array([.01, .01, .01, 3., 3.])       # one unit of a move (m, deg: turning is seen as much as moving) ...
PLACE_STRAIN = 5.                # ... (squared, per second it is made in); an awkwardness unit costs this many
PLACE_MIN_SECONDS = .05          # moves between anchors closer in time count as made in this long
PLACE_OWN = .3                   # a place off the anchor's own most comfortable pose: per squared unit (PLACE_SCALE)
# A figure is stroked in one direction: along a run of notes the hand goes the way the notes go and comes back in the
# air (the cat's back, a pipa's sweep), it does not rock to and fro on the keys. A move against the way the notes go
# costs this many times a move, except the step that turns the run back (``sweeps``).
PLACE_AGAINST = 4.
TURN_BACK = 5                    # semitones: a step against the run at least this far turns the figure back
RETURN_LIFT = .006               # m: the hand turning a figure back comes back this far above its keys (in the air)


def sweeps(order, notes):
    """Per step between consecutive anchors: (direction of the notes, -1/0/1; True where the step turns a run back),
    the direction from the mean pitch of each anchor's new keys; a turn back reverses a run of at least two steps one
    way, by at least ``TURN_BACK``, all from one hand position (the planner's wrist unchanged from the run's start to
    the turn): a figure under the hand. A run that carries the hand along (an arpeggio over octaves, then the next
    bass) is a leap back, not a stroke."""
    pitch = [np.mean([notes[k].pitch for k in a.incoming]) if a.incoming else np.nan for a in order]
    out, run, first = [], 0, 0
    for i, (p0, p1) in enumerate(zip(pitch, pitch[1:])):
        step = 0 if np.isnan(p0) or np.isnan(p1) else int(np.sign(p1 - p0))
        back = (step != 0 and run != 0 and step != np.sign(run) and abs(run) >= 2 and abs(p1 - p0) >= TURN_BACK
                and np.ptp([a.x_hint for a in order[first:i + 2]]) <= 1e-4)
        out.append((0 if back else step, back))
        if back or step == 0 or (run and step != np.sign(run)):
            run, first = step, (i + 1 if step == 0 else i)
        else:
            run += step
    # a direction counts only inside a cycle: a run that ends turning back
    swept, way = [], 0
    for (step, back), p0, p1 in zip(out[::-1], pitch[-2::-1], pitch[:0:-1]):
        if back:
            way = -int(np.sign(p1 - p0))
        elif step != way:
            way = 0
        swept.append((step if step == way else 0, back))
    return swept[::-1]


def choose_places(anchors, notes=None):
    """{id(anchor): index into its family} for ``anchors`` (one hand) by dynamic programming over time; with
    ``notes``, moving against the way the notes go costs more (``PLACE_AGAINST``, ``sweeps``)."""
    order = sorted([a for a in anchors if a.family is not None], key=lambda a: a.time)
    if not order:
        return {}
    ways = sweeps(order, notes) if notes is not None else [(0, False)] * (len(order) - 1)
    def strain(a):
        own = (((a.family[0] - a.family[0][0]) / PLACE_SCALE) ** 2).sum(axis=1)      # the pose's own is first
        return a.family[2] * PLACE_STRAIN + PLACE_OWN * own

    cost = strain(order[0])
    back = []
    for (prev, a), (way, _) in zip(zip(order, order[1:]), ways):
        dt = max(a.time - prev.time, PLACE_MIN_SECONDS)
        d = (a.family[0][:, None, :] - prev.family[0][None, :, :]) / PLACE_SCALE
        move = (d ** 2).sum(axis=2) / dt                 # [this place, previous place]
        if way:
            move = move + PLACE_AGAINST * np.maximum(0., -way * d[:, :, 0]) ** 2 / dt
        total = move + cost[None, :]
        back.append(np.argmin(total, axis=1))
        cost = total[np.arange(len(total)), back[-1]] + strain(a)
    out, i = {}, int(np.argmin(cost))
    for k in range(len(order) - 1, -1, -1):
        out[id(order[k])] = i
        if k:
            i = int(back[k - 1][i])
    return out


def family_pose(anchor, vector):
    """The anchor's settled pose (its family's) nearest to the wrist at ``vector``."""
    if anchor.family is None:
        return anchor.pose
    vectors, poses, _ = anchor.family
    return poses[int(np.argmin((((vectors - vector) / PLACE_SCALE) ** 2).sum(axis=1)))]


def placed(anchors, paths, solvers, notes):
    """The anchors with the poses of the places chosen for them (``choose_places``, as the wrist paths are), their
    key points where the path's wrist puts them (``moved_targets``: the path holds the place while the keys go down)."""
    chosen = {}
    for h in 'LR':
        chosen.update(choose_places([a for a in anchors if a.hand == h], notes))
    out = []
    for a in anchors:
        keys = [] if a.pose is None else [(notes[k].digit, notes[k].pitch) for k in a.incoming + a.held
                                          if notes[k].digit in a.pose.targets]
        if not keys:
            out.append(a)
            continue
        pose = a.family[1][chosen[id(a)]] if id(a) in chosen else a.pose
        v = np.asarray(paths[a.hand](a.time)[:5], float)
        keys = [(d, p) for d, p in keys if d in pose.targets]
        targets = {**pose.targets, **moved_targets(solvers[a.hand], pose, v, keys)}
        out.append(replace(a, pose=replace(pose, wrist=v[:3].astype(FLOAT), yaw=float(v[3]), roll=float(v[4]),
                                           targets=targets)))
    return out


def segments(anchors, notes, strokes):
    """Per anchor (sorted): its vector, when it starts holding, its down keys; and free windows between."""
    anchors = sorted(anchors, key=lambda a: a.time)
    chosen = choose_places(anchors, notes)
    rows = []
    for a in anchors:
        keys = list(a.incoming) + list(a.held)
        start = min((notes[k].touch for k in a.incoming), default=a.time)
        v = a.family[0][chosen[id(a)]].copy() if id(a) in chosen else _vector(a)
        rows.append((a, v, start, keys))
    return rows


def solve_path(anchors, notes, strokes, t0, t1):
    from scipy import sparse             # pipeline only (Blender's Python imports this module without scipy)
    rows = segments(anchors, notes, strokes)
    times = np.arange(t0, t1 + 1e-9, PATH_STEP)
    n = len(times)
    lb = np.full((n, 5), -np.inf); ub = np.full((n, 5), np.inf)
    target = np.full((n, 5), np.nan)
    holding = np.full(n, -1)                     # the anchor whose keys are down at a sample
    starts = np.array([r[2] for r in rows])
    for k, t in enumerate(times):
        i = int(np.searchsorted(starts, t + 1e-9, side='right')) - 1
        if i < 0:
            i = 0
        a, v, start, keys = rows[i]
        down = [notes[q] for q in keys if notes[q].touch <= t + 1e-9 and t <= notes[q].release + 1e-9]
        if t < start - 1e-9 or not down:
            continue
        holding[k] = i
        # the hand is at its place while the anchor's keys go down (from the touch until they are down), and over
        # the rest of their hold it may move on toward the next only among the places its keys are held from (the
        # extent of its family: a stretched chord has none); a key it would lose there is kept by an intermediate
        # anchor (the wrist path check's refinement), which has its places too
        # ... and where every key it still holds is one the next anchor holds too, among that one's places as well:
        # the hand sets off for the next chord while it keeps the key (a thumb held into an octave above it)
        if t <= max(a.time, start + SETTLE_SECONDS) + 1e-9:
            lb[k], ub[k] = v, v
            target[k] = v
            continue
        places = v[None] if a.family is None else a.family[0]
        if i + 1 < len(rows) and all(q in rows[i + 1][3] for q in keys
                                     if notes[q].touch <= t + 1e-9 and t <= notes[q].release + 1e-9):
            b, vb = rows[i + 1][0], rows[i + 1][1]
            places = np.concatenate([places, vb[None] if b.family is None else b.family[0]])
        else:
            target[k] = v                # held where it is: pulled to its place; on the way on: free between the two
        lb[k], ub[k] = places.min(axis=0), places.max(axis=0)
    # free windows: between an anchor's last release and the next anchor's first touch
    turns = [back for _, back in sweeps([r[0] for r in rows], notes)]
    for ((a, va, _, ka), (b, vb, sb, kb)), turn in zip(zip(rows, rows[1:]), turns):
        if not b.incoming:
            continue
        keep = set(b.held)
        free = max([notes[q].release for q in ka if q not in keep], default=a.time)
        touch = min(notes[q].touch for q in b.incoming)
        window = touch - free
        sel = (times > free) & (times < touch)
        if not sel.any():
            continue
        lb[sel, 2] = np.maximum(lb[sel, 2], min(va[2], vb[2]) - FREE_DROP)
        travel = float(np.linalg.norm(vb[:2] - va[:2]))
        lift = max(_arc(va, vb, not keep), strike_lift(a, b, notes, free, keep, travel), RETURN_LIFT if turn else 0.)
        lift = min(lift, LIFT_RATE * window)
        if lift > 1e-4:
            mid = (free + touch) / 2
            core = sel & (np.abs(times - mid) <= LIFT_SPAN * window / 2 + PATH_STEP / 2)
            lb[core, 2] = np.maximum(lb[core, 2], (va[2] + vb[2]) / 2 + lift)
    # landings: (sample, speed scale per dimension) for each first touch of an anchor after a free time or a move
    land = []
    for a, va, sa, ka in rows[1:]:
        if not a.incoming:
            continue
        touch = min(notes[q].touch for q in a.incoming)
        loud = min(max((max(notes[q].velocity for q in a.incoming) - 30) / 80., 0.), 1.)
        down = LAND_DOWN[0] + (LAND_DOWN[1] - LAND_DOWN[0]) * loud
        for k in np.nonzero((times >= touch - LAND_SECONDS) & (times < touch + SETTLE_SECONDS))[0]:
            if k + 1 < n:
                before = times[k] < touch
                land.append((k, np.array([LAND_SIDEWAYS, LAND_SIDEWAYS, down if before else np.inf, 40., 40.])))
    # ends pinned to the first / last pose
    first, last = rows[0][1], rows[-1][1]
    out = np.empty((n, 5))
    d3 = sparse.diags([-1., 3., -3., 1.], [0, 1, 2, 3], shape=(n - 3, n)) / PATH_STEP ** 3
    d2 = sparse.diags([1., -2., 1.], [0, 1, 2], shape=(n - 2, n)) / PATH_STEP ** 2
    d1 = sparse.diags([-1., 1.], [0, 1], shape=(n - 1, n)) / PATH_STEP
    for d in range(5):
        has = ~np.isnan(target[:, d])
        pull = sparse.diags(has.astype(float) / PULL_SCALE[d], 0, shape=(n, n)).tocsr()[has]
        rows_l = [(k, sc[d]) for k, sc in land]
        if rows_l:
            r = np.arange(len(rows_l)); ks = np.array([k for k, _ in rows_l]); sc = np.array([c for _, c in rows_l])
            vel = sparse.csr_matrix((np.r_[-1. / (PATH_STEP * sc), 1. / (PATH_STEP * sc)], (np.r_[r, r], np.r_[ks, ks + 1])),
                                    shape=(len(rows_l), n))
        else:
            vel = sparse.csr_matrix((0, n))
        A = sparse.vstack([d3 / JERK_SCALE[d], d2 / ACC_SCALE[d], d1 / VEL_SCALE[d], pull, vel]).tocsr()
        b = np.r_[np.zeros(n - 3), np.zeros(n - 2), np.zeros(n - 1), (target[has, d] / PULL_SCALE[d]), np.zeros(vel.shape[0])]
        lo, hi = lb[:, d].copy(), ub[:, d].copy()
        lo[0] = hi[0] = first[d]; lo[-1] = hi[-1] = last[d]
        hi = np.maximum(hi, lo + 1e-9)
        out[:, d] = chunked(A, b, lo, hi)
    return times, out


CHUNK_SECONDS, CHUNK_KEEP = 6., 4.    # solved exactly in overlapping windows, each fixed to the last at its start


def chunked(A, b, lo, hi):
    """Bounded least squares, exactly (bvls) over overlapping windows of samples; the first three samples of a
    window are fixed to the previous window's solution (position, speed, acceleration carry over)."""
    from scipy.optimize import lsq_linear
    A = A.tocsr()
    n = A.shape[1]
    first = np.array([A.indices[A.indptr[r]:A.indptr[r + 1]].min() if A.indptr[r + 1] > A.indptr[r] else 0 for r in range(A.shape[0])])
    last = np.array([A.indices[A.indptr[r]:A.indptr[r + 1]].max() if A.indptr[r + 1] > A.indptr[r] else 0 for r in range(A.shape[0])])
    size, keep = int(CHUNK_SECONDS / PATH_STEP), int(CHUNK_KEEP / PATH_STEP)
    x = np.zeros(n)
    i0 = 0
    while True:
        i1 = min(n, i0 + size)
        rows = np.nonzero((first >= i0) & (last < i1))[0]
        lo_w, hi_w = lo[i0:i1].copy(), hi[i0:i1].copy()
        if i0 > 0:
            lo_w[:3] = hi_w[:3] = x[i0:i0 + 3]
        res = lsq_linear(A[rows][:, i0:i1].toarray(), b[rows], bounds=(lo_w, np.maximum(hi_w, lo_w + 1e-12)), method='bvls')
        stop = i1 >= n
        x[i0:i1 if stop else i0 + keep] = res.x[:(i1 - i0) if stop else keep]
        if stop:
            return x
        i0 += keep - 3 if keep > 3 else keep


def wrist_path(anchors, notes, strokes, rest, duration):
    """``anchors`` of one hand (with poses and their depth families, pipeline.depth_family; without: held in place)."""
    anchors = sorted(anchors, key=lambda a: a.time)
    if not anchors:
        return WristPath([(0., rest, 0.), (duration, rest, 0.)])

    def ready(a):
        return min((strokes[k].hover for k in a.incoming), default=a.time)

    first, v0 = anchors[0], _vector(anchors[0])
    settle = ready(first)
    leave = settle - LEAD_IN_SECONDS
    last = anchors[-1]
    vl = _vector(last)
    end = max([last.time] + [n.release for n in notes.values()]) + KEY_RETURN_SECONDS
    times, values = solve_path(anchors, notes, strokes, settle, end)
    knots = [(min(0., leave - 1e-3), rest, 0.), (leave, rest, _arc(rest, v0, True))]
    knots += [(t, v, 0.) for t, v in zip(times, values)]
    knots[-1] = (times[-1], values[-1], _arc(vl, rest, True))
    knots.append((end + LEAD_IN_SECONDS, rest, 0.))
    if end + LEAD_IN_SECONDS < duration:
        knots.append((duration, rest, 0.))
    path = WristPath(knots)
    path.set_accents(list(notes.values()))
    return path


# ---------- contact points ----------
def contact_waypoints(anchors, notes):
    """Per note: (time, key-top pad point, finger values) at every anchor the key is part of.

    The pad may drift over its key between anchors (a held key under a moving hand)."""
    out = {}
    for a in sorted(anchors, key=lambda a: a.time):
        for k in a.incoming + a.held:
            digit = notes[k].digit
            depth = KEY_TRAVEL if k in a.incoming else notes[k].depth(a.time)
            out.setdefault(k, []).append((a.time, a.pose.targets[digit] + np.array([0., 0., depth], dtype=FLOAT),
                                          np.array(a.pose.values[digit], FLOAT), np.array(a.pose.wrist, FLOAT)))
    return out


def waypoint(points, t, path=None):
    """The pad's point on its key and the finger's seed at ``t``. Between two anchors the pad moves with
    the hand: as far as the wrist (``path``) has moved from one anchor's place to the next (by time when
    the wrist stays): a hand that waits before a leap keeps its fingers where they are."""
    if len(points) == 1 or t <= points[0][0]:
        return points[0][1], points[0][2]
    if t >= points[-1][0]:
        return points[-1][1], points[-1][2]
    i = max(j for j, p in enumerate(points) if p[0] <= t)
    a, b = points[i], points[i + 1]
    w = ease((t - a[0]) / max(b[0] - a[0], 1e-6))
    if path is not None and len(a) > 3:
        pa, pb = a[3], b[3]                 # the anchors' own wrists (a late leap is still under way at b's time)
        span = pb - pa
        if span @ span > 1e-6:
            w = float(np.clip((path(t)[:3] - pa) @ span / (span @ span), 0., 1.))
    return a[1] * (1 - w) + b[1] * w, a[2] * (1 - w) + b[2] * w


# ---------- finger strokes ----------
def hover_height(velocity):
    return HOVER_HEIGHT[0] + (HOVER_HEIGHT[1] - HOVER_HEIGHT[0]) * min(max((velocity - 30) / 80, 0.), 1.)


@dataclass
class Stroke:
    note: int
    hover: float                 # stroke start (pad at ``height`` above the key)
    touch: float
    release: float
    lift_end: float
    height: float
    lift_height: float
    poses: tuple = None          # finger values at hover, touch, release, lift end (``finger_track``)
    clearance: tuple = None      # (times, value deltas) keeping the strike and lift over the other keys

    def base_values(self, t):
        """Strike and lift in joint space: accelerating into the key, springing off it."""
        hover, touch, release, lifted = self.poses
        if t < self.touch:
            x = (t - self.hover) / max(self.touch - self.hover, 1e-6)
            return touch + (hover - touch) * (1 - x * x)
        x = (t - self.release) / max(self.lift_end - self.release, 1e-6)
        return release + (lifted - release) * lift_profile(x)

    def values(self, t):
        v = self.base_values(t)
        if self.clearance is None:
            return v
        times, deltas = self.clearance
        return v + np.array([np.interp(t, times, deltas[:, k]) for k in range(4)], dtype=FLOAT)


def strokes(notes):
    """Stroke timing of every note. The strike accelerates onto the key top at the key's own speed
    (travel / press time), so it lasts 2 * height / speed; consecutive strokes of one finger share the
    gap between them."""
    out = {}
    by_finger = {}
    for n in notes.values():
        by_finger.setdefault((n.hand, n.digit), []).append(n)
    for mine in by_finger.values():
        mine.sort(key=lambda n: n.touch)
        timing = []
        for n in mine:
            height = hover_height(n.velocity) * FINGER_REACH[n.digit]
            strike = 2 * height * max(n.onset - n.touch, 1e-3) / KEY_TRAVEL
            timing.append([strike, height, KEY_RETURN_SECONDS, LIFT_HEIGHT * FINGER_REACH[n.digit]])
        for (p, tp), (n, tn) in zip(zip(mine, timing), zip(mine[1:], timing[1:])):
            need = tp[2] + tn[0]
            gap = n.touch - p.release
            if gap < need:
                f = max(gap, 0.) / need
                tp[2], tp[3] = tp[2] * f, tp[3] * f
                tn[0], tn[1] = tn[0] * f, tn[1] * f
        for n, (strike, height, lift, lift_height) in zip(mine, timing):
            out[n.id] = Stroke(n.id, n.touch - strike, n.touch, n.release, n.release + lift, height, lift_height)
    return out


def stroke_floor(stroke, note, t):
    """How far the key floor is lowered for a stroke finger at ``t``; None while it rides its key.

    The floor is fully lowered at the key (touch, release) and back in place at the stroke's ends
    (hover, lift end), so the finger's pose is continuous across the ride."""
    if stroke.touch <= t <= stroke.release:
        return None
    if t < stroke.touch:
        x = (t - stroke.hover) / max(stroke.touch - stroke.hover, 1e-6)
        near = x * x
    else:
        near = 1 - lift_profile((t - stroke.release) / max(stroke.lift_end - stroke.release, 1e-6))
    return note.depth(t) + .0005 + FLOOR_RAMP * near


def stroke_target(stroke, note, points, t, path=None):
    """Pad target and IK seed of a stroke at ``t`` (hover .. lift end)."""
    top, seed = waypoint(points, min(max(t, stroke.touch), stroke.release), path)
    if t < stroke.touch:
        x = (t - stroke.hover) / max(stroke.touch - stroke.hover, 1e-6)
        z = stroke.height * (1 - x * x)
    elif t <= stroke.release:
        z = -note.depth(t)
    else:
        d0 = note.depth(stroke.release)
        x = (t - stroke.release) / max(stroke.lift_end - stroke.release, 1e-6)
        z = -d0 + (d0 + stroke.lift_height) * lift_profile(x)
    return top + np.array([0., 0., z], dtype=FLOAT), seed


@dataclass
class Keypose:
    time: float
    values: np.ndarray


class FingerTrack:
    """Strokes and free-time keyposes of one finger, plus a smooth key-clearance correction of the free time."""

    def __init__(self, strokes, keyposes):
        self.strokes = sorted(strokes, key=lambda s: s.hover)
        self.starts = np.array([s.hover for s in self.strokes])
        self.keyposes = sorted(keyposes, key=lambda k: k.time)
        self.key_times = np.array([k.time for k in self.keyposes])
        self.correction = None          # (times, deltas, slopes)
        self.room = None                # the same after the spacing (``room_corrections``)

    def set_correction(self, times, deltas):
        times, deltas = np.asarray(times, float), held(deltas)
        if len(times) > 1 and np.abs(deltas).max() > 0:
            self.correction = (times, deltas, pchip_slopes(times, deltas))

    def set_room(self, times, deltas):
        times, deltas = np.asarray(times, float), held(deltas)
        if len(times) > 1 and np.abs(deltas).max() > 0:
            self.room = (times, deltas, pchip_slopes(times, deltas))

    def room_values(self, t, values):
        return values if self.room is None else values + self._taper(t) * self._correction(t, self.room)

    def clear_shift(self, t):
        """The key-clearance correction at ``t`` (zero on the key)."""
        return np.zeros(4, dtype=FLOAT) if self.correction is None else self._taper(t) * self._correction(t)

    def _taper(self, t):
        """The corrections run through the free time and the strike and lift alike (one smooth curve: a
        correction switched off where a stroke joins the free time left the finger, still curled from its
        key, sweeping through the keys behind it) and fade out only as the pad reaches its key."""
        s = self.stroke(t)
        if s is None:
            return 1.
        if s.touch <= t <= s.release:
            return 0.
        gap = s.touch - t if t < s.touch else t - s.release
        return min(1., gap / CORRECTION_FADE)

    def _correction(self, t, correction=None):
        times, deltas, slopes = self.correction if correction is None else correction
        i = int(np.clip(np.searchsorted(times, t, side='right') - 1, 0, len(times) - 2))
        h = times[i + 1] - times[i]
        s = min(max((t - times[i]) / h, 0.), 1.)
        h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
        return h00 * deltas[i] + h10 * h * slopes[i] + h01 * deltas[i + 1] + h11 * h * slopes[i + 1]

    def stroke(self, t):
        i = np.searchsorted(self.starts, t, side='right') - 1
        if i >= 0 and t <= self.strokes[i].lift_end + 1e-9:
            return self.strokes[i]
        return None

    def keyposed(self, t):
        i = np.searchsorted(self.key_times, t, side='right') - 1
        if i < 0:
            return self.keyposes[0].values
        if i >= len(self.keyposes) - 1:
            return self.keyposes[-1].values
        a, b = self.keyposes[i], self.keyposes[i + 1]
        w = ease((t - a.time) / max(b.time - a.time, 1e-6))
        return a.values * (1 - w) + b.values * w

    def free_values(self, t):
        values = self.keyposed(t)
        return values if self.correction is None else values + self._taper(t) * self._correction(t)


def _sigmoid(x):
    return 1. / (1. + np.exp(-np.clip(x, -50., 50.)))


class KeyFloor:
    """Smooth lower bound for skin over the keyboard: the white-key top in front of the black keys,
    the black-key top behind their front edge (conservative between them), free in front of the keys."""

    EDGE = .002                  # width of the smooth steps (m)
    DROP = .05
    FRONT_LEAD = .008            # the step is complete at the keys' front edge (skin hooked over it is caught)

    def __init__(self, regions):
        keys = list(regions.keys.values())
        whites = [k for k in keys if not regions._black(k)]
        blacks = [k for k in keys if regions._black(k)]
        self.white_top = max(k['bounds_max'][2] for k in whites)
        self.black_top = max(k['bounds_max'][2] for k in blacks)
        self.white_front = min(k['bounds_min'][1] for k in whites)
        self.black_front = regions.black_front
        self.x = (min(k['bounds_min'][0] for k in keys), max(k['bounds_max'][0] for k in keys))

    def __call__(self, points):
        x, y = points[:, 0], points[:, 1]
        base = self.white_top + (self.black_top - self.white_top) * _sigmoid((y - self.black_front) / self.EDGE)
        over = _sigmoid((y - self.white_front + self.FRONT_LEAD) / self.EDGE) * _sigmoid((x - self.x[0]) / self.EDGE)             * _sigmoid((self.x[1] - x) / self.EDGE)
        low = self.white_top - self.DROP
        return low + (base - low) * over


class KeySurface:
    """The key tops as a continuous surface: the 1 mm key height field (checks.Heightfield), each key
    lowered by its current depth, bilinear between grid nodes. The finger's own key may be entered by
    ``OWN_KEY_TOLERANCE`` (its pad rests on it). Nodes in the gaps between keys are left out of the blend
    (weights renormalised over the key nodes): blended in as a floor far below, they hid the skin lying
    up to a millimetre over a neighbour's edge (a thumb's side on the next white key)."""

    NO_KEY = -1.                    # surface height (m) where no node is on a key: nothing to collide with
    CORNERS = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

    def __init__(self, heightfield):
        self.field = heightfield
        self.on_key = np.isfinite(heightfield.top)
        self.top = np.where(self.on_key, heightfield.top, self.NO_KEY)
        self._memo = {}

    def __call__(self, points, depths, own_pitch):
        return self._surface(points, depths, own_pitch)[0]

    def _surface(self, points, depths, own_pitch):
        """(surface height, share of it on keys a finger holding ``own_pitch`` may brush), read-only. Kept per exact
        input: a solver's difference step that moves one finger leaves the other fingers' skin where it was."""
        points = np.asarray(points)
        key = (points.dtype.str, points.shape, points.tobytes(), tuple(sorted(depths.items())), own_pitch)
        hit = self._memo.get(key)
        if hit is None:
            if len(self._memo) > 4096:
                self._memo.clear()
            hit = self._memo[key] = self._compute(points, depths, own_pitch)
            for a in hit:
                a.setflags(write=False)
        return hit

    def _compute(self, points, depths, own_pitch):
        f = self.field
        u = (points[:, :2] - f.origin) / CELL
        i = np.clip(np.floor(u).astype(int), 0, np.array(self.top.shape) - 2)
        w = np.clip(u - i, 0., 1.)
        table = depth_table(depths)
        # the four grid corners at once (rows), summed in corner order
        ix = i[:, 0] + self.CORNERS[:, :1]
        iy = i[:, 1] + self.CORNERS[:, 1:]
        weight = np.stack(((1 - w[:, 0]) * (1 - w[:, 1]), w[:, 0] * (1 - w[:, 1]), (1 - w[:, 0]) * w[:, 1], w[:, 0] * w[:, 1]))
        pitch = f.pitch[ix, iy]
        z = self.top[ix, iy] - table[pitch].astype(FLOAT)
        z = np.where(pitch == own_pitch, z - OWN_KEY_TOLERANCE, z)
        weight = weight * self.on_key[ix, iy]
        wz, ws = weight * z, weight * Heightfield.brushable(pitch, table, own_pitch)
        out, soft, total = ((wz[0] + wz[1]) + wz[2]) + wz[3], ((ws[0] + ws[1]) + ws[2]) + ws[3], ((weight[0] + weight[1]) + weight[2]) + weight[3]
        keyed = total > 1e-9
        safe = np.where(keyed, total, 1.)
        return np.where(keyed, out / safe, self.NO_KEY).astype(FLOAT), np.where(keyed, soft / safe, 0.).astype(FLOAT)

    def clearance(self, points, depths, own_pitch, brush=False):
        """Depth of each point below the key surface (0 above it). ``brush``: a finger holding ``own_pitch``
        may rest on unplayed neighbours (checks.brushed), blended by how much of the point is over them."""
        z, soft = self._surface(points, depths, own_pitch)
        pen = np.maximum(0., z - points[:, 2])
        if not brush or own_pitch is None:
            return pen
        return soft * brushed(pen, True) + (1 - soft) * pen


class Spacing:
    """Fingers 2..5 keep their order across the palm. A free finger makes room for its neighbours:
    its lateral position (palm frame) is pushed out by a smooth maximum to at least ``SPACING_SHARE``
    of the relaxed hand's spacing, and its sideways swing follows. The position is taken on the whole
    finger, its proximal and its distal half (``positions``): seen from above, a finger swung toward a
    curled neighbour crosses it at the middle joints while the fingertips are still apart. Playing
    fingers are never moved; a finger striking or lifting gives way in part (``make_room``'s ``give``). A free
    finger squeezed between two others that leave it too little room stays between them, in proportion to
    the relaxed spacing. The thumb passes under the hand by technique and is not constrained.

    The fingers are tied by their tendons (no finger opens alone over its range): where a neighbour has moved
    further away than the relaxed hand's gap, a free finger follows it by ``SPREAD_COUPLING`` of the stretch,
    passed on along the hand, so a hand spread by its outer fingers (an octave) fans out instead of one gap
    opening wide beside three fingers held together."""

    def __init__(self, solver):
        self.solver = solver
        relaxed = {d: ARCHED[d] for d in DIGITS}          # the playing hand the free fingers wait in
        tips = {d: solver.local_pad(d, relaxed)[:3].mean(axis=1) for d in DIGITS}
        lateral = tips[5] - tips[2]
        self.lateral = lateral / np.linalg.norm(lateral)
        self.gaps = {d: SPACING_SHARE * float((tips[d + 1] - tips[d]) @ self.lateral) for d in (2, 3, 4)}
        # proximal half: skin carried mostly by the finger's first bone (curling never moves it to the tip side)
        fingers = solver.rig.fingers[solver.hand]
        self.proximal = {}
        for d in DIGITS:
            skin = solver.clearance_skin[d]
            first = [k for k, b in enumerate(skin.bones) if b == fingers[d].names[0]]
            proximal = skin.weights[:, first].sum(axis=1) >= .5 if first else np.zeros(len(skin.weights), bool)
            if proximal.all() or not proximal.any():       # a finger skinned to one bone: split it in two along it
                forward = solver.local_clearance(d, relaxed)[1]
                proximal = forward < np.median(forward)
            self.proximal[d] = proximal
        whole = {d: self.positions(d, relaxed) for d in (2, 3, 4, 5)}
        self.whole_gaps = {d: SPACING_SHARE * (whole[d + 1] - whole[d]) for d in (2, 3, 4)}
        self.thumb_gap = whole[2] - self.positions(1, relaxed)     # the thumb's relaxed distance (the index follows it)

    def position(self, digit, values):
        return float(self.solver.local_pad(digit, values)[:3].mean(axis=1) @ self.lateral)

    def positions(self, digit, values):
        """Lateral position of the finger's proximal and distal half (its own skin, palm frame)."""
        lateral = self.lateral @ self.solver.local_clearance(digit, values)[:3]
        proximal = self.proximal[digit]
        return np.array([lateral[proximal].mean(), lateral[~proximal].mean()], dtype=FLOAT)

    def make_room(self, values, give):
        """``give``: {digit: 0 (on its key) .. 1 (free)}: the share of its make-room swing a finger takes."""
        fingers = self.solver.rig.fingers[self.solver.hand]
        s = {d: self.positions(d, values) for d in (2, 3, 4, 5)}
        want = {d: v.copy() for d, v in s.items()}
        moves = {d for d in (2, 3, 4, 5) if give.get(d, 1.) > 0}
        for d in (3, 4, 5):                        # push right of the left neighbour ...
            if d in moves:
                want[d] = _smooth_max(want[d], want[d - 1] + self.whole_gaps[d - 1])
        for d in (4, 3, 2):                        # ... and left of the right neighbour
            if d in moves:
                want[d] = -_smooth_max(-want[d], -(want[d + 1] - self.whole_gaps[d]))
        for d in (3, 4):                           # squeezed between both: in between, by the relaxed spacing
            if d in moves:
                left, right = self.whole_gaps[d - 1], self.whole_gaps[d]
                # how far both gaps are short, eased in over SPACING_SOFTNESS: no jump when a gap closes
                short = np.minimum(left - (want[d] - want[d - 1]), right - (want[d + 1] - want[d]))
                x = np.clip(short / SPACING_SOFTNESS, 0., 1.)
                tight = x * x * x * (10 + x * (-15 + 6 * x))          # minimum-jerk ease
                between = want[d - 1] + (want[d + 1] - want[d - 1]) * left / (left + right)
                want[d] = want[d] + (between - want[d]) * tight
        relaxed = {1: self.thumb_gap, **{d: self.whole_gaps[d] / SPACING_SHARE for d in (2, 3, 4)}}
        pulled = {**want, 1: self.positions(1, values)}          # the thumb pulls, it is not moved
        for _ in range(SPREAD_ROUNDS):             # the fan: follow a neighbour that went further than relaxed
            pull = {d: np.zeros(2, dtype=FLOAT) for d in moves}
            for d in moves:
                for n, gap, sign in ((d - 1, relaxed.get(d - 1), 1.), (d + 1, relaxed.get(d), -1.)):
                    if n not in pulled or gap is None:
                        continue
                    stretch = sign * (want[d] - pulled[n]) - gap        # > 0: further apart than relaxed
                    # a free neighbour pulls half as much as one on its key
                    pull[d] -= sign * SPREAD_COUPLING * _smooth_max(stretch, 0., SPREAD_SOFTNESS) * (1. - .5 * give.get(n, 1.))
            for d in sorted(moves):                # never past the spacing kept from either neighbour
                w = want[d] + pull[d] / SPREAD_ROUNDS
                lo = want[d - 1] + self.whole_gaps[d - 1] if d - 1 in want else -np.inf
                hi = want[d + 1] - self.whole_gaps[d] if d + 1 in want else np.inf
                # squeezed (no room between the two): left where the spacing put it
                want[d] = pulled[d] = np.where(lo <= hi, np.clip(w, lo, hi), want[d])
        out = dict(values)
        for d in (2, 3, 4, 5):
            shift = (want[d] - s[d]) * give.get(d, 1.)
            if d not in moves or np.abs(shift).max() < 1e-5:
                continue
            probe = np.array(values[d], FLOAT)
            probe[0] += 1.
            slope = self.positions(d, {**values, d: probe}) - s[d]          # metres per degree of swing
            swings = [shift[k] / slope[k] for k in range(2) if abs(slope[k]) > 1e-6 and abs(shift[k]) >= 1e-5]
            if not swings:
                continue
            v = np.array(values[d], FLOAT)
            # the larger of the proximal and distal needs, blended (no flip when they disagree in sign)
            weights = np.abs(swings) ** 4 + 1e-30
            swing = float(np.dot(swings, weights) / weights.sum())
            if give.get(d, 1.) >= 1.:
                # ... but a free finger's one swing moves both halves: keep each half between its neighbours (one half's
                # need taken whole swung the other half across a neighbour, a middle finger squeezed between the index
                # and ring finger on neighbouring keys); with no room for both, the swing that misses each alike
                low, high = -np.inf, np.inf
                for k in range(2):
                    if abs(slope[k]) < 1e-6:
                        continue
                    a = want[d - 1][k] + self.whole_gaps[d - 1][k] if d - 1 in want else -np.inf
                    b = want[d + 1][k] - self.whole_gaps[d][k] if d + 1 in want else np.inf
                    if a > b:
                        a = b = (a + b) / 2
                    ends = sorted(((a - s[d][k]) / slope[k], (b - s[d][k]) / slope[k]))
                    low, high = max(low, ends[0]), min(high, ends[1])
                swing = float(np.clip(swing, low, high)) if low <= high else float((low + high) / 2)
            v[0] = np.clip(v[0] + swing, fingers[d].lower[0], fingers[d].upper[0])
            out[d] = v
        return self.web_fan(out, give)

    def web_fan(self, values, give):
        """Two neighbouring fingers open no further than their web lets them comfortably (``kinematics.Rig.web``'s
        free limit). The web ties both: an opening beyond it is taken back by the two in proportion to their
        ``give``, all of it once their gives add up to one (a free finger follows its lifting neighbour), none
        while both are on their keys. Shared out in damped rounds, so a run of free fingers fans evenly and one
        between two neighbours too far apart is left in between, both its webs equally over."""
        rest, slope, free, _ = self.solver.rig.web(self.solver.hand)
        fingers = self.solver.rig.fingers[self.solver.hand]
        share = {d: give.get(d, 1.) for d in DIGITS}
        lateral = {d: float(values[d][0]) for d in DIGITS}
        start = dict(lateral)
        pairs = [(d, d + 1) for d in (1, 2, 3, 4) if share[d] + share[d + 1] > 0]
        for _ in range(WEB_ROUNDS):
            step = {d: 0. for d in DIGITS}
            for a, b in pairs:
                over = rest[a, b] + slope[a, b] * (lateral[b] - lateral[a]) - free[a, b]
                if over <= 0:
                    continue
                taken = over / slope[a, b] * min(1., share[a] + share[b]) / (share[a] + share[b])
                step[a] += taken * share[a]
                step[b] -= taken * share[b]
            for d in DIGITS:
                lateral[d] += .5 * step[d]
        out = dict(values)
        for d in DIGITS:
            if abs(lateral[d] - start[d]) < 1e-6:
                continue
            v = np.array(values[d], FLOAT)
            v[0] = np.clip(lateral[d], fingers[d].lower[0], fingers[d].upper[0])
            out[d] = v
        return out


def _smooth_max(a, b, softness=SPACING_SOFTNESS):
    return softness * np.logaddexp(a / softness, b / softness)


def distal_pitch(solver, digit, values, palm):
    """Downward tilt (deg) of the finger's last phalanx from the hand's forward direction: 0 level,
    90 vertical, beyond 90 curled back under (never folded back onto small values)."""
    finger = solver.rig.fingers[solver.hand][digit]
    axis = palm[:3, :3] @ finger.local_matrices(values)[2][:3, 1]
    forward = palm[:3, 1].copy()
    forward[2] = 0.
    return float(np.degrees(np.arctan2(-axis[2], axis @ forward / np.linalg.norm(forward))))


def pad_residual(solver, digit, values, palm):
    """Fingers 2..5 meet their key with the pad: penalise a last phalanx steeper than PAD_PITCH_MAX."""
    if digit == 1:
        return np.zeros(1, dtype=FLOAT)
    return np.array([max(0., distal_pitch(solver, digit, values, palm) - PAD_PITCH_MAX) / PAD_PITCH_SCALE], dtype=FLOAT)


class FingerIK:
    """Four-value pad IK with the wrist fixed; optionally keeps the finger's skin above the keys."""

    def __init__(self, solver: PoseSolver):
        self.solver = solver
        self.floor = KeyFloor(solver.regions)
        self.surface = KeySurface(solver.heightfield) if solver.heightfield is not None else None
        self.spacing = Spacing(solver)

    def ride(self, digit, palm, target, seed, depths, own_pitch, start=None, brush=False):
        """Pad on its (moving) key point, the finger's sides kept out of the neighbouring keys: where they would
        sink, the pad slips along its key toward the front (``RIDE_SLIP``) instead. ``start`` (the previous
        frame's values) keeps a finger strained against its limits on one solution."""
        if self.surface is None:
            return self.solve(digit, palm, target, seed)
        finger = self.solver.rig.fingers[self.solver.hand][digit]
        values = {d: RELAXED[d] for d in DIGITS}
        seed = np.asarray(seed, FLOAT)
        front = self.solver.regions.keys[own_pitch]['bounds_min'][1] + RIDE_FRONT_KEEP

        def residual(v):
            values[digit] = v
            point = self.solver.contact_point(self.solver.pad_world(digit, values, palm))
            skin = (palm @ self.solver.local_clearance(digit, values))[:3].T
            below = self.surface.clearance(skin, depths, own_pitch, brush=brush) / FLOOR_SCALE
            steady = (v - start) / RIDE_STEADY if start is not None else np.zeros(0, dtype=FLOAT)
            off = point - target
            slip = off[1] < 0.             # toward the key's front: the pad holds the key all along its length
            along = off[1] / (RIDE_SLIP if slip else .0002)
            return np.r_[off[0] / .0002, along, off[2] / .0002, max(0., front - point[1]) / .0002,
                         (v - seed) / IK_VALUE_SCALE, below, pad_residual(self.solver, digit, v, palm), steady]

        def run(x0):
            x, _ = PoseSolver._levenberg_marquardt(residual, np.clip(x0, finger.lower, finger.upper),
                                                   finger.lower, finger.upper, np.full(4, 1e-2, dtype=FLOAT), IK_ITERATIONS)
            values[digit] = x
            point = self.solver.contact_point(self.solver.pad_world(digit, values, palm))
            off = point - target
            off[1] = max(off[1], front - point[1], 0.) if off[1] < 0. else off[1]   # a slip toward the front is no miss
            return x, float(np.linalg.norm(off))

        x, error = run(seed if start is None else start)
        if start is not None and error > RIDE_RETRY:
            # the shape changed faster than the warm start can follow (a new anchor): solve from its pose
            fresh = run(seed)
            if fresh[1] < error - RIDE_SWITCH:
                x, error = fresh
        return x, error

    def below_floor(self, digit, values, palm, lowered):
        """Per skin point: metres below the key floor (lowered by ``lowered``), scaled."""
        points = (palm @ self.solver.local_clearance(digit, values))[:3].T
        return np.maximum(0., self.floor(points) - lowered - points[:, 2]) / FLOOR_SCALE

    def clear(self, digit, palm, values, lowered=0.):
        """A free finger: the smallest change of its values that keeps its skin above the keys."""
        others = {d: RELAXED[d] for d in DIGITS}
        start = np.asarray(values, FLOAT)
        if not self.below_floor(digit, {**others, digit: start}, palm, lowered).any():
            return start
        finger = self.solver.rig.fingers[self.solver.hand][digit]
        if digit in LIFT_PATH:
            def along(u):
                return np.clip(start + u * LIFT_PATH[digit], finger.lower, finger.upper).astype(FLOAT)
            if not self.below_floor(digit, {**others, digit: along(1.)}, palm, lowered).any():
                lo, hi = 0., 1.
                for _ in range(LIFT_PATH_STEPS):
                    mid = (lo + hi) / 2
                    if self.below_floor(digit, {**others, digit: along(mid)}, palm, lowered).any():
                        lo = mid
                    else:
                        hi = mid
                return along(hi)

        def residual(v):
            others[digit] = v
            return np.r_[(v - start) / (FREE_VALUE_SCALE * FINGER_JOINTS[digit]), self.below_floor(digit, others, palm, lowered)]

        # The finger lifts over the keys; its sideways swing is left to the spacing (dodging sideways
        # here would be undone by it, back onto a black key).
        lower, upper = finger.lower.copy(), finger.upper.copy()
        lower[0] = upper[0] = np.clip(start[0], finger.lower[0], finger.upper[0])

        def run(x0):
            x, _ = PoseSolver._levenberg_marquardt(residual, np.clip(x0, lower, upper), lower, upper, np.full(4, 1e-2, dtype=FLOAT),
                                                   IK_ITERATIONS)
            return x, float(np.sum(residual(x) ** 2))

        x, cost = run(start)
        if self.below_floor(digit, {**others, digit: x}, palm, lowered).max() > CLEAR_RETRY / FLOOR_SCALE:
            # a curled finger lifting from its own shape can hook its tip into the key: try from the relaxed curl
            y, other = run(np.r_[start[0], RELAXED[digit][1:]])
            if other < cost:
                x = y
        return x

    def solve(self, digit, palm, target, seed, lowered=None):
        """Values putting the pad's contact point on ``target`` (world); weak prior toward ``seed``.
        With ``lowered`` (own key depth) the skin also stays above the key floor."""
        finger = self.solver.rig.fingers[self.solver.hand][digit]
        values = {d: RELAXED[d] for d in DIGITS}
        seed = np.asarray(seed, FLOAT)

        def residual(v):
            values[digit] = v
            point = self.solver.contact_point(self.solver.pad_world(digit, values, palm))
            out = np.r_[(point - target) / .0002, (v - seed) / (IK_VALUE_SCALE * FINGER_JOINTS[digit]),
                        pad_residual(self.solver, digit, v, palm)]
            if lowered is not None:
                out = np.r_[out, self.below_floor(digit, values, palm, lowered)]
            return out

        x, _ = PoseSolver._levenberg_marquardt(residual, np.clip(seed, finger.lower, finger.upper),
                                               finger.lower, finger.upper, np.full(4, 1e-2, dtype=FLOAT), IK_ITERATIONS)
        values[digit] = x
        point = self.solver.contact_point(self.solver.pad_world(digit, values, palm))
        return x, float(np.linalg.norm(point - target))


def finger_track(hand, digit, notes, strokes_by_note, waypoints, path, solver, ik):
    """Keyposes at each stroke's start (hover) and end (lifted), relaxed hand in long gaps."""
    mine = sorted((replace(strokes_by_note[n.id]) for n in notes.values()
                   if n.hand == hand and n.digit == digit and n.id in waypoints), key=lambda s: s.hover)
    # free fingers wait in the playing hand's shape (pose.hand_shape), arched or flattened by the span of the keys the
    # hand is on around then (held, or about to be struck)
    mine_hand = [n for n in notes.values() if n.hand == hand and n.id in waypoints]

    def shape_at(t):
        pitches = [n.pitch for n in mine_hand if n.touch - PREPARE_SECONDS <= t <= n.release]
        return hand_shape(solver.span(pitches))[digit] if pitches else ARCHED[digit]

    # before the hand's first stroke and after its last the hand is at rest, its fingers loose (RELAXED): they
    # take the playing shape as the arm brings the hand onto the keys and let it go as it leaves
    hand_strokes = [strokes_by_note[n.id] for n in notes.values() if n.hand == hand and n.id in waypoints]
    first = min((s.hover for s in hand_strokes), default=None)
    last = max((s.lift_end for s in hand_strokes), default=None)
    keyposes, previous = [], None
    if first is not None:
        keyposes += [Keypose(first - LEAD_IN_SECONDS, RELAXED[digit]), Keypose(first - .4 * LEAD_IN_SECONDS, shape_at(first))]
    for k, s in enumerate(mine):
        note = notes[s.note]
        next_hover = mine[k + 1].hover if k + 1 < len(mine) else np.inf
        if previous is None:
            if s.hover - PREPARE_SECONDS > first - .4 * LEAD_IN_SECONDS:
                keyposes.append(Keypose(s.hover - PREPARE_SECONDS, shape_at(s.hover)))
        else:
            gap = s.hover - previous.lift_end
            if gap > RELAX_GAP_SECONDS:
                t1, t2 = previous.lift_end + min(.3 * gap, RETURN_SECONDS), s.hover - min(.3 * gap, PREPARE_SECONDS)
                keyposes += [Keypose(t1, shape_at(t1)), Keypose(t2, shape_at(t2))]
        poses = []
        # The finger takes its shape on the hand, and the hand carries it: the hover shape is solved
        # with the wrist where it will be at the touch, the lifted shape with the wrist of the release.
        for t, hand_at in ((s.hover, s.touch), (s.touch, s.touch), (s.release, s.release), (s.lift_end, s.release)):
            target, seed = stroke_target(s, note, waypoints[s.note], t, path)
            palm = solver.palm(*_split(path(hand_at)))
            target = target + (palm[:3, 3] - solver.palm(*_split(path(t)))[:3, 3]) * 0.   # key-fixed target
            values, _ = ik.solve(digit, palm, target, seed, stroke_floor(s, note, t))
            poses.append(values)
        s.poses = tuple(poses)
        keyposes += [Keypose(s.hover, poses[0]), Keypose(s.lift_end, poses[3])]
        # A finger swings sideways toward its next key only as far as the hand has come to it: one pointing at its
        # key while the hand is still on its way swings across its neighbours (crossings). Before its stroke its
        # sideways swing is the hover swing times the share of the way the wrist has covered.
        since = previous.lift_end if previous is not None else s.hover - PREPARE_SECONDS
        aim = s.hover - APPROACH_SECONDS
        if digit != 1 and aim > since + LIFT_LATERAL_SECONDS:
            to = path(s.touch)[:2]
            way = float(np.linalg.norm(path(since)[:2] - to))
            if way > APPROACH_MIN_TRAVEL:
                covered = float(np.clip(1. - np.linalg.norm(path(aim)[:2] - to) / way, 0., 1.))
                early = poses[0].copy()
                early[0] *= covered
                keyposes.append(Keypose(aim, early))
        lo, hi = LIFT_LATERAL_SOON
        let_go = (1. - LIFT_LATERAL_KEEP) * float(np.clip((next_hover - s.lift_end - lo) / (hi - lo), 0., 1.))
        if digit != 1 and let_go > 0 and next_hover - s.lift_end <= RELAX_GAP_SECONDS:   # longer: relaxed keyposes
            eased = poses[3].copy()
            eased[0] *= 1. - let_go
            keyposes.append(Keypose(s.lift_end + min(LIFT_LATERAL_SECONDS, .4 * (next_hover - s.lift_end)), eased))
        previous = s
    for p, n in zip(mine, mine[1:]):
        # lift end and next hover can coincide (a finger's quick notes share the time between them): the two
        # shapes are moved towards each other in place (keyposes hold the same arrays), so no frame jumps
        diff = n.poses[0] - p.poses[3]
        apart = float(np.abs(diff).max())
        allowed = TRANSITION_SPEED * max(n.hover - p.lift_end, 0.)
        if apart > allowed:
            share = (1. - allowed / apart) / 2.
            p.poses[3][:] += diff * share
            n.poses[0][:] -= diff * share
    if previous is not None:
        keyposes.append(Keypose(previous.lift_end + RETURN_SECONDS, shape_at(previous.lift_end + RETURN_SECONDS)))
    if last is not None:
        keyposes = [k for k in keyposes if k.time < last + RETURN_SECONDS + 1e-3]
        keyposes += [Keypose(last + RETURN_SECONDS + 1e-3, ARCHED[digit]),
                     Keypose(last + KEY_RETURN_SECONDS + LEAD_IN_SECONDS, RELAXED[digit])]
    else:
        keyposes = [Keypose(0., RELAXED[digit])]
    keyposes.sort(key=lambda k: k.time)
    track = FingerTrack(mine, keyposes)
    if ik.surface is not None:
        for s in mine:
            stroke_clearance(s, notes[s.note].pitch, digit, path, solver, ik, notes)
    # Off the key (free time, strike, lift): the values may let the skin dip into the keys under a moving
    # wrist. The smallest lift is solved on a coarse grid and interpolated, so the correction is smooth in
    # time; in a stroke the key floor is lowered toward its key as for the stroke's own poses.
    # from the earliest keypose (the lead-in runs before zero; a yielding path's own grid starts at zero)
    times = np.arange(min(path.times[0], keyposes[0].time), max(path.times[-1], keyposes[-1].time) + CLEAR_STEP, CLEAR_STEP)
    deltas = np.zeros((len(times), 4), dtype=FLOAT)
    for i, t in enumerate(times):
        s = track.stroke(t)
        if s is None:
            base, lowered = track.keyposed(t), 0.
        elif s.touch <= t <= s.release:
            continue
        else:
            base, lowered = s.values(t), stroke_floor(s, notes[s.note], t)
        deltas[i] = ik.clear(digit, solver.palm(*_split(path(t))), base, lowered) - base
    track.set_correction(times, deltas)
    return track


def key_depths(notes, t):
    """{pitch: depth} of every key down at ``t``."""
    out = {}
    for n in notes.values():
        if n.touch <= t:
            d = n.depth(t)
            if d > 0:
                out[n.pitch] = max(out.get(n.pitch, 0.), d)
    return out


def stroke_clearance(s, pitch, digit, path, solver, ik, notes):
    """The joint-space strike and lift may sweep the finger through a neighbouring (black) key. On a
    fine grid the smallest lift of the finger that keeps its skin over the real key tops (the keys'
    current depths, its own key free) is found; it fades out at the touch and the release, where the
    pad is on its key."""
    grid = np.r_[np.arange(s.hover, s.touch, STROKE_CLEAR_STEP), s.touch,
                 s.release, np.arange(s.release + STROKE_CLEAR_STEP, s.lift_end, STROKE_CLEAR_STEP), s.lift_end]
    nearby = {k: n for k, n in notes.items() if n.touch - 1. <= s.lift_end and n.returned >= s.hover}
    deltas, any_fix = np.zeros((len(grid), 4), dtype=FLOAT), False
    finger = solver.rig.fingers[solver.hand][digit]
    lower, upper = finger.lower.copy(), finger.upper.copy()
    others = {d: RELAXED[d] for d in DIGITS}
    for i, t in enumerate(grid):
        fade = min(1., abs(t - s.touch) / STROKE_CLEAR_FADE) if t <= s.touch else             min(1., abs(t - s.release) / STROKE_CLEAR_FADE)
        if fade <= 0.:
            continue
        palm = solver.palm(*_split(path(t)))
        base = s.base_values(t)
        depths = key_depths(nearby, t)

        def below(v):
            others[digit] = v
            skin = (palm @ solver.local_clearance(digit, others))[:3].T
            return np.maximum(0., ik.surface(skin, depths, pitch) - skin[:, 2]) / FLOOR_SCALE

        if below(base).max() * FLOOR_SCALE < STROKE_CLEAR_TOLERANCE:
            continue
        lower[0] = upper[0] = np.clip(base[0], finger.lower[0], finger.upper[0])
        x, _ = PoseSolver._levenberg_marquardt(lambda v: np.r_[(v - base) / (FREE_VALUE_SCALE * FINGER_JOINTS[digit]), below(v)],
                                               np.clip(base, lower, upper), lower, upper, np.full(4, 1e-2, dtype=FLOAT),
                                               IK_ITERATIONS)
        deltas[i] = (x - base) * fade
        any_fix = True
    if any_fix:
        # A finger that must clear a neighbour on its way down stays at least that clear from the stroke's
        # start (it descends once, not down-up-down); on the way up it keeps a clearance once taken.
        approach = np.where(grid <= s.touch)[0]
        lift = np.where(grid >= s.release)[0]
        for idx in (approach[::-1], lift):
            for k in range(4):
                held = 0.
                for i in idx:
                    if abs(deltas[i, k]) > abs(held):
                        held = deltas[i, k]
                    deltas[i, k] = held
        for i, t in enumerate(grid):
            # zero at the key (touch, release) and where the stroke joins the free track (hover, lift end):
            # the lift then reads as the finger rising a little before it strikes. The join eases in over
            # a share of the approach (lift), not a fixed instant: a fast ramp there read as a kick.
            if t <= s.touch:
                join = ease((t - s.hover) / max(STROKE_JOIN_SHARE * (s.touch - s.hover), 1e-6))
                key = min(1., abs(s.touch - t) / STROKE_CLEAR_FADE)
            else:
                join = ease((s.lift_end - t) / max(STROKE_JOIN_SHARE * (s.lift_end - s.release), 1e-6))
                key = min(1., abs(t - s.release) / STROKE_CLEAR_FADE)
            deltas[i] *= min(join, key)
        # A correction switching on and off between grid points reads as a tremor: spread each one over its
        # neighbours (a lift held a little longer is invisible), then keep it zero at the touch and release.
        spread = np.zeros_like(deltas)
        for i in range(len(grid)):
            w = np.exp(-.5 * ((grid - grid[i]) / STROKE_CLEAR_SMOOTH) ** 2)
            spread[i] = np.where(np.abs(deltas).max(axis=0) > 0, (w[:, None] * deltas).sum(0) / w.sum(), 0.)
            spread[i] = np.where(np.abs(spread[i]) > np.abs(deltas[i]), spread[i], deltas[i])
        for i, t in enumerate(grid):
            edge = s.touch if t <= s.touch else s.release
            spread[i] *= min(1., abs(t - edge) / STROKE_CLEAR_FADE)
        s.clearance = (grid, spread)


# ---------- per frame ----------
def frame_hand(t, path, tracks, notes, waypoints, solver, ik, depths=None, previous=None, apply_room=True):
    """Wrist vector, finger values {digit: 4} and the worst pad error on a pressed key at ``t``.
    ``depths`` (pitch: key depth now) keeps riding fingers out of the neighbouring keys. ``previous``
    ({digit: (note, values)}, updated) warm-starts a riding finger from the frame before."""
    vector = path(t)
    palm = solver.palm(*_split(vector))
    values, error = {}, 0.
    riding = [tracks[d].stroke(t) for d in DIGITS]
    stretch = stretched(notes[s.note].pitch for s in riding if s is not None and s.touch <= t <= s.release)
    for d in DIGITS:
        s = tracks[d].stroke(t)
        if s is None:
            values[d] = tracks[d].free_values(t)
            continue
        if not s.touch <= t <= s.release:
            values[d] = s.values(t)
            if t > s.release and previous is not None and previous.get(d, (None,))[0] == s.note:
                # the lift leaves from where the ride ended (its own solution), not the solved release shape
                x = (t - s.release) / max(s.lift_end - s.release, 1e-6)
                values[d] = values[d] + (previous[d][1] - s.poses[2]) * (1. - ease(x))
            values[d] = values[d] + tracks[d].clear_shift(t)
            continue
        target, seed = stroke_target(s, notes[s.note], waypoints[s.note], t, path)
        if depths is None:
            values[d], e = ik.solve(d, palm, target, seed)
        else:
            start = s.poses[1] if s.poses is not None else None      # the ride continues the strike's shape
            if previous is not None and d in previous and previous[d][0] == s.note:
                start = previous[d][1]
            values[d], e = ik.ride(d, palm, target, seed, depths, notes[s.note].pitch, start, stretch)
            if previous is not None:
                previous[d] = (s.note, values[d])
        error = max(error, e)
    # The smooth corrections were sampled on other shapes (a neighbouring sample held, a stroke's curl):
    # added to this one they may pass the joint limits.
    fingers = solver.rig.fingers[solver.hand]
    values = {d: np.clip(v, fingers[d].lower, fingers[d].upper) for d, v in values.items()}
    room = ik.spacing.make_room(values, {d: give_way(tracks[d].stroke(t), t) for d in DIGITS})
    if apply_room:
        for d in DIGITS:
            corrected = np.clip(tracks[d].room_values(t, room[d]), fingers[d].lower, fingers[d].upper)   # zero on the key (FingerTrack._taper)
            # The correction is sampled apart and spread over its neighbours in time: on a hand moving fast it may
            # belong to another place and push the finger into the keys it should clear. Taken only as far as it helps.
            if np.abs(corrected - room[d]).max() < 1e-6:
                room[d] = corrected
                continue
            base = ik.below_floor(d, {**room, d: room[d]}, palm, 0.).max()
            if ik.below_floor(d, {**room, d: corrected}, palm, 0.).max() > base + ROOM_WORSE / FLOOR_SCALE:
                share = min(ROOM_SHARES, key=lambda c: ik.below_floor(d, {**room, d: room[d] + c * (corrected - room[d])},
                                                                    palm, 0.).max())
                corrected = room[d] + share * (corrected - room[d])
            room[d] = corrected
    return vector, room, error


def give_way(stroke, t):
    """Share of the make-room swing a finger takes at ``t``: all when free, none on its key, in between while
    it strikes (less the nearer the key) or lifts (more the further off)."""
    if stroke is None:
        return 1.
    if stroke.touch <= t <= stroke.release:
        return 0.
    if t < stroke.touch:
        x = (t - stroke.hover) / max(stroke.touch - stroke.hover, 1e-6)
        return float(np.clip(1. - x * x, 0., 1.))
    x = (t - stroke.release) / max(stroke.lift_end - stroke.release, 1e-6)
    return float(np.clip(x * x, 0., 1.))


def room_corrections(times, path, tracks, notes, waypoints, solver, ik):
    """On a rolled hand the spacing's sideways swing axis is tilted: a free finger swung aside also
    sinks. Sampled after the spacing, the smallest lift that keeps each free finger over the keys;
    applied smoothly (FingerTrack.set_room), since a per-frame clear reads as a tremor."""
    deltas = {d: np.zeros((len(times), 4), dtype=FLOAT) for d in DIGITS}
    for i, t in enumerate(times):
        vector, room, _ = frame_hand(t, path, tracks, notes, waypoints, solver, ik, apply_room=False)
        palm = solver.palm(*_split(vector))
        for d in DIGITS:
            s = tracks[d].stroke(t)
            if s is None:
                deltas[d][i] = ik.clear(d, palm, room[d]) - room[d]
            elif not s.touch <= t <= s.release:
                deltas[d][i] = ik.clear(d, palm, room[d], stroke_floor(s, notes[s.note], t)) - room[d]
    return deltas


def contact_error(t, path, notes, waypoints, solver, ik):
    """Over the keys this hand holds at ``t`` (the ride only): (worst pad error, worst sideways swing of a
    finger 2..5 beyond its SIDEWAYS_FREE, degrees, worst depth of a free finger in the keys after its own
    clearance, metres)."""
    palm = solver.palm(*_split(path(t)))
    worst, swing, sunk = 0., 0., 0.
    busy = set()
    for n in notes.values():
        if n.touch - REFINE_FREE_NEAR <= t <= n.release + REFINE_FREE_NEAR:
            busy.add(n.digit)
        if n.touch <= t <= n.release and n.id in waypoints:
            top, seed = waypoint(waypoints[n.id], t, path)
            values, e = ik.solve(n.digit, palm, top - np.array([0., 0., n.depth(t)], dtype=FLOAT), seed)
            worst = max(worst, e)
            if n.digit != 1:
                swing = max(swing, abs(float(values[0])) - SIDEWAYS_FREE[n.digit])
    fingers = solver.rig.fingers[solver.hand]
    for d in DIGITS:
        if d not in busy:
            lifted = np.r_[RELAXED[d][0], fingers[d].upper[1], RELAXED[d][2:]]
            if not ik.below_floor(d, {**RELAXED, d: lifted}, palm, 0.).any():
                continue                    # lifted all the way it clears: no need to solve how little will do
            values = {**RELAXED, d: ik.clear(d, palm, RELAXED[d])}
            sunk = max(sunk, float(ik.below_floor(d, values, palm, 0.).max()) * FLOOR_SCALE)
    return worst, swing, sunk


def refine_score(errors):
    """``contact_error`` samples (n, 3) -> how far past what calls for an intermediate pose (> 1)."""
    errors = np.asarray(errors, FLOAT)
    return np.maximum.reduce([errors[:, 0] / REFINE_TOLERANCE, errors[:, 1] / REFINE_LATERAL, errors[:, 2] / REFINE_FREE])


def refine_points(times, scores, anchors):
    """Times needing an intermediate pose: the worst sample of every run with ``refine_score`` above 1, away
    from anchors."""
    taken = np.array(sorted(a.time for a in anchors))
    out, run = [], []
    for t, e in list(zip(times, scores)) + [(None, 0.)]:
        if e > 1.:
            run.append((e, t))
            continue
        if run:
            t_worst = max(run)[1]
            if not len(taken) or np.min(np.abs(taken - t_worst)) > .03:
                out.append(t_worst)
            run = []
    return out


class YieldingPath:
    """A wrist path moved sideways by ``offset`` (x, metres, on a time grid): an idle hand out of the other's way."""

    def __init__(self, base, times, offset):
        self.base, self.times, self.offset = base, np.asarray(times, float), np.asarray(offset, FLOAT)

    def __getattr__(self, name):
        if name in ('base', 'times', 'offset'):
            raise AttributeError(name)
        return getattr(self.base, name)

    def __call__(self, t):
        out = self.base(t)
        out[0] += float(np.interp(t, self.times, self.offset))
        return out


def yield_offsets(paths, notes, duration, apart, speed, busy_margin, rate=60.):
    """Sideways offsets per hand: where the wrists come closer than ``apart`` and only one hand is playing, the idle
    one moves out (left hand to the left, right to the right) by the shortfall, ramping at ``speed`` and held at
    zero while it plays (``busy_margin``: seconds before an onset / after a release). Both playing: unchanged."""
    times = np.arange(0., duration + 1. / rate, 1. / rate)
    x = {h: np.array([paths[h](t)[0] for t in times], dtype=FLOAT) for h in 'LR'}
    busy = {h: np.zeros(len(times), bool) for h in 'LR'}
    for n in notes.values():
        busy[n.hand] |= (times >= n.onset - busy_margin[0]) & (times <= n.release + busy_margin[1])
    short = np.maximum(0., apart - (x['R'] - x['L']))
    out = {}
    for h, sign in (('L', -1.), ('R', 1.)):
        other = 'R' if h == 'L' else 'L'
        want = np.where(~busy[h] & busy[other], short, 0.)
        want = np.where(~busy[h] & ~busy[other], short / 2, want)
        step = speed / rate
        env = want.copy()                       # rise and fall no faster than ``speed``
        for i in range(1, len(env)):
            env[i] = max(env[i], env[i - 1] - step)
        for i in range(len(env) - 2, -1, -1):
            env[i] = max(env[i], env[i + 1] - step)
        if busy[h].any():                       # zero while playing, reached at ``speed``
            env = np.minimum(env, _distance_to(busy[h]) * step)
        k = int(rate * .1) | 1                  # round the corners
        env = np.convolve(np.pad(env, k // 2, mode='edge'), np.ones(k, dtype=FLOAT) / k, mode='valid')
        env[busy[h]] = 0.
        out[h] = (times, sign * env)
    return out


def _distance_to(mask):
    """Frames from each frame to the nearest True frame of ``mask``."""
    n = len(mask)
    d = np.full(n, n, float)
    last = -n
    for i in range(n):
        if mask[i]:
            last = i
        d[i] = i - last
    last = 2 * n
    for i in range(n - 1, -1, -1):
        if mask[i]:
            last = i
        d[i] = min(d[i], last - i)
    return d
