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
from .checks import CELL, OWN_KEY_TOLERANCE, Heightfield, brushed, stretched
from .kinematics import DIGITS
from .pose import RELAXED, PoseSolver

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
CORRECTION_FADE = .12            # free-time key-clearance and room corrections fade over this next to a stroke (s)
ANTICIPATE_SECONDS = .06         # a hurried leap sets off this long before the key is released ...
ARRIVE_LATE_SECONDS = .05        # ... and lands this long after the next touch (within the press)
DEPART_DELAY = .02               # after the last needed key is released
ARC_RATIO, ARC_MAX = .15, .04    # wrist rises by this fraction of the travel (m), at most this
LIFT_RATE = .3                   # m/s: a hurried leap lifts only between the release and the next touch, at most
                                 # this times that time (a hop, not a slide over the keys)
ARC_MIN_DISTANCE = .02
LEAD_IN_SECONDS = 1.2            # from the rest pose to the first anchor
LIFT_HEIGHT = .008               # pad above the key top at the end of a lift (m)
HOVER_HEIGHT = (.002, .025)      # pad above the key before the stroke: soft starts on the key, loud from above
RELAX_GAP_SECONDS = .6           # free time after which a finger returns to the relaxed hand
PREPARE_SECONDS = .22            # a rested finger takes its shape for the next key this long before the stroke
RETURN_SECONDS = .3              # ... and returns to the relaxed hand within this after a lift
RIDE_RETRY = .001                # a warm-started ride further than this off its key is solved again from the anchor
IK_ITERATIONS = 20
STROKE_CLEAR_STEP = .01          # grid of the strike / lift clearance (s)
STROKE_CLEAR_FADE = .02          # ... faded out this close to the touch and the release
STROKE_CLEAR_TOLERANCE = .0005   # skin this far into another key is left alone (m)
STROKE_CLEAR_SMOOTH = .025       # gaussian width spreading a correction over neighbouring grid points (s)
FLOOR_SCALE = .0005              # skin below the key surface, metres per residual unit
FREE_VALUE_SCALE = 20.           # a free finger leaves its keyposed values by as little as possible (deg per unit)
IK_VALUE_SCALE = 15.             # prior toward the seed: fixes the finger's free direction (3 targets, 4 values)
FLOOR_RAMP = .015                # the key floor phases in over a stroke: lowered this much at the key (m)
SPACING_SHARE = .85              # free fingers keep this share of the relaxed fingertip spacing from their neighbours
SPACING_SOFTNESS = .002          # smooth maximum width (m)
SPREAD_COUPLING = .5             # share of a neighbour's stretch beyond the relaxed gap a free finger follows (tendons)
SPREAD_ROUNDS = 4                # ... passed on along the hand in this many rounds
SPREAD_SOFTNESS = .0005          # its onset eased over this (m): no pull at the relaxed gap
CLEAR_STEP = .05                 # free-finger key clearance is solved at this spacing and interpolated (s)
HOLD_SAMPLES = 3                 # ... each sample then holds the largest correction within this many samples (0.15 s)
REFINE_STEP = .04                # contact check spacing along the wrist path (s)
REFINE_TOLERANCE = .0015         # held-key pad error that calls for an intermediate pose (m) ...
SIDEWAYS_FREE = {1: 30., 2: 8., 3: 8., 4: 10., 5: 20.}   # deg a finger swings sideways on its key without strain
                                                 # (thumb and little finger spread for an octave, sharing it); coherence
                                                 # prefers less; the thumb does not call for intermediate poses
REFINE_LATERAL = 7.              # ... or a finger 2..5 swung sideways this far beyond SIDEWAYS_FREE on its key (deg)
LIFT_LATERAL_KEEP = .35          # a lifted finger lets its sideways swing go to this share, over the free time after
LIFT_LATERAL_SOON = (.15, .45)   # its lift, when its next stroke is further than this away (s; none when sooner) ...
LIFT_LATERAL_SECONDS = .2        # ... within at most this (the lift itself is too quick for it)
TRANSITION_SPEED = 400.          # deg/s: a finger's lifted shape and its next hover shape (solved apart) meet at
                                 # most this fast; when one finger's notes follow closely, both shapes meet halfway


# A pianist's hand: knuckles moderately bent and the finger joints curved, so the fingertip comes down
# onto the key (the calibrated relaxed hand is flatter: MCP 34, IP 18). The thumb keeps its relaxed shape.
# Geometric bends as measured on a reference pianist (FurElise dataset, performance 060: PIP ~27, DIP ~12 deg median): the last
# joint stays open, so the finger meets the key with its pad rather than its tip.
PLAYING_SHAPE = {1: RELAXED[1], **{d: np.array([0., 14., 42., 22.]) for d in (2, 3, 4, 5)}}   # arched: the palm stays off the keys
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
    deltas = np.asarray(deltas, float)
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
                values[-1], arcs[-1] = np.asarray(v, float), arc
                continue
            times.append(t)
            values.append(np.asarray(v, float))
            arcs.append(arc)
        self.times, self.values, self.arcs = np.array(times), np.array(values), np.array(arcs)
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
        self.accents = np.array(rows) if rows else np.zeros((0, 3))

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


def wrist_path(anchors, notes, strokes, rest, duration):
    """``anchors`` of one hand (with poses); intermediate poses have no incoming keys and are passed through."""
    anchors = sorted(anchors, key=lambda a: a.time)
    if not anchors:
        return WristPath([(0., rest, 0.), (duration, rest, 0.)])

    def ready(a):        # the wrist is settled when the first stroke of the anchor begins
        return min((strokes[k].hover for k in a.incoming), default=a.time)

    first, v0 = anchors[0], _vector(anchors[0])
    settle = ready(first)
    leave = settle - LEAD_IN_SECONDS
    knots = [(min(0., leave - 1e-3), rest, 0.), (leave, rest, _arc(rest, v0, True)), (settle, v0, 0.)]
    arrived = {}         # anchor index -> when the wrist is at its pose (after the touch, for a hurried leap)
    for i, (a, b) in enumerate(zip(anchors, anchors[1:])):
        va, vb = _vector(a), _vector(b)
        at = arrived.get(i, a.time)
        knots.append((at, va, 0.))
        if not b.incoming:
            # An intermediate pose holds fewer keys: the wrist stays where the others are reached until
            # they are released (a stretched chord has no slack for an early drift).
            keep = set(b.held)
            free = min(b.time - MIN_MOVE_SECONDS,
                       max([at] + [notes[k].release for k in a.incoming + a.held if k not in keep]))
            if free > at + 1e-3:
                knots.append((free, va, 0.))
            continue
        keep = set(b.held)
        free = min(b.time, max([at] + [notes[k].release for k in a.incoming + a.held if k not in keep]))
        natural = MOVE_BASE_SECONDS + float(np.linalg.norm(vb[:3] - va[:3])) / MOVE_SPEED
        travel = max(UNHURRIED * natural, MIN_TRAVEL_SECONDS)
        arrive = max(ready(b), free)
        start = free + DEPART_DELAY
        end = start + travel
        if end > arrive:
            end = max(arrive, min(b.time, start + natural))
            start = max(free, end - travel)
        if end - start < natural:
            # A leap the score leaves no time for: the wrist sets off while the finger still holds
            # the key and lands while the next finger is already pressing (as a pianist does).
            after = anchors[i + 2].time - 1e-3 if i + 2 < len(anchors) else np.inf
            onset = min(notes[k].onset for k in b.incoming)
            start = max(at, min(start, free - ANTICIPATE_SECONDS))
            end = min(max(end, min(b.time + ARRIVE_LATE_SECONDS, onset, start + natural)), after)
        if end - start < MIN_MOVE_SECONDS:
            start = max(at, end - MIN_MOVE_SECONDS)
        arc = _arc(va, vb, not keep)
        if start < free and arc:
            # set off while a key is still held: the hand hops from its release to the next touch
            touch = min(end, min(notes[k].touch for k in b.incoming))
            arc = (min(arc, LIFT_RATE * max(touch - free, 0.)), free, touch)
        if start > at + 1e-3:
            knots.append((start, va, arc))
        else:
            knots[-1] = (at, va, arc)
        if end < b.time - 1e-3:
            knots.append((end, vb, 0.))
        elif end > b.time + 1e-3:
            arrived[i + 1] = end
    last = anchors[-1]
    vl = _vector(last)
    end = max([last.time] + [n.release for n in notes.values()]) + KEY_RETURN_SECONDS
    knots += [(last.time, vl, 0.), (end, vl, _arc(vl, rest, True)), (end + LEAD_IN_SECONDS, rest, 0.)]
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
            out.setdefault(k, []).append((a.time, a.pose.targets[digit] + np.array([0., 0., depth]),
                                          np.array(a.pose.values[digit], float), np.array(a.pose.wrist, float)))
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
        return v + np.array([np.interp(t, times, deltas[:, k]) for k in range(4)])


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
            height = hover_height(n.velocity)
            strike = 2 * height * max(n.onset - n.touch, 1e-3) / KEY_TRAVEL
            timing.append([strike, height, KEY_RETURN_SECONDS, LIFT_HEIGHT])
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
    return top + np.array([0., 0., z]), seed


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

    def _taper(self, t):
        """Free-time corrections fade out next to a stroke, whose own poses are solved clear of the keys:
        switched off at the hover they snapped the finger into its stroke."""
        i = np.searchsorted(self.starts, t, side='right')
        gap = np.inf
        if i < len(self.strokes):
            gap = self.strokes[i].hover - t
        if i > 0:
            gap = min(gap, t - self.strokes[i - 1].lift_end)
        return min(1., max(0., gap) / CORRECTION_FADE)

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
    ``OWN_KEY_TOLERANCE`` (its pad rests on it)."""

    def __init__(self, heightfield):
        self.field = heightfield
        self.top = np.where(np.isfinite(heightfield.top), heightfield.top, -1.)

    def __call__(self, points, depths, own_pitch):
        return self._surface(points, depths, own_pitch)[0]

    def _surface(self, points, depths, own_pitch):
        """(surface height, share of it on keys a finger holding ``own_pitch`` may brush)."""
        f = self.field
        u = (points[:, :2] - f.origin) / CELL
        i = np.clip(np.floor(u).astype(int), 0, np.array(self.top.shape) - 2)
        w = np.clip(u - i, 0., 1.)
        out, soft = 0., 0.
        for dx, dy, weight in ((0, 0, (1 - w[:, 0]) * (1 - w[:, 1])), (1, 0, w[:, 0] * (1 - w[:, 1])),
                               (0, 1, (1 - w[:, 0]) * w[:, 1]), (1, 1, w[:, 0] * w[:, 1])):
            ix, iy = i[:, 0] + dx, i[:, 1] + dy
            pitch = f.pitch[ix, iy]
            z = self.top[ix, iy] - np.array([depths.get(int(p), 0.) for p in pitch])
            z = np.where(pitch == own_pitch, z - OWN_KEY_TOLERANCE, z)
            out = out + weight * z
            soft = soft + weight * (Heightfield.brushable(pitch, depths, own_pitch) & np.isfinite(f.top[ix, iy]))
        return out, soft

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
        relaxed = {d: RELAXED[d] for d in DIGITS}
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
        return np.array([lateral[proximal].mean(), lateral[~proximal].mean()])

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
            pull = {d: np.zeros(2) for d in moves}
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
            probe = np.array(values[d], float)
            probe[0] += 1.
            slope = self.positions(d, {**values, d: probe}) - s[d]          # metres per degree of swing
            swings = [shift[k] / slope[k] for k in range(2) if abs(slope[k]) > 1e-6 and abs(shift[k]) >= 1e-5]
            if not swings:
                continue
            v = np.array(values[d], float)
            # the larger of the proximal and distal needs, blended (no flip when they disagree in sign)
            weights = np.abs(swings) ** 4 + 1e-30
            v[0] = np.clip(v[0] + float(np.dot(swings, weights) / weights.sum()), fingers[d].lower[0], fingers[d].upper[0])
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
        return np.zeros(1)
    return np.array([max(0., distal_pitch(solver, digit, values, palm) - PAD_PITCH_MAX) / PAD_PITCH_SCALE])


class FingerIK:
    """Four-value pad IK with the wrist fixed; optionally keeps the finger's skin above the keys."""

    def __init__(self, solver: PoseSolver):
        self.solver = solver
        self.floor = KeyFloor(solver.regions)
        self.surface = KeySurface(solver.heightfield) if solver.heightfield is not None else None
        self.spacing = Spacing(solver)

    def ride(self, digit, palm, target, seed, depths, own_pitch, start=None, brush=False):
        """Pad on its (moving) key point, the finger's sides kept out of the neighbouring keys. ``start``
        (the previous frame's values) keeps a finger strained against its limits on one solution."""
        if self.surface is None:
            return self.solve(digit, palm, target, seed)
        finger = self.solver.rig.fingers[self.solver.hand][digit]
        values = {d: RELAXED[d] for d in DIGITS}
        seed = np.asarray(seed, float)

        def residual(v):
            values[digit] = v
            point = self.solver.contact_point(self.solver.pad_world(digit, values, palm))
            skin = (palm @ self.solver.local_clearance(digit, values))[:3].T
            below = self.surface.clearance(skin, depths, own_pitch, brush=brush) / FLOOR_SCALE
            steady = (v - start) / RIDE_STEADY if start is not None else np.zeros(0)
            return np.r_[(point - target) / .0002, (v - seed) / IK_VALUE_SCALE, below,
                         pad_residual(self.solver, digit, v, palm), steady]

        def run(x0):
            x, _ = PoseSolver._levenberg_marquardt(residual, np.clip(x0, finger.lower, finger.upper),
                                                   finger.lower, finger.upper, np.full(4, 1e-2), IK_ITERATIONS)
            values[digit] = x
            point = self.solver.contact_point(self.solver.pad_world(digit, values, palm))
            return x, float(np.linalg.norm(point - target))

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
        start = np.asarray(values, float)
        if not self.below_floor(digit, {**others, digit: start}, palm, lowered).any():
            return start
        finger = self.solver.rig.fingers[self.solver.hand][digit]

        def residual(v):
            others[digit] = v
            return np.r_[(v - start) / FREE_VALUE_SCALE, self.below_floor(digit, others, palm, lowered)]

        # The finger lifts over the keys; its sideways swing is left to the spacing (dodging sideways
        # here would be undone by it, back onto a black key).
        lower, upper = finger.lower.copy(), finger.upper.copy()
        lower[0] = upper[0] = np.clip(start[0], finger.lower[0], finger.upper[0])
        x, _ = PoseSolver._levenberg_marquardt(residual, np.clip(start, lower, upper), lower, upper, np.full(4, 1e-2),
                                               IK_ITERATIONS)
        return x

    def solve(self, digit, palm, target, seed, lowered=None):
        """Values putting the pad's contact point on ``target`` (world); weak prior toward ``seed``.
        With ``lowered`` (own key depth) the skin also stays above the key floor."""
        finger = self.solver.rig.fingers[self.solver.hand][digit]
        values = {d: RELAXED[d] for d in DIGITS}
        seed = np.asarray(seed, float)

        def residual(v):
            values[digit] = v
            point = self.solver.contact_point(self.solver.pad_world(digit, values, palm))
            out = np.r_[(point - target) / .0002, (v - seed) / IK_VALUE_SCALE,
                        pad_residual(self.solver, digit, v, palm)]
            if lowered is not None:
                out = np.r_[out, self.below_floor(digit, values, palm, lowered)]
            return out

        x, _ = PoseSolver._levenberg_marquardt(residual, np.clip(seed, finger.lower, finger.upper),
                                               finger.lower, finger.upper, np.full(4, 1e-2), IK_ITERATIONS)
        values[digit] = x
        point = self.solver.contact_point(self.solver.pad_world(digit, values, palm))
        return x, float(np.linalg.norm(point - target))


def finger_track(hand, digit, notes, strokes_by_note, waypoints, path, solver, ik):
    """Keyposes at each stroke's start (hover) and end (lifted), relaxed hand in long gaps."""
    mine = sorted((replace(strokes_by_note[n.id]) for n in notes.values()
                   if n.hand == hand and n.digit == digit and n.id in waypoints), key=lambda s: s.hover)
    relaxed = PLAYING_SHAPE[digit]          # free fingers wait curved, ready to play
    keyposes, previous = [], None
    for k, s in enumerate(mine):
        note = notes[s.note]
        next_hover = mine[k + 1].hover if k + 1 < len(mine) else np.inf
        if previous is None:
            keyposes.append(Keypose(s.hover - PREPARE_SECONDS, relaxed))
        else:
            gap = s.hover - previous.lift_end
            if gap > RELAX_GAP_SECONDS:
                keyposes += [Keypose(previous.lift_end + min(.3 * gap, RETURN_SECONDS), relaxed),
                             Keypose(s.hover - min(.3 * gap, PREPARE_SECONDS), relaxed)]
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
        keyposes.append(Keypose(previous.lift_end + RETURN_SECONDS, relaxed))
    else:
        keyposes = [Keypose(0., relaxed)]
    track = FingerTrack(mine, keyposes)
    # Free time: the keyposed values may let the skin dip into the keys under a moving wrist. The
    # smallest lift is solved on a coarse grid and interpolated, so the correction is smooth in time.
    times = np.arange(path.times[0], path.times[-1] + CLEAR_STEP, CLEAR_STEP)
    deltas = np.zeros((len(times), 4))
    for i, t in enumerate(times):
        if track.stroke(t) is None:
            base = track.keyposed(t)
            deltas[i] = ik.clear(digit, solver.palm(*_split(path(t))), base) - base
    track.set_correction(times, deltas)
    if ik.surface is not None:
        for s in mine:
            stroke_clearance(s, notes[s.note].pitch, digit, path, solver, ik, notes)
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
    deltas, any_fix = np.zeros((len(grid), 4)), False
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
        x, _ = PoseSolver._levenberg_marquardt(lambda v: np.r_[(v - base) / FREE_VALUE_SCALE, below(v)],
                                               np.clip(base, lower, upper), lower, upper, np.full(4, 1e-2),
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
    playing = {d for d in DIGITS if tracks[d].stroke(t) is not None}
    room = ik.spacing.make_room(values, {d: give_way(tracks[d].stroke(t), t) for d in DIGITS})
    if apply_room:
        for d in DIGITS:
            if d not in playing:
                room[d] = tracks[d].room_values(t, room[d])
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
    deltas = {d: np.zeros((len(times), 4)) for d in DIGITS}
    for i, t in enumerate(times):
        vector, room, _ = frame_hand(t, path, tracks, notes, waypoints, solver, ik, apply_room=False)
        palm = solver.palm(*_split(vector))
        for d in DIGITS:
            if tracks[d].stroke(t) is None:
                deltas[d][i] = ik.clear(d, palm, room[d]) - room[d]
    return deltas


def contact_error(t, path, notes, waypoints, solver, ik):
    """Over the keys this hand holds at ``t`` (the ride only): (worst pad error, worst sideways swing of a
    finger 2..5 beyond its SIDEWAYS_FREE, degrees)."""
    palm = solver.palm(*_split(path(t)))
    worst, swing = 0., 0.
    for n in notes.values():
        if n.touch <= t <= n.release and n.id in waypoints:
            top, seed = waypoint(waypoints[n.id], t, path)
            values, e = ik.solve(n.digit, palm, top - np.array([0., 0., n.depth(t)]), seed)
            worst = max(worst, e)
            if n.digit != 1:
                swing = max(swing, abs(float(values[0])) - SIDEWAYS_FREE[n.digit])
    return worst, swing


def refine_score(errors):
    """``contact_error`` samples (n, 2) -> how far past what calls for an intermediate pose (> 1)."""
    errors = np.asarray(errors, float)
    return np.maximum(errors[:, 0] / REFINE_TOLERANCE, errors[:, 1] / REFINE_LATERAL)


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
        self.base, self.times, self.offset = base, np.asarray(times, float), np.asarray(offset, float)

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
    x = {h: np.array([paths[h](t)[0] for t in times]) for h in 'LR'}
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
        env = np.convolve(np.pad(env, k // 2, mode='edge'), np.ones(k) / k, mode='valid')
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
