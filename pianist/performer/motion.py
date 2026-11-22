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

import json
import os
import sys

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
STRIKE_REST = float(os.environ.get('PIANOMOTION_STRIKE_REST', .24))   # a free hand this long before its next touch
                                 # lifts even for a single note (the gate of the strike/crawl ladder)
STRIKE_SHIFT = .025              # wrist travel (m) that counts as a position change
LIFT_WINDOWS = (.12, .25, .40, .7, 1.)         # free time (s) ...
# P3 ("多敲", chosen 2026-11-20): the short end rises enough to read as a strike at 8 mm, and the gate lets a
# 0.24 s free window lift at all, so fast passages get some. P0 was (0., .0062, .0135, .04, .065) with a 0.30 s
# gate; the ladder lives in output/strike_presets.py.
LIFT_HEIGHTS = (.0045, .011, .022, .048, .068)   # ... and the wrist's rise (m) at a middle loudness
# The ladder (output/strike_presets.py) is runnable without editing: PIANOMOTION_LIFT="h0,h1,h2,h3,h4" and
# PIANOMOTION_STRIKE_REST=0.24 override the two knobs that decide how much of the playing is struck.
if os.environ.get('PIANOMOTION_LIFT'):
    LIFT_HEIGHTS = tuple(float(x) for x in os.environ['PIANOMOTION_LIFT'].split(','))
LIFT_LOUDNESS = ((40., 110.), (.75, 1.35))     # velocity of the chord struck -> share of that rise (a loud chord from higher)
LIFT_SCALE = 1.                  # the piece's own share (inputs/<piece>/motion.json "lift"): its character, lyrical to spirited
LEAD_IN_SECONDS = 1.2            # from the rest pose to the first anchor
REST_HOLD = 1.0                  # the hands stay on the lap this long after the video starts: the first travel used to
# One shared intro for both hands (about 150 frames at 30 fps), the user's motion spec (2026-11-17 revision):
# a breath on the lap; a FLAT draw -- the fingertips keep sliding back along the lap while the wrist barely
# rises, the elbows splaying a little outward (room to adduct later); then, with no stop between the steps,
# the forearm carries the whole hand off the lap -- the relaxed hand hangs from the wrist -- and the elbow
# adducts and sinks so the forearm pokes the wrist up to the navel/chest midline, the wrist tension through
# the poke's latter half flattening the palm by its top; a brief wrist tension tilts the palm axis 45 deg off
# vertical (fingertips up), then the wrist lets go: the palm falls toward flat while the hand creeps a few
# centimetres forward -- the slow "pseudo-hover" that reads as a pause without ever stopping; then the slow
# reach finds the keys (slight droop over the key fronts, flat above them) and the slow weighted press.
# The fingers' second joints curl in as one group at lift-off (45-55 deg) and open only across the reach.
INTRO_BREATH_SECONDS = 1.2       # the still opening, set in pipeline as the video-time of ``leave``
REST_DRAW_T = .30                # the flat draw: fingertips sliding back on the lap, the wrist barely rising
REST_POKE_T = .45                # the crisp lift: carried off the lap, elbow in and down, forearm pokes up;
                                 # the palm flattens through the poke's latter half. One unbroken sweep with
                                 # the draw -- the speed peaks between the waypoints and never settles to
                                 # zero until the hover's tail (a stop-and-restart at the poke's top read
                                 # as two separate gestures)
REST_LIFT_T = REST_DRAW_T + REST_POKE_T
REST_REACH_T = 1.15              # nominal fall's length: the sail from the poke's top to the poise above the
                                 # first key -- the palm's fall rides it (the real length takes the slack)
REST_POISE_T = .55               # nominal still poise above the keys before the seat
REST_SEAT_T = .35                # the final settle into the playing pose -- the HAND stops there; the
                                 # first note itself is the FINGER's stroke (a tap, never the hand crawling
                                 # onto the key)
REST_HOVER_HEIGHT = .835         # wrist at the poke's top: midpoint between navel (.679) and chest (.990);
                                 # unchanged -- the lift reads clearly with the raised mid waypoint (.78),
                                 # no need to raise the apex itself
REST_DRAW_BACK = .030            # the flat draw drags the hand this far back along the lap, toward the body
                                 # (the +y sign is toward the keys: the draw SUBTRACTS). Capped at 30 mm: the
                                 # upper arm is 219 mm and already at full hang, a deeper drag would ask the
                                 # elbow to swing past its backward reach and the IK would wander
REST_DRAW_OUT = .004             # ... almost NO outward share: the draw is a BACKWARD drag along the lap
                                 # (the user: "正常路径我看起来是直接后撤肘"). At .020 the wrist slid 12-17
                                 # mm outward and the elbow splayed 8-16 mm -- and its zero-phase smoothing
                                 # started the splay ~0.2 s BEFORE the hand moved (visible around frame 30).
                                 # 4 mm keeps a hair of clearance for the later adduction without the flare.
REST_DRAW_UP = .025              # the wrist gains only this much height during the draw (fingers still on the
                                 # lap -- the thigh surface simply rises a little toward the hip)
REST_LIFT_MID_Z = .78            # wrist height as the hand is carried off the lap (the palm hangs behind
                                 # it). Originally .72 (just at the ribs); raised 6 cm per the user so the
                                 # lift reads clearly and the draw-to-lift bend is no longer a hairpin.
                                 # Still below the quiet apex (.822) and the poke top (.835): no inversion.
REST_HOVER_OUT = .10             # share of the sideways lap-to-key distance taken during the lift
REST_POKE_FWD = .012             # the poke's top sits this far forward of the lap rest -- barely: the lift
                                 # rises where the draw left the hand, the forward travel belongs to the
                                 # creep and the reach (a poke top 4.5 cm forward swung the upper arm ~20
                                 # deg forward, frames 46-59: the elbow was dragged 9 cm by the wrist path)
REST_FALL_CREST = .010           # the fall's sail crests this far above the poke-to-poise chord's height: the
                                 # world path curves (a height arc). Capped low: a bigger crest bounces the
                                 # hand UP off the poke's top (a release falls, it does not hop), and a
                                 # forward bow would overshoot past the poise and settle back (the wobble)
REST_REACH_FROM_KEY = (0., -.012, .007)   # the poise: a hair behind and above the wrist the strokes hold --
                                 # the seat is then the hand FINDING its key (a gentle ~1 cm settle, the
                                 # user's "按音去找琴键"), not a sudden translate or drop into the first note
REST_QUIET_POISE_T = .45         # the late hand's still beat above its keys (a quieter, shorter stay than the opener's)
REST_QUIET_CLEAR = .020          # the quiet pickup's pull toward the body: the lap rests UNDER the keyboard's
                                 # front overhang, so rising in place rams the curled fingertips into the
                                 # keys from below -- the pull clears the key fronts like the opener's draw does
REST_QUIET_APEX = .822           # the quiet arc's apex: flight height, a hair under the opener's poke top
REST_QUIET_FLAT_AT = .55         # the late hand's palm hangs until this share of its arc, then flattens; no tilt beat

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
# The playing range each joint is pushed back into (lo, hi; the first value is unused -- the sideways swing):
# the middle joint keeps its playing range, the last joint curls at least 22-24 (a distal phalanx ~45-48 deg
# down, 10 deg more than the fingers were sitting at), and the outer three keep a little knuckle bend instead
# of opening flat. The THUMB is left alone (its 8 deg knuckle is its own natural shape) and the INDEX's knuckle
# is left alone too: the user reads the index as a different finger, held flat and long over the key.
POSTURE_BANDS = {}
for _d in DIGITS:
    POSTURE_BANDS[_d] = ((0., 0.), (None, 2.), (18., 62.), (22., 45.))
POSTURE_BANDS[1] = ((0., 0.), (None, None), (None, None), (None, None))
POSTURE_BANDS[2] = ((0., 0.), (None, None), (18., 62.), (24., 45.))
PIP_POSTURE_SCALE = float(os.environ.get('PIANOMOTION_PIP_POSTURE', 2.5))  # deg outside that range per residual unit


def posture_flat(digit):
    """The bands for joints 1..3 as one flat array for the compiled ride (lo, hi per joint; sideways joint 0 is
    free, matching posture_residual and fast_settle.ride_residuals's ``scales[4 + 2*(j-1)]``); None becomes an
    unreachable bound."""
    return tuple(x for lo, hi in POSTURE_BANDS[digit][1:]
                 for x in (-1e4 if lo is None else lo, 1e4 if hi is None else hi))


def posture_residual(v, digit):
    """The joint hinges at ``v`` (one residual per joint, the sideways swing free)."""
    out = np.empty(3, dtype=FLOAT)
    for k, j in enumerate((1, 2, 3)):
        lo, hi = POSTURE_BANDS[digit][j]
        out[k] = max((-1e4 if lo is None else lo) - v[j], v[j] - (1e4 if hi is None else hi), 0.) / PIP_POSTURE_SCALE
    return out
PIP_BAND = None                  # diagnosis only: PIANOMOTION_PIP_BAND=lo,hi narrows every finger's middle joint
if os.environ.get('PIANOMOTION_PIP_BAND'):
    _lo, _hi = (float(x) for x in os.environ['PIANOMOTION_PIP_BAND'].split(','))
    PIP_BAND = (_lo, _hi)
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
# The lap rest's finger shape (deg per value: sideways, lift, knuckle, middle): the fingers lie almost flat
# along the thigh. A half curl (30 deg at the knuckle) read as a claw: the rig already tilts the palm's back
# up 15 deg (PALM_PITCH, for playing), so a curled knuckle stacks on it and arches the whole hand. On the lap
# the knuckle stays near its relaxed floor and the joints only bend enough to follow the thigh's curve.
LAP_FINGERS = {1: np.array([0., 0., 6., 4.], dtype=FLOAT),
               2: np.array([0., 0., 10., 6.], dtype=FLOAT),
               3: np.array([0., 0., 10., 6.], dtype=FLOAT),
               4: np.array([0., 0., 10., 6.], dtype=FLOAT),
               5: np.array([0., 0., 10., 6.], dtype=FLOAT)}
LAP_RELEASE_SECONDS = .6         # after the hover the fingers ease from the lap shape into the first note's
LAP_RETURN_SECONDS = 1.0         # after the last keypose of the piece they ease back onto the lap shape
LAP_FADE_SECONDS = .4            # the key-clearance / room corrections fade in this long past the lap rest
# Extra curl while the hand transits under the keybed and hovers there: the relaxed lap curl leaves the
# fingertips grazing the key fronts' underside (they sit a few millimetres into the keys' plan view), so
# the fingers tuck in like a hand really pulled out from under a piano, and unfold only on the approach.
# Extra curl from lift-off to the reach (the user's 2026-11-17 spec): the moment the hand leaves the lap the
# fingers' second joints (the middle joints) curl in as ONE group -- about 45-55 deg of natural tuck -- and
# hold it all the way to the reach, where they open toward the playing curl as the hand goes to its keys.
# The lap shape underneath carries ~10-16 deg (sideways, lift, knuckle, middle), so the tuck adds the rest.
LAP_TUCK = {1: np.array([0., 0., 12., 24.], dtype=FLOAT),
            2: np.array([0., 0., 16., 40.], dtype=FLOAT),
            3: np.array([0., 0., 16., 40.], dtype=FLOAT),
            4: np.array([0., 0., 16., 40.], dtype=FLOAT),
            5: np.array([0., 0., 16., 40.], dtype=FLOAT)}
LAP_TUCK_SECONDS = .30           # the group curl is in during the crisp lift (it starts at the draw's end)
LAP_TUCK_MIN_LIFT = .075         # m: the curl waits until the wrist has risen this far off the lap. Curling while
                                 # the palm still lies on the thigh drives the fingertips down into the leg (measured
                                 # on canonpd: a full curl at 5 mm lift sank the pads 7.4 mm; at ~30 mm it cleared).
                                 # Geometric, not a clock: any hand leaving any lap must clear it before it folds.
                                 # Raised to 75 mm for the Furina body: its fingers are ~40 % longer than canonpd's,
                                 # so a 40 deg curl drops the thumb tip that much further -- 28 mm still buried the
                                 # opening right thumb 13.7 mm into the thigh (measured on the 6 s slice).
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


def hermite_eval(times, values, slopes, t):
    """Cubic Hermite through (times, values) with ``slopes`` (pchip_slopes' output): dense sampling of a smooth
    node curve without linear sampling's velocity kink at every node."""
    if len(times) == 1:
        return np.array(values[0], FLOAT)
    i = int(np.clip(np.searchsorted(times, t, side='right') - 1, 0, len(times) - 2))
    h = times[i + 1] - times[i]
    s = min(max((t - times[i]) / h, 0.), 1.)
    a, b = values[i], values[i + 1]
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * a + h10 * h * slopes[i] + h01 * b + h11 * h * slopes[i + 1]


WRIST_DAMP_TAU = .06             # s; yaw/roll only. A wrist path's knots are pchip (C1), so the attitude
WRIST_DAMP_SUB = 8               # channels step their acceleration at every anchor (measured on dl35: yaw
WRIST_DAMP_MAX = 40000           # p95 612, roll 341 deg/s^2, while x/y/z sit at 2-3). The corner reads as the


def _gauss(x, step, tau):
    """Zero-phase Gaussian of a uniform series; the ends are extrapolated linearly, not copied (copying
    flat-lines a ramp)."""
    r = max(1, int(round(tau / step * 2.)))
    k = np.arange(-r, r + 1) * step
    w = np.exp(-.5 * (k / tau) ** 2)
    w /= w.sum()
    pad = np.concatenate([x[0] + (x[0] - x[1]) * np.arange(r, 0, -1), x,
                          x[-1] + (x[-1] - x[-2]) * np.arange(1, r + 1)])
    return np.convolve(pad, w, mode='valid')


def _dense_hermite(times, values, slopes, fine):
    """hermite_eval over a whole grid at once."""
    i = np.clip(np.searchsorted(times, fine, side='right') - 1, 0, len(times) - 2)
    h = (times[i + 1] - times[i])[:, None]
    s = np.clip(((fine - times[i]) / h[:, 0])[:, None], 0., 1.)
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    return h00 * values[i] + h10 * h * slopes[i] + h01 * values[i + 1] + h11 * h * slopes[i + 1]


def _attitude_table(times, values, slopes):
    """(fine times, damped yaw/roll columns) or (None, None) when the path is too short to smooth. The
    position columns are NOT touched: they keep the knots' own curve, resting segments included."""
    if len(times) < 3 or len(times) > WRIST_DAMP_MAX:
        return None, None
    step = float(np.median(np.diff(times)))
    fine = np.arange(times[0], times[-1] + step / WRIST_DAMP_SUB * .5, step / WRIST_DAMP_SUB)
    if len(fine) < len(times) * 2:
        return None, None
    dense = _dense_hermite(times, values, slopes, fine)
    h = step / WRIST_DAMP_SUB
    return fine, np.stack([_gauss(dense[:, c], h, WRIST_DAMP_TAU) for c in (3, 4)], 1)


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
        self.att_times, self.att_values = _attitude_table(self.times, self.values, self.slopes)
        self.accents = np.zeros((0, 3))          # (start, onset, depth): the wrist sinks into loud chords
        self.pitch_rise = None                  # (times, 0..1): the palm's pendulum through each free window

    def set_attitude(self, times, yaw, roll):
        """Yaw and roll from their own table (attitude.settle: the palm turned onto the forearm the elbows solve
        placed), read by ``__call__`` through np.interp; the knots' position columns and their curve stand."""
        self.att_times = np.asarray(times, float)
        self.att_values = np.stack([np.asarray(yaw, float), np.asarray(roll, float)], 1)

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
        if self.att_times is not None:          # yaw/roll from their own damped table; x/y/z above untouched
            out[3] = float(np.interp(t, self.att_times, self.att_values[:, 0]))
            out[4] = float(np.interp(t, self.att_times, self.att_values[:, 1]))
        return out


def _vector(anchor):
    return np.r_[anchor.pose.wrist, anchor.pose.yaw, anchor.pose.roll]


def _arc(va, vb, free):
    d = float(np.linalg.norm(vb[:2] - va[:2]))
    return min(ARC_MAX, ARC_RATIO * d) if free and d > ARC_MIN_DISTANCE else 0.


# --- the closing gesture (the hands leaving the keys after the last note) -------------------------------
# The user's spec (2026-11-18): one continuous, quicker-than-the-intro return. The wrist first lifts 2-3 cm
# back-up off the keys and immediately lets go -- the palm droops 20-25 deg while all five fingers curl in
# 30-40 deg to clear the white-key fronts. The upper arm then withdraws, the hand rides one smooth arc back
# over the lap (the elbow flares a hair past the waist), and the last 3-5 cm decelerate hard. The fingertips
# or the palm's front edge brush the thigh first, slide 1-2 cm along it, and the wrist settles flat.
# Two hands close TOGETHER when their last notes are within 1.5 s (the earlier hand waits in a very slow
# 1-2 cm rise); a wider gap closes each hand on its own. Nearest to mirror is never exact: a 0.05-0.08 s
# start difference and a few percent in arc / landing keep the pair from looking mechanical.
OUTRO_LIFT = .026              # wrist rises this far back-up off the last chord before travelling back (m)
OUTRO_CLEAR = .026             # ... and comes this far behind its lap rest while still high (m)
OUTRO_APEX = .036              # the crest of the return, above the last chord (m, over the end pose)
OUTRO_T = 3.80                 # the closing gesture itself, from lift-off to the hand at rest on the lap.
                               # 3.80 s = the intro's leave->land (measured from the same wrist path): the return
                               # is the intro played backwards, so it must take the same time. 2.20 s read as a
                               # hurried snap (the user's "too fast"). Only the outro knots use this constant.
                               # Longer than the old 1.35 s on purpose: the return covers 30-55 cm (the right
                               # hand comes back from the far end of the keyboard), and at 1.35 s it had to
                               # peak near 0.9 m/s and then brake hard -- read as a whip next to the opener.
OUTRO_HOLD = .26               # a still beat at the very end (the piece is over; the hands have arrived)
OUTRO_WAIT = 1.5               # last notes closer than this: the earlier hand waits, then both close
OUTRO_WAIT_LIFT = .016         # the waiting hand's wrist creeps up this far (m) instead of freezing on the keys
OUTRO_START_GAP = .06          # the two hands' starts differ this much (s) even when they close together
OUTRO_LAND_SLIDE = .012        # after the first touch the wrist slides this far along the thigh (m)
OUTRO_TUCK_SECONDS = .40       # fingers curl into the gathered shape over the first beat of the (3.8 s) return;
                               # scaled with OUTRO_T from .22 s so the gather does not read as a snap
OUTRO_UNFOLD_SECONDS = .80     # ... and open again over the last stretch onto the thigh (likewise scaled)
OUTRO_UNFOLD_CLEAR = .025      # the fingers are fully open by the time the wrist is this far above the lap (m)


def outro_path(v_from, rest, t0, side=1., lift=OUTRO_LIFT, apex=OUTRO_APEX, land=None, crest=None):
    """One continuous wrist track from the last playing hold ``v_from`` back to the lap ``rest``.

    Space: a C2 spline through lifted / pulled-back / descending / brush waypoints, sampled by arc length.
    Time: one monotone speed field over [t0, t0+OUTRO_T] shaped like the opener's -- a short gather, one
    broad crest, then a long decay -- so the hand is already slow 3-5 cm above the thigh, touches, slides
    a centimetre and stops, without ever coming to rest in the middle of the return.

    ``crest`` (0..1, the fraction of the time the speed peaks at) defaults to a fraction of where the apex
    falls along the path: a hand already over the lap reaches it a quarter of the way in, and a crest fixed
    at the usual .27 of the time lands in the descent, so the hand accelerates into the drop.
    """
    from scipy.interpolate import CubicSpline
    a = np.asarray(v_from, dtype=FLOAT).copy()
    r = np.asarray(rest, dtype=FLOAT).copy()
    land = r if land is None else np.asarray(land, dtype=FLOAT).copy()
    # A hand that is already over the lap needs no long clear corridor: pulling it forward only to bring it
    # back reads as a detour (the left hand's last chord sits 3 cm from its lap rest, the right's 37 cm
    # away). Scale the corridor by how far the hand actually has to travel.
    reach = float(np.linalg.norm(land - a))
    clear = OUTRO_CLEAR * float(np.clip(reach / .30, .45, 1.))
    # 1. Straight up-and-back off the keys first: the palm is about to droop, the fingertips must clear
    #    the key fronts before the hand travels (the same high corridor the intro uses in reverse).
    p_up = a.copy()
    p_up[0] += side * .002
    p_up[1] -= .020
    p_up[2] += lift
    # 2. Pull behind the lap rest while still HIGH, so the descent happens behind the key fronts' overhang.
    p_back = land.copy()
    p_back[0] += side * clear * .5
    p_back[1] -= clear
    p_back[2] = max(a[2], land[2]) + apex
    # 3. Descend in that clear corridor to just above the lap ...
    p_down = land.copy()
    p_down[0] += side * clear * .35
    p_down[1] -= clear * .8
    p_down[2] = land[2] + .012
    # 4. ... touch a centimetre or so toward the knee, then slide back along the thigh onto the settled
    #    place. The brush is the first contact; the last leg is a slow slide, not another descent.
    p_brush = land.copy()
    p_brush[1] -= OUTRO_LAND_SLIDE
    p_brush[2] = land[2] + .007
    pts = [a, p_up, p_back, p_down, p_brush, land]
    points = np.asarray(pts, dtype=float)
    # Round the interior corners (a spline through sharp waypoints trades a stop for a steering jerk).
    rounded = [points[0]]
    for i in range(1, len(points) - 1):
        p = points[i]
        a0, b0 = points[i - 1, :3] - p[:3], points[i + 1, :3] - p[:3]
        la, lb = np.linalg.norm(a0), np.linalg.norm(b0)
        rad = min(.020, .45 * la, .45 * lb)
        aa, bb = p.copy(), p.copy()
        aa[:3] += a0 * (rad / la)
        bb[:3] += b0 * (rad / lb)
        rounded.append(aa)
        for q in np.linspace(0., 1., 9)[1:-1]:
            rounded.append((1. - q) ** 2 * aa + 2. * (1. - q) * q * p + q * q * bb)
        rounded.append(bb)
    rounded.append(points[-1])
    points_r = np.asarray(rounded)
    chord = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points_r[:, :3], axis=0), axis=1))]
    chord /= chord[-1]
    spatial = CubicSpline(chord, points_r, axis=0, bc_type='natural')
    u = np.linspace(0., 1., max(401, len(points_r) * 32))
    dense = spatial(u)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense[:, :3], axis=0), axis=1))]
    arc /= arc[-1]
    # ONE speed field, shaped like the opener's: a short gather off the keys, one broad crest, then a
    # long decay -- the last 15% of the path takes the last third of the time, so the hand arrives at
    # the thigh slowly instead of braking out of a whip. A beta speed law gives that in one piece; the
    # old five rows accelerated, eased, then accelerated AGAIN into the descent (a visible hump before
    # the brake), which is what read worse than the intro's curve.
    #
    # The crest's PLACE matters as much as its height: it must sit before the apex, while the hand is
    # still travelling. A hand already over the lap (the left's last chord is 3 cm from its lap rest)
    # reaches the apex a quarter of the way along the path, so a crest fixed at .27 of the time lands in
    # the descent and the hand accelerates into the drop -- it read as a slap, while the right's, whose
    # apex is .64 of the path, never did. Put the crest at a fraction of the apex's own place instead.
    if crest is None:
        crest = float(np.clip(.55 * float(arc[int(np.argmax(dense[:, 2]))]), .12, .30))
    b = 1.6
    a_exp = b * crest / max(1e-6, 1. - crest)
    t1 = t0 + OUTRO_T
    t_hold = t1 - OUTRO_HOLD
    f = np.linspace(0., 1., 41)
    speed = np.maximum(f, 1e-9) ** a_exp * np.maximum(1. - f, 1e-9) ** b
    clock_s = np.r_[0., np.cumsum((speed[1:] + speed[:-1]) * .5)]
    clock_s = (clock_s / clock_s[-1])[:, None]
    clock_t = t0 + f * (t_hold - t0)
    clock_m = pchip_slopes(clock_t, clock_s)

    def progress(t):
        return float(np.clip(hermite_eval(clock_t, clock_s, clock_m, t)[0], 0., 1.))

    count = max(60, int(np.ceil((t1 - t0) * 60.)))
    sample_t = np.linspace(t0, t1, count + 1)
    sample_s = np.maximum.accumulate([progress(t) for t in sample_t])
    return [(float(t), spatial(float(np.interp(s, arc, u))), 0.) for t, s in zip(sample_t[1:], sample_s[1:])]


def lap_arrival(path, rest, clear=OUTRO_UNFOLD_CLEAR):
    """When the closing wrist first comes within ``clear`` of the lap's height (``rest`` is its landing vector).

    The return curve touches the thigh a centimetre or two above the settled place -- the last stretch is a
    slow slide along the leg, not another descent -- so the gesture's nominal end (``trip_outro_land``) is
    well after the hand is already on the leg. The fingers must be open by the CONTACT, not by the end:
    arriving still tucked drove the curled fingertips into the thigh (the lap pose is fitted with the
    relaxed curl, so the tuck's extra 40 deg of middle joint points straight down into it).
    """
    t0 = getattr(path, 'trip_outro', None)
    t1 = getattr(path, 'trip_outro_land', None)
    if t0 is None or t1 is None or t1 <= t0:
        return None
    rest = np.asarray(rest, dtype=float)
    ts = np.linspace(t0, t1, 241)
    z = np.array([path(t)[2] for t in ts], dtype=float)
    # Scan only after the return's highest point so a low keyboard / high thigh cannot match at t0.
    apex = int(np.argmax(z))
    below = np.nonzero(z[apex:] <= rest[2] + clear)[0]
    return float(ts[apex + below[0]]) if len(below) else None


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
    on_way = np.zeros(n, bool)                   # samples moving on toward the next chord while holding keys
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
        # past the settle the hold's family may take over only where the next chord holds these keys
        # too (the hand sets off while it keeps them); otherwise the hand stays at its place while its
        # keys are down -- the planner's travel model departs no sooner than the releases, and a wide
        # family box (a stretched chord's) lets the wrist slide clean off its own place inside a short
        # press, where the fingers cannot follow (the dragged thumb of the daylight piece's frame 2395)
        down_ids = [q for q in keys if notes[q].touch <= t + 1e-9 and t <= notes[q].release + 1e-9]
        carried = i + 1 < len(rows) and all(q in rows[i + 1][3] for q in down_ids)
        # A hand holding an octave or more is at the edge of its span: it has no reach to spare, so the few
        # millimetres the path saves by sinking while the keys are down cost it the keys. Canon's left hand
        # holding C2+C3 sinks 6 mm during the hold and the pads come back 16-33 mm off their keys, while the
        # same 6 mm on a narrower hold is harmless -- the anchor's own pose is fine, the fingers simply have
        # nothing left to absorb it. Such a hold keeps the full pin even when the next chord carries its keys.
        wide = max(notes[q].pitch for q in keys) - min(notes[q].pitch for q in keys) >= 12
        if not carried or wide:
            # the family box would let the wrist slide off its place while its keys are still down and the
            # pads cannot follow (a stretched chord's box is 27+ mm wide: the dragged thumb of the daylight
            # piece's frame 4783). Pinning the whole palm can cost the fingers' IK some reach on a short
            # press the hand departs from mid-key (R 66 @41.4, ~20 mm, frame 2482) -- tried pinning the
            # side alone, but the free axes slide on (L 70.1's 13 mm came back): keep the full pin.
            lb[k], ub[k] = v, v
            target[k] = v
            continue
        places = v[None] if a.family is None else a.family[0]
        if i + 1 < len(rows) and all(q in rows[i + 1][3] for q in keys
                                     if notes[q].touch <= t + 1e-9 and t <= notes[q].release + 1e-9):
            b, vb = rows[i + 1][0], rows[i + 1][1]
            places = np.concatenate([places, vb[None] if b.family is None else b.family[0]])
            # the box over both families has corners that are no place (one place's side, another's depth): pulled
            # along the way between the two chosen places, which both hold the keys
            end = max(a.time, start + SETTLE_SECONDS)
            share = float(np.clip((t - end) / max(rows[i + 1][2] - end, PATH_STEP), 0., 1.))
            target[k] = v + share * (vb - v)
            on_way[k] = True
        else:
            target[k] = v                # held where it is: pulled to its place; on the way on: free between the two
        lb[k], ub[k] = places.min(axis=0), places.max(axis=0)
    # free windows: between an anchor's last release and the next anchor's first touch
    rise = np.zeros(len(times))       # 0 at the release, 1 at the top of the lift, 0 at the next touch
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
            if window > 1e-3:
                # The palm's own pendulum: it turns up through the rise and gives the degrees back on the
                # descent, one unhurried up-and-back per free window -- not a vibration.
                rise[sel] = np.maximum(rise[sel], np.sin(np.pi * (times[sel] - free) / window))
    if os.environ.get('PIANOMOTION_DEBUG_SEGMENTS'):       # diagnosis only: the anchors' places and pair windows
        win = float(os.environ.get('PIANOMOTION_DEBUG_AT') or 0.)
        sel_dbg = (times > win - .4) & (times < win + .4)
        dbg = dict(rows=[dict(time=float(r[0].time), x=float(r[1][0])) for r in rows if abs(r[0].time - win) < 4.],
                   pairs=[dict(a_t=float(a.time), va_x=float(va[0]), b_t=float(b.time), vb_x=float(vb[0]))
                          for (a, va, _, ka), (b, vb, sb, kb) in zip(rows, rows[1:]) if abs(a.time - win) < 4.],
                   bounds=[[float(t), float(lb[k, 0]), float(ub[k, 0]), int(holding[k])]
                           for k, t in enumerate(times) if sel_dbg[k]])
        with open(os.environ['PIANOMOTION_DEBUG_SEGMENTS'], 'a') as f:
            json.dump(dbg, f)
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
    # ends pinned to the first / last pose. The LAST is the last PLAYING pose, not rows[-1]: the refiner
    # and the idle attitude leave stay-put anchors in the silence after the music (a lap or forearm pose
    # minutes ahead), and pinning the sampled horizon to one of those dragged the last held note down by
    # centimetres before its release -- the hand sank while the key was still down. The closing gesture
    # takes over after this path is truncated at the final release.
    first = rows[0][1]
    last = next((r[1] for r in reversed(rows) if r[0].incoming or r[0].held), rows[-1][1])
    out = np.empty((n, 5))
    d3 = sparse.diags([-1., 3., -3., 1.], [0, 1, 2, 3], shape=(n - 3, n)) / PATH_STEP ** 3
    d2 = sparse.diags([1., -2., 1.], [0, 1, 2], shape=(n - 2, n)) / PATH_STEP ** 2
    d1 = sparse.diags([-1., 1.], [0, 1], shape=(n - 1, n)) / PATH_STEP
    for d in range(5):
        has = ~np.isnan(target[:, d])
        # on the way between two places the depth is held to the way as firmly as the side: a depth from one place
        # with the side and turn of another holds no key
        scale = np.where(on_way, min(PULL_SCALE[d], PULL_SCALE[0]) if d == 1 else PULL_SCALE[d], PULL_SCALE[d])
        pull = sparse.diags(has.astype(float) / scale, 0, shape=(n, n)).tocsr()[has]
        rows_l = [(k, sc[d]) for k, sc in land]
        if rows_l:
            r = np.arange(len(rows_l)); ks = np.array([k for k, _ in rows_l]); sc = np.array([c for _, c in rows_l])
            vel = sparse.csr_matrix((np.r_[-1. / (PATH_STEP * sc), 1. / (PATH_STEP * sc)], (np.r_[r, r], np.r_[ks, ks + 1])),
                                    shape=(len(rows_l), n))
        else:
            vel = sparse.csr_matrix((0, n))
        A = sparse.vstack([d3 / JERK_SCALE[d], d2 / ACC_SCALE[d], d1 / VEL_SCALE[d], pull, vel]).tocsr()
        b = np.r_[np.zeros(n - 3), np.zeros(n - 2), np.zeros(n - 1), (target[has, d] / scale[has]), np.zeros(vel.shape[0])]
        lo, hi = lb[:, d].copy(), ub[:, d].copy()
        lo[0] = hi[0] = first[d]; lo[-1] = hi[-1] = last[d]
        hi = np.maximum(hi, lo + 1e-9)
        out[:, d] = chunked(A, b, lo, hi)
    return times, out, rise


CHUNK_SECONDS, CHUNK_KEEP = 6., 4.    # solved exactly in overlapping windows, each fixed to the last at its start


def bounded_lsq(A, b, lo, hi, iterations=None):
    """min |A x - b|^2 with lo <= x <= hi, exactly: a primal active set (one bound freed or met at a time, like bvls)
    over the normal equations, which are banded (the path's difference rows span a few samples) and solved banded,
    O(n) per step where bvls refactors the dense A. The optimum is unique (the jerk/acceleration/speed rows), so this
    is bvls's solution to rounding."""
    from scipy.linalg import cho_solve_banded, cholesky_banded
    A = np.asarray(A, float)
    n = A.shape[1]
    H, c = A.T @ A, A.T @ b
    nz = np.nonzero(np.abs(H) > 0)
    u = int(np.max(np.abs(nz[0] - nz[1]))) if len(nz[0]) else 0
    fixed = hi - lo <= 1e-12
    x = np.where(fixed, lo, np.clip(np.zeros(n), lo, hi))
    state = np.where(fixed, 3, np.where(x <= lo, 1, np.where(x >= hi, 2, 0)))      # 0 free, 1 at lo, 2 at hi, 3 fixed
    scale = max(1., float(np.abs(c).max()), float(np.abs(H).max()))
    tol = 1e-12 * scale

    def solve_free(free):
        idx = np.nonzero(free)[0]
        Hf = H[np.ix_(idx, idx)]
        rhs = c[idx] - H[np.ix_(idx, ~free)] @ x[~free]
        k = min(u, len(idx) - 1)
        ab = np.zeros((k + 1, len(idx)))
        for j in range(k + 1):
            ab[k - j, j:] = np.diagonal(Hf, j)
        return idx, cho_solve_banded((cholesky_banded(ab), False), rhs)

    for _ in range(iterations or 20 * n + 50):
        free = state == 0
        if free.any():
            idx, z = solve_free(free)
            low, high = z < lo[idx], z > hi[idx]
            if low.any() or high.any():
                # walk from x toward z until the first bound
                step = z - x[idx]
                with np.errstate(divide='ignore', invalid='ignore'):
                    t = np.where(low, (lo[idx] - x[idx]) / step, np.where(high, (hi[idx] - x[idx]) / step, np.inf))
                a = float(np.clip(t.min(), 0., 1.))
                x[idx] += a * step
                hit = (t <= a + 1e-15) & (low | high)
                x[idx[hit & low]] = lo[idx[hit & low]]
                x[idx[hit & high]] = hi[idx[hit & high]]
                state[idx[hit & low]] = 1
                state[idx[hit & high]] = 2
                continue
            x[idx] = z
        g = H @ x - c
        # a bound held with the gradient pulling inward is let go: the worst one
        wrong = np.where(state == 1, -g, np.where(state == 2, g, -np.inf))
        j = int(np.argmax(wrong))
        if wrong[j] <= tol:
            return x
        state[j] = 0
    from scipy.optimize import lsq_linear
    return lsq_linear(A, b, bounds=(lo, np.maximum(hi, lo + 1e-12)), method='bvls').x


def chunked(A, b, lo, hi, x0=None, dirty=None):
    """Bounded least squares, exactly (bounded_lsq; bvls stopped on its cost tolerance far from the optimum in some
    windows) over overlapping windows of samples; the first three samples of a window are fixed to the previous
    window's solution (position, speed, acceleration carry over). ``x0`` (an earlier solution) with ``dirty`` (the
    sample span whose bounds moved since) answers the same problem again: the windows that end before the span keep
    their samples, and past it the solve stops as soon as a window's solved start reproduces ``x0``'s -- from there
    on the problem is the one ``x0`` already solved, window for window."""
    A = A.tocsr()
    A.sort_indices()                    # the row's own first and last column: the row's span
    n = A.shape[1]
    first = np.zeros(A.shape[0], int)
    last = np.zeros(A.shape[0], int)
    nz = np.diff(A.indptr) > 0
    if nz.any():
        first[nz] = A.indices[A.indptr[:-1][nz]]
        last[nz] = A.indices[A.indptr[1:][nz] - 1]
    size, keep = int(CHUNK_SECONDS / PATH_STEP), int(CHUNK_KEEP / PATH_STEP)
    x = np.zeros(n) if x0 is None else np.array(x0, float)
    warm = x0 is not None and dirty is not None
    i0 = 0
    while True:
        i1 = min(n, i0 + size)
        stop = i1 >= n
        if warm and i1 < dirty[0]:          # before the change: the window's kept samples are x0's own
            if stop:
                return x
            i0 += keep - 3 if keep > 3 else keep
            continue
        rows = np.nonzero((first >= i0) & (last < i1))[0]
        lo_w, hi_w = lo[i0:i1].copy(), hi[i0:i1].copy()
        if i0 > 0:
            lo_w[:3] = hi_w[:3] = x[i0:i0 + 3]
        sol = bounded_lsq(A[rows][:, i0:i1].toarray(), b[rows], lo_w, np.maximum(hi_w, lo_w + 1e-12))
        if warm and not stop and i0 > dirty[1] and np.abs(sol[:3] - x0[i0:i0 + 3]).max() < 1e-9:
            return x                    # the carry-over matches the earlier answer: the rest is that answer too
        x[i0:i1 if stop else i0 + keep] = sol[:(i1 - i0) if stop else keep]
        if stop:
            return x
        i0 += keep - 3 if keep > 3 else keep


def wrist_path(anchors, notes, strokes, rest, duration, t_start=0., leave_override=None, land_override=None,
               quiet=False, outro_start=None, outro_variant=0.):
    """``anchors`` of one hand (with poses and their depth families, pipeline.depth_family; without: held in place).
    ``t_start``: the video's first moment in this path's own time. ``leave_override`` / ``land_override`` put the
    opening hand(s) on the shared intro (one breath, a quick lift, a short pause, a slow reach and a slow press);
    a hand that enters later gets its OWN window ending on its own first note (pipeline.hand_overrides), and
    ``quiet`` trades the ceremony for one calm arc off the lap to its keys. ``outro_start`` is the shared closing
    gesture's start (pipeline closings: the earlier hand may wait in a slow micro-rise); ``outro_variant`` breaks
    the mirror (-1/0/+1: a few percent in arc and landing). Every leg is sampled with an ease so the wrist
    accelerates and decelerates instead of moving at one speed."""
    anchors = sorted(anchors, key=lambda a: a.time)
    if not anchors or not notes:
        # A hand with no notes at all never plays this clip: it stays on the lap the whole time. A tail cut
        # can leave one hand silent with a stray refiner placeholder anchor; building a closing gesture from
        # that would lift a resting hand off the lap and sweep it past the keys at the end. No notes -> no
        # intro, no outro, just the rest pose (and no trip_outro / trip_leave markers for the pipeline).
        return WristPath([(0., rest, 0.), (duration, rest, 0.)])

    def ready(a):
        return min((strokes[k].hover for k in a.incoming), default=a.time)

    first, v0 = anchors[0], _vector(anchors[0])
    settle = ready(first)
    # the piece's own pace right after the hands land sets the hover's playback rate: the busiest stretch
    # the player must join. A fast follow-up (short gaps) shrinks the soothing drift, a slow opening lets
    # it breathe -- the intro hands over to the playing at the tempo the playing itself sets.
    gaps = [b.time - a.time for a, b in zip(anchors[:4], anchors[1:4]) if b.time - a.time > 1e-3]
    tempo = max(.55, min(1.35, (min(gaps) if gaps else .60) / .55))
    trip = REST_LIFT_T + REST_REACH_T + REST_POISE_T + REST_SEAT_T
    t_note = min((ready(a) for a in anchors if a.incoming), default=settle)
    leave = leave_override if leave_override is not None else max(t_note - trip, t_start + .15)
    land = land_override if land_override is not None else t_note
    if land <= leave + 1e-6:
        land = leave + trip
    # The hand's playing ends at its last anchor that actually holds / strikes a key. The wrist refiner may add
    # empty "stay put" anchors in the silence after the last note (they carry no incoming / held keys and sit up
    # against the sampling horizon); using one of those as the closing gesture's start would push the return past
    # the sampled domain and strand the hand above the keys (the v10 truncation bug). Ignore them: the key's own
    # release still sets the lower bound through the notes below.
    playing = [a for a in anchors if a.incoming or a.held]
    last = playing[-1] if playing else anchors[-1]
    end = max([last.time] + [n.release for n in notes.values()]) + KEY_RETURN_SECONDS
    times, values, rise = solve_path(anchors, notes, strokes, settle, end)
    if os.environ.get('PIANOMOTION_DEBUG_SEGMENTS'):       # diagnosis only: the solved grid around the window
        win = float(os.environ.get('PIANOMOTION_DEBUG_AT') or 0.)
        with open(os.environ['PIANOMOTION_DEBUG_SEGMENTS'], 'a') as f:
            json.dump(dict(grid=[[float(t), float(v[0])] for t, v in zip(times, values)
                                 if abs(t - win) < 0.7][::2]), f)
    scale = min(1., (land - leave) / trip)
    draw_t = REST_DRAW_T * scale
    poke_t = REST_POKE_T * scale
    lift_t = draw_t + poke_t
    seat_t = REST_SEAT_T * scale
    # The fall is ONE motion: from the poke's top the hand sails out to the keys while the palm falls from
    # its tilt toward the playing plane -- the droop and the travel are the same gesture (the user's spec:
    # "手腕卸力去让手掌朝放平的方向在空中下落，去找琴键"). The slack after it becomes a still POISE, and
    # the seat keeps ending exactly on the first note.
    if quiet:
        # the late hand's quiet approach (the user's pick: a simplified, hushed move): no flat draw, no
        # chest-high poke, no tilt beat -- ONE calm eased arc off the lap to its keys, a still beat above
        # them, then the same gentle seat. The palm hangs while the hand is low, flattens mid-arc.
        seat_start = land - seat_t
        reach_end = max(seat_start - REST_QUIET_POISE_T, leave + .9)
        tilt_end = draw_end = lift_end = leave + (reach_end - leave) * REST_QUIET_FLAT_AT
    else:
        draw_end = leave + draw_t
        lift_end = leave + lift_t
        seat_start = land - seat_t
        slack = max(seat_start - lift_end, .25)
        poise_t = min(.9 * scale, max(.45 * scale, slack * .30))
        fall_t = slack - poise_t
        tilt_end = lift_end + fall_t * .28       # the tension peaks early in the fall; then the palm falls out
        reach_end = lift_end + fall_t
    side = 1. if anchors[0].hand == 'R' else -1.
    # The flat draw: the upper arm carries the whole hand BACK toward the body, the fingertips sliding along
    # the lap surface (the wrist barely rises -- the thigh simply gets thicker toward the hip); the elbow
    # swings back with it and splays outward a hair, leaving room to adduct when the forearm pokes up.
    p_draw = rest.copy()
    p_draw[0] += side * REST_DRAW_OUT
    p_draw[1] -= REST_DRAW_BACK
    p_draw[2] += REST_DRAW_UP
    # The hand is carried off the lap and rises along ONE ARC toward the keys: the draw ends behind the
    # rest, the lift curves gently forward as it rises (the elbow drifts back-then-forward a few mm, never
    # swings), and the poke tops out just behind the lap rest -- the forward travel belongs to the hover's
    # creep and the reach after it, which keeps the last stretch to the first note a clear ~6 cm.
    p_lift = rest.copy()
    p_lift[0] += side * REST_DRAW_OUT * .4
    p_lift[1] -= REST_DRAW_BACK * .5
    p_lift[2] = max(REST_LIFT_MID_Z, p_draw[2] + .035)
    # The wrist the playing solve actually HOLDS once this hand's intro is over: its depth family settles
    # where the first figure wants it, up to a couple of centimetres off the anchor's raw pose. Seating the
    # hand on anything else means a one-frame yank at the handover (the hand arrives, then jumps to where
    # the strokes want it) -- the seat's target IS the solve's own first value, so the handover is seamless.
    hold = np.array([np.interp(max(settle, land), times, values[:, d]) for d in range(values.shape[1])],
                    dtype=FLOAT)
    p_poke = rest.copy()
    p_poke[0] += (hold[0] - rest[0]) * REST_HOVER_OUT
    p_poke[1] -= REST_POKE_FWD
    p_poke[2] = max(REST_HOVER_HEIGHT, p_lift[2] + .045)
    p_reach = hold.copy()
    p_reach[:3] += np.asarray(REST_REACH_FROM_KEY, dtype=FLOAT)
    # The fall is ONE shallow sail: the hand keeps drifting forward the whole way (linear share, eased by
    # the traversal -- monotone, no overshoot, no stall) while the height gently crests above the chord and
    # descends to the poise: the world path curves like a sail. SAMPLED (9 interior points, the crest
    # falling between them): one apex knot would make the pchip flatten its slope exactly there (a data
    # extremum) -- the sail stops dead on the crest and lurches down after it.
    p_fall_pts = [p_poke]
    for x in np.linspace(0., 1., 11)[1:-1]:
        pt = p_poke.copy()
        pt[0] += (p_reach[0] - p_poke[0]) * x
        pt[1] += (p_reach[1] - p_poke[1]) * x
        pt[2] += (p_reach[2] - p_poke[2]) * x + REST_FALL_CREST * np.sin(np.pi * x)
        p_fall_pts.append(pt)
    p_fall_pts.append(p_reach)
    p_press = hold.copy()

    def continuous_intro(pts, phases):
        """One rounded spatial curve and one monotone speed law through the whole intro.

        ``phases`` preserves the draw / poke / quiet-apex beats WITHOUT making them zero-speed knots.
        The first corners are replaced by small quadratic rounds before the C2 spline: interpolation
        through the raw sharp waypoints would merely trade the old stops for a steering jerk.
        """
        from scipy.interpolate import CubicSpline
        points = np.asarray(pts, dtype=float)
        rounded, skip = [points[0]], set()
        for i in range(1, len(points) - 1):
            if i in skip:
                continue
            p = points[i]
            if i <= 3:                       # draw/lift/poke (or the quiet pickup's three corners)
                j = min(i + 5, len(points) - 2) if i == 3 else i + 1
                skip.update(range(i + 1, j))  # the apex round spans several dense sail samples
                a0, b0 = points[i - 1, :3] - p[:3], points[j, :3] - p[:3]
                la, lb = np.linalg.norm(a0), np.linalg.norm(b0)
                if la < 1e-6 or lb < 1e-6:
                    rounded.append(p)
                    continue
                r = min(.020, .45 * la, .45 * lb)
                a, b = p.copy(), p.copy()
                a[:3] += a0 * (r / la)
                b[:3] += b0 * (r / lb)
                rounded.append(a)
                for q in np.linspace(0., 1., 13)[1:-1]:
                    rounded.append((1. - q) ** 2 * a + 2. * (1. - q) * q * p + q * q * b)
                rounded.append(b)
            else:
                rounded.append(p)
        rounded.append(points[-1])
        dedup = [rounded[0]]
        for pt in rounded[1:]:
            if np.linalg.norm(pt[:3] - dedup[-1][:3]) > 1e-6:
                dedup.append(pt)
        points_r = np.asarray(dedup)
        chord = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points_r[:, :3], axis=0), axis=1))]
        chord /= chord[-1]
        spatial = CubicSpline(chord, points_r, axis=0, bc_type='natural')
        u = np.linspace(0., 1., max(401, len(points_r) * 32))
        dense = spatial(u)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense[:, :3], axis=0), axis=1))]
        path_length = float(arc[-1])
        arc /= path_length

        def arc_near(p):
            j = int(np.argmin(np.linalg.norm(dense[:, :3] - p[:3], axis=1)))
            return float(arc[j])

        reach_arc = arc_near(points[-2])       # p_reach; the last point is the playing hold
        poise_arc = reach_arc + .35 * (1. - reach_arc)
        # The beats pin TIME-POSITION only; the speed law is pchip's monotone slope field over those rows.
        # Prescribing a slow speed AT a beat (the first cut) forced the interval after it to bulge far above
        # its chord slope to make up the distance -- a whip out of every turn. Pchip never exceeds the
        # neighbouring chord slopes, so the hand eases through each rounded corner near the interval's
        # natural pace instead of braking and lashing back out.
        clock = [(leave, 0.)]
        for p, t in phases:                   # a beat rides the curve; it is not a stop
            s = arc_near(p)
            if clock[-1][0] < t < reach_end and clock[-1][1] < s < reach_arc:
                clock.append((t, s))
        clock += [(reach_end, reach_arc), (seat_start, poise_arc), (land, 1.)]
        clock_t = np.asarray([r[0] for r in clock])
        clock_s = np.asarray([r[1] for r in clock])
        clock_m = pchip_slopes(clock_t, clock_s[:, None])

        def progress(t):
            return float(np.clip(hermite_eval(clock_t, clock_s[:, None], clock_m, t)[0], 0., 1.))

        def at(f):
            return spatial(float(np.interp(f, arc, u)))

        count = max(48, int(np.ceil((land - leave) * 60.)))
        sample_t = np.linspace(leave, land, count + 1)
        sample_s = np.maximum.accumulate([progress(t) for t in sample_t])
        return [(float(t), at(s), 0.) for t, s in zip(sample_t[1:], sample_s[1:])]

    knots = [(min(0., leave - 1e-3), rest, 0.), (leave, rest, 0.)]
    if quiet:
        # ONE calm curve: clearance pull, climb, high sail and the tiny seat all share the same spatial
        # spline and clock. The palm still flattens by the apex; there is no draw dwell or tilt beat.
        p_off = rest.copy()
        p_off[0] += side * REST_DRAW_OUT * .4
        p_off[1] -= REST_QUIET_CLEAR
        p_off[2] += .018
        p_up = rest.copy()
        p_up[0] += side * REST_DRAW_OUT * .6
        p_up[1] -= REST_QUIET_CLEAR + .004
        p_up[2] = max(REST_LIFT_MID_Z, p_off[2] + .035)
        p_top = rest.copy()
        p_top[0] += side * REST_DRAW_OUT * .8
        p_top[1] -= REST_QUIET_CLEAR + .004
        p_top[2] = max(REST_QUIET_APEX, p_up[2] + .035)
        pts = [rest, p_off, p_up, p_top]
        for x in np.linspace(0., 1., 11)[1:-1]:
            pt = p_top.copy()
            pt[0] += (p_reach[0] - p_top[0]) * x
            pt[1] += (p_reach[1] - p_top[1]) * x
            pt[2] += (p_reach[2] - p_top[2]) * x + REST_FALL_CREST * np.sin(np.pi * x)
            pts.append(pt)
        pts += [p_reach, p_press]
    else:
        # Draw, lift, poke, sail, near-poise and seat are ONE gesture. In particular the poke no longer
        # brakes to zero: its tension is communicated by the palm, while the wrist keeps travelling.
        pts = [rest, p_draw, p_lift] + p_fall_pts + [p_press]
    phases = [(p_top, tilt_end)] if quiet else [(p_draw, draw_end), (p_poke, lift_end)]
    knots += continuous_intro(pts, phases)
    if quiet:
        draw_end = leave                    # the quiet hand starts peeling at lift-off, not at its apex
        lift_end = tilt_end
    if settle > land + 1e-6:
        knots.append((settle, hold, 0.))        # this hand's first stroke comes later: wait on its key
    knots += [(t, v, 0.) for t, v in zip(times, values) if t > land + 1e-6]
    # The knots carry the solved hold: between them the spline cuts corners where the bounds change,
    # and a staccato press (about 40 ms) holds one or two of the PATH_STEP knots -- the smoothed curve
    # then skims off the chord's place while its pads are still down (the dragged thumb of the daylight
    # piece's frame 2395). So every press window's edges become knots, at the path's own solved values:
    # the curve cannot leave a place it is bounded to before the sampled window says so.
    edges = sorted({round(t, 6) for a in anchors for k in list(a.incoming) + list(a.held)
                    for t in (notes[k].touch, notes[k].release)})
    knots += [(t, np.array([np.interp(t, times, values[:, d]) for d in range(values.shape[1])], FLOAT), 0.)
              for t in edges if max(times[0], land) < t < times[-1]]
    knots[-1] = (times[-1], values[-1], 0.)       # the closing gesture owns the tail: no arc bump on it
    # Closing gesture: ONE continuous curve from the last playing hold back to the lap. The solved tail
    # after the final release is NOT trusted: the free solve can drop the wrist several centimetres as
    # the last finger lets go (it has no anchor left), while the closing gesture must begin from the
    # released chord. Cut the solved path at the last release and build the return from that value.
    # When the other hand finishes later (within OUTRO_WAIT), this hand waits in a very slow 1-2 cm rise
    # before joining it -- never a frozen hand lying on the keys. Otherwise it leaves at its own release.
    last_release = max((n.release for n in notes.values()), default=float(times[-1]))
    close_start = float(times[-1] if outro_start is None else outro_start)
    close_start = max(close_start, last_release + .005)
    hold_at = min(close_start, last_release + .001)
    waiting = close_start > last_release + .12
    hold = np.array([np.interp(hold_at, times, values[:, d]) for d in range(values.shape[1])], FLOAT)
    if hold_at < times[-1] - 1e-6:
        knots = [k for k in knots if k[0] <= hold_at + 1e-6] + [(hold_at, hold, 0.)]
    close_from = hold
    if waiting:
        # A shallow, slow lift while the other hand finishes: the wrist keeps living instead of parking.
        wait_to = hold.copy()
        wait_to[1] -= .004
        wait_to[2] += OUTRO_WAIT_LIFT
        knots.append((close_start - .08, wait_to, 0.))
        close_from = wait_to
    elif close_start > hold_at + .005:
        knots.append((close_start, hold, 0.))       # a short beat after the release, then the return
    lift = (OUTRO_LIFT - OUTRO_WAIT_LIFT if waiting else OUTRO_LIFT) * (1. + .04 * outro_variant)
    apex = OUTRO_APEX * (1. + .05 * outro_variant)
    # The two hands' arcs differ (lift / apex above), but they must LAND on the intro's own lap pose:
    # a millimetre of landing offset used to be spread over the idle tail after the gesture; with the
    # longer return that tail is short, and the leftover slid the settled hand 4 mm at the very end.
    land_pos = np.asarray(rest, dtype=FLOAT).copy()
    knots += outro_path(close_from, rest, close_start, side, lift=lift, apex=apex, land=land_pos)
    arrived = close_start + OUTRO_T
    if arrived < duration:
        knots.append((duration, rest, 0.))
    if os.environ.get('PIANOMOTION_DEBUG_KNOTS'):
        print('[knots %s] settle=%.3f first.anchor_t=%.3f n=%d | %s' % (
              anchors[0].hand, settle, first.time, len(anchors),
              ' | '.join('(%.2f: %.3f,%.3f,%.3f)' % (t, v[0], v[1], v[2]) for t, v, a in knots[:8])), flush=True)
    path = WristPath(knots)
    path.pitch_rise = (times, rise)
    path.trip_end = reach_end                 # the high-speed travel ends: the near-poise begins
    path.trip_draw_end = draw_end             # actual passage through the draw's waypoint on the whole curve
    path.trip_lift_end = lift_end             # actual passage through the poke / quiet apex
    path.trip_tilt_end = tilt_end             # quiet: == lift_end, so the tension beat is skipped
    path.trip_reach_end = reach_end           # the convex reach ends: the hand seats into the playing pose
    path.trip_land = land                     # the seat ends, the finger's stroke takes the first note
    path.trip_leave = leave                   # the flat draw's start
    path.trip_outro = close_start             # when the closing gesture begins (pitch + checks)
    path.trip_outro_land = close_start + OUTRO_T - OUTRO_HOLD   # palm + fingers reach the lap
    path.trip_outro_wait = waiting             # this hand waited for the other before closing
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
    clear_slopes: np.ndarray = None

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
        if self.clear_slopes is None:
            self.clear_slopes = pchip_slopes(times, deltas)
        return v + hermite_eval(times, deltas, self.clear_slopes, t)


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
        self.rest = None                # the lap rest's shape (LAP_FINGERS): held until ``rest_start``
        self.rest_start = None          # the hover's end: past it the fingers ease into the first note's shape
        self.rest_back = None           # the last keypose + a beat: after it the fingers ease back onto the lap
        self.rest_end = None            # closing: past this the hand rests on the lap (spacing skipped again)
        self.tuck = None                # LAP_TUCK: the extra curl while transiting under the keybed
        self.rest_tuck = None           # the leave time: the tuck curls in from here until ``rest_start``
        self.outro_end = None           # closing gesture: the fingers are back on the lap by this time
        self.outro_touch = None         # ... and the wrist first touches the thigh this much earlier (lap_arrival)
        self.outro_gathered = None      # closing-only compact curl; never changes the opener's LAP_TUCK
        self.outro_from = None          # frozen lift-off shape: do not chase moving playing keyposes
        self.outro_unfold = None        # measured departure from the key-front envelope

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
        key, sweeping through the keys behind it) and fade out only as the pad reaches its key. They stay
        out of the lap rest entirely (the hand is under the keybed there, nothing to clear) and ease in
        past it."""
        s = self.stroke(t)
        if self.rest_start is not None and t <= self.rest_start:
            return 0.
        if s is None:
            out = 1.
        elif s.touch <= t <= s.release:
            return 0.
        else:
            gap = s.touch - t if t < s.touch else t - s.release
            out = min(1., gap / CORRECTION_FADE)
        if self.rest_start is not None:
            out = min(out, ease((t - self.rest_start) / LAP_FADE_SECONDS))
        if self.rest_back is not None and t > self.rest_back:
            # the playing corrections belong to the keys: fade them out as the fingers gather onto the lap
            out = min(out, 1. - ease(min((t - self.rest_back) / OUTRO_TUCK_SECONDS, 1.)))
        return out

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

    def lap_shape(self, t):
        """The lap rest's shape at ``t``: the relaxed curl, plus the transit tuck once the hand leaves."""
        v = self.rest
        if self.tuck is not None and self.rest_tuck is not None and t > self.rest_tuck:
            w = ease(min((t - self.rest_tuck) / LAP_TUCK_SECONDS, 1.))
            v = v + self.tuck * w
        return v

    def keyposed(self, t):
        values = self._keyposed(t)
        if self.rest is None or not len(self.keyposes):
            return values
        start = self.rest_start
        if start is not None:
            end = self.strokes[0].hover if self.strokes else self.keyposes[0].time
            if t <= start:
                return self.lap_shape(t)
            if t < end:
                w = ease((t - start) / max(end - start, 1e-6))
                # the blend targets the shape HELD at the stroke's approach, frozen: a moving target would
                # drag the keyposes' own transitions (a 30-45 deg spread and back) into the settle, which
                # read as the finger sketching in the air (the user's poise note)
                return self.lap_shape(t) * (1. - w) + self._keyposed(end - 1e-4) * w
        if self.rest_back is not None and t > self.rest_back:
            # On the way back the fingers first curl in (a limp, gathered hand clearing the white-key fronts),
            # then unfold onto the thigh as the wrist slides onto it. A flat finger splayed straight across
            # the key ledge cuts into the keybed on the way down; the tuck is the hand's natural "pulled out
            # from under the piano" shape.
            if self.outro_end is not None:
                tuck_end = self.rest_back + OUTRO_TUCK_SECONDS
                # open onto the thigh by the CONTACT, not by the gesture's nominal end: the last stretch of
                # the return is a slide along the leg, so the hand is already on it a few tenths of a
                # second before ``outro_end``, and a tucked hand arriving there sinks its fingertips in.
                open_by = self.outro_end if self.outro_touch is None else self.outro_touch
                open_by = max(open_by, tuck_end + 1e-3)
                unfold = max(tuck_end, open_by - OUTRO_UNFOLD_SECONDS)
                measured_unfold = getattr(self, 'outro_unfold', None)
                if measured_unfold is not None:
                    unfold = float(np.clip(measured_unfold, tuck_end, open_by - 1e-3))
                gathered = getattr(self, 'outro_gathered', None)
                if gathered is None:
                    gathered = self.rest + self.tuck
                if t < tuck_end:
                    w = ease((t - self.rest_back) / max(tuck_end - self.rest_back, 1e-6))
                    initial = getattr(self, 'outro_from', None)
                    return (values if initial is None else initial) * (1. - w) + gathered * w
                if t < unfold:
                    return gathered
                w = ease(min((t - unfold) / max(open_by - unfold, 1e-6), 1.))
                return gathered * (1. - w) + self.rest * w
            w = ease(min((t - self.rest_back) / LAP_RETURN_SECONDS, 1.))
            return values * (1. - w) + self.rest * w
        return values

    def _keyposed(self, t):
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

    def many(self, queries, depths):
        """``_surface`` for several (points, own_pitch) at once under one ``depths``: the memo answers what it knows,
        the rest is computed in one pass (each point as alone: the same numbers)."""
        dkey = tuple(sorted(depths.items()))
        out, todo = [None] * len(queries), []
        for i, (points, own) in enumerate(queries):
            points = np.asarray(points)
            key = (points.dtype.str, points.shape, points.tobytes(), dkey, own)
            hit = self._memo.get(key)
            if hit is None:
                todo.append((i, key, points, own))
            else:
                out[i] = hit
        if todo:
            pts = np.concatenate([t[2] for t in todo])
            owns = np.concatenate([np.full(len(t[2]), -2 if t[3] is None else t[3], dtype=np.int16) for t in todo])
            z, soft = self._compute(pts, depths, owns)
            if len(self._memo) > 4096:
                self._memo.clear()
            at = 0
            for i, key, points, _ in todo:
                hit = (z[at:at + len(points)], soft[at:at + len(points)])
                at += len(points)
                for a in hit:
                    a.setflags(write=False)
                self._memo[key] = out[i] = hit
        return out

    def clearance_of(self, hit, points, own_pitch, brush=False):
        """``clearance`` from a ``_surface`` result."""
        z, soft = hit
        pen = np.maximum(0., z - points[:, 2])
        if not brush or own_pitch is None:
            return pen
        return soft * brushed(pen, True) + (1 - soft) * pen

    def _compute(self, points, depths, own_pitch):
        """``own_pitch``: one pitch, None, or one per point (-2: none)."""
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

    def bounds(self, digit):
        """The rig's joint limits, optionally narrowed to a PIP band. Diagnosis only
        (PIANOMOTION_PIP_BAND=lo,hi): it answers whether a press really needs the curl it takes, or whether
        the finger is only driven to its stop by an unreachable target."""
        finger = self.solver.rig.fingers[self.solver.hand][digit]
        lower, upper = finger.lower, finger.upper
        band = PIP_BAND
        if band is None:
            return lower, upper
        lower, upper = lower.copy(), upper.copy()
        lower[2] = max(lower[2], band[0])
        upper[2] = min(upper[2], band[1])
        return lower, upper

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
            posture = posture_residual(v, digit)
            return np.r_[off[0] / .0002, along, off[2] / .0002, max(0., front - point[1]) / .0002,
                         (v - seed) / IK_VALUE_SCALE, below, pad_residual(self.solver, digit, v, palm), steady, posture]

        from . import fast_settle       # (late: it reads this module's tables)
        fast = fast_settle.require()
        lower, upper = self.bounds(digit)

        def run(x0):
            if fast:
                x, _ = fast_settle.ride(fast_settle.pack(self.solver), digit, palm, target, seed, depths, own_pitch,
                                        x0, start, brush, front, lower, upper,
                                        IK_ITERATIONS, (IK_VALUE_SCALE, RIDE_SLIP, RIDE_STEADY, PIP_POSTURE_SCALE) + posture_flat(digit))
                x = np.asarray(x, FLOAT)
            else:
                x, _ = PoseSolver._levenberg_marquardt(residual, np.clip(x0, lower, upper),
                                                       lower, upper, np.full(4, 1e-2, dtype=FLOAT), IK_ITERATIONS)
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

    def solve(self, digit, palm, target, seed, lowered=None, start=None):
        """Values putting the pad's contact point on ``target`` (world); weak prior toward ``seed``.
        With ``lowered`` (own key depth) the skin also stays above the key floor. ``start`` is only the
        numerical starting point: the prior says where the finger WANTS to be, the start where it WAS. Along
        the finger's null space (a 3-D target, 4 joints) the target has no gradient, so even this weak prior
        decides the pose there -- which is exactly what we want it to do, and what it could not do while the
        start and the prior were the same chained pose (the finger then stayed wherever it had wandered,
        including pinned against a joint stop)."""
        finger = self.solver.rig.fingers[self.solver.hand][digit]
        values = {d: RELAXED[d] for d in DIGITS}
        seed = np.asarray(seed, FLOAT)

        def residual(v):
            values[digit] = v
            point = self.solver.contact_point(self.solver.pad_world(digit, values, palm))
            out = np.r_[(point - target) / .0002, (v - seed) / (IK_VALUE_SCALE * FINGER_JOINTS[digit]),
                        pad_residual(self.solver, digit, v, palm),
                        *posture_residual(v, digit)]
            if lowered is not None:
                out = np.r_[out, self.below_floor(digit, values, palm, lowered)]
            return out

        lower, upper = self.bounds(digit)
        x0 = seed if start is None else np.asarray(start, FLOAT)
        x, _ = PoseSolver._levenberg_marquardt(residual, np.clip(x0, lower, upper),
                                               lower, upper, np.full(4, 1e-2, dtype=FLOAT), IK_ITERATIONS)
        values[digit] = x
        point = self.solver.contact_point(self.solver.pad_world(digit, values, palm))
        return x, float(np.linalg.norm(point - target))


def finger_track(hand, digit, notes, strokes_by_note, waypoints, path, solver, ik):
    """Keyposes at each stroke's start (hover) and end (lifted), relaxed hand in long gaps."""
    fired = {'reconcile': 0, 'reconcile_deg': 0., 'let_go': 0, 'aim_early': 0, 'seed_chain': 0, 'pose_pair': 0}
    mine = sorted((replace(strokes_by_note[n.id]) for n in notes.values()
                   if n.hand == hand and n.digit == digit and n.id in waypoints), key=lambda s: s.hover)
    # free fingers wait in the playing hand's shape (pose.hand_shape), arched or flattened by the span of the keys the
    # hand is on around then (held, or about to be struck)
    mine_hand = [n for n in notes.values() if n.hand == hand and n.id in waypoints]

    # The stroke shapes are solved before the free-time keyposes, because a free pose is only meaningful once
    # the poses it waits between exist. (What they are used for: see shape_at below -- it is the hand's shape,
    # not the stroke's; deriving it from the strokes made a resting finger drift from one solved pose to the
    # next across the whole gap, which is a slow aimless turn, not a rest.)
    seed_previous = None
    for s in mine:
        note = notes[s.note]
        poses = []        # A finger that has just played keeps the configuration it lifted with into its next note: the whole
        # stroke is seeded from the previous stroke's lift pose, so the IK stays on one branch across the
        # notes instead of picking a fresh null-space solution for each one. After a real rest the waypoint's
        # own shape takes over again (the finger has had time to re-form).
        seed = None
        if seed_previous is not None and s.hover - seed_previous.lift_end <= RELAX_GAP_SECONDS:
            seed = seed_previous.poses[3].copy()
            fired['seed_chain'] += 1
        for t, hand_at in ((s.hover, s.touch), (s.touch, s.touch), (s.release, s.release), (s.lift_end, s.release)):
            target, wp_seed = stroke_target(s, note, waypoints[s.note], t, path)
            palm = solver.palm(*_split(path(hand_at)))
            # The four poses are one movement -- approach, press, hold, lift -- so each is seeded by the one
            # before it. Solving all four from the same waypoint seed let the weak seed prior (IK_VALUE_SCALE:
            # 15 deg of deviation buys a 1 mm better target) pick a different null-space branch per pose: the
            # lift, five millimetres above the touch, came back tens of degrees away from the pose it lifted
            # from. On a repeated note -- its next stroke 0.08-0.15 s away -- that is a full curl-and-uncurl
            # between two notes of the same finger, and it reads as the finger trembling. The approach keeps
            # the waypoint's own shape; only the pose chain inside the stroke is new.
            seed = wp_seed if seed is None else seed
            values, _ = ik.solve(digit, palm, target, seed, stroke_floor(s, note, t), start=seed)
            poses.append(values)
            seed = values
        s.poses = tuple(poses)
        seed_previous = s

    def shape_at(t):
        pitches = [n.pitch for n in mine_hand if n.touch - PREPARE_SECONDS <= t <= n.release]
        if pitches:
            base = hand_shape(solver.span(pitches))[digit]
        elif not mine_hand:
            base = ARCHED[digit]
        else:
            # Nothing on the keys: the shape of the keys nearest in time, not the empty hand's arch. A free finger
            # that straightened out and curled back between two chords read as the hand fidgeting while it held
            # nothing.
            near = min(mine_hand, key=lambda n: min(abs(t - (n.touch - PREPARE_SECONDS)), abs(t - n.release)))
            base = hand_shape(solver.span([near.pitch]))[digit]
        # The free finger HOLDS the hand's shape; it leaves and re-enters it through its own stroke's pose --
        # it is still its lift pose within RETURN_SECONDS of the lift, and already its hover pose within
        # PREPARE_SECONDS of the approach. Without the first, a finger uncurled out of the key it had just
        # played and curled back for the next one: a full curl-and-uncurl between two of its own notes, which
        # reads as trembling. Without the second the approach was a snap. Between them it holds one shape,
        # rather than drifting from one stroke's solved pose to the next across the whole gap (which is a
        # slow aimless turn, the "finger rotating for no reason").
        out = base
        if mine:
            before = max((s for s in mine if s.lift_end <= t), key=lambda s: s.lift_end, default=None)
            after = min((s for s in mine if s.hover >= t), key=lambda s: s.hover, default=None)
            if before is not None:
                w = float(np.clip((t - before.lift_end) / RETURN_SECONDS, 0., 1.))
                out = before.poses[3] * (1. - w) + out * w
            if after is not None:
                w = float(np.clip((after.hover - t) / PREPARE_SECONDS, 0., 1.))
                out = after.poses[0] * (1. - w) + out * w
        return out

    if os.environ.get('PIANOMOTION_FREEZE_FREE'):      # diagnosis only: is the free-time motion the keyposes at all?
        _frozen = hand_shape(solver.span([n.pitch for n in mine_hand]))[digit] if mine_hand else ARCHED[digit]
        shape_at = lambda t: _frozen

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
                # ONE shape for the whole gap. shape_at() reads the keys the hand is holding, and those change
                # as the other fingers play: re-evaluating it at both ends of the gap made the resting finger
                # re-form under the hand between two of its own notes -- a slow turn with no cause of its own,
                # and the bulk of the free-time rotation (the short stroke-to-stroke flicks were only ~8% of
                # it). It is fixed where the gap is, and the eases into the lift and into the next approach
                # are the only movement left.
                hold = shape_at(.5 * (t1 + t2))
                keyposes += [Keypose(t1, hold), Keypose(t2, hold.copy())]
        keyposes += [Keypose(s.hover, s.poses[0]), Keypose(s.lift_end, s.poses[3])]
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
                early = s.poses[0].copy()
                early[0] *= covered
                keyposes.append(Keypose(aim, early))
                fired['aim_early'] += 1
        lo, hi = LIFT_LATERAL_SOON
        let_go = (1. - LIFT_LATERAL_KEEP) * float(np.clip((next_hover - s.lift_end - lo) / (hi - lo), 0., 1.))
        if digit != 1 and let_go > 0 and next_hover - s.lift_end <= RELAX_GAP_SECONDS:   # longer: relaxed keyposes
            eased = s.poses[3].copy()
            eased[0] *= 1. - let_go
            keyposes.append(Keypose(s.lift_end + min(LIFT_LATERAL_SECONDS, .4 * (next_hover - s.lift_end)), eased))
            fired['let_go'] += 1
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
            fired['reconcile'] += 1
            fired['reconcile_deg'] += float(np.abs(diff).max()) * share * 2.
    if previous is not None:
        keyposes.append(Keypose(previous.lift_end + RETURN_SECONDS, shape_at(previous.lift_end + RETURN_SECONDS)))
    if last is not None:
        keyposes = [k for k in keyposes if k.time < last + RETURN_SECONDS + 1e-3]
        keyposes += [Keypose(last + RETURN_SECONDS + 1e-3, ARCHED[digit]),
                     Keypose(last + KEY_RETURN_SECONDS + LEAD_IN_SECONDS, RELAXED[digit])]
    else:
        keyposes = [Keypose(0., RELAXED[digit])]
    keyposes.sort(key=lambda k: k.time)
    # No two neighbouring poses may ask for more than TRANSITION_SPEED. The stroke pairs were already held to
    # this (the pass above); the free poses between them were not, and a free finger parked in one shape for a
    # tenth of a second and then taking the next one 0.08 s later swings its knuckle 37 deg in three frames
    # (Canon's R ring finger, blend frame 1904: -12.9 -> +24, its limit, in 50 ms) -- a pop with no note behind
    # it. Free poses are copies, so moving them cannot disturb a stroke's own solved shape.
    for a, b in zip(keyposes, keyposes[1:]):
        gap, diff = b.time - a.time, b.values - a.values
        if gap <= 1e-6 or a.values is b.values:
            continue
        apart = float(np.abs(diff).max())
        allowed = TRANSITION_SPEED * gap
        if apart > allowed:
            share = (1. - allowed / apart) / 2.
            a.values += diff * share
            b.values -= diff * share
            fired['pose_pair'] += 1
    if os.environ.get('PIANOMOTION_DEBUG_FINGERS'):      # diagnosis only: the track's knots around a window
        wins = [float(x) for x in (os.environ.get('PIANOMOTION_DEBUG_AT') or '').split(',') if x.strip()]
        if wins:
            with open(os.environ['PIANOMOTION_DEBUG_FINGERS'], 'a') as f:
                json.dump({f'{hand}{digit}': [[float(k.time), *[float(v) for v in k.values]]
                                              for k in keyposes
                                              if any(w - .8 < k.time < w + .8 for w in wins)]}, f)
    if os.environ.get('PIANOMOTION_PATCH_REPORT'):      # diagnosis only: which patch layers still fire
        print('PATCH %s%d notes=%d strokes=%d reconcile=%d (%.1f deg) let_go=%d aim_early=%d seed_chain=%d'
              % (hand, digit, len(mine_hand), len(mine), fired['reconcile'], fired['reconcile_deg'],
                 fired['let_go'], fired['aim_early'], fired['seed_chain']),
              file=sys.stderr, flush=True)
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
def frame_hand(t, path, tracks, notes, waypoints, solver, ik, depths=None, previous=None, apply_room=True,
               rest_slide=None, pitch=None):
    """Wrist vector, finger values {digit: 4} and the worst pad error on a pressed key at ``t``.
    ``depths`` (pitch: key depth now) keeps riding fingers out of the neighbouring keys. ``previous``
    ({digit: (note, values)}, updated) warm-starts a riding finger from the frame before. ``rest_slide``
    (world offset) shifts a hand that is resting on the lap so it follows the seat/leg as the body leans:
    IK then solves the arm to the moved wrist instead of pinning it in world space (the leg slides under
    a stationary hand otherwise, and the pads sink into it). Zero / None keeps the old fixed rest."""
    vector = path(t)
    if rest_slide is not None:
        vector = vector.copy()
        vector[:3] = vector[:3] + rest_slide
    palm = solver.palm(*_split(vector), pitch)
    values, error = {}, 0.
    riding = [tracks[d].stroke(t) for d in DIGITS]
    stretch = stretched(notes[s.note].pitch for s in riding if s is not None and s.touch <= t <= s.release)
    for d in DIGITS:
        s = tracks[d].stroke(t)
        if s is None:
            values[d] = tracks[d].free_values(t)
            continue
        if not s.touch <= t <= s.release:
            track = tracks[d]
            if (getattr(track, 'outro_gathered', None) is not None and track.rest_back is not None
                    and t > track.rest_back):
                # The piece has ended: its old lift stroke must not override the closing curl.
                values[d] = track.free_values(t)
                continue
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
    # The lap rest skips the finger spacing entirely: the fingers sit relaxed on the thigh (the spacing's
    # key clearance and neighbour gaps both read the hand under the keybed and would curl it back up). At both
    # ends of the piece -- before the first stroke and after the closing gesture lands -- the hand is at rest,
    # so the two gates must bracket the playing exactly alike (the user's spec: land as the intro began).
    # Aggregate rest gates across all active digits (not only digit 1 / thumb).
    t0 = tracks[DIGITS[0]]
    active_starts = [tracks[d].rest_start for d in DIGITS if tracks[d].strokes and tracks[d].rest_start is not None]
    if not active_starts:
        active_starts = [tracks[d].rest_start for d in DIGITS if tracks[d].rest_start is not None]
    rest_start = min(active_starts) if active_starts else None
    rest_ends = [tracks[d].rest_end for d in DIGITS if tracks[d].rest_end is not None]
    rest_end = min(rest_ends) if rest_ends else None
    if (rest_start is not None and t <= rest_start) or (rest_end is not None and t >= rest_end):
        return vector, values, error
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
    if os.environ.get('PIANOMOTION_DEBUG_FINGER_CURVES'):    # diagnosis only: the final per-frame finger values
        wins = [float(x) for x in (os.environ.get('PIANOMOTION_DEBUG_AT') or '').split(',') if x.strip()]
        if any(w - 1. < t < w + 1. for w in wins):
            with open(os.environ['PIANOMOTION_DEBUG_FINGER_CURVES'], 'a') as f:
                json.dump({'t': float(t), solver.hand: {d: [float(x) for x in room[d]] for d in DIGITS}}, f)
    if (getattr(t0, 'outro_gathered', None) is not None and t0.rest_back is not None
            and t > t0.rest_back):
        # Playing spacing is not the shape of a folded hand. Fade it out continuously during the curl,
        # then preserve the closing articulation exactly, as we do for the resting opener.
        active = any(s is not None and s.touch <= t <= s.release for s in riding)
        if not active:
            w = ease(min((t - t0.rest_back) / OUTRO_TUCK_SECONDS, 1.))
            room = {d: room[d] * (1. - w) + values[d] * w for d in DIGITS}
    # B1: a finger coming off a key can snap from its deeply-curled ride pose to the free hand shape in ONE frame
    # (measured on canonpd: L index at 135.87 s, knuckle -25 -> +20 deg in 16 ms, 45 deg/frame). The keypose
    # rate limit cannot see it: it happens at the stroke/free boundary, where ``stroke(t)`` stops returning the
    # stroke and the make-room spacing is switched on at once. Limit a FREE finger against the frame before,
    # on the final values, as the last word. Fingers actually riding a key are left alone (IK owns them, and a
    # limit here would keep the pad off its key). ``apply_room=False`` is the pre-pass that samples corrections;
    # it must stay raw.
    if apply_room and previous is not None:
        prev_final = previous.get('_final')
        pt = getattr(path, 'times', None)
        dt = float(np.median(np.diff(pt))) if pt is not None and len(pt) > 1 else 1. / 60.
        cap_deg = TRANSITION_SPEED * max(dt, 1e-4)
        now = {}
        for d in DIGITS:
            s = riding[DIGITS.index(d)]
            # Limit only the truly free stretch and the lift (after release). The strike/approach
            # (hover .. touch) is a deliberate move onto a key: cap it and the pad lands late.
            free = s is None or t > s.release
            v = room[d]
            if free and prev_final is not None and d in prev_final:
                p = prev_final[d]
                delta = v - p
                # per-joint bound; knuckle/PIP are the ones that pop. Keep the shape, only clamp the step.
                v = p + np.clip(delta, -cap_deg, cap_deg)
                room[d] = v
            now[d] = np.array(v, dtype=FLOAT, copy=True)
        previous['_final'] = now
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


def yield_offsets(paths, notes, duration, apart, speed, busy_margin, rate=60., closing=None, opening=None):
    """Sideways offsets per hand: where the wrists come closer than ``apart`` and only one hand is playing, the idle
    one moves out (left hand to the left, right to the right) by the shortfall, ramping at ``speed`` and held at
    zero while it plays (``busy_margin``: seconds before an onset / after a release). Both playing: unchanged.

    ``closing`` ({hand: trip_outro}): a hand that has started back to the lap is out of the way by definition --
    it descends BEHIND the keybed, not beside the playing hand. Letting the rule push it there shoved the left
    hand 3 cm sideways while it descended and then slid it across the thigh after it landed (the user: the hand
    "哆哆嗦嗦" on the way down and "不跟着腿面走" once down); its return must follow its own path.

    ``opening`` ({hand: trip_leave}): the mirror image at the other end. A hand still resting on the lap before
    it lifts off is out of the way already, and must follow its fitted lap pose exactly -- the right hand was
    pushed ~21 mm outward by a yield whose envelope propagated back into the still intro, sliding the thumb off
    the fitted spot and 7 mm into the thigh (the user: "起手式右手拇指穿腿里了"). Keep the rest stretch busy, the
    same gate as closing, so it lands as the intro began. Both are spatial facts about a hand on the lap, not a
    time window: any hand resting there, in any piece, is exempt."""

    # The paths run on the real clock, which starts at a NEGATIVE time (the intro is played out before
    # music 0). Sampling from 0 left the whole intro without its own grid points, and ``YieldingPath``
    # extrapolates its left endpoint to any t below the grid -- so a 14-21 mm yield measured at music 0 was
    # pasted, frozen, onto the resting hand for the whole intro, sliding it off the fitted thigh pose. Start
    # the grid at the paths' own first knot (the earliest of the two hands) so the intro is sampled too.
    t_first = min(float(np.asarray(getattr(paths[h], 'times', [0.]))[0]) for h in 'LR')
    times = np.arange(min(0., t_first) - 1. / rate, duration + 1. / rate, 1. / rate)
    x = {h: np.array([paths[h](t)[0] for t in times], dtype=FLOAT) for h in 'LR'}
    busy = {h: np.zeros(len(times), bool) for h in 'LR'}
    for n in notes.values():
        busy[n.hand] |= (times >= n.onset - busy_margin[0]) & (times <= n.release + busy_margin[1])
    for h in 'LR':
        start = (closing or {}).get(h)
        if start is not None:                   # reach zero before lift-off, at the yield's own speed
            busy[h] |= times >= start - .25
        leave = (opening or {}).get(h)
        if leave is not None:                   # still resting on the lap: keep its fitted pose, yield nothing
            busy[h] |= times <= leave + .25
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
