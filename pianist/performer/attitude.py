"""The palm points down the keys, on the forearm the render actually has.

posture.natural_orientation plans the hand's orientation from the *standard seat's* elbow -- the hand's plan must not
depend on where the performer sits -- while the elbows solve then places that elbow where the trunk's outline allows.
Those are not the same elbow: on the daylight piece the rendered forearm's own yaw comes out p50 11 deg, p95 39 deg
where the corridor's prediction sat at 4.

The corridor's objective was the wrist itself ("keep palm and forearm near straight"), and a straight wrist says
nothing about where the hand points: measured on the daylight piece, the fingers sit p50 13 deg, p95 27 deg off the
keys' length while the palm-vs-forearm angle is a comfortable 160 -- the hand is square to its own arm and crooked on
the keyboard, which is the angle the eye reads. So the objective here is the keyboard: the fingers point down the
keys (along the keyboard's length), and the palm-vs-forearm angle is a floor the turn may not cross (MIN_DEG), not a
target. Aligning the fingers costs almost nothing on that floor: p50 needs a 12.5 deg yaw turn (p95 27.5), and the
resulting angle is p5 145, p50 160 -- as comfortable as now, and better on the left hand's p5.

Two things are offered here, both reading the elbow the swing curve places (arm.closest_elbow with its angle):

* ``aim`` -- for the orientation pass: every anchor's relaxed orientation on that forearm, fed back through
  posture.natural_orientation's ``elbow``. The settle then moves the wrist and the palm together, so the fingers keep
  their keys; aiming the palm alone (turning the yaw at a fixed wrist) does not -- measured on the first 12 s, a
  17 deg turn pushed the fingers 49 mm off their keys.
* ``targets`` -- for the attitude refine: per frame, the yaw that puts the fingers on the keys, which is where the
  refine inserts an anchor (the wrist moves with the palm, by the settle's own pull).
"""
import math
import os

import numpy as np

from .kinematics import FLOAT
from .posture import finger_deg, palm_vs as _palm_deg

MIN_DEG = 145.           # floor on the palm-vs-forearm angle (180 straight): the wrist may not fold past this
REAIM_TURN = 5.          # deg the bulk re-aim may ask per anchor on the yaw
REAIM_ROLL = 4.          # ... and on the roll: a large ask makes the settle run the forearm to its ROLL_BOUNDS (20
                         # deg) to meet it, which folds the wrist further than the keys gain
ALIGN_TOL = 4.           # deg the fingers may sit off the keys' length and still count as pointing down the keys
STEP = .1                # s of the targets grid
DEG_STEP = 1.            # deg per yaw step while walking out from the path's own yaw
YAW_TRAVEL = 35.         # deg the wrist may deflect either way (radial/ulnar, the hand's own travel)


def _curve(data, h):
    """The swing curve as the frames read it: (knots, angles, slopes), the slopes cached on the curve."""
    from .motion import pchip_slopes
    got = data[h].get('_slopes')
    node = np.asarray(data[h]['times'], float)
    ang = np.asarray(data[h]['angles'], float)[:, None]
    if (got is None or got[0] is not data[h]['angles'] or got[1] is not data[h]['times']
            or got[2].shape != ang.shape):
        got = (data[h]['angles'], data[h]['times'], pchip_slopes(node, ang))
        data[h]['_slopes'] = got
    return node, ang, got[2]


def palm_at(solver, torso, curves, hand, v, t):
    """The rendered palm-vs-forearm angle (deg) for a wrist vector ``v`` at ``t``: the guard the attitude refine
    checks its anchors against -- a pose that turns the palm but folds the wrist further is not taken."""
    from .arm import closest_elbow
    from .motion import hermite_eval
    node, ang, slopes = _curve(curves, hand)
    arm = solver.arm
    phi = float(hermite_eval(node, ang, slopes, float(t))[0])
    shoulder = torso.shoulder_world(hand, float(t))[:3, 3]
    try:
        elbow = closest_elbow(shoulder, np.asarray(v[:3], float), arm.l1, arm.l2, arm.side, angle=phi)
    except ValueError:
        return 0.
    return _palm_deg(arm, float(v[3]), float(v[4]), np.asarray(v[:3], float) - elbow)


def finger_at(solver, v):
    """|deg| the fingers sit off the keys' length for a wrist vector ``v``: the refine's check that an anchor actually
    straightened the hand on the keyboard, not only on its own arm. (The palm's world orientation is q(yaw, roll)
    alone -- the elbow enters the wrist's comfort, not where the fingers point.)"""
    return abs(finger_deg(solver.arm, float(v[3]), float(v[4])))


def _frame_yaw(arm, v, elbow, y0=0., dy=0.):
    """((yaw, roll), palm-vs-forearm deg) for this frame: the least turn of the path's own wrist that puts the fingers
    down the keys (posture.turn_to_keys, walked out one degree at a time, nearest first).

    Roll is searched with the yaw, a few degrees either side, because the two trade against each other: a yaw ask the
    hand cannot meet in the plane is met by rolling the forearm (the settle took roll to its -20 bound on the 30 s
    test and folded the wrist to 132 deg), so a target that names only the yaw is a target the settle answers by
    folding. Where the plane cannot give the turn, the unrolled hand can.

    The floor bounds both: a pose that folds the wrist past MIN_DEG is not taken, even where it straightens the hand
    on the keyboard -- the two asks trade off at the extremes and the comfort floor wins. Where the travel runs out
    before the fingers are on the keys (the forearm comes in from too far sideways) the least crooked feasible pose is
    returned, so the hand still ends up closer than the path had it."""
    from .posture import turn_to_keys
    return turn_to_keys(arm, v[:3], float(v[3]), float(v[4]), elbow, tol=ALIGN_TOL, floor=MIN_DEG,
                        travel=YAW_TRAVEL, step=DEG_STEP, rolls=(0., -3., 3., -6., 6.), side=arm.side)


def aim(solvers, anchors, curves, torso):
    """{id(anchor): (yaw, roll)}: where each anchor's palm has to turn to sit within TARGET_DEG of the forearm the
    elbow curve places at its wrist.

    The anchor's *settled* yaw is the one measured -- not posture.natural_orientation's starting guess: the settle
    answers to the contacts, so its palm can sit 25 deg off the corridor's optimum while the corridor, reading the
    guess it started from, sees a straight wrist and never asks for anything. That was the first pass's blind spot."""
    from .arm import closest_elbow
    from .motion import hermite_eval
    from .coherence import vector_of
    from .posture import _calibration
    out = {}
    for h in 'LR':
        node, ang, slopes = _curve(curves, h)
        arm = solvers[h].arm
        y0, dy, _, _ = _calibration(solvers[h])
        for a in (a for a in anchors if a.hand == h):
            try:
                phi = float(hermite_eval(node, ang, slopes, a.time)[0])
                shoulder = torso.shoulder_world(h, a.time)[:3, 3]
                elbow = closest_elbow(shoulder, a.pose.wrist, arm.l1, arm.l2, arm.side, angle=phi)
            except ValueError:
                continue
            v = np.asarray(vector_of(a.pose), float)
            (y, r), _ = _frame_yaw(arm, v, elbow, y0, dy)
            y = v[3] + float(np.clip(y - v[3], -REAIM_TURN, REAIM_TURN))    # the refine adds the rest, guarded
            r = v[4] + float(np.clip(r - v[4], -REAIM_ROLL, REAIM_ROLL))
            out[id(a)] = np.array([y, r], dtype=FLOAT)
    return out


def idle_attitude(path, want, busy, rate=1. / 60., tau=.15, gate=3., speed=90., aim_z=None):
    """(times, yaw, roll) for a path whose *idle* stretches (``busy`` false) turn the palm onto the forearm: no key
    is held there, so nothing is bought by keeping the path's own yaw -- the wrist is simply folded in the air, which
    reads as a stiff hand. The playing frames keep the path's yaw exactly; the turn is smoothed and rate-limited so
    the hand comes off a chord already turning, not a frame after it. ``aim_z``: the wrist height below which the
    hand stops aiming -- a hand still on the lap (the piece has not started, or it is over) keeps its own relaxed
    yaw instead of snapping keyboardward at the first frame; the aim engages as the hand rises past it, so the turn
    reads as part of the lift, the way a real hand turns while it travels to the keys."""
    if len(want[0]) < 2:
        return np.asarray(want[0], float), np.zeros(len(want[0])), np.zeros(len(want[0]))
    times = np.arange(want[0][0], want[0][-1], rate)
    if len(times) < 2:
        cur = np.array([np.asarray(path(float(t)), float) for t in times]).reshape(len(times), 5)
        return times, cur[:, 3], cur[:, 4]
    cur = np.array([np.asarray(path(float(t)), float) for t in times])
    playing = np.interp(times, want[0], busy.astype(float)) > .05      # any overlap counts: a key is being kept
    aiming = np.ones(len(times), bool) if aim_z is None else cur[:, 2] >= aim_z
    if os.environ.get('PIANOMOTION_DEBUG_ATTITUDE'):       # diagnosis only: the idle-aim chase's inputs
        import json as _json
        raw = [float(x) for x in (os.environ.get('PIANOMOTION_DEBUG_AT') or '').split(',') if x.strip()]
        sel = np.zeros(len(times), bool)
        for win in (raw or [0.]):
            sel |= (times > win - 1.5) & (times < win + 1.5)
        with open(os.environ['PIANOMOTION_DEBUG_ATTITUDE'], 'a') as f:
            _json.dump(dict(rows=[[float(t), float(g), float(o), bool(p), bool(a)]
                                  for t, g, o, p, a in zip(times[sel], np.interp(times, want[0], want[1])[sel],
                                                           cur[sel, 3], playing[sel], aiming[sel])]), f)
    # Interpolate yaw & roll safely: non-finite samples in ``want`` fall back to ``own`` instead of poisoning
    # _gauss and latching _follow to NaN for the rest of the piece.
    own_y, own_r = cur[:, 3], cur[:, 4]
    goals = []
    for k, own in ((1, own_y), (2, own_r)):
        g = np.interp(times, want[0], want[k])
        goals.append(np.where(np.isfinite(g), g, own))
    gy, gr = goals
    # Joint Schmitt gate across (yaw, roll), reset outside active idle stretches so state never leaks across
    # playing/lap windows and a compensating +-3 deg roll from _frame_yaw stays coupled with its yaw.
    on_mask = np.zeros(len(times), bool)
    on = False
    for i in range(len(times)):
        if playing[i] or not aiming[i]:
            on = False
            continue
        gap = max(abs(float(gy[i]) - float(own_y[i])), abs(float(gr[i]) - float(own_r[i])))
        if not on and gap >= gate:
            on = True
        elif on and gap < gate * .25:
            on = False
        on_mask[i] = on
    out = []
    locked = playing | ~aiming
    for g, own in ((gy, own_y), (gr, own_r)):
        v = np.where(on_mask, g, own)
        target = np.where(locked, own, _gauss(v, rate, tau))
        out.append(_follow(own, target, speed * rate, locked=locked))
    return times, out[0], out[1]      # playing frames track the path's own wrist without speed clipping


def _gauss(values, step, tau):
    if len(values) < 2:
        return np.array(values, float)
    r = max(1, int(round(tau / step * 2.)))
    k = np.arange(-r, r + 1) * step
    w = np.exp(-.5 * (k / tau) ** 2)
    w /= w.sum()
    pad = np.concatenate([values[0] + (values[0] - values[1]) * np.arange(r, 0, -1), values,
                          values[-1] + (values[-1] - values[-2]) * np.arange(1, r + 1)])
    return np.convolve(pad, w, mode='valid')


def _rate_limit(values, step, rate):
    limit = rate * step
    out = values.copy()
    for i in range(1, len(out)):
        out[i] = min(max(out[i], out[i - 1] - limit), out[i - 1] + limit)
    for i in range(len(out) - 2, -1, -1):
        out[i] = min(max(out[i], out[i + 1] - limit), out[i + 1] + limit)
    return out


def _follow(cur, goal, limit, locked=None):
    """``cur`` chased toward ``goal`` at ``limit`` per step: smooth across playing/idle transitions, while
    allowing locked (playing) frames already on ``cur`` to follow fast settled wrist turns unclipped."""
    out = np.empty_like(cur)
    prev = float(cur[0])
    out[0] = prev
    for i in range(1, len(cur)):
        g = float(goal[i])
        if not math.isfinite(g):
            g = float(cur[i])
        if locked is not None and locked[i] and abs(prev - float(cur[i - 1])) <= 1e-6:
            prev = float(cur[i])
        else:
            prev += min(max(g - prev, -limit), limit)
        out[i] = prev
    return out


_ATT_CTX = None


def _att_init(ctx):
    global _ATT_CTX
    _ATT_CTX = ctx


def _att_chunk(job):
    from .arm import closest_elbow
    from .motion import hermite_eval
    h, times = job
    c = _ATT_CTX
    path, (node, ang, slopes), torso, arm, y0, dy = c['paths'][h], c['curves'][h], c['torso'], c['arms'][h], c['cal'][h][0], c['cal'][h][1]
    yaw = np.full(len(times), np.nan)
    roll = np.full(len(times), np.nan)
    for i, t in enumerate(times):
        v = np.asarray(path(float(t)), float)
        try:
            phi = float(hermite_eval(node, ang, slopes, float(t))[0])
            shoulder = torso.shoulder_world(h, float(t))[:3, 3]
            elbow = closest_elbow(shoulder, v[:3], arm.l1, arm.l2, arm.side, angle=phi)
        except ValueError:
            continue
        (yaw[i], roll[i]), _ = _frame_yaw(arm, v, elbow, y0, dy)
    return h, yaw, roll


def targets(paths, curves, torso, solvers, t0, t1, step=STEP, workers=1):
    """{hand: (times, yaw, roll)}: where the wrist would sit at each grid time if the fingers pointed down the keys on
    the elbow the swing curve places -- the attitude refine inserts an anchor wherever this sits far from the path."""
    from .arm import closest_elbow
    from .motion import hermite_eval
    from .posture import _calibration
    times = np.arange(t0, t1, step)
    if workers and workers > 1 and len(times) >= 128:
        from concurrent.futures import ProcessPoolExecutor
        c_curves = {h: _curve(curves, h) for h in 'LR'}
        c_arms = {h: solvers[h].arm for h in 'LR'}
        c_cal = {h: _calibration(solvers[h])[:2] for h in 'LR'}
        torso._invalidate_world_cache()
        ctx = dict(paths=paths, curves=c_curves, torso=torso, arms=c_arms, cal=c_cal)
        splits = [c for c in np.array_split(times, max(2, workers * 2)) if len(c)]
        jobs = [(h, c) for h in 'LR' for c in splits]
        parts = {'L': [], 'R': []}
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), initializer=_att_init, initargs=(ctx,)) as ex:
            for h, y, r in ex.map(_att_chunk, jobs):
                parts[h].append((y, r))
        return {h: (times, np.concatenate([p[0] for p in parts[h]]), np.concatenate([p[1] for p in parts[h]]))
                for h in 'LR'}
    out = {}
    for h in 'LR':
        node, ang, slopes = _curve(curves, h)
        arm = solvers[h].arm
        y0, dy, _, _ = _calibration(solvers[h])
        yaw = np.full(len(times), np.nan)
        roll = np.full(len(times), np.nan)
        for i, t in enumerate(times):
            v = np.asarray(paths[h](float(t)), float)
            try:
                phi = float(hermite_eval(node, ang, slopes, float(t))[0])
                shoulder = torso.shoulder_world(h, float(t))[:3, 3]
                elbow = closest_elbow(shoulder, v[:3], arm.l1, arm.l2, arm.side, angle=phi)
            except ValueError:
                continue
            (yaw[i], roll[i]), _ = _frame_yaw(arm, v, elbow, y0, dy)
        out[h] = (times, yaw, roll)
    return out
