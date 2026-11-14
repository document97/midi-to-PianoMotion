"""The palm follows the forearm the render actually has.

posture.natural_orientation plans the hand's orientation from the *standard seat's* elbow -- the hand's plan must not
depend on where the performer sits -- while the elbows solve then places that elbow where the trunk's outline allows.
Those are not the same elbow: on the daylight piece the rendered forearm's own yaw comes out p50 11 deg, p95 39 deg
where the corridor's prediction sat at 4. Across a held chord the settle lets the palm sag onto the arm it has, and
two thirds of the left hand's frames sit below 160 deg (p5 139), which reads as a wrist folded the wrong way.

Two things are offered here, both reading the elbow the swing curve places (arm.closest_elbow with its angle):

* ``aim`` -- for the orientation pass: every anchor's relaxed orientation on that forearm, fed back through
  posture.natural_orientation's ``elbow``. The settle then moves the wrist and the palm together, so the fingers keep
  their keys; aiming the palm alone (turning the yaw at a fixed wrist) does not -- measured on the first 12 s, a
  17 deg turn pushed the fingers 49 mm off their keys.
* ``targets`` -- for the attitude refine: per frame, the least yaw turn that reaches TARGET_DEG, which is where the
  refine inserts an anchor (the wrist moves with the palm, by the settle's own pull).
"""
import math

import numpy as np

from .kinematics import FLOAT, q_mat3

TARGET_DEG = 163.        # the palm-vs-forearm angle aimed at where the arm can give it (180 straight)
STEP = .1                # s of the targets grid
SCAN = 9                 # points between the palm's own yaw and the forearm's while looking for the least turn


def _palm_deg(arm, yaw, roll, forearm):
    axis = q_mat3(arm.palm_q(yaw, roll)) @ np.array([0., 1., 0.], dtype=FLOAT)
    return 180. - math.degrees(math.acos(np.clip(float(axis @ forearm) / max(np.linalg.norm(forearm), 1e-9), -1., 1.)))


def _curve(data, h):
    """The swing curve as the frames read it: (knots, angles, slopes), the slopes cached on the curve."""
    from .motion import pchip_slopes
    got = data[h].get('_slopes')
    node = np.asarray(data[h]['times'], float)
    ang = np.asarray(data[h]['angles'], float)[:, None]
    if got is None or got[0] is not data[h]['angles']:
        got = (data[h]['angles'], pchip_slopes(node, ang))
        data[h]['_slopes'] = got
    return node, ang, got[1]


def palm_at(solver, torso, curves, hand, v, t):
    """The rendered palm-vs-forearm angle (deg) for a wrist vector ``v`` at ``t``: the guard the attitude refine
    checks its anchors against -- a pose that turns the palm but folds the wrist further is not taken."""
    from .arm import closest_elbow
    from .motion import hermite_eval
    node, ang, slopes = _curve(curves, hand)
    arm = solver.arm
    phi = float(hermite_eval(node, ang, slopes, float(t))[0])
    shoulder = torso.shoulder_world(hand, float(t))[:3, 3]
    elbow = closest_elbow(shoulder, np.asarray(v[:3], float), arm.l1, arm.l2, arm.side, angle=phi)
    return _palm_deg(arm, float(v[3]), float(v[4]), np.asarray(v[:3], float) - elbow)


def _frame_yaw(arm, v, elbow, y0, dy):
    """(yaw, angle) for this frame: the least turn that puts the palm within TARGET_DEG of its forearm -- the path's
    own yaw where it is close enough already, or where the forearm gives no more."""
    forearm = v[:3] - elbow
    here = _palm_deg(arm, v[3], v[4], forearm)
    if here >= TARGET_DEG:
        return v[3], here
    full = (math.degrees(math.atan2(forearm[0], forearm[1])) - y0) / dy if abs(dy) > 1e-9 else v[3]
    best = _palm_deg(arm, full, v[4], forearm)
    if best <= here:                   # the forearm gives no more: leave the palm where the path put it
        return v[3], here
    aim = min(TARGET_DEG, best)
    for k in range(1, SCAN + 1):
        y = v[3] + (full - v[3]) * k / SCAN
        if _palm_deg(arm, y, v[4], forearm) >= aim:
            return y, aim
    return full, best


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
            y, _ = _frame_yaw(arm, v, elbow, y0, dy)
            out[id(a)] = np.array([y, v[4]], dtype=FLOAT)
    return out


def idle_attitude(path, want, busy, rate=1. / 60., tau=.15, gate=6., speed=160.):
    """(times, yaw, roll) for a path whose *idle* stretches (``busy`` false) turn the palm onto the forearm: no key
    is held there, so nothing is bought by keeping the path's own yaw -- the wrist is simply folded in the air, which
    reads as a stiff hand. The playing frames keep the path's yaw exactly; the turn is smoothed and rate-limited so
    the hand comes off a chord already turning, not a frame after it."""
    times = np.arange(want[0][0], want[0][-1], rate)
    cur = np.array([np.asarray(path(float(t)), float) for t in times])
    goal = np.interp(times, want[0], want[1])
    playing = np.interp(times, want[0], busy.astype(float)) > .05      # any overlap counts: a key is being kept
    yaw = np.where(playing, cur[:, 3], np.where(np.abs(goal - cur[:, 3]) > gate, goal, cur[:, 3]))
    yaw = _gauss(yaw, rate, tau)
    yaw = _rate_limit(yaw, rate, speed)
    return times, np.where(playing, cur[:, 3], yaw), cur[:, 4]        # the playing frames keep the path's own yaw


def _gauss(values, step, tau):
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


def targets(paths, curves, torso, solvers, t0, t1, step=STEP):
    """{hand: (times, yaw)}: what the palm's yaw would be at each grid time if it followed the placed elbow -- the
    attitude refine inserts an anchor wherever this sits far from the path's own yaw."""
    from .arm import closest_elbow
    from .motion import hermite_eval
    from .posture import _calibration
    times = np.arange(t0, t1, step)
    out = {}
    for h in 'LR':
        node, ang, slopes = _curve(curves, h)
        arm = solvers[h].arm
        y0, dy, _, _ = _calibration(solvers[h])
        yaw = np.full(len(times), np.nan)
        for i, t in enumerate(times):
            v = np.asarray(paths[h](float(t)), float)
            try:
                phi = float(hermite_eval(node, ang, slopes, float(t))[0])
                shoulder = torso.shoulder_world(h, float(t))[:3, 3]
                elbow = closest_elbow(shoulder, v[:3], arm.l1, arm.l2, arm.side, angle=phi)
            except ValueError:
                continue
            yaw[i] = _frame_yaw(arm, v, elbow, y0, dy)[0]
        out[h] = (times, yaw)
    return out
