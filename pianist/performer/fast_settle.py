"""coherence.settle's solve compiled (numba): the same residuals, bounds, start and bounded Levenberg-Marquardt, the
whole solve of one problem in machine code instead of some hundred small numpy calls per residual evaluation.

numba is required: without it the first solve stops immediately with how to install it (``require``; importing this
module still works inside Blender's Python, which only writes curves and never runs the solver). The rig's tables are
packed once per solver (``pack``); a problem is a tuple of small arrays (``problem``). The compiled code is kept on
disk (numba cache) and only rebuilt when this file changes.
"""
import math

import numpy as np

from .checks import BRUSH_COST, BRUSH_DEPTH, CELL, OWN_KEY_TOLERANCE
from .kinematics import DIGITS, child_euler
from .motion import FLOOR_SCALE, PAD_PITCH_MAX, PAD_PITCH_SCALE, KeyFloor, KeySurface, Spacing
from .pose import (CLEARANCE_SCALE, CONTACT_TOLERANCE, NARROW_TRANSITION, PAD_SOFTMIN_METRES, STEP_TOLERANCE, VALUE_SCALE,
                   WEB_SCALE, WRIST_SCALE)

njit = None
try:
    from numba import njit
except ImportError:
    pass
_jit = njit(cache=True, fastmath=False, nogil=True) if njit is not None else (lambda f: f)

F = np.float64
I = np.int64
FINGER_BONES = 3


def available():
    return njit is not None


def require():
    """Ensure numba is installed; raise ImportError otherwise (no slow numpy fallback)."""
    if njit is not None:
        return True
    raise ImportError('numba is required (pip install numba): the solver runs compiled with no numpy fallback.')


# ---------------------------------------------------------------- packing (python, once per solver / problem)
def pack(solver):
    """The rig's tables for the compiled solve, kept on the solver."""
    hit = getattr(solver, '_fast_pack', None)
    if hit is not None:
        return hit
    from .coherence import (FREE_CLEAR_SCALE, FREE_CURVED_SCALE, LATERAL_FREE, LATERAL_HINGE, LATERAL_SCALE,
                            NEUTRAL_ORIENTATION, ORDER_SCALE, PALM_HEIGHT, PALM_SCALE, THUMB_LATERAL_SCALE,
                            THUMB_TIP_SCALE, WEB_STRAIN, CENTRE_SCALE, WHITE_FRONT_SCALE, BLACK_FRONT_SCALE)
    rig, hand = solver.rig, solver.hand
    fingers = [rig.fingers[hand][d] for d in DIGITS]
    for f in fingers:
        assert not f.twist, 'fast_settle: finger twist not supported'
    e0 = np.array([[child_euler(f.row, i, 0.) for i in (0, 1)] for f in fingers], F)
    unit = np.array([[np.subtract(child_euler(f.row, i, 1.), e0[n][i]) for i in (0, 1)] for n, f in enumerate(fingers)], F)
    fp = (np.array([f.pre for f in fingers], F), np.array([f.root_loc for f in fingers], F),
          np.array([f.root_scale for f in fingers], F), np.array([f.child_loc for f in fingers], F),
          np.array([f.child_scale for f in fingers], F), np.array([f.links for f in fingers], F),
          np.array([f.neutral_matrix for f in fingers], F), np.array([f.normal for f in fingers], F),
          np.array([f.lift for f in fingers], F), np.array([f.base for f in fingers], F), e0, unit)
    owner = rig.bone_owner[hand]
    rigid = rig.neutral_wrist_inverse[hand]
    constants = {rig.wrist_bone[hand]: np.eye(4)}

    def source(name):
        if name in owner:
            return (owner[name] - 1) * FINGER_BONES + rig.fingers[hand][owner[name]].names.index(name)
        if name not in constants:
            constants[name] = rigid @ rig.neutral_bones[name]
        return len(DIGITS) * FINGER_BONES + list(constants).index(name)

    def padded(skins):
        points = max(s.weights.shape[0] for s in skins.values())
        bones = max(len(s.bones) for s in skins.values())
        weights = np.zeros((5, points, bones), F)
        local = np.zeros((5, bones, points, 4), F)
        sources = np.zeros((5, bones), I)
        count = np.zeros((5, 2), I)
        for d, s in skins.items():
            n, b = s.weights.shape
            weights[d - 1, :n, :b], local[d - 1, :b, :n] = s.weights, s.local
            sources[d - 1, :b] = [source(name) for name in s.bones]
            count[d - 1] = n, b
        return weights, local, sources, count
    pad = padded({d: rig.fingers[hand][d].pad for d in DIGITS})
    clear = padded({d: solver.clearance_skin[d] for d in DIGITS})
    const = np.array(list(constants.values()), F)
    arm = solver.arm
    forward = np.asarray(arm.palm_forward, F)
    palm = (np.asarray(arm.palm, F), forward / np.linalg.norm(forward), np.asarray(arm.scale['wrist'], F))
    regions = solver.regions
    pitches = max(regions.rows) + 1
    table = np.zeros((pitches, 8), F)
    for p, r in regions.rows.items():
        table[p] = [r['top'], *r['wide'], *r['narrow'], r['split'], *r['y']]
    surface = KeySurface(solver.heightfield)
    hf = solver.heightfield
    ks = (np.asarray(surface.top, F), surface.on_key.astype(np.bool_), np.asarray(hf.pitch, I), np.asarray(hf.origin, F))
    spacing = Spacing(solver)
    proximal = np.zeros((5, clear[0].shape[1]), np.bool_)
    for d in DIGITS:
        proximal[d - 1, :len(spacing.proximal[d])] = spacing.proximal[d]
    gaps = np.zeros((6, 2), F)
    for d, g in spacing.whole_gaps.items():
        gaps[d] = g
    rest, slope, free, hard = (np.asarray(a, F) for a in rig.web(hand))
    scales = np.array([CONTACT_TOLERANCE, CENTRE_SCALE, WHITE_FRONT_SCALE, BLACK_FRONT_SCALE, PAD_PITCH_MAX,
                       PAD_PITCH_SCALE, THUMB_TIP_SCALE, FLOOR_SCALE, FREE_CLEAR_SCALE, FREE_CURVED_SCALE, ORDER_SCALE,
                       LATERAL_HINGE, WEB_SCALE, WEB_STRAIN, PALM_SCALE, KeyFloor(regions).white_top + PALM_HEIGHT,
                       PAD_SOFTMIN_METRES, NARROW_TRANSITION, CELL, OWN_KEY_TOLERANCE, BRUSH_DEPTH, BRUSH_COST,
                       KeySurface.NO_KEY, STEP_TOLERANCE], F)
    per_digit = np.array([[THUMB_LATERAL_SCALE if d == 1 else LATERAL_SCALE, LATERAL_FREE[d]] for d in DIGITS], F)
    out = solver._fast_pack = (fp, pad, clear, const, palm, table, ks, (np.asarray(spacing.lateral, F), proximal, gaps),
                               (rest, slope, free, hard), scales, per_digit, np.asarray(NEUTRAL_ORIENTATION, F),
                               np.asarray(VALUE_SCALE, F),
                               (np.where(np.isfinite(hf.top), hf.top, 0.).astype(F), np.isfinite(hf.top),
                                np.asarray(WRIST_SCALE, F), np.array([CLEARANCE_SCALE], F)))
    return out


def problem(contacts, digits, x, lower, upper, values, depths, edges, blacks, centres, target, pull, natural, shape,
            curved_free, lifted, stretch, steps, iterations, curved, radial=None):
    """One settle problem as arrays (contacts sorted by digit). ``radial``: the wrist's lateral-bend wall as one
    row -- forearm unit (3), thumb-perpendicular unit (3), radial cap deg, ulnar cap deg, scale, enabled -- or None."""
    table = np.zeros(128, F)
    for p, d in depths.items():
        table[p] = d
    k = len(digits)
    return (np.array(digits, I), np.array([c.pitch for c in contacts], I), np.array([c.depth for c in contacts], F),
            np.asarray(x, F), np.asarray(lower, F), np.asarray(upper, F),
            np.array([values[d] for d in DIGITS], F), table, np.array(edges, F), np.array(blacks, np.bool_),
            np.array(centres, F), np.asarray(target, F), np.asarray(pull, F), np.asarray(natural, F),
            np.array([shape[d] for d in digits], F).reshape(k, 4),
            np.array([curved_free.get(d, np.zeros(4)) for d in DIGITS], F), np.asarray(lifted, F),
            bool(stretch), np.asarray(steps, F), int(iterations), bool(curved),
            np.asarray(radial, F) if radial is not None else np.zeros(11, F))


def pose_problem(contacts, digits, x, lower, upper, values, depths, prior_wrist, shape, stretch, steps, seeds,
                 seed_iterations, iterations, clearance, radial=None):
    """One PoseSolver.solve problem as arrays (contacts sorted by digit). ``radial``: the lateral-bend wall, one
    row -- forearm unit (3), thumb-perpendicular unit (3), radial cap deg, ulnar cap deg, scale, enabled -- or None."""
    table = np.zeros(128, F)
    for p, d in depths.items():
        table[p] = d
    k = len(digits)
    return (np.array(digits, I), np.array([c.pitch for c in contacts], I), np.array([c.depth for c in contacts], F),
            np.asarray(x, F), np.asarray(lower, F), np.asarray(upper, F), np.array([values[d] for d in DIGITS], F), table,
            np.asarray(prior_wrist, F), np.array([shape[d] for d in digits], F).reshape(k, 4), bool(stretch),
            np.asarray(steps, F), np.array(seeds, F).reshape(-1, 3), int(seed_iterations), int(iterations), bool(clearance),
            np.asarray(radial, F) if radial is not None else np.zeros(11, F))


def solve_pose(hand_pack, prob):
    """(x, iterations used) of PoseSolver.solve's seeded solve."""
    return pose_one(hand_pack, prob)


def ride(hand_pack, digit, palm, target, seed, depths, own_pitch, x0, start, brush, front, lower, upper, iterations,
         scales):
    """motion.FingerIK.ride's solve (x, iterations): the other fingers relaxed (``RELAXED``), ``scales``:
    (IK_VALUE_SCALE, RIDE_SLIP, RIDE_STEADY)."""
    from .pose import RELAXED
    table = np.zeros(128, F)
    for p, d in depths.items():
        table[p] = d
    has_start = start is not None
    R = (int(digit) - 1, np.asarray(palm, F)[:3].copy(), np.asarray(target, F), np.asarray(seed, F), table, int(own_pitch),
         np.asarray(start if has_start else seed, F), has_start, bool(brush), float(front), np.asarray(lower, F),
         np.asarray(upper, F), int(iterations), np.asarray(scales, F), np.array([RELAXED[d] for d in DIGITS], F),
         np.asarray(x0, F))
    return ride_lm(hand_pack, R)


def solve(hand_pack, prob):
    """(x, iterations used) of coherence.settle's solve, with its retry without the curved free fingers."""
    return solve_one(hand_pack, prob)


# ---------------------------------------------------------------- compiled code

@_jit
def mv3(a, v):
    return np.array([a[0, 0] * v[0] + a[0, 1] * v[1] + a[0, 2] * v[2], a[1, 0] * v[0] + a[1, 1] * v[1] + a[1, 2] * v[2],
                     a[2, 0] * v[0] + a[2, 1] * v[1] + a[2, 2] * v[2]])


@_jit
def mm3(a, b):
    out = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            out[i, j] = a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + a[i, 2] * b[2, j]
    return out


@_jit
def mm4(a, b, out):
    for i in range(4):
        for j in range(4):
            out[i, j] = a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + a[i, 2] * b[2, j] + a[i, 3] * b[3, j]


@_jit
def axis_matrix(axis, radians, out):
    x, y, z = axis[0], axis[1], axis[2]
    c, s = math.cos(radians), math.sin(radians)
    t = 1. - c
    out[0, 0] = t * x * x + c
    out[0, 1] = t * x * y - s * z
    out[0, 2] = t * x * z + s * y
    out[1, 0] = t * x * y + s * z
    out[1, 1] = t * y * y + c
    out[1, 2] = t * y * z - s * x
    out[2, 0] = t * x * z - s * y
    out[2, 1] = t * y * z + s * x
    out[2, 2] = t * z * z + c

@_jit
def euler_xyz(ex, ey, ez, out):
    cx, sx, cy, sy, cz, sz = math.cos(ex), math.sin(ex), math.cos(ey), math.sin(ey), math.cos(ez), math.sin(ez)
    out[0, 0] = cy * cz
    out[0, 1] = sx * sy * cz - cx * sz
    out[0, 2] = cx * sy * cz + sx * sz
    out[1, 0] = cy * sz
    out[1, 1] = sx * sy * sz + cx * cz
    out[1, 2] = cx * sy * sz - sx * cz
    out[2, 0] = -sy
    out[2, 1] = sx * cy
    out[2, 2] = cx * cy

@_jit
def compose(loc, rot, scale, out):
    for i in range(3):
        for j in range(3):
            out[i, j] = rot[i, j] * scale[j]
        out[i, 3] = loc[i]
        out[3, i] = 0.
    out[3, 3] = 1.

@_jit
def finger(fp, d, v, out, w):
    """kinematics.Finger._local_matrices of digit index d (0-based), values v (4,) -> out (3, 4, 4)."""
    pre, root_loc, root_scale, child_loc, child_scale, links, neutral, normal, lift, base, e0, unit = fp
    r1, r2, r3 = np.empty((3, 3)), np.empty((3, 3)), np.empty((3, 3))
    m, m2 = w[3], w[4]
    axis_matrix(lift[d], math.radians(v[1]), r1)
    lifted = mv3(r1, base[d])
    axis_matrix(normal[d], math.radians(v[0]), r2)
    axis = mv3(r2, lifted)
    b = base[d]
    k = np.array([b[1] * axis[2] - b[2] * axis[1], b[2] * axis[0] - b[0] * axis[2], b[0] * axis[1] - b[1] * axis[0]])
    sine = math.sqrt(k[0] * k[0] + k[1] * k[1] + k[2] * k[2])
    if sine < 1e-12:
        swing = neutral[d].copy()
    else:
        axis_matrix(k / sine, math.atan2(sine, b[0] * axis[0] + b[1] * axis[1] + b[2] * axis[2]), r3)
        swing = mm3(r3, neutral[d])
    compose(root_loc[d], swing, root_scale[d], m)
    mm4(pre[d], m, out[0])
    for i in range(2):
        bend = v[2 + i]
        euler_xyz(e0[d, i, 0] + bend * unit[d, i, 0], e0[d, i, 1] + bend * unit[d, i, 1],
                  e0[d, i, 2] + bend * unit[d, i, 2], r1)
        compose(child_loc[d, i], r1, child_scale[d, i], m2)
        t = np.empty((4, 4))
        mm4(out[i], links[d, i], t)
        mm4(t, m2, out[i + 1])

@_jit
def palm_matrix(pm, x, out):
    """PoseSolver.palm (3 x 4 rows)."""
    q0, fwd, scale = pm
    h = math.radians(x[4]) / 2
    n = math.sqrt(fwd[0] ** 2 + fwd[1] ** 2 + fwd[2] ** 2)
    a = np.array([math.cos(h), math.sin(h) * fwd[0] / n, math.sin(h) * fwd[1] / n, math.sin(h) * fwd[2] / n])
    b = qmul(a, q0)
    h = math.radians(x[3]) / 2
    a = np.array([math.cos(h), 0., 0., math.sin(h)])
    q = qmul(a, b)
    q = q / math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    w, qx, qy, qz = q[0], q[1], q[2], q[3]
    out[0, 0] = (1 - 2 * (qy * qy + qz * qz)) * scale[0]
    out[0, 1] = 2 * (qx * qy - qz * w) * scale[1]
    out[0, 2] = 2 * (qx * qz + qy * w) * scale[2]
    out[1, 0] = 2 * (qx * qy + qz * w) * scale[0]
    out[1, 1] = (1 - 2 * (qx * qx + qz * qz)) * scale[1]
    out[1, 2] = 2 * (qy * qz - qx * w) * scale[2]
    out[2, 0] = 2 * (qx * qz - qy * w) * scale[0]
    out[2, 1] = 2 * (qy * qz + qx * w) * scale[1]
    out[2, 2] = (1 - 2 * (qx * qx + qy * qy)) * scale[2]
    out[0, 3], out[1, 3], out[2, 3] = x[0], x[1], x[2]

@_jit
def qmul(a, b):
    w1, x1, y1, z1 = a[0], a[1], a[2], a[3]
    w2, x2, y2, z2 = b[0], b[1], b[2], b[3]
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

@_jit
def skin_local(skin, d, mats, out):
    """Skin points of digit index d in the wrist frame: out (P, 4); returns the point count."""
    weights, local, sources, count = skin
    n, nb = count[d, 0], count[d, 1]
    for p in range(n):
        for j in range(4):
            out[p, j] = 0.
    for b in range(nb):
        m = mats[sources[d, b]]
        for p in range(n):
            w = weights[d, p, b]
            if w == 0.:
                continue
            l0, l1, l2, l3 = local[d, b, p, 0], local[d, b, p, 1], local[d, b, p, 2], local[d, b, p, 3]
            for j in range(4):
                out[p, j] += w * (m[j, 0] * l0 + m[j, 1] * l1 + m[j, 2] * l2 + m[j, 3] * l3)
    return n

@_jit
def to_world(palm, loc, n, out):
    for p in range(n):
        for i in range(3):
            out[p, i] = palm[i, 0] * loc[p, 0] + palm[i, 1] * loc[p, 1] + palm[i, 2] * loc[p, 2] + palm[i, 3] * loc[p, 3]

@_jit
def surface(ks, sc, table, pt, own):
    """motion.KeySurface._compute of one point: (height, brushable share)."""
    top, on_key, pitch_grid, origin = ks
    cell, own_tol, no_key = sc[18], sc[19], sc[22]
    ux = (pt[0] - origin[0]) / cell
    uy = (pt[1] - origin[1]) / cell
    ix0 = min(max(int(math.floor(ux)), 0), top.shape[0] - 2)
    iy0 = min(max(int(math.floor(uy)), 0), top.shape[1] - 2)
    wx = min(max(ux - ix0, 0.), 1.)
    wy = min(max(uy - iy0, 0.), 1.)
    zs, ss, total = 0., 0., 0.
    for c in range(4):
        dx, dy = c % 2, c // 2
        w = (wx if dx else 1 - wx) * (wy if dy else 1 - wy)
        ix, iy = ix0 + dx, iy0 + dy
        p = pitch_grid[ix, iy]
        pressed = table[p] if p >= 0 else table[table.shape[0] + p]
        z = top[ix, iy] - pressed
        if p == own:
            z -= own_tol
        if not on_key[ix, iy]:
            w = 0.
        zs += w * z
        ss += w * (1. if (p != own and pressed <= 0.) else 0.)
        total += w
    if total > 1e-9:
        return zs / total, ss / total
    return no_key, 0.

@_jit
def contact_point(world, n, sc, out):
    zmin = 1e30
    for p in range(n):
        zmin = min(zmin, world[p, 2])
    s = 0.
    out[0], out[1], out[2] = 0., 0., 0.
    for p in range(n):
        w = math.exp(-(world[p, 2] - zmin) / sc[16])
        s += w
        for i in range(3):
            out[i] += w * world[p, i]
    for i in range(3):
        out[i] /= s

@_jit
def region(table, sc, pitch, depth, pt, out):
    top, w0, w1, n0, n1, split, y0, y1 = (table[pitch, 0], table[pitch, 1], table[pitch, 2], table[pitch, 3],
                                          table[pitch, 4], table[pitch, 5], table[pitch, 6], table[pitch, 7])
    s = 1. / (1. + math.exp(-max(-50., min(50., (pt[1] - split) / sc[17]))))
    lo = (1 - s) * w0 + s * n0
    hi = (1 - s) * w1 + s * n1
    out[0] = max(0., lo - pt[0]) - max(0., pt[0] - hi)
    out[1] = max(0., y0 - pt[1]) - max(0., pt[1] - y1)
    out[2] = pt[2] - (top - depth)

@_jit
def new_cache():
    """Kept finger matrices of one solve: (the values mats holds per finger (nan: none), the free fingers' lifted and
    curved matrices, which of those are made)."""
    return np.full((5, 4), np.nan), np.empty((5, 2, 3, 4, 4)), np.zeros((5, 2), np.bool_)


@_jit
def finger_kept(fp, d, v, mats, work, C):
    """finger() into mats unless mats already holds digit d at these values (a difference step moves one variable)."""
    held = C[0]
    for j in range(4):
        if held[d, j] != v[j]:
            finger(fp, d, v, mats[3 * d:3 * d + 3], work)
            for i in range(4):
                held[d, i] = v[i]
            return


@_jit
def pose_mats(H, Q, x, vals, mats, work, C):
    fp, const = H[0], H[3]
    digits = Q[0]
    vals[:, :] = Q[6]
    for i in range(len(digits)):
        for j in range(4):
            vals[digits[i] - 1, j] = x[5 + 4 * i + j]
    for d in range(5):
        finger_kept(fp, d, vals[d], mats, work, C)
    nb = 3 * 5
    for c in range(const.shape[0]):
        mats[nb + c] = const[c]

@_jit
def residuals(H, Q, x, curved, vals, mats, alt, work, loc, world, out, C):
    """coherence.settle's residuals at x into out; returns their count."""
    fp, pad, clear, const, pm, table, ks, spacing, web, sc, per_digit, neutral_or, value_scale = (
        H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7], H[8], H[9], H[10], H[11], H[12])
    (digits, pitches, depths, _x, _lo, _up, _values, dtable, edges, blacks, centres, target, pull, natural, shape,
     curved_free, lifted, stretch, _steps, _it, _curved, radial) = Q
    k = len(digits)
    pose_mats(H, Q, x, vals, mats, work, C)
    palm = np.empty((3, 4))
    palm_matrix(pm, x, palm)
    pt = np.empty(3)
    rr = np.empty(3)
    m = 0
    for i in range(k):
        d = digits[i] - 1
        n = skin_local(pad, d, mats, loc)
        to_world(palm, loc, n, world)
        contact_point(world, n, sc, pt)
        region(table, sc, pitches[i], depths[i], pt, rr)
        for j in range(3):
            out[m] = rr[j] / sc[0]
            m += 1
        out[m] = (pt[0] - centres[i]) / sc[1]
        m += 1
        if not blacks[i]:
            out[m] = max(0., pt[1] - edges[i]) / sc[2]
        else:
            out[m] = max(0., edges[i] - pt[1]) / sc[3]
        m += 1
        # last phalanx: pad pitch (fingers 2..5) or the thumb's tip angle
        last = mats[3 * d + 2]
        ax0 = palm[0, 0] * last[0, 1] + palm[0, 1] * last[1, 1] + palm[0, 2] * last[2, 1]
        ax1 = palm[1, 0] * last[0, 1] + palm[1, 1] * last[1, 1] + palm[1, 2] * last[2, 1]
        ax2 = palm[2, 0] * last[0, 1] + palm[2, 1] * last[1, 1] + palm[2, 2] * last[2, 1]
        if d == 0:
            out[m] = 0.
            m += 1
            out[m] = math.degrees(math.atan2(ax0, ax1)) / sc[6]
            m += 1
        else:
            f0, f1 = palm[0, 1], palm[1, 1]
            fn = math.sqrt(f0 * f0 + f1 * f1)
            pitch = math.degrees(math.atan2(-ax2, (ax0 * f0 + ax1 * f1) / fn))
            out[m] = max(0., pitch - sc[4]) / sc[5]
            m += 1
        n = skin_local(clear, d, mats, loc)
        to_world(palm, loc, n, world)
        for p in range(n):
            z, soft = surface(ks, sc, dtable, world[p], pitches[i])
            pen = max(0., z - world[p, 2])
            if stretch:
                brushed = max(pen - sc[20], 0.) + sc[21] * min(max(pen, 0.), sc[20])
                pen = soft * brushed + (1 - soft) * pen
            out[m] = pen / sc[7]
            m += 1
    # free fingers: lifted, then curved
    for d in range(5):
        free = True
        for i in range(k):
            if digits[i] - 1 == d:
                free = False
        if not free:
            continue
        own = mats[3 * d:3 * d + 3].copy()
        for variant in range(2 if curved else 1):
            if not C[2][d, variant]:
                finger(fp, d, lifted[0 if d == 0 else 1] if variant == 0 else curved_free[d], C[1][d, variant], work)
                C[2][d, variant] = True
            mats[3 * d:3 * d + 3] = C[1][d, variant]
            n = skin_local(clear, d, mats, loc)
            to_world(palm, loc, n, world)
            scale = sc[8] if variant == 0 else sc[9]
            for p in range(n):
                z, soft = surface(ks, sc, dtable, world[p], -2)
                out[m] = max(0., z - world[p, 2]) / scale
                m += 1
        mats[3 * d:3 * d + 3] = own
    # chord fingers 2..5 keep their order (wrist-frame lateral positions of their halves)
    lateral, proximal, gaps = spacing
    prev = -1
    pa0, pa1 = 0., 0.
    for i in range(k):
        dg = digits[i]
        if dg < 2:
            continue
        n = skin_local(clear, dg - 1, mats, loc)
        s0, c0, s1, c1 = 0., 0, 0., 0
        for p in range(n):
            v = lateral[0] * loc[p, 0] + lateral[1] * loc[p, 1] + lateral[2] * loc[p, 2]
            if proximal[dg - 1, p]:
                s0 += v
                c0 += 1
            else:
                s1 += v
                c1 += 1
        q0, q1 = s0 / max(c0, 1), s1 / max(c1, 1)
        if prev >= 2:
            need0, need1 = 0., 0.
            for g in range(prev, dg):
                need0 += gaps[g, 0]
                need1 += gaps[g, 1]
            out[m] = max(0., need0 / .85 * .6 - (q0 - pa0)) / sc[10]
            out[m + 1] = max(0., need1 / .85 * .6 - (q1 - pa1)) / sc[10]
            m += 2
        prev, pa0, pa1 = dg, q0, q1
    for i in range(k):
        out[m] = vals[digits[i] - 1, 0] / per_digit[digits[i] - 1, 0]
        m += 1
    for i in range(k):
        out[m] = max(0., abs(vals[digits[i] - 1, 0]) - per_digit[digits[i] - 1, 1]) / sc[11]
        m += 1
    rest, slope, wfree, hard = web
    for which in range(2):
        for i in range(k):
            for j in range(i + 1, k):
                a, b = digits[i], digits[j]
                opening = rest[a, b] + slope[a, b] * (vals[b - 1, 0] - vals[a - 1, 0])
                if which == 0:
                    out[m] = max(0., opening - hard[a, b]) / sc[12]
                else:
                    out[m] = max(0., opening - wfree[a, b]) / sc[12] * (sc[12] / sc[13])
                m += 1
    for j in range(5):
        out[m] = (x[j] - target[j]) / pull[j]
        m += 1
    out[m] = max(0., sc[15] - x[2]) / sc[14]
    m += 1
    for j in range(2):
        out[m] = (x[3 + j] - natural[j]) / neutral_or[j]
        m += 1
    if radial[9] > 0.:
        # the wrist's lateral-bend wall: the palm's long axis (palm column 1) folded against the forearm plane;
        # leaning past a cap costs -- thumb side past radial[6], little-finger side past radial[7], / radial[8]
        p0, p1, p2 = palm[0, 1], palm[1, 1], palm[2, 1]
        f0, f1, f2 = radial[0], radial[1], radial[2]
        pdot = p0 * f0 + p1 * f1 + p2 * f2
        pp0, pp1, pp2 = p0 - pdot * f0, p1 - pdot * f1, p2 - pdot * f2
        tdot = pp0 * radial[3] + pp1 * radial[4] + pp2 * radial[5]
        dev = math.degrees(math.asin(max(-1., min(1., tdot))))   # thumb-perp arrives a unit vector
        out[m] = (max(0., dev - radial[6]) + max(0., -dev - radial[7])) / radial[8]
        m += 1
    for i in range(k):
        for j in range(4):
            out[m] = (vals[digits[i] - 1, j] - shape[i, j]) / value_scale[j]
            m += 1
    return m

@_jit
def contact_error(H, Q, x, vals, mats, work, loc, world, C):
    pad, table, sc = H[1], H[5], H[9]
    digits, pitches, depths = Q[0], Q[1], Q[2]
    pose_mats(H, Q, x, vals, mats, work, C)
    palm = np.empty((3, 4))
    palm_matrix(H[4], x, palm)
    pt, rr = np.empty(3), np.empty(3)
    worst = 0.
    for i in range(len(digits)):
        n = skin_local(pad, digits[i] - 1, mats, loc)
        to_world(palm, loc, n, world)
        contact_point(world, n, sc, pt)
        region(table, sc, pitches[i], depths[i], pt, rr)
        for j in range(3):
            worst = max(worst, abs(rr[j]))
    return worst

@_jit
def linsolve(a, b):
    """Gaussian elimination with partial pivoting (np.linalg.solve)."""
    n = len(b)
    a = a.copy()
    b = b.copy()
    for c in range(n):
        p = c
        for r in range(c + 1, n):
            if abs(a[r, c]) > abs(a[p, c]):
                p = r
        if p != c:
            for j in range(n):
                a[c, j], a[p, j] = a[p, j], a[c, j]
            b[c], b[p] = b[p], b[c]
        piv = a[c, c]
        for r in range(c + 1, n):
            f = a[r, c] / piv
            if f != 0.:
                for j in range(c, n):
                    a[r, j] -= f * a[c, j]
                b[r] -= f * b[c]
    x = np.empty(n, a.dtype)
    for r in range(n - 1, -1, -1):
        s = b[r]
        for j in range(r + 1, n):
            s -= a[r, j] * x[j]
        x[r] = s / a[r, r]
    return x

@_jit
def normal_bound(jac, r, x, lo, up):
    """J^T r, J^T J with every variable held at a bound its gradient pushes past taken out (pose.bound_step)."""
    m, n = jac.shape
    g = np.zeros(n, np.float32)
    h = np.zeros((n, n), np.float32)
    for i in range(m):
        ri = r[i]
        for a in range(n):
            ja = jac[i, a]
            g[a] += ja * ri
            for b in range(n):
                h[a, b] += ja * jac[i, b]
    for a in range(n):
        if (x[a] <= lo[a] and g[a] > 0) or (x[a] >= up[a] and g[a] < 0):
            for b in range(n):
                h[a, b] = 0.
                h[b, a] = 0.
            h[a, a] = 1.
            g[a] = 0.
    return g, h


@_jit
def damped(h, g, lam):
    n = len(g)
    a = h.copy()
    for j in range(n):
        a[j, j] += np.float32(lam) * (h[j, j] + np.float32(1e-9))
    return linsolve(a, -g)


@_jit
def lm(H, Q, curved, cap):
    """PoseSolver._levenberg_marquardt on coherence.settle's residuals (float32 state, as the numpy solve)."""
    x0, lower, upper, steps, iterations = Q[3], Q[4], Q[5], Q[18], Q[19]
    sc = H[9]
    n = len(x0)
    vals = np.empty((5, 4))
    alt = np.empty((5, 4))
    mats = np.empty((15 + H[3].shape[0], 4, 4))
    work = np.empty((5, 4, 4))
    loc = np.empty((cap, 4))
    world = np.empty((cap, 3))
    buf = np.empty(16384)
    C = new_cache()
    x = x0.astype(np.float32)
    lo, up, st = lower.astype(np.float32), upper.astype(np.float32), steps.astype(np.float32)
    m = residuals(H, Q, x.astype(np.float64), curved, vals, mats, alt, work, loc, world, buf, C)
    r = buf[:m].astype(np.float32)
    cost = np.float32(np.dot(r, r))
    lam = 1e-2
    jac = np.empty((m, n), np.float32)
    for it in range(1, iterations + 1):
        for j in range(n):
            trial = x.copy()
            trial[j] += st[j]
            residuals(H, Q, trial.astype(np.float64), curved, vals, mats, alt, work, loc, world, buf, C)
            for i in range(m):
                jac[i, j] = (np.float32(buf[i]) - r[i]) / st[j]
        g, h = normal_bound(jac, r, x, lo, up)
        improved = False
        delta = np.zeros(n, np.float32)
        for _ in range(10):
            delta = damped(h, g, lam)
            trial = np.minimum(np.maximum(x + delta, lo), up)
            residuals(H, Q, trial.astype(np.float64), curved, vals, mats, alt, work, loc, world, buf, C)
            tr = buf[:m].astype(np.float32)
            tc = np.float32(np.dot(tr, tr))
            if tc < cost:
                x, r, cost, lam, improved = trial, tr, tc, max(lam / 3, 1e-7), True
                break
            lam *= 4
        if not improved or np.max(np.abs(delta)) < sc[23] or cost < 1e-6:
            return x, it
    return x, iterations

@_jit
def solve_one(H, Q):
    cap = max(H[1][3][:, 0].max(), H[2][3][:, 0].max())
    x, used = lm(H, Q, Q[20], cap)
    digits = Q[0]
    if Q[20] and len(digits) < 5:
        vals = np.empty((5, 4))
        mats = np.empty((15 + H[3].shape[0], 4, 4))
        work = np.empty((5, 4, 4))
        loc = np.empty((cap, 4))
        world = np.empty((cap, 3))
        if contact_error(H, Q, x.astype(np.float64), vals, mats, work, loc, world, new_cache()) > H[9][0] * 2.5:
            x, more = lm(H, Q, False, cap)
            used += more
    return x, used





# ---------------------------------------------------------------- PoseSolver.solve
@_jit
def penetration(H, dtable, pt, own, brush):
    """checks.Heightfield.penetration of one point (own: the finger's pitch, all points allowed)."""
    top, finite = H[13][0], H[13][1]
    origin, pitch_grid, sc = H[6][3], H[6][2], H[9]
    ix = int(math.floor((pt[0] - origin[0]) / sc[18]))
    iy = int(math.floor((pt[1] - origin[1]) / sc[18]))
    if ix < 0 or iy < 0 or ix >= top.shape[0] or iy >= top.shape[1]:
        return 0.
    if not finite[ix, iy]:
        return 0.
    p = pitch_grid[ix, iy]
    pressed = dtable[p] if p >= 0 else dtable[dtable.shape[0] + p]
    pen = top[ix, iy] - pressed - pt[2]
    if p == own:
        pen -= sc[19]
    if brush:
        pen = max(pen, 0.)
        if p != own and pressed <= 0.:
            pen = max(pen - sc[20], 0.) + sc[21] * min(max(pen, 0.), sc[20])
    return max(pen, 0.)


@_jit
def pose_residuals(H, Q, x, prior, clear, vals, mats, work, loc, world, out, C):
    """PoseSolver.solve's residuals at x into out; returns their count."""
    pad, clearskin, pm, table, web, sc, value_scale = H[1], H[2], H[4], H[5], H[8], H[9], H[12]
    wrist_scale, clear_scale = H[13][2], H[13][3][0]
    digits, pitches, depths, dtable, prior_wrist, shape, stretch = Q[0], Q[1], Q[2], Q[7], Q[8], Q[9], Q[10]
    radial = Q[16]
    k = len(digits)
    vals[:, :] = Q[6]
    for i in range(k):
        for j in range(4):
            vals[digits[i] - 1, j] = x[5 + 4 * i + j]
    for d in range(5):
        finger_kept(H[0], d, vals[d], mats, work, C)
    const = H[3]
    for c in range(const.shape[0]):
        mats[15 + c] = const[c]
    palm = np.empty((3, 4))
    palm_matrix(pm, x, palm)
    rest, slope, hard = web[0], web[1], web[3]
    m = 0
    for i in range(k):
        for j in range(i + 1, k):
            a, b = digits[i], digits[j]
            opening = rest[a, b] + slope[a, b] * (vals[b - 1, 0] - vals[a - 1, 0])
            out[m] = max(0., opening - hard[a, b]) / sc[12]
            m += 1
    pt, rr = np.empty(3), np.empty(3)
    for i in range(k):
        d = digits[i] - 1
        n = skin_local(pad, d, mats, loc)
        to_world(palm, loc, n, world)
        contact_point(world, n, sc, pt)
        region(table, sc, pitches[i], depths[i], pt, rr)
        for j in range(3):
            out[m] = rr[j] / sc[0]
            m += 1
        if clear:
            n = skin_local(clearskin, d, mats, loc)
            to_world(palm, loc, n, world)
            for p in range(n):
                out[m] = penetration(H, dtable, world[p], pitches[i], stretch) / clear_scale
                m += 1
    for j in range(5):
        out[m] = prior * (x[j] - prior_wrist[j]) / wrist_scale[j]
        m += 1
    if radial[9] > 0.:
        # the wrist's lateral-bend wall (coherence.RADIAL_CAP / ULNAR_CAP), the pose solve's share
        p0, p1, p2 = palm[0, 1], palm[1, 1], palm[2, 1]
        f0, f1, f2 = radial[0], radial[1], radial[2]
        pdot = p0 * f0 + p1 * f1 + p2 * f2
        pp0, pp1, pp2 = p0 - pdot * f0, p1 - pdot * f1, p2 - pdot * f2
        tdot = pp0 * radial[3] + pp1 * radial[4] + pp2 * radial[5]
        dev = math.degrees(math.asin(max(-1., min(1., tdot))))
        out[m] = prior * (max(0., dev - radial[6]) + max(0., -dev - radial[7])) / radial[8]
        m += 1
    for i in range(k):
        for j in range(4):
            out[m] = prior * (vals[digits[i] - 1, j] - shape[i, j]) / value_scale[j]
            m += 1
    return m


@_jit
def lm_pose(H, Q, x0, prior, clear, iterations, cap):
    """PoseSolver._levenberg_marquardt on PoseSolver.solve's residuals."""
    lower, upper, steps, sc = Q[4], Q[5], Q[11], H[9]
    n = len(x0)
    vals = np.empty((5, 4))
    mats = np.empty((15 + H[3].shape[0], 4, 4))
    work = np.empty((5, 4, 4))
    loc = np.empty((cap, 4))
    world = np.empty((cap, 3))
    buf = np.empty(16384)
    C = new_cache()
    x = x0.astype(np.float32)
    lo, up, st = lower.astype(np.float32), upper.astype(np.float32), steps.astype(np.float32)
    m = pose_residuals(H, Q, x.astype(np.float64), prior, clear, vals, mats, work, loc, world, buf, C)
    r = buf[:m].astype(np.float32)
    cost = np.float32(np.dot(r, r))
    lam = 1e-2
    jac = np.empty((m, n), np.float32)
    for it in range(1, iterations + 1):
        for j in range(n):
            trial = x.copy()
            trial[j] += st[j]
            pose_residuals(H, Q, trial.astype(np.float64), prior, clear, vals, mats, work, loc, world, buf, C)
            for i in range(m):
                jac[i, j] = (np.float32(buf[i]) - r[i]) / st[j]
        g, h = normal_bound(jac, r, x, lo, up)
        improved = False
        delta = np.zeros(n, np.float32)
        for _ in range(10):
            delta = damped(h, g, lam)
            trial = np.minimum(np.maximum(x + delta, lo), up)
            pose_residuals(H, Q, trial.astype(np.float64), prior, clear, vals, mats, work, loc, world, buf, C)
            tr = buf[:m].astype(np.float32)
            tc = np.float32(np.dot(tr, tr))
            if tc < cost:
                x, r, cost, lam, improved = trial, tr, tc, max(lam / 3, 1e-7), True
                break
            lam *= 4
        if not improved or np.max(np.abs(delta)) < sc[23] or cost < 1e-6:
            return x, it
    return x, iterations


@_jit
def pose_cost(H, Q, x, prior, clear, cap):
    vals = np.empty((5, 4))
    mats = np.empty((15 + H[3].shape[0], 4, 4))
    buf = np.empty(16384)
    m = pose_residuals(H, Q, x.astype(np.float64), prior, clear, vals, mats, np.empty((5, 4, 4)), np.empty((cap, 4)),
                       np.empty((cap, 3)), buf, new_cache())
    r = buf[:m].astype(np.float32)
    return np.float32(np.dot(r, r))


@_jit
def pose_one(H, Q):
    """PoseSolver.solve's seeds and phases: each seed reaches the keys, then adds clearance; the best is finished,
    then relaxed toward the prior."""
    x0, lower, upper, seeds, seed_it, iterations, clearance = Q[3], Q[4], Q[5], Q[12], Q[13], Q[14], Q[15]
    cap = max(H[1][3][:, 0].max(), H[2][3][:, 0].max())
    lo, up = lower.astype(np.float32), upper.astype(np.float32)
    it = 0
    best_cost = np.float32(np.inf)
    best = x0.astype(np.float32)
    for s in range(seeds.shape[0]):
        start = x0.astype(np.float32)
        start[0] += np.float32(seeds[s, 1])
        start[1] += np.float32(seeds[s, 2])
        start[3] = np.float32(seeds[s, 0])
        start = np.minimum(np.maximum(start, lo), up)
        trial, used = lm_pose(H, Q, start, .03, False, seed_it, cap)
        more = 0
        if clearance:
            trial, more = lm_pose(H, Q, trial, .03, clearance, seed_it, cap)
        it += used + more
        c = pose_cost(H, Q, trial, .03, clearance, cap)
        if s == 0 or c < best_cost:
            best_cost, best = c, trial
        if c < 1e-4:
            break
    x, used = lm_pose(H, Q, best, .03, clearance, iterations, cap)
    it += used
    x, used = lm_pose(H, Q, x, 1., clearance, iterations // 2, cap)
    return x, it + used


# ---------------------------------------------------------------- FingerIK.ride
@_jit
def ride_residuals(H, R, v, vals, mats, work, loc, world, out, C):
    pad, clearskin, sc = H[1], H[2], H[9]
    d, palm, target, seed, dtable, own, start, has_start, brush, front, scales = (
        R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8], R[9], R[13])
    vals[:, :] = R[14]
    for j in range(4):
        vals[d, j] = v[j]
    for f in range(5):
        finger_kept(H[0], f, vals[f], mats, work, C)
    const = H[3]
    for c in range(const.shape[0]):
        mats[15 + c] = const[c]
    pt = np.empty(3)
    n = skin_local(pad, d, mats, loc)
    to_world(palm, loc, n, world)
    contact_point(world, n, sc, pt)
    off0, off1, off2 = pt[0] - target[0], pt[1] - target[1], pt[2] - target[2]
    m = 0
    out[0] = off0 / .0002
    out[1] = off1 / (scales[1] if off1 < 0. else .0002)
    out[2] = off2 / .0002
    out[3] = max(0., front - pt[1]) / .0002
    m = 4
    for j in range(4):
        out[m] = (v[j] - seed[j]) / scales[0]
        m += 1
    n = skin_local(clearskin, d, mats, loc)
    to_world(palm, loc, n, world)
    for p in range(n):
        z, soft = surface(H[6], sc, dtable, world[p], own)
        pen = max(0., z - world[p, 2])
        if brush:
            brushed = max(pen - sc[20], 0.) + sc[21] * min(max(pen, 0.), sc[20])
            pen = soft * brushed + (1 - soft) * pen
        out[m] = pen / sc[7]
        m += 1
    if d == 0:
        out[m] = 0.
    else:
        last = mats[3 * d + 2]
        ax0 = palm[0, 0] * last[0, 1] + palm[0, 1] * last[1, 1] + palm[0, 2] * last[2, 1]
        ax1 = palm[1, 0] * last[0, 1] + palm[1, 1] * last[1, 1] + palm[1, 2] * last[2, 1]
        ax2 = palm[2, 0] * last[0, 1] + palm[2, 1] * last[1, 1] + palm[2, 2] * last[2, 1]
        f0, f1 = palm[0, 1], palm[1, 1]
        fn = math.sqrt(f0 * f0 + f1 * f1)
        pitch = math.degrees(math.atan2(-ax2, (ax0 * f0 + ax1 * f1) / fn))
        out[m] = max(0., pitch - sc[4]) / sc[5]
    m += 1
    if has_start:
        for j in range(4):
            out[m] = (v[j] - start[j]) / scales[2]
            m += 1
    # Posture: a finger has four joints for a three-dimensional target, so one direction is free. Spent
    # unguarded it drifts to a joint stop (the middle joint folded to its 8 deg limit and then held there by
    # the steady term for the whole note). These are hinges on that free direction only: nothing inside the
    # playing range, a growing cost outside it -- the pad's own position is untouched where it matters.
    for j in range(1, 4):
        lo, hi = scales[4 + 2 * (j - 1)], scales[5 + 2 * (j - 1)]
        out[m] = max(lo - v[j], v[j] - hi, 0.) / scales[3]
        m += 1
    return m


@_jit
def ride_lm(H, R):
    """PoseSolver._levenberg_marquardt on FingerIK.ride's residuals from ``start`` (or the seed)."""
    x0, lower, upper, iterations, sc = R[15], R[10], R[11], R[12], H[9]
    cap = max(H[1][3][:, 0].max(), H[2][3][:, 0].max())
    vals = np.empty((5, 4))
    mats = np.empty((15 + H[3].shape[0], 4, 4))
    work = np.empty((5, 4, 4))
    loc = np.empty((cap, 4))
    world = np.empty((cap, 3))
    buf = np.empty(4096)
    C = new_cache()
    lo, up = lower.astype(np.float32), upper.astype(np.float32)
    x = np.minimum(np.maximum(x0.astype(np.float32), lo), up)
    st = np.full(4, 1e-2, np.float32)
    m = ride_residuals(H, R, x.astype(np.float64), vals, mats, work, loc, world, buf, C)
    r = buf[:m].astype(np.float32)
    cost = np.float32(np.dot(r, r))
    lam = 1e-2
    jac = np.empty((m, 4), np.float32)
    for it in range(1, iterations + 1):
        for j in range(4):
            trial = x.copy()
            trial[j] += st[j]
            ride_residuals(H, R, trial.astype(np.float64), vals, mats, work, loc, world, buf, C)
            for i in range(m):
                jac[i, j] = (np.float32(buf[i]) - r[i]) / st[j]
        g, h = normal_bound(jac, r, x, lo, up)
        improved = False
        delta = np.zeros(4, np.float32)
        for _ in range(10):
            delta = damped(h, g, lam)
            trial = np.minimum(np.maximum(x + delta, lo), up)
            ride_residuals(H, R, trial.astype(np.float64), vals, mats, work, loc, world, buf, C)
            tr = buf[:m].astype(np.float32)
            tc = np.float32(np.dot(tr, tr))
            if tc < cost:
                x, r, cost, lam, improved = trial, tr, tc, max(lam / 3, 1e-7), True
                break
            lam *= 4
        if not improved or np.max(np.abs(delta)) < sc[23] or cost < 1e-6:
            return x, it
    return x, iterations
