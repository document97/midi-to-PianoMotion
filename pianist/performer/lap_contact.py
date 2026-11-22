"""Where the resting hands sit on the thighs -- fitted automatically from ANY seated model's mesh & rig.

``rig_export.py`` calls ``fit_lap_poses(arm, skin, rig)`` when exporting ``rig.json``, storing the fitted
per-hand pose under ``rig['hands'][h]['lap_pose']``:

- ``wrist``   (x, y, z) in world metres: centred laterally over the thigh's usable top dome band, pulled
              behind ``piano_clearance.front_y`` when needed, and lifted so EVERY vertex of the palm, thumb
              and all four fingers clears the thigh surface (zero penetration, never median-based).
- ``yaw``     aligned along the thigh's ``hip -> knee`` axis with a slight natural inward convergence.
- ``pitch``   balances palm-heel clearance against the curved index/middle fingertips so the hand forms a
              natural relaxed arch over the thigh's slope.
- ``roll``    balances inner (index) vs outer (ring/pinky) fingers across the thigh's transverse dome.
- ``fingers`` natural relaxed curved cascade across all five digits (including a gently curved thumb that
              stays open beside the index and 100% above the thigh), strictly clamped to ``[lower, upper]``
              so baked frames match the fitted pose verbatim.

``place(rest, hand, rig)``, ``pitch(hand, rig)`` and ``shape(hand, rig)`` read ``lap_pose`` directly from
``rig.data`` whenever present, falling back to the legacy module tables only for older ``rig.json`` files.
"""

import numpy as np

# Natural relaxed resting hand cascade (sideways, lift, knuckle/mcp, middle/ip in deg), strictly clamped to
# each model's own [finger.lower, finger.upper] during fitting.
BASE_CURL = {
    1: (0.0, -8.0, 18.0, 14.0),   # thumb: gently curved, lifted cleanly onto the thigh
    2: (0.0, -2.0, 24.0, 20.0),   # index: gentle relaxed arch
    3: (0.0, -1.0, 28.0, 22.0),   # middle
    4: (0.0,  0.0, 33.0, 25.0),   # ring
    5: (0.0,  2.0, 38.0, 28.0),   # pinky: most relaxed curl
}

CLEAR_MM = 1.5                    # mm minimum skin clearance above the thigh surface (zero penetration)

# Legacy fallback tables for older rig.json files exported before ``lap_pose`` was embedded.
OFFSET = {'L': (-4.8, -28.9, 49.9), 'R': (1.3, -28.9, 48.9)}
YAW = {'L': -3.6, 'R': 3.6}
ROLL = {'L': -7.6, 'R': 5.4}
PITCH = {'L': -4.2, 'R': -3.9}
FINGERS = {
    'L': {
        1: (-6.0, -22.0, 26.0, 20.3),
        2: (0.0, -3.8, 24.0, 20.0),
        3: (0.0, -2.0, 28.0, 22.0),
        4: (0.0, -0.2, 33.0, 25.0),
        5: (0.0, -2.0, 38.0, 28.0),
    },
    'R': {
        1: (-3.0, -10.0, 26.0, 20.3),
        2: (0.0, -3.8, 24.0, 20.0),
        3: (0.0, -2.0, 28.0, 22.0),
        4: (0.0, 1.6, 33.0, 25.0),
        5: (0.0, -2.0, 38.0, 28.0),
    },
}


def _pose_of(rig, hand):
    if rig is None:
        return None
    data = getattr(rig, 'data', rig)
    if isinstance(data, dict):
        return (data.get('hands') or {}).get(hand, {}).get('lap_pose')
    return None


def place(rest, hand, rig=None):
    """Move a hand's rest pose (x, y, z, yaw, roll) onto the fitted place, in place."""
    lp = _pose_of(rig, hand)
    if lp is not None:
        rest[hand][:3] = np.array(lp['wrist'], dtype=float)
        rest[hand][3] = float(lp['yaw'])
        rest[hand][4] = float(lp['roll'])
        return
    rest[hand][:3] += np.array(OFFSET[hand], dtype=float) * .001
    rest[hand][3] = YAW[hand]
    rest[hand][4] = ROLL[hand]


def pitch(hand, rig=None):
    """The resting palm's pitch (deg): fitted to this body's thigh slope and relaxed hand arch."""
    lp = _pose_of(rig, hand)
    if lp is not None:
        return float(lp['pitch'])
    return PITCH[hand]


def shape(hand, rig=None):
    """The resting finger shape: {digit: (sideways, lift, knuckle, middle)}."""
    lp = _pose_of(rig, hand)
    if lp is not None:
        raw = lp['fingers']
        return {int(d): np.array(raw[str(d)] if str(d) in raw else raw[int(d)], dtype=float) for d in (1, 2, 3, 4, 5)}
    return {d: np.array(v, dtype=float) for d, v in FINGERS[hand].items()}


def _scan1d(fn, lo, hi, n=13, refine=10, default=None):
    xs = np.linspace(lo, hi, n)
    fs = [fn(float(x)) for x in xs]
    i = int(np.argmin(fs))
    if default is not None and fs[i] >= 999.0:
        return float(default)
    a, b = float(xs[max(0, i - 1)]), float(xs[min(n - 1, i + 1)])
    phi = 0.5 * (3.0 - np.sqrt(5.0))
    c = a + phi * (b - a)
    d = b - phi * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(refine):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = a + phi * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = b - phi * (b - a)
            fd = fn(d)
    return c if fc <= fd else d


def fit_lap_poses(arm, skin, rig):
    """Fit natural, zero-penetration, gently-curved resting poses for both hands on ``arm``/``skin``.

    Runs inside Blender during ``rig_export.py`` on the seated character's neutral pose. Uses only the
    evaluated mesh and ``rig`` kinematics -- no model-specific constants -- and guarantees all finger values
    stay within ``[finger.lower, finger.upper]`` and all palm/thumb/finger skin vertices clear the thigh.
    """
    import bpy
    from mathutils import Vector
    from .arm import LAP_PITCH
    from .kinematics import DIGITS, SkinSet
    from .posture import natural_orientation
    from .regions import performer_solvers

    solvers = performer_solvers(rig)
    deps = bpy.context.evaluated_depsgraph_get()
    ev = skin.evaluated_get(deps)
    mw = ev.matrix_world
    inv = mw.inverted()
    names = {g.index: g.name for g in skin.vertex_groups}

    # Collect all bones carried by either arm so thigh raycasts cleanly skip arms/sleeves/hands.
    all_arm = set()
    for h in 'LR':
        ub = rig.data['hands'][h]['bones']['upper']
        if ub in arm.data.bones:
            u = arm.data.bones[ub]
            all_arm.update(b.name for b in [u] + list(u.children_recursive))

    polys = skin.data.polygons
    verts = skin.data.vertices
    arm_poly = np.zeros(len(polys), bool)
    for pi, p in enumerate(polys):
        w = 0.
        for vi in p.vertices:
            for g in verts[vi].groups:
                if names.get(g.group, '') in all_arm:
                    w += g.weight
        arm_poly[pi] = w > 0.5

    def raw_leg_z(x, y, z_top):
        org = inv @ Vector((float(x), float(y), float(z_top)))
        d_raw = inv.to_3x3() @ Vector((0., 0., -1.))
        scale = max(float(d_raw.length), 1e-9)
        d = d_raw / scale
        max_dist = 3.0 * scale
        eps = 1e-4 * scale
        for _ in range(16):
            hit, loc, _, pi = ev.ray_cast(org, d, distance=max_dist)
            if not hit:
                return None
            if pi < len(arm_poly) and not arm_poly[pi]:
                return float((mw @ loc).z)
            org = loc + d * eps
        return None

    mesh_ev = ev.to_mesh()
    arm_world = np.array(arm.matrix_world, float)
    arm_inv = np.linalg.inv(arm_world)

    def build_hand_skin(h):
        wb = rig.wrist_bone[h]
        finger_bones = {d: list(rig.fingers[h][d].names) for d in DIGITS}
        bone_to_part = {wb: 'palm'}
        for d, bnames in finger_bones.items():
            for b in bnames:
                bone_to_part[b] = d
        nb_inv = {b: np.linalg.inv(np.array(arm.pose.bones[b].matrix, float)) for b in bone_to_part}
        rows = {'palm': [], 1: [], 2: [], 3: [], 4: [], 5: []}
        tip_rows = {1: [], 2: [], 3: [], 4: [], 5: []}
        for v in verts:
            bw = {}
            dom_b, dom_w = None, 0.
            for g in v.groups:
                nm = names.get(g.group, '')
                if nm in bone_to_part and g.weight > 1e-4:
                    bw[nm] = g.weight
                    if g.weight > dom_w:
                        dom_b, dom_w = nm, g.weight
            if dom_b is None or dom_w < 0.25 or sum(bw.values()) < 0.5:
                continue
            p_world = np.array(mw @ mesh_ev.vertices[v.index].co, float)
            p_arm = (arm_inv @ np.r_[p_world, 1.])[:3]
            sw = sum(bw.values())
            entry = {'bones': {b: [w / sw, *(nb_inv[b] @ np.r_[p_arm, 1.])[:3].tolist()] for b, w in bw.items()}}
            part = bone_to_part[dom_b]
            rows[part].append(entry)
            if isinstance(part, int) and dom_b == finger_bones[part][-1]:
                tip_rows[part].append(entry)
        return ({k: SkinSet.from_rows(r) for k, r in rows.items() if r},
                {k: SkinSet.from_rows(r) for k, r in tip_rows.items() if r})

    def eval_hand_world(h, M_wrist, fvals, sets):
        mats = {rig.wrist_bone[h]: M_wrist}
        for d in DIGITS:
            f = rig.fingers[h][d]
            lms = f.local_matrices(fvals[d])
            for b, lm in zip(f.names, lms):
                mats[b] = M_wrist @ lm
        return {k: ss.points(mats, rig.arm_world) for k, ss in sets.items()}

    def clip_vals(h, d, v):
        f = rig.fingers[h][d]
        return np.clip(np.asarray(v, float), f.lower, f.upper)

    pc = rig.data.get('piano_clearance') or {}
    front_y = pc.get('front_y')
    side_prefix = {'L': '左', 'R': '右'}
    out = {}

    try:
        for h in 'LR':
            s = side_prefix[h]
            solver = solvers[h]
            sets, tips = build_hand_skin(h)
            lap = np.array(rig.data['hands'][h]['lap'], float)
            hip_bone = arm.pose.bones[s + '足']
            hip = np.array(arm.matrix_world @ hip_bone.head, float)
            knee = np.array(arm.matrix_world @ hip_bone.tail, float)
            axis_xy = (knee - hip)[:2]
            axis_xy /= np.linalg.norm(axis_xy)
            # Start rays ~12 cm above the hip/knee bone axis: always cleanly above the thigh top and below any
            # chest/coat overhang, even after place_feet pitches the thighs down.
            z_top = max(hip[2], knee[2]) + 0.12

            # Cached 4 mm thigh heightfield for this leg; filter out bench/floor hits (>70 mm below crest).
            step = 0.004
            y_min, y_max = min(hip[1], knee[1]) - 0.05, max(hip[1], knee[1]) + 0.08
            x_center = 0.5 * (hip[0] + knee[0])
            x_min, x_max = x_center - 0.16, x_center + 0.16
            gx = np.arange(x_min, x_max + 1e-6, step)
            gy = np.arange(y_min, y_max + 1e-6, step)
            gz = np.full((len(gx), len(gy)), np.nan)
            for ix, x in enumerate(gx):
                for iy, y in enumerate(gy):
                    z = raw_leg_z(x, y, z_top)
                    if z is not None:
                        gz[ix, iy] = z
            for iy in range(len(gy)):
                col = gz[:, iy]
                if np.isfinite(col).any():
                    gz[col < float(np.nanmax(col)) - 0.070, iy] = np.nan

            def leg_z_batch(pts):
                ix = np.rint((pts[:, 0] - x_min) / step).astype(int)
                iy = np.rint((pts[:, 1] - y_min) / step).astype(int)
                ok = (ix >= 0) & (ix < len(gx)) & (iy >= 0) & (iy < len(gy))
                res = np.full(len(pts), np.nan)
                res[ok] = gz[ix[ok], iy[ok]]
                return res

            hip_l = np.array(arm.matrix_world @ arm.pose.bones['左足'].head, float)
            hip_r = np.array(arm.matrix_world @ arm.pose.bones['右足'].head, float)
            mid_x = 0.5 * (hip_l[0] + hip_r[0])

            def thigh_band(x0, y0, drop=0.025):
                iy = int(np.clip(round((y0 - y_min) / step), 0, len(gy) - 1))
                col = gz[:, iy]
                if not np.isfinite(col).any():
                    return x0, 0.04, lap[2]
                valid = np.flatnonzero(np.isfinite(col))
                sgn = 1.0 if hip[0] >= mid_x else -1.0
                same_leg = valid[(gx[valid] - mid_x) * sgn >= -0.002]
                if len(same_leg) == 0:
                    same_leg = valid
                crest_z = float(np.max(col[same_leg]))
                top_set = set(same_leg[col[same_leg] >= crest_z - drop].tolist())
                peak = same_leg[int(np.argmax(col[same_leg]))]
                lo = hi = peak
                while (lo - 1) in top_set:
                    lo -= 1
                while (hi + 1) in top_set:
                    hi += 1
                return 0.5 * (gx[lo] + gx[hi]), 0.5 * (gx[hi] - gx[lo]), crest_z

            def gaps_of(pts):
                lz = leg_z_batch(pts)
                ok = np.isfinite(lz)
                return (pts[ok, 2] - lz[ok]) * 1000.0 if ok.any() else np.zeros(0)

            wrist = lap.copy()
            shoulder = np.asarray(solver.arm.pre_upper_world[:3, 3], float)
            yaw, roll = [float(v) for v in natural_orientation(solver, wrist, shoulder, keys=False)]
            pitch_deg = float(LAP_PITCH)
            fvals = {d: clip_vals(h, d, BASE_CURL[d]) for d in DIGITS}
            inward_sign = 1.0 if h == 'L' else -1.0

            def M_now():
                return rig.arms[h].arm_inverse @ solver.palm(wrist, yaw, roll, pitch_deg)

            # 1. Yaw along hip->knee with a slight natural inward convergence.
            for _ in range(3):
                pts = eval_hand_world(h, M_now(), fvals, sets)
                fwd = pts[3].mean(axis=0) - wrist
                dyaw = float(np.degrees(np.arctan2(axis_xy[1], axis_xy[0]) - np.arctan2(fwd[1], fwd[0])))
                yaw += ((dyaw + 180.0) % 360.0) - 180.0
            yaw += (3.0 if h == 'R' else -3.0)

            # 2. Stay cleanly behind the piano keybed front edge (>= 25 mm margin).
            if front_y is not None:
                pts = eval_hand_world(h, M_now(), fvals, sets)
                max_y = max(float(pts[d][:, 1].max()) for d in DIGITS)
                if max_y > front_y - 0.025:
                    wrist[1] -= (max_y - (front_y - 0.025))

            # 3. Centre the full hand laterally over the thigh's usable top dome band.
            for _ in range(3):
                pts = eval_hand_world(h, M_now(), fvals, sets)
                all_xy = np.vstack([pts[d] for d in DIGITS])
                cx = 0.5 * (all_xy[:, 0].min() + all_xy[:, 0].max())
                cy = float(all_xy[:, 1].mean())
                bcx, _, _ = thigh_band(cx, cy, drop=0.025)
                wrist[0] += bcx - cx

            # 4. Pitch: match palm-heel clearance to curved index/middle fingertips (natural arch).
            def pitch_err(p):
                M = rig.arms[h].arm_inverse @ solver.palm(wrist, yaw, roll, p)
                pts_p = eval_hand_world(h, M, fvals, {'palm': sets['palm'], 2: tips[2], 3: tips[3]})
                gp, g2, g3 = gaps_of(pts_p['palm']), gaps_of(pts_p[2]), gaps_of(pts_p[3])
                if not (len(gp) and len(g2) and len(g3)):
                    return 1e3
                return abs(float(gp.min()) - 0.5 * (float(g2.min()) + float(g3.min())))

            pitch_deg = _scan1d(pitch_err, -25.0, 15.0, n=21, refine=10, default=pitch_deg)

            # 5. Roll: balance inner (index) vs outer (ring/pinky) fingers across the thigh dome.
            def roll_err(r):
                M = rig.arms[h].arm_inverse @ solver.palm(wrist, yaw, r, pitch_deg)
                pts_r = eval_hand_world(h, M, fvals, {2: tips[2], 4: tips[4], 5: tips[5]})
                g2, g4, g5 = gaps_of(pts_r[2]), gaps_of(pts_r[4]), gaps_of(pts_r[5])
                if not (len(g2) and len(g4) and len(g5)):
                    return 1e3
                return abs(float(g2.min()) - 0.5 * (float(g4.min()) + float(g5.min())))

            roll = _scan1d(roll_err, roll - 25.0, roll + 25.0, n=21, refine=10, default=roll)

            # 6. Zero-penetration height + per-finger draping + curved open thumb (2 rounds).
            for _ in range(2):
                for _ in range(3):
                    pts = eval_hand_world(h, M_now(), fvals, sets)
                    mins = [float(gaps_of(pts[k]).min()) for k in ('palm', 2, 3, 4, 5) if len(gaps_of(pts[k]))]
                    if not mins:
                        break
                    wrist[2] -= (min(mins) - CLEAR_MM) * 0.001

                M = M_now()
                for d in (2, 3, 4, 5):
                    f = rig.fingers[h][d]
                    bv = clip_vals(h, d, BASE_CURL[d])

                    def f_long(lf, d=d, bv=bv):
                        v = clip_vals(h, d, [bv[0], lf, bv[2], bv[3]])
                        g = gaps_of(eval_hand_world(h, M, {**fvals, d: v}, {d: sets[d]})[d])
                        if not len(g):
                            return 1e3
                        mn = float(g.min())
                        return 12.0 * max(0., CLEAR_MM - mn) + abs(mn - CLEAR_MM) + 0.08 * abs(lf - bv[1])

                    best_lf = _scan1d(f_long, max(float(f.lower[1]), -20.0), min(float(f.upper[1]), 16.0), n=21, refine=8)
                    fvals[d] = clip_vals(h, d, [bv[0], best_lf, bv[2], bv[3]])

                # Thumb: gently curved, naturally open beside index (never tucked under palm), zero penetration.
                f1 = rig.fingers[h][1]
                pts2 = eval_hand_world(h, M, fvals, {2: tips[2]})[2]
                x2_mean = float(pts2[:, 0].mean())
                best_cost, best_v1 = 1e9, clip_vals(h, 1, BASE_CURL[1])
                for sd in np.linspace(-6.0, 6.0, 5):
                    for curl in (14.0, 20.0, 26.0):
                        pip = curl * 0.78
                        for lf in np.linspace(max(float(f1.lower[1]), -28.0), min(float(f1.upper[1]), 6.0), 18):
                            v = clip_vals(h, 1, [sd, lf, curl, pip])
                            pts1 = eval_hand_world(h, M, {**fvals, 1: v}, {1: sets[1], 't1': tips[1]})
                            inward_sep = (float(pts1['t1'][:, 0].mean()) - x2_mean) * inward_sign * 1000.0
                            if inward_sep < 22.0 or inward_sep > 44.0:
                                continue
                            g_all = gaps_of(pts1[1])
                            g_tip = gaps_of(pts1['t1'])
                            if not len(g_all):
                                continue
                            mn = float(g_all.min())
                            tip_mn = float(g_tip.min()) if len(g_tip) else mn
                            cost = (25.0 * max(0., CLEAR_MM - mn)
                                    + 1.5 * abs(tip_mn - (CLEAR_MM + 0.5))
                                    + 0.15 * abs(sd)
                                    + 0.04 * abs(lf + 10.0)
                                    + 0.04 * abs(curl - 20.0))
                            if cost < best_cost:
                                best_cost, best_v1 = cost, v
                fvals[1] = best_v1

                g1 = gaps_of(eval_hand_world(h, M_now(), fvals, {1: sets[1]})[1])
                if len(g1) and float(g1.min()) < 1.0:
                    wrist[2] += (CLEAR_MM - float(g1.min())) * 0.001

            # Verify arm IK reachability so resting frames never trigger stretched-arm fallback.
            arm_h = rig.arms[h]
            max_reach = 0.96 * (arm_h.l1 + arm_h.l2)
            diff = wrist - shoulder
            dist = float(np.linalg.norm(diff))
            if dist > max_reach and dist > 1e-6:
                wrist = shoulder + diff * (max_reach / dist)
                pts_r = eval_hand_world(h, M_now(), fvals, sets)
                all_g = np.concatenate([gaps_of(pts_r[k]) for k in ('palm', 1, 2, 3, 4, 5)])
                if len(all_g) and float(all_g.min()) < CLEAR_MM:
                    wrist[2] += (CLEAR_MM - float(all_g.min())) * 0.001

            # Log final mesh clearances for transparency.
            pts = eval_hand_world(h, M_now(), fvals, sets)
            mins_mm = {str(k): round(float(gaps_of(pts[k]).min()), 1) for k in ('palm', 1, 2, 3, 4, 5)
                       if len(gaps_of(pts[k]))}
            print(f'[lap fit] {h} wrist={np.round(wrist * 1000, 1).tolist()} yaw={yaw:.1f} '
                  f'roll={roll:.1f} pitch={pitch_deg:.1f} min_gap_mm={mins_mm}', flush=True)
            out[h] = dict(
                wrist=[round(float(x), 6) for x in wrist],
                yaw=round(float(yaw), 2),
                roll=round(float(roll), 2),
                pitch=round(float(pitch_deg), 2),
                fingers={str(d): [round(float(x), 2) for x in fvals[d]] for d in DIGITS},
            )
    finally:
        ev.to_mesh_clear()
    return out
