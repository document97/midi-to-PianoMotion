"""Universal physiological gaze, head-eye coordination, and blinking for any rigged pianist.

Works purely from rig geometry exported by rig_export.py (head bone, left/right eye bones,
and optional blink shape key) without character-specific hardcoding:
  - Builds a pianistic visual-attention trajectory (anticipatory look-ahead before leaps,
    salience-based hand commitment with dwell/hysteresis, calm intro/outro transitions,
    and gentle upward drift during long rests).
  - Splits gaze shifts between a slower partial head turn (head bone) and fast binocular
    eye saccades with exact vestibulo-ocular reflex (VOR) compensation.
  - Uses exact bone-frame conjugation (pre_inverse @ R_local @ world), so any rig's bone
    roll, outward eye axis offset, or coordinate convention maps accurately to world yaw/pitch.
  - Generates natural asymmetric blinking biased toward large saccades and phrase rests
    (written to the blink shape key in Blender when the mesh provides one).
"""
import math
import numpy as np

from .kinematics import FLOAT, axis_matrix, mat3_q
from .torso import HEAD_STEADY, _about, _gauss_time, _half, spine_transforms

# ---------- Attention & fixation parameters ----------
LOOKAHEAD_S = 0.34          # s eyes anticipate an upcoming leap or hand entry
LEAP_DIST_M = 0.12          # m horizontal shift that triggers an anticipatory gaze shift
CLOSE_SPAN_M = 0.42         # m: when hands are within this span, gaze stays calmly near their center
MIN_DWELL_S = 2.40          # s minimum hold before switching fixation between hands (calm, no darting)
SWITCH_MARGIN = 0.38        # salience margin needed to switch fixation to the other hand
SACCADE_TAU_S = 0.22        # s Gaussian smoothing for calm attention transitions
SACCADE_MAX_DEG_S = 45.0    # deg/s angular speed cap on gaze transitions
REST_DRIFT_S = 1.8          # s of musical silence before eyes drift softly upward/forward

# ---------- Head-eye coordination parameters ----------
# Subtle, restrained head & eyeball motion: gentle multi-second head follow (+/-4.5 deg yaw,
# -5..+1.5 deg pitch) paired with delicate eyeball micro-glances (+/-4.2 deg yaw, -2.2..+1.5 deg
# pitch) so the character feels alive without busy head bobbing or extreme eye rolling.
HEAD_YAW_SHARE = 0.14       # fraction of horizontal gaze yaw carried by the head bone
HEAD_PITCH_SHARE = 0.12     # fraction of vertical gaze pitch carried by the head bone
HEAD_ROLL_PER_YAW = 0.06    # subtle natural head tilt accompanying lateral gaze
HEAD_YAW_LIMIT = 4.5        # deg max head yaw (+ left / - right)
HEAD_PITCH_MIN = -5.0       # deg max downward head nod
HEAD_PITCH_MAX = 1.5        # deg max upward head tilt
HEAD_ROLL_LIMIT = 1.0       # deg max expressive head roll
HEAD_TAU_S = 0.90           # s head smoothing time constant (slow, calm phrase-level motion)
HEAD_MAX_DEG_S = 8.0        # deg/s head angular velocity cap

# ---------- Physiological / stylized-safe eyeball limits ----------
EYE_YAW_SHARE = 0.24        # horizontal residual yaw scale (delicate micro-gaze, never corner-rolling)
EYE_YAW_LIMIT = 4.2         # deg max horizontal eye turn inside head socket
EYE_PITCH_SHARE = 0.08      # vertical residual pitch scale
EYE_PITCH_MIN = -2.2        # deg max downward eye pitch inside head socket
EYE_PITCH_MAX = 1.5         # deg max upward eye pitch inside head socket
EYE_TAU_S = 0.16            # s eye smoothing time constant

# ---------- Natural blinking parameters ----------
BLINK_MEAN_S = 3.9          # average interval between blinks (~15 blinks/min)
BLINK_MIN_S = 1.8           # refractory period between blinks (except rare double-blinks)
BLINK_MAX_S = 6.2           # maximum interval before forcing a blink
BLINK_CLOSE_S = 0.065       # fast upper-eyelid closure
BLINK_HOLD_S = 0.025        # brief closed dwell
BLINK_OPEN_S = 0.135        # smoother eyelid opening
BLINK_SACCADE_DEG = 8.5     # gaze shift (deg) that biases a blink to occur


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def _rate_limit(series, dt, max_rate):
    out = np.array(series, dtype=float, copy=True)
    step = max_rate * dt
    for i in range(1, len(out)):
        d = out[i] - out[i - 1]
        if abs(d) > step:
            out[i] = out[i - 1] + math.copysign(step, d)
    return out


def _smooth_series(values, dt, tau, max_rate=None):
    if len(values) < 3:
        return np.array(values, dtype=float, copy=True)
    out = _gauss_time(np.asarray(values, dtype=float), dt, max(tau, dt))
    if max_rate is not None:
        out = _rate_limit(out, dt, max_rate)
        # light post-pass keeps C1 continuity after rate clamping
        out = _gauss_time(out, dt, max(dt * 1.5, tau * 0.25))
    return out


def _face_basis(eyes_cfg):
    """Orthonormal (right, forward, up) basis of the character's neutral face in world space."""
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    left_e = eyes_cfg.get('L')
    right_e = eyes_cfg.get('R')
    if left_e is not None and right_e is not None:
        lr = np.asarray(right_e['head'], dtype=float) - np.asarray(left_e['head'], dtype=float)
        lr[2] = 0.0
        n = float(np.linalg.norm(lr))
        if n > 1e-4:
            right = lr / n
            forward = np.cross(up, right)
            forward /= max(float(np.linalg.norm(forward)), 1e-9)
            return right, forward, up
    return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), up


def _neck_world_transforms(times, torso):
    """Per-frame (N, 4, 4) rigid world delta of the neck bone (matching Torso.pose exactly)."""
    n = len(times)
    out = np.broadcast_to(np.eye(4, dtype=float), (n, 4, 4)).copy()
    if torso is None:
        return out
    heads = [b['head'] for b in torso.spine]
    neck_head = torso.neck['head']
    for i, t in enumerate(times):
        t = float(t)
        cap = torso._cap(t)
        shift, turn, offset = torso._lean(t, cap, smooth=True)
        sway = torso._sway(t)
        chain = spine_transforms(shift, turn, heads, neck_head, torso.height, offset, sway)
        transform = chain[-1]
        lean_only = (spine_transforms(shift, turn, heads, neck_head, torso.height, sway=sway)[-1][:3, :3]
                     @ np.linalg.inv(spine_transforms(0.0, 0.0, heads, neck_head, torso.height, sway=sway)[-1][:3, :3]))
        steady_r = _half(lean_only, HEAD_STEADY).T
        neck_moved = (transform @ np.r_[neck_head, 1.0])[:3]
        out[i] = _about(neck_moved, steady_r) @ transform
    return out


def _key_y_bounds(row):
    if 'front_y' in row and 'back_y' in row:
        return float(row['front_y']), float(row['back_y'])
    if 'bounds_min' in row and 'bounds_max' in row:
        return float(row['bounds_min'][1]), float(row['bounds_max'][1])
    return -0.05, 0.15


def _keyboard_reference(rig):
    """Center (x, y, z) of the keyboard playing surface from rig.json."""
    kb = rig.data.get('keyboard') or {}
    if kb:
        xs, ys, zs = [], [], []
        for row in kb.values():
            if 'center_x' in row and 'top_z' in row:
                fy, by = _key_y_bounds(row)
                xs.append(float(row['center_x']))
                ys.append(0.5 * (fy + by))
                zs.append(float(row['top_z']))
        if xs:
            return np.array([float(np.median(xs)), float(np.median(ys)), float(np.median(zs))], dtype=float)
    return np.array([0.0, 0.0, 0.74], dtype=float)


def _hand_events(notes, rig):
    """Per hand, sorted list of (onset, release, x, y, z, velocity, is_leap, jump) tuples."""
    kb = rig.data.get('keyboard') or {}
    kb_ref = _keyboard_reference(rig)
    per = {'L': [], 'R': []}
    for n in notes.values():
        row = kb.get(str(int(n.pitch))) or kb.get(int(n.pitch))
        if row is not None and 'center_x' in row and 'top_z' in row:
            fy, by = _key_y_bounds(row)
            x = float(row['center_x'])
            y = float(fy + 0.35 * (by - fy))
            z = float(row['top_z'])
        else:
            x, y, z = float(kb_ref[0]), float(kb_ref[1]), float(kb_ref[2])
        vel = float(getattr(n, 'velocity', 80)) / 127.0
        onset = float(getattr(n, 'onset', getattr(n, 'start', 0.0)))
        per[n.hand].append((onset, float(n.release), x, y, z, vel))
    out = {}
    for h in 'LR':
        evs = sorted(per[h], key=lambda r: r[0])
        tagged = []
        prev_x, prev_end = None, -1e9
        for onset, rel, x, y, z, vel in evs:
            gap = onset - prev_end
            jump = abs(x - prev_x) if prev_x is not None else 0.0
            is_leap = (prev_x is None) or (gap >= 0.9) or (jump >= LEAP_DIST_M)
            tagged.append((onset, rel, x, y, z, vel, is_leap, jump))
            prev_x = x
            prev_end = max(prev_end, rel)
        out[h] = tagged
    return out


def compute_attention_targets(times, frames, notes, paths, rig):
    """Compute a natural 3D world-space gaze target (N, 3) for every frame.

    Combines anticipatory leap look-ahead, hand-salience commitment with dwell/hysteresis,
    and calm intro/outro/rest behavior.
    """
    n = len(times)
    dt = float(times[1] - times[0]) if n > 1 else (1.0 / 60.0)
    kb_ref = _keyboard_reference(rig)
    events = _hand_events(notes, rig)

    # Hand wrist positions on the frame grid (already solved in world space)
    w_pos = {h: np.asarray(frames[f'{h}_wrist'][:, :3], dtype=float) for h in 'LR'}
    # Approximate fingertip/key contact point slightly forward (+Y) of the wrist
    tip_pos = {h: w_pos[h] + np.array([0.0, 0.10, -0.03]) for h in 'LR'}

    # Piece timing milestones
    all_onsets = [e[0] for h in 'LR' for e in events[h]]
    all_releases = [e[1] for h in 'LR' for e in events[h]]
    first_note = min(all_onsets) if all_onsets else float(times[0])
    last_note = max(all_releases) if all_releases else float(times[-1])
    leaves = [getattr(paths[h], 'trip_leave', None) for h in 'LR']
    leaves = [float(t) for t in leaves if t is not None]
    first_leave = min(leaves) if leaves else first_note - 1.5
    outros = [getattr(paths[h], 'trip_outro', None) for h in 'LR']
    outros = [float(t) for t in outros if t is not None]
    outro_start = min(outros) if outros else last_note + 0.4
    outro_lands = [getattr(paths[h], 'trip_outro_land', None) for h in 'LR']
    outro_lands = [float(t) for t in outro_lands if t is not None]
    outro_land = max(outro_lands) if outro_lands else float(times[-1])

    # Precompute per-frame salience and look-ahead target for each hand
    salience = {'L': np.zeros(n, dtype=float), 'R': np.zeros(n, dtype=float)}
    hand_target = {h: tip_pos[h].copy() for h in 'LR'}
    active_any = np.zeros(n, dtype=bool)

    for h in 'LR':
        evs = events[h]
        # Slight melodic bias for right hand when both hands are equally busy
        melody_bias = 0.14 if h == 'R' else 0.0
        for onset, rel, x, y, z, vel, is_leap, jump in evs:
            # Activity window
            i0 = max(0, int(np.searchsorted(times, onset - 0.15)))
            i1 = min(n, int(np.searchsorted(times, rel + 0.20)))
            if i1 > i0:
                active_any[i0:i1] = True
                salience[h][i0:i1] += (0.55 + 0.45 * vel + melody_bias)
            # Anticipatory look-ahead window before leaps or re-entries
            if is_leap:
                boost = 1.15 + min(1.0, jump / 0.25)
                j0 = max(0, int(np.searchsorted(times, onset - LOOKAHEAD_S)))
                j1 = min(n, int(np.searchsorted(times, onset + 0.22)))
                if j1 > j0:
                    salience[h][j0:j1] = np.maximum(salience[h][j0:j1], boost + melody_bias)
                    # Blend hand target smoothly onto the upcoming landing key during look-ahead
                    w = _smoothstep((times[j0:j1] - (onset - LOOKAHEAD_S)) / max(LOOKAHEAD_S * 0.5, 1e-3))
                    tgt = np.array([x, y, z], dtype=float)
                    hand_target[h][j0:j1] = (1.0 - w[:, None]) * hand_target[h][j0:j1] + w[:, None] * tgt[None, :]
        # Smooth salience slightly so single 16th notes don't spike instantaneously
        salience[h] = _smooth_series(salience[h], dt, 0.12)

    # First landing target (for anticipatory intro gaze)
    first_evs = sorted((e for h in 'LR' for e in events[h]), key=lambda r: r[0])
    if first_evs:
        early = [e for e in first_evs if e[0] <= first_evs[0][0] + 0.35]
        first_key = np.mean([[e[2], e[3], e[4]] for e in early], axis=0)
    else:
        first_key = kb_ref.copy()

    # Soft "resting/reflective" target: poised focus toward fallboard / music desk before playing & on rests
    intro_rest_tgt = np.array([kb_ref[0], kb_ref[1] + 0.18, kb_ref[2] + 0.24], dtype=float)
    music_desk_tgt = np.array([kb_ref[0], kb_ref[1] + 0.24, kb_ref[2] + 0.30], dtype=float)
    lap_mid = 0.5 * (w_pos['L'][0] + w_pos['R'][0])
    outro_rest_tgt = 0.55 * np.array([kb_ref[0], kb_ref[1] + 0.06, kb_ref[2] + 0.08]) + 0.45 * (lap_mid + np.array([0.0, 0.08, 0.04]))

    # Step through frames with discrete fixation commitment (min dwell + hysteresis)
    # mode in {'BOTH', 'L', 'R', 'REST'}
    raw_target = np.zeros((n, 3), dtype=float)
    mode = 'REST'
    last_switch_t = float(times[0]) - 10.0
    silent_since = float(times[0])
    outro_anchor = None

    for i, t in enumerate(times):
        t = float(t)
        if t < first_leave - 0.15:
            raw_target[i] = intro_rest_tgt
            mode = 'REST'
            continue
        if t < first_note - LOOKAHEAD_S:
            # Transition smoothly from calm keyboard center to the first landing keys
            w = float(_smoothstep((t - (first_leave - 0.15)) / max((first_note - LOOKAHEAD_S) - (first_leave - 0.15), 0.25)))
            raw_target[i] = (1.0 - w) * intro_rest_tgt + w * first_key
            mode = 'BOTH'
            last_switch_t = t
            continue
        if t > outro_start + 0.20:
            w = float(_smoothstep((t - (outro_start + 0.20)) / max(outro_land - (outro_start + 0.20), 0.5)))
            if outro_anchor is None:
                outro_anchor = raw_target[i - 1].copy() if i > 0 else kb_ref.copy()
            raw_target[i] = (1.0 - w) * outro_anchor + w * outro_rest_tgt
            continue

        # Active playing region
        sL, sR = float(salience['L'][i]), float(salience['R'][i])
        if active_any[i] or max(sL, sR) > 0.35:
            silent_since = t
        if t - silent_since > REST_DRIFT_S and max(sL, sR) < 0.25:
            # Long musical pause: drift gently toward expressive soft focus
            w = float(_smoothstep((t - silent_since - REST_DRIFT_S) / 0.8))
            base = 0.5 * (hand_target['L'][i] + hand_target['R'][i])
            raw_target[i] = (1.0 - w) * base + w * music_desk_tgt
            mode = 'REST'
            continue

        span = abs(float(hand_target['R'][i, 0] - hand_target['L'][i, 0]))
        can_switch = (t - last_switch_t) >= MIN_DWELL_S
        # Urgent leap overrides normal dwell slightly (down to 0.32 s)
        urgent = max(sL, sR) >= 1.35 and (t - last_switch_t) >= 0.32

        desired = mode
        if span <= CLOSE_SPAN_M and max(sL, sR) < 1.35:
            desired = 'BOTH'
        else:
            if mode == 'L':
                if sR > sL + SWITCH_MARGIN:
                    desired = 'R'
            elif mode == 'R':
                if sL > sR + SWITCH_MARGIN:
                    desired = 'L'
            else:
                if sR > sL + 0.08:
                    desired = 'R'
                elif sL > sR + 0.08:
                    desired = 'L'
                else:
                    desired = 'BOTH'

        if desired != mode and (can_switch or urgent or mode == 'REST'):
            mode = desired
            last_switch_t = t

        if mode == 'L':
            # Bias 82% toward fixated hand, 18% toward keyboard center so gaze stays balanced
            tgt = 0.82 * hand_target['L'][i] + 0.18 * hand_target['R'][i]
        elif mode == 'R':
            tgt = 0.82 * hand_target['R'][i] + 0.18 * hand_target['L'][i]
        else:
            wL = max(0.15, sL)
            wR = max(0.15, sR)
            tgt = (wL * hand_target['L'][i] + wR * hand_target['R'][i]) / (wL + wR)

        # Keep vertical target anchored near keyboard surface (never chasing high air lifts wildly)
        tgt[2] = float(np.clip(tgt[2], kb_ref[2] - 0.06, kb_ref[2] + 0.08))
        raw_target[i] = tgt

    # Smooth raw attention target into crisp, organic saccades (no 60 Hz jitter)
    smooth_target = np.column_stack([
        _smooth_series(raw_target[:, k], dt, SACCADE_TAU_S) for k in range(3)
    ])
    return smooth_target


def _aim_yaw_pitch(origin, target, right, forward, up):
    """(yaw_deg, pitch_deg) from origin to target in orthonormal (right, forward, up) basis.

    +yaw rotates forward toward -right (character's left / bass side, R_up(+yaw)).
    +pitch rotates forward toward +up (looking up, R_right(+pitch)).
    """
    d = target - origin
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return 0.0, 0.0
    d /= dist
    dx = float(d @ right)
    dy = float(d @ forward)
    dz = float(d @ up)
    yaw = math.degrees(math.atan2(-dx, max(dy, 1e-4)))
    pitch = math.degrees(math.atan2(dz, max(math.hypot(dx, dy), 1e-4)))
    return yaw, pitch


def _generate_blinks(times, gaze_speed_deg_s, notes, seed=0):
    """Natural asymmetric blink curve in [0, 1], biased toward large saccades and phrase rests."""
    n = len(times)
    blink = np.zeros(n, dtype=np.float32)
    if n < 10:
        return blink, []
    rng = np.random.default_rng(seed)
    t0, t1 = float(times[0]), float(times[-1])

    # Identify phrase-boundary/rest bonus (moments with no note onset within +/-0.18 s)
    onsets = np.sort([float(getattr(nt, 'onset', getattr(nt, 'start', 0.0))) for nt in notes.values()]) if notes else np.zeros(0)

    def _near_onset(t):
        if len(onsets) == 0:
            return False
        idx = int(np.searchsorted(onsets, t))
        if idx < len(onsets) and abs(onsets[idx] - t) < 0.14:
            return True
        if idx > 0 and abs(t - onsets[idx - 1]) < 0.14:
            return True
        return False

    blink_times = []
    next_due = t0 + float(rng.uniform(1.0, 2.2))
    last_blink = t0 - BLINK_MIN_S

    dt = float(times[1] - times[0])
    # Detect saccade peaks (gaze angular speed > threshold)
    saccade_thresh = BLINK_SACCADE_DEG / 0.12

    i = 0
    while i < n:
        t = float(times[i])
        since = t - last_blink
        trigger = False
        if since >= BLINK_MIN_S:
            if t >= next_due + 1.2 or since >= BLINK_MAX_S:
                trigger = True
            elif t >= next_due - 0.6 and gaze_speed_deg_s[i] >= saccade_thresh:
                trigger = True
            elif t >= next_due and not _near_onset(t):
                trigger = True
        if trigger:
            blink_times.append(t)
            last_blink = t
            # Rare natural double-blink (~7%) when not right in a dense passage
            if rng.random() < 0.07 and t + 0.35 < t1 and not _near_onset(t + 0.30):
                blink_times.append(t + float(rng.uniform(0.28, 0.38)))
                last_blink = blink_times[-1]
            interval = float(np.clip(rng.normal(BLINK_MEAN_S, 0.95), BLINK_MIN_S + 0.3, BLINK_MAX_S))
            next_due = last_blink + interval
            i = int(np.searchsorted(times, last_blink + 0.25))
        else:
            i += 1

    # Paint realistic asymmetric blink envelopes onto the timeline
    for bt in blink_times:
        c_end = bt + BLINK_CLOSE_S
        h_end = c_end + BLINK_HOLD_S
        o_end = h_end + BLINK_OPEN_S
        j0 = max(0, int(np.searchsorted(times, bt)))
        j1 = min(n, int(np.searchsorted(times, o_end)) + 1)
        for j in range(j0, j1):
            tj = float(times[j])
            if tj < c_end:
                v = float(_smoothstep((tj - bt) / BLINK_CLOSE_S))
            elif tj < h_end:
                v = 1.0
            elif tj < o_end:
                v = 1.0 - float(_smoothstep((tj - h_end) / BLINK_OPEN_S))
            else:
                v = 0.0
            if v > blink[j]:
                blink[j] = v
    return blink, blink_times


def solve(times, frames, notes, paths, rig, torso):
    """Solve universal head micro-follow, binocular eye gaze, and natural blinking.

    Returns a dict of per-frame arrays (plus a summary ``report`` dict), or None if the rig
    has no exported ``eyes`` block.
    """
    eyes_cfg = rig.data.get('eyes')
    if not eyes_cfg:
        return None
    has_head = eyes_cfg.get('head') is not None
    eye_sides = [h for h in ('L', 'R') if eyes_cfg.get(h) is not None]
    if not has_head and not eye_sides:
        return None

    times = np.asarray(times, dtype=float)
    n = len(times)
    dt = float(times[1] - times[0]) if n > 1 else (1.0 / 60.0)
    right0, fwd0, up0 = _face_basis(eyes_cfg)

    # 1. Visual attention target in world space
    target = compute_attention_targets(times, frames, notes, paths, rig)

    # 2. Exact world transform of the neck bone at every frame
    t_neck = _neck_world_transforms(times, torso)

    # Midpoint of neutral eyes (or head bone) for computing head aim
    if len(eye_sides) == 2:
        eye_mid_rest = 0.5 * (np.asarray(eyes_cfg['L']['head'], float) + np.asarray(eyes_cfg['R']['head'], float))
    elif len(eye_sides) == 1:
        eye_mid_rest = np.asarray(eyes_cfg[eye_sides[0]]['head'], float)
    else:
        eye_mid_rest = np.asarray(eyes_cfg['head']['head'], float)

    # 3. Compute total gaze angles relative to neck-stabilized head, then allocate head follow
    raw_yaw = np.zeros(n, dtype=float)
    raw_pitch = np.zeros(n, dtype=float)
    for i in range(n):
        rn = t_neck[i, :3, :3]
        orig = (t_neck[i] @ np.r_[eye_mid_rest, 1.0])[:3]
        raw_yaw[i], raw_pitch[i] = _aim_yaw_pitch(orig, target[i], rn @ right0, rn @ fwd0, rn @ up0)

    # Gaze speed (for saccade-coupled blinking and diagnostics)
    gaze_speed = np.hypot(np.gradient(raw_yaw, dt), np.gradient(raw_pitch, dt)) if n > 1 else np.zeros(n)

    head_yaw = np.zeros(n, dtype=float)
    head_pitch = np.zeros(n, dtype=float)
    head_roll = np.zeros(n, dtype=float)
    head_q = None
    t_head = t_neck.copy()

    if has_head:
        h_cfg = eyes_cfg['head']
        h_pivot = np.asarray(h_cfg['head'], dtype=float)
        h_pre_inv = np.linalg.inv(np.asarray(h_cfg['pre'], dtype=float))[:3, :3]
        h_world = np.asarray(h_cfg['world'], dtype=float)[:3, :3]

        des_h_yaw = np.clip(raw_yaw * HEAD_YAW_SHARE, -HEAD_YAW_LIMIT, HEAD_YAW_LIMIT)
        des_h_pitch = np.clip(raw_pitch * HEAD_PITCH_SHARE, HEAD_PITCH_MIN, HEAD_PITCH_MAX)
        head_yaw = np.clip(_smooth_series(des_h_yaw, dt, HEAD_TAU_S, HEAD_MAX_DEG_S),
                           -HEAD_YAW_LIMIT, HEAD_YAW_LIMIT)
        head_pitch = np.clip(_smooth_series(des_h_pitch, dt, HEAD_TAU_S, HEAD_MAX_DEG_S),
                             HEAD_PITCH_MIN, HEAD_PITCH_MAX)
        head_roll = np.clip(-head_yaw * HEAD_ROLL_PER_YAW, -HEAD_ROLL_LIMIT, HEAD_ROLL_LIMIT)

        head_q = np.zeros((n, 4), dtype=FLOAT)
        for i in range(n):
            # Pure yaw (about up0) -> pitch (about right0) -> gentle roll (about fwd0) in neck frame
            r_in_neck = (axis_matrix(up0, math.radians(head_yaw[i])).astype(float)
                         @ axis_matrix(right0, math.radians(head_pitch[i])).astype(float)
                         @ axis_matrix(fwd0, math.radians(head_roll[i])).astype(float))
            head_q[i] = mat3_q((h_pre_inv @ r_in_neck @ h_world).astype(FLOAT))
            # Update exact world transform of head bone for eyeball VOR
            rn = t_neck[i, :3, :3]
            r_world_delta = rn @ r_in_neck @ rn.T
            p_moved = (t_neck[i] @ np.r_[h_pivot, 1.0])[:3]
            t_head[i] = _about(p_moved, r_world_delta) @ t_neck[i]

    # 4. Binocular eyeball aim inside the moving head (automatic VOR + vergence)
    out = dict(gaze_target=target.astype(FLOAT))
    if head_q is not None:
        out['head_q'] = head_q

    eye_stats = {}
    for h in eye_sides:
        e_cfg = eyes_cfg[h]
        e_pivot = np.asarray(e_cfg['head'], dtype=float)
        e_pre_inv = np.linalg.inv(np.asarray(e_cfg['pre'], dtype=float))[:3, :3]
        e_world = np.asarray(e_cfg['world'], dtype=float)[:3, :3]

        ey = np.zeros(n, dtype=float)
        ep = np.zeros(n, dtype=float)
        for i in range(n):
            rh = t_head[i, :3, :3]
            p_eye = (t_head[i] @ np.r_[e_pivot, 1.0])[:3]
            yaw_i, pitch_i = _aim_yaw_pitch(p_eye, target[i], rh @ right0, rh @ fwd0, rh @ up0)
            ey[i] = float(np.clip(yaw_i * EYE_YAW_SHARE, -EYE_YAW_LIMIT, EYE_YAW_LIMIT))
            ep[i] = float(np.clip(pitch_i * EYE_PITCH_SHARE, EYE_PITCH_MIN, EYE_PITCH_MAX))

        ey = np.clip(_smooth_series(ey, dt, EYE_TAU_S, SACCADE_MAX_DEG_S), -EYE_YAW_LIMIT, EYE_YAW_LIMIT)
        ep = np.clip(_smooth_series(ep, dt, EYE_TAU_S, SACCADE_MAX_DEG_S), EYE_PITCH_MIN, EYE_PITCH_MAX)

        eq = np.zeros((n, 4), dtype=FLOAT)
        for i in range(n):
            r_in_head = (axis_matrix(up0, math.radians(ey[i])).astype(float)
                         @ axis_matrix(right0, math.radians(ep[i])).astype(float))
            eq[i] = mat3_q((e_pre_inv @ r_in_head @ e_world).astype(FLOAT))
        out[f'{h}_eye_q'] = eq
        eye_stats[h] = dict(
            yaw_deg=[round(float(ey.min()), 1), round(float(ey.max()), 1)],
            pitch_deg=[round(float(ep.min()), 1), round(float(ep.max()), 1)],
        )

    # 5. Natural blinking (written in Blender if the mesh has a matching blink shape key)
    seed = int(round((times[-1] - times[0]) * 100)) + len(notes) * 31
    blink, blink_times = _generate_blinks(times, gaze_speed, notes, seed=seed)
    out['blink'] = blink

    dur_min = max(float(times[-1] - times[0]) / 60.0, 1e-3)
    out['report'] = dict(
        head_bone=eyes_cfg['head']['bone'] if has_head else None,
        eye_bones={h: eyes_cfg[h]['bone'] for h in eye_sides},
        blink_shape_key=eyes_cfg.get('blink_shape_key'),
        head_yaw_deg=[round(float(head_yaw.min()), 1), round(float(head_yaw.max()), 1)] if has_head else None,
        head_pitch_deg=[round(float(head_pitch.min()), 1), round(float(head_pitch.max()), 1)] if has_head else None,
        eyes=eye_stats,
        blinks=int(len(blink_times)),
        blinks_per_min=round(float(len(blink_times) / dur_min), 1),
    )
    return out
