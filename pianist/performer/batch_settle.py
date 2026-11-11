"""Many coherence.settle problems at once on the GPU (PyTorch, float32).

The same residuals, bounds, start and bounded Levenberg-Marquardt as coherence.settle (the curved free fingers
dropped and the solve repeated where they held the hand off its keys), every problem one row of a batch, on the
kinematics and key tables of batch_pose. The key surface is coherence's KeySurface (bilinear over the 1 mm height
field, gap nodes left out of the blend). Problems: (hand, contacts, vector, values, target, pull, iterations, front).
"""
import numpy as np

from .batch_pose import BatchPoseSolver, _Hand, FINGER_BONES
from .checks import BRUSH_COST, BRUSH_DEPTH, CELL, OWN_KEY_TOLERANCE, stretched
from .coherence import (BLACK_FRONT_IN, BLACK_FRONT_SCALE, CENTRE_SCALE, FREE_CLEAR_SCALE, FREE_CURVED_LIFT,
                        FREE_CURVED_SCALE, FREE_LIFTED, LATERAL_FREE, LATERAL_HINGE, LATERAL_SCALE, NEUTRAL_ORIENTATION,
                        ORDER_SCALE, PALM_HEIGHT, PALM_SCALE, POSITION_TRAVEL, START_MARGIN, THUMB_LATERAL_SCALE,
                        THUMB_TIP_SCALE, WEB_STRAIN, WHITE_FRONT_CLEAR, WHITE_FRONT_SCALE)
from .kinematics import DIGITS, FLOAT
from .motion import FLOOR_SCALE, PAD_PITCH_MAX, PAD_PITCH_SCALE, KeyFloor, KeySurface, Spacing
from .posture import natural_orientation
from .pose import CONTACT_TOLERANCE, RELAXED, ROLL_BOUNDS, VALUE_SCALE, WEB_SCALE, YAW_BOUNDS, HandPose, hand_shape


class _SettleHand(_Hand):
    def __init__(self, solver, dev):
        super().__init__(solver, dev)
        torch = self.torch
        t = lambda a, dtype=FLOAT: torch.as_tensor(np.asarray(a, dtype), device=dev)
        surface = KeySurface(solver.heightfield)
        self.ks_on = torch.as_tensor(surface.on_key, device=dev)
        self.ks_top = t(surface.top)
        self.ks_no_key = float(KeySurface.NO_KEY)
        self.white_top = float(KeyFloor(solver.regions).white_top)
        spacing = Spacing(solver)
        self.lateral_axis = t(spacing.lateral)
        points = self.clear['mask'].shape[1]
        proximal = np.zeros((5, points), bool)
        for d in DIGITS:
            proximal[d - 1, :len(spacing.proximal[d])] = spacing.proximal[d]
        self.proximal = t(proximal, bool)
        self.whole_gaps = {d: np.asarray(g, FLOAT) for d, g in spacing.whole_gaps.items()}   # (proximal, distal)
        self.lifted = t([FREE_LIFTED[d != 1] for d in DIGITS])
        self.lateral_scale = t([THUMB_LATERAL_SCALE if d == 1 else LATERAL_SCALE for d in DIGITS])
        self.lateral_free = t([LATERAL_FREE[d] for d in DIGITS])
        self.web_free = t(solver.rig.web(solver.hand)[2])
        self.thumb_last = FINGER_BONES - 1                  # bone index of the thumb's last phalanx (digit 1 first)

    # ----- key surface (motion.KeySurface._compute) -----
    def _surface(self, points, depth_map, own):
        """(height (N, P), share brushable (N, P)) of the key surface under ``points``; ``own`` (N,) pitch or -2."""
        torch = self.torch
        shape = self.ks_top.shape
        u = (points[..., :2] - self.hf_origin) / CELL
        i = torch.floor(u)
        ix0 = i[..., 0].clamp(0, shape[0] - 2).long()
        iy0 = i[..., 1].clamp(0, shape[1] - 2).long()
        wx = (u[..., 0] - ix0).clamp(0., 1.)
        wy = (u[..., 1] - iy0).clamp(0., 1.)
        total = torch.zeros_like(wx)
        z_sum = torch.zeros_like(wx)
        s_sum = torch.zeros_like(wx)
        for dx, dy, w in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)), (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
            ix, iy = ix0 + dx, iy0 + dy
            pitch = self.hf_pitch[ix, iy]
            pressed = torch.gather(depth_map, 1, pitch.flatten(1)).reshape(pitch.shape)
            z = self.ks_top[ix, iy] - pressed
            mine = pitch == own[:, None]
            z = torch.where(mine, z - OWN_KEY_TOLERANCE, z)
            w = w * self.ks_on[ix, iy].to(w.dtype)
            brushable = (~mine & (pressed <= 0.)).to(w.dtype)
            z_sum = z_sum + w * z
            s_sum = s_sum + w * brushable
            total = total + w
        keyed = total > 1e-9
        safe = torch.where(keyed, total, torch.ones_like(total))
        return (torch.where(keyed, z_sum / safe, torch.full_like(total, self.ks_no_key)),
                torch.where(keyed, s_sum / safe, torch.zeros_like(total)))

    def _local_points(self, skin, digit, mats):
        """Skin points of finger ``digit`` (N,) in the wrist frame (N, P, 3)."""
        torch = self.torch
        bones = mats[torch.arange(len(digit), device=self.dev)[:, None], skin['sources'][digit]]
        return torch.einsum('nbpi,nbji,npb->npj', skin['local'][digit], bones, skin['weights'][digit])[..., :3]

    def _variants(self, values, replace, x_palm):
        """Each finger's clearance skin (world, (N, 5, P, 3)) with that finger alone set to ``replace`` (N, 5, 4)."""
        torch = self.torch
        n = values.shape[0]
        eye = torch.eye(5, device=self.dev)
        varied = values[:, None] * (1 - eye)[None, :, :, None] + replace[:, None] * eye[None, :, :, None]   # (N, 5, 5, 4)
        mats = self._fingers(varied.reshape(n * 5, 5, 4))
        mats = torch.cat([mats, self.constants.expand(n * 5, *self.constants.shape)], 1)
        digit = torch.arange(5, device=self.dev).repeat(n)
        palm = x_palm.repeat_interleave(5, 0)
        pts = self._points(self.clear, digit, mats, palm)
        return pts.reshape(n, 5, *pts.shape[1:])

    def settle_residuals(self, x, c):
        """coherence.settle's residuals of every row: (N, m)."""
        torch = self.torch
        slots, mats, palm = self._pose(x, c)
        n, k = slots.shape[:2]
        onehot = c['onehot']
        values = c['values'] * (1 - onehot.sum(1))[..., None] + torch.einsum('nkd,nkv->ndv', onehot, slots)
        relu = torch.relu
        out = []
        cmask = self.clear['mask']
        for i in range(k):
            digit = c['digit'][:, i]
            point = self._contact_point(self._points(self.pad, digit, mats, palm), self.pad['mask'][digit])
            out.append(self._region(c['pitch'][:, i], c['depth'][:, i], point) / CONTACT_TOLERANCE)
            out.append(((point[:, 0] - c['centre'][:, i]) / CENTRE_SCALE)[:, None])
            black = c['black'][:, i]
            front = torch.where(black, relu(c['edge'][:, i] - point[:, 1]) / BLACK_FRONT_SCALE,
                                relu(point[:, 1] - c['edge'][:, i]) / WHITE_FRONT_SCALE)
            out.append(front[:, None])
            # motion.pad_residual (fingers 2..5) and coherence.thumb_tip_angle (the thumb), both from the last bone
            last = mats[torch.arange(n, device=self.dev), digit * FINGER_BONES + FINGER_BONES - 1]
            axis = torch.einsum('nij,nj->ni', palm[:, :, :3], last[:, :3, 1])
            forward = palm[:, :, 1].clone()
            forward[:, 2] = 0.
            pitch = torch.rad2deg(torch.atan2(-axis[:, 2], (axis * forward).sum(1) / forward.norm(dim=1)))
            thumb = digit == 0
            out.append(torch.where(thumb, torch.zeros_like(pitch), relu(pitch - PAD_PITCH_MAX) / PAD_PITCH_SCALE)[:, None])
            tip = torch.rad2deg(torch.atan2(axis[:, 0], axis[:, 1]))
            out.append(torch.where(thumb, tip / THUMB_TIP_SCALE, torch.zeros_like(tip))[:, None])
            skin = self._points(self.clear, digit, mats, palm)
            z, soft = self._surface(skin, c['depth_map'], c['pitch'][:, i])
            pen = relu(z - skin[..., 2])
            brushed = relu(pen - BRUSH_DEPTH) + BRUSH_COST * pen.clamp(0., BRUSH_DEPTH)
            pen = torch.where(c['stretch'][:, None], soft * brushed + (1 - soft) * pen, pen)
            out.append(torch.where(cmask[digit], pen, torch.zeros_like(pen)) / FLOOR_SCALE)
        # free fingers: lifted they clear the keys; keeping their curve preferably too
        none = torch.full((n,), -2, dtype=torch.long, device=self.dev)
        for replace, scale, on in ((self.lifted.expand(n, 5, 4), FREE_CLEAR_SCALE, c['free']),
                                   (c['curved_free'], FREE_CURVED_SCALE, c['free'] * c['curved'][:, None])):
            pts = self._variants(values, replace, palm)                       # (N, 5, P, 3)
            p = pts.shape[2]
            z, _ = self._surface(pts.reshape(n, 5 * p, 3), c['depth_map'], none)
            gap = relu(z.reshape(n, 5, p) - pts[..., 2]) * cmask[None].to(z.dtype) * on[..., None]
            out.append(gap.flatten(1) / scale)
        # chord fingers 2..5 keep their order across the palm (coherence._order)
        local = [self._local_points(self.clear, c['digit'][:, i], mats) @ self.lateral_axis for i in range(k)]
        halves = []
        for i in range(k):
            prox = self.proximal[c['digit'][:, i]] & cmask[c['digit'][:, i]]
            dist = ~self.proximal[c['digit'][:, i]] & cmask[c['digit'][:, i]]
            halves.append(torch.stack([(local[i] * prox).sum(1) / prox.sum(1).clamp(min=1),
                                       (local[i] * dist).sum(1) / dist.sum(1).clamp(min=1)], 1))
        for i in range(k - 1):
            gap = halves[i + 1] - halves[i]
            out.append(relu(c['need'][:, i] - gap) * c['order_on'][:, i, None] / ORDER_SCALE)
        lateral = slots[..., 0]
        digit = c['digit']
        out.append(lateral / self.lateral_scale[digit])
        out.append(relu(lateral.abs() - self.lateral_free[digit]) / LATERAL_HINGE)
        for i in range(k):
            for j in range(i + 1, k):
                a, b = digit[:, i] + 1, digit[:, j] + 1
                opening = self.web_rest[a, b] + self.web_slope[a, b] * (slots[:, j, 0] - slots[:, i, 0])
                out.append(relu(opening - self.web_hard[a, b])[:, None] / WEB_SCALE)
                out.append(relu(opening - self.web_free[a, b])[:, None] / WEB_STRAIN)
        out.append((x[:, :5] - c['target']) / c['pull'])
        out.append((relu(self.white_top + PALM_HEIGHT - x[:, 2]) / PALM_SCALE)[:, None])
        out.append((x[:, 3:5] - c['natural']) / self.ks_orientation)
        out.append(((slots - c['shape']) / self.value_scale).flatten(1))
        return torch.cat(out, 1)

    def solve_settle(self, problems):
        """problems: (contacts, vector, values, target, pull, iterations, front), all with the same contact count."""
        torch, s = self.torch, self.solver
        self.ks_orientation = torch.as_tensor(NEUTRAL_ORIENTATION, device=self.dev)
        rows = []
        for contacts, vector, values, target, pull, iterations, front in problems:
            contacts = sorted(contacts, key=lambda c: c.digit)
            digits = [c.digit for c in contacts]
            vector = np.asarray(vector, FLOAT)
            depths = {c.pitch: c.depth for c in contacts}
            depth_map = np.zeros(self.pitches, FLOAT)
            for c in contacts:
                depth_map[c.pitch] = c.depth
            base = np.array([np.asarray(values.get(d, RELAXED[d]), FLOAT) for d in DIGITS])
            shape = hand_shape(s.span(depths))
            lower = np.r_[vector[:3] - POSITION_TRAVEL, YAW_BOUNDS[0], ROLL_BOUNDS[0], *[self.lower[d] for d in digits]]
            upper = np.r_[vector[:3] + POSITION_TRAVEL, YAW_BOUNDS[1], ROLL_BOUNDS[1], *[self.upper[d] for d in digits]]
            margin = np.r_[np.zeros(3), np.full(len(lower) - 3, START_MARGIN)]
            x = np.clip(np.r_[vector, *[base[d - 1] for d in digits]], lower + margin, upper - margin)
            edges, blacks = [], []
            for c in contacts:
                black = s.regions.is_black(c.pitch)
                blacks.append(black)
                if black:
                    edges.append(s.regions.keys[c.pitch]['bounds_min'][1] + BLACK_FRONT_IN)
                else:
                    edges.append(s.regions.black_front - WHITE_FRONT_CLEAR if front is None
                                 else s.regions.keys[c.pitch]['bounds_min'][1] + front)
            need, order_on = [], []
            for a, b in zip(digits, digits[1:]):
                on = a >= 2
                order_on.append(float(on))
                need.append(sum(self.whole_gaps[d] for d in range(a, b)) / .85 * .6 if on else np.zeros(2, FLOAT))
            free = np.array([d not in digits for d in DIGITS], FLOAT)
            rows.append(dict(digit=np.array(digits, np.int64) - 1, onehot=np.eye(5, dtype=FLOAT)[np.array(digits) - 1],
                             pitch=np.array([c.pitch for c in contacts], np.int64),
                             depth=np.array([c.depth for c in contacts], FLOAT), depth_map=depth_map, values=base,
                             target=np.asarray(target, FLOAT), pull=np.asarray(pull, FLOAT),
                             natural=natural_orientation(s, vector[:3]).astype(FLOAT),
                             shape=np.array([shape[d] for d in digits], FLOAT),
                             curved_free=np.array([shape[d] + np.array([0., FREE_CURVED_LIFT, 0., 0.]) for d in DIGITS], FLOAT),
                             free=free, curved=np.float32(1.), stretch=stretched(depths),
                             black=np.array(blacks, bool), edge=np.array(edges, FLOAT),
                             centre=np.array([s.regions.centre(c.pitch)[0] for c in contacts], FLOAT),
                             need=np.array(need, FLOAT).reshape(len(digits) - 1, 2), order_on=np.array(order_on, FLOAT).reshape(len(digits) - 1),
                             lower=lower.astype(FLOAT), upper=upper.astype(FLOAT), x=x.astype(FLOAT), iterations=iterations))
        stack = lambda k, dtype=FLOAT: torch.as_tensor(np.stack([np.asarray(r[k]) for r in rows]).astype(dtype), device=self.dev)
        c = {k: stack(k, dt) for k, dt in (('digit', np.int64), ('onehot', FLOAT), ('pitch', np.int64), ('depth', FLOAT),
                                           ('depth_map', FLOAT), ('values', FLOAT), ('target', FLOAT), ('pull', FLOAT),
                                           ('natural', FLOAT), ('shape', FLOAT), ('curved_free', FLOAT), ('free', FLOAT),
                                           ('curved', FLOAT), ('stretch', bool), ('black', bool), ('edge', FLOAT),
                                           ('centre', FLOAT), ('need', FLOAT), ('order_on', FLOAT))}
        lower, upper, start = stack('lower'), stack('upper'), stack('x')
        iterations = max(r['iterations'] for r in rows)
        k = len(rows[0]['digit'])

        def residual(cc):
            f = lambda v, idx: self.settle_residuals(v, {key: t[idx] for key, t in cc.items()})
            f.key = ('settle', k)
            return f

        with torch.no_grad():
            x, used = self._lm(residual(c), start, lower, upper, iterations)
            error = self._contact_errors(x, c)
            retry = (c['free'].sum(1) > 0) & (error > CONTACT_TOLERANCE * 2.5)
            if bool(retry.any()):
                idx = torch.nonzero(retry)[:, 0]
                cr = {key: t[idx] for key, t in c.items()}
                cr['curved'] = torch.zeros_like(cr['curved'])
                xr, more = self._lm(residual(cr), start[idx], lower[idx], upper[idx], iterations)
                x[idx] = xr
                used[idx] = used[idx] + more
            error = self._contact_errors(x, c)
            _, mats, palm = self._pose(x, c)
            targets, worst = [], []
            no_brush = torch.zeros_like(c['stretch'])
            for i in range(k):
                digit = c['digit'][:, i]
                targets.append(self._contact_point(self._points(self.pad, digit, mats, palm), self.pad['mask'][digit]))
                pen = self._penetration(self._points(self.clear, digit, mats, palm), self.clear['mask'][digit],
                                        c['depth_map'], c['pitch'][:, i], no_brush)
                worst.append(pen.amax(1))
        x, error, used = x.cpu().numpy(), error.cpu().numpy(), used.cpu().numpy()
        targets, worst = torch.stack(targets, 1).cpu().numpy(), torch.stack(worst, 1).cpu().numpy()
        poses = []
        for r, row in enumerate(rows):
            digits = row['digit'] + 1
            values = {d: row['values'][d - 1].copy() for d in DIGITS}
            values.update({int(d): x[r, 5 + 4 * i:9 + 4 * i].copy() for i, d in enumerate(digits)})
            poses.append(HandPose(x[r, :3].copy(), float(x[r, 3]), float(x[r, 4]), values, float(error[r]),
                                  float(error[r]) <= CONTACT_TOLERANCE * 2.5, int(used[r]),
                                  {int(d): targets[r, i].copy() for i, d in enumerate(digits)}, float(worst[r].max())))
        return poses

    def _contact_errors(self, x, c):
        _, mats, palm = self._pose(x, c)
        errors = []
        for i in range(c['digit'].shape[1]):
            digit = c['digit'][:, i]
            point = self._contact_point(self._points(self.pad, digit, mats, palm), self.pad['mask'][digit])
            errors.append(self._region(c['pitch'][:, i], c['depth'][:, i], point).abs().amax(1))
        return self.torch.stack(errors, 1).amax(1)


class BatchSettler(BatchPoseSolver):
    """``settle`` for many problems: (hand, contacts, vector, values, target, pull, iterations, front) -> HandPoses."""

    def __init__(self, solvers, dev=None):
        super().__init__(solvers, dev)
        self.settlers = {h: _SettleHand(s, self.dev) for h, s in solvers.items()}

    def settle(self, problems):
        out = [None] * len(problems)
        groups = {}
        for i, (h, contacts, *_) in enumerate(problems):
            groups.setdefault((h, len(contacts), problems[i][6]), []).append(i)
        for (h, k, _), mine in groups.items():
            if not k:
                raise ValueError('settle needs contacts')
            for i, pose in zip(mine, self.settlers[h].solve_settle([problems[i][1:] for i in mine])):
                out[i] = pose
        return out
