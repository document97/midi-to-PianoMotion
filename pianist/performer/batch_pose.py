"""Many pose.PoseSolver.solve problems at once on the GPU (PyTorch, float32).

The same bounded Levenberg-Marquardt, seeds, residuals and selection as PoseSolver.solve, but every
problem and every seed is one row of a batch. A row has k contact slots, each naming its finger; the
fingers' skins are padded to one shape and looked up per row, so one batch holds every chord of a
hand with k contacts whichever fingers play it. The Jacobian comes from the numpy solver's forward
differences, all steps of all rows in one pass, so both solvers run one algorithm.

Used for the planner's pair table, thousands of independent solves (153 s on a Radeon 780M against 569 s for
the numpy solver on 16 processes). The anchor poses are a few hundred per piece: as fast on worker processes, so they stay there.
``device()`` is None without PyTorch or a GPU (or with PIANOMOTION_DEVICE=cpu); the callers
then keep the numpy solver on worker processes.
"""
import math
import os

import numpy as np

from .checks import BRUSH_COST, BRUSH_DEPTH, CELL, OWN_KEY_TOLERANCE, stretched
from .kinematics import DIGITS, FLOAT, child_euler
from .pose import (hand_shape, CLEARANCE_SCALE, CONTACT_TOLERANCE, WEB_SCALE, NARROW_TRANSITION, PAD_SOFTMIN_METRES, RELAXED, ROLL_BOUNDS,
                   SEED_ITERATIONS, SEEDS, STEP_TOLERANCE, VALUE_SCALE, WRIST_SCALE, WRIST_TRAVEL, YAW_BOUNDS, HandPose,
                   wrist_steps)

# GPU memory a solve may use: this share of what is free when it starts (an iGPU shares it with the CPU),
# or PIANOMOTION_GPU_MEMORY_GB. Passes are split into chunks of rows that fit, measured on the first pass
# of each kind, so a larger card runs larger chunks.
MEMORY_SHARE = float(os.environ.get('PIANOMOTION_GPU_MEMORY_SHARE', .4))
PROBE_ROWS = 128            # rows of the pass that measures a chunk's memory per row
BATCH_ROWS = 1 << 18        # problem-seed rows of one solve (their state is small; passes are chunked)
FINGER_BONES = 3            # per finger: root, MCP, IP
_device = False


def device():
    """The torch device for batched solves, or None (use the numpy solver)."""
    global _device
    if _device is False:
        _device = None
        if os.environ.get('PIANOMOTION_DEVICE', '').lower() != 'cpu':
            try:
                import torch
                if torch.cuda.is_available():      # CUDA and ROCm builds
                    _device = torch.device(os.environ.get('PIANOMOTION_DEVICE') or 'cuda')
            except ImportError:
                pass
    return _device


class _Replay:
    """fn() -> tuple of tensors over fixed buffers: run as is twice (the second time on a side stream, as
    CUDA/HIP graph capture asks), recorded as a graph then and replayed from the third call on. Replay runs
    the same kernels on the same buffers, so results do not change; PIANOMOTION_GPU_GRAPHS=0, or a graph
    that cannot be recorded, keeps every call a plain run."""
    enabled = os.environ.get('PIANOMOTION_GPU_GRAPHS', '1') != '0'

    def __init__(self, fn, dev):
        self.fn, self.dev, self.calls, self.graph = fn, dev, 0, None

    def __call__(self):
        import torch
        self.calls += 1
        if self.graph is not None:
            self.graph.replay()
            return self.out
        if self.calls != 2 or not _Replay.enabled or self.dev.type != 'cuda':
            return self.fn()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            out = self.fn()
        torch.cuda.current_stream().wait_stream(side)
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.out = self.fn()
            self.graph = graph
        except Exception as error:      # this device or build cannot record it: plain runs from now on
            _Replay.enabled = False
            print(f'batch_pose: GPU graphs off ({type(error).__name__}: {error})', flush=True)
            torch.cuda.synchronize()
        return out


class BatchPoseSolver:
    """Both hands' PoseSolvers (pose.hand_solvers or regions.performer_solvers) as batched torch solvers."""

    def __init__(self, solvers, dev=None):
        self.dev = dev or device()
        self.solvers = solvers
        self.hands = {h: _Hand(s, self.dev) for h, s in solvers.items()}

    def solve(self, problems, seeds=SEEDS, clearance=True, iterations=40):
        """``problems``: (hand, contacts, wrist or None, x_hint or None) each; returns HandPoses in order."""
        out = [None] * len(problems)
        groups = {}
        for i, (h, contacts, *_) in enumerate(problems):
            groups.setdefault((h, len(contacts)), []).append(i)
        for (h, k), mine in groups.items():
            if not k:       # nothing to reach: the numpy solver's pose
                for i in mine:
                    out[i] = self.solvers[h].solve([], wrist=problems[i][2], x_hint=problems[i][3], seeds=seeds,
                                                   clearance=clearance, iterations=iterations)
                continue
            step = max(1, BATCH_ROWS // len(seeds))
            for start in range(0, len(mine), step):
                part = mine[start:start + step]
                for i, pose in zip(part, self.hands[h].solve([problems[i][1:] for i in part], seeds, clearance, iterations)):
                    out[i] = pose
        return out


class _Hand:
    def __init__(self, solver, dev):
        import torch
        self.torch, self.dev, self.solver = torch, dev, solver
        t = lambda a, dtype=FLOAT: torch.as_tensor(np.asarray(a, dtype), device=dev)
        if 'PIANOMOTION_GPU_MEMORY_GB' in os.environ:
            self.budget = float(os.environ['PIANOMOTION_GPU_MEMORY_GB']) * 2 ** 30
        elif dev.type == 'cuda':
            self.budget = torch.cuda.mem_get_info(dev)[0] * MEMORY_SHARE
        else:
            self.budget = 2 ** 30
        self.row_bytes = {}
        # constants of the passes (a GPU graph cannot copy from the host while it records)
        self.bottom_row, self.z_axis = t([[0., 0., 0., 1.]]), t([0., 0., 1.])
        self.steps = t(wrist_steps(len(DIGITS)))
        rig, hand = solver.rig, solver.hand
        arm = solver.arm
        self.palm_q = t(arm.palm)
        self.palm_forward = t(np.asarray(arm.palm_forward) / np.linalg.norm(arm.palm_forward))
        self.wrist_scale = t(arm.scale['wrist'])
        self.relaxed = t([RELAXED[d] for d in DIGITS])
        self.value_scale, self.wrist_prior_scale = t(VALUE_SCALE), t(WRIST_SCALE)
        # key regions: top, wide lo/hi, narrow lo/hi, split, y lo/hi per pitch
        self.pitches = max(solver.regions.rows) + 1
        table = np.zeros((self.pitches, 8), FLOAT)
        for p, r in solver.regions.rows.items():
            table[p] = [r['top'], *r['wide'], *r['narrow'], r['split'], *r['y']]
        self.regions = t(table)
        hf = solver.heightfield
        self.hf_origin = t(hf.origin)
        self.hf_finite = torch.as_tensor(np.isfinite(hf.top), device=dev)
        self.hf_top = t(np.where(np.isfinite(hf.top), hf.top, -10.))
        self.hf_pitch = torch.as_tensor(hf.pitch.astype(np.int64), device=dev)
        self.hf_shape = hf.top.shape
        # the five fingers' parameters, stacked (digit first)
        fingers = [rig.fingers[hand][d] for d in DIGITS]
        e0 = [[child_euler(f.row, i, 0.) for i in (0, 1)] for f in fingers]
        unit = [[np.subtract(child_euler(f.row, i, math.degrees(1.)), e0[n][i]) for i in (0, 1)] for n, f in enumerate(fingers)]
        self.f = {k: t([getattr(f, k) for f in fingers]) for k in
                  ('pre', 'root_loc', 'root_scale', 'child_loc', 'child_scale', 'links', 'base', 'normal', 'lift')}
        self.f.update(neutral=t([f.neutral_matrix for f in fingers]), e0=t(e0), unit=t(unit))
        self.lower = {d: f.lower for d, f in zip(DIGITS, fingers)}
        self.web_rest, self.web_slope, _, self.web_hard = (t(a) for a in rig.web(hand))
        self.upper = {d: f.upper for d, f in zip(DIGITS, fingers)}
        # bone matrices of a row: 5 x 3 finger bones, then the bones that move with the wrist (constants)
        owner = rig.bone_owner[hand]
        rigid = rig.neutral_wrist_inverse[hand]
        constants = {rig.wrist_bone[hand]: np.eye(4, dtype=FLOAT)}

        def source(name):
            if name in owner:
                return (owner[name] - 1) * FINGER_BONES + rig.fingers[hand][owner[name]].names.index(name)
            if name not in constants:
                constants[name] = rigid @ rig.neutral_bones[name]
            return len(DIGITS) * FINGER_BONES + list(constants).index(name)

        def padded(skins):
            """{digit: SkinSet} -> weights (5, P, B), local (5, B, P, 4), bone sources (5, B), point mask (5, P)."""
            points = max(s.weights.shape[0] for s in skins.values())
            bones = max(len(s.bones) for s in skins.values())
            weights = np.zeros((5, points, bones), FLOAT)
            local = np.zeros((5, bones, points, 4), FLOAT)
            sources = np.full((5, bones), len(DIGITS) * FINGER_BONES, np.int64)   # padding: the wrist (weight 0)
            mask = np.zeros((5, points), bool)
            for d, s in skins.items():
                n, b = s.weights.shape
                weights[d - 1, :n, :b], local[d - 1, :b, :n] = s.weights, s.local
                sources[d - 1, :b] = [source(name) for name in s.bones]
                mask[d - 1, :n] = True
            return dict(weights=t(weights), local=t(local), sources=t(sources, np.int64), mask=t(mask, bool))
        self.pad = padded({d: rig.fingers[hand][d].pad for d in DIGITS})
        self.clear = padded({d: solver.clearance_skin[d] for d in DIGITS})
        self.constants = t(list(constants.values()))

    # ----- kinematics (leading batch dimensions broadcast) -----
    def _axis_matrix(self, axis, radians):
        """Rodrigues: unit axes (..., 3), radians (...) -> (..., 3, 3)."""
        torch = self.torch
        x, y, z = axis.expand(*radians.shape, 3).unbind(-1)
        c, s = torch.cos(radians), torch.sin(radians)
        t = 1 - c
        return torch.stack([torch.stack([t * x * x + c, t * x * y - s * z, t * x * z + s * y], -1),
                            torch.stack([t * x * y + s * z, t * y * y + c, t * y * z - s * x], -1),
                            torch.stack([t * x * z - s * y, t * y * z + s * x, t * z * z + c], -1)], -2)

    def _euler_xyz(self, e):
        torch = self.torch
        cx, cy, cz = torch.cos(e).unbind(-1)
        sx, sy, sz = torch.sin(e).unbind(-1)
        return torch.stack([torch.stack([cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz], -1),
                            torch.stack([cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz], -1),
                            torch.stack([-sy, sx * cy, cx * cy], -1)], -2)

    def _compose(self, loc, rotation, scale):
        torch = self.torch
        top = torch.cat([rotation * scale[..., None, :], loc.expand(*rotation.shape[:-2], 3)[..., None]], -1)
        bottom = self.bottom_row.expand(*rotation.shape[:-2], 1, 4)
        return torch.cat([top, bottom], -2)

    def _fingers(self, values):
        """kinematics.Finger.local_matrices of all five fingers for (N, 5, 4) values (deg): (N, 15, 4, 4)."""
        torch, f = self.torch, self.f
        rad = torch.deg2rad(values)
        lifted = (self._axis_matrix(f['lift'], rad[..., 1]) @ f['base'][..., None])[..., 0]
        swung = (self._axis_matrix(f['normal'], rad[..., 0]) @ lifted[..., None])[..., 0]
        # rotation taking base onto the swung axis (both unit): I + [v]x + [v]x^2 / (1 + c)
        v = torch.cross(f['base'].expand_as(swung), swung, dim=-1)
        c = (swung * f['base']).sum(-1)
        zero = torch.zeros_like(c)
        vx = torch.stack([torch.stack([zero, -v[..., 2], v[..., 1]], -1), torch.stack([v[..., 2], zero, -v[..., 0]], -1),
                          torch.stack([-v[..., 1], v[..., 0], zero], -1)], -2)
        arc = torch.eye(3, device=self.dev) + vx + vx @ vx / (1 + c)[..., None, None]
        out = [f['pre'] @ self._compose(f['root_loc'], arc @ f['neutral'], f['root_scale'])]
        for i in (0, 1):
            euler = f['e0'][:, i] + rad[..., 2 + i, None] * f['unit'][:, i]
            out.append(out[-1] @ f['links'][:, i] @ self._compose(f['child_loc'][:, i], self._euler_xyz(euler),
                                                                  f['child_scale'][:, i]))
        return torch.stack(out, 2).flatten(1, 2)

    def _palm(self, wrist, yaw, roll):
        """PoseSolver.palm: (N, 3, 4) rows of the wrist matrix."""
        torch = self.torch

        def q_axis(axis, degrees):
            half = torch.deg2rad(degrees)[:, None] / 2
            return torch.cat([torch.cos(half), torch.sin(half) * axis], -1)

        def q_mul(a, b):
            w1, x1, y1, z1 = a.unbind(-1)
            w2, x2, y2, z2 = b.unbind(-1)
            return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)
        q = q_mul(q_axis(self.z_axis, yaw), q_mul(q_axis(self.palm_forward, roll), self.palm_q.expand(yaw.shape[0], 4)))
        w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
        m = torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                         torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                         torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)
        return torch.cat([m * self.wrist_scale, wrist[:, :, None]], -1)

    def _points(self, skin, digit, mats, palm):
        """World skin points (N, P, 3) of each row's finger ``digit`` (N,) (0-based): blend skinning, then the palm."""
        torch = self.torch
        bones = mats[torch.arange(len(digit), device=self.dev)[:, None], skin['sources'][digit]]   # (N, B, 4, 4)
        acc = torch.einsum('nbpi,nbji,npb->npj', skin['local'][digit], bones, skin['weights'][digit])
        return torch.einsum('nij,npj->npi', palm, acc)

    def _contact_point(self, points, mask):
        """PoseSolver.contact_point over the unpadded points (``mask``)."""
        w = self.torch.softmax(self.torch.where(mask, -points[..., 2] / PAD_SOFTMIN_METRES, -1e30), dim=-1)
        return (w[..., None] * points).sum(-2)

    def _region(self, pitch, depth, point):
        """KeyRegions.residual, (N,) pitches -> (N, 3) metres."""
        torch = self.torch
        top, w0, w1, n0, n1, split, y0, y1 = self.regions[pitch].unbind(-1)
        x, y, z = point.unbind(-1)
        s = torch.sigmoid(torch.clamp((y - split) / NARROW_TRANSITION, -50., 50.))
        lo, hi = (1 - s) * w0 + s * n0, (1 - s) * w1 + s * n1
        relu = torch.relu
        return torch.stack([relu(lo - x) - relu(x - hi), relu(y0 - y) - relu(y - y1), z - (top - depth)], -1)

    def _penetration(self, points, mask, depth_map, own, brush):
        """checks.Heightfield.penetration with own=(own pitch, all points); (N, P) metres, 0 on padding."""
        torch = self.torch
        cell = torch.floor((points[..., :2] - self.hf_origin) / CELL).long()
        ok = (cell[..., 0] >= 0) & (cell[..., 1] >= 0) & (cell[..., 0] < self.hf_shape[0]) & (cell[..., 1] < self.hf_shape[1])
        ok = ok & mask
        ix = cell[..., 0].clamp(0, self.hf_shape[0] - 1)
        iy = cell[..., 1].clamp(0, self.hf_shape[1] - 1)
        top, pitch, finite = self.hf_top[ix, iy], self.hf_pitch[ix, iy], self.hf_finite[ix, iy] & ok
        pressed = torch.gather(depth_map, 1, pitch)
        pen = torch.where(finite, top - pressed - points[..., 2], torch.zeros_like(top))
        pen = torch.where(pitch == own[:, None], pen - OWN_KEY_TOLERANCE, pen)
        pen = torch.relu(pen)
        brushable = brush[:, None] & (pitch != own[:, None]) & (pressed <= 0.) & finite
        soft = torch.relu(pen - BRUSH_DEPTH) + BRUSH_COST * torch.clamp(pen, 0., BRUSH_DEPTH)
        return torch.where(ok, torch.where(brushable, soft, pen), torch.zeros_like(pen))

    def _pose(self, x, c):
        """Slot values (N, k, 4), every bone matrix of the hand (N, 15 + C, 4, 4) and the palm."""
        torch = self.torch
        slots = x[:, 5:].reshape(len(x), -1, 4)
        onehot = c['onehot']                                           # (N, k, 5)
        values = c['values'] * (1 - onehot.sum(1))[..., None] + torch.einsum('nkd,nkv->ndv', onehot, slots)
        mats = torch.cat([self._fingers(values), self.constants.expand(len(x), *self.constants.shape)], 1)
        return slots, mats, self._palm(x[:, :3], x[:, 3], x[:, 4])

    def residuals(self, x, c, prior, clear):
        """PoseSolver.solve's residuals, contact slot by slot: (N, m)."""
        torch = self.torch
        slots, mats, palm = self._pose(x, c)
        out = []
        for i in range(slots.shape[1]):
            digit = c['digit'][:, i]
            point = self._contact_point(self._points(self.pad, digit, mats, palm), self.pad['mask'][digit])
            out.append(self._region(c['pitch'][:, i], c['depth'][:, i], point) / CONTACT_TOLERANCE)
            if clear:
                points = self._points(self.clear, digit, mats, palm)
                pen = self._penetration(points, self.clear['mask'][digit], c['depth_map'], c['pitch'][:, i], c['stretch'])
                out.append(pen / CLEARANCE_SCALE)
        for i in range(slots.shape[1]):              # the web between every two contact fingers (PoseSolver.web_limit)
            for j in range(i + 1, slots.shape[1]):
                a, b = c['digit'][:, i] + 1, c['digit'][:, j] + 1
                opening = self.web_rest[a, b] + self.web_slope[a, b] * (slots[:, j, 0] - slots[:, i, 0])
                out.append(torch.relu(opening - self.web_hard[a, b])[:, None] / WEB_SCALE)
        out.append(prior * (x[:, :5] - c['prior_wrist']) / self.wrist_prior_scale)
        out.append((prior * (slots - c['shape']) / self.value_scale).flatten(1))
        return torch.cat(out, 1)

    def _chunked(self, key, run, count):
        """run(slice) -> tuple of tensors, over ``count`` rows in chunks that fit the memory budget, concatenated.

        The first full-size probe of each ``key`` measures the pass's memory per row."""
        torch = self.torch
        parts, start = [], 0
        if key not in self.row_bytes and count >= PROBE_ROWS:
            cuda = self.dev.type == 'cuda'
            if cuda:
                torch.cuda.synchronize(self.dev)
                base = torch.cuda.memory_allocated(self.dev)
                torch.cuda.reset_peak_memory_stats(self.dev)
            parts.append(run(slice(0, PROBE_ROWS)))
            peak = torch.cuda.max_memory_allocated(self.dev) - base if cuda else 2 ** 16 * PROBE_ROWS
            self.row_bytes[key] = max(peak / PROBE_ROWS, 1.)
            start = PROBE_ROWS
        step = max(PROBE_ROWS, int(self.budget // self.row_bytes.get(key, 2 ** 20)))
        while start < count:
            parts.append(run(slice(start, start + step)))
            start += step
        return tuple(torch.cat(p) for p in zip(*parts))

    def _eval(self, f, x, rows):
        return self._chunked(('residual',) + f.key, lambda s: (f(x[s], rows[s]),), len(x))[0]

    def _normal(self, f, x, rows, r):
        """Gradient J^T r (R, n) and J^T J (R, n, n) of rows' residuals, J by the numpy solver's forward
        differences (pose.wrist_steps), every step of every row in one pass. Differences, not exact
        derivatives: the key tops are steps of a 1 mm height field, and a difference step that crosses
        into a higher cell is what pushes skin off a key's edge (an exact derivative there is zero)."""
        torch = self.torch
        n_rows, n = x.shape
        steps = self.steps[:n]
        trial = (x[None] + torch.diag(steps)[:, None, :]).reshape(n * n_rows, n)
        moved = f(trial, rows.repeat(n)).reshape(n, n_rows, -1)
        jac = ((moved - r[None]) / steps[:, None, None]).permute(1, 2, 0)
        return torch.einsum('rmn,rm->rn', jac, r), jac.transpose(1, 2) @ jac

    def _trial(self, f, x, r, cost, lam, g, h, improved, moved, live, lower, upper, rows):
        """One damped step of every row still looking for one; rows that took a step keep it (pure: new state)."""
        torch = self.torch
        pending = live & ~improved
        a = h + lam[:, None, None] * torch.diag_embed(torch.diagonal(h, dim1=1, dim2=2) + 1e-9)
        chol, info = torch.linalg.cholesky_ex(a)        # J^T J + damping is symmetric positive definite
        delta = torch.cholesky_solve(-g[..., None], chol)[..., 0]
        trial = torch.minimum(torch.maximum(x + delta, lower), upper)
        tr = f(trial, rows)
        tc = (tr * tr).sum(1)
        better = pending & (info == 0) & torch.isfinite(tc) & (tc < cost)
        return (torch.where(better[:, None], trial, x), torch.where(better[:, None], tr, r), torch.where(better, tc, cost),
                torch.where(better, torch.clamp(lam / 3, min=1e-7), torch.where(pending, lam * 4, lam)),
                improved | better, torch.where(better, delta.abs().amax(1), moved))

    def _lm(self, f, x, lower, upper, iterations):
        """PoseSolver._levenberg_marquardt on every row; f(x, rows) -> residuals. Returns (x, iterations used).

        Rows are solved in chunks of one fixed size (the last one padded) that fit the memory budget; a row
        that has stopped stays in its chunk, frozen, so every pass has the same shapes and is replayed as a
        GPU graph (``graphs``) instead of thousands of separate kernel launches."""
        torch = self.torch
        n_rows, n = x.shape
        key = ('lm',) + f.key
        if key not in self.row_bytes and n_rows >= PROBE_ROWS and self.dev.type == 'cuda':
            probe = torch.arange(PROBE_ROWS, device=self.dev)
            torch.cuda.synchronize(self.dev)
            base = torch.cuda.memory_allocated(self.dev)
            torch.cuda.reset_peak_memory_stats(self.dev)
            self._normal(f, x[probe], probe, f(x[probe], probe))
            torch.cuda.synchronize(self.dev)
            self.row_bytes[key] = max((torch.cuda.max_memory_allocated(self.dev) - base) / PROBE_ROWS, 1.)
        size = min(n_rows, max(PROBE_ROWS, int(self.budget // self.row_bytes.get(key, 2 ** 20))))
        out_x, out_used = x.clone(), torch.full((n_rows,), iterations, dtype=torch.long, device=self.dev)
        state = None
        for start in range(0, n_rows, size):
            rows = torch.arange(start, start + size, device=self.dev).clamp(max=n_rows - 1)   # padded with the last row
            take = min(size, n_rows - start)
            if state is None:
                state = self._chunk_state(f, size, n)
            s = state
            s['rows'].copy_(rows)
            s['x'].copy_(x[rows])
            s['lower'].copy_(lower[rows])
            s['upper'].copy_(upper[rows])
            s['r'].copy_(f(s['x'], s['rows']))
            s['cost'].copy_((s['r'] * s['r']).sum(1))
            s['lam'].fill_(1e-2)
            s['live'].fill_(True)
            s['used'].fill_(iterations)
            for it in range(1, iterations + 1):
                if not bool(s['live'].any()):
                    break
                for k, v in zip(('g', 'h'), s['normal']()):
                    s[k].copy_(v)
                s['improved'].fill_(False)
                s['moved'].zero_()
                for _ in range(10):
                    for k, v in zip(('x', 'r', 'cost', 'lam', 'improved', 'moved'), s['trial']()):
                        s[k].copy_(v)
                    if not bool((s['live'] & ~s['improved']).any()):
                        break
                stop = s['live'] & (~s['improved'] | (s['moved'] < STEP_TOLERANCE) | (s['cost'] < 1e-6))
                s['used'].copy_(torch.where(stop, it, s['used']))
                s['live'] &= ~stop
            out_x[start:start + take] = s['x'][:take]
            out_used[start:start + take] = s['used'][:take]
        return out_x, out_used

    def _chunk_state(self, f, size, n):
        """Fixed buffers of one chunk and its two passes (normal equations, one trial step), as GPU graphs."""
        torch = self.torch
        dev = self.dev
        m = f(torch.zeros(1, n, device=dev), torch.zeros(1, dtype=torch.long, device=dev)).shape[1]
        s = dict(rows=torch.zeros(size, dtype=torch.long, device=dev), x=torch.zeros(size, n, device=dev),
                 lower=torch.zeros(size, n, device=dev), upper=torch.zeros(size, n, device=dev),
                 r=torch.zeros(size, m, device=dev), cost=torch.zeros(size, device=dev), lam=torch.zeros(size, device=dev),
                 g=torch.zeros(size, n, device=dev), h=torch.eye(n, device=dev).repeat(size, 1, 1),
                 improved=torch.zeros(size, dtype=torch.bool, device=dev), moved=torch.zeros(size, device=dev),
                 live=torch.ones(size, dtype=torch.bool, device=dev),
                 used=torch.zeros(size, dtype=torch.long, device=dev))
        s['normal'] = _Replay(lambda: self._normal(f, s['x'], s['rows'], s['r']), dev)
        s['trial'] = _Replay(lambda: self._trial(f, s['x'], s['r'], s['cost'], s['lam'], s['g'], s['h'], s['improved'],
                                                 s['moved'], s['live'], s['lower'], s['upper'], s['rows']), dev)
        return s

    # ----- solve -----
    def solve(self, problems, seeds, clearance, iterations):
        """``problems``: (contacts, wrist or None, x_hint or None), all with the same number of contacts."""
        torch, s = self.torch, self.solver
        rows = []
        for contacts, wrist, x_hint in problems:
            contacts = sorted(contacts, key=lambda c: c.digit)
            digits = [c.digit for c in contacts]
            if wrist is None:
                wrist = s.neutral.copy()
                wrist[0] = np.mean([s.regions.centre(c.pitch)[0] for c in contacts]) \
                    - np.mean([s._reach_x(d) for d in digits])
            if x_hint is None:
                x_hint = wrist[0]
            depth_map = np.zeros(self.pitches, FLOAT)
            for c in contacts:
                depth_map[c.pitch] = c.depth
            centre = np.array([x_hint, s.neutral[1], s.neutral[2]], FLOAT)
            values = np.array([RELAXED[d] for d in DIGITS], FLOAT)
            shape = hand_shape(s.span(c.pitch for c in contacts))
            lower = np.r_[centre - WRIST_TRAVEL, YAW_BOUNDS[0], ROLL_BOUNDS[0], *[self.lower[d] for d in digits]]
            upper = np.r_[centre + WRIST_TRAVEL, YAW_BOUNDS[1], ROLL_BOUNDS[1], *[self.upper[d] for d in digits]]
            x = np.clip(np.r_[wrist, 0., 0., *[values[d - 1] for d in digits]], lower, upper)
            rows.append(dict(digit=np.array(digits, np.int64) - 1, onehot=np.eye(5, dtype=FLOAT)[np.array(digits) - 1],
                             pitch=np.array([c.pitch for c in contacts], np.int64),
                             depth=np.array([c.depth for c in contacts], FLOAT), depth_map=depth_map, values=values,
                             prior_wrist=np.r_[x_hint, s.neutral[1], s.neutral[2], 0., 0.],
                             stretch=stretched(c.pitch for c in contacts), lower=lower, upper=upper, x=x,
                             shape=np.array([shape[d] for d in digits], FLOAT)))
        n_seeds = len(seeds)
        # problem-major, seed-minor rows
        stack = lambda k, dtype=FLOAT: torch.as_tensor(np.repeat(np.stack([r[k] for r in rows]).astype(dtype), n_seeds, 0),
                                                       device=self.dev)
        c = dict(digit=stack('digit', np.int64), onehot=stack('onehot'), pitch=stack('pitch', np.int64),
                 depth=stack('depth'), depth_map=stack('depth_map'), values=stack('values'),
                 prior_wrist=stack('prior_wrist'), stretch=stack('stretch', bool), shape=stack('shape'))
        lower, upper, x = stack('lower'), stack('upper'), stack('x')
        offsets = torch.as_tensor(np.tile(np.array(seeds, FLOAT), (len(rows), 1)), device=self.dev)
        x[:, 0] += offsets[:, 1]
        x[:, 1] += offsets[:, 2]
        x[:, 3] = offsets[:, 0]
        x = torch.minimum(torch.maximum(x, lower), upper)

        def residual(prior, clear):
            f = lambda v, idx: self.residuals(v, {k: t[idx] for k, t in c.items()}, prior, clear)
            f.key = (len(rows[0]['digit']), clear)       # what a pass's memory per row depends on
            return f

        with torch.no_grad():
            x, used = self._lm(residual(.03, False), x, lower, upper, SEED_ITERATIONS)
            if clearance:
                x, more = self._lm(residual(.03, True), x, lower, upper, SEED_ITERATIONS)
                used = used + more
            r = self._eval(residual(.03, clearance), x, torch.arange(len(x), device=self.dev))
        cost = (r * r).sum(1).reshape(len(rows), n_seeds).cpu().numpy()
        used = used.reshape(len(rows), n_seeds).cpu().numpy()
        # PoseSolver.solve tries the seeds in order and stops at the first close one: the best seed so far
        pick, spent = [], []
        for k in range(len(rows)):
            good = np.nonzero(cost[k] < 1e-4)[0]
            last = good[0] if len(good) else n_seeds - 1
            pick.append(k * n_seeds + int(np.argmin(cost[k, :last + 1])))
            spent.append(int(used[k, :last + 1].sum()))
        pick_t = torch.as_tensor(pick, device=self.dev)
        c = {k: t[pick_t] for k, t in c.items()}
        x, lower, upper = x[pick_t], lower[pick_t], upper[pick_t]
        with torch.no_grad():
            x, a = self._lm(residual(.03, clearance), x, lower, upper, iterations)
            x, b = self._lm(residual(1., clearance), x, lower, upper, iterations // 2)
            _, mats, palm = self._pose(x, c)
            errors, targets, worst = [], [], []
            no_brush = torch.zeros_like(c['stretch'])
            for i in range(c['digit'].shape[1]):
                digit = c['digit'][:, i]
                point = self._contact_point(self._points(self.pad, digit, mats, palm), self.pad['mask'][digit])
                errors.append(self._region(c['pitch'][:, i], c['depth'][:, i], point).abs().amax(1))
                targets.append(point)
                pen = self._penetration(self._points(self.clear, digit, mats, palm), self.clear['mask'][digit],
                                        c['depth_map'], c['pitch'][:, i], no_brush)
                worst.append(pen.amax(1))
        x = x.cpu().numpy()
        errors, worst = torch.stack(errors, 1).cpu().numpy(), torch.stack(worst, 1).cpu().numpy()
        targets = torch.stack(targets, 1).cpu().numpy()
        spent = np.array(spent) + (a + b).cpu().numpy()
        poses = []
        for k, row in enumerate(rows):
            digits = row['digit'] + 1
            values = {d: row['values'][d - 1].copy() for d in DIGITS}
            values.update({int(d): x[k, 5 + 4 * i:9 + 4 * i].copy() for i, d in enumerate(digits)})
            error = float(errors[k].max())
            poses.append(HandPose(x[k, :3].copy(), float(x[k, 3]), float(x[k, 4]), values, error,
                                  error <= CONTACT_TOLERANCE * 2.5, int(spent[k]),
                                  {int(d): targets[k, i].copy() for i, d in enumerate(digits)},
                                  float(worst[k].max()) if s.heightfield is not None else 0.))
        return poses
