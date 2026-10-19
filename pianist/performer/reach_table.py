"""Two-finger feasibility of the real rig, for the planner's hand model.

For every hand, finger pair and pitch interval, the pose solver (pose.py) decides
whether both pads can rest on their keys at once. The wrist is free, so the
answer depends only on the keyboard pattern: the lower key's pitch class and the
interval. Computed in parallel once per rig and cached next to it.
"""
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import hashlib
from pathlib import Path

import numpy as np

from .kinematics import DIGITS
from .pose import RELAXED, hand_solvers

MAX_INTERVAL = 16          # semitones; wider pairs are never one hand
REFERENCE_OCTAVE = {'L': 48, 'R': 72}   # C3 / C5: where each hand's patterns are sampled
PAIRS = tuple(combinations(range(1, 6), 2))

_worker = {}


def _init(rig_path):
    from .kinematics import Rig
    from .pose import hand_solvers
    rig = Rig(rig_path)
    _worker.update(rig=rig, solvers=hand_solvers(rig))


def _solve(case):
    from .anchors import KEY_TRAVEL
    from .pose import WIDE_SEEDS, Contact
    hand, f1, p1, f2, p2 = case
    solver = _worker['solvers'][hand]
    pose = solver.solve([Contact(f1, p1, KEY_TRAVEL), Contact(f2, p2, KEY_TRAVEL)], seeds=WIDE_SEEDS, clearance=False)
    return case, pose.ok, pose.contact_error


def cases():
    for hand in 'LR':
        base = REFERENCE_OCTAVE[hand]
        for f1, f2 in PAIRS:
            for pc in range(12):
                for interval in range(-MAX_INTERVAL, MAX_INTERVAL + 1):
                    if interval:
                        yield hand, f1, base + pc, f2, base + pc + interval


def compute(rig_path, workers, partial=None, stop_fraction=1.):
    """Solve every case; progress is saved to ``partial`` after each block so a stopped run resumes."""
    table = np.zeros((2, 5, 5, 12, 2 * MAX_INTERVAL + 1), dtype=bool)
    errors = np.full(table.shape, np.inf)
    jobs = list(cases())
    done = np.zeros(len(jobs), dtype=bool)
    if partial is not None and Path(partial).exists():
        saved = np.load(partial)
        table, errors, done = saved['table'], saved['errors'], saved['done']
    block = max(1, len(jobs) // 20)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(rig_path),)) as pool:
        for start in range(0, len(jobs), block):
            todo = [i for i in range(start, min(start + block, len(jobs))) if not done[i]]
            for i, ((hand, f1, p1, f2, p2), ok, error) in zip(todo, pool.map(_solve, [jobs[i] for i in todo], chunksize=16)):
                h = 'LR'.index(hand)
                for a, pa, b, pb in ((f1, p1, f2, p2), (f2, p2, f1, p1)):
                    index = (h, a - 1, b - 1, pa % 12, pb - pa + MAX_INTERVAL)
                    table[index], errors[index] = ok, error
                done[i] = True
            if partial is not None:
                np.savez_compressed(partial, table=table, errors=errors, done=done)
            print(f'pair table {100 * done.sum() // len(jobs)}%', flush=True)
            if done.mean() >= stop_fraction:
                break
    return table, errors, bool(done.all())


FINGERPRINT_TOLERANCE = 1e-4    # m: geometry closer than this to a cached table's reuses it


def fingerprint(rig_path):
    """What decides feasibility, as numbers: pad and skin points of probe poses, palm frames, key regions
    and tops; plus a digest of the solver code. The arm, torso and seating are not part of it."""
    from .kinematics import Rig
    rig = Rig(rig_path)
    solvers = hand_solvers(rig)
    parts = []
    probes = [RELAXED, {d: RELAXED[d] + np.array([8., -6., 20., 15.]) for d in DIGITS},
              {d: RELAXED[d] + np.array([-10., 12., -15., -10.]) for d in DIGITS}]
    for h, solver in solvers.items():
        for values in probes:
            for d in DIGITS:
                parts += [solver.local_pad(d, values).ravel(), solver.local_clearance(d, values).ravel()]
        for vector in ((0., 0.), (10., -7.)):
            parts.append(solver.palm(solver.neutral, *vector).ravel())
        parts.append(solver.neutral)
        for p in sorted(solver.regions.rows):
            row = solver.regions.rows[p]
            parts.append(np.r_[row['top'], row['wide'], row['narrow'], row['split'], row['y']])
    top = solvers['L'].heightfield.top
    parts.append(np.where(np.isfinite(top), top, -1.).ravel())
    from .anchors import KEY_TRAVEL        # the pairs are solved with the keys fully down
    parts.append(np.array([KEY_TRAVEL]))
    code = hashlib.sha256()
    for name in ('kinematics.py', 'pose.py', 'reach_table.py'):
        code.update((Path(__file__).parent / name).read_bytes())
    return np.concatenate(parts), code.hexdigest()[:16]


def cache_path(rig_path):
    """A cached table for this geometry and solver code, or the path a new one is saved to."""
    values, code = fingerprint(rig_path)
    folder = Path(rig_path).parent
    for path in sorted(folder.glob('pairs_*.npz')):
        if path.name.endswith('.partial.npz'):
            continue
        with np.load(path) as data:
            if 'fingerprint' not in data or str(data['code']) != code:
                continue
            saved = data['fingerprint']
            if saved.shape == values.shape and np.abs(saved - values).max() < FINGERPRINT_TOLERANCE:
                return path
    digest = hashlib.sha256(np.round(values, 5).tobytes()).hexdigest()[:8]
    return folder / f'pairs_{code[:8]}{digest}.npz'


def load_or_compute(rig_path, workers=8, stop_fraction=1.):
    """Cached table; an unfinished computation resumes from its partial file."""
    path = cache_path(rig_path)
    if path.exists():
        data = np.load(path)
        return data['table'], data['errors']
    partial = path.with_name(path.stem + '.partial.npz')
    table, errors, complete = compute(rig_path, workers, partial, stop_fraction)
    if not complete:
        raise SystemExit(f'pair table paused at {stop_fraction:.0%}; run again to resume ({partial.name})')
    values, code = fingerprint(rig_path)
    np.savez_compressed(path, table=table, errors=errors, fingerprint=values, code=np.array(code))
    partial.unlink(missing_ok=True)
    return table, errors


if __name__ == '__main__':
    # python -m performer.reach_table [workers] [stop_fraction]   (cwd: pianist)
    import sys
    from .pipeline import RIG
    load_or_compute(RIG,
                    int(sys.argv[1]) if len(sys.argv) > 1 else 16, float(sys.argv[2]) if len(sys.argv) > 2 else 1.)
    print('pair table complete', flush=True)
