"""The planner's per-finger wrist windows, sampled from the rig the motion is solved with.

python -m performer.hand_ranges   (cwd: pianist) -> inputs/instrument/pad_rom_envelopes.json

For each hand and finger: the sideways (keyboard x) offset of the pad's contact point from the wrist over every
pose the pose solver may use -- each finger value on a grid over the rig's own bounds, the palm turned over
pose.YAW_BOUNDS and rolled over pose.ROLL_BOUNDS. The planner puts the wrist within [key - high_x, key - low_x]
for a finger on a key, so a chord the rig can hold (reach_table) always has a wrist position. The extremes are
found on a grid (every bound included) and then refined by a bounded optimisation over the four values, yaw and roll.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .kinematics import DIGITS, Rig
from .pose import RELAXED, ROLL_BOUNDS, YAW_BOUNDS, hand_solvers

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'inputs' / 'instrument' / 'pad_rom_envelopes.json'
VALUE_STEPS = 9            # grid points per finger value, bounds included
YAW_STEPS, ROLL_STEPS = 5, 3


def sample(rig_path):
    rig = Rig(rig_path)
    solvers = hand_solvers(rig)
    palms = {h: [s.palm(s.neutral, yaw, roll) for yaw in np.linspace(*YAW_BOUNDS, YAW_STEPS)
                 for roll in np.linspace(*ROLL_BOUNDS, ROLL_STEPS)] for h, s in solvers.items()}
    envelopes = {}
    for h, s in solvers.items():
        envelopes[h] = {}
        for d in DIGITS:
            finger = rig.fingers[h][d]
            axes = [np.linspace(lo, hi, VALUE_STEPS) for lo, hi in zip(finger.lower, finger.upper)]
            values = {k: RELAXED[k] for k in DIGITS}
            offsets = np.empty((VALUE_STEPS,) * 4 + (len(palms[h]),))
            for index in itertools.product(range(VALUE_STEPS), repeat=4):
                values[d] = np.array([axes[k][i] for k, i in enumerate(index)])
                local = s.local_pad(d, values)
                for p, palm in enumerate(palms[h]):
                    offsets[index + (p,)] = s.contact_point((palm @ local)[:3].T)[0] - s.neutral[0]
            lower = np.r_[finger.lower, YAW_BOUNDS[0], ROLL_BOUNDS[0]]
            upper = np.r_[finger.upper, YAW_BOUNDS[1], ROLL_BOUNDS[1]]

            def offset(x):
                values[d] = x[:4]
                return s.contact_point(s.pad_world(d, values, s.palm(s.neutral, x[4], x[5])))[0] - s.neutral[0]

            def extreme(sign):
                *index, p = np.unravel_index(np.argmin(sign * offsets), offsets.shape)
                yaw, roll = np.linspace(*YAW_BOUNDS, YAW_STEPS)[p // ROLL_STEPS], np.linspace(*ROLL_BOUNDS, ROLL_STEPS)[p % ROLL_STEPS]
                x0 = np.r_[[axes[k][i] for k, i in enumerate(index)], yaw, roll]
                best = minimize(lambda x: sign * offset(x), x0, bounds=list(zip(lower, upper)), method='L-BFGS-B')
                return min(sign * offsets.min(), float(best.fun)) * sign

            envelopes[h][str(d)] = dict(low_x=extreme(1.), high_x=extreme(-1.),
                                        sampled_low_x=float(offsets.min()), sampled_high_x=float(offsets.max()))
    return envelopes


def main():
    from .pipeline import RIG
    envelopes = sample(RIG)
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = dict(schema='pad-rom-envelopes', envelopes=envelopes,
               interpretation='Pad contact point x minus wrist x over the rig finger bounds, palm yaw over '
                              f'{list(YAW_BOUNDS)} deg and roll over {list(ROLL_BOUNDS)} deg: grid extremes refined by '
                              'bounded optimisation (sampled_* are the grid values).',
               rig=str(Path(RIG).name), rig_sha256=hashlib.sha256(Path(RIG).read_bytes()).hexdigest(), tool_sha256=code,
               grid=dict(value_steps=VALUE_STEPS, yaw_steps=YAW_STEPS, roll_steps=ROLL_STEPS))
    OUT.write_text(json.dumps(out, indent=2), encoding='utf-8')
    for h in 'LR':
        print(h, {d: (round(e['low_x'] * 1000, 1), round(e['high_x'] * 1000, 1)) for d, e in envelopes[h].items()})
    print('written', OUT)


if __name__ == '__main__':
    main()
