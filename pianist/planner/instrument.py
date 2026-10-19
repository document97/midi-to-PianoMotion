"""Keyboard and fixed-palm hand tables for O(1) geometric screens.

Each table reproduces one existing necessary condition exactly:
- pad pair span: global_planning.pad_reach_bounds.impossible_hand_pairs
- fixed-palm pair span: global_planning.rotation_coverage.impossible_fixed_palm_pairs
- wrist interval: global_planning.transport_feasibility.wrist_interval
"""
import math

import numpy as np

LOWEST_PITCH = 21
KEY_COUNT = 88
HANDS = ('L', 'R')
FINGERS = (1, 2, 3, 4, 5)


class Instrument:
    def __init__(self, keyboard, envelopes, radii, numerical_reserve=.001):
        pitches = sorted(int(p) for p in keyboard)
        if pitches != list(range(LOWEST_PITCH, LOWEST_PITCH + KEY_COUNT)):
            raise ValueError('An 88-key keyboard from A0 is required')
        rows = [keyboard[str(p)] for p in pitches]
        self.low = np.array([r['bounds_min'][:2] for r in rows], dtype=float)
        self.high = np.array([r['bounds_max'][:2] for r in rows], dtype=float)
        self.center_x = np.array([r['center_x'] for r in rows], dtype=float)
        # Same black-key test as the existing proposal heuristic.
        self.is_black = np.array([r['bounds_min'][1] > -.025 for r in rows])
        self.wrist_low = {}
        self.wrist_high = {}
        self.pair_ok = {}
        for hand in HANDS:
            low = np.full((6, KEY_COUNT), -math.inf)
            high = np.full((6, KEY_COUNT), math.inf)
            for finger in FINGERS:
                envelope = envelopes[hand].get(str(finger))
                if envelope is None:
                    continue
                low[finger] = self.low[:, 0] - envelope['high_x']
                high[finger] = self.high[:, 0] - envelope['low_x']
            self.wrist_low[hand], self.wrist_high[hand] = low, high
            self.pair_ok[hand] = self._pair_table(envelopes[hand], radii[hand], numerical_reserve)

    def _pair_table(self, envelopes, radii, reserve):
        gap_x = np.maximum(0., np.maximum(self.low[:, None, 0] - self.high[None, :, 0],
                                          self.low[None, :, 0] - self.high[:, None, 0]))
        gap_y = np.maximum(0., np.maximum(self.low[:, None, 1] - self.high[None, :, 1],
                                          self.low[None, :, 1] - self.high[:, None, 1]))
        separation = np.hypot(gap_x, gap_y)
        table = np.ones((6, KEY_COUNT, 6, KEY_COUNT), dtype=bool)
        for f1 in FINGERS:
            for f2 in FINGERS:
                ok = np.ones((KEY_COUNT, KEY_COUNT), dtype=bool)
                r1, r2 = radii.get(str(f1)), radii.get(str(f2))
                if r1 is not None and r2 is not None:
                    ok &= separation <= r1 + r2 + 2 * reserve
                a, b = envelopes.get(str(f1)), envelopes.get(str(f2))
                if a is not None and b is not None:
                    maximum = max(abs(a['high_x'] - b['low_x']), abs(b['high_x'] - a['low_x']))
                    ok &= gap_x <= maximum
                table[f1, :, f2, :] = ok
        return table

    def index(self, pitch):
        return pitch - LOWEST_PITCH

    def wrist_interval(self, hand, finger, pitch):
        i = pitch - LOWEST_PITCH
        return self.wrist_low[hand][finger, i], self.wrist_high[hand][finger, i]

    def pair_feasible(self, hand, f1, p1, f2, p2):
        return bool(self.pair_ok[hand][f1, p1 - LOWEST_PITCH, f2, p2 - LOWEST_PITCH])
