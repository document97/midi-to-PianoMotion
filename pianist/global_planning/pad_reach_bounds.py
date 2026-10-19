"""Conservative necessary hand-span bounds for a fixed rigid-bone LBS rig.

Arbitrary bone rotations are allowed, so ROM and finger direction can only
make the actual span smaller. Unknown rig conditions disable pruning.
"""
import math


def impossible_hand_pairs(notes, keyboard, radii, numerical_reserve=.001):
    if not math.isfinite(numerical_reserve) or numerical_reserve < 0:
        raise ValueError('Nonnegative finite bound reserve required')
    failures = []
    for i, left in enumerate(notes):
        for right in notes[i+1:]:
            if left['hand'] != right['hand']:
                continue
            hand = left['hand']
            r1 = radii[hand].get(str(left['finger']))
            r2 = radii[hand].get(str(right['finger']))
            if r1 is None or r2 is None:
                continue
            a, b = keyboard[str(left['pitch'])], keyboard[str(right['pitch'])]
            gaps = [max(0., a['bounds_min'][axis]-b['bounds_max'][axis],
                            b['bounds_min'][axis]-a['bounds_max'][axis]) for axis in (0, 1)]
            required = math.hypot(*gaps)
            maximum = r1+r2+2*numerical_reserve
            if required > maximum:
                failures.append(dict(hand=hand, ids=[left['id'], right['id']],
                    fingers=[left['finger'], right['finger']],
                    minimum_key_roof_separation_mm=required*1000,
                    conservative_pad_pair_span_mm=maximum*1000))
    return failures
