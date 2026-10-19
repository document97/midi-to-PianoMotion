"""Screen mandatory crossed wrists under the current noncrossing style.

The driver has no coordinated arm-crossing/layering mode. This configurable
policy is not an anatomical impossibility or a reason to delete source notes.
"""
import math
from .transport_feasibility import wrist_interval


def arm_order_rejections(owners, keyboard, envelopes, minimum_order_m=0.):
    if not math.isfinite(minimum_order_m) or minimum_order_m < 0:
        raise ValueError('Finite nonnegative wrist-order preference required')
    banks = {hand: [n for n in owners if n['hand'] == hand] for hand in ('L', 'R')}
    if not all(banks.values()):
        return []
    ranges = {}
    for hand, notes in banks.items():
        intervals = [wrist_interval(n, keyboard, envelopes) for n in notes]
        if any(interval is None for interval in intervals):
            return []  # Unknown coverage must not exclude a candidate.
        ranges[hand] = [max(interval[0] for interval in intervals),
                        min(interval[1] for interval in intervals)]
        if ranges[hand][0] > ranges[hand][1]:
            return []  # The separate same-hand span screen handles this.
    if ranges['L'][0] + minimum_order_m <= ranges['R'][1] + 1e-10:
        return []
    return [dict(kind='unsupported_crossed_wrist_style',
        ids=sorted(n['id'] for n in owners), wrist_intervals_m=ranges,
        minimum_order_m=minimum_order_m,
        minimum_forced_crossing_mm=(ranges['L'][0] - ranges['R'][1]) * 1000,
        necessary_for_this_owner_pair=False, whole_note_unreachability_proved=False,
        anatomical_impossibility_proved=False)]

