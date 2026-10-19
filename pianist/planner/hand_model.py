"""Hand model interface used by the search.

The search never manipulates wrist coordinates itself; it asks the hand model
to place a hand on a fingered chord from its previous pose. This is the seam
where note planning and motion merge: a motion-aware model can implement the
same methods by solving the actual wrist and finger pose (seeded from the
previous pose and cached by ``(hand, previous.signature(), keys)``), so poses
on the chosen path become motion keyframes instead of being solved again.

``GeometricHandModel`` implements the interface with the fixed-palm tables
(O(1) per key); its pose is the wrist x coordinate only.
"""
from dataclasses import dataclass

from .instrument import LOWEST_PITCH

POSE_RESOLUTION_METRES = .005   # poses closer than this are the same search state


@dataclass(frozen=True)
class WristPose:
    x: float

    def signature(self):
        return round(self.x / POSE_RESOLUTION_METRES)


@dataclass(frozen=True)
class Placement:
    pose: object
    travel_metres: float        # wrist path length from the previous pose
    window: tuple               # admissible wrist range used (model specific)


class GeometricHandModel:
    def __init__(self, instrument):
        self.instrument = instrument

    # --- keyboard facts used by musical costs ---
    def key_center(self, pitch):
        return self.instrument.center_x[pitch - LOWEST_PITCH]

    def is_black(self, pitch):
        return bool(self.instrument.is_black[pitch - LOWEST_PITCH])

    # --- necessary screens ---
    def pair_feasible(self, hand, f1, p1, f2, p2):
        return self.instrument.pair_feasible(hand, f1, p1, f2, p2)

    def key_window(self, hand, finger, pitch):
        """Wrist range in which ``finger`` can hold ``pitch``."""
        return self.instrument.wrist_interval(hand, finger, pitch)

    def chord_window(self, hand, keys):
        low, high = -float('inf'), float('inf')
        for finger, pitch in keys:
            a, b = self.key_window(hand, finger, pitch)
            low, high = max(low, a), min(high, b)
        return None if low > high else (low, high)

    # --- placement ---
    def place(self, hand, previous, keys):
        """Least-travel pose holding every (finger, pitch) in ``keys``; None if impossible."""
        window = self.chord_window(hand, keys)
        if window is None:
            return None
        low, high = window
        if previous is None:
            return Placement(WristPose((low + high) / 2), 0., window)
        x = min(max(previous.x, low), high)
        return Placement(WristPose(x), abs(x - previous.x), window)

    def hands_ordered(self, left, right):
        """Existing non-crossing fixed-palm style."""
        if left is None or right is None:
            return True
        return left.x <= right.x + 1e-9


class RigHandModel(GeometricHandModel):
    """Fixed-palm wrist windows, with two-finger feasibility of the real rig.

    ``pair_table[hand, f1-1, f2-1, p1 % 12, p2-p1+max]`` comes from the whole-hand
    pose solver that also generates the motion (performer.reach_table), so the
    planner never chooses a chord shape the animated hand cannot hold.
    """

    def __init__(self, instrument, pair_table):
        super().__init__(instrument)
        self.pair_table = pair_table
        self.max_interval = (pair_table.shape[-1] - 1) // 2

    def pair_feasible(self, hand, f1, p1, f2, p2):
        interval = p2 - p1
        if f1 == f2 or interval == 0 or abs(interval) > self.max_interval:
            return False
        return bool(self.pair_table['LR'.index(hand), f1 - 1, f2 - 1, p1 % 12, interval + self.max_interval])

    def chord_window(self, hand, keys):
        for i, (f1, p1) in enumerate(keys):
            for f2, p2 in keys[i + 1:]:
                if not self.pair_feasible(hand, f1, p1, f2, p2):
                    return None
        return super().chord_window(hand, keys)
