"""The damper pedal of the source (CC64) as a pedal depth curve for the foot (no Blender).

The damper lifts or falls halfway through the pedal stroke, so each stroke is centred on its MIDI event: the
sound changes exactly at the source time and the foot starts moving half a stroke earlier. A quick
pedal change (up and down again within a stroke) only half-lifts the pedal, as a real foot does.
"""
import numpy as np

from midi_input.damper import DamperClock

STROKE_SECONDS = .08       # pedal travel, down or up


def damper_intervals(timeline):
    """(n, 2) seconds the damper pedal is down, over all channels; an interval still open at the end has end inf."""
    rows = [(begin, np.inf if end is None else end)
            for intervals in DamperClock(timeline).intervals.values() for begin, end in intervals]
    return np.array(sorted(rows), float).reshape(-1, 2)


def _stroke(x):
    s = np.clip(x, 0., 1.)
    return s * s * (3. - 2. * s)


def damper_depth(intervals, times):
    """Pedal depth 0 (up) .. 1 (down) at ``times``."""
    times = np.asarray(times, float)
    depth = np.zeros(len(times))
    for begin, end in intervals:
        down = _stroke((times - begin) / STROKE_SECONDS + .5)
        up = _stroke((end - times) / STROKE_SECONDS + .5) if np.isfinite(end) else 1.
        depth = np.maximum(depth, np.minimum(down, up))
    return depth
