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


PEDAL_LEAD = .05                 # the pedal is down this long before a key lifts that it holds the sound of
PEDAL_GAP = .2                   # a pedal-up shorter than this between two extra pedals (keys lifted early) is not made


def _merged(rows, gap=0.):
    """(begin, end, extra) rows merged where they overlap (or lie closer than ``gap`` between two extra rows); a merged
    row is extra only when all of it is."""
    out = []
    for begin, end, extra in sorted(tuple(r) for r in rows):
        if out and begin <= out[-1][1] + (gap if extra and out[-1][2] else 0.):
            out[-1] = [out[-1][0], max(out[-1][1], end), out[-1][2] and extra]
        else:
            out.append([begin, end, extra])
    return out


def pedal_union(intervals, extra):
    """The score's damper intervals with ``extra`` (begin, end) ones for keys lifted early, merged. The score's own
    pedal changes stay however quick (legato pedalling lifts for a few hundredths of a second): a key lifts early only
    after the score's last pedal change before its end (coherence.free_releases), so no extra interval covers one."""
    score = _merged([(float(b), float(e), False) for b, e in np.asarray(intervals, float).reshape(-1, 2)])   # channels
    out = _merged(score + [(float(b), float(e), True) for b, e in extra], PEDAL_GAP)
    return np.array([r[:2] for r in out], float).reshape(-1, 2)


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
