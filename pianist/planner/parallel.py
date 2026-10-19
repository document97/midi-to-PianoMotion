"""Segment-parallel planning with exact sequential stitching.

Target machines: 8 or 16 equal-performance cores. The score is cut into
``workers * segments_per_worker`` segments of equal note count (work is close
to proportional to notes), each planned from a free state after a warm-up
span. The stitcher replays segment decisions in order from the true state;
where a decision is infeasible at a seam, a short sequential beam search from
the true state repairs it. The result is one consistent plan; parallelism
never weakens a constraint.
"""
from concurrent.futures import ProcessPoolExecutor
import time

from .search import Planner, initial_state

SUPPORTED_WORKERS = (8, 16)
_WORKER = {}


def _init_worker(factory, factory_args, planner_kwargs):
    score, instrument = factory(*factory_args)
    _WORKER['planner'] = Planner(score, instrument, **planner_kwargs)


def _plan_segment(bounds):
    first, warm_first, last = bounds
    planner = _WORKER['planner']
    best = planner.run(planner.score.groups[warm_first:last])[0]
    return [d for d in best.decisions() if d.group >= first]


def segment_bounds(groups, count, warm_seconds):
    """(first, warm_first, last) group ranges with near-equal note counts."""
    total = sum(len(g.notes) for g in groups)
    cuts, running, target = [0], 0, total / count
    for i, g in enumerate(groups):
        running += len(g.notes)
        if running >= target * len(cuts) and i + 1 < len(groups) and len(cuts) < count:
            cuts.append(i + 1)
    cuts.append(len(groups))
    bounds = []
    for first, last in zip(cuts, cuts[1:]):
        warm = first
        while warm > 0 and groups[warm - 1].start > groups[first].start - warm_seconds:
            warm -= 1
        bounds.append((first, warm, last))
    return bounds


def stitch(planner, decisions, repair_groups=24):
    """Replay decisions from the true state; repair infeasible stretches."""
    groups = planner.score.groups
    state, index, repairs = initial_state(), 0, []
    while index < len(groups):
        successors = planner.expand(state, groups[index], forced=decisions[index].assignment)
        if successors:
            state = min(successors, key=lambda s: s.cost)
            index += 1
            continue
        stop = min(len(groups), index + repair_groups)
        state = planner.run(groups[index:stop], state)[0]
        repairs.append((groups[index].start, groups[stop - 1].start))
        for d in state.decisions():
            if index <= d.group < stop:
                decisions[d.group] = d
        index = stop
    return state, repairs


def plan_parallel(factory, factory_args, planner_kwargs=None, workers=8,
                  segments_per_worker=2, warm_seconds=4.):
    if workers not in SUPPORTED_WORKERS:
        raise ValueError(f'Planned for {SUPPORTED_WORKERS} equal cores, got {workers}')
    planner_kwargs = planner_kwargs or {}
    started = time.perf_counter()
    score, instrument = factory(*factory_args)
    planner = Planner(score, instrument, **planner_kwargs)
    bounds = segment_bounds(score.groups, workers * segments_per_worker, warm_seconds)
    decisions = [None] * len(score.groups)
    with ProcessPoolExecutor(workers, initializer=_init_worker,
                             initargs=(factory, factory_args, planner_kwargs)) as pool:
        for rows in pool.map(_plan_segment, bounds):
            for d in rows:
                decisions[d.group] = d
    searched = time.perf_counter() - started
    state, repairs = stitch(planner, decisions)
    return dict(state=state, planner=planner, segments=len(bounds), workers=workers,
                search_seconds=searched, total_seconds=time.perf_counter() - started, repairs=repairs)


def plan_serial(factory, factory_args, planner_kwargs=None):
    started = time.perf_counter()
    score, instrument = factory(*factory_args)
    planner = Planner(score, instrument, **(planner_kwargs or {}))
    state = planner.run(score.groups)[0]
    elapsed = time.perf_counter() - started
    return dict(state=state, planner=planner, segments=1, workers=1,
                search_seconds=elapsed, total_seconds=elapsed, repairs=[])
