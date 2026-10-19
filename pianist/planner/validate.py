"""Independent plan check with the reach rules in global_planning.

Nothing here reuses the planner's tables, so a table or search bug shows up
as a violation instead of being silently trusted.
"""
from global_planning.arm_order_policy import arm_order_rejections
from global_planning.pad_reach_bounds import impossible_hand_pairs
from global_planning.preparation_finger_order import violations as order_violations
from global_planning.rotation_coverage import impossible_fixed_palm_pairs
from global_planning.transport_feasibility import transport_rejections

from .music import KEY_RETURN_SECONDS


def validate(plan_notes, score, keyboard, envelopes, radii, hard_speed):
    """``plan_notes``: dicts with id, pitch, start, touch, hand, finger, physical_release."""
    problems = []
    by_index = {n['index']: n for n in plan_notes}
    ordered = sorted(plan_notes, key=lambda n: (n['start'], n['pitch']))
    for n in ordered:
        if not score.release_is_legal(n['index'], n['physical_release']):
            problems.append(dict(kind='damper_incompatible_release', id=n['id']))
        if n['physical_release'] < n['start'] - 1e-9:
            problems.append(dict(kind='release_before_onset', id=n['id']))
    # Same key: lift and return before the next touch (any hand).
    by_pitch = {}
    for n in ordered:
        by_pitch.setdefault(n['pitch'], []).append(n)
    for rows in by_pitch.values():
        for a, b in zip(rows, rows[1:]):
            if a['physical_release'] + KEY_RETURN_SECONDS > b['touch'] + 1e-9:
                problems.append(dict(kind='key_restrike_overlap', ids=[a['id'], b['id']]))
    # One finger owns one key at a time (touch .. lift + return).
    by_finger = {}
    for n in ordered:
        by_finger.setdefault((n['hand'], n['finger']), []).append(n)
    for rows in by_finger.values():
        for a, b in zip(rows, rows[1:]):
            if a['physical_release'] + KEY_RETURN_SECONDS > b['touch'] + 1e-9:
                problems.append(dict(kind='finger_double_booked', ids=[a['id'], b['id']]))
    # Simultaneous-owner screens at each onset, with the global_planning rules.
    starts = sorted({n['start'] for n in ordered})
    for t in starts:
        incoming = [n for n in ordered if abs(n['start'] - t) < 1e-9]
        touch = min(n['touch'] for n in incoming)
        owners = [n for n in ordered
                  if n['start'] <= t + 1e-9 and n['physical_release'] + KEY_RETURN_SECONDS > touch + 1e-9]
        for row in impossible_hand_pairs(owners, keyboard, radii):
            problems.append(dict(kind='pad_span', time=t, **row))
        for row in impossible_fixed_palm_pairs(owners, keyboard, envelopes):
            problems.append(dict(kind='fixed_palm_span', time=t, **row))
        for row in arm_order_rejections(owners, keyboard, envelopes):
            problems.append(dict(time=t, **row))
        for row in order_violations(owners, [n['id'] for n in incoming]):
            problems.append(dict(time=t, **row))
    # Wrist transport: necessary overlap and the configured hard speed limit.
    assigned = [dict(n) for n in ordered]
    for t in starts:
        ids = [n['id'] for n in ordered if abs(n['start'] - t) < 1e-9]
        window = [n for n in assigned if n['start'] <= t + 1e-9 and n['start'] > t - 4.]
        for row in transport_rejections(window, ids, keyboard, envelopes, maximum_average_speed=hard_speed):
            problems.append(dict(time=t, **row))
    return problems
