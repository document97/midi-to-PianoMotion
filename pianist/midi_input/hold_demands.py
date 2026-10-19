"""Musical hold obligations for joint fingering and hand placement planning.

Only the music decides them: no physical limits, finger assignments or
piece-specific overrides. Hand labels from the MIDI tracks are kept for
reference; the planner chooses the actual, reachable releases later.
"""
from .damper import DamperClock, EPSILON
from .timeline import TempoClock


def build_hold_demands(timeline, keyboard=None):
    damper = DamperClock(timeline)
    clock = TempoClock(timeline['ticks_per_quarter'], timeline['tempo_events'])
    notes = []
    unresolved = []
    for note in timeline['notes']:
        if note['end'] is None:
            unresolved.append(dict(id=note['id'], reason='Unclosed MIDI note'))
            continue
        contract = damper.release_contract(note, note['end'])
        release = contract['earliest_equivalent_release']
        if not contract['model_resolved'] or release is None:
            unresolved.append(dict(id=note['id'], reason='Unsupported or unresolved damper state'))
            continue
        notes.append(dict(id=note['id'], source_event=note['source_event'], pitch=note['pitch'],
            velocity=note['velocity'], channel=note['channel'], start=note['start'], end=note['end'],
            start_tick=note['start_tick'], end_tick=note['end_tick'],
            hand=note.get('hand'), hand_origin=note.get('hand_origin'),
            earliest_equivalent_release=release,
            expected_damper_closure=contract['expected_damper_closure']))
    groups = {}
    for note in notes:
        groups.setdefault(note['start_tick'], []).append(note)
    windows = []
    for tick, incoming in sorted(groups.items()):
        start = clock.seconds_at_tick(tick)
        required = [note for note in notes if note['start'] < start - EPSILON
                    and note['earliest_equivalent_release'] > start + EPSILON]
        owners = required + incoming
        hands = {}
        for hand in ('L', 'R', None):
            selected = [note for note in owners if note['hand'] == hand]
            pitches = sorted({note['pitch'] for note in selected})
            span = None
            if keyboard is not None and pitches and all(str(p) in keyboard for p in pitches):
                centers = [(keyboard[str(p)]['bounds_min'][0] + keyboard[str(p)]['bounds_max'][0]) / 2
                           for p in pitches]
                span = (max(centers) - min(centers)) * 1000
            hands[hand or 'unassigned'] = dict(note_ids=[note['id'] for note in selected],
                distinct_pitches=pitches, distinct_keys=len(pitches), key_center_span_mm=span)
        windows.append(dict(start_tick=tick, time=start, quarter_bpm=clock.quarter_bpm_at_seconds(start),
            incoming_ids=[note['id'] for note in incoming], required_held_ids=[note['id'] for note in required],
            hands=hands, total_distinct_keys=len({note['pitch'] for note in owners})))
    return dict(schema='musical-hold-demands', source=timeline['source'],
        source_sha256=timeline['source_sha256'], input_notes=len(timeline['notes']),
        resolved_notes=len(notes), notes=notes, onset_windows=windows, unresolved=unresolved,
        unsupported_controls=damper.unsupported, notes_removed=0,
        global_fingering_planned=False, physical_releases_planned=False,
        complete_performance_verified=False,
        interpretation='Earliest releases preserve binary MIDI damper closure. Incoming strikes remain obligations even if pedal permits immediate release. Hand labels are input evidence, not a new allocation. Actual stroke/return duration, all five fingers, reachability and continuous motion must be solved jointly.')
