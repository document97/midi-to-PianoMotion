"""Selected preview clock from source MIDI events, independent of finger gates."""
import math
from .damper_open_tail import DamperClock


def selected_preview_clock(timeline, plan, duration):
    if plan.get('musical_damper_policy')!='final_open_source_interval':
        raise ValueError('New preview requires an explicitly declared open-tail policy')
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Positive finite preview duration required')
    if plan.get('source_note_inventory') != timeline['notes']:
        raise ValueError('Preview and plan have different source inventories')
    by_id = {n['id']: n for n in timeline['notes']}
    if len(by_id) != len(timeline['notes']):
        raise ValueError('Duplicate source note identity')
    damper = DamperClock(timeline)
    rows = []
    identities = set()
    for selected in plan['notes']:
        note_id = selected['id']
        if note_id in identities or note_id not in by_id:
            raise ValueError('Duplicate or unknown selected note identity')
        identities.add(note_id)
        raw = by_id[note_id]
        fields = ('source_event', 'pitch', 'velocity', 'channel',
                  'start', 'end', 'start_tick', 'end_tick')
        if any(selected.get(k) != raw.get(k) for k in fields):
            raise ValueError('Selected preview changed source music: ' + str(note_id))
        if raw['start'] >= duration:
            continue
        contract = damper.release_contract(raw, selected['physical_release'])
        if not contract['model_resolved'] or not contract['damping_compatible']:
            raise ValueError('Selected physical release violates source damper clock: ' + str(note_id))
        closure = contract['expected_damper_closure']
        rows.append(dict(id=note_id, pitch=raw['pitch'], velocity=raw['velocity'],
                         channel=raw['channel'], start=raw['start'], midi_note_off=raw['end'],
                         physical_release=selected['physical_release'], damper_closure=closure,
                         hold_seconds=(duration if closure is None else closure)-raw['start'],
                         preview_horizon=duration if closure is None else None,
                         open_pedal_tail=closure is None))
    return rows
