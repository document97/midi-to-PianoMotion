"""Preserve source-final open CC64 intervals without inventing pedal-off.

An already released key and its MIDI note-off have the same future damping
when both lie in the final open pedal interval and precede the source EOF.
The damping time remains None: no finite closure is fabricated. Existing
finite-closure contracts and unsupported channel controls are unchanged.
"""
import math
from .damper import DamperClock as FiniteDamperClock, EPSILON


class DamperClock(FiniteDamperClock):
    def __init__(self, timeline):
        super().__init__(timeline)
        self.source_end = float(timeline['duration'])
        if not math.isfinite(self.source_end) or self.source_end < 0:
            raise ValueError('Finite source duration required for open-tail equivalence')

    def release_contract(self, note, physical_release):
        contract = super().release_contract(note, physical_release)
        if contract['model_resolved']:
            return contract
        channel = note['channel']
        if any(row['channel'] == channel for row in self.unsupported):
            return contract
        if (contract['expected_damper_closure'] is not None
                or contract['planned_damper_closure'] is not None
                or physical_release > self.source_end+EPSILON
                or note['end'] > self.source_end+EPSILON):
            return contract
        # Both None values must name the same explicit final open interval,
        # not two unrelated unresolved conditions or a future key release.
        interval = next((begin for begin, end in self.intervals[channel]
                         if end is None and begin <= min(note['end'], physical_release)+EPSILON), None)
        if interval is None:
            return contract
        return dict(contract, earliest_equivalent_release=max(note['start'], interval),
                    damping_compatible=True, model_resolved=True,
                    shortened_damper_interval_seconds=0.,
                    open_tail_policy='same-final-open-interval-before-source-end',
                    eof_is_not_pedal_off=True,
                    closure_interpretation='No pedal closure exists in the source; both released keys await the same future channel closure.')
