"""Binary MIDI 1.0 damper constraints, separate from physical reachability.

CC64 <=63 is off, >=64 is on. Original same-tick event order is retained.
This models when a damper may close, not acoustic decay, half pedal, repedal
resonance, physical key travel, or a proof that a hand can retain a note.
"""
from bisect import bisect_right
import math

EPSILON = 1e-8
class DamperClock:
    def __init__(self, timeline):
        self.events = {channel: [] for channel in range(16)}
        self.intervals = {channel: [] for channel in range(16)}
        self.identities = {}
        self.unsupported = []
        open_since = {}
        state = {channel: False for channel in range(16)}
        for index, event in enumerate(timeline['events']):
            self.identities[f"{event['track']}:{event['order']}"] = index
            message = event['message']
            if message['type'] != 'control_change':
                continue
            channel, control = message['channel'], message['control']
            if control in (66, 120, 121, 123):
                self.unsupported.append(dict(channel=channel, control=control,
                    seconds=event['seconds'], event_index=index))
            if control != 64:
                continue
            down = message['value'] >= 64
            point = (event['seconds'], index)
            self.events[channel].append((point, down))
            if down and not state[channel]:
                open_since[channel] = event['seconds']
            if not down and state[channel]:
                self.intervals[channel].append((open_since.pop(channel), event['seconds']))
            state[channel] = down
        for channel, begin in open_since.items():
            self.intervals[channel].append((begin, None))
        self.points = {channel: [row[0] for row in rows] for channel, rows in self.events.items()}

    def damping_time(self, channel, release, event_index=math.inf):
        """First damper closure after a key release, or unknown beyond EOF."""
        if not math.isfinite(release) or release < 0:
            raise ValueError('A finite nonnegative key release is required')
        rows = self.events[channel]
        position = bisect_right(self.points[channel], (release, event_index))
        if position == 0 or not rows[position - 1][1]:
            return release
        return next((point[0] for point, down in rows[position:] if not down), None)

    def release_contract(self, note, physical_release):
        if note['end'] is None:
            raise ValueError('An unclosed note has no finite release contract')
        if not math.isfinite(physical_release) or physical_release < note['start'] - EPSILON:
            raise ValueError('Physical release precedes its musical onset')
        channel, end = note['channel'], note['end']
        end_index = self.identities[note['note_off_event']]
        desired = self.damping_time(channel, end, end_index)
        actual = desired if abs(physical_release - end) <= EPSILON else self.damping_time(channel, physical_release)
        earliest = end
        if desired is not None:
            for begin, finish in self.intervals[channel]:
                if finish is not None and abs(finish - desired) <= EPSILON and begin <= end <= finish + EPSILON:
                    earliest = min(earliest, max(note['start'], begin))
        supported = not any(row['channel'] == channel for row in self.unsupported)
        resolved = supported and desired is not None and actual is not None
        compatible = resolved and abs(desired - actual) <= EPSILON
        return dict(channel=channel, midi_note_off=end, physical_release=physical_release,
                    key_gate_difference_seconds=physical_release - end,
                    expected_damper_closure=desired, planned_damper_closure=actual,
                    earliest_equivalent_release=earliest if desired is not None and supported else None,
                    damping_compatible=compatible, model_resolved=resolved,
                    shortened_damper_interval_seconds=max(0., desired - actual) if resolved else None)
