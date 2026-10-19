"""Source notes, onset groups and legal physical-release bounds.

A key may be lifted before its MIDI note-off. Under the source damper (CC64)
from ``earliest_equivalent_release`` on (hold-demand proof, re-checked with
DamperClock) the sound is identical. An earlier lift loses only the sound
between the damper closure it causes and the source closure: a key lifted
while the pedal is down keeps sounding until the pedal rises. The search
prices that loss against omitting other notes and the report lists it.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from midi_input.damper_open_tail import DamperClock

KEY_RETURN_SECONDS = .072        # existing provisional key-return reserve
MINIMUM_PRESS_SECONDS = .030     # existing minimum key press
MINIMUM_DOWNSTROKE_SECONDS = .018  # existing wrist arrival deadline before onset
CHORD_WINDOW_SECONDS = .050       # onsets this close are fingered as one chord


def downstroke_seconds(velocity):
    """Key descent time (touch to keybed): ~110 ms pianissimo to ~20 ms fortissimo (velocity 30 .. 110)."""
    return float(np.clip(.110 - (velocity - 30) * .0011, .020, .110))


@dataclass(frozen=True)
class Note:
    index: int
    id: int
    pitch: int
    velocity: int
    start: float
    end: float
    touch: float
    pedal_release: float      # earliest lift with identical damping
    earliest_release: float   # earliest permitted lift (minimum press)
    group: int


@dataclass(frozen=True)
class Group:
    index: int
    start: float
    touch: float
    notes: tuple              # note indices, ascending pitch


class Score:
    def __init__(self, timeline, demands, end=None):
        self.timeline = timeline
        self.damper = DamperClock(timeline)
        raw_by_id = {n['id']: n for n in timeline['notes']}
        rows = sorted((d for d in demands['notes'] if end is None or d['start'] <= end + 1e-9),
                      key=lambda d: (d['start'], d['pitch'], d['id']))
        # Near-simultaneous onsets (rolled or humanised chords) are fingered
        # together; every note keeps its own source start and touch.
        group_of_tick, group, anchor, pitches = {}, -1, None, set()
        for d in rows:
            at = d['start']
            key = d['start_tick']
            if key in group_of_tick:
                pitches.add(d['pitch'])
                continue
            if anchor is None or at - anchor > CHORD_WINDOW_SECONDS or d['pitch'] in pitches:
                group, anchor, pitches = group + 1, at, set()
            group_of_tick[key] = group
            pitches.add(d['pitch'])
        self.raw = []
        notes = []
        for index, d in enumerate(rows):
            raw = raw_by_id[d['id']]
            at = d['start']
            floor = at + MINIMUM_PRESS_SECONDS
            pedal = d['earliest_equivalent_release']
            pedal = d['end'] if pedal is None else min(d['end'], max(pedal, floor))
            if not self.damper.release_contract(raw, pedal)['damping_compatible']:
                pedal = d['end']
            earliest = min(pedal, floor)
            notes.append(Note(index, d['id'], d['pitch'], d['velocity'], at, d['end'],
                              at - downstroke_seconds(d['velocity']), pedal, earliest, group_of_tick[d['start_tick']]))
            self.raw.append(raw)
        self.notes = tuple(notes)
        # Source damper closure per note; None (final open pedal) means the source end.
        self.source_end = self.damper.source_end
        self._closure = []
        for n, raw in zip(notes, self.raw):
            closure = self.damper.release_contract(raw, n.end)['expected_damper_closure']
            self._closure.append(self.source_end if closure is None else closure)
        members = {}
        for n in notes:
            members.setdefault(n.group, []).append(n.index)
        self.groups = tuple(Group(g, min(notes[i].start for i in m), min(notes[i].touch for i in m),
                                  tuple(sorted(m, key=lambda i: notes[i].pitch)))
                            for g, m in sorted(members.items()))

    def outer_voice_notes(self):
        """Indices of notes that are the highest or lowest source note sounding at their onset."""
        outer = set()
        for n in self.notes:
            sounding = [m.pitch for m in self.notes if m.start <= n.start + 1e-9 and m.end > n.start + 1e-9]
            if n.pitch >= max(sounding) or n.pitch <= min(sounding):
                outer.add(n.index)
        return outer

    def sound_end(self, note_index, release):
        """When the damper stops the string if the key is lifted at ``release``."""
        if release >= self.notes[note_index].end - 1e-9:
            return self._closure[note_index]
        closure = self.damper.damping_time(self.raw[note_index]['channel'], release)
        return self.source_end if closure is None else closure

    def sounding_seconds(self, note_index):
        """Source sound length: onset to the source damper closure."""
        return max(self._closure[note_index] - self.notes[note_index].start, 1e-6)

    def audible_shortening(self, note_index, release):
        """Seconds of sound lost by lifting at ``release`` (0 when damping is identical)."""
        if release >= self.notes[note_index].pedal_release - 1e-9:
            return 0.
        return max(0., self._closure[note_index] - self.sound_end(note_index, release))

    def release_is_legal(self, note_index, release):
        note = self.notes[note_index]
        return note.earliest_release - 1e-9 <= release <= note.end + 1e-9


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def hand_split_pitches(score, window_seconds=1.5, default=60.):
    """Per-group pitch dividing the two hands' registers (1-D two-means).

    Uses the notes sounding within +/- ``window_seconds``; a window without two
    separated clusters (spread under an octave) inherits the previous split.
    """
    starts = np.array([n.start for n in score.notes])
    pitches = np.array([n.pitch for n in score.notes], dtype=float)
    splits, previous = [], default
    for group in score.groups:
        near = np.sort(pitches[np.abs(starts - group.start) <= window_seconds])
        if len(near) >= 2 and near[-1] - near[0] >= 12:
            best, cut = None, previous
            for k in range(1, len(near)):
                if near[k] == near[k - 1]:
                    continue
                left, right = near[:k], near[k:]
                spread = left.var() * len(left) + right.var() * len(right)
                if best is None or spread < best:
                    best, cut = spread, (near[k - 1] + near[k]) / 2
            previous = cut
        splits.append(previous)
    return splits
