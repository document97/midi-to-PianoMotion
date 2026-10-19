"""Listening preview of a note selection.

Sounding durations follow the source note-off and CC64 clock (the existing
preview_clock), synthesised with the same GM piano samples and envelope as
lib/render_bound_piano_audio.py. Audible legato shortenings made
by the planner are rendered: the sound of such a note stops when the source
damper (CC64) closes after its key lift, not at the lift itself.
"""
import math
from pathlib import Path
import sys
import wave

import numpy as np

from midi_input.damper_open_tail import DamperClock
from midi_input.preview_clock import selected_preview_clock

TOOLS = Path(__file__).resolve().parents[1] / 'lib'
SOUNDBANK = Path('C:/Windows/System32/drivers/gm.dls')
RATE = 44100


def _regions():
    sys.path.insert(0, str(TOOLS))
    from render_bound_piano_audio import piano_regions
    return piano_regions(SOUNDBANK)


def render(timeline, note_ids, path, duration=None, cut_releases=None):
    """Write a normalised preview; returns the number of sounding notes."""
    audio, count = synthesize(timeline, note_ids, duration, cut_releases)
    audio = audio * (.88 / max(float(np.abs(audio).max()), 1e-8))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(RATE)
        writer.writeframes((audio * 32767).astype('<i2').tobytes())
    return count


def synthesize(timeline, note_ids, duration=None, cut_releases=None):
    """Unnormalised stereo signal (so two plans can be compared at one gain) and its note count.

    ``cut_releases``: {note id: key lift} for notes whose sound the plan shortens.
    """
    cut_releases = cut_releases or {}
    duration = duration or float(timeline['duration']) + 1.
    wanted = set(note_ids)
    plan = dict(musical_damper_policy='final_open_source_interval', source_note_inventory=timeline['notes'],
                notes=[dict(n, physical_release=n['end']) for n in timeline['notes'] if n['id'] in wanted])
    rows = selected_preview_clock(timeline, plan, duration)
    damper = DamperClock(timeline)
    channel = {n['id']: n['channel'] for n in timeline['notes']}
    regions = _regions()
    audio = np.zeros((math.ceil(duration * RATE), 2))
    for note in rows:
        pitch, hold = note['pitch'], note['hold_seconds']
        if note['id'] in cut_releases:
            closure = damper.damping_time(channel[note['id']], cut_releases[note['id']])
            if closure is not None:
                hold = min(hold, closure - note['start'])
        length = min(duration - note['start'], hold + .65)
        region = next(r for r in regions if r['low'] <= pitch <= r['high'])
        t = np.arange(math.ceil(length * RATE)) / RATE
        ratio = 2 ** ((pitch - region['unity'] - region['cents'] / 100) / 12)
        positions = t * region['rate'] * ratio
        if region['loop_length']:
            begin = region['loop_start']
            positions = np.where(positions >= begin, begin + np.mod(positions - begin, region['loop_length']), positions)
        else:
            positions = np.minimum(positions, len(region['sound']) - 1)
        sound = np.interp(positions, np.arange(len(region['sound'])), region['sound'])
        decay = 2.8 - 1.4 * np.clip((pitch - 36) / 60, 0, 1)
        envelope = np.exp(-t / decay) * np.minimum(1, t / .004) * np.exp(-np.maximum(0, t - hold) / .16)
        sound *= envelope * (note['velocity'] / 127) ** 1.6
        pan = np.clip((pitch - 60) / 42, -.65, .65)
        gains = np.array([math.sqrt((1 - pan) / 2), math.sqrt((1 + pan) / 2)])
        start = round(note['start'] * RATE)
        count = min(len(sound), len(audio) - start)
        audio[start:start + count] += sound[:count, None] * gains
    return audio, len(rows)
