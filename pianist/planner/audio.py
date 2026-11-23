"""Listening preview of a note selection.

``render`` (the preview.wav of a run) plays the selected notes with the source CC64 on a sampled grand piano
(FluidSynth with Dore Mark's Fazioli F308 SF2; see TOOLS_DIR) and writes WAV with one uniform gain (peak to
PEAK_DB), nothing else: no compression, EQ or effects (FluidSynth's reverb and chorus off). A note the planner shortens is let go at its key
lift, so its sound stops where the source damper closes after it.

``synthesize`` is the plainer in-house sampler (a General MIDI DLS bank, the envelope of lib/render_bound_piano_audio.py)
the fidelity measure compares plans with: unnormalised, at one gain for two plans. Windows ships the bank it was
developed against; elsewhere PIANOMOTION_SOUNDBANK names a DLS file.
"""
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import wave

import numpy as np

from midi_input.damper_open_tail import DamperClock
from midi_input.preview_clock import selected_preview_clock

TOOLS = Path(__file__).resolve().parents[1] / 'lib'
SOUNDBANK_ENV = 'PIANOMOTION_SOUNDBANK'   # a General MIDI DLS bank for the in-house sampler
RATE = 44100
TOOLS_DIR = Path(__file__).resolve().parents[2] / 'tools'     # downloaded once, never committed (README)
FLUIDSYNTH = Path(os.environ.get('PIANOMOTION_FLUIDSYNTH', TOOLS_DIR / 'fluidsynth' / 'bin' / 'fluidsynth.exe'))
SOUNDFONT = Path(os.environ.get('PIANOMOTION_SOUNDFONT', TOOLS_DIR / 'fazioli' / "Dore Mark's Fazioli F308-v2.6.sf2"))
SYNTH_GAIN = .5                       # FluidSynth's master gain: its 16-bit output clips at 1.0 with this piano; halved
                                      # again if a render still reaches full scale (the WAV's gain is set after)
PEAK_DB = -1.
MIDI_TICKS_PER_SECOND = 2000          # the preview MIDI's clock (0.5 ms)


def soundbank():
    """The GM DLS bank ``synthesize`` reads: ``PIANOMOTION_SOUNDBANK`` first, then the platform's own.

    Windows ships ``gm.dls``; macOS ships the CoreAudio DLS bank. Other systems have no fixed location, so a
    Linux run must set the variable. The parser in lib/render_bound_piano_audio.py reads DLS, not SF2.
    """
    configured = os.environ.get(SOUNDBANK_ENV)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'{SOUNDBANK_ENV}={configured} is not a file')
        return path
    candidates = []
    if sys.platform == 'win32':
        candidates.append(Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / 'drivers' / 'gm.dls')
    elif sys.platform == 'darwin':
        candidates.append(Path('/System/Library/Components/CoreAudio.component/Contents/Resources/gs_instruments.dls'))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f'no General MIDI DLS bank for the fidelity preview; set {SOUNDBANK_ENV} to a .dls file '
        '(Windows: %SystemRoot%\\System32\\drivers\\gm.dls)')


def _regions():
    sys.path.insert(0, str(TOOLS))
    from render_bound_piano_audio import piano_regions
    return piano_regions(soundbank())


def render(timeline, note_ids, path, duration=None, cut_releases=None, lead=0.):
    """Write the preview WAV of ``note_ids`` (``cut_releases``: {note id: key lift} for notes the plan shortens);
    returns the number of sounding notes. ``duration`` pads or trims the file to that many seconds; ``lead``:
    seconds of silence before the piece (the hands' silent lead-in)."""
    cut_releases = cut_releases or {}
    wanted = set(note_ids)
    events = []
    for n in timeline['notes']:
        if n['id'] in wanted:
            events.append((n['start'], 1, bytes([0x90 | n['channel'], n['pitch'], max(1, n['velocity'])])))
            if n['id'] in cut_releases and cut_releases[n['id']] < n['end']:
                # a key the plan lifts early counts every pedal event at its lift instant as already happened
                # (DamperClock.damping_time), so a lift as the pedal goes down is caught by it
                events.append((cut_releases[n['id']], 3, bytes([0x80 | n['channel'], n['pitch'], 0])))
            else:
                events.append((n['end'], 0, bytes([0x80 | n['channel'], n['pitch'], 0])))
    for e in timeline['events']:
        m = e['message']
        if m['type'] == 'control_change':
            # at one instant: source releases, the pedal going up, new notes, the pedal going down, planned lifts
            order = 0 if m['control'] == 64 and m['value'] < 64 else 2
            events.append((e['seconds'], order, bytes([0xB0 | m['channel'], m['control'], m['value']])))
    events.sort(key=lambda e: (round(e[0] * MIDI_TICKS_PER_SECOND), e[1]))   # by MIDI tick: an instant is one tick
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        midi, raw = Path(folder) / 'preview.mid', Path(folder) / 'preview.wav'
        midi.write_bytes(_midi_file(events))
        gain = SYNTH_GAIN
        while True:
            subprocess.run([str(FLUIDSYNTH), '-ni', '-q', '-R', '0', '-C', '0', '-g', str(gain), '-r', str(RATE), '-F', str(raw),
                            str(SOUNDFONT), str(midi)], check=True, capture_output=True)
            with wave.open(str(raw), 'rb') as reader:
                channels, width = reader.getnchannels(), reader.getsampwidth()
                samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype='<i2').reshape(-1, channels)
            if np.abs(samples.astype(np.int32)).max() < 32767 or gain < .01:
                break
            gain /= 2
        audio = samples / 32768.
    if duration is not None:
        frames = math.ceil(duration * RATE)
        audio = np.pad(audio, ((0, max(0, frames - len(audio))), (0, 0)))[:frames]
    audio = np.pad(audio, ((round(lead * RATE), 0), (0, 0)))
    audio = audio * (10 ** (PEAK_DB / 20) / max(float(np.abs(audio).max()), 1e-8))
    with wave.open(str(path), 'wb') as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(RATE)
        writer.writeframes(np.round(audio * 32767).astype('<i2').tobytes())
    return sum(1 for n in timeline['notes'] if n['id'] in wanted)


def _midi_file(events):
    """Type-0 standard MIDI file of (seconds, order, message bytes), at 120 bpm (MIDI_TICKS_PER_SECOND)."""
    def number(value):
        out = [value & 0x7F]
        value >>= 7
        while value:
            out.append(0x80 | value & 0x7F)
            value >>= 7
        return bytes(reversed(out))
    track = bytearray(number(0) + b'\xff\x51\x03' + (500000).to_bytes(3, 'big'))
    last = 0
    for seconds, _, message in events:
        tick = max(0, round(seconds * MIDI_TICKS_PER_SECOND))
        track += number(tick - last) + message
        last = tick
    track += number(0) + b'\xff\x2f\x00'
    return (b'MThd' + struct.pack('>IHHH', 6, 0, 1, MIDI_TICKS_PER_SECOND // 2)
            + b'MTrk' + struct.pack('>I', len(track)) + track)


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
