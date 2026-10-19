"""Planner inputs for a piece from a MIDI file.

python inputs/build_piece.py <file.mid> <name> [--right TRACK --left TRACK] [--credit TEXT]   (cwd: pianist)
  ->  inputs/<name>/{timeline,demands}.json, the MIDI copied to inputs/<name>/source/

--right/--left name the tracks holding each hand (e.g. piano-midi.de's 'Piano right' = 1, 'Piano left' = 2). They only
label the notes so pipeline.score_hands can compare the plan's hands with the score's; the planner picks hands itself.
Without them a track named 'Right hand'/'Left hand' is still recognised; otherwise the notes stay unassigned.
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from midi_input.timeline import load_midi
from midi_input.hold_demands import build_hold_demands


def channels(midi_path, track):
    import mido
    return sorted({m.channel for m in mido.MidiFile(midi_path).tracks[track] if m.type == 'note_on'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('midi')
    ap.add_argument('name')
    ap.add_argument('--right', type=int)
    ap.add_argument('--left', type=int)
    ap.add_argument('--credit', default='')
    a = ap.parse_args()
    folder = ROOT / 'inputs' / a.name
    (folder / 'source').mkdir(parents=True, exist_ok=True)
    midi_path = folder / 'source' / Path(a.midi).name
    if Path(a.midi).resolve() != midi_path.resolve():
        shutil.copyfile(a.midi, midi_path)
    if a.credit:
        (folder / 'source' / 'SOURCE.txt').write_text(a.credit + '\n', encoding='utf-8')
    hand_map = {}
    for hand, track in (('R', a.right), ('L', a.left)):
        if track is not None:
            hand_map.update({f'{track}:{c}': hand for c in channels(midi_path, track)})
    timeline = load_midi(midi_path, hand_map)
    demands = build_hold_demands(timeline)
    for name, data in (('timeline', timeline), ('demands', demands)):
        (folder / f'{name}.json').write_text(json.dumps(data, default=str), encoding='utf-8')
    hands = Counter(n['hand'] or 'unassigned' for n in timeline['notes'])
    pedal = sum(1 for e in timeline['events'] if e['message']['type'] == 'control_change' and e['message']['control'] == 64)
    print(f"{a.name}: {len(timeline['notes'])} notes {dict(hands)}, {timeline['duration']:.1f} s, pedal events {pedal}, "
          f"diagnostics {len(timeline['diagnostics'])}, unresolved holds {len(demands.get('unresolved', []))}")


if __name__ == '__main__':
    main()
