"""Input loading, plan export and listening-relevant report."""
from pathlib import Path

from .instrument import Instrument
from .music import KEY_RETURN_SECONDS, Score, load_json

DATA = Path(__file__).resolve().parents[1]
PIECE_FILES = ('timeline', 'demands')    # a piece's inputs (inputs/build_piece.py writes them)
DEFAULT_INPUTS = dict(
    keyboard_plan='inputs/instrument/keyboard.json',
    envelopes='inputs/instrument/pad_rom_envelopes.json',     # sampled from the rig (performer/hand_ranges.py)
    radii='inputs/instrument/pad_reach_conservative.json',  # bone-length bound of the rig, scaled with the character (exact for a uniform scale)
)
REGISTERS = (('bass <C3', 0, 48), ('low C3-B3', 48, 60), ('middle C4-B4', 60, 72),
             ('high C5-B5', 72, 84), ('top >=C6', 84, 128))


def load_raw(inputs):
    """``inputs``: the piece's PIECE_FILES paths (and optionally other DEFAULT_INPUTS overrides)."""
    missing = [k for k in PIECE_FILES if k not in inputs]
    if missing:
        raise ValueError(f'piece inputs missing: {missing}')
    paths = {k: DATA / v for k, v in {**DEFAULT_INPUTS, **inputs}.items()}
    return dict(timeline=load_json(paths['timeline']), demands=load_json(paths['demands']),
                keyboard=load_json(paths['keyboard_plan'])['keyboard'],
                envelopes=load_json(paths['envelopes'])['envelopes'], radii=load_json(paths['radii'])['radii'],
                paths={k: str(v) for k, v in paths.items()})


def build(inputs, end=None):
    """Picklable factory used by every worker process."""
    raw = load_raw(inputs)
    return (Score(raw['timeline'], raw['demands'], end),
            Instrument(raw['keyboard'], raw['envelopes'], raw['radii']))


def plan_notes(state, score):
    """Retained notes with hand, finger and physical release."""
    lifted = {}
    for decision in state.decisions():
        for note, release in decision.releases:
            lifted[note] = min(release, lifted.get(note, release))
    rows = []
    for decision in state.decisions():
        for note, hand, finger in decision.assignment:
            n = score.notes[note]
            release = lifted.get(note, n.end)
            rows.append(dict(index=note, id=n.id, pitch=n.pitch, velocity=n.velocity, start=n.start,
                             end=n.end, touch=n.touch, hand=hand, finger=finger,
                             physical_release=release, busy_end=release + KEY_RETURN_SECONDS,
                             earliest_equivalent_release=n.earliest_release))
    return sorted(rows, key=lambda r: (r['start'], r['pitch']))


def register_report(source_notes, kept_ids, curve=None):
    """Omission per register over the complete source inventory (dicts with id, pitch).

    With ``curve``, each register also gets its expected ratio (mean over its notes).
    """
    out = []
    for name, low, high in REGISTERS:
        notes = [n for n in source_notes if low <= n['pitch'] < high]
        omitted = [n for n in notes if n['id'] not in kept_ids]
        row = dict(register=name, notes=len(notes), omitted=len(omitted),
                   omitted_ratio=len(omitted) / len(notes) if notes else 0.)
        if curve is not None and notes:
            row['expected_ratio'] = sum(curve.ratio(n['pitch']) for n in notes) / len(notes)
        out.append(row)
    return out
