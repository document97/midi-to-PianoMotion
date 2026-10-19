"""Listening-fidelity metric: how much of the source's band energy a plan keeps.

Both the full source selection and a plan are synthesised at one gain and compared
in 1/3-octave-like bands over 100 ms frames. ``deficit_db`` is the frame-average
loudness the plan lacks (positive = quieter than the source) in frames where the
source band is audible; it is a proxy for "the bass / treble is missing", not a
substitute for listening.
"""
import numpy as np

from .audio import RATE, synthesize

FRAME = 4096
HOP = 4410
EDGES = [40 * 2 ** (k / 3) for k in range(0, 22)]   # 40 Hz .. ~5 kHz
BANDS = (('bass <130Hz', 0, 130), ('low-mid 130-260', 130, 260), ('mid 260-520', 260, 520),
         ('high-mid 520-1k', 520, 1050), ('treble >1k', 1050, 6000))


def band_levels(audio):
    mono = audio.mean(axis=1)
    window = np.hanning(FRAME)
    frames = np.array([mono[i:i + FRAME] * window for i in range(0, len(mono) - FRAME, HOP)])
    spectrum = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(FRAME, 1 / RATE)
    out = np.stack([spectrum[:, (freqs >= low) & (freqs < high)].sum(axis=1) for _, low, high in
                    ((None, a, b) for a, b in zip(EDGES, EDGES[1:]))], axis=1)
    return 10 * np.log10(out + 1e-12)


def compare(timeline, kept_ids, cut_releases, source_ids=None, reference=None):
    if reference is None:
        source_ids = source_ids or [n['id'] for n in timeline['notes']]
        reference = band_levels(synthesize(timeline, source_ids)[0])
    plan = band_levels(synthesize(timeline, kept_ids, cut_releases=cut_releases)[0])
    frames = min(len(reference), len(plan))
    reference, plan = reference[:frames], plan[:frames]
    floor = reference.max() - 60
    centres = np.array([np.sqrt(a * b) for a, b in zip(EDGES, EDGES[1:])])
    report = {}
    for name, low, high in BANDS:
        cols = (centres >= low) & (centres < high)
        audible = reference[:, cols] > floor
        deficit = np.where(audible, np.maximum(reference[:, cols] - plan[:, cols], 0.), 0.)
        report[name] = round(float(deficit.sum() / max(audible.sum(), 1)), 3)
    report['overall'] = round(float(np.mean(list(report.values()))), 3)
    return report, reference
