"""Per-frame physical checks of the generated motion (skin points against the keys).

Key tops are rasterised from the exported key meshes (upward faces) into a 1 mm
height field that remembers which key owns each cell, so a pressed key is
lowered by its current depth. A skin point below the surface of the key under it
penetrates that key. The pressing finger's own pad sits on its key by design;
it is allowed ``OWN_KEY_TOLERANCE``.

Brushed keys: a finger holding its own key down may rest on an unplayed neighbouring key and push it
down by up to ``BRUSH_DEPTH``, as a small hand does in a stretched chord. A key sounds only near the
bottom of its travel (escapement ~8-9 mm of 10), so a 3 mm push is silent. The solvers pay
``BRUSH_COST`` per metre of such a dip (they keep clear where they can); beyond it the dip is penetration.
The pushed keys are animated (pipeline.brushed_keys) and checked.
"""
import numpy as np

CELL = .001
OWN_KEY_TOLERANCE = .0008
BRUSH_DEPTH = .003               # deepest silent push of an unplayed key by a finger resting on it (m)
BRUSH_COST = .3                  # solver cost of a brushing millimetre relative to a penetrating one ...
BRUSH_MIN_SPAN = 9               # ... only while the hand holds a stretch this wide (semitones); otherwise full


def stretched(pitches):
    """A hand holding keys this far apart may rest a finger on a neighbour (the solvers' brush discount)."""
    pitches = list(pitches)
    return len(pitches) > 1 and max(pitches) - min(pitches) >= BRUSH_MIN_SPAN


def brushed(pen, brushable):
    """Penetration as the solvers see it: a brushable key yields ``BRUSH_DEPTH`` at ``BRUSH_COST``."""
    soft = np.maximum(pen - BRUSH_DEPTH, 0.) + BRUSH_COST * np.clip(pen, 0., BRUSH_DEPTH)
    return np.where(brushable, soft, pen)


class Heightfield:
    def __init__(self, key_meshes):
        tris = {int(p): np.array(m['triangles']) for p, m in key_meshes.items()}
        allpts = np.concatenate([t.reshape(-1, 3) for t in tris.values()])
        self.origin = allpts[:, :2].min(axis=0) - CELL
        size = np.ceil((allpts[:, :2].max(axis=0) + CELL - self.origin) / CELL).astype(int) + 1
        self.top = np.full(size, -np.inf)
        self.pitch = np.zeros(size, dtype=np.int16)
        for pitch, t in tris.items():
            n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            for a, b, c in t[n[:, 2] > .5]:
                lo = np.floor((np.minimum(np.minimum(a, b), c)[:2] - self.origin) / CELL).astype(int)
                hi = np.ceil((np.maximum(np.maximum(a, b), c)[:2] - self.origin) / CELL).astype(int)
                gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing='ij')
                px, py = self.origin[0] + gx * CELL, self.origin[1] + gy * CELL
                v0, v1 = b[:2] - a[:2], c[:2] - a[:2]
                den = v0[0] * v1[1] - v1[0] * v0[1]
                if abs(den) < 1e-14:
                    continue
                wx, wy = px - a[0], py - a[1]
                u = (wx * v1[1] - v1[0] * wy) / den
                v = (v0[0] * wy - wx * v0[1]) / den
                inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
                z = a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2])
                cells = (gx[inside], gy[inside])
                higher = z[inside] > self.top[cells]
                self.top[cells[0][higher], cells[1][higher]] = z[inside][higher]
                self.pitch[cells[0][higher], cells[1][higher]] = pitch

    def penetration(self, points, depths, own=None, brush=False):
        """Per point: metres below the (pressed) key surface under it; ``own``: (pitch, mask) allowance;
        ``brush``: unplayed keys other than ``own`` may be pushed down silently (``brushed``)."""
        idx = np.floor((points[:, :2] - self.origin) / CELL).astype(int)
        ok = (idx[:, 0] >= 0) & (idx[:, 1] >= 0) & (idx[:, 0] < self.top.shape[0]) & (idx[:, 1] < self.top.shape[1])
        out = np.zeros(len(points))
        ix, iy = idx[ok, 0], idx[ok, 1]
        top = self.top[ix, iy]
        pitch = self.pitch[ix, iy]
        lowered = top - np.array([depths.get(int(p), 0.) for p in pitch])
        pen = np.where(np.isfinite(top), lowered - points[ok, 2], 0.)
        if own is not None:
            own_pitch, mask = own
            allow = mask[ok] & (pitch == own_pitch)
            pen = np.where(allow, pen - OWN_KEY_TOLERANCE, pen)
            if brush:
                pen = brushed(np.maximum(pen, 0.), self.brushable(pitch, depths, own_pitch) & np.isfinite(top))
        out[ok] = np.maximum(pen, 0.)
        return out

    @staticmethod
    def brushable(pitch, depths, own_pitch):
        return (pitch != own_pitch) & np.array([depths.get(int(p), 0.) <= 0. for p in pitch], dtype=bool)

    def split(self, points, depths, own_pitch):
        """For a finger holding ``own_pitch``: (penetration per point of keys it may not brush,
        {pitch: deepest push wanted of each brushable key})."""
        idx = np.floor((points[:, :2] - self.origin) / CELL).astype(int)
        ok = (idx[:, 0] >= 0) & (idx[:, 1] >= 0) & (idx[:, 0] < self.top.shape[0]) & (idx[:, 1] < self.top.shape[1])
        hard = np.zeros(len(points))
        wanted = {}
        ix, iy = idx[ok, 0], idx[ok, 1]
        top, pitch = self.top[ix, iy], self.pitch[ix, iy]
        lowered = top - np.array([depths.get(int(p), 0.) for p in pitch])
        pen = np.where(np.isfinite(top), lowered - points[ok, 2], 0.)
        pen = np.where(pitch == own_pitch, pen - OWN_KEY_TOLERANCE, pen)
        pen = np.maximum(pen, 0.)
        soft = self.brushable(pitch, depths, own_pitch) & np.isfinite(top)
        hard[ok] = np.where(soft, 0., pen)
        for p in np.unique(pitch[soft & (pen > 0)]):
            wanted[int(p)] = float(pen[soft & (pitch == p)].max())
        return hard, wanted
