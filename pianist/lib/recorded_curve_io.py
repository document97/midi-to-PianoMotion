"""Write recorded LINEAR knots without Blender's proximity deduplication.

FCurve.update() merges knots within its frame threshold. A physical release
event may be closer to a regular solve sample, and neither may be discarded.
Keep editable linear keys, check float32 storage, and reject silent loss.
"""
import numpy as np


def replace_linear_curve(curve, frames, values):
    frames = np.asarray(frames, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if frames.ndim != 1 or values.shape != frames.shape or not len(frames):
        raise ValueError('Recorded curve requires matching nonempty scalar arrays')
    if not np.all(np.isfinite(frames)) or not np.all(np.isfinite(values)):
        raise ValueError('Recorded curve contains nonfinite knots')
    stored = np.column_stack((frames, values)).astype(np.float32)
    if not np.all(np.diff(frames) > 0) or not np.all(np.diff(stored[:, 0]) > 0):
        raise ValueError('Recorded times must remain distinct in Blender float32 frames')
    if len(curve.sampled_points) or len(curve.modifiers):
        raise ValueError('Recorded linear writer does not replace sampled or modified curves')
    points = curve.keyframe_points
    for index in range(len(points) - 1, -1, -1):
        points.remove(points[index], fast=True)
    points.add(len(frames))
    points.foreach_set('co', stored.ravel())
    for point in points:
        point.interpolation = 'LINEAR'
    # Separate APIs avoid FCurve.update() and its implicit deduplication.
    points.sort()
    points.handles_recalc()
    curve.id_data.update_tag()
    observed = np.empty((len(points), 2), dtype=np.float32)
    points.foreach_get('co', observed.ravel())
    if not np.array_equal(observed, stored):
        raise ValueError('Blender changed recorded knot count, clock or value')

