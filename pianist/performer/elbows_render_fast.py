"""Compiled pchip slopes + batched hermite evaluation for the elbow solve's renderer.

The elbow wrist-ask re-renders the palm angle hundreds of times. Each _rendered call
recomputed pchip_slopes over the whole ~25k-point curve in a Python loop (~140 ms)
and then evaluated it one frame at a time in another Python loop (~190 ms). Those two
loops are ~330 ms of the ~350 ms the renderer spends per call, 558 calls per piece.

This module compiles both loops. Results are bit-for-bit the same formulas as
motion.pchip_slopes / motion.hermite_eval (Fritsch-Carlson slopes, cubic Hermite),
so enabling it changes no animation. numba is optional: available() is False under
Blender's Python and _rendered keeps its original path there.
PIANOMOTION_FAST_RENDER=0 forces the numpy path for A/B cross-checking.
"""
import os

import numpy as np

OPTED_OUT = os.environ.get('PIANOMOTION_FAST_RENDER', '1') == '0'
njit = None
if not OPTED_OUT:
    try:
        from numba import njit as _njit
        njit = _njit
    except ImportError:
        njit = None


def available():
    return njit is not None


if njit is not None:

    @njit(cache=True, fastmath=False, nogil=True)
    def _slopes(times, values):
        """Fritsch-Carlson monotone cubic slopes, values (n, k) -> (n, k)."""
        n, k = values.shape
        h = times[1:] - times[:-1]
        slopes = np.zeros((n, k))
        for i in range(1, n - 1):
            a0, a1 = h[i - 1], h[i]
            w1, w2 = 2. * a1 + a0, a1 + 2. * a0
            for j in range(k):
                da = (values[i, j] - values[i - 1, j]) / a0
                db = (values[i + 1, j] - values[i, j]) / a1
                if da * db > 0.:
                    slopes[i, j] = (w1 + w2) / (w1 / da + w2 / db)
        return slopes

    @njit(cache=True, fastmath=False, nogil=True)
    def _eval_one(times, values, slopes, t):
        n = len(times)
        if n == 0:
            return 0.
        if n == 1:
            return values[0, 0]
        # searchsorted(times, t, side='right') - 1, clipped to [0, n-2]
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) >> 1
            if times[mid] <= t:
                lo = mid + 1
            else:
                hi = mid
        i = lo - 1
        if i < 0:
            i = 0
        elif i > n - 2:
            i = n - 2
        h = times[i + 1] - times[i]
        s = (t - times[i]) / h
        if s < 0.:
            s = 0.
        elif s > 1.:
            s = 1.
        s2, s3 = s * s, s * s * s
        h00, h10 = 2. * s3 - 3. * s2 + 1., s3 - 2. * s2 + s
        h01, h11 = -2. * s3 + 3. * s2, s3 - s2
        return (h00 * values[i, 0] + h10 * h * slopes[i, 0]
                + h01 * values[i + 1, 0] + h11 * h * slopes[i + 1, 0])

    @njit(cache=True, fastmath=False, nogil=True)
    def _eval_many(times, values, slopes, queries, out):
        for i in range(len(queries)):
            out[i] = _eval_one(times, values, slopes, queries[i])
        return out


def slopes(times, values):
    """Compiled pchip_slopes; values (n, k)."""
    return _slopes(np.ascontiguousarray(times, dtype=np.float64),
                   np.ascontiguousarray(values, dtype=np.float64))


def eval_many(times, values, t_query):
    """Compiled batched hermite evaluation of a pchip curve at 1-d query times."""
    t = np.ascontiguousarray(times, dtype=np.float64)
    v = np.ascontiguousarray(values, dtype=np.float64).reshape(len(t), -1)
    q = np.ascontiguousarray(np.atleast_1d(t_query), dtype=np.float64)
    sl = _slopes(t, v)
    out = np.empty(len(q), dtype=np.float64)
    _eval_many(t, v, sl, q, out)
    return out
