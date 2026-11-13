"""elbows.Trunk's hull test, compiled (numba).

The numpy version builds a temporary per call -- ``up`` is (angles, UP, 3), the hull lookup gathers
(angles, UP, edges) twice and takes a min along the last axis -- which is some 60 k comparisons and
five large temporaries for one scan sample (120 angles x 20 capsule points x 26 hull edges). The scan
takes one such call per sample, and a 164 s piece is 3 280 samples a hand, again per cap round and
again per honest rescan of the wrist ask: the temporaries, not the arithmetic, are the cost.

Here the whole thing is one kernel over the sample axis, ``prange`` across the samples (the samples are
independent: each has its own shoulder, wrist and trunk matrix). One sample does its circle, its 20
capsule points and its 26-edge min in machine code, writing into preallocated arrays -- no temporary
larger than a scalar. The per-sample result is identical to Trunk.swing's by construction: the same
elbow_circle (including its two ``1e-4`` reach guards, reported as ``ok`` instead of raised), the same
nearest-slice rule (searchsorted then the nearer of the two neighbours), the same padded hull rows.

numba is optional: without it ``available()`` is False and Trunk falls back to the numpy path, which is
what runs under Blender's Python as well. PIANOMOTION_FAST_ELBOWS=0 forces the numpy path for
cross-checking. The compiled code is cached on disk and rebuilt only when this file changes.
"""
import math
import os

import numpy as np

OPTED_OUT = os.environ.get('PIANOMOTION_FAST_ELBOWS', '1') == '0'
njit = None
prange = range
if not OPTED_OUT:
    try:
        from numba import njit, prange as _prange
        prange = _prange
    except ImportError:
        njit = None


def available():
    return njit is not None


if njit is not None:

    @njit(cache=True, fastmath=False, nogil=True)
    def _depth_local(lx, ly, lz, keys, origin, normal, tlo, step):
        """One point's depth in the outline: Trunk.depth for a single point already in the bone's frame."""
        k = int(math.floor((ly - tlo) / step))
        if k < keys[0] or k > keys[-1]:
            return -1.
        n = keys.shape[0]
        lo, hi = 0, n
        while lo < hi:                      # searchsorted(keys, k, 'left')
            mid = (lo + hi) >> 1
            if keys[mid] < k:
                lo = mid + 1
            else:
                hi = mid
        j = lo
        if j > n - 1:
            j = n - 1
        near = j - 1 if j > 0 else 0
        if abs(k - keys[near]) > abs(keys[j] - k):
            near = j
        best = 1e30
        for m in range(origin.shape[1]):
            d = (origin[near, m, 0] - lx) * normal[near, m, 0] + (origin[near, m, 1] - lz) * normal[near, m, 1]
            if d < best:
                best = d
        return best

    @njit(cache=True, fastmath=False, nogil=True, parallel=True)
    def _swing_block(S, W, R, O, l1, l2, side, rup, rfo, keys, origin, normal, tlo, step, up, fo, cosv, sinv,
                     depth, rise, ok):
        """Fill depth/rise/(ok) for every sample: same numbers as Trunk.swing, many samples at once."""
        n = S.shape[0]
        nang = cosv.shape[0]
        nup, nfo = up.shape[0], fo.shape[0]
        for i in prange(n):
            sx, sy, sz = S[i, 0], S[i, 1], S[i, 2]
            wx, wy, wz = W[i, 0], W[i, 1], W[i, 2]
            dx, dy, dz = wx - sx, wy - sy, wz - sz
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist >= l1 + l2 - 1e-4 or dist <= abs(l1 - l2) + 1e-4:
                ok[i] = False                # elbow_circle's reach guards, the scan's "stretched" sample
                continue
            ok[i] = True
            ux, uy, uz = dx / dist, dy / dist, dz / dist
            a = (l1 * l1 - l2 * l2 + dist * dist) / (2. * dist)
            radius = math.sqrt(max(0., l1 * l1 - a * a))
            cx, cy, cz = sx + a * ux, sy + a * uy, sz + a * uz
            dn = -uz                         # dot(downward (0,0,-1), u)
            tx, ty, tz = -ux * dn, -uy * dn, -1. - uz * dn
            tn = math.sqrt(tx * tx + ty * ty + tz * tz)
            if tn < .01:
                tx = side - ux * side * ux
                ty = -uy * side * ux
                tz = -uz * side * ux
                tn = math.sqrt(tx * tx + ty * ty + tz * tz)
            tx, ty, tz = tx / tn, ty / tn, tz / tn
            qx = uy * tz - uz * ty
            qy = uz * tx - ux * tz
            qz = ux * ty - uy * tx
            r00, r01, r02 = R[i, 0, 0], R[i, 0, 1], R[i, 0, 2]
            r10, r11, r12 = R[i, 1, 0], R[i, 1, 1], R[i, 1, 2]
            r20, r21, r22 = R[i, 2, 0], R[i, 2, 1], R[i, 2, 2]
            ox, oy, oz = O[i, 0], O[i, 1], O[i, 2]
            for j in range(nang):
                ex = cx + radius * (cosv[j] * tx + sinv[j] * qx)
                ey = cy + radius * (cosv[j] * ty + sinv[j] * qy)
                ez = cz + radius * (cosv[j] * tz + sinv[j] * qz)
                best_u = -1e30
                for k in range(nup):
                    f = up[k]
                    px, py, pz = sx + f * (ex - sx), sy + f * (ey - sy), sz + f * (ez - sz)
                    ddx, ddy, ddz = px - ox, py - oy, pz - oz
                    lx = ddx * r00 + ddy * r10 + ddz * r20
                    ly = ddx * r01 + ddy * r11 + ddz * r21
                    lz = ddx * r02 + ddy * r12 + ddz * r22
                    d = _depth_local(lx, ly, lz, keys, origin, normal, tlo, step)
                    if d > best_u:
                        best_u = d
                best_f = -1e30
                for k in range(nfo):
                    f = fo[k]
                    px, py, pz = ex + f * (wx - ex), ey + f * (wy - ey), ez + f * (wz - ez)
                    ddx, ddy, ddz = px - ox, py - oy, pz - oz
                    lx = ddx * r00 + ddy * r10 + ddz * r20
                    ly = ddx * r01 + ddy * r11 + ddz * r21
                    lz = ddx * r02 + ddy * r12 + ddz * r22
                    d = _depth_local(lx, ly, lz, keys, origin, normal, tlo, step)
                    if d > best_f:
                        best_f = d
                depth[i, j] = max(best_u + rup, best_f + rfo)
                rise[i, j] = ez - wz


def pack(trunk):
    """The outline's arrays for the kernel, built once per Trunk (the owner keeps it)."""
    return dict(keys=np.ascontiguousarray(trunk.keys, dtype=np.int64),
                origin=np.ascontiguousarray(trunk.origin, dtype=np.float64),
                normal=np.ascontiguousarray(trunk.normal, dtype=np.float64),
                lo=float(trunk.lo), step=float(trunk.step))


def swing_block(packed, S, W, R, O, l1, l2, side, rup, rfo, cosv, sinv, up, fo):
    """(depth, rise, ok) for every sample: depth (n, angles), rise (n, angles), the elbow's rise above the wrist."""
    S = np.ascontiguousarray(S, dtype=np.float64)
    W = np.ascontiguousarray(W, dtype=np.float64)
    R = np.ascontiguousarray(R, dtype=np.float64)
    O = np.ascontiguousarray(O, dtype=np.float64)
    n, nang = len(S), len(cosv)
    depth = np.empty((n, nang), dtype=np.float64)
    rise = np.empty((n, nang), dtype=np.float64)
    ok = np.zeros(n, dtype=np.bool_)
    _swing_block(S, W, R, O, float(l1), float(l2), float(side), float(rup), float(rfo),
                 packed['keys'], packed['origin'], packed['normal'], packed['lo'], packed['step'],
                 np.ascontiguousarray(up, dtype=np.float64), np.ascontiguousarray(fo, dtype=np.float64),
                 np.ascontiguousarray(cosv, dtype=np.float64), np.ascontiguousarray(sinv, dtype=np.float64),
                 depth, rise, ok)
    return depth, rise, ok
