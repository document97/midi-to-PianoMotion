"""The performer's key regions and playing wrist: white keys may also be played a little between the
black keys, and the hand sits back and low enough to meet the keys with the finger pads.

The planner's pair table keeps the conservative regions of pose.KeyRegions (wide front part only), so
the plan is unchanged; the performer's anchors and frames use these. Confined to the 33 mm in front of
the black keys, a long finger beside the thumb had to curl until it met the key with its tip; pianists
let it land further in, on the narrow part between the black keys.
"""
import numpy as np

from .pose import X_MARGIN, hand_solvers

WHITE_DEEP = .012               # how far behind the black keys' front a white-key pad may land (m); 8 mm
                                # gave more ride penetration over the full piece, not less
# The calibrated neutral wrist (from the bake's seat) sits over the white keys' front edge: from there the
# fingers can only reach the keys curled steeply down onto their tips. Further back and lower, they meet
# the keys with the pad (last phalanx ~40 deg; measured on test chords), about 3 cm above the key tops.
PLAYING_WRIST_OFFSET = np.array([0., -.08, -.025])


def deepen(regions):
    keys = regions.keys
    for p, row in regions.rows.items():
        if regions.is_black(p):
            continue
        lo, hi = keys[p]['bounds_min'], keys[p]['bounds_max']
        left = keys[p - 1]['bounds_max'][0] if p - 1 in keys and regions.is_black(p - 1) else lo[0]
        right = keys[p + 1]['bounds_min'][0] if p + 1 in keys and regions.is_black(p + 1) else hi[0]
        row['narrow'] = (max(lo[0], left) + X_MARGIN, min(hi[0], right) - X_MARGIN)
        row['y'] = (row['y'][0], regions.black_front + WHITE_DEEP)
    return regions


def performer_solvers(rig):
    """Pose solvers for the performer (the planner's pair table keeps pose.hand_solvers)."""
    solvers = hand_solvers(rig)
    for solver in solvers.values():
        deepen(solver.regions)
        solver.neutral = solver.neutral + PLAYING_WRIST_OFFSET
    return solvers
