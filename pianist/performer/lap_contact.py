"""Where the resting hands sit on the thighs.

``rig_export.lap`` gives the point of the thigh's surface the wrist goes to, but a hand is a body with
thickness and a thigh is a round slope: laid on that point at the lap pitch, the palm sinks a few
millimetres into the leg, the little finger sits ~10 mm inside it, and the index finger hangs ~15 mm above
it (the surface falls away toward the knee and toward the outside of the thigh). No single rigid placement
fixes that -- the fingers have to follow the slope, each a little differently.

So the pose itself was fitted against the rendered body: the wrist searched in a box around the lap point,
the hand rolled a few degrees either side of its natural hang, and each finger's knuckle and middle joint
bent a few degrees, until every finger pad lay on the leg within 2 mm and no part of either hand was more
than a millimetre inside anything. That is the same clearance idea the elbows use against the trunk,
measured on the real surface instead of a proxy, and done once because the seat is fixed: ``rig_export``'s
seating places the body and the same mesh follows it.

The numbers below belong to one body. A different mesh, a new seat or a re-exported rig needs them
measured again -- that is one command on the built blend, ~15 s for both hands:

    blender -b <run>/performer.blend --python output/fit_lap.py -- \
        --rig <base.json> --frame 19 --write release/pianist/performer/lap_contact.py

It prints the block it wrote, and the pads / skin numbers it reached, so a new body can be checked the
same way this one was. The full arm is dropped from the surface it measures against (palm included), or
the pads read as buried in their own hand.

The numbers are millimetres from the lap point, degrees from the natural hang, and finger values
(sideways, lift, knuckle, middle) in degrees, per hand.
"""

import numpy as np

OFFSET = {'L': np.array([5.9, 8.9, 8.8]),          # mm
          'R': np.array([3.8, -0.0, 16.2])}
ROLL = {'L': -13.4, 'R': 12.6}                    # deg, absolute (the natural hang is -10 / +10)
FINGERS = {
    'L': {1: (1.9, 2.0, 15.6, 8.3), 2: (5.0, -10.0, 16.2, 8.4), 3: (3.8, -5.9, 13.6, 7.2),
          4: (0.7, -1.1, 10.6, 6.2), 5: (0.4, 3.4, 8.6, 5.6)},
    'R': {1: (-6.2, -4.5, 12.6, 6.9), 2: (-5.0, -17.6, 20.3, 9.8), 3: (-5.0, -11.5, 16.9, 8.7),
          4: (-1.7, -6.8, 13.6, 7.2), 5: (0.3, -2.3, 11.3, 6.6)},
}


def place(rest, hand):
    """Move a hand's rest pose (x, y, z, yaw, roll) onto the fitted place, in place."""
    rest[hand][:3] += OFFSET[hand] * .001
    rest[hand][4] = ROLL[hand]


def shape(hand):
    """The resting finger shape: {digit: (sideways, lift, knuckle, middle)}."""
    return {d: np.array(v, dtype=float) for d, v in FINGERS[hand].items()}
