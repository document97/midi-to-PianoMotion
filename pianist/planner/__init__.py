"""Fast musical note planner (hand, finger, release, omission).

Pure Python/numpy, no Blender. Geometry screens are precomputed tables that
are tested against the reach rules in global_planning; the finished plan is
independently re-validated with those rules (planner.validate).
Mesh-level contact solving belongs to the motion stage, not to this planner.
"""
