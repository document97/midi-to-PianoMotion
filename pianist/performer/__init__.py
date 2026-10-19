"""Merged single-pass performer: note planning (planner) + motion, written to a .blend.

Blender is used twice and only in one process: to export the rig once
(rig_export.py) and to write the finished animation (blender_main.py).
Everything in between is plain numpy (kinematics.py, motion.py) and runs in
a pool of ordinary Python worker processes, so the solve scales to 8 / 16
cores without starting more than one Blender process.
"""
