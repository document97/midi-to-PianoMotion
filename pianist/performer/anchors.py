"""Hand poses at every touch of the plan (motion keyframes), solved in parallel.

An anchor is one hand at the touch time of one of its onset groups. Its
contacts are the incoming keys (at the key top) and every key the hand still
holds (at its current depth). The planner screens chords pairwise with the
rig's own table; three or more held keys can still exceed the hand. Such an
anchor is repaired the way the planner itself lifts keys: the held key whose
early lift costs the least sound (pedal-held lifts cost nothing) is released
before the touch, and the anchors it affected are solved again. A reachable
anchor whose fingers splay to the side holding a key the pedal already
sustains lets go of it too: the sound is identical and the hand is free.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from itertools import combinations

import numpy as np

from planner.music import KEY_RETURN_SECONDS

from .pose import Contact

REACH_TOLERANCE = .0025         # contact error above which a held key is lifted early (the frame follower absorbs less)
STRAIN_SWING = 18.              # contact finger sideways swing (deg) above which a silent lift frees the hand
REPAIR_CUT_SECONDS = .05        # an unreachable anchor may shorten a held note's sound by at most this; beyond it
                                # the pose keeps its error (an octave at the hand's limit misses by 3-4 mm, mostly
                                # depth) rather than cutting a long bass note by seconds
REPAIR_FAR = .006               # an anchor missing by more than this may cut up to REPAIR_FAR_CUT_SECONDS
REPAIR_FAR_CUT_SECONDS = .3
SILENT_LIFT_LEAD = .3           # a lift that costs no sound happens this long before the touch (or at pedal catch)
KEY_TRAVEL = .010               # full key depth: a grand's white key dips ~10 mm at its front


def ease(x):
    """Minimum-jerk blend (rest to rest)."""
    x = min(max(x, 0.), 1.)
    return x * x * x * (10 + x * (-15 + 6 * x))


def press_profile(x):
    """Key travel during the press: the finger drives it down at a steady speed until it stops on the keybed."""
    return min(max(x, 0.), 1.)


def lift_profile(x):
    """Release: the key springs back fastest at first (the finger leaves with it)."""
    x = min(max(x, 0.), 1.)
    return 1 - (1 - x) ** 3


@dataclass
class NoteMotion:
    """One played note as the hand sees it."""
    id: int
    index: int
    hand: str
    digit: int
    pitch: int
    velocity: int
    touch: float
    onset: float
    release: float              # physical key lift
    earliest_release: float
    pedal_release: float        # earliest lift with identical sound
    group: int

    @property
    def returned(self):
        return self.release + KEY_RETURN_SECONDS

    def depth(self, t):
        """Key travel (m) at time ``t`` (performance_timing.key_depth)."""
        if t < self.touch or t >= self.returned:
            return 0.
        if t < self.onset:
            return KEY_TRAVEL * press_profile((t - self.touch) / max(self.onset - self.touch, 1e-6))
        if t <= self.release:
            return KEY_TRAVEL
        return KEY_TRAVEL * (1 - lift_profile((t - self.release) / KEY_RETURN_SECONDS))

    def held_at(self, t):
        return self.touch < t - 1e-9 and self.release > t + 1e-9


@dataclass
class Anchor:
    hand: str
    group: int
    time: float
    x_hint: float
    incoming: tuple             # note ids
    held: tuple                 # note ids
    pose: object = None
    lifted: tuple = ()          # held notes released early to make the pose feasible
    attempts: list = field(default_factory=list)

    def contacts(self, notes):
        """Incoming keys are solved pressed (their longest phase); the touch is a small lift from it."""
        return [Contact(notes[i].digit, notes[i].pitch, KEY_TRAVEL) for i in self.incoming] + \
               [Contact(notes[i].digit, notes[i].pitch, notes[i].depth(self.time)) for i in self.held]


def note_motions(rows, score):
    """``rows``: planner.problem.plan_notes output."""
    out = {}
    for r in rows:
        n = score.notes[r['index']]
        out[r['id']] = NoteMotion(r['id'], r['index'], r['hand'], r['finger'], r['pitch'], r['velocity'], r['touch'],
                                  r['start'], r['physical_release'], n.earliest_release, n.pedal_release, n.group)
    return out


def build_anchors(notes, wrist_x):
    """``wrist_x``: {(group, hand): planner wrist x}."""
    anchors = []
    for hand in 'LR':
        mine = [n for n in notes.values() if n.hand == hand]
        for group in sorted({n.group for n in mine}):
            incoming = sorted((n for n in mine if n.group == group), key=lambda n: n.pitch)
            t = min(n.touch for n in incoming)
            held = tuple(n.id for n in mine if n.group != group and n.held_at(t))
            anchors.append(Anchor(hand, group, t, wrist_x[(group, hand)], tuple(n.id for n in incoming), held))
    return anchors


# ---------- parallel solve ----------
_worker = {}


def _init(rig_path):
    from .kinematics import Rig
    rig = Rig(rig_path)
    from .regions import performer_solvers
    _worker['solvers'] = performer_solvers(rig)


def _solve(job):
    hand, contacts, x_hint = job
    solver = _worker['solvers'][hand]
    wrist = solver.neutral.copy()
    wrist[0] = x_hint
    return solver.solve(contacts, wrist=wrist, x_hint=x_hint)


def lift_cost(note, t):
    """(audible seconds lost, lift time) for releasing a held key before a touch at ``t``; None if illegal."""
    lift = max(note.earliest_release, min(max(note.pedal_release, t - SILENT_LIFT_LEAD), t - KEY_RETURN_SECONDS))
    if lift + KEY_RETURN_SECONDS > t + 1e-9 or lift >= note.release:
        return None
    return max(0., note.pedal_release - lift), lift


def strain(pose):
    """Largest sideways swing of a contact finger (deg)."""
    return max((abs(float(v[0])) for d, v in pose.values.items() if d != 1), default=0.)


def solve_anchors(anchors, notes, rig_path, workers, max_rounds=6):
    """Solve every anchor; repair infeasible ones by early lifts. Mutates ``notes`` releases."""
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(rig_path),)) as pool:
        pending = list(range(len(anchors)))
        for _ in range(max_rounds):
            jobs = [(anchors[i].hand, anchors[i].contacts(notes), anchors[i].x_hint) for i in pending]
            for i, pose in zip(pending, pool.map(_solve, jobs, chunksize=4)):
                anchors[i].pose = pose
            failed = [i for i in pending if anchors[i].held and anchors[i].pose.contact_error > REACH_TOLERANCE]
            strained = [i for i in pending if anchors[i].held and i not in failed and strain(anchors[i].pose) > STRAIN_SWING]
            if not failed and not strained:
                break
            # Candidate lifts: every subset of held keys, cheapest audible loss first, solved in parallel.
            options = []
            for i in failed + strained:
                a = anchors[i]
                for size in range(1, len(a.held) + 1):
                    for subset in combinations(a.held, size):
                        costs = [lift_cost(notes[k], a.time) for k in subset]
                        cap = 0. if i in strained else (REPAIR_FAR_CUT_SECONDS if a.pose.contact_error > REPAIR_FAR
                                                        else REPAIR_CUT_SECONDS)
                        if None in costs or sum(c for c, _ in costs) > cap:
                            continue
                        keep = tuple(k for k in a.held if k not in subset)
                        options.append((sum(c for c, _ in costs), size, i, subset, keep,
                                        {k: lift for k, (_, lift) in zip(subset, costs)}))
            options.sort(key=lambda o: (o[2], o[0], o[1]))
            jobs = [(anchors[i].hand, replace(anchors[i], held=keep).contacts(notes), anchors[i].x_hint)
                    for _, _, i, _, keep, _ in options]
            chosen = {}
            for option, pose in zip(options, pool.map(_solve, jobs, chunksize=4)):
                i = option[2]
                relieved = i in failed or strain(pose) < strain(anchors[i].pose) - 3.
                if pose.contact_error <= REACH_TOLERANCE and relieved and i not in chosen:
                    chosen[option[2]] = (option, pose)
            if not chosen:
                break
            before = [a.held for a in anchors]
            for i, ((_, _, _, subset, keep, lifts), pose) in chosen.items():
                anchors[i].lifted += subset
                anchors[i].held = keep
                anchors[i].pose = pose
                for k, lift in lifts.items():
                    notes[k].release = min(notes[k].release, lift)
            # a key lifted earlier is no longer held by later anchors: those are solved again
            for a in anchors:
                a.held = tuple(k for k in a.held if notes[k].held_at(a.time))
            pending = [j for j, a in enumerate(anchors) if a.held != before[j] and j not in chosen]
            if not pending:
                break
    return anchors


def summary(anchors):
    errors = np.array([a.pose.contact_error for a in anchors])
    return dict(anchors=len(anchors), failed=int(sum(a.pose.contact_error > REACH_TOLERANCE for a in anchors)),
                repaired=int(sum(bool(a.lifted) for a in anchors)),
                lifted_keys=int(sum(len(a.lifted) for a in anchors)),
                worst_contact_mm=float(errors.max() * 1000) if len(errors) else 0.)
