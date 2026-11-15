"""Beam search over onset groups: hand, finger, key lift and omission.

Per group, each hand independently proposes (incoming subset, finger tuple)
candidates against its own state; hand pairs with disjoint subsets are then
combined. Held keys are lifted lazily: a key keeps sounding (through the
pedal where possible) and is lifted only when its finger, its key or the
wrist is needed. All geometry goes through the hand model interface
(hand_model.py), so a motion-aware model can replace the geometric one.
"""
from dataclasses import dataclass, replace
from itertools import combinations
import math

from .hand_model import GeometricHandModel, RigHandModel, WristPose
from .instrument import FINGERS, HANDS
from .music import KEY_RETURN_SECONDS, MINIMUM_DOWNSTROKE_SECONDS, hand_split_pitches

INF = math.inf
CROSSING_SECONDS = .8
REPEAT_SECONDS = .5             # a finger hopping to another key sooner than this costs repeat_finger
KEY_REPEAT_SECONDS = .5         # a key played again sooner than this by another finger costs key_finger_change
CROSSING_AGAIN_SECONDS = .6
HANDS_NEAR_SECONDS = .4         # a hand's keys of this recent past still mark where its fingers are
HAND_EDGE = .01                 # a hand reaches this far beyond the finger next to the other hand (m)
REPEAT_MAX_COST = 3.            # a finger hop costs at most this: well under an omitted note (omission >= 8)
ARCH_PEAK_SECONDS = .8          # a same-finger hop this long squeezes the whole relax cycle (lift .3 + prepare .22,
                                # motion.py) into the gap: the full up-and-down arch lands inside the phrase and reads
                                # as a flick. Below the glide window the finger barely leaves its key (a glide);
                                # above the fade the arch gets its dwell and reads as natural relaxation
ARCH_FADE_SECONDS = 1.4         # the flick charge fades out by this gap


def _thumb_reach(finger):
    """Weight of the thumb passing ``finger``: under the index or middle finger by technique, the ring finger
    hardly, beyond the little finger only by moving the whole hand."""
    return 1. + 2. * max(0, finger - 3)


def _crossed(direction, f1, p1, f2, p2):
    """Two non-thumb fingers whose order across the hand contradicts their keys' order."""
    return f1 >= 2 and f2 >= 2 and f1 != f2 and p1 != p2 and direction * (p1 - p2) * (f1 - f2) < 0


def _arch_squeeze(gap):
    """(0..1) how visibly the same finger's relax cycle is compressed between two keys: 0 below the glide
    window (the finger barely leaves its key), rising to the peak where the fixed lift+prepare cycle just
    fits into the gap with no dwell left, easing back to 0 once the arch reads as unhurried relaxation."""
    lo, peak, hi = REPEAT_SECONDS, ARCH_PEAK_SECONDS, ARCH_FADE_SECONDS
    if gap <= lo or gap >= hi:
        return 0.
    x = (gap - lo) / (peak - lo) if gap <= peak else (hi - gap) / (hi - peak)
    return x * x * (3. - 2. * x)          # smoothstep: no cliff at either end


@dataclass(frozen=True)
class Weights:
    omission: float = 8.             # per omitted note, scaled by importance
    wrist_metre: float = 10.         # per metre of wrist travel
    soft_speed: float = 1.5          # m/s average wrist speed above which travel costs more
    speed_excess: float = 4.         # per m/s above soft_speed
    hard_speed: float = 4.           # m/s average; above it is rejected
    lift_speed: float = 4.           # m/s average the release timing plans for (keys are held until the wrist must
                                     # leave at this speed); lower values buy unhurried hand travel with sound
    stretch_metre: float = 40.       # per metre beyond a relaxed finger spacing (without the rig's pair strain)
    strain_unit: float = 9.          # per unit of the rig hand's strain holding a shape (hand_model.shape_strain) ...
    strain_free: float = 0.          # ... beyond this many: none, as the planner compares fingerings (a finger swung
                                     # to its sideways limit is only AWKWARD_SIDEWAYS units, under the performer's free)
    finger_spacing: float = .0235    # relaxed spacing per finger index, fingers 2..5 (one white key; the performer
                                     # measures its rig's hand instead)
    thumb_spacing: float = .0235     # relaxed thumb-to-index spacing
    outer_on_black: float = .6       # thumb or little finger on a black key
    thumb_crossing: float = .4       # thumb passing under/over a held finger, or right after its key
    crossing_again: float = 2.       # another thumb crossing soon after the last one (CROSSING_AGAIN_SECONDS):
                                     # a scale passes the thumb now and then, a fast figure must not every note
    early_release: float = .05       # lifting before note-off with identical sound
    shortening: float = 1.            # share of a note's omission cost for cutting all of its sustain (fidelity metric: a cut bass ~ an omitted one)
    repeat_finger: float = 3.        # same finger on a different key soon after (REPEAT_SECONDS): more the sooner
                                     # and the further it jumps (a finger hopping in a fast figure)
    key_finger_change: float = 1.5   # a key played again soon after (KEY_REPEAT_SECONDS) by another finger: a repeated
                                     # note of a figure keeps its finger, so the hand keeps its place
    finger_crossing: float = 2.      # fingers 2..5 crossing each other (sequence within CROSSING_SECONDS, or over a held key);
                                     # only the thumb passes under the hand by technique
    wrong_side: float = 3.           # per octave beyond the local hand split
    hands_overlap: float = 100.      # per metre the hands' finger ranges (keys held or just played, widened by the
                                     # fingers on the inner side) overlap: one hand must not land on the other
    lift_pedalled: bool = True       # a held key the pedal already sustains (lifting it sounds the same) is let go
                                     # when the hand plays on: it does not shape the hand (stretch, crossings)


@dataclass(frozen=True)
class Held:
    note: int
    finger: int
    pitch: int
    end: float
    pedal: float     # earliest lift with identical sound
    earliest: float  # earliest permitted lift


@dataclass(frozen=True)
class HandState:
    pose: object = None          # hand-model pose; None before the first touch
    last_touch: float = -INF
    pin_release: float = -INF    # latest lift of a key that pinned the pose since last_touch
    held: tuple = ()
    last_finger_use: tuple = ()  # ((finger, pitch, touch), ...): each finger's latest key
    last_release: tuple = ()     # ((finger, pitch, lift), ...): each finger's latest key let go, and when
    last_crossing: float = -INF  # touch of the hand's latest thumb crossing

    def signature(self):
        pose = -1 << 30 if self.pose is None else self.pose.signature()   # sortable
        last = tuple((f, p) for f, p, t in self.last_finger_use if t >= self.last_touch - 1e-9)
        return (pose, tuple((h.finger, h.pitch) for h in self.held), last)


@dataclass(frozen=True)
class Decision:
    group: int
    assignment: tuple          # ((note, hand, finger), ...); notes not listed are omitted
    releases: tuple            # ((note, lift time), ...) for keys lifted before note-off
    poses: tuple               # ((hand, pose), ...) placed in this group


@dataclass
class State:
    cost: float
    hands: dict
    trail: tuple = None        # (Decision, parent trail): shared persistent list

    def signature(self):
        return (self.hands['L'].signature(), self.hands['R'].signature())

    def decisions(self):
        out, node = [], self.trail
        while node is not None:
            out.append(node[0])
            node = node[1]
        return out[::-1]


@dataclass(frozen=True)
class HandOption:
    cost: float
    state: HandState
    releases: tuple
    fingers: tuple


def initial_state(rest_x=None):
    """``rest_x``: {hand: wrist x (m)} where each hand rests before the piece: its first touch travels from there
    (without it a hand's first key costs no travel, and an idle hand takes a note far from its own register)."""
    return State(0., {h: HandState(pose=None if rest_x is None else WristPose(rest_x[h])) for h in HANDS})


class Planner:
    def __init__(self, score, instrument, weights=Weights(), beam_width=48, finger_options=4, model=None,
                 outer_voice_bonus=1., pair_table=None, pair_strain=None, shape_penalty=None, rest_x=None, reach_x=None):
        """``pair_table``: two-finger feasibility of the animated rig (performer.reach_table); without it
        the fixed-palm screens are used. ``shape_penalty``: {(hand, frozenset of (note, finger) on the keys)}: cost
        of the animated hand's awkwardness in that shape (performer.pipeline.awkward_shapes); the search weighs it
        against other fingers, an omitted note or a held key let go early. ``reach_x``: {hand: (lowest, highest) key
        centre x} the hand reaches (performer.torso.reach_window); a key beyond both hands' reach is omitted."""
        self.score, self.w = score, weights
        outer = score.outer_voice_notes()
        self.voice_scale = [outer_voice_bonus if i in outer else 1. for i in range(len(score.notes))]
        self.model = model or (RigHandModel(instrument, pair_table, pair_strain) if pair_table is not None
                               else GeometricHandModel(instrument))
        self.beam_width, self.finger_options = beam_width, finger_options
        self.split = hand_split_pitches(score)
        self.shape_penalty = shape_penalty or {}
        self.rest_x = rest_x
        self.reach_x = reach_x or {}
        self.last_reason = None

    def initial_state(self):
        return initial_state(self.rest_x)

    def _reject(self, reason):
        self.last_reason = reason
        return None

    def _importance(self, note):
        n = self.score.notes[note]
        value = 1. + .5 * n.velocity / 127. + .5 * min(n.end - n.start, 2.) / 2.
        return value * self.voice_scale[note]

    # ---------- state normalisation ----------
    @staticmethod
    def _advance(hs, touch):
        """Drop keys whose natural lift and key return finished before ``touch``."""
        kept, pin = [], hs.pin_release
        for h in hs.held:
            if h.end + KEY_RETURN_SECONDS <= touch + 1e-9:
                pin = max(pin, h.end)
            else:
                kept.append(h)
        return replace(hs, held=tuple(kept), pin_release=pin)

    def _lift_cost(self, held, release):
        """Small cost for any early lift, plus the musical cost of sound actually cut."""
        score = self.score
        start = score.notes[held.note].start
        cost = self.w.early_release * (held.end - release) / max(held.end - start, 1e-6)
        if release < held.pedal - 1e-9:
            # The string sounds on until the damper closes (pedal up), not just until the lift.
            lost = score.audible_shortening(held.note, release) / score.sounding_seconds(held.note)
            cost += self.w.shortening * self.w.omission * self._importance(held.note) * lost
        return cost

    @staticmethod
    def _lift_time(held, touch):
        """Latest lift before ``touch`` that is no later than the pedal makes inaudible."""
        return min(held.end, max(held.earliest, min(held.pedal, touch - KEY_RETURN_SECONDS)))

    # ---------- one hand ----------
    def _conflicts(self, hand, held, incoming):
        """True if a held key cannot coexist with the incoming (finger, pitch) set."""
        direction = 1 if hand == 'R' else -1
        for finger, pitch in incoming:
            if held.finger == finger or held.pitch == pitch:
                return True
            if not self.model.pair_feasible(hand, held.finger, held.pitch, finger, pitch):
                return True
            # Existing configured non-thumb preparation order.
            if held.finger != 1 and finger != 1 and direction * (pitch - held.pitch) * (finger - held.finger) < 0:
                return True
        return False

    def evaluate_hand(self, hand, hs, group, notes, fingers):
        """HandOption for ``hand`` playing ``notes`` with ``fingers``; None if infeasible."""
        score, w, model = self.score, self.w, self.model
        touch = group.touch
        deadline = group.start - MINIMUM_DOWNSTROKE_SECONDS
        incoming = [(f, score.notes[n].pitch) for n, f in zip(notes, fingers)]
        window = model.chord_window(hand, incoming)
        if window is None:
            return self._reject('incoming_span')
        lifted, kept = [], []
        for h in hs.held:
            a, b = model.key_window(hand, h.finger, h.pitch)
            compatible = not self._conflicts(hand, h, incoming) and a <= window[1] and b >= window[0]
            pedalled = w.lift_pedalled and h.earliest <= h.pedal <= touch - KEY_RETURN_SECONDS + 1e-9
            (kept if compatible and not pedalled else lifted).append(h)
        if any(h.earliest + KEY_RETURN_SECONDS > touch + 1e-9 for h in lifted):
            return self._reject('held_key_not_liftable')
        kept, freed, awkward = self._awkward_lift(hand, kept, notes, fingers, touch)
        lifted += freed
        keys = [(h.finger, h.pitch) for h in kept] + incoming
        placement = model.place(hand, hs.pose, keys)
        if placement is None:
            return self._reject('held_span')
        # One hand position covers the keys of its last touch and the new ones, or the hand moves: then the wrist
        # travels at least between where each set is comfortable (a finger reaching across from the old position
        # to a key no one position holds with the last ones is no free move)
        recent = [(f, p) for f, p, t in hs.last_finger_use
                  if t >= hs.last_touch - 1e-9 and all(p != q for _, q in keys)]
        if recent and model.chord_window(hand, keys + recent) is None:
            moved = abs(model.comfortable_x(hand, keys) - model.comfortable_x(hand, recent))                 if model.chord_window(hand, recent) is not None else 0.
            if moved > placement.travel_metres:
                placement = replace(placement, travel_metres=moved)
        releases = [(h.note, self._lift_time(h, touch)) for h in lifted]
        cost = w.wrist_metre * placement.travel_metres
        if placement.travel_metres > 1e-6:
            # The wrist must start moving by ``latest_start``; lift only as early as that needs.
            latest_start = deadline - placement.travel_metres / w.hard_speed
            planned_start = deadline - placement.travel_metres / min(w.lift_speed, w.hard_speed)
            releases = [(h.note, max(h.earliest, min(r, planned_start))) for h, (_, r) in zip(lifted, releases)]
            start = max([hs.last_touch, hs.pin_release] + [r for _, r in releases])
            if start > latest_start + 1e-9:
                return self._reject('travel_too_fast')
            cost += w.speed_excess * max(0., placement.travel_metres / (deadline - start) - w.soft_speed)
        for h, (_, r) in zip(lifted, releases):
            cost += self._lift_cost(h, r)
        shape_cost, crossed = self._shape_cost(hand, hs, group, kept, incoming)
        cost += shape_cost + awkward
        new_held = [Held(n, f, score.notes[n].pitch, score.notes[n].end, score.notes[n].pedal_release,
                         score.notes[n].earliest_release) for n, f in zip(notes, fingers)]
        state = HandState(pose=placement.pose, last_touch=touch, pin_release=-INF,
                          last_crossing=touch if crossed else hs.last_crossing,
                          held=tuple(sorted(kept + new_held, key=lambda h: (h.finger, h.pitch))),
                          last_finger_use=tuple(sorted({**{u[0]: u for u in hs.last_finger_use},
                                                        **{f: (f, p, touch) for f, p in incoming}}.values())),
                          last_release=tuple(sorted({**{u[0]: u for u in hs.last_release},
                                                     **{h.finger: (h.finger, h.pitch, r) for h, (_, r) in zip(lifted, releases)}}.values())))
        return HandOption(cost, state, tuple(releases), tuple(fingers))

    def _awkward_lift(self, hand, kept, notes, fingers, touch):
        """(kept, lifted early, awkwardness cost): the held keys whose early lift (at its sound cost) makes the
        shape's awkwardness (``shape_penalty``) cheapest; none when keeping them all costs least."""
        if not self.shape_penalty:
            return kept, [], 0.
        playing = frozenset(zip(notes, fingers))

        def penalty(held):
            return self.shape_penalty.get((hand, playing | {(h.note, h.finger) for h in held}), 0.)

        best = (penalty(kept), kept, [])
        if best[0] <= 0.:
            return kept, [], 0.
        liftable = [h for h in kept if h.earliest + KEY_RETURN_SECONDS <= touch + 1e-9]
        for size in range(1, len(liftable) + 1):
            for gone in combinations(liftable, size):
                stay = [h for h in kept if h not in gone]
                cost = penalty(stay) + sum(self._lift_cost(h, self._lift_time(h, touch)) for h in gone)
                if cost < best[0] - 1e-9:
                    best = (cost - sum(self._lift_cost(h, self._lift_time(h, touch)) for h in gone), stay, list(gone))
        return best[1], best[2], best[0]

    def _shape_cost(self, hand, hs, group, kept, incoming):
        w, model = self.w, self.model
        cost = 0.
        shape = sorted([(h.finger, h.pitch) for h in kept] + incoming)
        strain = model.shape_strain(hand, shape) if len(shape) > 1 and hasattr(model, 'shape_strain') else None
        if strain is not None:
            # how strained the animated hand holds the shape (measured on the rig, as the performer prices it)
            cost += w.strain_unit * max(0., strain - w.strain_free)
        else:
            for (f1, p1), (f2, p2) in zip(shape, shape[1:]):
                gap = abs(model.key_center(p2) - model.key_center(p1))
                relaxed = (w.thumb_spacing + (f2 - 2) * w.finger_spacing) if f1 == 1 else (f2 - f1) * w.finger_spacing
                cost += w.stretch_metre * max(0., gap - relaxed * 1.6)
        direction = 1 if hand == 'R' else -1
        split = self.split[group.index]
        # The whole hand shifting (an octave or a chord moved as one shape) is wrist travel, not a finger hopping.
        last = {u[0]: u[1] for u in hs.last_finger_use if u[2] >= hs.last_touch - 1e-9}
        moved = {p - last[f] for f, p in incoming if f in last}
        shifted = len(incoming) >= 2 and all(f in last for f, _ in incoming) and len(moved) == 1
        crossings = 0
        for f, p in incoming:
            cost += w.wrong_side * max(0., direction * (split - p)) / 12.
            if f in (1, 5) and model.is_black(p):
                cost += w.outer_on_black
            if f == 1:
                crossings += sum(_thumb_reach(h.finger) for h in kept if direction * (p - h.pitch) > 0)
            elif any(h.finger == 1 and direction * (h.pitch - p) > 0 for h in kept):
                crossings += _thumb_reach(f)
            for used_finger, used_pitch, used_touch in hs.last_finger_use:
                if used_finger == f and used_pitch != p and not shifted:
                    gap = group.touch - used_touch
                    hop = .5 + min(abs(p - used_pitch), 12) / 6.      # a further jump costs more
                    soon = max(0., 1. - gap / REPEAT_SECONDS)         # a fast figure must not hop a finger
                    cost += min(w.repeat_finger * soon * hop, REPEAT_MAX_COST)
                    squeeze = _arch_squeeze(gap)                      # the compressed arch between keys: the
                    if squeeze > 0.:                                  # visible flick -- prefer a free finger
                        cost += min(w.repeat_finger * squeeze * hop, REPEAT_MAX_COST)
                elif used_finger != f and used_pitch == p:
                    cost += w.key_finger_change * max(0., 1. - (group.touch - used_touch) / KEY_REPEAT_SECONDS)
                if used_touch < hs.last_touch - 1e-9:
                    continue                     # crossings: the hand's last group only
                if 1 in (f, used_finger) and f != used_finger and not any(h.pitch == used_pitch for h in kept)                         and direction * (p - used_pitch) * (f - used_finger) < 0 and group.touch - used_touch < CROSSING_SECONDS:
                    crossings += _thumb_reach(max(f, used_finger))   # passing the thumb right after a key was let go
                if _crossed(direction, f, p, used_finger, used_pitch) and group.touch - used_touch < CROSSING_SECONDS:
                    cost += w.finger_crossing
            cost += w.finger_crossing * sum(_crossed(direction, f, p, h.finger, h.pitch) for h in kept)
        if crossings:
            cost += w.thumb_crossing * crossings
            cost += w.crossing_again * max(0., 1. - (group.touch - hs.last_crossing) / CROSSING_AGAIN_SECONDS)
        return cost, crossings > 0

    def hand_candidates(self, hand, hs, group, playable):
        """Best finger tuples for every subset of ``playable`` notes (ascending pitch)."""
        out = {(): [HandOption(0., hs, (), ())]}
        pool = FINGERS if hand == 'R' else FINGERS[::-1]   # chord fingers follow pitch order
        for size in range(1, len(playable) + 1):
            for subset in combinations(playable, size):
                options = [o for fingers in combinations(pool, size)
                           if (o := self.evaluate_hand(hand, hs, group, subset, fingers)) is not None]
                if options:
                    options.sort(key=lambda o: (o.cost, o.fingers))
                    out[subset] = options[:self.finger_options]
        return out

    # ---------- one group ----------
    def _restrike(self, hands, group):
        """Lift any held key that a group note restrikes; return (playable, lifts)."""
        playable, lifts = [], []
        for n in group.notes:
            pitch, blocked = self.score.notes[n].pitch, False
            for hand in HANDS:
                for held in hands[hand].held:
                    if held.pitch != pitch:
                        continue
                    if held.earliest + KEY_RETURN_SECONDS <= group.touch + 1e-9:
                        lift = self._lift_time(held, group.touch)
                        lifts.append((held.note, lift))
                        hands[hand] = replace(hands[hand], pin_release=max(hands[hand].pin_release, lift),
                                              held=tuple(x for x in hands[hand].held if x is not held))
                    else:
                        blocked = True
            if not blocked:
                playable.append(n)
        return playable, lifts

    def expand(self, state, group, forced=None):
        """Successor states; ``forced`` fixes the assignment (used to replay a plan)."""
        hands = {h: self._advance(state.hands[h], group.touch) for h in HANDS}
        playable, lifts = self._restrike(hands, group)
        if forced is None:
            # a hand-split score (a pianist's own division) is kept: each hand is offered only its own notes
            candidates = {h: self.hand_candidates(h, hands[h], group, [n for n in playable if self.score.notes[n].hand in (None, h)
                                                                       and self._reachable(h, self.score.notes[n].pitch)])
                          for h in HANDS}
        else:
            candidates = self._forced_candidates(hands, group, playable, forced)
            if candidates is None:
                return []
        successors = []
        for left_notes, left_options in candidates['L'].items():
            for right_notes, right_options in candidates['R'].items():
                if set(left_notes) & set(right_notes):
                    continue
                left_keys = {self.score.notes[n].pitch for n in left_notes}
                right_keys = {self.score.notes[n].pitch for n in right_notes}
                if left_keys & right_keys:
                    continue                     # one key, two hands: a unison of two voices is played once
                played = set(left_notes) | set(right_notes)
                sounding = left_keys | right_keys
                omission = sum(self.w.omission * self._importance(n) for n in group.notes
                               if n not in played and self.score.notes[n].pitch not in sounding)
                for lo in left_options:
                    for ro in right_options:
                        if not self.model.hands_ordered(lo.state.pose, ro.state.pose):
                            continue
                        apart = self.w.hands_overlap * self._hands_overlap(lo.state, ro.state, group.touch)
                        assignment = tuple(sorted([(n, 'L', f) for n, f in zip(left_notes, lo.fingers)] +
                                                  [(n, 'R', f) for n, f in zip(right_notes, ro.fingers)]))
                        poses = tuple((h, o.state.pose) for h, o, notes in
                                      (('L', lo, left_notes), ('R', ro, right_notes)) if notes)
                        decision = Decision(group.index, assignment, tuple(lifts) + lo.releases + ro.releases, poses)
                        successors.append(State(state.cost + lo.cost + ro.cost + omission + apart,
                                                {'L': lo.state, 'R': ro.state}, (decision, state.trail)))
        return successors

    def _reachable(self, hand, pitch):
        if hand not in self.reach_x:
            return True
        lo, hi = self.reach_x[hand]
        return lo <= self.model.key_center(pitch) <= hi

    def _hands_overlap(self, left, right, touch):
        """Metres the left hand's finger range reaches past the right hand's (0 when apart)."""
        edges = []
        for hs, sign in ((left, 1.), (right, -1.)):
            # a key let go is near for a while after its lift, however long ago it was struck
            keys = [(h.finger, h.pitch) for h in hs.held] +                    [(f, p) for f, p, t in hs.last_finger_use + hs.last_release if t >= touch - HANDS_NEAR_SECONDS]
            if not keys:
                return 0.
            inner = [self.model.key_center(p) + sign * ((f - 1) * self.w.finger_spacing + HAND_EDGE) for f, p in keys]
            edges.append(max(inner) if sign > 0 else min(inner))
        return max(0., edges[0] - edges[1])

    def _forced_candidates(self, hands, group, playable, forced):
        candidates = {}
        for hand in HANDS:
            chosen = sorted(((n, f) for n, h, f in forced if h == hand), key=lambda x: self.score.notes[x[0]].pitch)
            if any(n not in playable for n, _ in chosen):
                return None
            if not chosen:
                candidates[hand] = {(): [HandOption(0., hands[hand], (), ())]}
                continue
            notes, fingers = zip(*chosen)
            option = self.evaluate_hand(hand, hands[hand], group, notes, fingers)
            if option is None:
                return None
            candidates[hand] = {notes: [option]}
        return candidates

    def run(self, groups, state=None):
        beam = [state or self.initial_state()]
        for group in groups:
            successors = [s for b in beam for s in self.expand(b, group)]
            if not successors:
                raise RuntimeError(f'No successor at group {group.index}')
            successors.sort(key=lambda s: (s.cost, s.signature()))
            seen, beam = set(), []
            for s in successors:
                key = s.signature()
                if key not in seen:
                    seen.add(key)
                    beam.append(s)
                    if len(beam) == self.beam_width:
                        break
        return beam
