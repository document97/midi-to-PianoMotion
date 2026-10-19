"""Configured neutral non-thumb order at first touch, not a human limit.

Source onsets alone miss two fingers sharing keys during downstroke. Thumb
crossings and already released contacts are deliberately outside this screen.
Every admitted proposal still requires native geometry and continuous motion.
"""
EPS = 1e-7


def violations(notes, incoming_ids):
    incoming = set(incoming_ids)
    rows = {}
    for target in notes:
        if target['id'] not in incoming or target['finger'] == 1:
            continue
        for old in notes:
            if (old['id'] == target['id'] or old['hand'] != target['hand']
                    or old['finger'] == 1 or old['finger'] == target['finger']
                    or old['pitch'] == target['pitch']):
                continue
            clock=max(old['touch'],target['touch'])
            if clock>min(old['physical_release'],target['physical_release'])+EPS:
                continue
            direction = 1 if target['hand'] == 'R' else -1
            if ((target['pitch'] - old['pitch']) * direction
                    * (target['finger'] - old['finger'])) < 0:
                key = tuple(sorted((target['id'], old['id'])))
                rows[key] = dict(kind='neutral_nonthumb_preparation_order',
                    hand=target['hand'], ids=[old['id'], target['id']],
                    fingers=[old['finger'], target['finger']],
                    pitches=[old['pitch'], target['pitch']], time=clock,
                    necessary_for_this_owner_pair=False,
                    human_unreachability_proof=False)
    return list(rows.values())
