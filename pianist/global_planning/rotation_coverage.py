"""Two-finger spans a hand with its calibrated palm orientation cannot cover (a fast screen before the rig's pair table)."""


def impossible_fixed_palm_pairs(notes, keyboard, envelopes):
    failures=[]
    for i,left in enumerate(notes):
        for right in notes[i+1:]:
            if left['hand']!=right['hand']:continue
            h=left['hand'];a=envelopes[h].get(str(left['finger']));b=envelopes[h].get(str(right['finger']))
            if a is None or b is None:continue
            ka,kb=keyboard[str(left['pitch'])],keyboard[str(right['pitch'])]
            required=max(0.,ka['bounds_min'][0]-kb['bounds_max'][0],kb['bounds_min'][0]-ka['bounds_max'][0])
            maximum=max(abs(a['high_x']-b['low_x']),abs(b['high_x']-a['low_x']))
            if required>maximum:
                failures.append(dict(hand=h,ids=[left['id'],right['id']],fingers=[left['finger'],right['finger']],
                    minimum_key_roof_separation_mm=required*1000,conservative_pad_pair_span_mm=maximum*1000,
                    scope='Fixed calibrated palm orientation and current ROM only; not an omission proof'))
    return failures
