"""Fixed-palm reach intervals and an explicit development movement budget.

Span/overlap exclusions are necessary conditions under the bound rig model.
The speed budget is a style policy, not a universal human speed limit.
"""
import math


def wrist_interval(note, keyboard, envelopes):
    envelope=envelopes[note['hand']].get(str(note['finger']))
    if envelope is None:return None
    key=keyboard[str(note['pitch'])]
    return key['bounds_min'][0]-envelope['high_x'],key['bounds_max'][0]-envelope['low_x']


def transport_rejections(assigned, incoming_ids, keyboard, envelopes,
                         maximum_average_speed=1.5, minimum_downstroke=.018):
    if not math.isfinite(maximum_average_speed) or maximum_average_speed<=0:
        raise ValueError('Positive finite movement budget required')
    if not math.isfinite(minimum_downstroke) or minimum_downstroke<=0:
        raise ValueError('Positive finite existing driver downstroke required')
    incoming=set(incoming_ids);failures=[]
    for target in assigned:
        if target['id'] not in incoming:continue
        b=wrist_interval(target,keyboard,envelopes)
        if b is None:continue
        deadline=target['start']-minimum_downstroke
        for old in assigned:
            if old['id'] in incoming or old['hand']!=target['hand'] or old['start']>=target['start']:
                continue
            a=wrist_interval(old,keyboard,envelopes)
            if a is None:continue
            distance=max(0.,a[0]-b[1],b[0]-a[1])
            if distance<=0:continue
            available=deadline-old['physical_release']
            row=dict(hand=target['hand'],ids=[old['id'],target['id']],fingers=[old['finger'],target['finger']],
                previous_release=old['physical_release'],latest_incoming_touch=deadline,
                minimum_wrist_translation_mm=distance*1000,available_seconds=max(0.,available),
                maximum_average_speed_m_s=maximum_average_speed,
                whole_note_unreachability_proved=False)
            if available<=0:
                failures.append(dict(row,kind='fixed_palm_first_touch_overlap',necessary_for_this_owner_pair=True))
            elif distance/available>maximum_average_speed:
                failures.append(dict(row,kind='wrist_transport_speed_budget',minimum_average_speed_m_s=distance/available,
                                     necessary_for_this_owner_pair=False))
    return failures
