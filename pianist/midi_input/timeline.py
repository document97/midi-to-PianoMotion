"""Lossless musical event clocks and workload inputs for future motion planning.

Counts use quarter notes explicitly. They are musical demands, not anatomical
limits or a proof of physical finger availability. No note is discarded.
"""
from bisect import bisect_right
from collections import defaultdict,deque
import hashlib
from io import BytesIO
from pathlib import Path
import mido


class TempoClock:
    def __init__(self,ticks_per_quarter,events):
        if ticks_per_quarter<=0:
            raise ValueError('SMPTE timing requires a separate explicit clock')
        self.ticks_per_quarter=ticks_per_quarter
        bank={0:500000}
        for event in events:
            if event['tempo']<=0:
                raise ValueError('MIDI tempo must be positive')
            bank[event['tick']]=event['tempo']
        self.rows=[]
        seconds=0.
        previous_tick=0
        previous_tempo=500000
        for tick,tempo in sorted(bank.items()):
            seconds+=(tick-previous_tick)*previous_tempo/(1e6*ticks_per_quarter)
            self.rows.append(dict(tick=tick,seconds=seconds,tempo=tempo,
                quarter_bpm=60e6/tempo))
            previous_tick,previous_tempo=tick,tempo
        self.ticks=[row['tick'] for row in self.rows]
        self.seconds=[row['seconds'] for row in self.rows]

    def seconds_at_tick(self,tick):
        if tick<0:
            raise ValueError('Negative musical time')
        row=self.rows[bisect_right(self.ticks,tick)-1]
        return row['seconds']+(tick-row['tick'])*row['tempo']/(1e6*self.ticks_per_quarter)

    def tick_at_seconds(self,seconds):
        if seconds<0:
            raise ValueError('Negative musical time')
        row=self.rows[bisect_right(self.seconds,seconds)-1]
        return row['tick']+(seconds-row['seconds'])*1e6*self.ticks_per_quarter/row['tempo']

    def quarter_bpm_at_seconds(self,seconds):
        if seconds<0:
            raise ValueError('Negative musical time')
        return self.rows[bisect_right(self.seconds,seconds)-1]['quarter_bpm']


def load_midi(path,hand_map=None):
    path=Path(path).resolve()
    payload=path.read_bytes()
    midi=mido.MidiFile(file=BytesIO(payload))
    if midi.type==2:
        raise ValueError('Asynchronous type-2 tracks need explicit separate performances')
    if midi.ticks_per_beat<=0:
        raise ValueError('SMPTE timing is not this quarter-note clock')
    hand_map=hand_map or {}
    if any(value not in ('L','R',None) for value in hand_map.values()):
        raise ValueError('Hand mapping values must be L, R or null')
    events=[]
    names={}
    for track,stream in enumerate(midi.tracks):
        names[track]=stream.name
        tick=0
        for order,message in enumerate(stream):
            if message.time<0 or int(message.time)!=message.time:
                raise ValueError('Invalid MIDI delta tick')
            tick+=message.time
            events.append(dict(tick=int(tick),track=track,order=order,message=message.dict()))
    events.sort(key=lambda row:(row['tick'],row['track'],row['order']))
    tempo_events=[dict(tick=row['tick'],tempo=row['message']['tempo'],
        track=row['track'],order=row['order']) for row in events if row['message']['type']=='set_tempo']
    clock=TempoClock(midi.ticks_per_beat,tempo_events)
    diagnostics=[]
    tempo_at={}
    for event in tempo_events:
        old=tempo_at.get(event['tick'])
        if old is not None and old!=event['tempo']:
            diagnostics.append(dict(kind='conflicting_tempo_at_same_tick',tick=event['tick'],
                policy='Last event in deterministic track/message order',previous=old,chosen=event['tempo']))
        tempo_at[event['tick']]=event['tempo']
    notes=[]
    active=defaultdict(deque)
    for event in events:
        event['seconds']=clock.seconds_at_tick(event['tick'])
        message=event['message']
        kind=message['type']
        if kind not in ('note_on','note_off'):
            continue
        key=(message['channel'],message['note'])
        if kind=='note_on' and message['velocity']>0:
            identity=f"{event['track']}:{event['order']}"
            declared=names[event['track']].strip().lower()
            hand=hand_map.get(f"{event['track']}:{message['channel']}",
                {'left hand':'L','right hand':'R'}.get(declared))
            if active[key]:
                diagnostics.append(dict(kind='overlapping_same_channel_pitch',tick=event['tick'],
                    channel=key[0],pitch=key[1],note_off_pairing='FIFO; MIDI supplies no voice identity'))
            note=dict(id=len(notes),source_event=identity,track=event['track'],
                channel=key[0],pitch=key[1],velocity=message['velocity'],
                start_tick=event['tick'],end_tick=None,start=event['seconds'],end=None,
                hand=hand,hand_origin='explicit track/channel map' if f"{event['track']}:{key[0]}" in hand_map
                    else ('explicit track name' if hand else 'unassigned'))
            notes.append(note)
            active[key].append(note)
        elif active[key]:
            note=active[key].popleft()
            note.update(end_tick=event['tick'],end=event['seconds'],
                note_off_event=f"{event['track']}:{event['order']}")
        else:
            diagnostics.append(dict(kind='unmatched_note_off',tick=event['tick'],
                track=event['track'],channel=key[0],pitch=key[1]))
    for bank in active.values():
        for note in bank:
            diagnostics.append(dict(kind='unclosed_note',id=note['id'],source_event=note['source_event']))
    end_tick=max((event['tick'] for event in events),default=0)
    return dict(schema='midi-musical-timeline',source=str(path),
        source_sha256=hashlib.sha256(payload).hexdigest(),midi_type=midi.type,
        ticks_per_quarter=midi.ticks_per_beat,duration=clock.seconds_at_tick(end_tick),
        tempo_events=tempo_events,tempo_map=clock.rows,tracks=names,events=events,
        notes=notes,diagnostics=diagnostics,notes_removed=0,
        event_order='Absolute tick, original track, original message order',
        physical_holds_planned=False,global_fingering_planned=False,
        interpretation='Raw musical note-off times and all pedal/control/meta events are retained. Hand labels require explicit evidence. Quarter BPM is not an inferred conducting pulse.')


def workload(timeline,start,horizon_quarters=2.,keyboard=None):
    if start<0 or horizon_quarters<=0:
        raise ValueError('A nonnegative start and positive quarter-note horizon are required')
    clock=TempoClock(timeline['ticks_per_quarter'],timeline['tempo_events'])
    tick=clock.tick_at_seconds(start)
    end=clock.seconds_at_tick(tick+horizon_quarters*timeline['ticks_per_quarter'])
    result=dict(start=start,end=end,horizon_quarters=horizon_quarters,
        quarter_bpm_at_start=clock.quarter_bpm_at_seconds(start),hands={})
    for hand in ('L','R',None):
        bank=[note for note in timeline['notes'] if note['hand']==hand]
        incoming=[note for note in bank if start-1e-9<=note['start']<end-1e-9]
        held=[note for note in bank if note['start']<=start and (note['end'] is None or note['end']>start)]
        boundaries=sorted({start,*[note['start'] for note in incoming],
            *[note['end'] for note in bank if note['end'] is not None and start<=note['end']<end]})
        simultaneous=max((sum(note['start']<=time and (note['end'] is None or note['end']>time)
            for note in bank) for time in boundaries),default=0)
        pitches=sorted({note['pitch'] for note in incoming+held})
        span_mm=None
        if keyboard is not None and pitches and all(str(pitch) in keyboard for pitch in pitches):
            positions=[(keyboard[str(pitch)]['bounds_min'][0]+keyboard[str(pitch)]['bounds_max'][0])*.5 for pitch in pitches]
            span_mm=(max(positions)-min(positions))*1000
        result['hands'][hand or 'unassigned']=dict(note_on_count=len(incoming),
            onset_groups=len({note['start_tick'] for note in incoming}),
            note_ons_per_quarter=len(incoming)/horizon_quarters,
            note_ons_per_second=len(incoming)/(end-start),
            midi_held_at_start=len(held),maximum_simultaneous_midi_holds=simultaneous,
            pitch_span_semitones=(max(pitches)-min(pitches)) if pitches else 0,
            keyboard_center_span_mm=span_mm,incoming_ids=[note['id'] for note in incoming],
            unresolved_held_ids=[note['id'] for note in bank if note['end'] is None and note['start']<end])
    result['interpretation']='Musical demand and keyboard geometry only. Simultaneous MIDI holds, pedal sound and physical finger occupancy need separate motion planning.'
    return result
