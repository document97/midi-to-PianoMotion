"""The acoustic grand piano regions of an installed GM DLS bank (Windows gm.dls), for the offline preview.

No live MIDI/audio device is opened. Raw sample assets are not exported.
"""
import struct
import numpy as np

def chunks(data,start,end):
    result=[]
    while start+8<=end:
        tag=data[start:start+4];size=struct.unpack_from('<I',data,start+4)[0]
        result.append((tag,start+8,size,data[start+8:start+12] if tag==b'LIST' else b''));start+=8+size+size%2
    return result

def piano_regions(path):
    data=path.read_bytes();root=chunks(data,12,len(data));instruments=next(r for r in root if r[3]==b'lins');waves=next(r for r in root if r[3]==b'wvpl')
    wave_rows=chunks(data,waves[1]+4,waves[1]+waves[2]);regions=[]
    for item in chunks(data,instruments[1]+4,instruments[1]+instruments[2]):
        fields=chunks(data,item[1]+4,item[1]+item[2]);header=next(r for r in fields if r[0]==b'insh');count,bank,program=struct.unpack_from('<III',data,header[1])
        if bank!=0 or program!=0:continue
        group=next(r for r in fields if r[3]==b'lrgn')
        for region in chunks(data,group[1]+4,group[1]+group[2]):
            fields=chunks(data,region[1]+4,region[1]+region[2]);header=next(r for r in fields if r[0]==b'rgnh');low,high=struct.unpack_from('<HH',data,header[1]);sample=next(r for r in fields if r[0]==b'wsmp');size,unity,cents,attenuation,options,loop_count=struct.unpack_from('<IHhiII',data,sample[1])
            loop_start,loop_length=(struct.unpack_from('<II',data,sample[1]+size+8) if loop_count else (0,0))
            link=next(r for r in fields if r[0]==b'wlnk');index=struct.unpack_from('<I',data,link[1]+8)[0];item=wave_rows[index];fields=chunks(data,item[1]+4,item[1]+item[2]);fmt=next(r for r in fields if r[0]==b'fmt ');format,channels,rate,byte_rate,align,bits=struct.unpack_from('<HHIIHH',data,fmt[1]);raw=next(r for r in fields if r[0]==b'data')
            if format!=1 or bits!=16:raise ValueError('Expected installed PCM piano sample')
            sound=np.frombuffer(data[raw[1]:raw[1]+raw[2]],dtype='<i2').astype(float)/32768
            if channels>1:sound=sound.reshape(-1,channels).mean(axis=1)
            regions.append({'low':low,'high':high,'unity':unity,'cents':cents,'rate':rate,'loop_start':loop_start,'loop_length':loop_length,'sound':sound})
        break
    if not regions:raise ValueError('No acoustic grand piano in installed GM bank')
    return regions
