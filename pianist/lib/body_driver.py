"""Measured seating correction and slow torso assistance, in world space."""
import math
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector


def apply_seating(arm, cal):
    offset=Vector(cal.get('ergonomics',{}).get('world_root_offset',[0,0,0]))
    root=arm.parent or arm
    if 'performer_seating_offset' in root:
        if (Vector(root['performer_seating_offset'])-offset).length>1e-8:
            raise ValueError('Scene already has a different seating correction')
        return
    world=root.matrix_world.copy();world.translation+=offset;root.matrix_world=world
    seat=bpy.data.objects.get('bench_seat')
    if seat:
        world=seat.matrix_world.copy();world.translation+=offset;seat.matrix_world=world
    for name in ('bench_leg','bench_leg.001'):
        leg=bpy.data.objects.get(name)
        if leg is None or leg.type!='MESH':continue
        leg.data=leg.data.copy();world=leg.matrix_world.copy();inverse=world.inverted()
        points=[world@v.co for v in leg.data.vertices];low=min(p.z for p in points);high=max(p.z for p in points)
        if high-low<1e-6:raise ValueError('Bench support has no height')
        for vertex,point in zip(leg.data.vertices,points):
            point.x+=offset.x;point.y+=offset.y
            point.z=low+(point.z-low)*(1+offset.z/(high-low));vertex.co=inverse@point
    support=cal.get('ergonomics',{}).get('footrest')
    if support and bpy.data.objects.get('pedal_extender') is None:
        x0,y0,x1,y1=support['bounds_xy'];floor=support['floor_z'];top=support['top_z']
        bpy.ops.mesh.primitive_cube_add(size=1,location=((x0+x1)/2,(y0+y1)/2,(floor+top)/2))
        obj=bpy.context.object;obj.name='pedal_extender';obj.dimensions=(x1-x0,y1-y0,top-floor)
        material=bpy.data.materials.new('pedal_extender_dark');material.diffuse_color=(.035,.028,.023,1);obj.data.materials.append(material)
    root['performer_seating_offset']=list(offset);bpy.context.view_layer.update()


class BodyDriver:
    def __init__(self,arm,drivers,config):
        self.arm=arm;self.drivers=drivers
        self.bone=arm.pose.bones.get('上半身') if config.get('torso_assistance',False) else None
        self.reference={h:d.pre_upper_world.copy() for h,d in drivers.items()}
        self.timeline_key=None
        if self.bone:
            self.world=arm.matrix_world@self.bone.matrix;self.pivot=self.world.translation.copy()
            self.pre=arm.matrix_world@self.bone.parent.matrix@self.bone.parent.bone.matrix_local.inverted()@self.bone.bone.matrix_local
            self.bone.rotation_mode='QUATERNION'

    def pose(self,timing,t):
        if self.bone is None:return {}
        if self.timeline_key!=id(timing):
            rows=[]
            for n in timing.notes:
                states=timing.hand_states[n['hand']]
                state=next((s for s in states if abs(s['time']-n['start'])<1e-5),None)
                if state is None:continue
                delta=state['wrist'][0]-self.reference[n['hand']].translation.x
                need=math.copysign(max(0,abs(delta)-.15),delta)
                rows.append((n['start'],need,(n['velocity']/127)**.6))
            self.rows=np.asarray(timing.body_support if timing.body_support is not None else rows);self.timeline_key=id(timing)
            self.first=min(n['start'] for n in timing.notes);self.last=max(n['physical_release'] for n in timing.notes)
        def smooth(x):
            x=np.clip(x,0,1);return x*x*x*(10+x*(-15+6*x))
        rows=self.rows[np.abs(self.rows[:,0]-t)<.8]
        shift=0.
        if len(rows):
            weights=np.exp(-.5*((rows[:,0]-t)/.35)**2)*rows[:,2]*smooth((.8-np.abs(rows[:,0]-t))/.2)
            # A neutral weight prevents an isolated note from producing a
            # jump when its compact preparation window first enters view.
            shift=float(np.sum(rows[:,1]*weights)/(.25+np.sum(weights)))
        shift=np.clip(shift,-.11,.11)*smooth((t-self.first+.85)/.65)*(1-smooth((t-self.last)/.7))
        height=np.mean([m.translation.z for m in self.reference.values()])-self.pivot.z
        limit=math.radians(16);raw=math.asin(np.clip(shift/max(.05,height),-1,1));angle=limit*math.tanh(raw/limit)
        rotation=Quaternion(Vector((0,1,0)),float(angle)).to_matrix().to_4x4()
        transform=Matrix.Translation(self.pivot)@rotation@Matrix.Translation(-self.pivot)
        for h,d in self.drivers.items():d.pre_upper_world=transform@self.reference[h]
        return {self.bone.name:(self.pre.inverted()@transform@self.world).to_quaternion().normalized()}
