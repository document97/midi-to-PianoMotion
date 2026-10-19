"""Analytic shoulder/elbow/wrist transport for the existing avatar hierarchy.

The elbow follows a continuous preference on its reach circle. Skin twist
helpers remain part of the rig; the previous unconstrained four-link IK is
disabled. World joint positions are checked against unscaled segment lengths.
"""
import math
import numpy as np
import bpy
from mathutils import Matrix,Quaternion,Vector


def basis(axis,forward):
    y=Vector(forward).normalized();x=Vector(axis)-y*Vector(axis).dot(y)
    x.normalize();z=x.cross(y).normalized()
    return Matrix(((x.x,y.x,z.x),(x.y,y.y,z.y),(x.z,y.z,z.z)))


def closest_elbow(shoulder,wrist,l1,l2,side,reference=None):
    s=np.asarray(shoulder,dtype=float);w=np.asarray(wrist,dtype=float)
    d=w-s;distance=float(np.linalg.norm(d))
    if distance>=l1+l2-.0001 or distance<=abs(l1-l2)+.0001:
        raise ValueError(f'Arm target outside unstretched reach: {distance:.6f}')
    u=d/distance;a=(l1*l1-l2*l2+distance*distance)/(2*distance)
    radius=math.sqrt(max(0,l1*l1-a*a));center=s+a*u
    downward=np.array([0.,0.,-1.]);t=downward-u*np.dot(downward,u)
    if np.linalg.norm(t)<.01:t=np.array([side,0.,0.])-u*side*u[0]
    t/=np.linalg.norm(t);q=np.cross(u,t)
    # The forearm mainly points along the keys' depth direction. The elbow
    # preference also keeps it below the wrist and away from the torso.
    preferred=np.array([w[0],s[1]+.055,w[2]-.035])
    def cost(angle):
        point=center+radius*(t*math.cos(angle)+q*math.sin(angle))
        error=point-preferred
        value=8*error[0]**2+error[1]**2+2*error[2]**2
        value+=20*max(0,point[2]-w[2]-.025)**2
        if reference is not None:value+=.4*float(np.sum((point-np.asarray(reference))**2))
        return value,point
    grid=np.linspace(-math.pi,math.pi,81)
    candidates=center+radius*(np.cos(grid)[:,None]*t+np.sin(grid)[:,None]*q)
    errors=candidates-preferred
    costs=(errors*errors)@np.array([8.,1.,2.])+20*np.maximum(0,candidates[:,2]-w[2]-.025)**2
    if reference is not None:costs+=.4*np.sum((candidates-np.asarray(reference))**2,axis=1)
    angle=grid[int(np.argmin(costs))];step=math.pi/40
    for _ in range(12):
        options=[angle-step,angle,angle+step];angle=min(options,key=lambda theta:cost(theta)[0]);step*=.5
    return Vector(cost(angle)[1])


class ArmDriver:
    def __init__(self,arm,hand,palm_quaternion,palm_forward=None):
        self.arm=arm;self.hand=hand;self.side='左' if hand=='L' else '右';self.sign=-1 if hand=='L' else 1
        self.upper=arm.pose.bones[self.side+'腕'];self.elbow=arm.pose.bones[self.side+'ひじ'];self.wrist=arm.pose.bones[self.side+'手首']
        self.twists=[arm.pose.bones[self.side+name] for name in ('腕捩','手捩')]
        self.palm=Quaternion(palm_quaternion).normalized()
        self.palm_forward=Vector(palm_forward if palm_forward is not None else self.palm@Vector((0,1,0))).normalized()
        for bone in self.twists:
            for c in bone.constraints:
                if c.type=='IK':c.mute=True
        for bone in [self.upper,self.elbow]+self.twists:
            bone.rotation_mode='QUATERNION';bone.rotation_quaternion=(1,0,0,0)
            bone.location=(0,0,0);bone.scale=(1,1,1)
        bpy.context.view_layer.update()
        world=arm.matrix_world
        s=world@self.upper.head;e=world@self.elbow.head;w=world@self.wrist.head
        self.l1=(e-s).length;self.l2=(w-e).length
        self.rest_upper_q=(world@self.upper.matrix).to_quaternion()
        self.rest_elbow_q=(world@self.elbow.matrix).to_quaternion()
        self.hinge=(world@self.elbow.matrix).to_3x3().col[0].normalized()
        # setup() applies the calibrated neutral wrist before any planning.
        # Choose the bend-axis polarity once there. Rechoosing its hemisphere
        # for each pose creates a half turn when the reference dot crosses zero.
        self.hinge_polarity=None
        self.rest_upper_basis=basis(self.hinge,e-s)
        self.rest_elbow_basis=basis(self.hinge,w-e)
        self.scales={b.name:(world@b.matrix).decompose()[2] for b in (self.upper,self.elbow,self.wrist)}
        self.pre_upper_world=world@self.upper.parent.matrix@self.upper.parent.bone.matrix_local.inverted()@self.upper.bone.matrix_local
        self.upper_elbow_link=self.upper.bone.matrix_local.inverted()@self.elbow.bone.matrix_local
        self.elbow_wrist_link=self.elbow.bone.matrix_local.inverted()@self.wrist.bone.matrix_local
        self.forearm_twist=self.twists[-1]
        self.elbow_twist_link=self.elbow.bone.matrix_local.inverted()@self.forearm_twist.bone.matrix_local
        self.twist_wrist_link=self.forearm_twist.bone.matrix_local.inverted()@self.wrist.bone.matrix_local

    def hinge_axis(self,upper_dir,forearm_dir):
        hinge=upper_dir.cross(forearm_dir)
        if hinge.length<1e-8:raise ValueError('Arm bending plane is singular')
        hinge.normalize()
        if self.hinge_polarity is None:self.hinge_polarity=1 if hinge.dot(self.hinge)>=0 else -1
        return hinge*self.hinge_polarity

    def local_pose(self,position,palm_yaw=0.,forearm_roll=0.,world_matrices=False):
        """Same analytic geometry in a static body frame, without graph updates."""
        shoulder=self.pre_upper_world.translation;wrist=Vector(position)
        elbow=closest_elbow(shoulder,wrist,self.l1,self.l2,self.sign)
        upper_dir=(elbow-shoulder).normalized();forearm_dir=(wrist-elbow).normalized()
        hinge=self.hinge_axis(upper_dir,forearm_dir)
        upper_q=(basis(hinge,upper_dir)@self.rest_upper_basis.transposed()).to_quaternion()@self.rest_upper_q
        elbow_q=(basis(hinge,forearm_dir)@self.rest_elbow_basis.transposed()).to_quaternion()@self.rest_elbow_q
        upper=Matrix.LocRotScale(shoulder,upper_q,self.scales[self.upper.name])
        forearm=Matrix.LocRotScale(elbow,elbow_q,self.scales[self.elbow.name])
        twist_q=Quaternion(Vector((0,1,0)),math.radians(forearm_roll))
        palm_q=Quaternion(Vector((0,0,1)),math.radians(palm_yaw))@Quaternion(self.palm_forward,math.radians(forearm_roll))@self.palm
        palm=Matrix.LocRotScale(wrist,palm_q,self.scales[self.wrist.name])
        wrist_pre=forearm@self.elbow_twist_link@twist_q.to_matrix().to_4x4()@self.twist_wrist_link
        if world_matrices:
            return {self.upper.name:upper,self.elbow.name:forearm,self.wrist.name:palm,
                    self.forearm_twist.name:forearm@self.elbow_twist_link@twist_q.to_matrix().to_4x4()}
        matrices=(self.pre_upper_world.inverted()@upper,(upper@self.upper_elbow_link).inverted()@forearm,wrist_pre.inverted()@palm)
        result={b.name:m.to_quaternion().normalized() for b,m in zip((self.upper,self.elbow,self.wrist),matrices)}
        result[self.forearm_twist.name]=twist_q;return result

    def _matrix(self,bone,position,rotation):
        world=Matrix.LocRotScale(position,rotation,self.scales[bone.name])
        bone.matrix=self.arm.matrix_world.inverted()@world

    def apply(self,position,reference_elbow=None):
        # Parent body pose may change. Its root point remains authoritative.
        parent=self.upper.parent
        pre=parent.matrix@parent.bone.matrix_local.inverted()@self.upper.bone.matrix_local
        shoulder=self.arm.matrix_world@pre.translation
        wrist=Vector(position)
        elbow=closest_elbow(shoulder,wrist,self.l1,self.l2,self.sign,reference_elbow)
        upper_dir=(elbow-shoulder).normalized();forearm_dir=(wrist-elbow).normalized()
        hinge=self.hinge_axis(upper_dir,forearm_dir)
        upper_q=(basis(hinge,upper_dir)@self.rest_upper_basis.transposed()).to_quaternion()@self.rest_upper_q
        elbow_q=(basis(hinge,forearm_dir)@self.rest_elbow_basis.transposed()).to_quaternion()@self.rest_elbow_q
        self._matrix(self.upper,shoulder,upper_q)
        bpy.context.view_layer.update()
        self._matrix(self.elbow,elbow,elbow_q)
        bpy.context.view_layer.update()
        self._matrix(self.wrist,wrist,self.palm)
        bpy.context.view_layer.update()
        actual=self.arm.matrix_world@self.wrist.head
        local_translation={b.name:float(b.location.length)*float(self.arm.matrix_world.to_scale().x)*1000 for b in (self.upper,self.elbow,self.wrist)}
        return {'shoulder':list(shoulder),'elbow':list(elbow),'wrist':list(actual),
                'target_error_mm':(actual-wrist).length*1000,'local_translation_mm':local_translation,
                'segment_error_mm':max(abs((elbow-shoulder).length-self.l1),abs((wrist-elbow).length-self.l2))*1000}
