"""Shared contact footprint from the measured physical key geometry."""
GEOMETRY_SOURCE = 'live measured released key meshes at frame 1'


def key_depth_gain(obj):
    """World Z displacement per unit of local delta_location.z."""
    from mathutils import Matrix,Vector
    parent=(obj.parent.matrix_world@obj.matrix_parent_inverse if obj.parent else Matrix.Identity(4))
    axis=parent.to_3x3()@Vector((0,0,1))
    if abs(axis.x)>1e-7 or abs(axis.y)>1e-7 or axis.z<=0:
        raise ValueError('Key driver requires a vertical positive local Z axis: '+obj.name)
    return float(axis.z)


def contact_bounds(key, margin=.0005):
    return (key['bounds_min'][0]+margin, key['bounds_max'][0]-margin,
            key['bounds_min'][1]+margin, key['bounds_max'][1]-margin)


def measured_keyboard():
    """Capture authoritative key surfaces at frame 1 from the current scene."""
    import bpy
    scene=bpy.context.scene
    prior=(scene.frame_current,scene.frame_subframe)
    scene.frame_set(1)
    bpy.context.view_layer.update()
    result={}
    for row in key_solids():
        if abs(row['object'].delta_location.z)>1e-6:
            raise ValueError('Reference keys must be released at frame 1')
        low,high=row['low'],row['high']
        result[str(row['pitch'])]={'center_x':float((low[0]+high[0])/2),
            'top_z':float(high[2]),'bounds_min':[float(v) for v in low],
            'bounds_max':[float(v) for v in high],
            'local_z_world_gain':key_depth_gain(row['object'])}
    if len(result)!=88:
        raise ValueError('Expected 88 physical key meshes, got '+str(len(result)))
    scene.frame_set(prior[0],subframe=prior[1])
    bpy.context.view_layer.update()
    return result


def key_solids():
    """Build closed key BVHs in the currently evaluated reference pose."""
    import bpy
    import numpy as np
    from mathutils.bvhtree import BVHTree
    rows=[]
    for obj in bpy.data.objects:
        if not obj.name.startswith('key_') or obj.type!='MESH': continue
        points=[obj.matrix_world@v.co for v in obj.data.vertices]
        polygons=[list(p.vertices) for p in obj.data.polygons]
        edge_counts={}
        for polygon in polygons:
            for a,b in zip(polygon,polygon[1:]+polygon[:1]):
                edge=tuple(sorted((a,b)))
                edge_counts[edge]=edge_counts.get(edge,0)+1
        if any(count!=2 for count in edge_counts.values()):
            raise ValueError('Key mesh is not a closed manifold: '+obj.name)
        cloud=np.asarray(points,dtype=float)
        obj.data.calc_loop_triangles()
        faces=cloud[np.asarray([t.vertices[:] for t in obj.data.loop_triangles],dtype=int)]
        normals=np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0])
        normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
        roofs=faces[normals[:,2]>.5]
        rows.append({'pitch':int(obj.name[4:]),'object':obj,
                     'tree':BVHTree.FromPolygons(points,polygons),
                     'low':cloud.min(axis=0),'high':cloud.max(axis=0),
                     'origin':np.asarray(obj.matrix_world.translation,dtype=float),
                     'mesh_vertices':len(points),'mesh_polygons':len(polygons),
                     'roof_triangles':roofs})
    return rows


def inside_solid(tree,point):
    from mathutils import Vector
    direction=Vector((.371,.557,.743)).normalized()
    origin=point.copy()
    hits=0
    for _ in range(16):
        location,normal,index,distance=tree.ray_cast(origin,direction,2)
        if location is None: return hits%2==1
        hits+=1
        origin=location+direction*1e-6
    raise ValueError('Unexpected repeated ray hits')


def deepest_witness(points,rows,tolerance=0):
    import numpy as np
    from mathutils import Vector
    worst=None
    if not len(points) or not rows:return None
    positioned=[]
    for row in rows:
        offset=row.get('offset')
        if offset is None:offset=np.asarray(row['object'].matrix_world.translation)-row['origin']
        positioned.append((row,offset,row['low']+offset,row['high']+offset))
    global_low=np.min([low for row,offset,low,high in positioned],axis=0)
    global_high=np.max([high for row,offset,low,high in positioned],axis=0)
    selected=np.flatnonzero(np.all((points>global_low)&(points<global_high),axis=1))
    if not len(selected):return None
    subset=points[selected]
    for row,offset,low,high in positioned:
        indices=selected[np.flatnonzero(np.all((subset>low)&(subset<high),axis=1))]
        for index in indices:
            point=Vector(points[index]-offset)
            nearest,normal,face,distance=row['tree'].find_nearest(point)
            if distance is None or distance<=tolerance or not inside_solid(row['tree'],point): continue
            if worst is None or distance>worst['depth']:
                worst={'index':int(index),'pitch':row['pitch'],'depth':float(distance),
                       'point':[float(v) for v in points[index]]}
    return worst
