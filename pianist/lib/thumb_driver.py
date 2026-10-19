"""Calibrated thumb articulation and contact geometry for the character's rig.

CMC uses a minimum-twist swing of the first metacarpal. Its three rotation
components are coupled by that transport, not independent Euler actuators.
MCP and IP are independent bounded inputs. Bounds are avatar working bounds,
not medical ROM. The solver never moves the wrist or another finger.
"""
import math

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from key_geometry import contact_bounds, deepest_witness
from thumb_articulation import child_euler

_REST_SKIN_CACHE={}


def animation_curves(obj):
    if not obj.animation_data or not obj.animation_data.action:
        return []
    action = obj.animation_data.action
    result = []
    for layer in action.layers:
        for strip in layer.strips:
            bag = strip.channelbag(obj.animation_data.action_slot)
            if bag:
                result.extend(bag.fcurves)
    return result


def set_time(scene, time):
    frame = 1 + time * scene.render.fps / scene.render.fps_base
    integer = math.floor(frame)
    scene.frame_set(integer, subframe=frame-integer)


def mesh_world(skin):
    ev = skin.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = ev.to_mesh()
    try:
        coords = np.empty((len(mesh.vertices), 3), dtype=np.float64)
        normals = np.empty_like(coords)
        mesh.vertices.foreach_get('co', coords.ravel())
        mesh.vertices.foreach_get('normal', normals.ravel())
        matrix = np.asarray(ev.matrix_world, dtype=np.float64)
        coords = coords @ matrix[:3, :3].T + matrix[:3, 3]
        normals = normals @ np.linalg.inv(matrix[:3, :3])
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
        return coords, normals
    finally:
        ev.to_mesh_clear()


def calibrate(arm, skin, workspace):
    """Fix contact membership in the neutral pose, never at each strike."""
    for mod in skin.modifiers:
        if mod.type == 'ARMATURE' and mod.use_deform_preserve_volume:
            raise ValueError('Thumb prediction requires linear armature skinning')
    coords, normals = mesh_world(skin)
    skin.data.calc_loop_triangles()
    result = {'schema': 'thumb-calibration', 'hands': {},
              'contact_normal_min_down_dot': .25,
              'neutral_contact_min_down_dot': .20,
              'working_bounds_degrees': {'swing_lateral': [-12, 12],
                 'swing_lift': [-24, 24], 'mcp_delta': [0, 45],
                 'ip_delta': [0, 24]},
              'limits_kind': 'avatar working bounds; not anatomical ROM'}
    for hand, side, sign in [('L', '左', 1), ('R', '右', -1)]:
        names = [side+'親指'+suffix for suffix in ('０', '１', '２')]
        terminal_group = skin.vertex_groups[names[-1]].index
        chain_groups = {skin.vertex_groups[name].index for name in names}
        tip = arm.matrix_world @ arm.pose.bones[names[-1]].tail
        terminal = [v.index for v in skin.data.vertices if any(
            g.group == terminal_group and g.weight > .5 for g in v.groups)]
        # The neutral downward-facing distal soft surface is the lateral
        # contact patch. Fix those vertex IDs before any pose optimization.
        patch = [i for i in terminal if np.linalg.norm(coords[i]-tip) < .014
                 and normals[i, 2] < -result['neutral_contact_min_down_dot']]
        if len(patch) < 5:
            raise ValueError(f'{hand}: insufficient neutral lateral contact patch')
        patch_lookup = {v:i for i,v in enumerate(patch)}
        triangles = [[patch_lookup[v] for v in t.vertices]
                     for t in skin.data.loop_triangles
                     if all(v in patch_lookup for v in t.vertices)]
        if not triangles:
            raise ValueError(f'{hand}: no connected contact faces')
        affected = [v.index for v in skin.data.vertices if any(
            g.group in chain_groups and g.weight > .001 for g in v.groups)]
        other_groups = {skin.vertex_groups[side+'人指'+s].index
                        for s in ('１', '２', '３')}
        index_surface = [v.index for v in skin.data.vertices if
            sum(g.weight for g in v.groups if g.group in other_groups) > .8]
        result['hands'][hand] = {
            'bone_names': names, 'sign': sign,
            'working_bounds_degrees': result['working_bounds_degrees'],
            'base_child_euler': workspace[hand]['fingers']['1']['curl_base_eulers'],
            'contact_patch': patch, 'contact_mode': 'neutral_lateral_distal_surface',
            'contact_triangles_in_patch': triangles,
            'affected_vertices': affected, 'terminal_vertices': terminal,
            'index_surface_vertices': index_surface,
            'bone_rest_axes': {name: [list(arm.data.bones[name].matrix_local.to_3x3().col[i])
                                    for i in range(3)] for name in names}}
    return result


class ThumbFrame:
    """Exact linear-skin delta prediction in the current wrist frame."""
    def __init__(self, arm, skin, row, coords, normals):
        self.arm, self.skin, self.row = arm, skin, row
        self.names = row['bone_names']
        self.bones = [arm.pose.bones[name] for name in self.names]
        self.indices = np.asarray(row['affected_vertices'], dtype=np.int32)
        self.coords = coords[self.indices].copy()
        self.normals = normals.copy()
        lookup = {int(v): i for i, v in enumerate(self.indices)}
        self.patch = np.asarray([lookup[i] for i in row['contact_patch']])
        self.patch_triangles = row.get('contact_triangles_in_patch')
        self.terminal = np.asarray([lookup[i] for i in row['terminal_vertices']])
        self.index_points = coords[row['index_surface_vertices']]
        self.world = arm.matrix_world.copy()
        self.old = [b.matrix.copy() for b in self.bones]
        root = self.bones[0]
        self.pre = root.parent.matrix @ root.parent.bone.matrix_local.inverted() @ root.bone.matrix_local
        self.links = [self.bones[i-1].bone.matrix_local.inverted() @ self.bones[i].bone.matrix_local
                      for i in (1, 2)]
        self.q0 = root.rotation_euler.to_quaternion() if root.rotation_mode != 'QUATERNION' else root.rotation_quaternion.copy()
        self.root_basis = root.matrix_basis.copy()
        self.child_basis = [b.matrix_basis.copy() for b in self.bones[1:]]
        self.base_child = row['base_child_euler']
        self.root_world_q = (self.world @ self.old[0]).to_quaternion()
        self.base_axis = self.root_world_q @ Vector((0, 1, 0))
        side = '左' if row['sign'] == 1 else '右'
        across = arm.pose.bones[side+'小指１'].head-arm.pose.bones[side+'人指１'].head
        forward = arm.pose.bones[side+'中指１'].head-arm.pose.bones[side+'手首'].head
        normal = (self.world.to_3x3() @ across.cross(forward)).normalized()
        if normal.z < 0:
            normal = -normal
        self.palm_normal = normal
        self.lift_axis = self.base_axis.cross(normal).normalized()
        self.rest_points, self.weights = [], []
        transform = arm.matrix_world.inverted() @ skin.matrix_world
        cache_key=(arm.as_pointer(),skin.as_pointer(),tuple(self.names),tuple(self.indices),
                   tuple(v for r in transform for v in r))
        cached=_REST_SKIN_CACHE.get(cache_key)
        if cached:
            self.rest_points,self.weights=cached
            return
        raw = np.asarray([list(transform @ skin.data.vertices[int(i)].co)+[1.0]
                          for i in self.indices], dtype=np.float64)
        for b in self.bones:
            group = skin.vertex_groups[b.name].index
            weights = np.asarray([next((g.weight for g in skin.data.vertices[int(i)].groups
                                        if g.group == group), 0.0) for i in self.indices])
            self.weights.append(weights)
            self.rest_points.append(raw @ np.asarray(b.bone.matrix_local.inverted()).T)
        _REST_SKIN_CACHE[cache_key]=(self.rest_points,self.weights)

    def set_root_reference(self, local_quaternion):
        """Use an explicit wrist-relative reference without inheriting root spikes.

        The evaluated matrices stay intact for skin delta prediction;
        only the swing reference changes.
        """
        self.root_world_q = (self.world @ self.pre).to_quaternion() @ Quaternion(local_quaternion).normalized()
        self.base_axis = self.root_world_q @ Vector((0, 1, 0))
        self.lift_axis = self.base_axis.cross(self.palm_normal).normalized()

    def matrices(self, values):
        lateral, lift, mcp, ip = values
        axis = Quaternion(self.palm_normal, math.radians(lateral)) @ (
            Quaternion(self.lift_axis, math.radians(lift)) @ self.base_axis)
        swing = self.base_axis.rotation_difference(axis)
        world_q = swing @ self.root_world_q
        local_q = (self.world @ self.pre).to_quaternion().inverted() @ world_q
        loc, _, scale = self.root_basis.decompose()
        root = self.pre @ Matrix.LocRotScale(loc, local_q, scale)
        matrices = [root]
        for index, bend in enumerate((mcp, ip)):
            euler = child_euler(self.row, index, bend)
            from mathutils import Euler
            child_q = Euler(euler, 'XYZ').to_quaternion()
            loc, _, scale = self.child_basis[index].decompose()
            matrices.append(matrices[-1] @ self.links[index] @ Matrix.LocRotScale(loc, child_q, scale))
        return local_q, matrices

    def predict(self, values):
        q, matrices = self.matrices(values)
        points = self.coords.copy()
        world = np.asarray(self.world)
        for new, old, rest, weights in zip(matrices, self.old, self.rest_points, self.weights):
            delta = world @ (np.asarray(new)-np.asarray(old))
            points += (rest @ delta.T)[:, :3]*weights[:, None]
        transform = (self.world @ matrices[-1]).to_3x3() @ (self.world @ self.old[-1]).to_3x3().inverted()
        normals = self.normals[self.row['contact_patch']] @ np.asarray(transform).T
        return points, normals, q

    def apply(self, values):
        q, _ = self.matrices(values)
        root = self.bones[0]
        root.rotation_mode = 'QUATERNION'
        root.rotation_quaternion = q
        for index, bend in enumerate(values[2:]):
            euler = child_euler(self.row, index, bend)
            self.bones[index+1].rotation_mode = 'XYZ'
            self.bones[index+1].rotation_euler = euler
        bpy.context.view_layer.update()
        return q


def contact(points, patch, normals, key, depth, triangles=None):
    lo, hi, front, back = contact_bounds(key)
    cloud = points[patch]
    if triangles:
        # Clip actual fixed skin faces to the key footprint. A vertex crossing
        # its edge must not make the contact height jump to another vertex.
        candidates = []
        samples = np.column_stack((cloud, -normals[:,2]))
        planes = [(0,lo,1),(0,hi,-1),(1,front,1),(1,back,-1),(3,.25,1)]
        for triangle in triangles:
            polygon = [samples[i] for i in triangle]
            for axis,bound,sign in planes:
                distances = [sign*(p[axis]-bound) for p in polygon]
                if all(d>=0 for d in distances):
                    continue
                if all(d<0 for d in distances):
                    polygon=[]
                    break
                clipped=[]
                for i,a in enumerate(polygon):
                    b=polygon[(i+1)%len(polygon)]
                    da,db=distances[i],distances[(i+1)%len(polygon)]
                    if da>=0: clipped.append(a)
                    if (da>=0)!=(db>=0):
                        clipped.append(a+(b-a)*da/(da-db))
                polygon=clipped
            candidates.extend(polygon)
        if not candidates:
            return None
        selected = min(candidates,key=lambda p:p[2])
        return float(selected[2]-(key['top_z']-depth)),selected[:3],float(selected[3])
    valid = (cloud[:, 0] >= lo) & (cloud[:, 0] <= hi) & (cloud[:, 1] >= front) & (cloud[:, 1] <= back) & (normals[:, 2] <= -.25)
    if not np.any(valid):
        return None
    subset = np.flatnonzero(valid)
    i = subset[np.argmin(cloud[valid, 2])]
    point = cloud[i]
    return float(point[2]-(key['top_z']-depth)), point, float(-normals[i, 2])


def clearance(points, keys):
    witness=deepest_witness(points,keys)
    minimum=-witness['depth'] if witness else 0.0
    return minimum


def surface_distance(thumb, index):
    # Dense surface samples, not a claim of exact triangle self-intersection.
    if not len(thumb) or not len(index):
        return float('inf')
    return float(np.sqrt(np.min(np.sum((thumb[:, None]-index[None, :])**2, axis=2))))


def track_height(frame, values, key, desired_z, on_key, material_anchor=None):
    """Compensate CMC height as the authored wrist and child baseline move."""
    values = list(values)
    working=frame.row.get('working_bounds_degrees',{'swing_lateral':[-12,12],'swing_lift':[-24,24]})
    lateral_bounds=working['swing_lateral'];lift_bounds=working['swing_lift']
    def height(trial):
        points, normals, q = frame.predict(trial)
        minimum = float(np.min(points[frame.patch, 2]))
        if material_anchor is not None:
            z = float(np.asarray(material_anchor) @ points[frame.patch,2])
        elif on_key:
            hit = contact(points, frame.patch, normals, key, 0, frame.patch_triangles)
            if hit is None:
                return None
            z = hit[1][2]
        else:
            z = minimum
        return z, points, normals, q
    first = height(values)
    if first is None:
        candidates = []
        for offset in (0, -1, 1, -2, 2, -4, 4, -6, 6, -8, 8):
            trial = list(values)
            trial[0] = max(lateral_bounds[0], min(lateral_bounds[1], trial[0]+offset))
            measured = height(trial)
            if measured:
                candidates.append((abs(offset), trial, measured))
        if not candidates:
            return None
        _, values, first = min(candidates, key=lambda r:r[0])
    measured = first
    for _ in range(6):
        if abs(measured[0]-desired_z) <= .000025:
            break
        trial = list(values)
        trial[1] += .2 if values[1] < lift_bounds[1]-.2 else -.2
        shifted = height(trial)
        if shifted is None:
            break
        slope = (shifted[0]-measured[0])/(trial[1]-values[1])
        if abs(slope) < 1e-7:
            break
        next_values = list(values)
        next_values[1] = max(lift_bounds[0], min(lift_bounds[1], values[1]-(measured[0]-desired_z)/slope))
        next_measure = height(next_values)
        if next_measure is None:
            break
        values = next_values
        measured = next_measure
    return values, measured, abs(measured[0]-desired_z)


def fit(frame, key, depth, keys, preferred, iterations=6):
    working=frame.row.get('working_bounds_degrees',{'swing_lateral':[-12,12],'swing_lift':[-24,24],'mcp_delta':[0,45],'ip_delta':[0,24]})
    bounds = [working[name] for name in ('swing_lateral','swing_lift','mcp_delta','ip_delta')]
    baseline_distance = surface_distance(frame.coords[frame.terminal], frame.index_points)
    required_distance = min(.0015, baseline_distance-.0001)
    best_feasible=None
    def evaluate(values):
        nonlocal best_feasible
        values = list(values)
        # Solve height inside each articulation trial. Otherwise coordinate
        # descent rewards straight joints because changing a hinge alone
        # temporarily loses contact, even when a compensating CMC swing exists.
        tracked=track_height(frame,values,key,key['top_z']-depth,True)
        if tracked is None: return 1e8,None,values
        values=tracked[0]
        points, normals, q = frame.predict(values)
        hit = contact(points, frame.patch, normals, key, depth, frame.patch_triangles)
        if hit is None:
            return 1e8, None, values
        gap, point, normal_down = hit
        clear = clearance(points, keys)
        separation = surface_distance(points[frame.terminal], frame.index_points)
        error = gap*1000
        score = 100*max(0, abs(error)-.12)**2+.02*error**2
        score += .035*(values[2]-preferred[0])**2+.055*(values[3]-preferred[1])**2
        score += .04*values[0]**2+.02*values[1]**2
        score += .4*max(0, .45-normal_down)**2
        score += 1500*max(0, -1.3-clear*1000)**2
        score += 1500*max(0, (required_distance-separation)*1000)**2
        metrics={'gap_mm': error, 'clearance_mm': clear*1000,
                       'index_surface_gap_mm': separation*1000, 'normal_down_dot': normal_down,
                       'point': list(point), 'root_quaternion': list(q)}
        if abs(error)<=.6 and clear>=-.0015 and separation>=required_distance:
            if best_feasible is None or score<best_feasible[0]:
                best_feasible=(score,list(values),metrics)
        return score,metrics,values
    starts = [(0, 0, *preferred), (0, 0, 4, 2), (0, 0, 0, 0)]
    # Side contact with a neighbouring black key produces separate feasible
    # basins. Local descent from almost straight joints cannot cross that
    # barrier. Rank a bounded coarse grid before refining the best shapes.
    coarse=[]
    for lateral in (-12,-8,-4,0,4,8,12):
        for mcp in (0,8,16,24,30,36,45):
            for ip in (0,4,8,12,18,24):
                score,metrics,values=evaluate([lateral,0,mcp,ip])
                if metrics is not None:
                    coarse.append((score,values))
    starts.extend(values for _,values in sorted(coarse,key=lambda r:r[0])[:4])
    best = None
    for initial in starts:
        values = list(initial)
        score, metrics, values = evaluate(values)
        for step in (5, 2, 1, .5, .25, .1)[:iterations]:
            for _ in range(4):
                improved = False
                for axis in (0, 2, 3):
                    for direction in (-1, 1):
                        trial = list(values)
                        trial[axis] = max(bounds[axis][0], min(bounds[axis][1], trial[axis]+direction*step))
                        trial_score, trial_metrics, trial = evaluate(trial)
                        if trial_score < score-1e-8:
                            values, score, metrics = trial, trial_score, trial_metrics
                            improved = True
                if not improved:
                    break
        feasible = metrics is not None and abs(metrics['gap_mm']) <= .6 and metrics['clearance_mm'] >= -1.5 and metrics['index_surface_gap_mm'] >= required_distance*1000
        if feasible and (best is None or score < best[0]):
            best = (score, values, metrics)
    return best_feasible
