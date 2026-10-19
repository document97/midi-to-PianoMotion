"""Explicit signed hinge mapping for modular thumb joints.

An opt-in joint spec defines the model zero and the direction of positive
flexion. Without a spec a joint keeps the default mapping.
Neither Euler values nor these model zeros are clinical goniometry.
"""
import math


def child_euler(row, index, bend_degrees):
    specs = row.get('joint_articulation', {})
    spec = specs.get(row.get('child_joint_names',('MCP', 'IP'))[index])
    if spec is None:
        result = list(row['base_child_euler'][index])
        result[0] += row['sign'] * math.radians(bend_degrees)
        return result
    result = list(spec['zero_euler'])
    result[spec['axis']] += spec['flexion_sign'] * math.radians(bend_degrees)
    return result


def calibrate_joint_direction(arm, skin, row, index, pad_ids):
    """Calibrate an existing MCP/IP hinge against the model's own pulp face."""
    from mathutils import Vector
    bone = arm.data.bones[row['bone_names'][index+1]]
    transform = arm.matrix_world.inverted() @ skin.matrix_world
    normals = transform.to_3x3().inverted().transposed()
    pad_rest = sum((normals @ skin.data.vertices[i].normal for i in pad_ids), Vector()).normalized()
    inverse = bone.matrix_local.to_3x3().inverted()
    pad_local = (inverse @ pad_rest).normalized()
    proximal_local = (inverse @ bone.parent.matrix_local.to_3x3().col[1]).normalized()
    sign = 1 if pad_local.z > 0 else -1
    if abs(pad_local.z) < .8:
        raise ValueError('Annotated pad does not establish the transverse X hinge sign')
    zero = math.atan2(proximal_local.z, proximal_local.y)
    return {'axis': 0, 'flexion_sign': sign, 'zero_euler': [zero, 0, 0],
            'zero_kind': 'proximal/distal alignment projected into the existing X hinge plane',
            'positive_direction': 'toward annotated rounded model-pad face',
            'pad_landmark_vertices': list(pad_ids), 'pad_normal_in_bone_rest': list(pad_local),
            'rest_lateral_residual_degrees': math.degrees(math.asin(proximal_local.x)),
            'bounds_degrees': [0, 45 if index == 0 else 24], 'clinical_axis_calibration': False}


def signed_joint_geometry(arm, row, index):
    """Measure a signed posed hinge from bone directions, not Euler values."""
    from mathutils import Vector
    bone = arm.pose.bones[row['bone_names'][index+1]]
    parent = bone.parent
    pre = parent.matrix @ parent.bone.matrix_local.inverted() @ bone.bone.matrix_local
    world = (arm.matrix_world @ pre).to_3x3()
    spec = row['joint_articulation'][row.get('child_joint_names',('MCP', 'IP'))[index]]
    axis = world.col[0].normalized() * spec['flexion_sign']
    proximal = (arm.matrix_world @ parent.matrix).to_3x3().col[1].normalized()
    # Remove the small static out-of-plane error before measuring the hinge.
    forward = (proximal-axis*proximal.dot(axis)).normalized()
    inward = axis.cross(forward).normalized()
    distal = (arm.matrix_world.to_3x3() @ (bone.tail-bone.head)).normalized()
    return {'signed_flexion_degrees': math.degrees(math.atan2(distal.dot(inward), distal.dot(forward))),
            'side_component_degrees': math.degrees(math.asin(max(-1, min(1, distal.dot(axis))))),
            'inward_direction_world': list(inward)}
