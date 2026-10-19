"""Remove animation curves from an object's layered action (Blender 4.4+ actions)."""


def remove_curves(obj,predicate):
    if not obj.animation_data or not obj.animation_data.action:return
    action=obj.animation_data.action
    for layer in action.layers:
        for strip in layer.strips:
            bag=strip.channelbag(obj.animation_data.action_slot)
            if bag:
                for curve in list(bag.fcurves):
                    if predicate(curve):bag.fcurves.remove(curve)
