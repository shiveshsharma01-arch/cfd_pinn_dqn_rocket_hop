"""Check the saved failed-flight scene after reopening it."""
import math
import bpy

scene = bpy.context.scene
rig = bpy.data.objects["ROCKET_RIG | animate location Z"]
camera = scene.camera
assert scene.frame_start == 1 and scene.frame_end == 840
assert scene.render.fps == 30
assert scene["trace_file"].endswith("step000000_trace.csv")
assert len([o for o in bpy.data.objects if o.name.startswith("Impact dust puff")]) == 18
scene.frame_set(1)
assert abs(rig.location.z) < 1e-4
peak_marker = next(m for m in scene.timeline_markers if m.name.startswith("Peak |"))
scene.frame_set(peak_marker.frame)
assert abs(rig.location.z - 621.204) < 0.5
assert abs(camera.location.z - rig.location.z - 12) < 1e-4
scene.frame_set(784)
assert abs(rig.location.z) < 1e-4
assert not bpy.data.objects["HUD engine | IMPACT"].hide_render
scene.frame_set(839)
assert math.degrees(rig.rotation_euler.y) > 80
print("Verified early-DQN scene: trace source, flight, impact HUD and tip-over.")
