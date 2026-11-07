"""Check version-3 HUD and terrain after reopening the saved Blender file."""

import bpy


scene = bpy.context.scene
rig = bpy.data.objects["ROCKET_RIG | animate location Z"]
camera = scene.camera
assert len([obj for obj in bpy.data.objects if obj.name.startswith("Nozzle particle")]) == 18
assert len([obj for obj in bpy.data.objects if "mountain range" in obj.name.lower()]) == 2
for frame, expected_height, expected_digit, expected_engine in (
    (46, 13, "0", "ASCENT BURN"),
    (398, 511, "5", "COAST"),
    (794, 0, "0", "TOUCHDOWN"),
):
    scene.frame_set(frame)
    assert abs(camera.location.z - rig.location.z - 12.0) < 1e-5
    assert abs(rig.location.z - expected_height) < 1.0
    assert not bpy.data.objects[f"HUD altitude digit 0 value {expected_digit}"].hide_render
    assert not bpy.data.objects[f"HUD engine | {expected_engine}"].hide_render
print("Verified v03: mountain ranges, tracking camera and flight HUD at launch, peak and touchdown.")
