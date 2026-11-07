"""Render half-resolution frames sampled at 10 fps from the animation."""

from pathlib import Path

import bpy


scene = bpy.context.scene
output = Path(__file__).resolve().parent / "draft_frames"
output.mkdir(exist_ok=True)
scene.render.resolution_percentage = 50
scene.render.fps = 10
scene.frame_step = 3
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(output / "frame_")
bpy.ops.render.render(animation=True)
print(f"DRAFT_FRAMES={output}")
