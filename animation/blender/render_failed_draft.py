"""Render a 10 fps review sequence from the early DQN failure scene."""
from pathlib import Path
import bpy

scene = bpy.context.scene
folder = Path(__file__).resolve().parent / "failed_frames"
folder.mkdir(exist_ok=True)
scene.render.resolution_percentage = 50
scene.render.fps = 10
scene.frame_step = 3
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(folder / "frame_")
bpy.ops.render.render(animation=True)
print(f"FAILED_FRAMES={folder}")
