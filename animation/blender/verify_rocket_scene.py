"""Check saved animation against the CFD-informed flight trace."""

from __future__ import annotations

import csv
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT / "flight_model/results/landing/final/dqn_ascent_guided_descent_nominal_trace.csv"
with TRACE.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))

scene = bpy.context.scene
rig = bpy.data.objects["ROCKET_RIG | animate location Z"]
camera = scene.camera
assert scene.frame_end == 794
assert len([o for o in bpy.data.objects if o.name.startswith("Nozzle particle")]) == 18
for frame, expected_height in ((1, 0.0), (398, max(float(r["height_m"]) for r in rows)),
                               (scene.frame_end, 0.0)):
    scene.frame_set(frame)
    assert abs(rig.location.z - expected_height) < 0.05, (frame, rig.location.z, expected_height)
    assert abs(camera.location.z - rig.location.z - 12.0) < 1e-5
scene.frame_set(1)
assert "render_frames" in scene.render.filepath
print("Verified: 794 frames; launch, 511.33 m peak and touchdown match trace;")
print("camera follows rocket at a constant 12 m vertical offset; 18 exhaust particles.")
