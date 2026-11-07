"""Animate the rocket scene from the frozen 500 m hybrid-controller trace.

Run with Blender 5.2:
  blender --background rocket_pad_v01.blend --python animate_rocket_pad.py

All flight positions come from the exported simulator CSV. Exhaust geometry is
an illustrative visual effect, not a combustion prediction from the 300 K air
CFD nozzle model. Camera and lighting are deliberately neutral first choices.
"""

from __future__ import annotations

import bisect
import csv
import math
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
TRACE_FILE = ROOT / "flight_model/results/landing/final/dqn_ascent_guided_descent_nominal_trace.csv"
OUTPUT_BLEND = OUTPUT_DIR / "rocket_hop_v02.blend"
PREVIEW_LAUNCH = OUTPUT_DIR / "rocket_hop_launch_preview.png"
PREVIEW_ASCENT = OUTPUT_DIR / "rocket_hop_ascent_preview.png"
FRAME_DIR = OUTPUT_DIR / "render_frames"
FPS = 30


def load_trace():
    with TRACE_FILE.open(newline="", encoding="utf-8") as stream:
        rows = []
        for raw in csv.DictReader(stream):
            rows.append({
                "time": float(raw["time_s"]),
                "height": float(raw["height_m"]),
                "speed": float(raw["vertical_speed_m_s"]),
                "fuel": float(raw["fuel_remaining_kg"]),
                "spool": float(raw["actuator_spool"]),
                "command": float(raw["command_kpa"] or 0.0),
            })
    if len(rows) < 3 or abs(rows[0]["time"]) > 1e-9:
        raise ValueError("Trace must begin at time zero and include multiple samples")
    if any(b["time"] <= a["time"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Trace times are not strictly increasing")
    if abs(rows[-1]["height"]) > 1e-6 or min(r["height"] for r in rows) < 0:
        raise ValueError("Trace must end at ground contact with nonnegative height")
    return rows, [r["time"] for r in rows]


def interpolate(rows, times, time_s):
    if time_s >= times[-1]:
        return rows[-1]
    index = max(0, bisect.bisect_right(times, time_s) - 1)
    a, b = rows[index], rows[index + 1]
    fraction = (time_s - a["time"]) / (b["time"] - a["time"])
    return {key: a[key] + fraction * (b[key] - a[key])
            for key in ("time", "height", "speed", "fuel", "spool")}


def emission_material(name, color, strength, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, alpha)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Emission Color"].default_value = (*color, 1.0)
    shader.inputs["Emission Strength"].default_value = strength
    shader.inputs["Roughness"].default_value = 1.0
    shader.inputs["Alpha"].default_value = alpha
    if alpha < 1.0:
        mat.surface_render_method = "DITHERED"
    return mat


def add_plume_cone(name, rig, collection, radius_bottom, radius_top,
                   depth, center_z, material):
    bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=radius_bottom,
                                    radius2=radius_top, depth=depth,
                                    location=(0, 0, center_z))
    obj = bpy.context.object
    obj.name = name
    for old in list(obj.users_collection):
        old.objects.unlink(obj)
    collection.objects.link(obj)
    obj.parent = rig
    obj.data.materials.append(material)
    for face in obj.data.polygons:
        face.use_smooth = True
    return obj


def make_particle(name, rig, collection, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=4, radius=1.0,
                                         location=(0, 0, 0))
    obj = bpy.context.object
    obj.name = name
    for old in list(obj.users_collection):
        old.objects.unlink(obj)
    collection.objects.link(obj)
    obj.parent = rig
    obj.data.materials.append(material)
    for face in obj.data.polygons:
        face.use_smooth = True
    return obj


def add_target(rig, collection):
    target = bpy.data.objects.new("Camera target | rocket centre", None)
    collection.objects.link(target)
    target.parent = rig
    target.location = (0, 0, 8.5)
    target.empty_display_size = 0.5
    return target


def main():
    rows, times = load_trace()
    scene = bpy.context.scene
    rig = bpy.data.objects.get("ROCKET_RIG | animate location Z")
    camera = scene.camera
    if rig is None or camera is None:
        raise RuntimeError("Open rocket_pad_v01.blend before running this script")
    fx = bpy.data.collections.new("05  Exhaust visualisation | not CFD combustion")
    scene.collection.children.link(fx)
    target = add_target(rig, fx)
    camera.name = "Vertical tracking camera | provisional framing"
    camera.data.type = "PERSP"
    camera.data.lens = 52
    camera.constraints.clear()
    track = camera.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    # A neutral, physically plausible clear-sky setup. Final art direction can
    # change this without changing flight data or rocket geometry.
    world = scene.world
    world.name = "Daylight sky | neutral draft"
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    background = nodes.get("Background")
    for link in list(background.inputs["Color"].links):
        links.remove(link)
    sky = nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_elevation = math.radians(42)
    sky.sun_rotation = math.radians(115)
    sky.air_density = 1.0
    links.new(sky.outputs["Color"], background.inputs["Color"])
    background.inputs["Strength"].default_value = 0.24
    sun = bpy.data.objects.get("Temporary daylight — replace later")
    if sun:
        sun.name = "Daylight sun | neutral draft"
        sun.data.energy = 1.6
        sun.data.angle = math.radians(3)
        sun.rotation_euler = (math.radians(48), math.radians(-18),
                              math.radians(-35))
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = -0.8
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    FRAME_DIR.mkdir(exist_ok=True)
    scene.render.filepath = str(FRAME_DIR / "rocket_hop_")

    outer_mat = emission_material("Exhaust | pale blue", (0.15, 0.40, 0.95),
                                  1.8, 0.48)
    core_mat = emission_material("Exhaust | warm core", (1.0, 0.48, 0.16),
                                 2.0, 0.82)
    particle_mat = emission_material("Exhaust | warm particles", (1.0, 0.58, 0.22),
                                     1.4)
    outer = add_plume_cone("Exhaust envelope | actuator-linked", rig, fx,
                           0.75, 0.20, 4.4, -1.7, outer_mat)
    core = add_plume_cone("Exhaust core | actuator-linked", rig, fx,
                          0.28, 0.12, 2.7, -0.85, core_mat)
    particles = [make_particle(f"Nozzle particle {i + 1:02d}", rig, fx,
                               particle_mat) for i in range(18)]
    nozzle_light_data = bpy.data.lights.new("Engine reflected light", "POINT")
    nozzle_light = bpy.data.objects.new("Engine reflected light", nozzle_light_data)
    fx.objects.link(nozzle_light)
    nozzle_light.parent = rig
    nozzle_light.location = (0, 0, 0.3)
    nozzle_light_data.color = (1.0, 0.70, 0.40)
    nozzle_light_data.shadow_soft_size = 2.5

    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = math.ceil(times[-1] * FPS) + 1
    apex = max(rows, key=lambda item: item["height"])
    apex_frame = 1 + round(apex["time"] * FPS)
    scene.timeline_markers.new("Launch", frame=1)
    scene.timeline_markers.new("Peak height", frame=apex_frame)
    scene.timeline_markers.new("Touchdown", frame=scene.frame_end)
    scene["trace_file"] = TRACE_FILE.relative_to(ROOT).as_posix()
    scene["flight_source"] = "Frozen DQN ascent + guided descent simulator trace"
    scene["exhaust_note"] = "Illustrative visual effect; CFD uses 300 K air and has no combustion"

    for frame in range(scene.frame_start, scene.frame_end + 1):
        time_s = min((frame - 1) / FPS, times[-1])
        state = interpolate(rows, times, time_s)
        height = state["height"]
        opening = max(0.0, min(1.0, state["spool"]))
        rig.location = (0, 0, height)
        rig.keyframe_insert(data_path="location", frame=frame)
        camera.location = (25, -39, 12 + height)
        camera.keyframe_insert(data_path="location", frame=frame)
        outer.scale = (opening, opening, opening)
        core.scale = (opening, opening, opening)
        outer.keyframe_insert(data_path="scale", frame=frame)
        core.keyframe_insert(data_path="scale", frame=frame)
        nozzle_light_data.energy = 160 * opening
        nozzle_light_data.keyframe_insert(data_path="energy", frame=frame)
        for i, particle in enumerate(particles):
            phase = (time_s * (3.7 + 0.08 * (i % 5)) + i / len(particles)) % 1.0
            angle = i * 2.399963 + 0.5 * time_s
            radius = (0.10 + 0.60 * phase) * (0.75 + 0.25 * math.sin(i * 7.1))
            particle.location = (radius * math.cos(angle),
                                 radius * math.sin(angle),
                                 0.42 - 5.2 * phase)
            size = (0.045 + 0.09 * (1 - phase)) * opening
            particle.scale = (size, size, size)
            particle.keyframe_insert(data_path="location", frame=frame)
            particle.keyframe_insert(data_path="scale", frame=frame)

    scene.frame_set(1)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_perspective = "CAMERA"
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND))
    for frame, path in ((1 + round(1.5 * FPS), PREVIEW_LAUNCH),
                        (1 + round(10.0 * FPS), PREVIEW_ASCENT),
                        (scene.frame_end, OUTPUT_DIR / "rocket_hop_touchdown_preview.png")):
        scene.frame_set(frame)
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
    scene.frame_set(1)
    print(f"ANIMATED_SCENE={OUTPUT_BLEND}")
    print(f"FRAMES={scene.frame_start}:{scene.frame_end}")
    print(f"DURATION_S={times[-1]:.6f}")
    print(f"APOGEE_M={apex['height']:.6f}")
    print(f"APOGEE_FRAME={apex_frame}")


if __name__ == "__main__":
    main()
