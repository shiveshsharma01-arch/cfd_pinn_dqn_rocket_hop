"""Animate the saved zero-step DQN failure in a separate Blender project.

Open rocket_hop_v04.blend before running. The trajectory and HUD come from the
recorded simulator trace. The post-touchdown tip-over and dust are illustrative
impact effects, not a structural or combustion simulation.
"""

from __future__ import annotations

import bisect
import csv
import math
import sys
from pathlib import Path

import bpy


FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
TRACE = ROOT / "flight_model/results/landing/seed0_v3/step000000_trace.csv"
OUTPUT = FOLDER / "rocket_hop_failed_step000000.blend"
sys.path.insert(0, str(FOLDER))
from enhance_rocket_hop import add_hud, make_material, text_object, update_variants, set_visible  # noqa: E402


def clear_animation(obj):
    if obj.animation_data:
        obj.animation_data_clear()


def move_to_collection(obj, collection):
    for old in list(obj.users_collection):
        old.objects.unlink(obj)
    collection.objects.link(obj)


def add_dust(scene, impact_frame):
    collection = bpy.data.collections.new("08  Illustrative impact dust")
    scene.collection.children.link(collection)
    dust = make_material("Impact dust | warm grey", (0.43, 0.39, 0.32),
                         roughness=1.0, alpha=0.20)
    for index in range(18):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6,
                                             radius=1.0, location=(0, 0, 0))
        obj = bpy.context.object
        obj.name = f"Impact dust puff {index + 1:02d}"
        move_to_collection(obj, collection)
        obj.data.materials.append(dust)
        obj.visible_shadow = False
        angle = 2 * math.pi * index / 18
        phase = (index % 4) / 4
        for frame, progress in ((impact_frame - 1, 0),
                                (impact_frame + 8, 0.25),
                                (impact_frame + 25, 0.75),
                                (impact_frame + 55, 1.0)):
            radius = (1.1 + 7.0 * progress) * (0.82 + 0.22 * phase)
            obj.location = (radius * math.cos(angle), radius * math.sin(angle),
                            0.4 + (0.9 + phase) * progress)
            size = 0.001 if progress == 0 else (0.15 + 0.7 * progress) * (0.8 + phase)
            obj.scale = (size, size, size * 0.55)
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="scale", frame=frame)
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_render", frame=impact_frame - 1)
        obj.hide_render = False
        obj.keyframe_insert(data_path="hide_render", frame=impact_frame)
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_render", frame=impact_frame + 56)
    return collection


def rebuild_hud(scene, camera, rows, times, impact_frame):
    old = bpy.data.collections.get("07  Camera HUD | keyframed data")
    if old:
        for obj in list(old.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old)
    collection = bpy.data.collections.new("07  Camera HUD | early DQN attempt")
    scene.collection.children.link(collection)
    digits, status = add_hud(camera, collection)
    bpy.data.objects["HUD title"].data.body = "500 m  |  EARLY DQN"
    white = bpy.data.materials.get("HUD | neutral white")
    impact = text_object("HUD engine | IMPACT", "IMPACT", -1.10, 0.86,
                         0.072, camera, collection, white)
    status["IMPACT"] = impact
    note = text_object("HUD impact note", "FAILED LANDING  |  75.1 m/s IMPACT",
                       -0.27, -0.95, 0.067, camera, collection, white)
    for obj in collection.objects:
        obj.visible_shadow = False
    for frame, shown in ((1, False), (impact_frame, True)):
        set_visible(note, shown, frame)

    previous = {name: [None] * len(slots) for name, slots in digits.items()}
    previous_status = None
    for frame in range(scene.frame_start, scene.frame_end + 1):
        time_s = min((frame - 1) / 30.0, times[-1])
        index = max(0, bisect.bisect_right(times, time_s) - 1)
        a, b = rows[index], rows[min(index + 1, len(rows) - 1)]
        ta, tb = float(a["time_s"]), float(b["time_s"])
        fraction = (time_s - ta) / (tb - ta) if tb > ta else 0.0
        value = lambda key: float(a[key]) + fraction * (float(b[key]) - float(a[key]))
        altitude = max(0, min(999, round(value("height_m"))))
        spool = max(0, min(100, round(100 * value("actuator_spool"))))
        pressure = round(float(a["command_kpa"] or 0))
        fuel = max(0, min(99, round(value("fuel_remaining_kg"))))
        numbers = {"altitude": f"{altitude:03d}",
                   "pressure": f"{pressure:04d}",
                   "actuator": f"{spool:03d}",
                   "fuel": f"{fuel:02d}"}
        for field, string in numbers.items():
            for slot, char in enumerate(string):
                digit = int(char)
                update_variants(digits[field][slot], previous[field][slot], digit, frame)
                previous[field][slot] = digit
        if frame >= impact_frame:
            current_status = "IMPACT"
        elif pressure == 0:
            current_status = "COAST"
        elif spool < 75:
            current_status = "IGNITION"
        elif value("vertical_speed_m_s") >= 0:
            current_status = "ASCENT BURN"
        else:
            current_status = "LANDING BURN"
        if previous_status is None:
            for name, obj in status.items():
                set_visible(obj, name == current_status, frame)
        elif current_status != previous_status:
            set_visible(status[previous_status], False, frame)
            set_visible(status[current_status], True, frame)
        previous_status = current_status


def main():
    with TRACE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 100 or abs(float(rows[-1]["height_m"])) > 1e-6:
        raise ValueError("Expected the saved zero-step DQN touchdown trace")
    times = [float(row["time_s"]) for row in rows]
    scene = bpy.context.scene
    fps = 30
    impact_frame = 1 + math.ceil(times[-1] * fps)
    scene.frame_start = 1
    scene.frame_end = impact_frame + 56
    scene.render.fps = fps
    scene.render.resolution_percentage = 100
    rig = bpy.data.objects["ROCKET_RIG | animate location Z"]
    camera = scene.camera
    outer = bpy.data.objects["Exhaust envelope | actuator-linked"]
    core = bpy.data.objects["Exhaust core | actuator-linked"]
    particles = [bpy.data.objects[f"Nozzle particle {i + 1:02d}"] for i in range(18)]
    light = bpy.data.objects["Engine reflected light"].data
    for obj in (rig, camera, outer, core, *particles):
        clear_animation(obj)
    clear_animation(light)
    rig.rotation_mode = "XYZ"
    apex_time = float(max(rows, key=lambda row: float(row["height_m"]))["time_s"])
    apex_frame = 1 + round(apex_time * fps)
    for marker in list(scene.timeline_markers):
        scene.timeline_markers.remove(marker)
    for label, frame in (("Launch | early DQN", 1),
                         ("Peak | 621 m", apex_frame),
                         ("Failed touchdown | 75.1 m/s", impact_frame)):
        scene.timeline_markers.new(label, frame=frame)

    for frame in range(1, impact_frame + 1):
        time_s = min((frame - 1) / fps, times[-1])
        index = max(0, bisect.bisect_right(times, time_s) - 1)
        a, b = rows[index], rows[min(index + 1, len(rows) - 1)]
        ta, tb = float(a["time_s"]), float(b["time_s"])
        blend = (time_s - ta) / (tb - ta) if tb > ta else 0
        height = float(a["height_m"]) + blend * (float(b["height_m"]) - float(a["height_m"]))
        opening = float(a["actuator_spool"]) + blend * (float(b["actuator_spool"]) - float(a["actuator_spool"]))
        rig.location = (0, 0, max(0, height))
        rig.rotation_euler = (0, 0, 0)
        rig.keyframe_insert(data_path="location", frame=frame)
        rig.keyframe_insert(data_path="rotation_euler", frame=frame)
        camera.location = (25, -39, 12 + max(0, height))
        camera.keyframe_insert(data_path="location", frame=frame)
        for obj in (outer, core):
            obj.scale = (opening,) * 3
            obj.keyframe_insert(data_path="scale", frame=frame)
        light.energy = 160 * opening
        light.keyframe_insert(data_path="energy", frame=frame)
        for i, particle in enumerate(particles):
            phase = (time_s * (3.7 + 0.08 * (i % 5)) + i / len(particles)) % 1
            angle = i * 2.399963 + 0.5 * time_s
            radius = (0.10 + 0.60 * phase) * (0.75 + 0.25 * math.sin(i * 7.1))
            particle.location = (radius * math.cos(angle), radius * math.sin(angle),
                                 0.42 - 5.2 * phase)
            size = (0.045 + 0.09 * (1 - phase)) * opening
            particle.scale = (size,) * 3
            particle.keyframe_insert(data_path="location", frame=frame)
            particle.keyframe_insert(data_path="scale", frame=frame)

    for frame, tilt in ((impact_frame, 0), (impact_frame + 15, 24),
                        (impact_frame + 35, 66), (impact_frame + 55, 83)):
        rig.location = (0, 0, 0)
        rig.rotation_euler = (0, math.radians(tilt), 0)
        rig.keyframe_insert(data_path="location", frame=frame)
        rig.keyframe_insert(data_path="rotation_euler", frame=frame)
        camera.location = (25, -39, 12)
        camera.keyframe_insert(data_path="location", frame=frame)
    for obj in (outer, core, *particles):
        obj.scale = (0,) * 3
        obj.keyframe_insert(data_path="scale", frame=impact_frame + 1)
    light.energy = 0
    light.keyframe_insert(data_path="energy", frame=impact_frame + 1)
    add_dust(scene, impact_frame)
    rebuild_hud(scene, camera, rows, times, impact_frame)
    scene["trace_file"] = TRACE.relative_to(ROOT).as_posix()
    scene["failure_result"] = "DQN step 0: 621.204 m peak; 75.106 m/s touchdown"
    scene["impact_note"] = "Post-contact tilt and dust are illustrative, not simulator or structural outputs"
    scene["comparison_note"] = "Successful v04 animation uses DQN ascent plus guided descent; not pure DQN"
    scene.frame_set(1)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT))
    for frame, label in ((46, "launch"), (apex_frame, "apogee"),
                         (impact_frame + 42, "crash")):
        scene.frame_set(frame)
        scene.render.filepath = str(FOLDER / f"rocket_hop_failed_step000000_{label}.png")
        bpy.ops.render.render(write_still=True)
    print(f"FAILURE_SCENE={OUTPUT}")
    print(f"IMPACT_FRAME={impact_frame}; END_FRAME={scene.frame_end}")


if __name__ == "__main__":
    main()
