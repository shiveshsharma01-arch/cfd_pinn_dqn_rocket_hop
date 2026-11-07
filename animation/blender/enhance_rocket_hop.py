"""Add scale-reference mountains, neutral surface detail, and a flight HUD.

Run with rocket_hop_v02.blend loaded in Blender 5.2. The HUD is keyframed
geometry, so it works after reopening the .blend with no auto-run scripts.
"""

from __future__ import annotations

import bisect
import csv
import math
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent
TRACE = ROOT / "flight_model/results/landing/final/dqn_ascent_guided_descent_nominal_trace.csv"
OUTPUT = FOLDER / "rocket_hop_v03.blend"


def make_material(name, color, roughness=0.8, alpha=1.0, emission=0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, alpha)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Alpha"].default_value = alpha
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*color, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission
    if alpha < 1:
        mat.surface_render_method = "DITHERED"
    return mat


def add_surface_noise(material, *, scale, strength, distance):
    tree = material.node_tree
    bsdf = tree.nodes.get("Principled BSDF")
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 4.0
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = distance
    tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def ridge(x, peaks):
    total = 0.0
    for centre, height, width in peaks:
        total += height * math.exp(-0.5 * ((x - centre) / width) ** 2)
    rough = 12 * math.sin(x / 43) + 6 * math.sin(x / 16) + 2 * math.sin(x / 7)
    return max(0.0, total + rough)


def make_mountain_range(collection, name, y_centre, spread, peaks, rock, snow):
    nx, ny = 190, 58
    vertices = []
    faces = []
    for iy in range(ny + 1):
        y = y_centre - spread + 2 * spread * iy / ny
        falloff = math.exp(-0.5 * ((y - y_centre) / (spread * 0.46)) ** 2)
        for ix in range(nx + 1):
            x = -1450.0 + 2900.0 * ix / nx
            z = ridge(x, peaks) * falloff
            vertices.append((x, y, z - 0.35))
    for iy in range(ny):
        for ix in range(nx):
            a = iy * (nx + 1) + ix
            faces.append((a, a + 1, a + nx + 2, a + nx + 1))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    mesh.materials.append(rock)
    mesh.materials.append(snow)
    for face in mesh.polygons:
        z = sum(mesh.vertices[v].co.z for v in face.vertices) / len(face.vertices)
        face.material_index = 1 if z > 275 + 18 * math.sin(face.index * 0.37) else 0
        face.use_smooth = True
    obj["purpose"] = "Visual scale reference only; not part of the flight dynamics"
    return obj


def text_object(name, body, x, y, size, camera, collection, material):
    data = bpy.data.curves.new(name, "FONT")
    data.body = body
    data.size = size
    data.extrude = 0.0
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    obj.parent = camera
    obj.location = (x, y, -5.0)
    data.materials.append(material)
    return obj


def add_hud(camera, collection):
    white = make_material("HUD | neutral white", (0.92, 0.95, 0.96), emission=1.0)
    muted = make_material("HUD | subdued label", (0.64, 0.75, 0.80), emission=0.8)
    panel_mat = make_material("HUD | solid graphite", (0.025, 0.042, 0.052),
                              alpha=1.0, emission=0.08)
    vertices = [(-1.68, -0.02, -5.04), (-0.40, -0.02, -5.04),
                (-0.40, 1.26, -5.04), (-1.68, 1.26, -5.04)]
    mesh = bpy.data.meshes.new("HUD panel mesh")
    mesh.from_pydata(vertices, [], [(0, 1, 2, 3)])
    mesh.materials.append(panel_mat)
    panel = bpy.data.objects.new("HUD | top-left panel", mesh)
    collection.objects.link(panel)
    panel.parent = camera
    text_object("HUD title", "500 m  |  VERTICAL HOP", -1.54, 1.08, 0.085,
                camera, collection, white)
    fields = [
        ("ENGINE", 0.86),
        ("ALT", 0.66),
        ("INLET", 0.46),
        ("VALVE", 0.26),
        ("FUEL", 0.06),
    ]
    for label, y in fields:
        text_object(f"HUD label | {label}", label, -1.54, y, 0.068,
                    camera, collection, muted)
    # Four engine-state alternatives. Bool keyframes switch exactly on frame.
    status = {}
    for name in ("IGNITION", "ASCENT BURN", "COAST", "LANDING BURN", "TOUCHDOWN"):
        status[name] = text_object(f"HUD engine | {name}", name,
                                   -1.10, 0.86, 0.072,
                                   camera, collection, white)
    # Each digit has ten objects. Only one is visible for each slot. Using
    # keyframed visibility avoids Blender auto-run scripts and works on reopen.
    digits = {}
    specs = {
        "altitude": (3, -1.16, 0.66, "m"),
        "pressure": (4, -1.16, 0.46, "kPa"),
        "actuator": (3, -1.16, 0.26, "%"),
        "fuel": (2, -1.16, 0.06, "kg"),
    }
    for field, (count, x, y, unit) in specs.items():
        digits[field] = []
        for slot in range(count):
            variants = [text_object(f"HUD {field} digit {slot} value {digit}",
                                    str(digit), x + 0.080 * slot, y, 0.095,
                                    camera, collection, white)
                        for digit in range(10)]
            digits[field].append(variants)
        text_object(f"HUD unit | {field}", unit, x + 0.080 * count + 0.025,
                    y, 0.065, camera, collection, muted)
    return digits, status


def set_visible(obj, shown, frame):
    obj.hide_render = not shown
    obj.hide_viewport = not shown
    obj.keyframe_insert(data_path="hide_render", frame=frame)
    obj.keyframe_insert(data_path="hide_viewport", frame=frame)


def update_variants(variants, old_value, new_value, frame):
    if old_value is None:
        for i, obj in enumerate(variants):
            set_visible(obj, i == new_value, frame)
    elif old_value != new_value:
        set_visible(variants[old_value], False, frame)
        set_visible(variants[new_value], True, frame)


def main():
    scene = bpy.context.scene
    camera = scene.camera
    scenery = bpy.data.collections.new("06  Mountain ranges | altitude reference")
    hud_collection = bpy.data.collections.new("07  Camera HUD | keyframed data")
    scene.collection.children.link(scenery)
    scene.collection.children.link(hud_collection)

    rock = make_material("Mountains | weathered rock", (0.34, 0.37, 0.37), 0.95)
    snow = rock
    add_surface_noise(rock, scale=11, strength=0.18, distance=0.22)
    ground = bpy.data.materials.get("Ground | muted sand")
    if ground:
        add_surface_noise(ground, scale=7, strength=0.12, distance=0.08)
    paint = bpy.data.materials.get("Rocket | warm white")
    if paint:
        add_surface_noise(paint, scale=80, strength=0.025, distance=0.015)
    make_mountain_range(
        scenery, "Near mountain range | peaks below 500 m", 620, 400,
        [(-1050, 230, 185), (-720, 350, 145), (-400, 245, 180),
         (-75, 390, 170), (280, 300, 145), (580, 420, 160), (980, 260, 200)],
        rock, snow,
    )
    far_rock = make_material("Mountains | distant blue-grey", (0.36, 0.40, 0.42), 0.98)
    far_snow = far_rock
    make_mountain_range(
        scenery, "Far mountain range | atmospheric depth", 1320, 610,
        [(-1000, 300, 240), (-600, 420, 240), (-210, 290, 190),
         (210, 450, 220), (690, 340, 230), (1100, 380, 200)],
        far_rock, far_snow,
    )

    # Less stylized lighting while retaining quick EEVEE previews.
    world = scene.world
    sky = next((node for node in world.node_tree.nodes
                if node.bl_idname == "ShaderNodeTexSky"), None)
    if sky:
        sky.sun_elevation = math.radians(55)
        sky.air_density = 1.15
    background = world.node_tree.nodes.get("Background")
    background.inputs["Strength"].default_value = 0.18
    sun = bpy.data.objects.get("Daylight sun | neutral draft")
    if sun:
        sun.data.energy = 2.0
        sun.data.angle = math.radians(1.5)
    scene.view_settings.exposure = -0.55

    digits, status = add_hud(camera, hud_collection)
    with TRACE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    times = [float(row["time_s"]) for row in rows]
    old_digits = {field: [None] * len(slots) for field, slots in digits.items()}
    old_status = None
    for frame in range(scene.frame_start, scene.frame_end + 1):
        time_s = min((frame - 1) / scene.render.fps, times[-1])
        index = max(0, bisect.bisect_right(times, time_s) - 1)
        a = rows[index]
        b = rows[min(index + 1, len(rows) - 1)]
        ta, tb = float(a["time_s"]), float(b["time_s"])
        blend = (time_s - ta) / (tb - ta) if tb > ta else 0.0
        value = lambda key: float(a[key]) + blend * (float(b[key]) - float(a[key]))
        altitude = max(0, min(999, round(value("height_m"))))
        actuator = max(0, min(100, round(100 * value("actuator_spool"))))
        pressure = round(float(a["command_kpa"] or 0))
        fuel = max(0, min(99, round(value("fuel_remaining_kg"))))
        numbers = {"altitude": f"{altitude:03d}",
                   "pressure": f"{pressure:04d}",
                   "actuator": f"{actuator:03d}",
                   "fuel": f"{fuel:02d}"}
        for field, string in numbers.items():
            for slot, character in enumerate(string):
                digit = int(character)
                update_variants(digits[field][slot], old_digits[field][slot],
                                digit, frame)
                old_digits[field][slot] = digit
        speed = value("vertical_speed_m_s")
        if frame == scene.frame_end:
            current_status = "TOUCHDOWN"
        elif pressure == 0:
            current_status = "COAST"
        elif actuator < 75:
            current_status = "IGNITION"
        elif speed >= 0:
            current_status = "ASCENT BURN"
        else:
            current_status = "LANDING BURN"
        if old_status is None:
            for name, obj in status.items():
                set_visible(obj, name == current_status, frame)
        elif old_status != current_status:
            set_visible(status[old_status], False, frame)
            set_visible(status[current_status], True, frame)
        old_status = current_status

    scene["mountain_note"] = "Visual altitude reference; mountain geometry is illustrative"
    scene["hud_note"] = "Pressure is commanded inlet kPa; actuator percent is opening, not measured thrust"
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT))
    for frame, name in ((46, "rocket_hop_v03_launch.png"),
                        (398, "rocket_hop_v03_apogee.png"),
                        (scene.frame_end, "rocket_hop_v03_touchdown.png")):
        scene.frame_set(frame)
        scene.render.filepath = str(FOLDER / name)
        bpy.ops.render.render(write_still=True)
    print(f"SCENE={OUTPUT}")
    print(f"MOUNTAINS=2 ranges; HUD=engine, altitude, pressure, actuator, fuel")


if __name__ == "__main__":
    main()
