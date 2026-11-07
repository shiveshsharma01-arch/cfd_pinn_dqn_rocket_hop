"""Build the first, static Blender scene for the 500 m hop project.

Run with Blender 5.2 in background mode. The rocket and scenery are original
procedural geometry; the scene deliberately has no flight animation yet.
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Vector


OUTPUT_DIR = Path(__file__).resolve().parent
BLEND_FILE = OUTPUT_DIR / "rocket_pad_v01.blend"
PREVIEW_FILE = OUTPUT_DIR / "rocket_pad_preview.png"


def material(name: str, color: tuple[float, float, float], *, metallic=0.0,
             roughness=0.55):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return mat


def move_to_collection(obj, collection):
    for old in list(obj.users_collection):
        old.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def bevel(obj, width=0.06, segments=3):
    mod = obj.modifiers.new("Softened edges", "BEVEL")
    mod.width = width
    mod.segments = segments
    obj.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")


def cylinder(name, radius, depth, location, mat, collection, vertices=64):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius,
                                        depth=depth, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    move_to_collection(obj, collection)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def cube(name, location, scale, mat, collection):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    move_to_collection(obj, collection)
    return obj


def fin_mesh(name, angle, mat, collection, parent):
    # A thick, swept quadrilateral in a radial/vertical plane.
    contour = [(0.82, 1.65), (3.05, 0.9), (2.52, 3.45), (0.9, 5.15)]
    half = 0.085
    verts = [(r, -half, z) for r, z in contour]
    verts += [(r, half, z) for r, z in contour]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7)]
    faces += [(i, (i + 1) % 4, (i + 1) % 4 + 4, i + 4)
              for i in range(4)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.parent = parent
    obj.rotation_euler.z = angle
    obj.data.materials.append(mat)
    bevel(obj, 0.035, 2)
    return obj


def annulus_sector(name, inner, outer, angle_a, angle_b, z, mat, collection):
    points = []
    subdivisions = 5
    for radius in (inner, outer):
        for i in range(subdivisions + 1):
            a = angle_a + (angle_b - angle_a) * i / subdivisions
            points.append((radius * math.cos(a), radius * math.sin(a), z))
    faces = []
    for i in range(subdivisions):
        faces.append((i, i + 1, subdivisions + i + 2, subdivisions + i + 1))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def main():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0

    terrain = bpy.data.collections.new("01  Ground and sky")
    pad = bpy.data.collections.new("02  Launch pad")
    rocket = bpy.data.collections.new("03  Rocket — animate this rig later")
    preview = bpy.data.collections.new("04  Temporary preview camera and light")
    for collection in (terrain, pad, rocket, preview):
        scene.collection.children.link(collection)

    chalk = material("Rocket | warm white", (0.86, 0.88, 0.84), metallic=0.22)
    charcoal = material("Rocket | graphite", (0.035, 0.052, 0.064), metallic=0.52)
    copper = material("Rocket | orange identification", (0.92, 0.26, 0.075), metallic=0.2)
    steel = material("Pad | brushed steel", (0.23, 0.28, 0.30), metallic=0.62)
    concrete = material("Pad | pale concrete", (0.43, 0.45, 0.40))
    earth = material("Ground | muted sand", (0.48, 0.45, 0.35))
    warning = material("Pad | safety amber", (0.98, 0.57, 0.10))
    dark = material("Pad | asphalt", (0.11, 0.14, 0.15))

    ground = cube("Ground plain", (0, 0, -0.62), (500, 500, 0.9), earth, terrain)
    ground["purpose"] = "Static scenery; 1 Blender unit = 1 metre"
    # A subdued access road anchors the pad without making this a complex site.
    road = cube("Access road", (0, -63, -0.145), (8, 120, 0.02), dark, terrain)
    bevel(road, 0.15)

    base = cylinder("Circular concrete foundation", 10.0, 0.45,
                    (0, 0, 0.0), concrete, pad, 96)
    bevel(base, 0.15)
    deck = cylinder("Steel launch deck", 7.4, 0.22,
                    (0, 0, 0.32), steel, pad, 96)
    bevel(deck, 0.09)
    inner = cylinder("Dark engine recess", 2.55, 0.04,
                     (0, 0, 0.45), dark, pad, 96)
    for i in range(24):
        start = 2 * math.pi * i / 24
        end = 2 * math.pi * (i + 1) / 24 - 0.012
        annulus_sector(f"Safety stripe {i + 1:02d}", 5.6, 6.6,
                       start, end, 0.437,
                       warning if i % 2 == 0 else dark, pad)
    for i in range(4):
        a = math.pi * (i + 0.5) / 2
        r = 4.55
        x, y = r * math.cos(a), r * math.sin(a)
        post = cylinder(f"Service post {i + 1}", 0.17, 0.9,
                        (x, y, 0.86), steel, pad, 20)
        bevel(post, 0.04)
        cylinder(f"Amber marker {i + 1}", 0.23, 0.09,
                 (x, y, 1.36), warning, pad, 20)

    rig = bpy.data.objects.new("ROCKET_RIG | animate location Z", None)
    rocket.objects.link(rig)
    rig.empty_display_type = "PLAIN_AXES"
    rig.empty_display_size = 3.5
    rig["description"] = "All rocket meshes are children. Keep this origin at ground level."
    rig["scale"] = "1 Blender unit = 1 metre; vehicle is illustrative, not a real design"

    body = cylinder("Main propellant stage", 1.08, 10.85,
                    (0, 0, 7.875), chalk, rocket, 96)
    body.parent = rig
    bevel(body, 0.06)
    skirt = cylinder("Lower engine skirt", 1.14, 1.25,
                     (0, 0, 1.84), charcoal, rocket, 96)
    skirt.parent = rig
    bevel(skirt, 0.05)
    band = cylinder("Orange flight identification band", 1.095, 0.64,
                    (0, 0, 10.55), copper, rocket, 96)
    band.parent = rig
    bevel(band, 0.025)
    upper_band = cylinder("Graphite upper band", 1.092, 0.26,
                          (0, 0, 12.82), charcoal, rocket, 96)
    upper_band.parent = rig
    # Rounded shoulder under the nose.
    collar = cylinder("Nose shoulder", 1.085, 0.23,
                      (0, 0, 13.4), charcoal, rocket, 96)
    collar.parent = rig
    bpy.ops.mesh.primitive_cone_add(vertices=96, radius1=1.08, radius2=0.035,
                                    depth=3.45, location=(0, 0, 15.23))
    nose = bpy.context.object
    nose.name = "Tapered nose cone"
    nose.data.materials.append(chalk)
    move_to_collection(nose, rocket)
    nose.parent = rig
    for poly in nose.data.polygons:
        poly.use_smooth = True
    for i in range(4):
        fin_mesh(f"Stabilizing fin {i + 1}", math.pi / 4 + i * math.pi / 2,
                 copper, rocket, rig)
    nozzle = cylinder("Engine nozzle bell", 0.55, 0.82,
                      (0, 0, 0.95), charcoal, rocket, 64)
    nozzle.parent = rig
    bevel(nozzle, 0.04)
    throat = cylinder("Engine throat", 0.34, 0.08,
                      (0, 0, 0.51), dark, rocket, 64)
    throat.parent = rig
    for i in range(4):
        a = math.pi / 4 + i * math.pi / 2
        x, y = 1.38 * math.cos(a), 1.38 * math.sin(a)
        foot = cube(f"Landing foot {i + 1}", (x, y, 0.64),
                    (0.82, 0.48, 0.19), charcoal, rocket)
        foot.rotation_euler.z = a
        foot.parent = rig
        bevel(foot, 0.05)

    # Temporary inspection camera and neutral light; shot design comes later.
    camera_data = bpy.data.cameras.new("Preview camera")
    camera = bpy.data.objects.new("Preview camera — replace later", camera_data)
    preview.objects.link(camera)
    camera.location = (25, -39, 12)
    camera_data.type = "PERSP"
    camera_data.lens = 52
    look_at(camera, (0, 0, 8.5))
    scene.camera = camera
    sun_data = bpy.data.lights.new("Temporary daylight", "SUN")
    sun = bpy.data.objects.new("Temporary daylight — replace later", sun_data)
    preview.objects.link(sun)
    sun.rotation_euler = (math.radians(30), math.radians(-25), math.radians(-30))
    sun_data.energy = 2.0
    sun_data.angle = math.radians(12)

    world = bpy.data.worlds.new("Clear blue sky — provisional")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.31, 0.55, 0.78, 1)
    background.inputs["Strength"].default_value = 0.6
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.render.filepath = str(PREVIEW_FILE)

    scene.frame_set(1)
    scene.frame_start = 1
    scene.frame_end = 250
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_FILE))
    bpy.ops.render.render(write_still=True)
    print(f"SCENE={BLEND_FILE}")
    print(f"PREVIEW={PREVIEW_FILE}")
    print(f"ROCKET_HEIGHT_M={15.23 + 3.45 / 2:.3f}")


if __name__ == "__main__":
    main()
