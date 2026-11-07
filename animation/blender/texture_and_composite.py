"""Material, daylight and compositor pass over the frozen v03 flight scene."""
from pathlib import Path
import math
import bpy
import bmesh

FOLDER = Path(__file__).resolve().parent
TEXTURES = FOLDER / "textures"
OUT = FOLDER / "rocket_hop_v04.blend"


def image(asset, kind):
    paths = list(TEXTURES.glob(f"{asset}_{kind}_1k.*"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one image for {asset} {kind}")
    img = bpy.data.images.load(str(paths[0]), check_existing=True)
    if kind != "Diffuse":
        img.colorspace_settings.name = "Non-Color"
    return img


def pbr(name, asset, *, tile=(1, 1, 1), tint=(1, 1, 1, 1), roughness=1):
    mat = bpy.data.materials.get(name)
    if mat is None:
        raise KeyError(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    texco = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = tile
    nt.links.new(texco.outputs["Generated"], mapping.inputs["Vector"])
    diffuse = nt.nodes.new("ShaderNodeTexImage")
    diffuse.image = image(asset, "Diffuse")
    nt.links.new(mapping.outputs["Vector"], diffuse.inputs["Vector"])
    color = nt.nodes.new("ShaderNodeMixRGB")
    color.blend_type = "MULTIPLY"
    color.inputs[0].default_value = 1
    color.inputs[2].default_value = tint
    nt.links.new(diffuse.outputs["Color"], color.inputs[1])
    nt.links.new(color.outputs["Color"], bsdf.inputs["Base Color"])
    rough = nt.nodes.new("ShaderNodeTexImage")
    rough.image = image(asset, "Rough")
    nt.links.new(mapping.outputs["Vector"], rough.inputs["Vector"])
    mult = nt.nodes.new("ShaderNodeMath")
    mult.operation = "MULTIPLY"
    mult.inputs[1].default_value = roughness
    nt.links.new(rough.outputs["Color"], mult.inputs[0])
    nt.links.new(mult.outputs[0], bsdf.inputs["Roughness"])
    normal = nt.nodes.new("ShaderNodeTexImage")
    normal.image = image(asset, "nor_gl")
    nt.links.new(mapping.outputs["Vector"], normal.inputs["Vector"])
    normap = nt.nodes.new("ShaderNodeNormalMap")
    normap.inputs["Strength"].default_value = 0.55
    nt.links.new(normal.outputs["Color"], normap.inputs["Color"])
    nt.links.new(normap.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def metal_paint(name, *, base, metallic, roughness, noise_scale):
    mat = bpy.data.materials[name]
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*base, 1)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    texco = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = noise_scale
    noise.inputs["Detail"].default_value = 3
    nt.links.new(texco.outputs["Generated"], noise.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.2
    ramp.color_ramp.elements[0].color = (roughness * 0.72,) * 3 + (1,)
    ramp.color_ramp.elements[1].position = 0.8
    ramp.color_ramp.elements[1].color = (min(roughness * 1.25, 1),) * 3 + (1,)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Roughness"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = 0.006
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def add_ring(name, z, radius, depth, material, collection, rig):
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=radius, depth=depth,
                                        location=(0, 0, z))
    obj = bpy.context.object
    obj.name = name
    for old in list(obj.users_collection):
        old.objects.unlink(obj)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    obj.parent = rig
    bevel = obj.modifiers.new("Rolled lip", "BEVEL")
    bevel.width = 0.009
    bevel.segments = 2
    obj.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
    for polygon in obj.data.polygons:
        polygon.use_smooth = True


def main():
    scene = bpy.context.scene
    pbr("Ground | muted sand", "dirt", tile=(3000, 3000, 1),
        tint=(0.72, 0.68, 0.58, 1))
    pbr("Pad | pale concrete", "concrete_floor", tile=(8, 8, 1),
        tint=(0.75, 0.75, 0.71, 1))
    pbr("Mountains | weathered rock", "rock_01", tile=(55, 18, 5),
        tint=(0.70, 0.73, 0.69, 1))
    pbr("Mountains | distant blue-grey", "rock_01", tile=(60, 19, 6),
        tint=(0.57, 0.66, 0.68, 1))
    pbr("Pad | asphalt", "concrete_floor", tile=(6, 70, 1),
        tint=(0.27, 0.30, 0.30, 1))
    ground = bpy.data.objects["Ground plain"]
    for vertex in ground.data.vertices:
        vertex.co.x *= 40
        vertex.co.y *= 40
    for name in ("Near mountain range | peaks below 500 m",
                 "Far mountain range | atmospheric depth"):
        mountain = bpy.data.objects[name]
        for vertex in mountain.data.vertices:
            vertex.co.z *= 0.18
        mesh_edit = bmesh.new()
        mesh_edit.from_mesh(mountain.data)
        low_faces = [face for face in mesh_edit.faces
                     if max(vertex.co.z for vertex in face.verts) < 2.0]
        bmesh.ops.delete(mesh_edit, geom=low_faces, context="FACES")
        mesh_edit.to_mesh(mountain.data)
        mesh_edit.free()
    metal_paint("Rocket | warm white", base=(0.72, 0.76, 0.76),
                metallic=0.33, roughness=0.4, noise_scale=110)
    metal_paint("Rocket | graphite", base=(0.08, 0.10, 0.11),
                metallic=0.7, roughness=0.42, noise_scale=85)
    metal_paint("Rocket | orange identification", base=(0.54, 0.19, 0.06),
                metallic=0.43, roughness=0.4, noise_scale=100)
    metal_paint("Pad | brushed steel", base=(0.21, 0.23, 0.23),
                metallic=0.78, roughness=0.49, noise_scale=95)

    rig = bpy.data.objects["ROCKET_RIG | animate location Z"]
    rocket = bpy.data.collections["03  Rocket — animate this rig later"]
    seam = bpy.data.materials["Rocket | graphite"]
    for i, z in enumerate((4.1, 7.5, 11.8), 1):
        add_ring(f"Stage panel seam {i:02d}", z, 1.087, 0.022, seam, rocket, rig)

    world = scene.world
    sky = next(n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeTexSky")
    sky.sun_elevation = math.radians(38)
    sky.sun_rotation = math.radians(110)
    sky.air_density = 1.0
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.32
    sun = bpy.data.objects["Daylight sun | neutral draft"]
    sun.data.energy = 2.8
    sun.data.angle = math.radians(4)
    sun.rotation_euler = (math.radians(43), math.radians(-18), math.radians(-35))
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = -0.3
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    scene.use_nodes = True
    if scene.compositing_node_group is None:
        scene.compositing_node_group = bpy.data.node_groups.new("Rocket daylight composite", "CompositorNodeTree")
    tree = scene.compositing_node_group
    nodes = tree.nodes
    nodes.clear()
    links = tree.links
    layers = nodes.new("CompositorNodeRLayers")
    glare = nodes.new("CompositorNodeGlare")
    glare.inputs["Type"].default_value = "Fog Glow"
    glare.inputs["Quality"].default_value = "Medium"
    glare.inputs["Threshold"].default_value = 2.2
    glare.inputs["Strength"].default_value = 0.08
    tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    composite = nodes.new("NodeGroupOutput")
    links.new(layers.outputs["Image"], glare.inputs["Image"])
    links.new(glare.outputs["Image"], composite.inputs["Image"])
    scene["visual_pass"] = "CC0 Poly Haven textures; procedural painted metal; daylight and subtle glow"
    scene["asset_license"] = "Poly Haven CC0: dirt, concrete_floor, rock_01"
    scene.frame_set(1)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT))
    for frame, label in ((46, "launch"), (398, "apogee"), (scene.frame_end, "touchdown")):
        scene.frame_set(frame)
        scene.render.filepath = str(FOLDER / f"rocket_hop_v04_{label}.png")
        bpy.ops.render.render(write_still=True)
    print(f"FINAL_SCENE={OUT}")


if __name__ == "__main__":
    main()
