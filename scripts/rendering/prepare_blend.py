# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.

import argparse
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np
import yaml
from mathutils import Matrix


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    print("Uncaught exception in prepare_blend.py:", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
    sys.exit(1)


sys.excepthook = handle_exception


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(
        description=(
            "Build a clean BAT3R rendering .blend scene (and optional canonical *_shape.gltf) "
            "from user-downloaded .glb/.gltf meshes."
        )
    )
    parser.add_argument(
        "--category-config",
        type=Path,
        required=True,
        help="Path to configs/category/<category>.yaml.",
    )
    parser.add_argument(
        "--input-mesh",
        type=Path,
        default=None,
        help="Input .glb/.gltf path for single-model categories (e.g. chimp, elephant).",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="MODEL_NAME=MESH_PATH",
        help="Repeatable mapping for multi-model categories (e.g. --model horse_male-rd=/path/to/male.gltf).",
    )
    parser.add_argument(
        "--out-blend",
        type=Path,
        required=True,
        help="Output path for the generated <category>.blend file.",
    )
    parser.add_argument(
        "--out-shapes-dir",
        type=Path,
        default=None,
        help="Optional shapes/ directory to also export <model_name>/<model_name>_shape.gltf (+ .bin).",
    )
    parser.add_argument(
        "--drop-submeshes",
        nargs="+",
        default=[],
        help=(
            "Sub-meshes to discard before joining, matched by sub-mesh object name or assigned "
            "material name (extends rendering.models.<name>.drop_submeshes in category config)."
        ),
    )
    return parser.parse_args(argv)


def resolve_model_inputs(args: argparse.Namespace, models_cfg: dict) -> dict[str, Path]:
    if args.input_mesh is not None and args.model:
        raise ValueError("Specify either --input-mesh or --model, not both.")
    if args.input_mesh is None and not args.model:
        raise ValueError("Must specify --input-mesh (single-model) or --model NAME=PATH (multi-model).")

    model_paths: dict[str, Path] = {}
    if args.input_mesh is not None:
        if len(models_cfg) != 1:
            raise ValueError(
                f"Category config defines {len(models_cfg)} models ({list(models_cfg.keys())}); "
                f"use --model NAME=PATH for each model instead of --input-mesh."
            )
        only_name = next(iter(models_cfg))
        model_paths[only_name] = args.input_mesh
    else:
        for spec in args.model:
            if "=" not in spec:
                raise ValueError(f"Invalid --model '{spec}': expected MODEL_NAME=MESH_PATH.")
            name, path_str = spec.split("=", 1)
            if name not in models_cfg:
                raise KeyError(
                    f"Unknown model '{name}'; configured models: {list(models_cfg.keys())}"
                )
            model_paths[name] = Path(path_str)

    missing = [name for name in models_cfg if name not in model_paths]
    if missing:
        raise ValueError(f"Missing input mesh for configured model(s): {missing}")

    for name, path in model_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Input mesh for '{name}' not found: {path}")

    return model_paths


def setup_base_scene() -> None:
    """Create the exact BAT3R Blender 4.5 scene (World HDRI, Compositor mask, Light, Camera)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # Render & color management defaults (Standard / Medium Contrast are NOT overwritten at render time)
    scene.frame_set(0)  # blender_render_posed.py expects frame 0 for mask_{file_id}_#.png -> _0.png
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.compositor_device = "CPU"
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium Contrast"
    scene.view_settings.exposure = 0.0

    # Enable Object Index pass so Render Layers exposes the IndexOB socket in Compositor
    scene.view_layers[0].use_pass_object_index = True

    # World HDRI shader: Texture Coordinate -> Mapping -> Environment Texture -> Background(0.5) -> Output
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    wtree = world.node_tree
    wtree.nodes.clear()

    tex_coord = wtree.nodes.new("ShaderNodeTexCoord")
    mapping = wtree.nodes.new("ShaderNodeMapping")
    mapping.name = "Mapping"
    env_tex = wtree.nodes.new("ShaderNodeTexEnvironment")
    env_tex.name = "Environment Texture"
    bg = wtree.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 0.5
    world_out = wtree.nodes.new("ShaderNodeOutputWorld")

    wtree.links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    wtree.links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
    wtree.links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    wtree.links.new(bg.outputs["Background"], world_out.inputs["Surface"])

    # Compositor graph: Render Layers [IndexOB] > 0.0 -> File Output (label="mask", PNG BW)
    scene.use_nodes = True
    ctree = scene.node_tree
    ctree.nodes.clear()

    rl = ctree.nodes.new("CompositorNodeRLayers")
    comp = ctree.nodes.new("CompositorNodeComposite")
    math_node = ctree.nodes.new("CompositorNodeMath")
    math_node.operation = "GREATER_THAN"
    math_node.use_clamp = False
    math_node.inputs[1].default_value = 0.0

    mask_out = ctree.nodes.new("CompositorNodeOutputFile")
    mask_out.label = "mask"
    mask_out.format.file_format = "PNG"
    mask_out.format.color_mode = "BW"

    ctree.links.new(rl.outputs["Image"], comp.inputs["Image"])
    index_socket = rl.outputs.get("IndexOB") or rl.outputs.get("Object Index")
    if index_socket is None:
        raise RuntimeError(
            f"Could not find Object Index socket on Render Layers node; available: {[s.name for s in rl.outputs]}"
        )
    ctree.links.new(index_socket, math_node.inputs[0])
    ctree.links.new(math_node.outputs["Value"], mask_out.inputs[0])

    # Camera (lens and matrix_world are overwritten per sample by blender_render_posed.py)
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 31.17691421508789
    cam_data.clip_start = 0.1
    cam_data.clip_end = 100.0
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.location = (2.4809, 2.4809, 2.0257)
    cam_obj.rotation_euler = (1.2978, 0.0, -0.9863)
    scene.camera = cam_obj

    # Default Point Light (matching horses.blend / chimp.blend / elephant.blend)
    light_data = bpy.data.lights.new("Light", type="POINT")
    light_data.energy = 1000.0
    light_data.color = (1.0, 1.0, 1.0)
    light_data.shadow_soft_size = 0.1
    light_obj = bpy.data.objects.new("Light", light_data)
    scene.collection.objects.link(light_obj)
    light_obj.location = (4.0762, 1.0055, 5.9039)
    light_obj.rotation_euler = (0.6503, 0.0552, 1.8664)


def build_randomizable_texture_material(mat_name: str) -> bpy.types.Material:
    """Create the 4-node texture-swapping shader expected by blender_render_posed.py."""
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    ntree = mat.node_tree
    ntree.nodes.clear()

    bsdf = ntree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["IOR"].default_value = 1.45
    mat_out = ntree.nodes.new("ShaderNodeOutputMaterial")
    tex_node = ntree.nodes.new("ShaderNodeTexImage")
    tex_node.name = "Image Texture"
    sep_color = ntree.nodes.new("ShaderNodeSeparateColor")
    sep_color.mode = "RGB"
    sep_color.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)

    ntree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    ntree.links.new(sep_color.outputs["Green"], bsdf.inputs["Roughness"])
    ntree.links.new(sep_color.outputs["Blue"], bsdf.inputs["Metallic"])
    ntree.links.new(bsdf.outputs["BSDF"], mat_out.inputs["Surface"])
    return mat


def _matches_drop_list(obj: bpy.types.Object, drop_names: set[str]) -> bool:
    if obj.name in drop_names or (obj.data and obj.data.name in drop_names):
        return True
    if obj.type == "MESH" and obj.data:
        for mat in obj.data.materials:
            if mat is not None and mat.name in drop_names:
                return True
    return False


def _is_skinned_to(obj: bpy.types.Object, arm: bpy.types.Object) -> bool:
    if obj.type != "MESH":
        return False
    if obj.parent == arm:
        return True
    return any(m.type == "ARMATURE" and m.object == arm for m in obj.modifiers)


def import_and_prepare_model(
    model_name: str,
    mesh_path: Path,
    model_cfg: dict,
    *,
    randomize_textures: bool,
    extra_drop_submeshes: list[str],
    out_shapes_dir: Path | None,
) -> tuple[bpy.types.Object, bpy.types.Object]:
    before_objs = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    new_objs = [o for o in bpy.data.objects if o not in before_objs]

    armatures = [o for o in new_objs if o.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(
            f"Expected exactly 1 armature when importing {mesh_path}, found {len(armatures)}: "
            f"{[a.name for a in armatures]}"
        )
    arm = armatures[0]

    skinned_meshes = sorted(
        (o for o in new_objs if _is_skinned_to(o, arm)),
        key=lambda o: o.name,
    )
    if not skinned_meshes:
        raise RuntimeError(f"No skinned meshes found under armature '{arm.name}' in {mesh_path}")

    drop_names = set(model_cfg.get("drop_submeshes", [])) | set(extra_drop_submeshes)
    keep_meshes = []
    dropped_meshes = []
    for m in skinned_meshes:
        if _matches_drop_list(m, drop_names):
            dropped_meshes.append(m)
        else:
            keep_meshes.append(m)

    if dropped_meshes:
        print(
            f"[{model_name}] Dropped {len(dropped_meshes)} sub-mesh(es): "
            f"{[(m.name, [mt.name for mt in m.data.materials if mt]) for m in dropped_meshes]}"
        )
    if not keep_meshes:
        raise RuntimeError(
            f"All sub-meshes were dropped for '{model_name}' using drop_submeshes={sorted(drop_names)}"
        )

    # Bake armature world transform and clear Sketchfab/glTF wrapper parents
    bpy.context.view_layer.update()
    arm_mw = arm.matrix_world.copy()
    arm.parent = None
    if np.allclose(np.array(arm_mw), np.eye(4), atol=1e-5):
        arm.matrix_world = Matrix.Identity(4)
    else:
        arm.matrix_world = arm_mw

    # Delete dropped sub-meshes, wrapper empties, and stray helper meshes (e.g. Icosphere)
    keep_set = {arm, *keep_meshes}
    for obj in list(new_objs):
        if obj not in keep_set:
            bpy.data.objects.remove(obj, do_unlink=True)

    # Join remaining skinned sub-meshes into a single mesh object
    if len(keep_meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for m in keep_meshes:
            m.hide_set(False)
            m.select_set(True)
        bpy.context.view_layer.objects.active = keep_meshes[0]
        bpy.ops.object.join()
    mesh_obj = keep_meshes[0]

    # Ensure clean parenting to armature with identity local transform
    if mesh_obj.parent != arm:
        mesh_mw = mesh_obj.matrix_world.copy()
        mesh_obj.parent = arm
        mesh_obj.matrix_world = mesh_mw
    for mod in mesh_obj.modifiers:
        if mod.type == "ARMATURE":
            mod.object = arm

    # Rename armature & mesh to match configs/category/<cat>.yaml
    target_arm_name = model_cfg["armature"]
    target_obj_name = model_cfg["object"]
    arm.name = target_arm_name
    arm.data.name = target_arm_name
    mesh_obj.name = target_obj_name

    # Attach texture-swapping shader when randomize_textures is enabled (e.g. Horse)
    if randomize_textures:
        mat = build_randomizable_texture_material(target_obj_name)
        mesh_obj.data.materials.clear()
        mesh_obj.data.materials.append(mat)

    print(
        f"[{model_name}] Prepared armature={arm.name!r} ({len(arm.data.bones)} bones), "
        f"mesh={mesh_obj.name!r} ({len(mesh_obj.data.vertices)} verts, {len(mesh_obj.data.polygons)} polys, "
        f"mats={[m.name for m in mesh_obj.data.materials if m]})"
    )

    # Optional .gltf export for bat3r/skin.py (export_materials='NONE' guarantees 1 primitive)
    if out_shapes_dir is not None:
        shape_dir = out_shapes_dir / model_name
        shape_dir.mkdir(parents=True, exist_ok=True)
        gltf_out = shape_dir / f"{model_name}_shape.gltf"

        bpy.ops.object.select_all(action="DESELECT")
        arm.select_set(True)
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm

        bpy.ops.export_scene.gltf(
            filepath=str(gltf_out),
            export_format="GLTF_SEPARATE",
            use_selection=True,
            export_materials="NONE",
            export_yup=True,
        )
        print(f"[{model_name}] Exported canonical GLTF to {gltf_out}")

    return arm, mesh_obj


def main() -> None:
    args = parse_args()

    with args.category_config.open("r", encoding="utf-8") as f:
        category_cfg = yaml.safe_load(f)

    rendering_cfg = category_cfg.get("rendering")
    if not rendering_cfg or not rendering_cfg.get("models"):
        raise ValueError(f"Missing 'rendering.models' in {args.category_config}")

    models_cfg = rendering_cfg["models"]
    randomize_textures = bool(rendering_cfg.get("randomize_textures", False))
    model_paths = resolve_model_inputs(args, models_cfg)

    setup_base_scene()

    for model_name, model_cfg in models_cfg.items():
        import_and_prepare_model(
            model_name=model_name,
            mesh_path=model_paths[model_name],
            model_cfg=model_cfg,
            randomize_textures=randomize_textures,
            extra_drop_submeshes=args.drop_submeshes,
            out_shapes_dir=args.out_shapes_dir,
        )

    # Purge orphaned meshes/materials/images (e.g. dropped sub-mesh textures) before saving
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

    args.out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out_blend))
    print(f"Saved Blender scene to {args.out_blend}")


if __name__ == "__main__":
    main()
