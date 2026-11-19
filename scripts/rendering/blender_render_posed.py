# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.

import bpy
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

import yaml
import numpy as np
from mathutils import Vector, Matrix, Quaternion
from tqdm import tqdm


# Force Blender to exit with non-zero code if a Python exception occurs
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    print("❌ Uncaught exception in Blender script:", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
    sys.exit(1)


sys.excepthook = handle_exception

MODEL_TO_SEX = {
    'horse_female-rd': 'female',
    'horse_male-rd': 'male',
}
SEXES = ['female', 'male']
OBJ_NAMES = {
    'female': 'Horse_Female',
    'male': 'Horse_Male',
}
ARM_NAMES = {
    'female': 'Arm_Horse_Female',
    'male': 'Arm_Horse_Male',
}

CAMERA_FOV_DEG = 60

z_up_to_y_up = Matrix((
    (1.0,  0.0,  0.0, 0.0),
    (0.0,  0.0,  1.0, 0.0),
    (0.0, -1.0,  0.0, 0.0),
    (0.0,  0.0,  0.0, 1.0)
))

np.random.seed(42)

# Find separator index
argv = sys.argv
argv = argv[argv.index("--") + 1:]  # all arguments after --
rank = int(argv[0])
world_size = int(argv[1])

root_dir = Path.cwd()
textures_dir = root_dir / "textures"
hdr_dir = root_dir / "hdrs"

if not len(argv) > 2:
    raise ValueError("3 args must be provided")

data_root = Path(argv[2])
out_dir = data_root
pose_ext = 'npy'

meta_dir = data_root / "metadata"
pose_dir_in = data_root / "poses"
cameras_dir_in = data_root / "cameras_in"

# FIND SAMPLES:
train_ids_path = data_root / "train_ids.json"

cam_suffix = "_camera.txt"
avail_ids = set()
with os.scandir(str(cameras_dir_in)) as entries:
    for entry in entries:
        if entry.is_file() and entry.name.endswith(cam_suffix):
            avail_ids.add(entry.name[:-len(cam_suffix)])

with open(train_ids_path, 'r') as f:
    train_ids = set(json.load(f))

all_train_file_ids = sorted([cid for cid in avail_ids if cid.split('_')[0] in train_ids])
print(len(all_train_file_ids), all_train_file_ids[:15])

total = len(all_train_file_ids)
piece = math.ceil(total / world_size)
offset = rank * piece
print(f"Number of renders: {piece}, offset: {offset}")

renders_dir = out_dir / "renders"
masks_dir = out_dir / "masks"
cameras_dir = out_dir / "cameras"

for dir in [out_dir, renders_dir, masks_dir, cameras_dir]:
    if not dir.is_symlink():
        dir.mkdir(exist_ok=True)


textures = {
    sex: sorted(list((textures_dir / sex).glob("*.png")))
    for sex in SEXES
}
hdrs = sorted([p for p in hdr_dir.iterdir() if p.is_file() and p.suffix.lower() in ['.exr', '.hdr']])

TEXTURE_CACHE = {}


def set_texture(obj_name, tex_path):
    path_str = str(tex_path)
    if path_str not in TEXTURE_CACHE:
        TEXTURE_CACHE[path_str] = bpy.data.images.load(path_str)
    tex_image = TEXTURE_CACHE[path_str]

    mat = bpy.data.materials[obj_name]
    tex_node = mat.node_tree.nodes["Image Texture"]
    tex_node.image = tex_image


def set_world_texture(tex_path, rot_radians=0):
    path_str = str(tex_path)
    if path_str not in TEXTURE_CACHE:
        TEXTURE_CACHE[path_str] = bpy.data.images.load(path_str)
    new_hdri = TEXTURE_CACHE[path_str]

    world = bpy.context.scene.world
    nodes = world.node_tree.nodes
    env_tex = nodes["Environment Texture"]
    env_tex.image = new_hdri

    mapping = nodes["Mapping"]
    mapping.inputs["Rotation"].default_value[2] = rot_radians


def render_scene(render_settings, mask_output_node, file_id):
    render_path = renders_dir / f"{file_id}_rgb.png"
    mask_fname = f"mask_{file_id}_#.png"

    render_settings.filepath = str(render_path)
    mask_output_node.file_slots[0].path = mask_fname

    bpy.ops.render.render(write_still=True)

    # blender appends something like sequence number?
    # get rid of it here for consistency
    mask_path = masks_dir / mask_fname.replace("#", "0")
    mask_path.rename(mask_path.with_name(f"{file_id}_mask.png"))


def sample_texture_idx(file_id: str, num_textures: int) -> int:
    h = hashlib.sha256(file_id.encode("utf-8")).hexdigest()
    return int(h, 16) % num_textures


def read_pose_blender(path: Path):
    data = np.load(path)
    if path.suffix == '.npy':
        return [Matrix(pose.tolist()) for pose in data]

    raw_poses = data["poses"]  # Assuming shape (N, 7) or (7,)

    if raw_poses.ndim == 1:
        raw_poses = raw_poses[np.newaxis, :]

    m_targets = []
    for pose in raw_poses:
        quat_data = pose[:4]
        pos_data = pose[4:]

        # Assuming torch data is [w, x, y, z]
        rot = Quaternion(quat_data)
        loc = Vector(pos_data)

        # Build M_target (4x4 Matrix): Translation @ Rotation
        mat_rot = rot.to_matrix().to_4x4()
        mat_loc = Matrix.Translation(loc)
        m_target = mat_loc @ mat_rot
        m_targets.append(m_target)

    return m_targets


def apply_pose(arm, m_targets, node_names):
    for bone in arm.data.bones:
        bone_index = node_names.index(bone.name)

        m_target = m_targets[bone_index]
        if bone.name == "Base_bone":
            m_target = z_up_to_y_up.inverted() @ m_target

        p_bone = arm.pose.bones[bone.name]
        if bone.parent:
            m_rest_relative = bone.parent.matrix_local.inverted() @ bone.matrix_local
        else:
            m_rest_relative = bone.matrix_local

        # m_target = m_rest_relative @ matrix_basis
        p_bone.matrix_basis = m_rest_relative.inverted() @ m_target


def prepare_camera(world2camera):
    camera2world = (world2camera @ z_up_to_y_up).inverted()
    return camera2world


def get_transform_error_blender(T_pred, T_gt):
    t_pred = T_pred.to_translation()
    t_gt = T_gt.to_translation()
    trans_error = (t_pred - t_gt).length

    R_pred = T_pred.to_3x3()
    R_gt = T_gt.to_3x3()
    R_rel = R_gt.transposed() @ R_pred
    rot_error_rad = R_rel.to_quaternion().angle
    rot_error_deg = math.degrees(rot_error_rad)

    return trans_error, rot_error_deg


# --------------

scene = bpy.context.scene
scene.use_nodes = True
render_settings = scene.render
cam = scene.camera

cam_data = cam.data
cam_data.type = 'PERSP'

render_settings.engine = 'CYCLES'
scene.cycles.device = 'GPU'
# scene.cycles.device = 'CPU'
scene.cycles.samples = 1024
scene.cycles.use_denoising = False
# scene.cycles.denoiser = 'OPTIX'
scene.cycles.max_bounces = 6
scene.render.use_persistent_data = True
scene.cycles.use_tiling = False
scene.cycles.tile_size = 4096
scene.cycles.sample_clamp_indirect = 5.0
# scene.cycles.filter_glossy = 1.0
# scene.world.cycles.sample_as_light = True

view_layer = scene.view_layers[0]
view_layer.use_pass_z = False
view_layer.use_pass_normal = False
view_layer.use_pass_vector = False
view_layer.use_pass_uv = False
view_layer.use_pass_object_index = True

# Set the device_type
cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
cycles_prefs.compute_device_type = "OPTIX"
cycles_prefs.get_devices()

found_gpu = False
for d in cycles_prefs.devices:
    print(f"Found device: {d.name} ({d.type})", file=sys.stderr)
    if d.type == 'OPTIX':
        d.use = True
        found_gpu = True
        print(f"✅ Enabled OPTIX Device: {d.name}", file=sys.stderr)
    else:
        d.use = False

if not found_gpu:
    print("❌ NO OPTIX DEVICES FOUND! Check NVIDIA drivers.", file=sys.stderr)

print(f"Active Compute Device: {cycles_prefs.compute_device_type}", file=sys.stderr)

render_settings.use_compositing = True
render_settings.resolution_x = 512
render_settings.resolution_y = 512
render_settings.image_settings.file_format = 'PNG'
render_settings.image_settings.color_mode = 'RGB'
render_settings.film_transparent = False

tree = scene.node_tree
mask_output_node = next(n for n in tree.nodes if n.type == "OUTPUT_FILE" and n.label == "mask")
mask_output_node.base_path = str(masks_dir)
mask_output_node.format.file_format = 'PNG'
mask_output_node.format.color_mode = 'BW'

# TODO: During category unification, consider deriving node_names on the fly from the
# canonical .gltf mesh files (or loading per-category node_names alongside the mesh assets).
node_names = {}
for s in SEXES:
    with open(f"node_names_{s}.yaml") as f:
        node_names[s] = yaml.safe_load(f)

end = min(offset + piece, total)
for idx in tqdm(range(offset, end), file=sys.stderr):
    file_id = all_train_file_ids[idx]

    with (meta_dir / f"{file_id}_metadata.txt").open() as f:
        meta = yaml.safe_load(f)

    sex = MODEL_TO_SEX[meta["model_name"]]

    cam_data.lens = meta["focal_length"]
    camera_matrix_world_np = np.loadtxt(cameras_dir_in / f"{file_id}_camera.txt")
    camera_matrix_world = Matrix(camera_matrix_world_np)
    cam.matrix_world = camera_matrix_world

    active_obj_name = OBJ_NAMES[sex]
    active_obj = bpy.data.objects[active_obj_name]
    active_arm = bpy.data.objects[ARM_NAMES[sex]]
    active_obj.pass_index = 1

    pose_path = pose_dir_in / f"{file_id}_pose.{pose_ext}"
    pose = read_pose_blender(pose_path)
    apply_pose(active_arm, pose, node_names[sex])

    for s in SEXES:
        obj_name = OBJ_NAMES[s]
        arm_name = ARM_NAMES[s]
        obj = bpy.data.objects[obj_name]
        arm = bpy.data.objects[arm_name]

        obj.hide_render = not (s == sex)
        obj.hide_set(not (s == sex))
        arm.hide_render = not (s == sex)
        arm.hide_set(not (s == sex))

    texture_idx = sample_texture_idx(file_id, len(textures[sex]))
    set_texture(active_obj_name, textures[sex][texture_idx])

    hdr_idx = sample_texture_idx(file_id + "_hdr", len(hdrs))
    rot_deg = sample_texture_idx(file_id + "_rot", 360)
    set_world_texture(hdrs[hdr_idx], math.radians(rot_deg))

    render_scene(render_settings, mask_output_node, file_id)

    cam_to_world = np.array(cam.matrix_world)
    cam_path = cameras_dir / f"{file_id}_camera.txt"
    np.savetxt(cam_path, cam_to_world, fmt="%.6f")
