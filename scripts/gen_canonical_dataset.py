# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.

from pathlib import Path
import yaml

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from bat3r.alt_dataset import _read_shape
from bat3r.bones import ShapeProcessingData
from bat3r.skin import OneMeshGltf
from bat3r.eval_utils import (
    get_camera_matrix_from_spherical,
    sample_view,
)
from bat3r.utils import to_blender_camera


def write_sample(
    file_id: str,
    shape_id: str,
    shape: OneMeshGltf,
    shape_data: ShapeProcessingData,
    view_params: dict,
    focal_length: float,
    frame_cfg: DictConfig,
    output_dir: Path,
):
    joint_pos_dict = {}
    for joint_idx, node_idx in enumerate(shape.joints):
        joint_pos_dict[shape.node_names[node_idx]] = shape_data.canonical_joints[joint_idx]

    world2cam, _ = get_camera_matrix_from_spherical(
        azimuth=view_params["azimuth"],
        elevation=view_params["elevation"],
        roll=view_params["roll"],
        joint_pos_dict=joint_pos_dict,
        target_points=shape_data.vertices,
        focal_length=focal_length,
        frame_cfg=frame_cfg,
        margin=view_params["margin"],
    )

    cam_pose_blender = to_blender_camera(world2cam.cpu().numpy())
    np.savetxt(output_dir / "cameras_in" / f"{file_id}_camera.txt", cam_pose_blender, fmt="%.6f")
    np.save(
        output_dir / "poses" / f"{file_id}_pose.npy",
        shape_data.local_to_parent_transforms.cpu().numpy(),
    )

    meta = {
        "model_name": shape_id,
        "frame": 0,
        "action": "canonical",
        "focal_length": float(focal_length),
    }
    meta_path = output_dir / "metadata" / f"{file_id}_metadata.txt"
    with meta_path.open("w+") as meta_f:
        yaml.dump(meta, meta_f, sort_keys=False)


@hydra.main(config_path="../configs", config_name="real", version_base="1.3")
def main(cfg: DictConfig):
    log_dir = Path(cfg.log_dir)
    output_dir = log_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cameras_in").mkdir(exist_ok=True)
    (output_dir / "poses").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)
    print("output_dir:", str(output_dir))

    np.random.seed(42)
    torch.manual_seed(42)

    shape_id = cfg.shape_id
    shape_dir = Path(cfg.dataset_root) / "shapes" / shape_id
    shape = _read_shape(shape_dir / f"{shape_id}_shape.gltf")
    shape_data = ShapeProcessingData(shape, device="cpu")

    num_samples = cfg.gen_canonical_dataset.num_samples
    f_min, f_max = cfg.gen_canonical_dataset.focal_length_range

    for idx in tqdm(range(num_samples)):
        view_params = sample_view(view_cfg=cfg.view_sampling)
        file_id = f"{idx:06d}"
        focal_length = float(np.random.uniform(f_min, f_max))
        write_sample(
            file_id=file_id,
            shape_id=shape_id,
            shape=shape,
            shape_data=shape_data,
            view_params=view_params,
            focal_length=focal_length,
            frame_cfg=cfg.body_centric_frame,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
