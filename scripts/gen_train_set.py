import json
import time
import yaml
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.multiprocessing import Process, Queue, set_start_method, Event
from torch.utils import data
from tqdm import tqdm

from bat3r import bones
from bat3r.bones import ShapeProcessingData
from bat3r.eval_utils import (
    DotDict,
    align_procrustes_new,
    chamfer_dist,
    estimate_camera_pose_new_with_focal,
    get_view_type,
    load_model,
    get_body_centric_camera_pose,
    sample_view,
    get_camera_matrix_from_spherical,
)
from bat3r.alt_dataset import _read_shape
from bat3r.refine_kinematic_chain import MeshFittingConfig, fit_mesh_to_pointcloud
from bat3r.dataset import PointmapBatch
from bat3r.pointmaps import extract_from_amodal_pointmap
from bat3r.utils import to_blender_camera


def get_metrics_new(output: dict, cfg: DictConfig, output_dir: Path | None = None):
    output = DotDict(output)

    seed = int(output.file_id)
    np.random.seed(seed)
    torch.manual_seed(seed)

    resolution = output.resolution

    canon_pc = output.canon[output.seq_mask]
    rec_pc = output.posed[output.seq_mask]
    inlier_mask = output.inlier_mask[output.seq_mask]
    canonical_indices = output.canonical_indices[output.seq_mask][inlier_mask]

    dist_canon = torch.cdist(canon_pc, output.canon_gt_pc)
    canon_row_min = torch.argmin(dist_canon, dim=1)
    canon_col_min = torch.argmin(dist_canon, dim=0)

    cd_canon_1 = torch.mean(dist_canon[np.arange(dist_canon.shape[0]), canon_row_min])
    cd_canon_2 = torch.mean(dist_canon[canon_col_min, np.arange(dist_canon.shape[1])])
    cd_canon = 0.5 * (cd_canon_1 + cd_canon_2)

    rel2world = output.rel2world
    joint_pos = output.joint_pos
    joint_pos_sym = output.joint_pos_sym
    fitted_shape = output.fitted_shape
    _, rotation, translation = bones.kabsch_umeyama(
        fitted_shape[canonical_indices],
        rec_pc[inlier_mask],
        scale=False,
        reflection=False,
        rotation=True,
    )
    world2rel_no_scale = bones.to_transform(1.0, rotation, translation)
    scale = torch.linalg.norm(rel2world[:3, 0]).item()

    rec_pc_aligned_scaled = bones.__transform(output.rel2world_scaled, rec_pc)
    cd_fitted_to_rec = chamfer_dist(rec_pc_aligned_scaled, output.fitted_shape_scaled)[1]
    
    fitted_aligned_to_can = align_procrustes_new(fitted_shape, output.canon_gt_pc)
    cd_fitted_to_can = chamfer_dist(fitted_aligned_to_can, output.canon_gt_pc)[1]

    # ESTIMATE CAMERA
    world2cam_new, focal_length, inlier_mask_new = estimate_camera_pose_new_with_focal(
        posed_pointmap=output.posed_pointmap,
        occupancy_mask=output.occupancy_mask,
        world2rel_no_scale=world2rel_no_scale,
        resolution=resolution,
        scale=scale,
    )

    joint_pos_dict = {}
    joint_pos_dict_sym = {}
    for joint_idx, node_idx in enumerate(output.shape.joints):
        joint_pos_dict[output.shape.node_names[node_idx]] = joint_pos[joint_idx]
        joint_pos_dict_sym[output.shape.node_names[node_idx]] = joint_pos_sym[joint_idx]

    spherical_camera_pose = get_body_centric_camera_pose(
        cam2world=torch.inverse(world2cam_new),
        joint_pos_dict=joint_pos_dict,
        frame_cfg=cfg.body_centric_frame,
    )
    view_type = get_view_type(spherical_camera_pose["azimuth"])

    # Exclude fixed joints (handling negative indices)
    num_joints = len(output.shape_data.canonical_joints)
    fixed_set = {
        j if j >= 0 else num_joints + j
        for j in cfg.mesh_fitting.fixed_joints
    }
    active_idx = [i for i in range(num_joints) if i not in fixed_set]
    canon_subset = output.shape_data.canonical_joints[active_idx]
    posed_subset = joint_pos[active_idx]

    joints_aligned = align_procrustes_new(posed_subset, canon_subset, scale=False)
    joints_dist = torch.sum(torch.square(joints_aligned - canon_subset))

    # SAVE ALL
    if cfg.gen_train_set.include_estimated_camera:
        cam_poses = [world2cam_new]
        cam_pose_types = ["estim_new"]
        focal_lengths = [focal_length.item() if isinstance(focal_length, torch.Tensor) else focal_length]
        is_sym_list = [False]
    else:
        cam_poses = []
        cam_pose_types = []
        focal_lengths = []
        is_sym_list = []

    if cfg.gen_train_set.num_views_by_type is not None:
        if view_type == "front":
            n_views = cfg.gen_train_set.num_views_by_type.front
        elif view_type == "rear":
            n_views = cfg.gen_train_set.num_views_by_type.rear
        else:
            n_views = cfg.gen_train_set.num_views_by_type.other
    else:
        n_views = cfg.gen_train_set.num_views

    for _ in range(n_views):
        view_params = sample_view(view_cfg=cfg.view_sampling)
        other_view_type = get_view_type(view_params["azimuth"])
        sample_focal_length = np.random.uniform(85, 200)

        is_sym = bool(np.random.rand() < 0.5)
        curr_fitted_shape = output.fitted_shape_sym if is_sym else fitted_shape
        curr_joint_pos_dict = joint_pos_dict_sym if is_sym else joint_pos_dict

        world2cam_other, _ = get_camera_matrix_from_spherical(
            azimuth=view_params["azimuth"],
            elevation=view_params["elevation"],
            roll=view_params["roll"],
            joint_pos_dict=curr_joint_pos_dict,
            target_points=curr_fitted_shape,
            focal_length=sample_focal_length,
            margin=view_params["margin"],
            frame_cfg=cfg.body_centric_frame,
        )
        cam_poses.append(world2cam_other)
        cam_pose_types.append(other_view_type)
        focal_lengths.append(sample_focal_length)
        is_sym_list.append(is_sym)
        
    if output_dir is not None:
        for idx, (cam_pose, f_len, is_sym) in enumerate(zip(cam_poses, focal_lengths, is_sym_list)):
            cam_pose_blender = to_blender_camera(cam_pose.cpu().numpy())
            curr_transforms = output.local_joint_transforms_sym if is_sym else output.local_joint_transforms
            np.save(output_dir / "poses" / f"{output.file_id}_{idx:02d}_pose.npy", curr_transforms.cpu().numpy())
            np.savetxt(output_dir / "cameras_in" / f"{output.file_id}_{idx:02d}_camera.txt", cam_pose_blender, fmt="%.6f")
            meta = {
                "model_name": output.shape_id,
                "focal_length": float(f_len),
                "is_sym": is_sym,
            }
            meta_path = output_dir / "metadata" / f"{output.file_id}_{idx:02d}_metadata.txt"
            with meta_path.open("w+") as meta_f:
                yaml.dump(meta, meta_f, sort_keys=False)

    inliers = dict(
        mesh_fitting_inliers=torch.sum(inlier_mask).item(),
        mesh_fitting_inliers_frac=torch.mean(inlier_mask.to(torch.float32)).item(),
        new_cam_inliers=torch.sum(inlier_mask_new).item(),
        new_cam_inlier_frac=torch.mean(inlier_mask_new.to(torch.float32)).item(),
    )
    return dict(
        file_id=output.file_id,
        cd_canon=cd_canon.item(),
        cd_fitted_to_rec=cd_fitted_to_rec,
        cd_fitted_to_can=cd_fitted_to_can,
        joints_dist=joints_dist.item(),
        n_views=len(cam_poses),
        view_types=','.join(cam_pose_types),
        **inliers,
        **spherical_camera_pose,
    )


class Producer(Process):
    def __init__(
        self,
        rank: int,
        input_q: Queue,
        output_q: Queue,
        cfg: DictConfig,
        mesh_fitting_cfg: MeshFittingConfig,
        data_root: Path,
        output_dir: Path | None = None,
    ):
        super().__init__()
        self.rank = rank
        self.input_q = input_q
        self.output_q = output_q
        self.cfg = cfg
        self.mesh_fitting_cfg = mesh_fitting_cfg
        self.output_dir = output_dir
        self.data_root = data_root

    def run(self):
        self.cfg.device = f"cuda:{self.rank}"
        torch.cuda.set_device(self.cfg.device)
        torch.set_num_threads(6)
        
        shape_id = self.cfg.shape_id

        shape_root = self.data_root / "shapes"
        shapes = {}
        all_geo_dists = {}
        shape_proc_data = {}
        for p in shape_root.iterdir():
            if p.is_dir() and p.name == shape_id:
                shape = _read_shape(p / f"{p.stem}_shape.gltf")
                shapes[shape_id] = shape
                all_geo_dists[shape_id] = torch.from_numpy(
                    np.load(p / f"{p.stem}_geo_dists.npz")['dists']
                ).to(dtype=torch.float32, device=self.cfg.device)
                shape_proc_data[shape_id] = ShapeProcessingData(shape, device=self.cfg.device)

        shape = shapes[shape_id]
        shape_data = shape_proc_data[shape_id]

        module = load_model(self.cfg)

        while True:
            batch = self.input_q.get()
            # Exit the process
            if batch is None:
                break

            batch = PointmapBatch(*batch)
            batch.to(self.cfg.device)

            masks = batch.mask
            file_ids = batch.data_id

            with torch.inference_mode():
                output = module.model(batch.input_image)

            canon, posed, seq_mask = module.predict(
                None,
                masks,
                confidence_threshold=self.cfg.confidence_threshold,
                output=output,
            )

            canon_pointmap, posed_pointmap, occupancy, confidence = extract_from_amodal_pointmap(
                output, module.num_layers, activate_occupancy=True
            )
            occupancy_mask = occupancy.squeeze(4) * masks[..., None]
            occupancy_mask = (occupancy_mask > 0.5).to(dtype=torch.float32, device=occupancy.device)

            confidence = confidence.squeeze(4)
            per_layer_confidence = (confidence * occupancy_mask).sum((1, 2)) / occupancy_mask.sum((1, 2))
            per_sample_confidence = (confidence * occupancy_mask).sum((1, 2, 3)) / occupancy_mask.sum((1, 2, 3))

            resolution = canon_pointmap.shape[1]

            mesh_fit_res = fit_mesh_to_pointcloud(
                cfg=self.mesh_fitting_cfg,
                predicted_canon=canon,
                predicted_pose=posed,
                seq_mask=seq_mask,
                mesh_proc_data=shape_data,
                geo_dists=all_geo_dists[shape_id],
                return_metrics=True,
            )
            rel2world = mesh_fit_res['global_transform']
            local_joint_transforms = mesh_fit_res['local_joint_transforms']
            fitted_shape = mesh_fit_res['fitted_shape']
            joint_pos = mesh_fit_res['joint_pos']
            joint_angles_rad = mesh_fit_res['joint_angles']
            inlier_mask = mesh_fit_res['inlier_mask']
            canonical_indices = mesh_fit_res['canonical_indices']
            fitting_metrics = mesh_fit_res['metrics']

            local_joint_transforms_sym, fitted_shape_sym, joint_pos_sym = bones.mirror_local_joint_transforms(local_joint_transforms, shape_data)

            for bidx in range(len(file_ids)):
                args = dict(
                    resolution=resolution,
                    file_id=file_ids[bidx],
                    canon_pointmap=canon_pointmap[bidx],
                    posed_pointmap=posed_pointmap[bidx],
                    occupancy_mask=occupancy_mask[bidx],
                    canon=canon[bidx],
                    posed=posed[bidx],
                    seq_mask=seq_mask[bidx],
                    canon_gt_pc=shape_data.vertices,
                    shape=shape,
                    shape_data=shape_data,
                    shape_id=shape_id,
                    rel2world=rel2world[bidx],
                    local_joint_transforms=local_joint_transforms[bidx],
                    local_joint_transforms_sym=local_joint_transforms_sym[bidx],
                    fitted_shape=fitted_shape[bidx],
                    rel2world_scaled=mesh_fit_res['global_transform_scaled'][bidx],
                    fitted_shape_scaled=mesh_fit_res['fitted_shape_scaled'][bidx],
                    fitted_shape_sym=fitted_shape_sym[bidx],
                    joint_pos=joint_pos[bidx],
                    joint_pos_sym=joint_pos_sym[bidx],
                    inlier_mask=inlier_mask[bidx],
                    canonical_indices=canonical_indices[bidx],
                )
                metrics = get_metrics_new(args, output_dir=self.output_dir, cfg=self.cfg)
                metrics.update(dict(
                    joint_rots_angles_rad=torch.mean(joint_angles_rad[bidx]).item(),
                    joint_rots_angles_rad_sq=torch.mean(torch.square(joint_angles_rad[bidx])).item(),
                ))
                metrics["confidence"] = per_sample_confidence[bidx].item()
                for idx, val in enumerate(per_layer_confidence[bidx]):
                    metrics[f"confidence_{idx}"] = val.item()

                for k, v in fitting_metrics.items():
                    metrics[f"fitting_{k}"] = v[bidx].item()

                self.output_q.put(metrics)


def producer_wrapper(loader, queue, event):
    for batch in loader:
        queue.put(batch)
    while not event.is_set():
        time.sleep(0.5)


@hydra.main(config_path="../configs", config_name="real", version_base="1.3")
def main(cfg: DictConfig):
    set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    n_gpus = torch.cuda.device_count()

    input_q = Queue(maxsize=4 * max(1, n_gpus))
    output_q = Queue(maxsize=4 * max(1, n_gpus))
    stop_producer = Event()

    log_dir = Path(cfg.log_dir)
    weights_path = log_dir / cfg.ckpt_name
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    refinement_iter = cfg.get("refinement_iter", 1)
    mesh_fitting_cfg = MeshFittingConfig(**OmegaConf.to_container(cfg.mesh_fitting, resolve=True))

    print(f"Refinement Iter: {refinement_iter}")
    print(f"Mesh Fitting Config: {mesh_fitting_cfg}")

    dataset = hydra.utils.instantiate(cfg.dataset)

    train_ids_path = Path(cfg.dataset_root) / "train_ids.json"
    with train_ids_path.open() as f:
        train_ids = set(json.load(f))

    output_dir = log_dir / "output"
    if not output_dir.is_symlink():
        output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "poses").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)
    (output_dir / "cameras_in").mkdir(exist_ok=True)
    print("output_dir:", str(output_dir))

    # Process all train samples
    train_ind = [idx for idx in range(len(dataset)) if dataset.ids[idx] in train_ids]
    train_dataset = data.Subset(dataset, train_ind)

    loader = hydra.utils.instantiate(cfg.dataloader, dataset=train_dataset, collate_fn=dataset.collate_fn)

    producer_proc = Process(target=producer_wrapper, args=(loader, input_q, stop_producer))
    producer_proc.start()

    num_workers = max(1, n_gpus)
    processes = [
        Producer(
            rank=idx,
            input_q=input_q,
            output_q=output_q,
            cfg=cfg,
            mesh_fitting_cfg=mesh_fitting_cfg,
            output_dir=output_dir,
            data_root=dataset.root,
        )
        for idx in range(num_workers)
    ]
    for proc in processes:
        proc.start()

    rows = []
    for _ in tqdm(range(len(train_dataset))):
        row = output_q.get()
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.mean(numeric_only=True))
    df.to_csv(output_dir / f"{weights_path.stem}-train.csv", index=False)
    df.to_csv(output_dir / "train.csv", index=False)

    for _ in range(num_workers):
        input_q.put(None)
    stop_producer.set()

    for proc in processes + [producer_proc]:
        proc.join()


if __name__ == "__main__":
    main()
