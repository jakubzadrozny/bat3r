# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.

import json
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.multiprocessing import Event, Process, Queue, set_start_method
from torch.utils import data
from tqdm import tqdm

from bat3r import bones
from bat3r.alt_dataset import _read_shape
from bat3r.bones import ShapeProcessingData
from bat3r.dataset import PointmapBatch
from bat3r.eval_utils import (
    DotDict,
    align_procrustes_new,
    chamfer_dist,
    decompose_transform,
    estimate_camera_pose_new,
    get_active_joint_indices,
    get_body_centric_camera_pose,
    get_transform_error,
    load_model,
)
from bat3r.icp import icp
from bat3r.pointmaps import extract_from_amodal_pointmap
from bat3r.refine_kinematic_chain import MeshFittingConfig, fit_mesh_to_pointcloud


def get_metrics_new(output: dict, cfg: DictConfig):
    output = DotDict(output)
    resolution = output.resolution

    canon_pc = output.canon[output.seq_mask]
    rec_pc = output.posed[output.seq_mask]
    n_points = rec_pc.shape[0]
    inlier_mask = output.inlier_mask[output.seq_mask]
    canonical_indices = output.canonical_indices[output.seq_mask][inlier_mask]

    dist_canon = torch.cdist(canon_pc, output.canon_gt_pc)
    canon_row_min = torch.argmin(dist_canon, dim=1)
    canon_col_min = torch.argmin(dist_canon, dim=0)

    cd_canon_1_rms = torch.mean(dist_canon[np.arange(dist_canon.shape[0]), canon_row_min].pow(2))
    cd_canon_2_rms = torch.mean(dist_canon[canon_col_min, np.arange(dist_canon.shape[1])].pow(2))
    cd_canon_rms = torch.sqrt((cd_canon_1_rms + cd_canon_2_rms) / 2.0)

    cd_canon_1 = torch.mean(dist_canon[np.arange(dist_canon.shape[0]), canon_row_min])
    cd_canon_2 = torch.mean(dist_canon[canon_col_min, np.arange(dist_canon.shape[1])])
    cd_canon = (cd_canon_1 + cd_canon_2) / 2.0

    corresp_gt_pc = output.posed_gt_pc[canon_row_min]

    scale_factor, rotation, translation = bones.kabsch_umeyama(
        rec_pc, corresp_gt_pc, scale=True, reflection=False, rotation=True,
    )
    rec_to_gt_transform = bones.to_transform(scale_factor, rotation, translation)
    rec_pc_aligned = bones.__transform(rec_to_gt_transform, rec_pc)
    _, cd_aligned_procr, cd_aligned_procr_rms = chamfer_dist(rec_pc_aligned, output.posed_gt_pc)
    rec_to_gt_scale, rec_to_gt_angle_deg, rec_to_gt_trans = decompose_transform(rec_to_gt_transform)

    rec_pc_icp, _ = icp(rec_pc_aligned, output.posed_gt_pc, optimize_rotation=True)
    _, cd_aligned_icp, cd_aligned_icp_rms = chamfer_dist(rec_pc_icp, output.posed_gt_pc)

    rec_pc_aligned_inliers = rec_pc_aligned[inlier_mask]
    rec_pc_icp_inliers, _ = icp(rec_pc_aligned_inliers, output.posed_gt_pc, optimize_rotation=True)
    _, cd_aligned_icp_inliers, cd_aligned_icp_rms_inliers = chamfer_dist(rec_pc_icp_inliers, output.posed_gt_pc)

    rec_pc_aligned_no_rot = align_procrustes_new(rec_pc, corresp_gt_pc, rotation=False)
    rec_pc_aligned_no_rot, _ = icp(rec_pc_aligned_no_rot, output.posed_gt_pc, optimize_rotation=False)
    dist_posed, cd, cd_rms = chamfer_dist(rec_pc_aligned_no_rot, output.posed_gt_pc)

    rel2world = output.rel2world
    joint_pos = output.joint_pos
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

    rec_pc_aligned = bones.__transform(output.rel2world_scaled, rec_pc)
    _, cd_fitted_to_rec, cd_fitted_to_rec_rms = chamfer_dist(rec_pc_aligned, output.fitted_shape_scaled)

    fitted_aligned_to_gt = align_procrustes_new(fitted_shape, output.posed_gt_pc)
    _, cd_fitted_to_gt, cd_fitted_to_gt_rms = chamfer_dist(fitted_aligned_to_gt, output.posed_gt_pc)

    fitted_aligned_to_can = align_procrustes_new(fitted_shape, output.canon_gt_pc)
    _, cd_fitted_to_can, cd_fitted_to_can_rms = chamfer_dist(fitted_aligned_to_can, output.canon_gt_pc)

    # FIND FITTED TO GT TRANSFORM
    _, rotation, translation = bones.kabsch_umeyama(
        fitted_shape, output.posed_gt_pc_world_space, scale=False, reflection=False, rotation=True,
    )
    fitted_to_gt_transform = bones.to_transform(1.0, rotation, translation)

    # ESTIMATE CAMERA
    world2cam_new, inlier_mask_new = estimate_camera_pose_new(
        posed_pointmap=output.posed_pointmap,
        occupancy_mask=output.occupancy_mask,
        world2rel_no_scale=world2rel_no_scale,
        focal_length=output.focal_length,
        resolution=resolution,
        scale=scale,
    )

    camera_mtx_orig_scale = world2cam_new.clone()
    camera_mtx_orig_scale[:3, 3] = camera_mtx_orig_scale[:3, 3] * (1 / scale)
    camera_mtx_aligned = camera_mtx_orig_scale @ torch.inverse(fitted_to_gt_transform)
    camera_trans_error, camera_rot_error_deg = get_transform_error(camera_mtx_aligned, output.model_view)

    joint_pos_dict = {}
    for joint_idx, node_idx in enumerate(output.shape.joints):
        joint_pos_dict[output.shape.node_names[node_idx]] = joint_pos[joint_idx]

    spherical_camera_pose = get_body_centric_camera_pose(
        cam2world=torch.inverse(world2cam_new),
        joint_pos_dict=joint_pos_dict,
        frame_cfg=cfg.body_centric_frame,
    )

    active_idx = get_active_joint_indices(
        len(output.canonical_joints), cfg.mesh_fitting.fixed_joints
    )
    canon_joints_subset = output.canonical_joints[active_idx]
    joints_pos_gt = output.joints_pos[active_idx]
    joints_pos_pred = joint_pos[active_idx]

    joints_aligned_can = align_procrustes_new(joints_pos_pred, canon_joints_subset, scale=False)
    joints_dist = torch.sum(torch.square(joints_aligned_can - canon_joints_subset))

    # PA-MPJPE (Procrustes Aligned Mean Per-Joint Position Error)
    joints_aligned_gt = align_procrustes_new(joints_pos_pred, joints_pos_gt, scale=False)
    mpjpe = torch.mean(torch.linalg.norm(joints_aligned_gt - joints_pos_gt, dim=-1))

    inliers = dict(
        mesh_fitting_inliers=torch.sum(inlier_mask).item(),
        mesh_fitting_inliers_frac=torch.mean(inlier_mask.to(torch.float32)).item(),
        new_cam_inliers=torch.sum(inlier_mask_new).item(),
        new_cam_inlier_frac=torch.mean(inlier_mask_new.to(torch.float32)).item(),
    )
    rec_to_gt = dict(
        rec_to_gt_scale=rec_to_gt_scale,
        rec_to_gt_angle_deg=rec_to_gt_angle_deg,
        rec_to_gt_trans=rec_to_gt_trans,
    )
    camera_errors = dict(
        camera_trans_error=camera_trans_error,
        camera_rot_error=camera_rot_error_deg,
    )
    return dict(
        file_id=output.file_id,
        cd_canon=cd_canon.item(),
        cd_canon_rms=cd_canon_rms.item(),
        cd=cd,
        cd_rms=cd_rms,
        cd_aligned_procr=cd_aligned_procr,
        cd_aligned_procr_rms=cd_aligned_procr_rms,
        cd_aligned_icp=cd_aligned_icp,
        cd_aligned_icp_rms=cd_aligned_icp_rms,
        cd_aligned_icp_inliers=cd_aligned_icp_inliers,
        cd_aligned_icp_rms_inliers=cd_aligned_icp_rms_inliers,
        cd_fitted_to_rec=cd_fitted_to_rec,
        cd_fitted_to_gt=cd_fitted_to_gt,
        cd_fitted_to_can=cd_fitted_to_can,
        cd_fitted_to_rec_rms=cd_fitted_to_rec_rms,
        cd_fitted_to_gt_rms=cd_fitted_to_gt_rms,
        cd_fitted_to_can_rms=cd_fitted_to_can_rms,
        joints_dist=joints_dist.item(),
        mpjpe=mpjpe.item(),
        n_points=n_points,
        **camera_errors,
        **inliers,
        **rec_to_gt,
        **spherical_camera_pose,
    )


class Producer(Process):
    def __init__(
        self,
        rank: int,
        input_q: Queue,
        output_q: Queue,
        cfg: DictConfig,
        data_root: Path,
        mesh_fitting_cfg: MeshFittingConfig,
        output_dir: Path | None = None,
    ):
        super().__init__()
        self.rank = rank
        self.input_q = input_q
        self.output_q = output_q
        self.cfg = cfg
        self.output_dir = output_dir
        self.data_root = data_root
        self.mesh_fitting_cfg = mesh_fitting_cfg

    def run(self):
        self.cfg.device = f"cuda:{self.rank}"
        torch.cuda.set_device(self.cfg.device)
        torch.set_num_threads(6)

        shape_root = self.data_root / "shapes"
        shapes = {}
        all_geo_dists = {}
        shape_proc_data = {}
        for p in shape_root.iterdir():
            if p.is_dir():
                shape_id = p.name
                shape = _read_shape(p / f"{p.stem}_shape.gltf")
                shapes[shape_id] = shape
                all_geo_dists[shape_id] = torch.from_numpy(
                    np.load(p / f"{p.stem}_geo_dists.npz")["dists"]
                ).to(dtype=torch.float32, device=self.cfg.device)
                shape_proc_data[shape_id] = ShapeProcessingData(shape, device=self.cfg.device)

        module = load_model(self.cfg)

        while True:
            batch = self.input_q.get()
            if batch is None:
                break

            batch = PointmapBatch(*batch)

            loss, output, batch = module.validation_step(batch, reduction="per-sample")
            masks = batch.mask
            file_ids = batch.data_id
            render_args = batch.render_args
            shape_id = render_args["shape_id"][0]
            shape_data = shape_proc_data[shape_id]
            canon, posed, seq_mask = module.predict(
                None,
                masks,
                confidence_threshold=self.cfg.confidence_threshold,
                output=output,
            )

            with torch.no_grad():
                eval_losses, canon_pointmap, posed_pointmap, occupancy = module.compute_eval_losses(
                    output, batch.model_targets, mask=batch.mask
                )

            occupancy_mask = occupancy.squeeze(4) * masks[..., None]
            # TO INCLUDE CONFIDENCE:   
            # if confidence_threshold is not None:
            #     occupancy_mask = occupancy_mask * (confidence >= confidence_threshold).to(torch.float32)
            occupancy_mask = (occupancy_mask > 0.5).to(dtype=torch.float32, device=occupancy.device)

            _, _, _, confidence = extract_from_amodal_pointmap(
                output, module.num_layers, activate_occupancy=True
            )
            confidence = confidence.squeeze(4)
            per_layer_confidence = (confidence * occupancy_mask).sum((1, 2)) / occupancy_mask.sum((1, 2))
            per_sample_confidence = (confidence * occupancy_mask).sum((1, 2, 3)) / occupancy_mask.sum((1, 2, 3))

            resolution = canon_pointmap.shape[1]

            eval_losses["confidence_estim"] = per_sample_confidence
            eval_losses["per_layer_confidence_estim"] = per_layer_confidence

            mesh_fit_res = fit_mesh_to_pointcloud(
                cfg=self.mesh_fitting_cfg,
                predicted_canon=canon,
                predicted_pose=posed,
                seq_mask=seq_mask,
                mesh_proc_data=shape_data,
                geo_dists=all_geo_dists[shape_id],
                return_metrics=False,
            )
            rel2world = mesh_fit_res["global_transform"]
            fitted_shape = mesh_fit_res["fitted_shape"]
            rel2world_scaled = mesh_fit_res["global_transform_scaled"]
            fitted_shape_scaled = mesh_fit_res["fitted_shape_scaled"]
            joint_pos = mesh_fit_res["joint_pos"]
            joint_angles_rad = mesh_fit_res["joint_angles"]
            inlier_mask = mesh_fit_res["inlier_mask"]
            canonical_indices = mesh_fit_res["canonical_indices"]

            for bidx in range(len(file_ids)):
                if render_args["shape_id"][bidx] != shape_id:
                    print(f"shape_id mismatch {shape_id} != {render_args['shape_id'][bidx]}")
                args = dict(
                    resolution=resolution,
                    file_id=file_ids[bidx],
                    vis=render_args["vis"][bidx],
                    verts_ndc=render_args["verts_ndc"][bidx].to(posed.device),
                    canon_pointmap=canon_pointmap[bidx],
                    posed_pointmap=posed_pointmap[bidx],
                    occupancy_mask=occupancy_mask[bidx],
                    canon=canon[bidx],
                    posed=posed[bidx],
                    seq_mask=seq_mask[bidx],
                    canon_gt_pc=render_args["canonical_verts"][bidx].to(posed.device),
                    posed_gt_pc=render_args["posed_verts_camera_space"][bidx].to(posed.device),
                    posed_gt_pc_world_space=render_args["pose_verts"][bidx].to(posed.device),
                    shape=shapes[shape_id],
                    focal_length=render_args["focal_length"][bidx].item(),
                    model_view=render_args["model_view"][bidx].to(posed.device),
                    joints_pos=render_args["joint_positions"][bidx].to(posed.device),
                    canonical_joints=shape_data.canonical_joints,
                    rel2world=rel2world[bidx],
                    fitted_shape=fitted_shape[bidx],
                    rel2world_scaled=rel2world_scaled[bidx],
                    fitted_shape_scaled=fitted_shape_scaled[bidx],
                    joint_pos=joint_pos[bidx],
                    inlier_mask=inlier_mask[bidx],
                    canonical_indices=canonical_indices[bidx],
                )
                metrics = get_metrics_new(args, cfg=self.cfg, output_dir=self.output_dir)
                metrics.update(dict(
                    joint_rots_angles_rad=torch.mean(joint_angles_rad[bidx]).item(),
                    joint_rots_angles_rad_sq=torch.mean(torch.square(joint_angles_rad[bidx])).item(),
                ))

                metrics["loss"] = loss[bidx].item()
                for key in ["canon_loss", "posed_loss", "occupancy_loss", "confidence", "confidence_estim"]:
                    metrics[key] = eval_losses[key][bidx].item()
                    for idx, val in enumerate(eval_losses[f"per_layer_{key}"][bidx]):
                        metrics[f"{key}_{idx}"] = val.item()

                self.output_q.put(metrics)


def producer_wrapper(loaders, queue, event):
    for loader in loaders:
        for batch in loader:
            queue.put(batch)
    while not event.is_set():
        time.sleep(0.5)


@hydra.main(config_path="../configs", config_name="infer", version_base="1.3")
def main(cfg: DictConfig):
    set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")

    n_gpus = torch.cuda.device_count()
    num_workers = max(1, n_gpus)

    input_q = Queue(maxsize=4 * num_workers)
    output_q = Queue(maxsize=4 * num_workers)
    stop_producer = Event()

    log_dir = Path(cfg.log_dir)
    weights_path = log_dir / cfg.ckpt_name
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    dataset = hydra.utils.instantiate(cfg.dataset)
    dataset_root = Path(cfg.dataset_root)

    if cfg.split == "train":
        train_ids_txt = dataset_root / "train_ids-orig.txt"
        train_ids_json = dataset_root / "train_ids.json"
        if train_ids_txt.exists():
            with train_ids_txt.open() as f:
                test_ids = set(file_id.strip() for file_id in f.readlines())
        elif train_ids_json.exists():
            with train_ids_json.open() as f:
                test_ids = set(json.load(f))
        else:
            raise FileNotFoundError(f"Neither train_ids-orig.txt nor train_ids.json found in {dataset_root}")
    elif cfg.split == "test":
        test_ids_file = dataset_root / "test_ids.json"
        with test_ids_file.open() as f:
            test_ids = set(json.load(f))
    else:
        raise ValueError(f"Unknown split: {cfg.split}")
    print(f"Infer on {len(test_ids)} IDs")

    mesh_fitting_cfg = MeshFittingConfig(**OmegaConf.to_container(cfg.mesh_fitting, resolve=True))

    pose_csv_path = dataset_root / "spherical_camera_pose-2.csv"
    loaders = []
    num_samples = 0
    if pose_csv_path.exists():
        dataset_df = pd.read_csv(pose_csv_path, dtype={"file_id": "string"})
        dataset_df.set_index("file_id", inplace=True)
        shape_ids = dataset_df["shape_id"].unique().tolist()
        for shape_id in shape_ids:
            shape_ind = [
                idx
                for idx in range(len(dataset))
                if dataset.ids[idx] in test_ids and dataset_df.loc[dataset.ids[idx], "shape_id"] == shape_id
            ]
            sub_dataset = data.Subset(dataset, shape_ind)
            print(f"{shape_id}: {len(shape_ind)}")
            loader = hydra.utils.instantiate(
                cfg.dataloader, dataset=sub_dataset, collate_fn=dataset.collate_fn
            )
            loaders.append(loader)
            num_samples += len(shape_ind)
        if num_samples != len(test_ids):
            raise ValueError(
                f"Expected {len(test_ids)} samples for split '{cfg.split}', found {num_samples}"
            )
    else:
        test_ind = [
            idx
            for idx in range(len(dataset))
            if dataset.ids[idx].split("_")[0] in test_ids
        ]
        found_ids = {dataset.ids[idx].split("_")[0] for idx in test_ind}
        missing_ids = test_ids - found_ids
        if missing_ids:
            raise ValueError(
                f"Missing {len(missing_ids)}/{len(test_ids)} IDs from dataset for split '{cfg.split}': "
                f"{sorted(missing_ids)[:5]}"
            )
        sub_dataset = data.Subset(dataset, test_ind)
        print(f"test ind: {len(test_ind)}")
        loader = hydra.utils.instantiate(
            cfg.dataloader, dataset=sub_dataset, collate_fn=dataset.collate_fn
        )
        loaders.append(loader)
        num_samples = len(test_ind)

    producer_proc = Process(target=producer_wrapper, args=(loaders, input_q, stop_producer))
    producer_proc.start()

    processes = [
        Producer(
            rank=idx,
            input_q=input_q,
            output_q=output_q,
            cfg=cfg,
            data_root=dataset.root,
            mesh_fitting_cfg=mesh_fitting_cfg,
        )
        for idx in range(num_workers)
    ]
    for proc in processes:
        proc.start()

    rows = []
    for _ in tqdm(range(num_samples)):
        row = output_q.get()
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.mean(numeric_only=True))
    if cfg.split == "test":
        df.to_csv(log_dir / f"{weights_path.stem}.csv", index=False)
    elif cfg.split == "train":
        df.to_csv(log_dir / f"{weights_path.stem}-train.csv", index=False)

    for _ in range(num_workers):
        input_q.put(None)
    stop_producer.set()

    for proc in processes + [producer_proc]:
        proc.join()


if __name__ == "__main__":
    main()
