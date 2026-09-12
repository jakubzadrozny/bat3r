import time
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from contextlib import contextmanager
from scipy import stats
from scipy.optimize import least_squares

import bat3r.raster.utils as ut
from bat3r import bones
from bat3r.pointmaps import PointmapModule


class NoCandidatesError(Exception):
    """Exception raised when no valid Tz candidates are found during camera pose estimation."""
    pass


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_model(cfg) -> PointmapModule:
    weights_path = Path(cfg.log_dir) / cfg.ckpt_name
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    module: PointmapModule = hydra.utils.instantiate(cfg.module, device=cfg.device)

    # map to CPU
    state_dicts = torch.load(weights_path, map_location="cpu")
    if "model_state" in state_dicts:
        model_weight_dict = state_dicts["model_state"]
    else:
        model_weight_dict = state_dicts

    module.model.load_state_dict(
        model_weight_dict,
        strict=False,
    )

    module.model.to(cfg.device)
    module.model.eval()
    module.device = cfg.device
    return module


def chamfer_dist(X, Y):
    dist = torch.cdist(X, Y)
    cost_X_to_Y = torch.amin(dist, dim=1)
    cost_Y_to_X = torch.amin(dist, dim=0)
    cd = (cost_X_to_Y.mean() + cost_Y_to_X.mean()) / 2.0
    cd_sq = (cost_X_to_Y.pow(2).mean() + cost_Y_to_X.pow(2).mean()) / 2.0
    return dist, cd.item(), torch.sqrt(cd_sq).item()


def align_procrustes_new(X, Y, rotation=True, scale=True):
    scale_factor, rotation, translation = bones.kabsch_umeyama(
        X, Y, scale=scale, reflection=False, rotation=rotation,
    )
    transform = bones.to_transform(scale_factor, rotation, translation)
    X_transf = bones.__transform(transform, X)
    return X_transf


def scale_relative_to(X, centroid, scale):
    return (X - centroid) * scale + centroid


def scale_in_place(X, scale):
    assert X.ndim == 2
    centroid = torch.mean(X, dim=0)
    return (X - centroid) * scale + centroid


def get_matching_pixel(
    canon_pointmap: torch.Tensor,
    occupancy_mask: torch.Tensor,
    canon_verts: torch.Tensor,
    layers: list[int] = [0],
    match_dist_threshold: float = 0.2,
    inf_dist: float = 10000,
):
    canon_pointmap = canon_pointmap[..., layers, :]
    occupancy_mask = occupancy_mask[..., layers, None]
    dist = (
        torch.cdist(canon_pointmap, canon_verts) + (1.0 - occupancy_mask) * inf_dist
    )
    dist = torch.amin(dist, dim=-2)
    
    pix_match_dist, pix_match = torch.min(dist.flatten(end_dim=1), dim=-1)
    matched_verts = torch.unique(pix_match[pix_match_dist < match_dist_threshold])

    reduced_dist, best_v = torch.min(dist, dim=0)
    match_dist, best_u = torch.min(reduced_dist, dim=0)
    best_v = best_v[best_u, torch.arange(best_v.shape[1])]
    matching_pixel = torch.stack((best_u, best_v), dim=-1).to(torch.float32)
    return matched_verts, matching_pixel


def get_matching_pixel_batched(
    canon_pointmap: torch.Tensor,
    occupancy_mask: torch.Tensor,
    canon_verts: torch.Tensor,
    layers: list[int] = [0],
    match_dist_threshold: float = 0.2,
    inf_dist: float = 10000,
):
    B, H, W, _, _ = canon_pointmap.shape
    V = canon_verts.shape[1]
    
    canon_pointmap = canon_pointmap[..., layers, :]
    occupancy_mask = occupancy_mask[..., layers, None]
    
    # Flatten spatial dims but keep layers for reduction
    # (B, H*W, L, 3)
    pts = canon_pointmap.flatten(1, 2)
    # (B, H*W, L, 1)
    occ = occupancy_mask.flatten(1, 2)
    
    # We need cdist between (B, H*W*L, 3) and (B, V, 3) because cdist doesn't support 4D
    pts_flat = pts.flatten(1, 2) # (B, H*W*L, 3)
    
    dist = torch.cdist(pts_flat, canon_verts) # (B, H*W*L, V)
    
    # Reshape back to (B, H*W, L, V)
    dist = dist.view(B, H*W, -1, V)
    
    # Add inf_dist
    dist = dist + (1.0 - occ) * inf_dist
    
    # Min over L -> (B, H*W, V)
    dist = torch.amin(dist, dim=2)
    
    # 1. Find matched vertices
    # Min over V -> (B, H*W)
    pix_match_dist, pix_match = torch.min(dist, dim=-1)
    
    valid_match = pix_match_dist < match_dist_threshold
    
    src = torch.ones_like(pix_match, dtype=torch.float32)
    src[~valid_match] = 0
    
    matched_counts = torch.zeros((B, V), dtype=torch.float32, device=dist.device)
    matched_counts.scatter_add_(1, pix_match, src)
    matched_verts_mask = matched_counts > 0
    
    # 2. Find matching pixel for each vertex
    # dist is (B, H*W, V). We need (B, H, W, V) to get u, v.
    dist = dist.view(B, H, W, V)
    
    # Min over H -> (B, W, V)
    val_h, idx_h = torch.min(dist, dim=1)
    # Min over W -> (B, V)
    val_w, idx_w = torch.min(val_h, dim=1)
    
    best_u = idx_w # (B, V)
    best_v = torch.gather(idx_h, 1, best_u.unsqueeze(1)).squeeze(1) # (B, V)
    
    matching_pixel = torch.stack((best_u, best_v), dim=-1).to(torch.float32)
    
    return matched_verts_mask, matching_pixel


def get_matching_err(
    matching_pixel: torch.Tensor,
    verts_ndc: torch.Tensor,
    vis_mask: torch.Tensor,
    resolution: int,
) -> tuple[float, float]:
    gt_uv_coords = resolution * 0.5 * (verts_ndc[:, :2] + 1)
    matching_err = torch.sqrt(torch.sum((gt_uv_coords - matching_pixel)**2, dim=-1))
    matching_err_all = torch.mean(matching_err)
    matching_err_vis = torch.mean(matching_err[vis_mask])
    return matching_err_all.item(), matching_err_vis.item()


def get_matching_err_batched(
    matching_pixel: torch.Tensor,
    verts_ndc: torch.Tensor,
    vis_mask: torch.Tensor,
    resolution: int,
):
    gt_uv_coords = resolution * 0.5 * (verts_ndc[..., :2] + 1)
    matching_err = torch.sqrt(torch.sum((gt_uv_coords - matching_pixel)**2, dim=-1))
    
    matching_err_all = torch.mean(matching_err, dim=1)
    
    vis_mask = vis_mask.bool()
    vis_err_sum = (matching_err * vis_mask).sum(dim=1)
    vis_count = vis_mask.sum(dim=1).float()
    matching_err_vis = vis_err_sum / (vis_count + 1e-8)
    
    return matching_err_all, matching_err_vis


def get_grid(res, device):
    u = torch.arange(res, device=device)
    v = torch.arange(res, device=device)
    V, U = torch.meshgrid(v, u)
    grid = torch.stack((U, V), dim=-1)
    return grid


def count_inliers(
    points_3d: torch.Tensor,
    points_2d: torch.Tensor,
    world2cam: torch.Tensor,
    resolution: int,
    focal_length: float,
    threshold: float = 5.0, # In pixels
):
    projected_pts = project_points(
        points_3d=points_3d,
        world2cam=world2cam,
        focal_length=focal_length,
        resolution=resolution,
    )
    err = torch.sum(torch.square(projected_pts - points_2d), dim=-1)
    inlier_mask = err < (threshold**2)
    return inlier_mask


def estimate_camera_pose(
    verts_3d: torch.Tensor,
    matched_verts: torch.Tensor,
    matching_pixel: torch.Tensor,
    resolution: int,
    focal_length: float,
    num_corresp: int = 50,
    sensor_height: float = 36.0,
):
    device = verts_3d.device
    sampled_verts_ind = torch.randperm(matched_verts.shape[0], device=device)[:num_corresp]
    sampled_verts = matched_verts[sampled_verts_ind]
    correspondences_3d = verts_3d[sampled_verts].cpu().numpy()
    correspondences_2d = matching_pixel[sampled_verts].cpu().numpy()

    focal_length_scaled = resolution * focal_length / sensor_height
    cx = (resolution-1) / 2
    cy = (resolution-1) / 2

    K = np.array([
        [focal_length_scaled, 0, cx],
        [0,  focal_length_scaled, cy],
        [0,   0,  1]
    ], dtype=np.float32)
    distCoeffs = np.zeros(5, dtype=np.float32)  # no distortion

    ret, rvec, tvec = cv2.solvePnP(
        objectPoints=correspondences_3d,
        imagePoints=correspondences_2d,
        cameraMatrix=K,
        distCoeffs=distCoeffs,
        flags=cv2.SOLVEPNP_EPNP,
    )

    convention_change = torch.eye(4, dtype=torch.float32, device=device)
    convention_change[1, 1] = -1
    convention_change[2, 2] = -1

    R, _ = cv2.Rodrigues(rvec)
    RT = np.eye(4, dtype=np.float32)
    RT[:3, :3] = R
    RT[:3, 3] = tvec[:, 0]
    RT = torch.from_numpy(RT).to(device=device, dtype=torch.float32)
    RT = convention_change @ RT

    inlier_mask = count_inliers(
        points_3d=verts_3d[matched_verts],
        points_2d=matching_pixel[matched_verts],
        resolution=resolution,
        focal_length=focal_length,
        world2cam=RT,
    )
    return RT, inlier_mask


def project_points(
    points_3d: torch.Tensor,
    world2cam: torch.Tensor,
    focal_length: float,
    resolution: int = 160,
    sensor_height: float = 36.0,
):
    _cot_half_fov = ut.cot_half_fov(focal_length, sensor_height)
    cot_half_fov = torch.tensor([_cot_half_fov], dtype=torch.float32, device=points_3d.device)
    projection = ut.perspective_matrix(cot_half_fov)[0]
    
    model_view_verts = ut.apply_transform(points_3d, world2cam)

    clip_verts = ut.apply_projection(model_view_verts, projection)
    clip_verts = clip_verts / clip_verts[:, 3:]

    u = torch.clamp(resolution * 0.5 * (clip_verts[:, 0] + 1), 0, resolution)
    v = torch.clamp(resolution * 0.5 * (clip_verts[:, 1] + 1), 0, resolution)

    # points_cam = points_3d @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    
    # focal_length_scaled = resolution * focal_length / sensor_height
    # cx = (resolution - 1) / 2
    # cy = (resolution - 1) / 2
    
    # depth = -points_cam[:, 2]
    # depth = torch.clamp(depth, min=1e-5)
    
    # u = (focal_length_scaled * points_cam[:, 0]) / depth + cx
    # v = cy - (focal_length_scaled * points_cam[:, 1]) / depth

    return torch.stack([u, v], dim=-1)


def estimate_translation_bounded_2pt(
    pts_rel: torch.Tensor,         # Shape: (N, 3)
    matching_pixels: torch.Tensor, # Shape: (N, 2)
    resolution: int,
    focal_length: float,
    xy_limit: float = 0.3,        # Hard limit for Tx/Ty (e.g., 0.05 = 5cm)
    xy_penalty_weight = 100.0,
    subsample: bool = True,
    num_corresp: int = 2000,
    sensor_height: float = 36.0,
    ransac_threshold: float = 5.0, # In pixels
    ransac_iters: int = 100,
    refine: bool = True,
) -> tuple[tuple[float, float, float], torch.Tensor]:
    """
    Estimates translation (Tz, Tx, Ty) using a 2-point RANSAC solver with bounds.
    
    Assumes:
      - OpenGL Convention: Camera looks down -Z, +Y is Up.
      - Input 'pts_rel' are in camera space but missing the translation offset.
      - Tx and Ty are small and bounded by 'xy_limit'.
    
    Returns:
        tuple: ((Tz, Tx, Ty), inliers_mask)
    """
    assert pts_rel.ndim == 2
    assert pts_rel.shape[1] == 3
    assert pts_rel.shape[0] == matching_pixels.shape[0]
    assert matching_pixels.ndim == 2
    device = pts_rel.device
    N = pts_rel.shape[0]
    
    # 1. Subsample for Speed
    # ----------------------
    if subsample and N > num_corresp:
        perm = torch.randperm(N, device=device)[:num_corresp]
        pts_subset = pts_rel[perm]
        pix_subset = matching_pixels[perm]
    else:
        pts_subset = pts_rel
        pix_subset = matching_pixels
        
    # 2. Precompute Normalized Coordinates
    # ------------------------------------
    f_pix = resolution * focal_length / sensor_height
    cx = (resolution - 1) / 2.0
    cy = (resolution - 1) / 2.0
    
    # Normalized coords (u_hat, v_hat)
    u_hat = (pix_subset[:, 0] - cx) / f_pix
    v_hat = (pix_subset[:, 1] - cy) / f_pix
    
    X_sub, Y_sub, Z_sub = pts_subset[:, 0], pts_subset[:, 1], pts_subset[:, 2]
    
    best_inlier_count = -1
    best_T = (0.0, 0.0, 0.0) # (Tz, Tx, Ty)

    # 3. RANSAC Loop
    # --------------
    # We solve the linear system for OpenGL projection:
    #   u_hat = (X + Tx) / -(Z + Tz)  =>  Tx + u_hat*Tz = -X - u_hat*Z
    #   v_hat = (Y + Ty) /  (Z + Tz)  =>  Ty - v_hat*Tz = -Y + v_hat*Z
    
    n_hypotheses = 0
    num_subset = pts_subset.shape[0]
    for _ in range(ransac_iters):
        # Sample 2 unique points
        idx = torch.randperm(num_subset)[:2]
        
        # Extract components for the pair
        u_h, v_h = u_hat[idx], v_hat[idx]
        x_i, y_i, z_i = X_sub[idx], Y_sub[idx], Z_sub[idx]
        
        # Build A (4x3) and b (4)
        # Columns: [Tx, Ty, Tz]
        A = torch.tensor([
            [1.0, 0.0,  u_h[0]],
            [0.0, 1.0, -v_h[0]],
            [1.0, 0.0,  u_h[1]],
            [0.0, 1.0, -v_h[1]]
        ], device=device)
        
        b = torch.tensor([
            -x_i[0] - u_h[0]*z_i[0],
            -y_i[0] + v_h[0]*z_i[0],
            -x_i[1] - u_h[1]*z_i[1],
            -y_i[1] + v_h[1]*z_i[1]
        ], device=device)
        
        # Solve Ax = b
        sol = torch.linalg.lstsq(A, b).solution
        cand_Tx, cand_Ty, cand_Tz = sol[0], sol[1], sol[2]

        # Check 1: Hard Bounds on X/Y Translation
        if abs(cand_Tx) > xy_limit or abs(cand_Ty) > xy_limit:
            continue
            
        # Check 2: Physical Validity (Points must be in front of camera)
        # OpenGL: Z_final must be negative
        if (z_i[0] + cand_Tz) > -0.01 or (z_i[1] + cand_Tz) > -0.01:
            continue

        # Verify Inliers (Vectorized)
        Z_est = Z_sub + cand_Tz
        # If Z_est is not negative, the point is behind camera. 
        # Clamp to small negative to avoid div/0, error check handles the rest.
        depth = torch.clamp(-Z_est, min=1e-5)
        
        u_proj = (f_pix * (X_sub + cand_Tx)) / depth + cx
        v_proj = cy - (f_pix * (Y_sub + cand_Ty)) / depth

        error_sq = (pix_subset[:, 0] - u_proj)**2 + (pix_subset[:, 1] - v_proj)**2
        
        # Mark points behind camera as outliers
        invalid_depth = Z_est > -0.01
        error_sq[invalid_depth] = float('inf')
        
        count = torch.sum(error_sq < ransac_threshold**2)

        n_hypotheses += 1

        if count > best_inlier_count:
            best_inlier_count = count
            best_T = (cand_Tz.item(), cand_Tx.item(), cand_Ty.item())

    # 4. Final Inlier Calculation & Refinement
    # ----------------------------------------
    best_Tz, best_Tx, best_Ty = best_T
    
    # Compute final inliers on FULL dataset
    Z_full = pts_rel[:, 2] + best_Tz
    depth_full = torch.clamp(-Z_full, min=1e-5)
    
    u_final = (f_pix * (pts_rel[:, 0] + best_Tx)) / depth_full + cx
    v_final = cy - (f_pix * (pts_rel[:, 1] + best_Ty)) / depth_full
    
    err_full = (matching_pixels[:, 0] - u_final)**2 + (matching_pixels[:, 1] - v_final)**2
    # Ensure points behind camera are not inliers
    err_full[Z_full > -0.01] = float('inf')
    
    final_inliers = err_full < (ransac_threshold**2)

    if not refine:
        return best_T, final_inliers

    # Refinement using Scipy (CPU)
    valid_idx = torch.nonzero(final_inliers).squeeze()
    if valid_idx.numel() < 3:
        return best_T, final_inliers

    pts_np = pts_rel[valid_idx].cpu().numpy()
    pix_np = matching_pixels[valid_idx].cpu().numpy()
    
    def residual(params):
        tz, tx, ty = params
        
        # OpenGL depth = -(Z + Tz)
        depth_r = -(pts_np[:, 2] + tz)
        depth_r = np.maximum(depth_r, 1e-5)
        
        u_r = (f_pix * (pts_np[:, 0] + tx)) / depth_r + cx
        v_r = cy - (f_pix * (pts_np[:, 1] + ty)) / depth_r
        
        reproj_err = np.concatenate([u_r - pix_np[:, 0], v_r - pix_np[:, 1]])
        
        # These extra "residuals" add (weight * tx)^2 + (weight * ty)^2 to the cost
        reg_terms = np.array([tx * xy_penalty_weight, ty * xy_penalty_weight])
        
        return np.concatenate([reproj_err, reg_terms])

    # Optimization Bounds
    # Tz: (-inf, max_valid_z] (must keep objects in front)
    # Tx, Ty: [-xy_limit, +xy_limit]
    tz_bound = -np.max(pts_np[:, 2]) - 0.01
    
    res = least_squares(
        residual,
        x0=[best_Tz, best_Tx, best_Ty],
        bounds=(
            [-np.inf, -xy_limit, -xy_limit], 
            [tz_bound, xy_limit,  xy_limit]
        ),
        ftol=1e-4
    )
    
    return tuple(res.x), final_inliers


def estimate_pose_and_focal_search(
    pts_rel: torch.Tensor,         # Shape: (N, 3)
    matching_pixels: torch.Tensor, # Shape: (N, 2)
    resolution: int,
    sensor_height: float = 36.0,
    focal_range: tuple[float, float] = (30.0, 300.0),
    focal_steps: int = 20,
    xy_limit: float = 0.3,        # Hard limit for Tx/Ty (e.g., 0.05 = 5cm)
    xy_penalty: float = 100.0,    # Regularization weight for refinement
    ransac_threshold: float = 5.0,
    ransac_iters: int = 50,
    num_corresp: int = 1000,
    refine: bool = True,
):
    """
    Grid search wrapper that robustly estimates (Tz, Tx, Ty) and Focal Length.
    
    1. Grid Search: Iterates over focal lengths, using a bounded 2-point RANSAC
       solver to find the best translation (Tz, Tx, Ty) for each candidate.
    2. Refinement: Jointly optimizes [Tz, Tx, Ty, Focal] with soft constraints
       on Tx/Ty and hard physical bounds on Tz and Focal.
    """
    # Generate focal candidates
    f_candidates = torch.linspace(focal_range[0], focal_range[1], focal_steps)
    
    best_score = -1
    best_T = (0.0, 0.0, 0.0) # (Tz, Tx, Ty)
    best_f_mm = 0.0
    best_inliers = None

    # --- 1. Grid Search Loop ---
    # We iterate over reasonable focal lengths to initialize the non-linear problem
    for f_mm in f_candidates:
        f_val = f_mm.item()
        
        # Use the robust 2-point solver defined previously.
        # Note: We disable the internal 'refine' of the helper to keep the loop fast.
        # We only refine the global winner at the end.
        translation, inliers = estimate_translation_bounded_2pt(
            pts_rel=pts_rel,
            matching_pixels=matching_pixels,
            resolution=resolution,
            focal_length=f_val,
            xy_limit=xy_limit,
            sensor_height=sensor_height,
            ransac_threshold=ransac_threshold,
            ransac_iters=ransac_iters,
            num_corresp=num_corresp,
            refine=False  # Speed up grid search
        )
            
        count = inliers.sum().item()
        if count > best_score:
            best_score = count
            best_T = translation
            best_f_mm = f_val
            best_inliers = inliers

    # If grid search failed (e.g. no valid depth found), return safe defaults
    if best_inliers is None:
        avg_f = (focal_range[0] + focal_range[1]) / 2.0
        return (0.0, 0.0, 0.0), avg_f, torch.zeros(len(pts_rel), dtype=torch.bool)
    
    if not refine:
        return best_T, best_f_mm, best_inliers

    # --- 2. Joint Refinement (Tz, Tx, Ty, Focal) ---
    # Move to CPU/NumPy for scipy.optimize
    valid_idx = torch.nonzero(best_inliers).squeeze()
    
    pts_np = pts_rel[valid_idx].cpu().numpy()
    pix_np = matching_pixels[valid_idx].cpu().numpy()
    
    cx = (resolution - 1) / 2.0
    cy = (resolution - 1) / 2.0
    
    # Unpack best initial guess
    tz_init, tx_init, ty_init = best_T
    
    # Calculate physical bounds for Tz
    # Object must be in front of camera (Z_cam < -near_plane)
    # Z_cam = z + tz  =>  tz < -z - near_plane
    near_plane = 0.01
    tz_upper_bound = -np.max(pts_np[:, 2]) - near_plane
    
    # Ensure init respects the bound (handle noisy edge cases)
    if tz_init > tz_upper_bound:
        tz_init = tz_upper_bound - 0.01

    def residual(params):
        tz, tx, ty, f_mm = params
        f_pix = f_mm * resolution / sensor_height
        
        # OpenGL Depth: -(Z + Tz)
        # Note: We use the clamped depth for stability, but bounds prevent hitting it.
        depth = -(pts_np[:, 2] + tz)
        depth = np.maximum(depth, 1e-5)
        
        u_proj = (f_pix * (pts_np[:, 0] + tx)) / depth + cx
        v_proj = cy - (f_pix * (pts_np[:, 1] + ty)) / depth
        
        reproj_err = np.concatenate([u_proj - pix_np[:, 0], v_proj - pix_np[:, 1]])
        
        # Soft Penalty for Tx, Ty
        # Adds (weight * val)^2 to the cost function
        penalty = np.array([tx * xy_penalty, ty * xy_penalty])
        
        return np.concatenate([reproj_err, penalty])

    # Optimize [Tz, Tx, Ty, Focal]
    res = least_squares(
        residual,
        x0=[tz_init, tx_init, ty_init, best_f_mm],
        bounds=(
            [-np.inf, -xy_limit, -xy_limit, focal_range[0]], # Lower
            [tz_upper_bound, xy_limit, xy_limit, focal_range[1]] # Upper
        ),
        ftol=1e-4
    )
    
    final_tz, final_tx, final_ty, final_f_mm = res.x
    
    return (final_tz, final_tx, final_ty), final_f_mm, best_inliers


def estimate_camera_pose_new(
    posed_pointmap: torch.Tensor,
    occupancy_mask: torch.Tensor,
    world2rel_no_scale: torch.Tensor,
    focal_length: float,
    resolution: int,
    scale: float,
):
    first_layer_mask = (occupancy_mask > 0.5)[:, :, 0]
    points_rel = posed_pointmap[:, :, 0, :][first_layer_mask]
    points_rel_scaled = scale_in_place(points_rel, scale)
    grid = get_grid(resolution, device=posed_pointmap.device)
    pixels = grid[first_layer_mask]
    (Tz, Tx, Ty), cam_inlier_mask = estimate_translation_bounded_2pt(
        pts_rel=points_rel_scaled,
        matching_pixels=pixels,
        resolution=resolution,
        focal_length=focal_length,
    )
    world2cam_new = world2rel_no_scale.clone()
    T_offset = torch.tensor([Tx, Ty, Tz], device=world2cam_new.device, dtype=torch.float32)
    world2cam_new[:3, 3] += T_offset
    return world2cam_new, cam_inlier_mask


def estimate_camera_pose_new_with_focal(
    posed_pointmap: torch.Tensor,
    occupancy_mask: torch.Tensor,
    world2rel_no_scale: torch.Tensor,
    resolution: int,
    scale: float,
):
    first_layer_mask = (occupancy_mask > 0.5)[:, :, 0]
    points_rel = posed_pointmap[:, :, 0, :][first_layer_mask]
    points_rel_scaled = scale_in_place(points_rel, scale)
    grid = get_grid(resolution, device=posed_pointmap.device)
    pixels = grid[first_layer_mask]
    (Tz, Tx, Ty), focal_length, cam_inlier_mask = estimate_pose_and_focal_search(
        pts_rel=points_rel_scaled,
        matching_pixels=pixels,
        resolution=resolution,
    )
    world2cam_new = world2rel_no_scale.clone()
    T_offset = torch.tensor([Tx, Ty, Tz], device=world2cam_new.device, dtype=torch.float32)
    world2cam_new[:3, 3] += T_offset
    return world2cam_new, focal_length, cam_inlier_mask


def get_transform_error(T_pred: torch.Tensor, T_gt: torch.Tensor) -> tuple[float, float]:
    t_pred = T_pred[:3, 3]
    t_gt = T_gt[:3, 3]
    trans_error = torch.linalg.norm(t_pred - t_gt)
    
    R_pred = T_pred[:3, :3]
    R_gt = T_gt[:3, :3]
    R_rel = R_gt.T @ R_pred
    return trans_error.item(), get_rotation_mag(R_rel)


def get_rotation_mag(R: torch.Tensor, in_deg: bool = True) -> float | torch.Tensor:
    is_batched = R.dim() > 2
    if not is_batched:
        R = R.unsqueeze(0)
    
    # 1. Compute the Trace (sum of diagonal elements)
    # R is (..., 3, 3), so we sum R[ii]
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    
    # 2. Solve for cos(theta)
    # Trace(R) = 1 + 2 * cos(theta)
    # cos(theta) = (Trace(R) - 1) / 2
    cos_theta = (trace - 1.0) / 2.0
    
    # 3. Clip for numerical stability
    # Floating point errors can make cos_theta slightly > 1.0 or < -1.0, 
    # which causes torch.acos to return NaN.
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    # 4. Calculate Angle
    theta = torch.acos(cos_theta)
    if in_deg:
        theta = torch.rad2deg(theta)
    
    if not is_batched:
        return theta.item()
    return theta


def decompose_transform(T: torch.Tensor) -> tuple[float, float, float]:
    t = T[:3, 3]
    trans_norm = torch.linalg.norm(t).item()

    M = T[:3, :3]
    scale = torch.linalg.norm(M[:, 0]).item()

    if scale > 1e-9:
        R = M / scale
    else:
        torch.eye(3, device=T.device, dtype=T.dtype)

    rot_angle_deg = get_rotation_mag(R)
    return scale, rot_angle_deg, trans_norm


def get_transform_error_batched(T_pred: torch.Tensor, T_gt: torch.Tensor):
    t_pred = T_pred[:, :3, 3]  # (B, 3)
    t_gt = T_gt[:, :3, 3]      # (B, 3)
    trans_error = torch.linalg.norm(t_pred - t_gt, axis=1)

    R_pred = T_pred[:, :3, :3]  # (B, 3, 3)
    R_gt = T_gt[:, :3, :3]      # (B, 3, 3)
    R_rel = torch.transpose(R_gt, (0, 2, 1)) @ R_pred  # (B, 3, 3)

    rot_error_deg = get_rotation_mag(R_rel)
    return trans_error, rot_error_deg


def get_rotation_angle_rad(T: torch.Tensor) -> tuple[float, float]:
    R = T[..., :3, :3]  # (B, 3, 3)
    return get_rotation_mag(R, in_deg=False)


def get_body_centric_coord_sys(joint_pos_dict: dict[str, torch.Tensor]):
    # Forward (Z): Hips -> Neck
    z_axis = joint_pos_dict["Spine_02"] - joint_pos_dict["Spine_03"]
    z_axis /= torch.linalg.norm(z_axis)
    
    # Right Temp: Left Hip -> Right Hip
    right_temp = joint_pos_dict["hip_b.R"] - joint_pos_dict["hip_b.L"]
    right_temp /= torch.linalg.norm(right_temp)
    
    # Up (Y): Perpendicular to Spine and Thighs (Dorsal direction)
    y_axis = torch.linalg.cross(right_temp, z_axis)
    y_axis /= torch.linalg.norm(y_axis)
    
    # Right Corrected (X): Ensure orthogonality
    # x_axis = np.cross(y_axis, z_axis)
    x_axis = torch.linalg.cross(z_axis, y_axis)
    x_axis /= torch.linalg.norm(x_axis)
    
    # Horse Rotation Matrix (Columns = Basis Vectors)
    R_horse = torch.column_stack((x_axis, y_axis, z_axis))
    return R_horse


def get_body_centric_camera_pose(cam2world: torch.Tensor, joint_pos_dict: dict[str, torch.Tensor]):
    cam_pos_world = cam2world[:3, 3]
    rel_pos_world = cam_pos_world - joint_pos_dict["Spine_03"]

    R_horse = get_body_centric_coord_sys(joint_pos_dict)
    y_axis = R_horse[:, 1]
    
    # Project into Horse Frame
    # V_local = R_horse.T * V_world
    pos_local = rel_pos_world @ R_horse 
    
    radius = torch.linalg.norm(pos_local)
    
    # Elevation: Angle from the X-Z plane (Spinal plane)
    # arcsin(y / r)
    elevation = torch.rad2deg(torch.arcsin(pos_local[1] / radius))
    
    # Azimuth: Angle in the X-Z plane
    # arctan2(x, z). 0 is Forward (Z), +/- 180 is Behind
    azimuth = torch.rad2deg(torch.arctan2(pos_local[0], pos_local[2]))
    
    # 4. Calculate Camera Roll
    # We need the Camera's Local vectors in World Space
    # Assuming standard GLTF/OpenGL camera: 
    # -Z is Forward (Look), +Y is Up, +X is Right
    # cam_right_world = cam2world[:3, 0]
    cam_up_world    = cam2world[:3, 1]
    cam_fwd_world   = -cam2world[:3, 2] # Look vector
    
    # Project Horse's Up vector (y_axis) onto Camera's Image Plane
    # The image plane is perpendicular to cam_fwd_world
    
    # Projection formula: V_proj = V - (V . N) * N
    horse_up_projected = y_axis - torch.dot(y_axis, cam_fwd_world) * cam_fwd_world
    
    # Normalize (handle edge case where cam looks straight down horse's Y)
    norm = torch.linalg.norm(horse_up_projected)
    if norm < 1e-6:
        # Singularity: Camera looking straight down the Horse's Up vector
        # Roll is undefined or can be set to 0
        roll = 0.0
    else:
        horse_up_projected /= norm
        
        # Calculate angle between Camera Up and Projected Horse Up
        # Dot product gives cos(angle)
        dot = torch.dot(horse_up_projected, cam_up_world)
        # Cross product (along look axis) gives sign
        cross = torch.linalg.cross(horse_up_projected, cam_up_world)
        det = torch.dot(cross, cam_fwd_world)
        
        roll = torch.rad2deg(torch.arctan2(det, dot))

    return {
        "azimuth": azimuth.item(),
        "elevation": elevation.item(),
        "radius": radius.item(),
        "roll": roll.item(),
    }


def get_camera_matrix_from_spherical(
    azimuth: float,
    elevation: float,
    roll: float,
    joint_pos_dict: dict[str, torch.Tensor],
    target_points: torch.Tensor,
    focal_length: float,
    sensor_height: float = 36.0,
    margin: float = 0.1,
):
    """
    Calculates a camera pose that maintains the requested Az/El/Roll orientation
    but shifts the camera position (Pan/Dolly) to perfectly frame the target_points.
    """
    device = target_points.device

    R_horse = get_body_centric_coord_sys(joint_pos_dict)
    y_axis = R_horse[:, 1]
    
    # Calculate Direction in Horse Frame
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    
    local_y = np.sin(el_rad)
    xz_len = np.cos(el_rad)
    local_x = xz_len * np.sin(az_rad)
    local_z = xz_len * np.cos(az_rad)
    
    dir_local = torch.tensor([local_x, local_y, local_z], device=device, dtype=torch.float32)
    dir_world = R_horse @ dir_local
    dir_world /= torch.linalg.norm(dir_world)
    
    # Camera Forward (Look) is opposite to the direction vector
    cam_fwd = -dir_world 
    
    # Handle Roll
    horse_up = y_axis
    base_up = horse_up - torch.dot(horse_up, cam_fwd) * cam_fwd
    if torch.linalg.norm(base_up) < 1e-6:
        y_vec = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)
        base_up = y_vec - torch.dot(y_vec, cam_fwd) * cam_fwd
    base_up /= torch.linalg.norm(base_up)
    
    roll_rad = np.radians(roll)
    c, s = np.cos(roll_rad), np.sin(roll_rad)
    cam_up = base_up * c + torch.linalg.cross(cam_fwd, base_up) * s
    cam_right = torch.linalg.cross(cam_fwd, cam_up)
    
    # R_cam columns: [Right, Up, Back] (assuming standard GLTF/OpenGL)
    cam_back = -cam_fwd
    R_cam = torch.column_stack((cam_right, cam_up, cam_back))
    
    # --- 2. Center and Frame the Points (New Logic) ---
    
    # Transform points into "Camera Rotation Space" relative to the Spine
    # This aligns the points so X is "Screen Right", Y is "Screen Up", Z is "Depth"
    center_ref = joint_pos_dict["Spine_03"]
    rel_vecs = target_points - center_ref
    
    # Rotate vectors: v_local = v_world . R_cam
    p_local = rel_vecs @ R_cam
    x = p_local[:, 0]
    y = p_local[:, 1]
    z = p_local[:, 2]

    # --- 3. Calculate Optimal Depth (The "Dolly") ---
    slope = sensor_height / (2.0 * focal_length)
    
    # Apply Margin
    # We want the point to hit (1.0 - margin) of the sensor edge
    limit = slope * (1.0 - margin)

    # --- Solve X Axis (Pan & Dolly) ---
    # We need to satisfy:
    # Left Edge:  Sx*D - Ox >= Sx*z - x
    # Right Edge: Sx*D + Ox >= Sx*z + x

    # Solve the system:
    # Sx*D - Ox = A
    # Sx*D + Ox = B
    # -> 2*Sx*D = A + B
    # -> 2*Ox = B - A
    
    # Calculate the "boundary pressure" for every single point
    left_constraints  = limit * z - x
    right_constraints = limit * z + x
    
    # The required values are determined by the "worst case" points
    L_max = torch.amax(left_constraints, dim=0)
    R_max = torch.amax(right_constraints, dim=0)
    D_req_x = (L_max + R_max) / (2.0 * limit)
    
    # --- Solve Y Axis (Tilt & Dolly) ---
    top_constraints = limit * z + y
    bot_constraints = limit * z - y

    B_max = torch.amax(bot_constraints, dim=0)
    T_max = torch.amax(top_constraints, dim=0)
    D_req_y = (B_max + T_max) / (2.0 * limit)
    
    # --- Combine ---
    # We must be far enough back to satisfy BOTH Width and Height
    D_final = torch.max(D_req_x, D_req_y)
    
    # Ensure camera is not inside the object (Near clip safety)
    # The camera must be at least slightly in front of the closest point
    # Note: 'z' is positive towards camera. We need D > max(z)
    min_safe_dist = torch.max(z) + 0.1
    D_final = torch.clamp(D_final, min=min_safe_dist)

    # --- 4. Pass 2: Re-Scan for Visual Center ---
    # Now that D_final is fixed, we project points to find the new visual bounds.
    # The "Constraint Winner" from Step 3 might NOT be the visual edge at this new depth!
    
    denom = torch.clamp(D_final - z, min=1e-3)
    
    # Project to "Screen Space" (assuming 0 offset for now)
    x_proj = x / denom
    y_proj = y / denom
    
    # Find the indices of the points that appear Leftmost/Rightmost/Top/Bottom on screen
    idx_L = torch.argmin(x_proj)
    idx_R = torch.argmax(x_proj)
    idx_B = torch.argmin(y_proj)
    idx_T = torch.argmax(y_proj)
    
    d_L = denom[idx_L]
    d_R = denom[idx_R]
    d_B = denom[idx_B]
    d_T = denom[idx_T]
    
    # Weighted Average Formula
    Ox_final = (x[idx_L] * d_R + x[idx_R] * d_L) / (d_L + d_R)
    Oy_final = (y[idx_B] * d_T + y[idx_T] * d_B) / (d_B + d_T)
    
    # --- 4. Final Transform ---
    world_offset = (cam_right * Ox_final) + \
                   (cam_up    * Oy_final) + \
                   (cam_back  * D_final)
                   
    final_cam_pos = center_ref + world_offset
    
    cam2world = torch.eye(4, device=device, dtype=torch.float32)
    cam2world[:3, :3] = R_cam
    cam2world[:3, 3] = final_cam_pos

    world2cam = torch.inverse(cam2world)
    world2cam[3, :3] = 0
    return world2cam, cam2world


params_roll = {'front': {'df': 1.275, 'mu': -0.177, 'std': 4.848}, 'left': {'df': 1.625, 'mu': -6.331, 'std': 5.142}, 'right': {'df': 1.672, 'mu': 6.318, 'std': 5.128}, 'rear': {'df': 1.297, 'mu': 0.089, 'std': 4.802}}
params_el = {'front': {'df': 1.747, 'mu': 8.246, 'std': 5.488}, 'left': {'df': 2.308, 'mu': 0.946, 'std': 6.926}, 'right': {'df': 2.195, 'mu': 1.388, 'std': 6.794}, 'rear': {'df': 1.536, 'mu': -5.692, 'std': 4.859}}

def get_view_type(az):
    if (az < -150) or (az >= 150):
        return "rear"
    if (az >= -150) and (az < -30):
        return "right"
    if (az >= -30) and (az < 30):
        return "front"
    if (az >= 30) and (az < 150):
        return "left"
    raise ValueError(f"incorrect azimuth {az}")

def get_other_views(azimuth):
    view_types = ["front", "rear", "right", "left"]
    view_type = get_view_type(azimuth)
    return [x for x in view_types if x != view_type]

def sample_view(view_type: str | None = None):
    if view_type is None:
        az = np.random.uniform(-180, 180)
        view_type = get_view_type(az)
    elif view_type == "front":
        az = np.random.uniform(-30, 30)
    elif view_type == "rear":
        if np.random.uniform() < 0.5:
            az = np.random.uniform(-180, -150)
        else:
            az = np.random.uniform(150, 180)
    elif view_type == "right":
        az = np.random.uniform(-150, -30)
    elif view_type == "left":
        az = np.random.uniform(30, 150)
    else:
        print(f"UNKNOWN VIEW TYPE {view_type}")

    df_roll = params_roll[view_type]['df']
    mu_roll = params_roll[view_type]['mu']
    std_roll = params_roll[view_type]['std']
    roll = stats.t.rvs(df_roll, mu_roll, std_roll)

    df_el = params_el[view_type]['df']
    mu_el = params_el[view_type]['mu']
    std_el = params_el[view_type]['std']
    el = stats.t.rvs(df_el, mu_el, std_el)

    margin = np.random.uniform(0.1, 0.25)

    return {
        'azimuth': az,
        'elevation': el,
        'roll': roll,
        'margin': margin,
    }

VIEWS = ['front', 'rear', 'right', 'left']
VIEW_SAMPLE_PROBS = {
    'front': [0.05, 0.15, 0.4, 0.4],
    'rear': [0.15, 0.05, 0.4, 0.4],
    'right': [0.3, 0.3, 0.1, 0.3],
    'left': [0.3, 0.3, 0.3, 0.1],
}

def sample_other_view_type(view_type):
    return np.random.choice(VIEWS, p=VIEW_SAMPLE_PROBS[view_type])


class BlockTimer:
    def __init__(self):
        self.times = {}

    def __call__(self, name):
        return _TimerContext(self, name)

    def record(self, name, dt):
        self.times.setdefault(name, []).append(dt)

    def summary(self):
        for name, vals in self.times.items():
            print(f"{name:20s}: {sum(vals)/len(vals):.6f} s (avg over {len(vals)} runs)")


class _TimerContext:
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name

    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        self.prof_ctx = torch.autograd.profiler.record_function(self.name)
        self.prof_ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        torch.cuda.synchronize()
        dt = time.perf_counter() - self.t0
        self.parent.record(self.name, dt)
        self.prof_ctx.__exit__(exc_type, exc, tb)


class PeriodicTimer:
    def __init__(self, name: str, interval: int = 100):
        self.name = name
        self.interval = interval
        self.times = []
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        self.times.append(self.start.elapsed_time(self.end))
        if len(self.times) >= self.interval:
            print(f"{self.name}: {sum(self.times)/len(self.times):.2f} ms")
            self.times = []


@contextmanager
def mps_timer(label):
    torch.mps.synchronize()
    t0 = time.perf_counter()
    yield
    torch.mps.synchronize()
    print(f"{label}: {(time.perf_counter() - t0)*1000:.2f} ms")


# def estimate_depth_offset(
#     pts_rel: np.ndarray,      # P_world (N, 3)
#     matching_pixels: np.ndarray,# Pixel coordinates (N, 2)
#     resolution: int,
#     focal_length: float,
#     subsample: bool = True,
#     num_corresp: int = 2000,    # RANSAC needs a healthy pool of points
#     sensor_height: float = 36.0,
#     ransac_threshold: float = 5.0, # in pixels
#     ransac_iters: int = 100,
#     refine: bool = False,
# ):
#     """
#     Recovers the full world-to-camera transform given the 'relative' transform
#     (which lacks absolute depth) and 2D-3D correspondences.
#     """
#     _pts_rel = pts_rel
#     _matching_pixels = matching_pixels
#     if subsample and pts_rel.shape[0] > num_corresp:
#         # Randomly sample if we have too many points to keep RANSAC fast
#         pi = np.random.permutation(pts_rel.shape[0])[:num_corresp]
#         pts_rel = pts_rel[pi]
#         matching_pixels = matching_pixels[pi]

#     X_rel = pts_rel[:, 0]
#     Y_rel = pts_rel[:, 1]
#     Z_rel = pts_rel[:, 2]

#     focal_length_scaled = resolution * focal_length / sensor_height
#     cx = (resolution-1) / 2
#     cy = (resolution-1) / 2
    
#     u_obs = matching_pixels[:, 0]
#     v_obs = matching_pixels[:, 1]

#     # Pre-compute denominators to identify stable axes
#     denom_u = u_obs - cx
#     denom_v = v_obs - cy
    
#     # Compute candidate Tz values from both U and V
#     # We use a small epsilon to avoid division by zero
#     cand_Tz_u = -(focal_length_scaled * X_rel) / (denom_u + 1e-6) - Z_rel
#     cand_Tz_v = (focal_length_scaled * Y_rel) / (denom_v + 1e-6) - Z_rel

#     # For each point, pick the candidate from the dimension (u or v) 
#     # that is further from the optical center (more stable)
#     use_v = torch.abs(denom_v) > torch.abs(denom_u)
#     candidates = torch.where(use_v, cand_Tz_v, cand_Tz_u)

#     best_inlier_count = -1
#     best_Tz = 0.0

#     # Filter invalid candidates (e.g. those that put points behind camera)
#     # This assumes standard pinhole (+Z depth). If OpenGL (-Z depth), 
#     # we might need to check -(Z_rel + Tz) > 0. 
#     # We rely on reprojection error to select the correct one.
#     valid_mask = (Z_rel + candidates) < -0.1
#     candidate_pool = candidates[valid_mask]
    
#     # If no valid candidates, return input matrix
#     if len(candidate_pool) == 0:
#         raise NoCandidatesError("No valid Tz candidates found.")

#     for _ in range(ransac_iters):
#         # Sample one hypothesis
#         idx = torch.randint(0, len(candidate_pool), (1,), device=candidate_pool.device)
#         curr_Tz = candidate_pool[idx]

#         # Project all points
#         Z_est = Z_rel + curr_Tz
#         # Guard against zero depth div
#         depth = torch.clamp(-Z_est, min=1e-5)
        
#         u_proj = (focal_length_scaled * X_rel) / depth + cx
#         v_proj = cy - (focal_length_scaled * Y_rel) / depth

#         # Calculate error
#         error_sq = (u_obs - u_proj)**2 + (v_obs - v_proj)**2
#         inliers = error_sq < (ransac_threshold**2)
#         count = torch.sum(inliers)

#         if count > best_inlier_count:
#             best_inlier_count = count
#             best_Tz = curr_Tz

#     # recompute inliers on all points
#     X = _pts_rel[:, 0]
#     Y = _pts_rel[:, 1]
#     Z = _pts_rel[:, 2]
#     _u_obs = _matching_pixels[:, 0]
#     _v_obs = _matching_pixels[:, 1]
#     best_Z = Z + best_Tz
#     best_depth = torch.clamp(-best_Z, min=1e-5)
#     best_u_proj = (focal_length_scaled * X) / best_depth + cx
#     best_v_proj = cy - (focal_length_scaled * Y) / best_depth
#     best_error_sq = (_u_obs - best_u_proj)**2 + (_v_obs - best_v_proj)**2
#     best_inliers = best_error_sq < (ransac_threshold**2)

#     if not refine:
#         return best_Tz.item(), best_inliers
    
#     _pts_rel = _pts_rel[best_inliers].cpu().numpy()
#     _matching_pixels = _matching_pixels[best_inliers].cpu().numpy()
#     X = _pts_rel[:, 0]
#     Y = _pts_rel[:, 1]
#     Z = _pts_rel[:, 2]
#     u_obs = _matching_pixels[:, 0]
#     v_obs = _matching_pixels[:, 1]

#     def residual(tz_offset):
#         # Total Z (OpenGL: look down -Z)
#         Z_total = Z + tz_offset
#         depth = -Z_total
#         depth = np.maximum(depth, 1e-5)
        
#         # OpenGL Projection
#         u_proj = (focal_length_scaled * X) / depth + cx
#         v_proj = cy - (focal_length_scaled * Y) / depth

#         return np.concatenate([u_proj - u_obs, v_proj - v_obs])
    
#     Tz_init = best_Tz.item()
#     near_plane = 0.01
#     tz_upper_bound = -np.max(Z) - near_plane
#     if Tz_init > tz_upper_bound:
#         Tz_init = tz_upper_bound - 0.01

#     res = least_squares(
#         residual,
#         x0=[Tz_init],
#         bounds=([-np.inf], [tz_upper_bound]),
#         ftol=1e-4
#     )
#     final_Tz_offset = res.x[0]
#     return final_Tz_offset, best_inliers

# def estimate_pose_and_focal_search(
#     pts_rel: torch.Tensor,
#     matching_pixels: torch.Tensor,
#     resolution: int,
#     sensor_height: float = 36.0,
#     focal_range: tuple[float, float] = (30.0, 300.0),
#     focal_steps: int = 20,
#     ransac_threshold: float = 5.0,
#     ransac_iters: int = 50,
#     refine: bool = False,
# ):
#     """
#     Grid search wrapper that reuses estimate_camera_pose_from_rel to find 
#     both T_z and Focal Length. Includes a final refinement step.
#     """
#     f_candidates = torch.linspace(focal_range[0], focal_range[1], focal_steps)
    
#     best_score = -1
#     best_depth_offset = None
#     best_f_mm = 0.0
#     best_inliers = None

#     # --- 1. Grid Search Loop ---
#     for f_mm in f_candidates:
#         f_val = f_mm.item()
#         try:
#             # Reuse the existing single-focal solver
#             depth_offset, inliers = estimate_depth_offset(
#                 pts_rel=pts_rel,
#                 matching_pixels=matching_pixels,
#                 resolution=resolution,
#                 focal_length=f_val,
#                 sensor_height=sensor_height,
#                 ransac_threshold=ransac_threshold,
#                 ransac_iters=ransac_iters,
#             )
            
#             count = inliers.sum().item()
#             if count > best_score:
#                 best_score = count
#                 best_depth_offset = depth_offset
#                 best_f_mm = f_val
#                 best_inliers = inliers
                
#         except NoCandidatesError:
#             continue # Skip if no valid candidates found for this f

#     # If grid search failed completely, return input
#     if best_depth_offset is None:
#         return 0, (focal_range[0] + focal_range[1]) / 2
    
#     if not refine:
#         return best_depth_offset, best_f_mm, best_inliers
    
#     # --- Refinement (Joint Optimization) ---
#     # We move to CPU/NumPy for scipy.optimize
#     pts_rel = pts_rel[best_inliers].detach().cpu().numpy()
#     obs = matching_pixels[best_inliers].detach().cpu().numpy()

#     X, Y, Z = pts_rel[:, 0], pts_rel[:, 1], pts_rel[:, 2]
#     u_obs, v_obs = obs[:, 0], obs[:, 1]

#     cx = (resolution - 1) / 2
#     cy = (resolution - 1) / 2

#     # Extract the Tz we found from the pose matrix
#     # best_pose[2, 3] = original_Tz + new_offset
#     # We want to optimize the *total* Tz directly
#     Tz_init = best_depth_offset

#     # In OpenGL, Z must be negative.
#     # So (Z_rel + Tz) < -near_plane  =>  Tz < -near_plane - Z_rel
#     # This must hold for ALL points, so we look at the 'worst case' (max Z_rel)
    
#     # Upper bound: The object cannot cross the near plane
#     near_plane = 0.01
#     tz_upper_bound = -np.max(Z) - near_plane
#     if Tz_init > tz_upper_bound:
#         Tz_init = tz_upper_bound - 0.01

#     def residual(params):
#         tz_offset, f_mm = params
#         f_pix = f_mm * resolution / sensor_height
        
#         # Total Z (OpenGL: look down -Z)
#         Z_total = Z + tz_offset
#         depth = -Z_total
        
#         # Clamp depth to avoid singularities
#         depth = np.maximum(depth, 1e-5)
        
#         # OpenGL Projection
#         u_est = (f_pix * X) / depth + cx
#         v_est = cy - (f_pix * Y) / depth
        
#         return np.concatenate([u_est - u_obs, v_est - v_obs])

#     # Optimize both Tz and Focal Length
#     res = least_squares(
#         residual,
#         x0=[Tz_init, best_f_mm],
#         bounds=([-np.inf, focal_range[0]], [tz_upper_bound, focal_range[1]]),
#         ftol=1e-4
#     )
#     final_Tz_offset, final_f_mm = res.x
#     return final_Tz_offset, final_f_mm, best_inliers