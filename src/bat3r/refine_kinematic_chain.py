import torch
import torch.nn.functional as F
from dataclasses import dataclass

import bat3r.third_party.pytorch3d as t3d
from bat3r import bones, skin, trees
from bat3r.bones import ShapeProcessingData
from bat3r.skin import OneMeshGltf
from bat3r.eval_utils import get_rotation_angle_rad
# from bat3r.third_party.pytorch3d.loss import point_mesh_face_distance


@dataclass
class MeshFittingConfig:
    algo: str = 'grad'
    distance_threshold: float = 0.3
    # grad
    grad_num_steps: int = 200
    angle_loss_weight: float = 1e-5
    repulsion_weight: float = 1.0
    repulsion_radius: float = 0.05
    is_stranger_threshold: float = 0.3
    edge_length_loss_weight: float = 100.0
    local_volume_loss_weight: float = 20.0
    scale_sym_loss_weight: float = 0.01
    scale_bounds_loss_weight: float = 1000.0
    scale_reg_loss_weight: float = 0.01
    lr: float = 2e-2
    fixed_joints: list[int] | None = None
    optimize_bone_scales: bool = True
    subsample: int | None = None
    # greedy
    greedy_num_steps: int = 3
    joint_threshold: float = 0.5

    def get_grad_kwargs(self):
        return dict(
            distance_threshold=self.distance_threshold,
            num_iters=self.grad_num_steps,
            angle_loss_weight=self.angle_loss_weight,
            repulsion_weight=self.repulsion_weight,
            repulsion_radius=self.repulsion_radius,
            is_stranger_threshold=self.is_stranger_threshold,
            edge_length_loss_weight=self.edge_length_loss_weight,
            local_volume_loss_weight=self.local_volume_loss_weight,
            scale_sym_loss_weight=self.scale_sym_loss_weight,
            scale_bounds_loss_weight=self.scale_bounds_loss_weight,
            scale_reg_loss_weight=self.scale_reg_loss_weight,
            lr=self.lr,
            fixed_joints=list(self.fixed_joints) if self.fixed_joints else None,
            optimize_bone_scales=self.optimize_bone_scales,
            subsample=self.subsample,
        )

    def get_greedy_kwargs(self):
        return dict(
            distance_threshold=self.distance_threshold,
            num_iters=self.greedy_num_steps,
            joint_threshold=self.joint_threshold,
            scale=True,
        )


def fit_mesh_to_pointcloud(
    cfg: MeshFittingConfig,
    predicted_canon: torch.Tensor,
    predicted_pose: torch.Tensor,
    seq_mask: torch.Tensor,
    mesh_proc_data: ShapeProcessingData,
    geo_dists: torch.Tensor | None = None,
    return_metrics: bool = False,
):
    if cfg.algo == 'grad':
        return refine_kinematic_chain_batched(
            predicted_canon=predicted_canon,
            predicted_pose=predicted_pose,
            seq_mask=seq_mask,
            mesh_proc_data=mesh_proc_data,
            geo_dists=geo_dists,
            return_metrics=return_metrics,
            **cfg.get_grad_kwargs(),
        )
    elif cfg.algo == 'greedy':
        rel2world, local_joint_transforms, fitted_shape, joint_pos, joint_rots, inlier_mask, canonical_indices = bones.fit_kinematic_chain_batched(
            predicted_canon=predicted_canon,
            predicted_pose=predicted_pose,
            seq_mask=seq_mask,
            mesh_proc_data=mesh_proc_data,
            **cfg.get_greedy_kwargs(),
        )
        joint_angles_rad = get_rotation_angle_rad(joint_rots)
        
        return {
            'global_transform': rel2world,
            'local_joint_transforms': local_joint_transforms,
            'fitted_shape': fitted_shape,
            'global_transform_scaled': rel2world,
            'local_joint_transforms_scaled': local_joint_transforms,
            'fitted_shape_scaled': fitted_shape,
            'joint_pos': joint_pos,
            'joint_angles': joint_angles_rad,
            'inlier_mask': inlier_mask,
            'canonical_indices': canonical_indices,
            'sampled_ind': None,
            'metrics': {} if return_metrics else None,
        }
    else:
        raise ValueError(f"unknown mesh fitting algo '{cfg.algo}'")


def get_chamfer_dist(X, Y, weights: tuple[float] = (0.5, 0.5), delta: float = 0.1):
    dist = torch.cdist(X, Y)
    w1, w2 = weights
    min_dist_1 = torch.amin(dist, dim=1)
    min_dist_2 = torch.amin(dist, dim=0)
    loss_1 = F.huber_loss(min_dist_1, torch.zeros_like(min_dist_1), reduction='mean', delta=delta)
    loss_2 = F.huber_loss(min_dist_2, torch.zeros_like(min_dist_2), reduction='mean', delta=delta)
    cd = w1 * loss_1 + w2 * loss_2
    return cd


def get_face_to_bone_map(mesh_vertex_joints, mesh_vertex_weights, faces):
    dominant_weight_idx = torch.argmax(mesh_vertex_weights, dim=1) # (V,)
    v_bone_ids = mesh_vertex_joints[torch.arange(mesh_vertex_joints.shape[0]), dominant_weight_idx]
    face_vertex_bones = v_bone_ids[faces]
    face_bone_ids, _ = torch.mode(face_vertex_bones, dim=1)
    return face_bone_ids, v_bone_ids


def get_mesh_volume(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Computes the volume of a mesh using the Divergence Theorem.
    Differentiable and capable of handling batches if needed.
    
    Args:
        vertices: (N, 3) tensor of vertex positions.
        faces: (F, 3) tensor of face indices (integers).
    
    Returns:
        Scalar tensor representing the signed volume.
    """
    # 1. Center vertices to make the calculation translation-invariant
    # This is critical for non-watertight meshes to prevent volume 
    # fluctuations just from moving the mesh.
    centered_verts = vertices - vertices.mean(dim=0, keepdim=True)
    
    # 2. Gather triangle vertices
    # faces is (F, 3), so we get three (F, 3) tensors of vertices
    v1 = centered_verts[faces[:, 0]]
    v2 = centered_verts[faces[:, 1]]
    v3 = centered_verts[faces[:, 2]]
    
    # 3. Compute Signed Tetrahedron Volume
    # V = 1/6 * sum( (v1 x v2) . v3 )
    cross_prod = torch.cross(v1, v2, dim=1)
    dot_prod = torch.sum(cross_prod * v3, dim=1)
    
    return torch.abs(torch.sum(dot_prod) / 6.0)


def get_per_bone_volumes(vertices: torch.Tensor, faces: torch.Tensor, face_bone_ids: torch.Tensor, num_bones: int):
    """
    Computes volume per bone cluster efficiently.
    """
    # 1. Compute Centroids Per Bone
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    
    # Center of each face (F, 3)
    face_centers = (v1 + v2 + v3) / 3.0
    
    # Sum face centers per bone
    bone_sums = torch.zeros(num_bones, 3, device=vertices.device)
    bone_sums.index_add_(0, face_bone_ids, face_centers)
    
    # Count faces per bone
    bone_counts = torch.zeros(num_bones, device=vertices.device)
    bone_counts.index_add_(0, face_bone_ids, torch.ones_like(face_bone_ids, dtype=torch.float))
    
    # Calculate Centroids (Handle divide by zero safely)
    # The centroid for empty bones will be 0, but it won't be used.
    bone_centroids = bone_sums / (bone_counts.unsqueeze(1) + 1e-8)
    
    # 2. Compute Signed Volumes Relative to Bone Centroid
    # Expand bone centroids back to faces
    face_origins = bone_centroids[face_bone_ids]
    
    rel_v1 = v1 - face_origins
    rel_v2 = v2 - face_origins
    rel_v3 = v3 - face_origins
    
    cross_prod = torch.cross(rel_v1, rel_v2, dim=1)
    dot_prod = torch.sum(cross_prod * rel_v3, dim=1)
    tet_volumes = dot_prod / 6.0
    
    # 3. Sum volumes per bone
    bone_volumes = torch.zeros(num_bones, device=vertices.device)
    bone_volumes.index_add_(0, face_bone_ids, tet_volumes)
    
    return torch.abs(bone_volumes), bone_counts


def get_per_bone_volumes_batched(vertices: torch.Tensor, faces: torch.Tensor, face_bone_ids: torch.Tensor, num_bones: int):
    """
    Computes volume per bone cluster efficiently for batched vertices.
    vertices: (B, V, 3)
    faces: (F, 3)
    face_bone_ids: (F,)
    """
    B, V, _ = vertices.shape
    
    # 1. Compute Centroids Per Bone
    v1 = vertices[:, faces[:, 0]] # (B, F, 3)
    v2 = vertices[:, faces[:, 1]]
    v3 = vertices[:, faces[:, 2]]
    
    face_centers = (v1 + v2 + v3) / 3.0 # (B, F, 3)
    
    # Sum face centers per bone
    index = face_bone_ids.view(1, -1, 1).expand(B, -1, 3)
    bone_sums = torch.zeros(B, num_bones, 3, device=vertices.device, dtype=vertices.dtype)
    bone_sums.scatter_add_(1, index, face_centers)
    
    # Count faces per bone
    bone_counts = torch.zeros(num_bones, device=vertices.device, dtype=vertices.dtype)
    bone_counts.index_add_(0, face_bone_ids, torch.ones_like(face_bone_ids, dtype=vertices.dtype))
    
    # Calculate Centroids (Handle divide by zero safely)
    bone_centroids = bone_sums / (bone_counts.view(1, -1, 1) + 1e-8) # (B, num_bones, 3)
    
    # 2. Compute Signed Volumes Relative to Bone Centroid
    face_origins = torch.gather(bone_centroids, 1, index) # (B, F, 3)
    
    cross_prod = torch.cross(v1 - face_origins, v2 - face_origins, dim=2)
    dot_prod = torch.sum(cross_prod * (v3 - face_origins), dim=2) # (B, F)
    tet_volumes = dot_prod / 6.0
    
    # 3. Sum volumes per bone
    bone_volumes = torch.zeros(B, num_bones, device=vertices.device, dtype=vertices.dtype)
    index_vol = face_bone_ids.view(1, -1).expand(B, -1)
    bone_volumes.scatter_add_(1, index_vol, tet_volumes)
    
    return torch.abs(bone_volumes), bone_counts


def get_triangle_areas(verts: torch.Tensor, faces: torch.Tensor):
    """
    Penalizes deviations in surface area for every individual triangle.
    """
    v1 = verts[faces[:, 0]]
    v2 = verts[faces[:, 1]]
    v3 = verts[faces[:, 2]]

    # Cross product magnitude represents 2 * Area
    cross_prod = torch.cross(v2 - v1, v3 - v1, dim=1)
    
    # Area = 0.5 * norm(cross_product)
    # Adding a small epsilon for gradient stability of sqrt at 0
    cross_prod_sq = torch.sum(cross_prod**2, dim=1)
    areas = 0.5 * torch.sqrt(cross_prod_sq + 1e-10)
    return areas


def get_edges_and_lengths(vertices: torch.Tensor, faces: torch.Tensor):
    """
    Extracts unique edges and their squared lengths from the canonical mesh.
    """
    e1 = faces[:, [0, 1]]
    e2 = faces[:, [1, 2]]
    e3 = faces[:, [2, 0]]
    all_edges = torch.cat([e1, e2, e3], dim=0)

    # sort vertex indices to handle duplicates (e.g. edge [0,5] == [5,0])
    all_edges, _ = torch.sort(all_edges, dim=1)
    unique_edges = torch.unique(all_edges, dim=0)

    p1 = vertices[unique_edges[:, 0]]
    p2 = vertices[unique_edges[:, 1]]
    target_lens_sq = torch.sum((p1 - p2)**2, dim=1)

    return unique_edges, target_lens_sq


def edge_length_loss(
    verts: torch.Tensor,
    edge_indices: torch.Tensor,
    target_lens_sq: torch.Tensor,
):
    p1 = verts[edge_indices[:, 0]]
    p2 = verts[edge_indices[:, 1]]
    current_lens_sq = torch.sum((p1 - p2)**2, dim=1)
    return torch.mean((current_lens_sq - target_lens_sq) ** 2)


def edge_length_loss_batched(
    verts: torch.Tensor,
    edge_indices: torch.Tensor,
    target_lens_sq: torch.Tensor,
    reduction: str = 'sum',
):
    # Supports batched verts (B, V, 3) via broadcasting
    p1 = verts[:, edge_indices[:, 0]]
    p2 = verts[:, edge_indices[:, 1]]
    current_lens_sq = torch.sum((p1 - p2)**2, dim=-1)
    loss = torch.mean((current_lens_sq - target_lens_sq.unsqueeze(0)) ** 2, dim=1)
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    return loss


def get_pairwise_repulsion_loss_knn(
    verts: torch.Tensor,
    is_stranger: torch.Tensor,
    radius: float = 0.05,
    k: int = 100,
    subsample: bool = False,
):
    if subsample:
        N = verts.shape[0]
        N_sub = int(0.7 * N)
        idx = torch.randperm(N, device=verts.device)[:N_sub]
        verts = verts[idx]
        is_stranger = is_stranger[idx][:, idx]

    with torch.no_grad():
        dists = torch.cdist(verts, verts)
        _, knn_idx = torch.topk(dists, k=k+1, largest=False)
        knn_idx = knn_idx[:, 1:]

    nbr_verts = verts[knn_idx]
    diff = verts.unsqueeze(1) - nbr_verts
    dist_sq = torch.sum(diff**2, dim=-1)

    penalty = torch.relu(radius**2 - dist_sq) ** 2
    nbr_mask = is_stranger.gather(1, knn_idx).float().detach()
    return torch.sum(penalty * nbr_mask)


def get_pairwise_repulsion_loss_knn_batched(
    verts: torch.Tensor,
    is_stranger: torch.Tensor,
    radius: float = 0.05,
    k: int = 100,
    subsample: bool = False,
    reduction: str = 'sum',
):
    if subsample:
        B, N, _ = verts.shape
        N_sub = int(0.7 * N)
        idx = torch.randperm(N, device=verts.device)[:N_sub]
        verts = verts[:, idx]
        is_stranger = is_stranger[idx][:, idx]

    B, V, _ = verts.shape
    with torch.no_grad():
        dists = torch.cdist(verts, verts)
        _, knn_idx = torch.topk(dists, k=k+1, largest=False, dim=-1)
        knn_idx = knn_idx[..., 1:]

    batch_idx = torch.arange(B, device=verts.device).view(B, 1, 1).expand(B, V, k)
    nbr_verts = verts[batch_idx, knn_idx] # (B, V, k, 3)
    
    dist_sq = torch.sum((verts.unsqueeze(2) - nbr_verts)**2, dim=-1) # (B, V, k)
    penalty = torch.relu(radius**2 - dist_sq) ** 2
    
    is_stranger_expanded = is_stranger.unsqueeze(0).expand(B, -1, -1)
    nbr_mask = torch.gather(is_stranger_expanded, 2, knn_idx).float().detach()
    loss = torch.sum(penalty * nbr_mask, dim=(1, 2))
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    return loss


def get_pairwise_repulsion_loss(
    verts: torch.Tensor,
    is_stranger: torch.Tensor,
    radius: float = 0.05,
    subsample: bool = False,
    debug: bool = False,
):
    if subsample:
        N = verts.shape[0]
        N_sub = int(0.4 * N)
        idx = torch.randperm(N, device=verts.device)[:N_sub]
        verts = verts[idx]
        is_stranger = is_stranger[idx][:, idx]

    curr_dist_sq = torch.cdist(verts, verts)**2

    if debug:
        is_over_radius = curr_dist_sq < (radius**2)
        is_repulsed = is_over_radius & is_stranger.to(torch.bool)
        repulsed_ind = list(set(x.item() for x in torch.argwhere(is_repulsed).flatten()))
        return repulsed_ind

    diff = (radius ** 2) - curr_dist_sq
    penalty = torch.relu(diff) ** 2
    return torch.sum(penalty * is_stranger)


def geman_mcclure_loss(x, delta, reduction="mean"):
    """
    r: residuals (any shape)
    delta: scale parameter (float or tensor)
    """
    delta2 = delta * delta
    r2 = torch.sum(x * x, dim=-1)
    loss = r2 / (r2 + delta2)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    

def cauchy_loss(x, delta, reduction="mean"):
    r2 = (x * x).sum(dim=-1)
    delta2 = delta * delta
    loss = 0.5 * delta2 * torch.log1p(r2 / delta2)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def refine_kinematic_chain(
    predicted_canon: torch.Tensor,
    predicted_pose: torch.Tensor,
    mesh: OneMeshGltf,
    initial_local_joint_transforms: torch.Tensor | None = None,
    initial_global_transform: torch.Tensor | None = None,
    distance_threshold: float = 0.3,
    scale: bool = True,
    num_iters: int = 200,
    loss_type: str = 'huber',
    delta: float = 0.1,
    angle_loss_weight: float = 1e-5,
    local_volume_loss_weight: float = 50.0,
    edge_length_loss_weight: float = 100.0,
    repulsion_weight: float = 0,
    repulsion_radius: float = 0.05,
    geo_dists: torch.Tensor | None = None,
    is_stranger_threshold: float = 0.3,
    cd_weights: tuple[float] = (0.5, 0.5),
    warmup_iters: float = 25.0,
    lr=1e-2,
    subsample_repulsion: bool = False,
    fixed_joints: list[int] | None = None,
):
    predicted_canon = predicted_canon.detach()
    _device = predicted_canon.device

    template_verts = mesh.vertices.to(dtype=torch.float32, device=_device)
    dist = torch.cdist(template_verts, predicted_canon)
    _dist, canonical_indices = torch.min(dist, dim=0)
    inlier_mask = _dist < distance_threshold
    # weights = torch.relu(1 - torch.square(_dist[inlier_mask] / distance_threshold))
    # weight_sum = torch.sum(weights)
    canonical_indices = canonical_indices[inlier_mask]
    predicted_pose = predicted_pose[inlier_mask]

    joints = mesh.joints.tolist()
    joint_to_index = {joint: idx for idx, joint in enumerate(joints)}
    num_joints = mesh.local_joint_transforms.shape[0]
    parents, children, _, roots = trees.recover_parents(mesh.nodes_parents_list)
    topo_order = trees.topsort(children, roots)

    mesh_vertex_joints = mesh.vertex_joints.to(_device)
    mesh_vertex_weights = mesh.vertex_weights.to(dtype=torch.float32, device=_device)

    if fixed_joints is None:
        joint_mask = torch.zeros((num_joints, 1), dtype=torch.float32, device=_device)
        joint_mask[joints, :] = 1.0
    else:
        joint_mask = torch.ones((num_joints, 1), dtype=torch.float32, device=_device)
        joint_mask[fixed_joints, :] = 0.0

    faces = mesh.faces.to(_device)
    face_bone_ids, _ = get_face_to_bone_map(mesh_vertex_joints, mesh_vertex_weights, faces)
    # target_volume = get_mesh_volume(template_verts, faces)
    target_bone_vols, bone_counts = get_per_bone_volumes(
        template_verts, faces, face_bone_ids, num_joints
    )
    # Important: Mask out bones with 0 faces
    active_bone_mask = bone_counts > 0
    edge_indices, target_edge_lens_sq = get_edges_and_lengths(template_verts, faces)

    use_repulsion = (geo_dists is not None) and repulsion_weight > 0
    if use_repulsion:
        is_stranger = (geo_dists.to(device=_device) > is_stranger_threshold).to(dtype=torch.float32).detach()

    inverse_bind = (
        bones.__transpose(mesh.inverse_bind_matrices).to(dtype=torch.float32, device=_device).contiguous()
    )
    local_to_parent_transf = bones.get_local_to_parent_transforms(
        inverse_bind, joint_to_index, parents, num_joints
    )
    
    if initial_local_joint_transforms is not None:
        local_extra_rotation_mat = local_to_parent_transf.inverse() @ initial_local_joint_transforms
        local_extra_rotation = t3d.matrix_to_axis_angle(local_extra_rotation_mat[:, :3, :3])
    else:
        local_extra_rotation = torch.zeros((num_joints, 3), dtype=torch.float32, device=_device)
    local_extra_rotation.requires_grad = True
    optim = torch.optim.Adam([local_extra_rotation], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.5)

    if initial_global_transform is not None:
        _initial_glob_rot_mat = initial_global_transform[:3, :3]
        glob_scale = torch.mean(torch.linalg.norm(_initial_glob_rot_mat, dim=-1), dim=-1)
        glob_rot_mat = _initial_glob_rot_mat / glob_scale
        glob_t = initial_global_transform[:3, 3]
    else:
        glob_scale, glob_rot_mat, glob_t = bones.kabsch_umeyama(
            predicted_pose,
            template_verts[canonical_indices],
            scale=scale,
            reflection=False,
            rotation=True,
        )
    glob_rot = t3d.matrix_to_axis_angle(glob_rot_mat)
    glob_scale.requires_grad = True
    glob_rot.requires_grad = True
    glob_t.requires_grad = True
    _optim = torch.optim.Adam([glob_scale, glob_rot, glob_t], lr=1e-3)

    t = 0
    while True:
        # local_extra_rotation_mat = torch.zeros(num_joints, 4, 4, dtype=torch.float32)
        # local_extra_rotation_mat[:, :4, :4] = torch.eye(4, dtype=torch.float32)
        _local_extra_rotation_mat = t3d.axis_angle_to_matrix(local_extra_rotation * joint_mask)
        local_extra_rotation_mat = t3d.Rotate(_local_extra_rotation_mat).get_matrix()
        local_joint_transforms = local_to_parent_transf @ local_extra_rotation_mat

        global_joint_transforms = skin.compute_global_nodes_transforms_tree(
            local_joint_transforms, topo_order, parents, joint_to_index, transpose=True
        )
        # global_joint_transforms = skin.compute_global_nodes_transforms(
        #     local_joint_transforms, joints, mesh.nodes_parents_list, transpose=True
        # )
        template_pc = bones.articulate_template_mesh(
            bones.__transpose(global_joint_transforms),
            template_verts,
            bones.__transpose(inverse_bind),
            mesh_vertex_joints,
            mesh_vertex_weights,
        )

        # reconstruction_pc = predicted_pose
        # scale_factor, rotation, translation = kabsch_umeyama(
        #     predicted_pose,
        #     template_pc[canonical_indices],
        #     scale=scale,
        #     reflection=False,
        #     rotation=True,
        # )
        # global_transform = to_transform(scale_factor, rotation, translation)
        
        glob_rot_mat = t3d.axis_angle_to_matrix(glob_rot)
        global_transform = t3d.Rotate(glob_rot_mat).get_matrix()[0]
        # # .translate(glob_t)
        global_transform[:3, :3] *= glob_scale
        global_transform[:3, 3] = glob_t
        reconstruction_pc = bones.__transform(global_transform, predicted_pose)

        if loss_type == 'cd':
            data_loss = get_chamfer_dist(reconstruction_pc, template_pc, weights=cd_weights)
        elif loss_type == 'point_to_face':
            data_loss = point_mesh_face_distance(
                verts=template_pc,
                faces=faces,
                reconstruction_pc=reconstruction_pc,
            )
        elif loss_type in 'huber':
            dists = torch.linalg.norm(reconstruction_pc - template_pc[canonical_indices], dim=-1)
            data_loss = F.huber_loss(dists, torch.zeros_like(dists), reduction='mean', delta=delta)
            # data_loss = torch.sum(weights * _data_loss) / weight_sum
        elif loss_type == 'geman_mcclure':
            diff = reconstruction_pc - template_pc[canonical_indices]
            data_loss = geman_mcclure_loss(diff, delta, reduction='mean')
            # data_loss = torch.sum(weights * _data_loss) / weight_sum
        elif loss_type == 'cauchy':
            diff = reconstruction_pc - template_pc[canonical_indices]
            data_loss = cauchy_loss(diff, delta, reduction='mean')
            # data_loss = torch.sum(weights * _data_loss) / weight_sum
        else:
            raise ValueError(f"uknown loss type {loss_type}")
        
        angles = torch.linalg.norm(local_extra_rotation, dim=-1)
        angle_loss = torch.mean(angles*angles)
        # angle_loss = torch.mean(torch.sum(local_extra_rotation**2, dim=-1))
        # for idx, name in enumerate(mesh.node_names):
        #     print(idx, name, angles[idx])
        
        # current_volume = get_mesh_volume(template_pc, faces)
        # global_vol_loss = (current_volume - target_volume)**2
        current_bone_vols, _ = get_per_bone_volumes(
            template_pc, faces, face_bone_ids, num_joints
        )
        vol_diff = (current_bone_vols - target_bone_vols)[active_bone_mask]
        local_vol_loss = torch.mean(vol_diff*vol_diff)
        edge_loss = edge_length_loss(template_pc, edge_indices, target_edge_lens_sq)

        if use_repulsion:
            repulsion_loss = get_pairwise_repulsion_loss_knn(
                template_pc, 
                is_stranger, 
                radius=repulsion_radius, 
                subsample=subsample_repulsion,
            )
        else:
            repulsion_loss = torch.tensor([0], device=_device, dtype=torch.float32)
        
        decay_factor = max(1.0, 10.0 - 9.0 * (t / warmup_iters))

        # if t % 50 == 0:
        #     print(
        #         f"iter {t}: data={data_loss.item():.4f}, "
        #         f"angle={angle_loss.item():.4f}, "
        #         f"glob vol={global_vol_loss.item():.4f}, "
        #         f"local vol={local_vol_loss.item():.6f}, "
        #         f"edge={edge_loss.item():.8f}, "
        #         f"repulsion={repulsion_loss.item():6f}, "
        #         f"decay={decay_factor:.4f}, "
        #         f"lr={optim.param_groups[0]['lr']:.6f}"
        #     )
            # print(global_transform)
            # history.append((global_transform.detach(), local_joint_transforms.detach(), template_pc.detach()))

        loss = (
            data_loss
            + (angle_loss_weight * decay_factor * angle_loss)
            + (local_volume_loss_weight * decay_factor * local_vol_loss)
            + (edge_length_loss_weight * decay_factor * edge_loss)
            + (repulsion_weight * repulsion_loss)
        )
        # + (global_volume_loss_weight * decay_factor * global_vol_loss)
        optim.zero_grad()
        _optim.zero_grad()
        loss.backward()
        optim.step()
        _optim.step()
        scheduler.step()
        t += 1

        if t > num_iters:
            # fig, axs = plt.subplots(1, 2, figsize=(10, 6))
            # axs[0].plot(range(len(cd_hist)), cd_hist)
            # axs[1].plot(range(len(angle_hist)), angle_hist)
            joint_pos = global_joint_transforms[..., :3, 3].detach() # <- the matrices are transposed here
            return global_transform.detach(), local_joint_transforms.detach(), template_pc.detach(), joint_pos, angles.detach(), inlier_mask, canonical_indices


def get_chamfer_dist_batched(X, Y, seq_mask, weights=(0.5, 0.5), delta=0.1, reduction='sum'):
    if Y.ndim == 2:
        Y = Y.unsqueeze(0).expand(X.shape[0], -1, -1)
    
    dist = torch.cdist(X, Y) # (B, N, V)
    
    # X to Y
    min_dist_X_to_Y = torch.amin(dist, dim=-1) # (B, N)
    loss_X_to_Y = F.huber_loss(min_dist_X_to_Y, torch.zeros_like(min_dist_X_to_Y), reduction='none', delta=delta)
    loss_X_to_Y = (loss_X_to_Y * seq_mask).sum(dim=1) / (seq_mask.sum(dim=1) + 1e-8)
    
    # Y to X
    dist_masked = dist + (1 - seq_mask.unsqueeze(-1)) * 1e6
    min_dist_Y_to_X = torch.amin(dist_masked, dim=-2) # (B, V)
    loss_Y_to_X = F.huber_loss(min_dist_Y_to_X, torch.zeros_like(min_dist_Y_to_X), reduction='none', delta=delta)
    loss_Y_to_X = loss_Y_to_X.mean(dim=1)
    
    loss = weights[0] * loss_X_to_Y + weights[1] * loss_Y_to_X
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    return loss


def fp_sample(canon: torch.Tensor, verts: torch.Tensor, canonical_indices: torch.Tensor, geo_dists: torch.Tensor, K: int, mask: torch.Tensor | None = None):
    """
    Args:
        canon: (B, N, 3) or (N, 3) tensor of predicted points.
        verts: (M, 3) tensor of mesh vertices.
        canonical_indices: (B, N) or (N,) tensor of long/int, where assignments[i] is the index of the nearest vertex in V.
        geo_dists: (M, M) precomputed geodesic distance matrix.
        K: Number of points to sample.
        mask: (B, N) or (N,) boolean tensor of valid points to sample from (useful for padded batches).
        
    Returns:
        selected_indices: (B, K) or (K,) tensor of indices of the sampled points in P.
    """
    is_unbatched = canon.dim() == 2
    if is_unbatched:
        canon = canon.unsqueeze(0)
        canonical_indices = canonical_indices.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    B, N, _ = canon.shape
    device = canon.device

    # 1. Precompute the static Euclidean offsets for all points in P
    # E[b, i] = ||p_{b,i} - v_{a(b,i)}||
    assigned_vertices = verts[canonical_indices]
    E = torch.norm(canon - assigned_vertices, dim=-1) # Shape: (B, N)

    # 2. Initialize FPS trackers
    selected_indices = torch.zeros((B, K), dtype=torch.long, device=device)
    # Track the minimum distance from each point to the selected set
    min_dists = torch.full((B, N), float('inf'), device=device)
    
    if mask is not None:
        mask = mask.bool()
        min_dists[~mask] = -1.0

    # 3. Pick a random starting point for each batch element
    if mask is not None:
        # Vectorized random choice from valid elements: assign random scores, argmax
        random_scores = torch.rand((B, N), device=device)
        random_scores[~mask] = -1.0
        curr_idx = torch.argmax(random_scores, dim=1)
    else:
        curr_idx = torch.randint(0, N, (B,), device=device)

    selected_indices[:, 0] = curr_idx
    batch_indices = torch.arange(B, device=device)
    min_dists[batch_indices, curr_idx] = 0.0

    # 4. Vectorized FPS Loop
    for i in range(1, K):
        # We need the distances from all points in P to the newly selected point (curr_idx)
        a_curr = canonical_indices[batch_indices, curr_idx]
        
        # Look up geodesic distance from all assigned vertices to the current point's vertex
        geo_dists_to_curr = geo_dists[canonical_indices, a_curr.unsqueeze(1)] # Shape: (B, N)
        
        # Calculate full topology-aware distance
        dists_to_curr = E + geo_dists_to_curr + E[batch_indices, curr_idx].unsqueeze(1)
        
        # Fix the self-distance edge case (distance to itself is exactly 0)
        dists_to_curr[batch_indices, curr_idx] = 0.0
        
        # Update the minimum distance to the selected set
        min_dists = torch.minimum(min_dists, dists_to_curr)
        if mask is not None:
            min_dists[~mask] = -1.0
        
        # The next point is the one farthest from the selected set
        curr_idx = torch.argmax(min_dists, dim=1)
        selected_indices[:, i] = curr_idx

    if is_unbatched:
        return selected_indices.squeeze(0)
    return selected_indices


def refine_kinematic_chain_batched(
    predicted_canon: torch.Tensor,
    predicted_pose: torch.Tensor,
    seq_mask: torch.Tensor,
    mesh_proc_data: ShapeProcessingData,
    initial_local_joint_transforms: torch.Tensor | None = None,
    initial_global_transform: torch.Tensor | None = None,
    distance_threshold: float = 0.3,
    scale: bool = True,
    num_iters: int = 200,
    loss_type: str = 'huber',
    delta: float = 0.1,
    angle_loss_weight: float = 1e-5,
    local_volume_loss_weight: float = 50.0,
    edge_length_loss_weight: float = 100.0,
    scale_sym_loss_weight: float = 10.0,
    scale_bounds_loss_weight: float = 1000.0,
    scale_reg_loss_weight: float = 0.1,
    repulsion_weight: float = 0,
    repulsion_radius: float = 0.05,
    geo_dists: torch.Tensor | None = None,
    is_stranger_threshold: float = 0.3,
    cd_weights: tuple[float] = (0.5, 0.5),
    warmup_iters: float = 25.0,
    lr=1e-2,
    subsample_repulsion: bool = False,
    fixed_joints: list[int] | None = None,
    return_metrics: bool = False,
    optimize_bone_scales: bool = True,
    subsample: int | None = None,
):
    predicted_canon = predicted_canon.detach()
    _device = predicted_canon.device
    B, N, _ = predicted_canon.shape

    template_verts = mesh_proc_data.vertices
    
    # Correspondence finding
    dist = torch.cdist(predicted_canon, template_verts.unsqueeze(0).expand(B, -1, -1))
    _dist, canonical_indices = torch.min(dist, dim=2) # (B, N)
    inlier_mask = (_dist < distance_threshold) & seq_mask.bool()
    
    full_canonical_indices = canonical_indices.clone()
    full_inlier_mask = inlier_mask
    # print("refine inlier count:", full_inlier_mask.sum())

    if subsample is not None:
        sampled_ind = fp_sample(
            canon=predicted_canon,
            verts=mesh_proc_data.vertices,
            canonical_indices=canonical_indices,
            geo_dists=geo_dists,
            K=subsample,
            mask=inlier_mask,
        )
        batch_idx_sample = torch.arange(B, device=_device).view(B, 1).expand(B, subsample)
        
        predicted_pose = predicted_pose[batch_idx_sample, sampled_ind]
        canonical_indices = canonical_indices[batch_idx_sample, sampled_ind]
        inlier_mask = inlier_mask[batch_idx_sample, sampled_ind]
        assert inlier_mask.all()
        predicted_canon = predicted_canon[batch_idx_sample, sampled_ind]
        N = subsample
    else:
        sampled_ind = None
        
    face_bone_ids, _ = get_face_to_bone_map(
        mesh_proc_data.vertex_joints, mesh_proc_data.vertex_weights, mesh_proc_data.faces
    )
    
    target_bone_vols, bone_counts = get_per_bone_volumes(
        template_verts, mesh_proc_data.faces, face_bone_ids, mesh_proc_data.num_joints
    )
    active_bone_mask = bone_counts > 0
    edge_indices, target_edge_lens_sq = get_edges_and_lengths(template_verts, mesh_proc_data.faces)

    use_repulsion = repulsion_weight > 0
    if use_repulsion:
        is_stranger = (geo_dists.to(device=_device) > is_stranger_threshold).to(dtype=torch.float32).detach()

    inverse_bind = mesh_proc_data.inverse_bind_matrices
    local_to_parent_transf = mesh_proc_data.local_to_parent_transforms

    if initial_local_joint_transforms is not None:
        local_extra_rotation_mat = local_to_parent_transf.inverse() @ initial_local_joint_transforms
        local_extra_rotation = t3d.matrix_to_axis_angle(local_extra_rotation_mat[..., :3, :3])
    else:
        local_extra_rotation = torch.zeros((B, mesh_proc_data.num_joints, 3), dtype=torch.float32, device=_device)
    local_extra_rotation.requires_grad = True

    joint_mask = torch.zeros((1, mesh_proc_data.num_joints, 1), dtype=torch.float32, device=_device)
    joint_mask[:, mesh_proc_data.joints, :] = 1.0
    if fixed_joints is not None:
        joint_mask[:, fixed_joints, :] = 0.0
    # print("JOINTS MASK:", joint_mask[0, :, 0].sum(), joint_mask[0, :, 0].mean())

    if initial_global_transform is not None:
        _initial_glob_rot_mat = initial_global_transform[..., :3, :3]
        glob_scale = torch.mean(torch.linalg.norm(_initial_glob_rot_mat, dim=-1), dim=-1)
        glob_rot_mat = _initial_glob_rot_mat / glob_scale.unsqueeze(-1).unsqueeze(-1)
        glob_t = initial_global_transform[..., :3, 3]
    else:
        batch_idx = torch.arange(B, device=_device).view(B, 1).expand(B, N)
        matched_template = template_verts.unsqueeze(0).expand(B, -1, -1)[batch_idx, canonical_indices]
        
        glob_scale, glob_rot_mat, glob_t = bones.kabsch_umeyama_batched(
            predicted_pose,
            matched_template,
            sequence_mask=inlier_mask,
            scale=scale,
            reflection=False,
            rotation=True,
        )
        glob_scale = glob_scale.squeeze(-1)

    # global_transform = bones.to_transform(glob_scale, glob_rot_mat, glob_t)
    # predicted_pose_aligned = bones.__transform(global_transform, predicted_pose)
    
    if optimize_bone_scales:
        # bone_scales = bones.initialize_bone_scales(
        #     reconstruction_pc=predicted_pose_aligned,
        #     canonical_indices=canonical_indices,
        #     inlier_mask=inlier_mask,
        #     mesh_proc_data=mesh_proc_data,
        # )
        bone_scales = torch.ones((B, mesh_proc_data.num_joints), dtype=torch.float32, device=_device)
        bone_scales.requires_grad = True
        optim = torch.optim.Adam([local_extra_rotation, bone_scales], lr=lr)
    else:
        bone_scales = torch.ones((B, mesh_proc_data.num_joints), dtype=torch.float32, device=_device)
        optim = torch.optim.Adam([local_extra_rotation], lr=lr)
        
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.5)

    counts = mesh_proc_data.joint_affected_vertices.to(dtype=torch.float32, device=_device)
    active_mask = joint_mask[0, :, 0] > 0
    active_counts = counts * active_mask.float()
    counts_sum = active_counts.sum()
    if counts_sum > 0:
        normalized_weights = active_counts / counts_sum
    else:
        num_active = active_mask.sum()
        normalized_weights = active_mask.float() / (num_active + 1e-8)

    glob_rot = t3d.matrix_to_axis_angle(glob_rot_mat)
    glob_scale.requires_grad = True
    glob_rot.requires_grad = True
    glob_t.requires_grad = True
    _optim = torch.optim.Adam([glob_scale, glob_rot, glob_t], lr=1e-3)

    t = 0
    while True:
        # Wrap parameters to canonical range [-pi, pi] to ensure correct regularization
        with torch.no_grad():
            current_angles = torch.linalg.norm(local_extra_rotation, dim=-1)
            mask = current_angles > torch.pi
            if mask.any():
                # w' = w * (1 - 2pi / |w|) -> flips axis and sets mag to 2pi - |w|
                angle_scale = 1.0 - (2 * torch.pi) / (current_angles[mask] + 1e-8)
                local_extra_rotation.data[mask] *= angle_scale.unsqueeze(-1)
                
            # if optimize_bone_scales:
            #     bone_scales[joint_mask.squeeze(-1).expand(B, -1) < 1.0] = 1.0

        _local_extra_rotation_mat = t3d.axis_angle_to_matrix(local_extra_rotation * joint_mask)
        # _local_extra_rotation_mat = t3d.axis_angle_to_matrix(local_extra_rotation)
        local_extra_rotation_mat = (
            t3d.Rotate(_local_extra_rotation_mat.flatten(0, 1))
            .get_matrix().view(B, mesh_proc_data.num_joints, 4, 4)
        )
        local_joint_transforms_unscaled = local_to_parent_transf @ local_extra_rotation_mat
        
        global_joint_transforms_unscaled = skin.compute_global_nodes_transforms_tree_batched(
            local_joint_transforms_unscaled, mesh_proc_data.topo_order, mesh_proc_data.parents, mesh_proc_data.joint_to_index, transpose=True
        )

        template_pc_unscaled = bones.articulate_template_mesh(
            bones.__transpose(global_joint_transforms_unscaled),
            template_verts,
            bones.__transpose(inverse_bind),
            mesh_proc_data.vertex_joints,
            mesh_proc_data.vertex_weights,
        )
        
        local_joint_transforms_scaled = bones.scale_local_joint_transforms(
            local_joint_transforms=local_joint_transforms_unscaled,
            bone_scales=bone_scales,
            shape_data=mesh_proc_data,
        )

        global_joint_transforms_scaled = skin.compute_global_nodes_transforms_tree_batched(
            local_joint_transforms_scaled, mesh_proc_data.topo_order, mesh_proc_data.parents, mesh_proc_data.joint_to_index, transpose=True
        )

        template_pc_scaled = bones.articulate_template_mesh(
            bones.__transpose(global_joint_transforms_scaled),
            template_verts,
            bones.__transpose(inverse_bind),
            mesh_proc_data.vertex_joints,
            mesh_proc_data.vertex_weights,
        )

        glob_rot_mat = t3d.axis_angle_to_matrix(glob_rot)
        global_transform = t3d.Rotate(glob_rot_mat).get_matrix()
        global_transform[:, :3, :3] *= glob_scale.view(B, 1, 1)
        global_transform[:, :3, 3] = glob_t
        reconstruction_pc = bones.__transform(global_transform, predicted_pose)

        batch_idx = torch.arange(B, device=_device).view(B, 1).expand(B, N)
        matched_template = template_pc_scaled[batch_idx, canonical_indices]
        diff = reconstruction_pc - matched_template
        if loss_type == 'cd':
            data_loss = get_chamfer_dist_batched(reconstruction_pc, template_pc_scaled, inlier_mask.float(), weights=cd_weights)
        elif loss_type in ["huber", "geman_mcclure", "cauchy"]:
            if loss_type == "huber":
                dists = torch.linalg.norm(diff, dim=-1)
                _data_loss = F.huber_loss(dists, torch.zeros_like(dists), reduction='none', delta=delta)
            elif loss_type == 'geman_mcclure':
                _data_loss = geman_mcclure_loss(diff, delta, reduction='none')
            elif loss_type == 'cauchy':
                _data_loss = cauchy_loss(diff, delta, reduction='none')
            data_loss = ((_data_loss * inlier_mask).sum(dim=1) / (inlier_mask.sum(dim=1) + 1e-8)).sum()
        else:
            raise ValueError(f"uknown loss type {loss_type}")

        angle_loss = (local_extra_rotation**2).sum(dim=-1).mean(dim=1).sum()
        # angle_loss = ((local_extra_rotation**2).sum(dim=-1) * normalized_weights.unsqueeze(0)).sum(dim=1).sum()
        angles = torch.linalg.norm(local_extra_rotation.detach(), dim=-1)
        
        current_bone_vols, _ = get_per_bone_volumes_batched(
            template_pc_unscaled, mesh_proc_data.faces, face_bone_ids, mesh_proc_data.num_joints
        )
        vol_diff = (current_bone_vols - target_bone_vols.unsqueeze(0))[:, active_bone_mask]
        local_vol_loss = (vol_diff*vol_diff).mean(dim=1).sum()
        
        edge_loss = edge_length_loss_batched(template_pc_unscaled, edge_indices, target_edge_lens_sq)

        if use_repulsion:
            repulsion_loss = get_pairwise_repulsion_loss_knn_batched(
                template_pc_unscaled, 
                is_stranger, 
                radius=repulsion_radius, 
                subsample=subsample_repulsion,
            )
        else:
            repulsion_loss = torch.tensor([0], device=_device, dtype=torch.float32)
        
        if optimize_bone_scales:
            scale_sym_diff = torch.log(bone_scales.clamp(min=1e-4)) - torch.log(bone_scales[:, mesh_proc_data.joint_sym_corresp].clamp(min=1e-4))
            scale_sym_loss = (
                (scale_sym_diff * scale_sym_diff) * mesh_proc_data.has_joint_sym_corresp.unsqueeze(0)
            ).mean(dim=1).sum()
            
            scale_bounds_diff = torch.relu(0.5 - bone_scales) + torch.relu(bone_scales - 2.0)
            scale_bounds_loss = (scale_bounds_diff * scale_bounds_diff).mean(dim=1).sum()
            
            scale_reg_diff = torch.log(bone_scales.clamp(min=1e-4))
            scale_reg_loss = (scale_reg_diff * scale_reg_diff).mean(dim=1).sum()
        else:
            scale_sym_loss = 0.0
            scale_bounds_loss = 0.0
            scale_reg_loss = 0.0
        
        decay_factor = max(1.0, 10.0 - 9.0 * (t / warmup_iters))

        loss = (
            data_loss
            + (angle_loss_weight * decay_factor * angle_loss)
            + (local_volume_loss_weight * decay_factor * local_vol_loss)
            + (edge_length_loss_weight * decay_factor * edge_loss)
            + (repulsion_weight * repulsion_loss) # * decay factor here as well?
        )
        if optimize_bone_scales:
            loss += (scale_sym_loss_weight * scale_sym_loss) + (scale_bounds_loss_weight * scale_bounds_loss) + (scale_reg_loss_weight * scale_reg_loss)

        if t < num_iters:
            optim.zero_grad()
            _optim.zero_grad()
            loss.backward()
            optim.step()
            _optim.step()
            scheduler.step()

        t += 1

        if t > num_iters:

            # if node_names is not None:
            #     for node_idx in range(bone_scales.shape[1]):
            #         print(node_names[node_idx], bone_scales[0, node_idx].item())

            joint_pos = global_joint_transforms_unscaled[..., :3, 3].detach()
            
            with torch.no_grad():
                batch_idx = torch.arange(B, device=_device).view(B, 1).expand(B, N)
                matched_template_unscaled = template_pc_unscaled[batch_idx, canonical_indices]
                
                glob_scale_unscaled, glob_rot_mat_unscaled, glob_t_unscaled = bones.kabsch_umeyama_batched(
                    predicted_pose,
                    matched_template_unscaled,
                    sequence_mask=inlier_mask,
                    scale=scale,
                    reflection=False,
                    rotation=True,
                )
                
                global_transform_unscaled = bones.to_transform(
                    glob_scale_unscaled.view(B, 1, 1), 
                    glob_rot_mat_unscaled, 
                    glob_t_unscaled
                )
            
            if return_metrics:
                with torch.no_grad():
                    batch_idx = torch.arange(B, device=_device).view(B, 1).expand(B, N)
                    matched_template = template_pc_scaled[batch_idx, canonical_indices]
                    reconstruction_pc = bones.__transform(global_transform, predicted_pose)
                    diff = reconstruction_pc - matched_template
                    
                    if loss_type == 'cd':
                        final_data_loss = get_chamfer_dist_batched(reconstruction_pc, template_pc_scaled, inlier_mask.float(), weights=cd_weights, delta=delta, reduction='none')
                    elif loss_type in ["huber", "geman_mcclure", "cauchy"]:
                        if loss_type == "huber":
                            dists = torch.linalg.norm(diff, dim=-1)
                            _data_loss = F.huber_loss(dists, torch.zeros_like(dists), reduction='none', delta=delta)
                        elif loss_type == 'geman_mcclure':
                            _data_loss = geman_mcclure_loss(diff, delta, reduction='none')
                        elif loss_type == 'cauchy':
                            _data_loss = cauchy_loss(diff, delta, reduction='none')
                        final_data_loss = ((_data_loss * inlier_mask).sum(dim=1) / (inlier_mask.sum(dim=1) + 1e-8))
                    
                    final_angle_loss = (local_extra_rotation**2).sum(dim=-1).mean(dim=1)
                    # final_angle_loss = ((local_extra_rotation**2).sum(dim=-1) * normalized_weights.unsqueeze(0)).sum(dim=1)
                    
                    current_bone_vols, _ = get_per_bone_volumes_batched(
                        template_pc_unscaled, mesh_proc_data.faces, face_bone_ids, mesh_proc_data.num_joints
                    )
                    vol_diff = (current_bone_vols - target_bone_vols.unsqueeze(0))[:, active_bone_mask]
                    final_local_vol_loss = (vol_diff*vol_diff).mean(dim=1)
                    
                    final_edge_loss = edge_length_loss_batched(template_pc_unscaled, edge_indices, target_edge_lens_sq, reduction='none')
                    
                    if use_repulsion:
                        final_repulsion_loss = get_pairwise_repulsion_loss_knn_batched(
                            template_pc_unscaled, is_stranger, radius=repulsion_radius, subsample=subsample_repulsion, reduction='none'
                        )
                    else:
                        final_repulsion_loss = torch.zeros(B, device=_device, dtype=torch.float32)

                    if optimize_bone_scales:
                        scale_sym_diff = torch.log(bone_scales.clamp(min=1e-4)) - torch.log(bone_scales[:, mesh_proc_data.joint_sym_corresp].clamp(min=1e-4))
                        # print("bone scales range:", bone_scales.min().item(), bone_scales.max().item())
                        final_scale_sym_loss = (
                            (scale_sym_diff * scale_sym_diff) * mesh_proc_data.has_joint_sym_corresp.unsqueeze(0)
                        ).mean(dim=1)
                        
                        scale_bounds_diff = torch.relu(0.5 - bone_scales) + torch.relu(bone_scales - 2.0)
                        final_scale_bounds_loss = (scale_bounds_diff * scale_bounds_diff).mean(dim=1)
                        
                        final_scale_reg_loss = (torch.log(bone_scales.clamp(min=1e-4)) ** 2).mean(dim=1)
                    else:
                        final_scale_sym_loss = torch.zeros(B, device=_device, dtype=torch.float32)
                        final_scale_bounds_loss = torch.zeros(B, device=_device, dtype=torch.float32)
                        final_scale_reg_loss = torch.zeros(B, device=_device, dtype=torch.float32)

                    final_total_loss = (
                        final_data_loss
                        + (angle_loss_weight * final_angle_loss)
                        + (local_volume_loss_weight * final_local_vol_loss)
                        + (edge_length_loss_weight * final_edge_loss)
                        + (repulsion_weight * final_repulsion_loss)
                    )
                    if optimize_bone_scales:
                        final_total_loss += (scale_sym_loss_weight * final_scale_sym_loss) + (scale_bounds_loss_weight * final_scale_bounds_loss) + (scale_reg_loss_weight * final_scale_reg_loss)

                    metrics = {
                        'data_loss': final_data_loss,
                        'angle_loss': final_angle_loss,
                        'volume_loss': final_local_vol_loss,
                        'edge_loss': final_edge_loss,
                        'repulsion_loss': final_repulsion_loss,
                        'scale_sym_loss': final_scale_sym_loss,
                        'scale_bounds_loss': final_scale_bounds_loss,
                        'scale_reg_loss': final_scale_reg_loss,
                        'total_loss': final_total_loss,
                    }
            else:
                metrics = None

            # joint_pos, angles.detach(), full_inlier_mask, full_canonical_indices, metrics
            return {
                'global_transform': global_transform_unscaled.detach(),
                'local_joint_transforms': local_joint_transforms_unscaled.detach(),
                'fitted_shape': template_pc_unscaled.detach(),
                'global_transform_scaled': global_transform.detach(),
                'local_joint_transforms_scaled': local_joint_transforms_scaled.detach(),
                'fitted_shape_scaled': template_pc_scaled.detach(),
                'joint_pos': joint_pos,
                'joint_angles': angles.detach(),
                'inlier_mask': full_inlier_mask,
                'canonical_indices': full_canonical_indices,
                'sampled_ind': sampled_ind,
                'metrics': metrics,
            }
