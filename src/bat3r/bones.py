
import torch
import einops
from scipy.spatial import KDTree

from bat3r import skin, trees
from bat3r.skin import OneMeshGltf


class ShapeProcessingData:
    vertices: torch.Tensor
    joints: list[int]
    num_joints: int
    local_joint_transforms: torch.Tensor
    vertex_joints: torch.Tensor
    vertex_weights: torch.Tensor
    inverse_bind_matrices: torch.Tensor
    canonical_joints: torch.Tensor
    joint_affected_vertices: torch.Tensor

    def __init__(self, shape: OneMeshGltf, device, sym_corresp_threshold: float = 0.005, joint_threshold: float = 0.5):
        self.device = device
        inverse_bind_matrices = shape.inverse_bind_matrices.to(dtype=torch.float32, device=device)
        rest_global_joint_transf = torch.inverse(inverse_bind_matrices)
        self.canonical_joints = rest_global_joint_transf[..., 3, :3]

        parents, children, descendants, roots = trees.recover_parents(shape.nodes_parents_list)
        self.children = children
        self.parents = parents
        self.descendants = descendants
        self.topo_order = trees.topsort(children, roots)
        self.joints = shape.joints.tolist()
        self.joint_to_index = {joint: idx for idx, joint in enumerate(self.joints)}
        
        node_depths = {}
        for node in self.topo_order:
            p = self.parents[node]
            node_depths[node] = 0 if p is None else node_depths[p] + 1
        self.joint_depths = torch.tensor([node_depths[j] for j in self.joints], dtype=torch.float32, device=device)

        self.inverse_bind_matrices = self.__transpose(inverse_bind_matrices).contiguous()
        self.bind_matrices = self.__transpose(rest_global_joint_transf).contiguous()
        self.num_joints = shape.local_joint_transforms.shape[0]
        self.local_to_parent_transforms = get_local_to_parent_transforms(
            self.inverse_bind_matrices, self.joint_to_index, parents, self.num_joints
        )
        self.local_to_parent_inv = torch.linalg.inv(self.local_to_parent_transforms)

        self.vertices = shape.vertices.to(dtype=torch.float32, device=device)
        self.faces = shape.faces.to(device)
        self.vertex_joints = shape.vertex_joints.to(device)
        self.vertex_weights = shape.vertex_weights.to(dtype=torch.float32, device=device)

        self.vertices_sym = self.vertices.clone()
        self.vertices_sym[..., 0] *= -1
        dist = torch.cdist(self.vertices_sym, self.vertices)
        nearest_sym_dist, self.sym_corresp = torch.min(dist, dim=1)
        self.has_sym_corresp = nearest_sym_dist < sym_corresp_threshold
        # print(self.has_sym_corresp.float().mean())
        
        canonical_joints_sym = self.canonical_joints.clone()
        canonical_joints_sym[..., 0] *= -1
        dist_joints = torch.cdist(canonical_joints_sym, self.canonical_joints)

        # CHIMP ONLY:
        # Tie-breakers for overlapping joints:
        # 1. Prefer joints with the same topological depth
        # depth_penalty = torch.abs(self.joint_depths.unsqueeze(1) - self.joint_depths.unsqueeze(0)) * 1e-4
        # 2. Prefer self-matching for central joints
        # self_penalty = (1.0 - torch.eye(len(self.joints), device=device)) * 1e-5
        
        # dist_joints_penalized = dist_joints + depth_penalty + self_penalty
        # _, joint_sym_corresp = torch.min(dist_joints_penalized, dim=1)

        # REST:
        joint_sym_corresp = torch.zeros(len(self.joints), dtype=torch.long, device=device)
        for i in range(len(self.joints)):
            min_dist = dist_joints[i].min().item()
            candidates = torch.where(dist_joints[i] <= min_dist + 1e-4)[0].tolist()

            best_j = candidates[0]
            best_prefix_len = -1

            name_i = shape.node_names[self.joints[i]]
            if len(candidates) > 1:
                for j in candidates:
                    name_j = shape.node_names[self.joints[j]]
                    prefix_len = 0
                    for c1, c2 in zip(name_i, name_j):
                        if c1 == c2:
                            prefix_len += 1
                        else:
                            break
                    
                    if prefix_len > best_prefix_len:
                        best_prefix_len = prefix_len
                        best_j = j
                    elif prefix_len == best_prefix_len:
                        if dist_joints[i, j] < dist_joints[i, best_j]:
                            best_j = j
                        
            joint_sym_corresp[i] = best_j
        
        # Use the original distances to check the threshold limit
        nearest_joint_sym_dist = dist_joints[torch.arange(len(self.joints), device=device), joint_sym_corresp]
        has_joint_sym_corresp = nearest_joint_sym_dist < (sym_corresp_threshold * 10.0) # slightly relaxed for skeleton rigs
        self.has_joint_sym_corresp = torch.full((self.num_joints,), False, dtype=torch.bool, device=device)
        self.joint_sym_corresp = torch.arange(self.num_joints, device=device)
        for joint_idx, node_idx in enumerate(self.joints):
            self.has_joint_sym_corresp[node_idx] = has_joint_sym_corresp[joint_idx]
            self.joint_sym_corresp[node_idx] = self.joints[joint_sym_corresp[joint_idx]]
        print(self.has_joint_sym_corresp.float().mean())

        self.joint_affected_vertices = torch.zeros(self.num_joints, dtype=torch.long, device=device)
        for node_idx in self.joints:
            target_nodes = [node_idx] + [d for d, _ in self.descendants.get(node_idx, [])]
            target_joint_indices = [self.joint_to_index[n] for n in target_nodes if n in self.joint_to_index]
            if not target_joint_indices:
                continue
            target_joints_tensor = torch.tensor(target_joint_indices, device=device)
            is_target_joint = (self.vertex_joints[:, :, None] == target_joints_tensor).any(dim=-1)
            is_above_threshold = self.vertex_weights > joint_threshold
            affected_mask = (is_target_joint & is_above_threshold).any(dim=-1)
            self.joint_affected_vertices[node_idx] = affected_mask.sum()
            # print(shape.node_names[node_idx], affected_mask.sum())

    def __transpose(self, tensor: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(tensor, "... c r -> ... r c")


def to_transform(
    scale_factor: float, rotation: torch.Tensor, translation: torch.Tensor
):
    matrix = torch.zeros(*rotation.shape[:-2], 4, 4, device=rotation.device)
    matrix[..., :, :] = torch.eye(4, device=rotation.device)

    matrix[..., :3, 3] = translation
    matrix[..., :3, :3] = scale_factor * rotation[..., :3, :3]
    return matrix


def __transpose(tensor: torch.Tensor) -> torch.Tensor:
    return einops.rearrange(tensor, "... c r -> ... r c")


def __transform(transform: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return vector @ __transpose(transform[..., :3, :3]) + transform[..., :3, 3].unsqueeze(-2)


def kabsch_umeyama(
    a: torch.Tensor, b: torch.Tensor, rotation: bool, scale: bool, reflection: bool, translation: bool = True,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """
    calculates the optimal rigid transform from a to b
    b = [sR | t] a

    umeyama's algorithm
    https://en.wikipedia.org/wiki/Kabsch_algorithm
    https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/

    a: (N, 3)
    b: (N, 3)
    rotation: bool - compute rotation?
    scale: bool - compute scale?
    reflection: bool - allow reflection?

    returns:
        scale_factor: float - scale factor
        rotation: (3, 3) - rotation matrix
        translation: (3,) - translation vector
    """

    assert a.shape == b.shape, "a,b must have the same shape"

    *extra, n, m = a.shape
    device = a.device

    if translation:
        a_mean = a.mean(axis=-2)
        b_mean = b.mean(axis=-2)
    else:
        a_mean = torch.zeros_like(a.mean(axis=-2))
        b_mean = torch.zeros_like(b.mean(axis=-2))
    
    A = a - a_mean
    B = b - b_mean

    # normalized covariance <(3, n) @ (n, 3)>
    covariance = (__transpose(B) @ A) / n

    # Add epsilon for numerical stability (handles N=1 case or collinear points)
    covariance = covariance + 1e-6 * torch.eye(3, device=device)

    U, D, Vh = torch.linalg.svd(covariance)

    def eye():
        _eye = torch.zeros(*extra, 3, 3, device=device)
        _eye[..., :, :] = torch.eye(3, device=device)
        return _eye

    # correct signs, no reflection
    Signs = eye()
    if not reflection:
        Signs[..., 2, 2] = torch.sign(torch.linalg.det(U) * torch.linalg.det(Vh))

    rotation = U @ Signs @ Vh if rotation else eye()

    var_A = (A**2).sum(axis=-1).mean(axis=-1)
    scale_factor = torch.trace(torch.diag(D) @ Signs) / var_A if scale else 1

    translation = b_mean - scale_factor * rotation @ a_mean
    return scale_factor, rotation, translation


def kabsch_umeyama_batched(
    a: torch.Tensor, 
    b: torch.Tensor, 
    sequence_mask: torch.Tensor, 
    rotation: bool = True, 
    scale: bool = True, 
    reflection: bool = False, 
    translation: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched Kabsch-Umeyama algorithm with mask support.
    
    Args:
        a: (B, N, 3) Source point cloud
        b: (B, N, 3) Target point cloud
        sequence_mask: (B, N, 1) or (B, N) Indicates valid points (1.0) vs padding (0.0)
        
    Returns:
        scale_factor: (B, 1)
        rotation: (B, 3, 3)
        translation: (B, 3)
    """
    
    assert a.shape == b.shape, "a and b must have the same shape"

    if sequence_mask.dim() == 2:
        sequence_mask = sequence_mask.unsqueeze(-1)
    sequence_mask = sequence_mask.float()
    
    B_size, N, D = a.shape
    device = a.device

    weights = sequence_mask.sum(dim=1)
    # Avoid division by zero/small numbers by clamping, but keep track of valid batches
    valid_mask = weights > 1e-4
    weights_clamped = torch.where(valid_mask, weights, torch.ones_like(weights))
    
    if translation:
        a_mean = (a * sequence_mask).sum(dim=1) / weights_clamped
        b_mean = (b * sequence_mask).sum(dim=1) / weights_clamped
    else:
        a_mean = torch.zeros((B_size, 3), device=device)
        b_mean = torch.zeros((B_size, 3), device=device)

    A = (a - a_mean.unsqueeze(1)) * sequence_mask
    B = (b - b_mean.unsqueeze(1)) * sequence_mask
    
    covariance = (__transpose(B) @ A) / weights_clamped.unsqueeze(-1)
    # For invalid batches, covariance might be garbage or zero. We'll mask the output later, but SVD needs to be safe.
    
    # Add epsilon for numerical stability (handles N=1 case or collinear points)
    covariance = covariance + 1e-6 * torch.eye(3, device=device).unsqueeze(0)

    U, S, Vh = torch.linalg.svd(covariance)

    Signs = torch.eye(3, device=device).unsqueeze(0).repeat(B_size, 1, 1)
    
    if not reflection:
        det_U = torch.linalg.det(U)
        det_Vh = torch.linalg.det(Vh)
        # Set the last diagonal element to sign(det(U) * det(Vh))
        Signs[:, 2, 2] = torch.sign(det_U * det_Vh)

    if rotation:
        R = U @ Signs @ Vh
    else:
        R = torch.eye(3, device=device).unsqueeze(0).repeat(B_size, 1, 1)

    # Mask out invalid rotations (set to Identity)
    R = torch.where(valid_mask.view(B_size, 1, 1), R, torch.eye(3, device=device).unsqueeze(0))

    # 7. Compute Scale
    if scale:
        var_A = (A ** 2).sum(dim=(1, 2)) / weights_clamped.squeeze(-1)
        # Trace of (S * Signs)
        # S is vector of singular values. Signs is diagonal.
        # This effectively calculates trace(D @ Signs)
        trace_S = (S * Signs.diagonal(dim1=-2, dim2=-1)).sum(dim=-1)
        scale_factor = trace_S / var_A
        scale_factor = scale_factor.unsqueeze(-1) # (B, 1)
    else:
        scale_factor = torch.ones((B_size, 1), device=device)

    # Mask out invalid scales (set to 1.0)
    scale_factor = torch.where(valid_mask.view(B_size, 1), scale_factor, torch.ones_like(scale_factor))

    t = b_mean - (scale_factor.unsqueeze(-1) * (R @ a_mean.unsqueeze(-1))).squeeze(-1)
    # Mask out invalid translations (set to 0.0 - though b_mean/a_mean are likely 0 already if masked)
    t = torch.where(valid_mask.view(B_size, 1), t, torch.zeros_like(t))
    
    return scale_factor, R, t


def get_point_correspondence(
    predicted_canon: torch.Tensor,
    vertex_lookup: callable,
):
    """
    get the correspondence from the canonical prediction and canonical-space mesh

    args:
        predicted_canon: (N, 3) - canonical vertices of the predicted mesh
        vertex_lookup: callable - lookup function that returns the indices of the closest vertices in the target canonical vertices

    returns:
        canonical_indices: (P,) - indices of the closest vertices in the target canonical vertices
            note: P <= N, == sum(inlier_mask)
        inlier_mask: (N,) - boolean mask of the valid vertices
    """
    with torch.no_grad():
        canonical_indices, inlier_mask = vertex_lookup(predicted_canon)

    return canonical_indices, inlier_mask


def estimate_joints(
    predicted_canon: torch.Tensor,
    predicted_pose: torch.Tensor,
    skinned_mesh: OneMeshGltf,
    vertex_lookup: callable,
    joint_threshold: float,
    scale: bool,
):
    """
    1. get the correspondence from the canonical prediction and canonical-space mesh
    2. estimate the global joint transforms from the prior mesh (inverse_bind @ skinned_mesh.vertices) to posed point cloud
    3. apply to the prior mesh, and return
    """

    assert predicted_canon.shape[-1] == 3, "must be shape (P 3)"
    assert predicted_pose.shape[-1] == 3, "must be shape (P 3)"
    assert len(predicted_canon.shape) == 2, "must be shape (P 3)"
    assert len(predicted_pose.shape) == 2, "must be shape (P 3)"
    assert torch.allclose(
        skinned_mesh.inverse_bind_matrices[:, 3, :3],
        torch.zeros(
            skinned_mesh.inverse_bind_matrices.shape[0],
            1,
            3,
            dtype=torch.float64,
            device=skinned_mesh.inverse_bind_matrices.device,
        ),
    ), "must be column major"

    # strip gradients from the canonical points
    predicted_canon = predicted_canon.detach()

    canonical_indices, inlier_mask = get_point_correspondence(
        predicted_canon, vertex_lookup
    )

    _device = predicted_canon.device
    skin_points = skinned_mesh.vertices.to(dtype=torch.float32, device=_device)[canonical_indices]
    predicted_pose = predicted_pose[inlier_mask]

    global_joint_transforms, success_mask = _estimate_global_joints(
        skin_points,
        predicted_pose,
        canonical_indices,
        skinned_mesh.joints,
        skinned_mesh.inverse_bind_matrices.to(dtype=torch.float32, device=_device),
        skinned_mesh.vertex_joints,
        skinned_mesh.vertex_weights,
        joint_threshold,
        scale,
    )

    return global_joint_transforms, success_mask


def _estimate_global_joints(
    template_pointcloud: torch.Tensor,
    reconstruction_pointcloud: torch.Tensor,
    canonical_indices: torch.Tensor,
    mesh_joints: torch.Tensor,
    inverse_bind: torch.Tensor,
    mesh_vertex_joints: torch.Tensor,
    mesh_vertex_weights: torch.Tensor,
    joint_threshold: float,
    scale: bool,
):
    """
    estimate the global joint transforms from the prior mesh (inverse_bind @ skinned_mesh.vertices) to posed point cloud

    args:
        template_pointcloud: (N, 3) - canonical vertices of the prior mesh
        reconstruction_pointcloud: (P, 3) - posed vertices of the reconstruction
        canonical_indices: (P,) - indices of the closest vertices in the target canonical vertices
        mesh_joints: (J,) - indices of the joints in the prior mesh
        inverse_bind: (J, 4, 4) - inverse bind matrices of the prior mesh
        mesh_vertex_joints: (V,) - indices of the joints for each vertex in the prior mesh
        mesh_vertex_weights: (V,) - weights for each vertex in the prior mesh
        joint_threshold: float - threshold for the joint weights

    returns:
        joint_transforms_est: (J, 4, 4) - estimated global joint transforms
    """

    _device = template_pointcloud.device

    joint_transforms_est = torch.zeros_like(inverse_bind, device=_device)
    joint_transforms_est[..., :4, :4] = torch.eye(4, device=_device)

    success_mask = torch.zeros(inverse_bind.shape[0], dtype=torch.bool, device=_device)

    for bone_index in mesh_joints:
        # get the vertex weights for the prior mesh
        transform = _single_joint_transform_est(
            bone_index,
            template_pointcloud,
            reconstruction_pointcloud,
            canonical_indices,
            inverse_bind,
            mesh_vertex_joints,
            mesh_vertex_weights,
            joint_threshold,
            scale,
        )

        if transform is None:
            continue

        joint_transforms_est[bone_index] = transform
        success_mask[bone_index] = True

    return joint_transforms_est, success_mask


def _single_joint_transform_est(
    joint_index: int,
    template_pointcloud: torch.Tensor,
    reconstruction_pointcloud: torch.Tensor,
    canonical_indices: torch.Tensor,
    joint_inverse_bind: torch.Tensor,
    mesh_vertex_joints: torch.Tensor,
    mesh_vertex_weights: torch.Tensor,
    joint_threshold: float,
    scale: bool,
) -> torch.Tensor:
    # TODO: invert matching direction: match selected mesh vertices to predicted canon PC
    joint_mask = (mesh_vertex_joints == joint_index).nonzero(as_tuple=True)
    vert_weights = mesh_vertex_weights[joint_mask]

    # drop vertices that have weights below threshold
    thresholded_mask = tuple(j[vert_weights > joint_threshold] for j in joint_mask)

    if len(vert_weights) == 0:
        return None

    cloud_to_bone, mesh_to_bone = (
        thresholded_mask[0][None, :]
        .eq(canonical_indices[:, None])
        .nonzero(as_tuple=True)
    )

    bone_posed = reconstruction_pointcloud[cloud_to_bone]

    if len(bone_posed) == 0:
        return None

    bone_canon = template_pointcloud[cloud_to_bone]

    canon_local = __transform(joint_inverse_bind[joint_index], bone_canon)

    scale_factor, rotation, translation = kabsch_umeyama(
        canon_local,
        bone_posed,
        scale=scale,
        reflection=False,
        rotation=True,
    )

    transform = to_transform(scale_factor, rotation, translation)
    return transform


def make_lookup(
    target_canonical_vertices: torch.Tensor, distance_threshold: float | None
):
    """
    make a lookup function that returns the indices of the closest vertices in the target canonical vertices
    optionally apply a threshold on the distance to reject outliers

    args:
        target_canonical_vertices: (N, 3) - canonical vertices of the target mesh
        distance_threshold: float | None - threshold on the distance to reject outliers
            if None, all vertices are returned
            if not None, only vertices within the threshold are returned

    returns:
        lookup: callable
            lookup(query_vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
            returns:
                indices: (N,) - indices of the closest vertices in the target canonical vertices
                mask: (N,) - boolean mask of the valid vertices
    """

    kdtree = KDTree(target_canonical_vertices.cpu())

    def lookup(query_vertices: torch.Tensor):
        res = kdtree.query(query_vertices.cpu())
        dists, indices = (torch.tensor(t) for t in res)
        if distance_threshold is not None:
            mask = dists <= distance_threshold
        else:
            mask = torch.ones_like(dists, dtype=torch.bool)

        return indices[mask], mask

    return lookup


def reconstruction_to_canonical(
    reconstruction_predictions: torch.Tensor,
    canonical_predictions: torch.Tensor,
    joint_transforms: torch.Tensor,
    inverse_bind: torch.Tensor,
    vertex_joints: torch.Tensor,
    vertex_weights: torch.Tensor,
    success_mask: torch.Tensor,
    lookup: callable,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    obtain the reconstruction shape in the canonical space, ready for animations!

    args:
        reconstruction_predictions: (N, 3)
        canonical_predictions: (N, 3)
        joint_transforms: (J, 4, 4) joint transforms
        inverse_bind: (J, 4, 4) inverse bind matrices
        vertex_joints: (P, 4) indices of J
        vertex_weights: (P, 4) weights float
        success_mask: (J,) boolean mask of the valid joints
        lookup: callable - lookup function that returns the indices of the closest vertices in the target canonical vertices

    returns:
        reconstruction_in_canonical: (N, 3) - reconstruction in the canonical space
        error: (N,) - error between the canonical reconstruction and canonical predictions
    """
    joint_transforms = joint_transforms @ inverse_bind.to(dtype=torch.float32)

    canonical_indices, inlier_mask = get_point_correspondence(
        canonical_predictions, lookup
    )

    reconstruction_predictions = reconstruction_predictions[inlier_mask]
    canonical_predictions = canonical_predictions[inlier_mask]

    rec_joints = vertex_joints[canonical_indices]
    vertex_weights = vertex_weights.to(dtype=torch.float32)
    rec_weights = vertex_weights[canonical_indices]

    joint_success = success_mask[rec_joints]
    valid = joint_success.any(dim=1)

    rec_weights = rec_weights[valid]
    rec_joints = rec_joints[valid]
    reconstruction_predictions = reconstruction_predictions[valid]
    canonical_predictions = canonical_predictions[valid]
    joint_success = joint_success[valid]

    rec_weights[~joint_success] = 0.0

    # re-normalize weights
    sum_weights = rec_weights.sum(dim=-1, keepdim=True)
    rec_weights = rec_weights / (sum_weights + 1e-8)

    # sum over the joints
    final_transforms = (
        joint_transforms[rec_joints] * rec_weights[..., None, None]
    ).sum(dim=1)

    # get the inverse transform
    final_transforms = torch.linalg.inv(final_transforms)

    reconstruction_in_canonical = (
        reconstruction_predictions[:, None, :]
        @ __transpose(final_transforms[:, :3, :3])
    )[:, 0] + final_transforms[:, :3, 3]

    canonical_error = reconstruction_in_canonical - canonical_predictions
    canonical_error = canonical_error.pow(2).sum(dim=-1).sqrt()

    return reconstruction_in_canonical, canonical_error


def articulate_template_mesh(
    global_joint_transforms: torch.Tensor,
    vertices: torch.Tensor,
    inverse_bind: torch.Tensor,
    mesh_vertex_joints: torch.Tensor,
    mesh_vertex_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Articulate the template mesh into the posed configuration using estimated global joint transforms.

    Args:
        joint_transforms: (J, 4, 4)
            Estimated global joint transforms in world/posed space.
        ----

    Returns:
        posed_vertices: (V, 3)
            Template mesh vertices articulated into the estimated pose.
    """
    skinning_matrices = skin.compute_skinning_matrices(
        global_joint_transforms,
        inverse_bind,
        mesh_vertex_joints,
        mesh_vertex_weights,
    )
    posed_vertices = skin.transform_vertices(vertices, skinning_matrices)
    return posed_vertices


def estimate_joints_reverse(
    predicted_canon: torch.Tensor,
    predicted_pose: torch.Tensor,
    skinned_mesh: OneMeshGltf,
    distance_threshold: float,
    # vertex_lookup: callable,
    joint_threshold: float,
    scale: bool,
):
    assert predicted_canon.shape[-1] == 3, "must be shape (P 3)"
    assert predicted_pose.shape[-1] == 3, "must be shape (P 3)"
    assert len(predicted_canon.shape) == 2, "must be shape (P 3)"
    assert len(predicted_pose.shape) == 2, "must be shape (P 3)"
    assert torch.allclose(
        skinned_mesh.inverse_bind_matrices[:, 3, :3],
        torch.zeros(
            skinned_mesh.inverse_bind_matrices.shape[0],
            1,
            3,
            dtype=torch.float64,
            device=skinned_mesh.inverse_bind_matrices.device,
        ),
    ), "must be column major"

    # strip gradients from the canonical points
    predicted_canon = predicted_canon.detach()

    _device = predicted_canon.device
    template_verts = skinned_mesh.vertices.to(dtype=torch.float32, device=_device)

    # canonical_indices, inlier_mask = get_point_correspondence(
    #     predicted_canon, vertex_lookup
    # )
    dist = torch.cdist(template_verts, predicted_canon)
    _dist, canonical_indices = torch.min(dist, dim=0)
    inlier_mask = _dist < distance_threshold
    canonical_indices = canonical_indices[inlier_mask]
    predicted_pose = predicted_pose[inlier_mask]

    global_joint_transforms, success_mask = _estimate_global_joints_reverse(
        template_verts,
        predicted_pose,
        canonical_indices,
        skinned_mesh.joints,
        skinned_mesh.inverse_bind_matrices.to(dtype=torch.float32, device=_device),
        skinned_mesh.vertex_joints,
        skinned_mesh.vertex_weights,
        joint_threshold,
        scale,
    )

    return global_joint_transforms, success_mask


def _estimate_global_joints_reverse(
    template_pointcloud: torch.Tensor,
    reconstruction_pointcloud: torch.Tensor,
    canonical_indices: torch.Tensor,
    mesh_joints: torch.Tensor,
    inverse_bind: torch.Tensor,
    mesh_vertex_joints: torch.Tensor,
    mesh_vertex_weights: torch.Tensor,
    joint_threshold: float,
    scale: bool,
):
    _device = template_pointcloud.device

    joint_transforms_est = torch.zeros_like(inverse_bind, device=_device)
    joint_transforms_est[..., :4, :4] = torch.eye(4, device=_device)

    success_mask = torch.zeros(inverse_bind.shape[0], dtype=torch.bool, device=_device)

    for bone_index in mesh_joints:
        # get the vertex weights for the prior mesh
        transform = _single_joint_transform_est_reverse(
            bone_index,
            template_pointcloud,
            reconstruction_pointcloud,
            canonical_indices,
            inverse_bind,
            mesh_vertex_joints,
            mesh_vertex_weights,
            joint_threshold,
            scale,
        )

        if transform is None:
            continue

        joint_transforms_est[bone_index] = transform
        success_mask[bone_index] = True

    return joint_transforms_est, success_mask


def _single_joint_transform_est_reverse(
    joint_index: int,
    template_pointcloud: torch.Tensor,
    reconstruction_pointcloud: torch.Tensor,
    canonical_indices: torch.Tensor,
    joint_inverse_bind: torch.Tensor,
    mesh_vertex_joints: torch.Tensor,
    mesh_vertex_weights: torch.Tensor,
    joint_threshold: float,
    scale: bool,
) -> torch.Tensor:
    joint_mask = (mesh_vertex_joints == joint_index)
    weights_mask = mesh_vertex_weights > joint_threshold
    
    template_mask = torch.any(joint_mask & weights_mask, dim=1)
    template_ind = torch.arange(template_pointcloud.shape[0])[template_mask]
    if len(template_ind) == 0:
        return None
    
    rec_mask = torch.isin(canonical_indices, template_ind)
    bone_posed = reconstruction_pointcloud[rec_mask]

    if len(bone_posed) == 0:
        return None

    bone_canon = template_pointcloud[canonical_indices[rec_mask]]

    canon_local = __transform(joint_inverse_bind[joint_index], bone_canon)

    scale_factor, rotation, translation = kabsch_umeyama(
        canon_local,
        bone_posed,
        scale=scale,
        reflection=False,
        rotation=True,
    )

    transform = to_transform(scale_factor, rotation, translation)
    return transform


def get_local_to_parent_transforms(inverse_bind, joint_to_index, parents, num_joints):
    _device = inverse_bind.device
    local_to_parent_transform = torch.zeros((num_joints, 4, 4), dtype=torch.float32, device=_device)
    local_to_parent_transform[..., :4, :4] = torch.eye(4)

    for joint, joint_index in joint_to_index.items():        
        parent = parents[joint]
        if parent is None or parent not in joint_to_index:
            parent_inverse_bind = torch.eye(4, device=_device)
        else:
            parent_idx = joint_to_index[parent]
            parent_inverse_bind = inverse_bind[parent_idx]
        joint_bind = torch.linalg.inv(inverse_bind[joint_index])
        local_to_parent_transform[joint] = parent_inverse_bind @ joint_bind
    
    return local_to_parent_transform


def fit_kinematic_chain(
    predicted_canon: torch.Tensor,
    predicted_pose: torch.Tensor,
    mesh: OneMeshGltf,
    distance_threshold: float,
    joint_threshold: float,
    scale: bool = True,
    num_iters: int = 2,
    # fixed_joints: list[int] | None = None,
):
    predicted_canon = predicted_canon.detach()
    _device = predicted_canon.device

    template_pc = mesh.vertices.to(dtype=torch.float32, device=_device)
    dist = torch.cdist(template_pc, predicted_canon)
    _dist, canonical_indices = torch.min(dist, dim=0)
    inlier_mask = _dist < distance_threshold
    canonical_indices = canonical_indices[inlier_mask]
    predicted_pose = predicted_pose[inlier_mask]

    joints = mesh.joints.tolist()
    joint_to_index = {joint: idx for idx, joint in enumerate(joints)}
    num_joints = mesh.local_joint_transforms.shape[0]
    parents, children, descendants, roots = trees.recover_parents(mesh.nodes_parents_list)
    topo_order = trees.topsort(children, roots)
    # print("Tree sanity check:", tree_sanity_check(parents, children, roots))

    inverse_bind = (
        __transpose(mesh.inverse_bind_matrices).to(dtype=torch.float32, device=_device).contiguous()
    )
    orig_local_transforms = get_local_to_parent_transforms(
        inverse_bind, joint_to_index, parents, num_joints
    )
    previous_local_transforms = orig_local_transforms.clone()

    mesh_vertex_joints = mesh.vertex_joints.to(_device)
    mesh_vertex_weights = mesh.vertex_weights.to(dtype=torch.float32, device=_device)
    
    # history = []
    for _ in range(num_iters):
        global_transform, residual_local_transforms = _fit_kinematic_chain(
            template_verts=template_pc,
            reconstruction_pc=predicted_pose,
            canonical_indices=canonical_indices,
            joints=joints,
            topo_order=topo_order,
            joint_to_index=joint_to_index,
            parents=parents,
            node_parents_dict=mesh.nodes_parents_list,
            descendants=descendants,
            orig_local_transforms=orig_local_transforms,
            previous_local_transforms=previous_local_transforms,
            inverse_bind=inverse_bind,
            mesh_vertex_joints=mesh_vertex_joints,
            mesh_vertex_weights=mesh_vertex_weights,
            joint_threshold=joint_threshold,
            scale=scale,
        )
        # TO CONSIDER:
        # decouple orig_local_transform (ie from local joint to parent joint)
        # from residual rotation (articulation), apply former on the left of current
        # estimated residual rotation and latter on the right
        # but it reduces to the same thing, right?
        # definitely lower priority
        # PROBABLY EVEN BETTER: forget about the previous rotation for the current joint
        # and optimize it from scratch (only the rotations below (descendants' rots) matter
        # for the fitting the current joint in the iterated process - but i believe it seriously
        # does not matter here)
        previous_local_transforms = previous_local_transforms @ residual_local_transforms

    global_joint_transforms = skin.compute_global_nodes_transforms(
        previous_local_transforms, joints, mesh.nodes_parents_list, transpose=True
    )
    fitted_shape = articulate_template_mesh(
        __transpose(global_joint_transforms),
        template_pc,
        __transpose(inverse_bind),
        mesh_vertex_joints,
        mesh_vertex_weights,
    )
    # history.append((global_transform, previous_local_transforms, fitted_shape, joint_pos))

    # TODO: shouldn't we recompute the global transform here

    joint_pos = global_joint_transforms[..., :3, 3] # <- the matrices are transposed here
    joint_rots = torch.linalg.inv(orig_local_transforms) @ previous_local_transforms
    return global_transform, previous_local_transforms, fitted_shape, joint_pos, joint_rots, inlier_mask, canonical_indices


def _fit_kinematic_chain(
    template_verts: torch.Tensor,
    reconstruction_pc: torch.Tensor,
    canonical_indices: torch.Tensor,
    joints: list[int],
    topo_order: list[int],
    joint_to_index: dict[int, int],
    parents: dict[int, int],
    node_parents_dict: dict[int, list[int]],
    descendants: dict[int, list[tuple[int, int]]],
    orig_local_transforms: torch.Tensor,
    previous_local_transforms: torch.Tensor,
    inverse_bind: torch.Tensor,
    mesh_vertex_joints: torch.Tensor,
    mesh_vertex_weights: torch.Tensor,
    joint_threshold: float,
    scale: bool = True,
):
    _device = reconstruction_pc.device
    
    global_joint_transforms = skin.compute_global_nodes_transforms(
        previous_local_transforms, joints, node_parents_dict, transpose=True
    )

    template_pc = articulate_template_mesh(
        __transpose(global_joint_transforms),
        template_verts,
        __transpose(inverse_bind),
        mesh_vertex_joints,
        mesh_vertex_weights,
    )

    scale_factor, rotation, translation = kabsch_umeyama(
        reconstruction_pc,
        template_pc[canonical_indices],
        scale=scale,
        reflection=False,
        rotation=True,
    )
    global_transform = to_transform(scale_factor, rotation, translation)
    reconstruction_pc = __transform(global_transform, reconstruction_pc)

    residual_local_transforms = torch.zeros_like(previous_local_transforms, device=_device)
    residual_local_transforms[..., :4, :4] = torch.eye(4, device=_device)
    # global_joint_transforms = torch.zeros_like(inverse_bind)
    # global_joint_transforms[..., :4, :4] = torch.eye(4)

    for joint in topo_order:
        if joint not in joint_to_index:
            continue
        joint_index = joint_to_index[joint]
    
        parent = parents[joint]
        if parent is None or parent not in joint_to_index:
            parent_idx = None
        else:
            parent_idx = joint_to_index[parent]

        temp_local_transforms = orig_local_transforms.clone()
        for d, _ in descendants[joint]:
            temp_local_transforms[d] = previous_local_transforms[d]
        temp_global_transforms = skin.compute_global_nodes_transforms(
            temp_local_transforms, joints, node_parents_dict, transpose=True
        )
        temp_template_pc = articulate_template_mesh(
            __transpose(temp_global_transforms),
            template_verts,
            __transpose(inverse_bind),
            mesh_vertex_joints,
            mesh_vertex_weights,
        )

        residual_local_transform, bone_canon, bone_posed = _fit_kinematic_chain_single(
            joint_with_index=(joint, joint_index),
            parent_with_index=(parent, parent_idx),
            reconstruction_pc=reconstruction_pc,
            template_pc=temp_template_pc,
            joint_to_index=joint_to_index,
            canonical_indices=canonical_indices,
            inverse_bind=inverse_bind,
            previous_local_transforms=previous_local_transforms,
            global_joint_transforms=global_joint_transforms,
            mesh_vertex_joints=mesh_vertex_joints,
            mesh_vertex_weights=mesh_vertex_weights,
            descendants=descendants,
            joint_threshold=joint_threshold,
        )
        if residual_local_transform is not None:
            residual_local_transforms[joint] = residual_local_transform
        
        # TEMPORARY:
        full_local_transforms = previous_local_transforms @ residual_local_transforms
        global_joint_transforms = skin.compute_global_nodes_transforms(
            full_local_transforms, joints, node_parents_dict, transpose=True
        )
        # assert torch.allclose(global_joint_transforms, _global_joint_transforms, atol=1e-7)

        # TODO: shouldn't we recompute the global transform here, like below:

        # PERMANENT:
        # global_joint_transforms[joint_index] = local_to_parent_transforms[joint] @ local_joint_transforms[joint]
        # if parent_idx is not None:
        #     global_joint_transforms[joint_index] = global_joint_transforms[parent_idx] @ global_joint_transforms[joint_index]
        
        if residual_local_transform is not None and False:
            bone_canon_transformed = __transform(residual_local_transform, bone_canon)
            current_canon_pc = articulate_template_mesh(
                mesh, __transpose(global_joint_transforms).to(torch.float64)
            ).to(torch.float32)

            ps.remove_all_structures()
            ps_canon_pc = ps.register_point_cloud("CANON OG", temp_template_pc.numpy(), enabled=False)
            # ps_canon_pc = ps.register_point_cloud("CANON", template_pc.numpy(), enabled=False)
            ps_current_canon_pc = ps.register_point_cloud("CURRENT CANON", current_canon_pc.numpy())
            ps_rec_pc = ps.register_point_cloud("REC", reconstruction_pc.numpy(), enabled=False)
            
            ps_bone_canon_transf_pc = ps.register_point_cloud("BONE CANON TRANSF", bone_canon_transformed.numpy())
            ps_bone_canon_pc = ps.register_point_cloud("BONE CANON", bone_canon.numpy())
            ps_bone_posed_pc = ps.register_point_cloud("BONE POSED", bone_posed.numpy())
            ps.show()
    
    return global_transform, residual_local_transforms


def _fit_kinematic_chain_single(
    joint_with_index,
    parent_with_index,
    reconstruction_pc,
    template_pc,
    joint_to_index,
    canonical_indices,
    inverse_bind,
    previous_local_transforms,
    global_joint_transforms,
    mesh_vertex_joints,
    mesh_vertex_weights,
    descendants,
    joint_threshold,
    descendant_level=1,
):
    _device = reconstruction_pc.device

    joint, joint_index = joint_with_index
    _, parent_index = parent_with_index

    parent_global_inv = (
        torch.linalg.inv(global_joint_transforms[parent_index]) 
        if parent_index is not None else torch.eye(4)
    )
    local_to_parent_inv = torch.linalg.inv(previous_local_transforms[joint])

    _descendants = [d for d, dist in descendants[joint] if dist <= descendant_level]
    descendant_indices = torch.tensor(
        [joint_to_index[d] for d in _descendants] + [joint_index], dtype=torch.int32, device=_device
    )
    joint_mask = torch.isin(mesh_vertex_joints, descendant_indices)
    weights_mask = mesh_vertex_weights > joint_threshold
    
    template_mask = torch.any(joint_mask & weights_mask, dim=1)
    # template_ind = torch.arange(template_pc.shape[0])[template_mask]
    template_ind = torch.nonzero(template_mask, as_tuple=True)[0]
    if len(template_ind) == 0:
        return None, None, None
    
    rec_mask = torch.isin(canonical_indices, template_ind)
    bone_posed = reconstruction_pc[rec_mask]
    bone_posed = __transform(parent_global_inv, bone_posed)
    bone_posed = __transform(local_to_parent_inv, bone_posed)

    if len(bone_posed) == 0:
        return None, None, None

    bone_canon = template_pc[canonical_indices[rec_mask]]
    bone_canon = __transform(inverse_bind[joint_index], bone_canon)

    scale_factor, rotation, translation = kabsch_umeyama(
        bone_canon,
        bone_posed,
        scale=False,
        reflection=False,
        rotation=True,
        translation=False,
    )
    return to_transform(scale_factor, rotation, translation), bone_canon, bone_posed


def force_strict_local_rotations(
    mesh: OneMeshGltf,
    pose: torch.Tensor,
):
    parents, _, _, _ = trees.recover_parents(mesh.nodes_parents_list)
    joints = mesh.joints.tolist()
    joint_to_index = {joint: idx for idx, joint in enumerate(joints)}
    inverse_bind = __transpose(mesh.inverse_bind_matrices).to(torch.float32)
    num_joints = mesh.local_joint_transforms.shape[0]
    local_to_parent_transforms = get_local_to_parent_transforms(
        inverse_bind, joint_to_index, parents, num_joints
    )

    joint_rots = torch.linalg.inv(local_to_parent_transforms) @ pose
    joint_rots[:, :3, 3] = 0
    local_joint_transforms = local_to_parent_transforms @ joint_rots
    return local_joint_transforms


def global_to_local(global_joint_transforms: torch.Tensor, parents, joints, num_nodes):
    local_joint_transforms = torch.zeros((num_nodes, 4, 4), dtype=torch.float32)
    local_joint_transforms[..., :4, :4] = torch.eye(4)

    for jidx, joint in enumerate(joints):
        parent = parents[joint]
        if parent is None or parent not in joints:
            local_joint_transforms[joint] = global_joint_transforms[jidx]
        else:
            parent_idx = joints.index(parent)
            local_joint_transforms[joint] = torch.linalg.inv(global_joint_transforms[parent_idx]) @ global_joint_transforms[jidx]

    return local_joint_transforms


def fit_kinematic_chain_batched(
    predicted_canon: torch.Tensor,
    predicted_pose: torch.Tensor,
    seq_mask: torch.Tensor,
    mesh_proc_data: ShapeProcessingData,
    distance_threshold: float,
    joint_threshold: float,
    scale: bool = True,
    num_iters: int = 2,
):
    """
    Batched version of fit_kinematic_chain.
    
    Args:
        predicted_canon: (B, N, 3)
        predicted_pose: (B, N, 3)
        mesh: OneMeshGltf (shared across batch)
        seq_mask: (B, N) boolean mask of valid points
    """
    predicted_canon = predicted_canon.detach()
    B, N, _ = predicted_canon.shape
    
    template_pc = mesh_proc_data.vertices
    
    # Correspondence finding (Batched)
    # dist: (B, N, V)
    dist = torch.cdist(predicted_canon, template_pc.unsqueeze(0).expand(B, -1, -1))
    _dist, canonical_indices = torch.min(dist, dim=2) # (B, N)
    inlier_mask = (_dist < distance_threshold) & seq_mask.bool()
    
    # We keep predicted_pose as (B, N, 3) but use inlier_mask in calculations

    previous_local_transforms = mesh_proc_data.local_to_parent_transforms.unsqueeze(0).expand(B, -1, -1, -1).clone()
    for _ in range(num_iters):
        global_transform, residual_local_transforms = _fit_kinematic_chain_batched(
            template_verts=template_pc,
            reconstruction_pc=predicted_pose,
            canonical_indices=canonical_indices,
            inlier_mask=inlier_mask,
            mesh_proc_data=mesh_proc_data,
            previous_local_transforms=previous_local_transforms,
            joint_threshold=joint_threshold,
            scale=scale,
        )
        previous_local_transforms = previous_local_transforms @ residual_local_transforms

    # Compute final global transforms
    global_joint_transforms = skin.compute_global_nodes_transforms_tree_batched(
        previous_local_transforms, mesh_proc_data.topo_order, mesh_proc_data.parents, mesh_proc_data.joint_to_index, transpose=True
    )
    joint_pos = global_joint_transforms[..., :3, 3] 
    joint_rots = torch.linalg.inv(mesh_proc_data.local_to_parent_transforms) @ previous_local_transforms
    
    fitted_shape = articulate_template_mesh(
        __transpose(global_joint_transforms),
        template_pc,
        __transpose(mesh_proc_data.inverse_bind_matrices),
        mesh_proc_data.vertex_joints,
        mesh_proc_data.vertex_weights,
    )

    batch_idx = torch.arange(B, device=predicted_canon.device).view(B, 1).expand(-1, canonical_indices.shape[1])
    matched_template = fitted_shape[batch_idx, canonical_indices]
    scale_factor, rotation, translation = kabsch_umeyama_batched(
        predicted_pose,
        matched_template,
        sequence_mask=inlier_mask,
        scale=scale,
        reflection=False,
        rotation=True,
    )
    global_transform = to_transform(scale_factor.unsqueeze(-1), rotation, translation)
    
    return global_transform, previous_local_transforms, fitted_shape, joint_pos, joint_rots, inlier_mask, canonical_indices


def _fit_kinematic_chain_batched(
    template_verts: torch.Tensor,
    reconstruction_pc: torch.Tensor,
    canonical_indices: torch.Tensor,
    inlier_mask: torch.Tensor,
    mesh_proc_data: ShapeProcessingData,
    previous_local_transforms: torch.Tensor,
    joint_threshold: float,
    scale: bool = True,
):
    _device = reconstruction_pc.device
    B = reconstruction_pc.shape[0]
    
    # 1. Global Alignment
    # Compute current global transforms
    global_joint_transforms = skin.compute_global_nodes_transforms_tree_batched(
        previous_local_transforms, mesh_proc_data.topo_order, mesh_proc_data.parents, mesh_proc_data.joint_to_index, transpose=True
    )

    # Articulate template
    template_pc_articulated = articulate_template_mesh(
        __transpose(global_joint_transforms),
        template_verts,
        __transpose(mesh_proc_data.inverse_bind_matrices),
        mesh_proc_data.vertex_joints,
        mesh_proc_data.vertex_weights,
    )

    # Gather matched points: (B, N, 3)
    batch_idx = torch.arange(B, device=_device).view(B, 1).expand(-1, canonical_indices.shape[1])
    matched_template = template_pc_articulated[batch_idx, canonical_indices]

    scale_factor, rotation, translation = kabsch_umeyama_batched(
        reconstruction_pc,
        matched_template,
        sequence_mask=inlier_mask,
        scale=scale,
        reflection=False,
        rotation=True,
    )
    global_transform = to_transform(scale_factor.unsqueeze(-1), rotation, translation)

    # Apply global transform to reconstruction_pc
    reconstruction_pc_aligned = __transform(global_transform, reconstruction_pc)

    residual_local_transforms = torch.zeros_like(previous_local_transforms, device=_device)
    residual_local_transforms[..., :4, :4] = torch.eye(4, device=_device)

    # 2. Joint Loop
    for idx, joint in enumerate(mesh_proc_data.topo_order):
        if joint not in mesh_proc_data.joint_to_index:
            continue
        joint_index = mesh_proc_data.joint_to_index[joint]
    
        parent = mesh_proc_data.parents[joint]
        if parent is None or parent not in mesh_proc_data.joint_to_index:
            parent_idx = None
        else:
            parent_idx = mesh_proc_data.joint_to_index[parent]

        # Construct temp local transforms
        temp_local_transforms = mesh_proc_data.local_to_parent_transforms.unsqueeze(0).expand(B, -1, -1, -1).clone()
        # Update descendants with previous estimates
        for d, _ in mesh_proc_data.descendants[joint]:
            temp_local_transforms[:, d] = previous_local_transforms[:, d].clone()
            
        # Compute temp global transforms
        temp_global_transforms = skin.compute_global_nodes_transforms_tree_batched(
            temp_local_transforms, mesh_proc_data.topo_order, mesh_proc_data.parents, mesh_proc_data.joint_to_index, transpose=True
        )
        
        # Articulate temp template
        temp_template_pc = articulate_template_mesh(
            __transpose(temp_global_transforms),
            template_verts,
            __transpose(mesh_proc_data.inverse_bind_matrices),
            mesh_proc_data.vertex_joints,
            mesh_proc_data.vertex_weights,
        )

        residual_local_transform = _fit_kinematic_chain_single_batched(
            joint_with_index=(joint, joint_index),
            parent_with_index=(parent, parent_idx),
            reconstruction_pc=reconstruction_pc_aligned,
            template_pc=temp_template_pc,
            canonical_indices=canonical_indices,
            inlier_mask=inlier_mask,
            previous_local_transforms=previous_local_transforms,
            global_joint_transforms=global_joint_transforms, # Note: using the 'current' global transforms for parent inv
            mesh_proc_data=mesh_proc_data,
            joint_threshold=joint_threshold,
            scale=False, # Usually False for internal joints
        )
        
        if residual_local_transform is not None:
            residual_local_transforms[:, joint] = residual_local_transform

        full_local_transforms = previous_local_transforms @ residual_local_transforms
        global_joint_transforms = skin.compute_global_nodes_transforms_tree_batched(
            full_local_transforms, mesh_proc_data.topo_order, mesh_proc_data.parents, mesh_proc_data.joint_to_index, transpose=True
        )
        
        # OPTIONAL:
        # Recompute global alignment
        # current_template_pc = articulate_template_mesh(
        #     __transpose(global_joint_transforms),
        #     template_verts,
        #     __transpose(mesh_proc_data.inverse_bind_matrices),
        #     mesh_proc_data.vertex_joints,
        #     mesh_proc_data.vertex_weights,
        # )
        # matched_template = current_template_pc[batch_idx, canonical_indices]
        # scale_factor, rotation, translation = kabsch_umeyama_batched(
        #     reconstruction_pc,
        #     matched_template,
        #     sequence_mask=inlier_mask,
        #     scale=scale,
        #     reflection=False,
        #     rotation=True,
        # )
        # global_transform = to_transform(scale_factor.unsqueeze(-1), rotation, translation)
        # reconstruction_pc_aligned = __transform(global_transform, reconstruction_pc)

        # GEMINI SOLUTION:
        # # Update global transform for the current joint to reflect changes in parent and self
        # current_local = previous_local_transforms[:, joint] @ residual_local_transforms[:, joint]
        # if parent_idx is not None:
        #     global_joint_transforms[:, joint_index] = global_joint_transforms[:, parent_idx] @ current_local
        # else:
        #     global_joint_transforms[:, joint_index] = current_local

    return global_transform, residual_local_transforms


def _fit_kinematic_chain_single_batched(
    joint_with_index: tuple[int, int],
    parent_with_index: tuple[int, int],
    reconstruction_pc,
    template_pc,
    canonical_indices,
    inlier_mask,
    previous_local_transforms,
    global_joint_transforms,
    mesh_proc_data: ShapeProcessingData,
    joint_threshold: float,
    descendant_level: int = 1,
    scale: float = False,
):
    _device = reconstruction_pc.device
    B = reconstruction_pc.shape[0]

    joint, joint_index = joint_with_index
    _, parent_index = parent_with_index

    # Parent Global Inverse: (B, 4, 4)
    if parent_index is not None:
        parent_global_inv = torch.linalg.inv(global_joint_transforms[:, parent_index])
    else:
        parent_global_inv = torch.eye(4, device=_device).unsqueeze(0).expand(B, -1, -1)

    # Local to Parent Inverse: (B, 4, 4)
    local_to_parent_inv = torch.linalg.inv(previous_local_transforms[:, joint])

    # Identify vertices belonging to this joint/descendants
    _descendants = [d for d, dist in mesh_proc_data.descendants[joint] if dist <= descendant_level]
    descendant_indices = torch.tensor(
        [mesh_proc_data.joint_to_index[d] for d in _descendants] + [joint_index], dtype=torch.int32, device=_device
    )

    # Masks (V,)
    joint_mask = torch.isin(mesh_proc_data.vertex_joints, descendant_indices)
    weights_mask = mesh_proc_data.vertex_weights > joint_threshold
    template_mask = torch.any(joint_mask & weights_mask, dim=1)
    template_ind = torch.nonzero(template_mask, as_tuple=True)[0] # Indices into V

    if len(template_ind) == 0:
        return None
    
    # Identify points in reconstruction_pc that correspond to these vertices
    # rec_mask: (B, N)
    rec_mask = torch.isin(canonical_indices, template_ind)
    
    # Combined mask for Kabsch
    # (B, N)
    mask = rec_mask & inlier_mask
    
    # Transform reconstruction_pc to local frame
    # (B, N, 3)
    bone_posed = __transform(parent_global_inv, reconstruction_pc)
    bone_posed = __transform(local_to_parent_inv, bone_posed)
    
    # Transform template_pc (which is temp_template_pc) to local frame
    # We need to gather the corresponding points first
    # (B, N, 3)
    batch_idx = torch.arange(B, device=_device).view(B, 1).expand(-1, canonical_indices.shape[1])
    matched_template = template_pc[batch_idx, canonical_indices]
    
    # Apply inverse bind to bring to local rest frame
    # inverse_bind[joint_index]: (4, 4) -> expand to (B, 4, 4)
    inv_bind = mesh_proc_data.inverse_bind_matrices[joint_index].unsqueeze(0).expand(B, -1, -1)
    bone_canon = __transform(inv_bind, matched_template)

    scale_factor, rotation, translation = kabsch_umeyama_batched(
        bone_canon,
        bone_posed,
        sequence_mask=mask,
        scale=scale,
        reflection=False,
        rotation=True,
        translation=False,
    )

    return to_transform(scale_factor.unsqueeze(-1), rotation, translation)


def initialize_bone_scales(
    reconstruction_pc,
    canonical_indices,
    inlier_mask,
    mesh_proc_data: ShapeProcessingData,
    joint_threshold: float = 0.5,
):
    _device = reconstruction_pc.device
    B = reconstruction_pc.shape[0]
    num_joints = mesh_proc_data.local_to_parent_transforms.shape[0]
    bones_scales = torch.ones((B, num_joints), dtype=torch.float32, device=_device)
    for joint in mesh_proc_data.topo_order:
        if joint not in mesh_proc_data.joint_to_index:
            continue
        # joint_index = mesh_proc_data.joint_to_index[joint]
    
        bones_scales[:, joint] = _initialize_bone_scale(
            node_idx=joint,
            reconstruction_pc=reconstruction_pc,
            canonical_indices=canonical_indices,
            inlier_mask=inlier_mask,
            mesh_proc_data=mesh_proc_data,
            joint_threshold=joint_threshold,
        )
    
    return bones_scales


def _initialize_bone_scale(
    node_idx: int,
    reconstruction_pc,
    canonical_indices,
    inlier_mask,
    mesh_proc_data: ShapeProcessingData,
    joint_threshold: float = 0.5,
):
    _device = reconstruction_pc.device
    B = reconstruction_pc.shape[0]
    joint_index = mesh_proc_data.joint_to_index[node_idx]

    # Masks (V,)
    joint_mask = mesh_proc_data.vertex_joints == joint_index
    weights_mask = mesh_proc_data.vertex_weights > joint_threshold
    template_mask = torch.any(joint_mask & weights_mask, dim=1)
    template_ind = torch.nonzero(template_mask, as_tuple=True)[0] # Indices into V

    if len(template_ind) < 10:
        # print(f"not enough points for joint idx {joint_index} (node idx: {mesh_proc_data.joints[joint_index]})")
        return torch.ones((B,), dtype=torch.float32, device=_device)
    
    # Identify points in reconstruction_pc that correspond to these vertices
    # rec_mask: (B, N)
    rec_mask = torch.isin(canonical_indices, template_ind)
    
    # Combined mask for Kabsch
    # (B, N)
    mask = rec_mask & inlier_mask

    batch_mask = torch.sum(mask, dim=-1) > 25
    # if not batch_mask.all():
    #     print(f"problem with joint: {joint_index}, node: {mesh_proc_data.joints[joint_index]}")
    #     print(torch.sum(mask, dim=-1))

    inv_bind = mesh_proc_data.inverse_bind_matrices[joint_index].unsqueeze(0).expand(B, -1, -1)
    template_pc = mesh_proc_data.vertices.unsqueeze(0).expand(B, -1, -1)
    
    # Transform reconstruction_pc to local frame
    # (B, N, 3)
    bone_posed = __transform(inv_bind, reconstruction_pc)
    
    # Transform template_pc (which is temp_template_pc) to local frame
    # We need to gather the corresponding points first
    # (B, N, 3)
    batch_idx = torch.arange(B, device=_device).view(B, 1).expand(-1, canonical_indices.shape[1])
    matched_template = template_pc[batch_idx, canonical_indices]
    
    # Apply inverse bind to bring to local rest frame
    # inverse_bind[joint_index]: (4, 4) -> expand to (B, 4, 4)
    bone_canon = __transform(inv_bind, matched_template)

    scale_factor, rotation, translation = kabsch_umeyama_batched(
        bone_canon,
        bone_posed,
        sequence_mask=mask,
        scale=True,
        reflection=False,
        rotation=True,
        translation=True,
    )
    scale_factor = scale_factor.squeeze(-1)
    scale_factor[~batch_mask] = 1.0
    if torch.isinf(scale_factor).any():
        print("WARN:", joint_index, mesh_proc_data.joints[joint_index], "is INF")
    return scale_factor


def scale_local_joint_transforms(
    local_joint_transforms: torch.Tensor,
    bone_scales: torch.Tensor,
    shape_data: ShapeProcessingData,
) -> torch.Tensor:
    device = local_joint_transforms.device
    B, num_nodes = local_joint_transforms.shape[:2]
    
    # scales_adjusted = bone_scales.clone()
    scaled_transforms = []

    # def follow_tree(node, scale_accum):
    #     scale_accum *= scales_adjusted[node]
    #     # print(node, scale_accum, scales_adjusted[node])
    #     parent = shape_data.parents[node]
    #     if parent is None:
    #         return scale_accum
    #     else:
    #         return follow_tree(parent, scale_accum)
        
    # for node_idx in range(num_nodes):
    #     parent = shape_data.parents[node_idx]
    #     parent_scale = bone_scales[parent] if parent is not None else 1.0
    #     scales_adjusted[node_idx] = bone_scales[node_idx] / parent_scale

    for node_idx in range(num_nodes):
        parent = shape_data.parents[node_idx]
        # scale_adjusted = scales_adjusted[node_idx]
        parent_scale = bone_scales[:, parent] if parent is not None else 1.0
        scale_adjusted = bone_scales[:, node_idx] / parent_scale
        # scales_adjusted[node_idx] = scale_adjusted
        # scale_accum = follow_tree(node_idx, 1.0)
        # print(node_idx, node_names[node_idx], bone_scales[node_idx].item(), scale_adjusted.item(), scale_accum)
        # S = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
        # S[:, :3, :3] *= scale_adjusted
        # S[[0, 1, 2], [0, 1, 2]] = scale_factor
        # local_joint_transforms[:, node_idx] = local_joint_transforms[:, node_idx] @ S
    
        scale_xyz = scale_adjusted.view(B, 1).expand(B, 3)
        scale_w = torch.ones((B, 1), dtype=torch.float32, device=device)
        S_diag = torch.cat([scale_xyz, scale_w], dim=-1)
        S = torch.diag_embed(S_diag)
        
        scaled_transforms.append(local_joint_transforms[:, node_idx] @ S)
        
    return torch.stack(scaled_transforms, dim=1)


def mirror_local_joint_transforms(
    local_joint_transforms: torch.Tensor, 
    shape_data: ShapeProcessingData
) -> torch.Tensor:
    """
    Creates a symmetrically mirrored pose by swapping symmetric joints 
    and reflecting their transformation matrices across the YZ plane.
    
    Args:
        local_joint_transforms: (B, num_joints, 4, 4) tensor of local transforms.
        shape_data: ShapeProcessingData instance containing symmetry mappings.
        
    Returns:
        local_joint_transforms_sym: (B, num_joints, 4, 4) mirrored local transforms.
    """
    device = local_joint_transforms.device
    B = local_joint_transforms.shape[0]

    local_extra_rotation_mat = shape_data.local_to_parent_inv @ local_joint_transforms
    
    # 1. Structural Swapping
    # shape_data.joint_sym_corresp maps each joint index to its symmetric counterpart.
    # Center-line joints (e.g., spine) naturally map to themselves.
    swapped_extra_transforms = local_extra_rotation_mat[:, shape_data.joint_sym_corresp]
    
    delta_L = swapped_extra_transforms[..., :3, :3]  # Pure 3x3 rotation
    
    # 3. Compute pure 3x3 Global Rest Rotations by traversing the hierarchy
    G_rest = torch.zeros((shape_data.num_joints, 3, 3), dtype=torch.float32, device=device)
    for joint in range(shape_data.num_joints):
        if joint in shape_data.joint_to_index:
            G_rest[joint] = shape_data.bind_matrices[shape_data.joint_to_index[joint]][:3, :3]
        else:
            G_rest[joint] = torch.eye(3, device=device, dtype=torch.float32)

    # Expand to batch size
    G_L = G_rest[shape_data.joint_sym_corresp].unsqueeze(0).expand(B, -1, -1, -1)
    G_R = G_rest.unsqueeze(0).expand(B, -1, -1, -1)
    
    # 4. World-Space Reflection Matrix (Flip X-axis)
    M_3x3 = torch.eye(3, device=device)
    M_3x3[0, 0] = -1.0
    M_3x3 = M_3x3.view(1, 1, 3, 3)
    
    # 5. Compute the Local Axis Mapping (F)
    # This maps the mirrored Left axes to the Right axes perfectly
    F = torch.transpose(G_R, -2, -1) @ M_3x3 @ G_L
    
    # 6. Snap F to the nearest perfect axis permutation
    # Rigs are often slightly asymmetric (e.g., right leg is modeled 2 degrees off).
    # Snapping forces the mapping to be a perfect 90/180 degree axis flip
    F_abs = torch.abs(F)
    max_idx = torch.argmax(F_abs, dim=-1, keepdim=True)
    F_snapped = torch.zeros_like(F)
    F_snapped.scatter_(-1, max_idx, torch.sign(torch.gather(F, -1, max_idx)))
    
    # 7. Apply the perfect similarity transform
    # delta_R = F @ delta_L @ F^T (This natively negates the angle for correct outward bending!)
    delta_R = F_snapped @ delta_L @ torch.transpose(F_snapped, -2, -1)
    
    # 8. Reconstruct the 4x4 matrix and apply to the local rest pose
    local_joint_rotation_mat_sym = torch.zeros_like(swapped_extra_transforms)
    local_joint_rotation_mat_sym[..., :3, :3] = delta_R
    local_joint_rotation_mat_sym[..., 3, 3] = 1.0

    local_joint_transforms_sym = shape_data.local_to_parent_transforms @ local_joint_rotation_mat_sym

    global_joint_transforms_sym = skin.compute_global_nodes_transforms_tree_batched(
        local_joint_transforms_sym, shape_data.topo_order, shape_data.parents, shape_data.joint_to_index, transpose=True
    )

    joint_pos = global_joint_transforms_sym[..., :3, 3]

    fitted_shape = articulate_template_mesh(
        __transpose(global_joint_transforms_sym),
        shape_data.vertices,
        __transpose(shape_data.inverse_bind_matrices),
        shape_data.vertex_joints,
        shape_data.vertex_weights,
    )
    return local_joint_transforms_sym, fitted_shape, joint_pos
