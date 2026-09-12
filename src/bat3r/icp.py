import torch

from bat3r import bones
from bat3r.eval_utils import chamfer_dist


def _kabsch_umeyama_weighted_for_icp(
    a: torch.Tensor, b: torch.Tensor, rotation: bool, scale: bool, reflection: bool, 
    translation: bool = True, weights: torch.Tensor | None = None,
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

    if weights is None:
        weights = torch.ones(a.shape[:-1], dtype=a.dtype, device=device)

    weights = weights.unsqueeze(-1)
    w_sum = weights.sum(axis=-2)

    if translation:
        a_mean = (a * weights).sum(axis=-2) / w_sum
        b_mean = (b * weights).sum(axis=-2) / w_sum
    else:
        a_mean = torch.zeros_like((a * weights).sum(axis=-2) / w_sum)
        b_mean = torch.zeros_like((b * weights).sum(axis=-2) / w_sum)
    
    A = a - a_mean
    B = b - b_mean

    # normalized covariance <(3, n) @ (n, 3)>
    covariance = (bones.__transpose(B) @ (A * weights)) / w_sum

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

    rotation_mat = U @ Signs @ Vh if rotation else eye()

    var_A = ((A**2).sum(axis=-1) * weights.squeeze(-1)).sum(axis=-1) / w_sum.squeeze(-1)
    if scale:
        if rotation:
            scale_factor = torch.trace(torch.diag(D) @ Signs) / var_A
        else:
            scale_factor = covariance.diagonal(dim1=-2, dim2=-1).sum(-1) / var_A
    else:
        scale_factor = 1.0

    translation = b_mean - scale_factor * rotation_mat @ a_mean
    return scale_factor, rotation_mat, translation


def icp(X, Y, num_iters=50, tol=1e-6, optimize_scale=True, optimize_rotation=True):
    """
    Bidirectional ICP using PyTorch tooling.
    Assumes X and Y are already in a good initial alignment.
    """
    X_current = X.clone()

    best_cost = chamfer_dist(X, Y)[2]
    X_best = X.clone()  # If ICP fails instantly, this guarantees we return Identity
    last_cost = best_cost

    # To balance the bidirectional matches if point clouds are different sizes
    len_X, len_Y = len(X), len(Y)
    use_weights = len_X != len_Y
    if use_weights:
        weights = torch.ones(len_X + len_Y, dtype=X.dtype, device=X.device)
        weights[:len_X] = 1.0 / len_X
        weights[len_X:] = 1.0 / len_Y
    else:
        weights = None

    for _ in range(num_iters):
        dist = torch.cdist(X_current, Y)

        _, idx_Y = torch.min(dist, dim=1)
        matched_Y_for_X = Y[idx_Y]

        _, idx_X = torch.min(dist, dim=0)
        matched_X_for_Y = X_current[idx_X]

        X_bidir = torch.cat([X_current, matched_X_for_Y], dim=0)
        Y_bidir = torch.cat([matched_Y_for_X, Y], dim=0)

        kwargs = dict(scale=optimize_scale, rotation=optimize_rotation, reflection=False, translation=True)
        if use_weights:
            kwargs['weights'] = weights 

        scale_factor, R, t = _kabsch_umeyama_weighted_for_icp(X_bidir, Y_bidir, **kwargs)
        
        # 7. Apply step transform 
        T_step = bones.to_transform(scale_factor if optimize_scale else 1.0, R, t)
        X_current = bones.__transform(T_step, X_current)

        current_cost = chamfer_dist(X_current, Y)[2]
        
        if current_cost < best_cost:
            best_cost = current_cost
            X_best = X_current.clone()
            
        if tol is not None and (last_cost - current_cost) <= tol:
            break
            
        last_cost = current_cost

    final_scale_factor, R_final, t_final = _kabsch_umeyama_weighted_for_icp(
        X, X_best, 
        scale=optimize_scale, rotation=optimize_rotation, reflection=False, translation=True
    )
    
    T_final = bones.to_transform(final_scale_factor if optimize_scale else 1.0, R_final, t_final) 
    return X_best, T_final


def icp_start(X, Y, percentile=5.0, optimize_rotation=True):
    X_centered = X - X.mean(dim=0)
    Y_centered = Y - Y.mean(dim=0)
    
    def get_principle_scale(centered_pc):
        cov = (centered_pc.T @ centered_pc) / (centered_pc.shape[0] - 1)
        _, evecs = torch.linalg.eigh(cov)
        principal_dir = evecs[:, -1]
        proj = centered_pc @ principal_dir
        
        if percentile == 0:
            return proj.max() - proj.min()
            
        q = torch.tensor([percentile / 100.0, 1.0 - (percentile / 100.0)], device=proj.device, dtype=proj.dtype)
        quantiles = torch.quantile(proj, q)
        return quantiles[1] - quantiles[0]

    scale_X = get_principle_scale(X_centered)
    scale_Y = get_principle_scale(Y_centered)
    scale_factor = (scale_Y / scale_X).item()
    scale_factor = 1.0
    
    X_mean = X.mean(dim=0)
    Y_mean = Y.mean(dim=0)
    
    T_init = torch.eye(4, device=X.device, dtype=X.dtype)
    T_init[:3, :3] *= scale_factor
    T_init[:3, 3] = Y_mean - scale_factor * X_mean
    
    X_init = bones.__transform(T_init, X)

    if X_init.shape[0] > 10000:
        indices = torch.randperm(X_init.shape[0], device=X_init.device)[:10000]
        X_init_sub = X_init[indices]
    else:
        X_init_sub = X_init
        
    _, T_icp = icp(X_init_sub, Y, optimize_scale=True, optimize_rotation=optimize_rotation)
    
    T_final = T_icp @ T_init
    X_final = bones.__transform(T_final, X)
    return X_final, T_final


def icp_animodel(X, Y, optimize_rotation=True):
    """
    Animodel-specific ICP alignment that explores multiple initializations:
    1. Run ICP alignment twice:
       - with no extra rotation
       - with an extra 180 degree rotation around Y-axis
    3. Return the best transform based on Chamfer Distance.
    """
    device = X.device
    dtype = X.dtype
    
    # 2. Setup Y-axis rotations (0 and 180 degrees)
    T_y0 = torch.eye(4, device=device, dtype=dtype)
    T_y180 = torch.eye(4, device=device, dtype=dtype)
    T_y180[0, 0] = -1.0
    T_y180[2, 2] = -1.0
    
    best_cd = float('inf')
    best_X = None
    best_T = None
    
    for T_y in [T_y0, T_y180]:
        # T_init = T_y @ T_x90
        T_init = T_y
        X_init = bones.__transform(T_y, X)
        
        X_icp, T_icp = icp_start(X_init, Y, optimize_rotation=optimize_rotation)
        T_final = T_icp @ T_init
        cd = chamfer_dist(X_icp, Y)[1]
        
        if cd < best_cd:
            best_cd = cd
            best_X = X_icp
            best_T = T_final
            
    return best_X, best_T
