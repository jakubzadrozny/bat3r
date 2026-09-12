import logging
from pathlib import Path

import einops
import numpy as np
import torch
import torch.utils.data as tud
import trimesh
# from psbody.mesh import Mesh
# from psbody.mesh.visibility import visibility_compute

import bat3r.raster.utils as ut
import bat3r.dataset as dd
from bat3r.utils import (
    rescale_im_and_mask,
    GLTF2,
    process_gltf,
    OneMeshGltf,
    read_meta,
    read_camera,
    read_fuse_image,
    read_mask,
    read_image,
)
from bat3r.skin import quaternion_to_matrix, skin_mesh
try:
    from PIL import Image
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _read_coo_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a dense tensor, and matching mask for a given sparse COO npz file.
    contains keys: shape tuple[int, ...], indices (N,len(shape)), values (N, C)

    returns:
        tensor: (*shape)
        mask: (*shape[:-1])
    """

    data = np.load(path)
    shape = data["shape"]
    indices = data["indices"]
    values = data["values"]

    tensor = np.zeros(shape, dtype=values.dtype)
    mask = np.zeros(shape[:-1], dtype=np.bool_)

    if indices.shape[-1] == 3:
        tensor[indices[:, 0], indices[:, 1], indices[:, 2]] = values
        mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    elif indices.shape[-1] == 2:
        tensor[indices[:, 0], indices[:, 1]] = values
        mask[indices[:, 0], indices[:, 1]] = True
    else:
        raise ValueError(f"Invalid indices shape: {indices.shape}")

    return tensor, mask


def _read_pointmap_npz(path: Path) -> torch.Tensor:
    pointmap, points_mask = (torch.from_numpy(t) for t in _read_coo_npz(path))
    return pointmap


def _read_feats_npz(path: Path) -> torch.Tensor:
    feats, feats_mask = (torch.from_numpy(f) for f in _read_coo_npz(path))

    if feats.is_floating_point():
        return feats

    feats = feats.float() / 127 - 1
    return feats


class DualPmDataset(tud.Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int,
        num_layers: int,
        include_ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        exclude_5000s: bool = True,
        **kwargs,
    ):
        if isinstance(root, str):
            root = Path(root)

        self.root = root
        self.resolution = image_size
        self.image_size = (image_size, image_size)
        self.num_layers = num_layers

        self._feats_dir = self.root / "features"
        if not self._feats_dir.exists():
            raise FileNotFoundError(f"Features directory {self._feats_dir} does not exist")

        self._mask_dir = self.root / "masks"
        if not self._mask_dir.exists():
            raise FileNotFoundError(f"Masks directory {self._mask_dir} does not exist")

        self._render_dir = self.root / "renders"
        if self._render_dir.exists():
            # exclude 5000s as they are corrupted..
            ids = (
                p.stem.split("_rgb")[0]
                for p in self._render_dir.glob("*_rgb.png")
                if not exclude_5000s or not p.stem.isnumeric() or int(p.stem) % 5000
            )
        else:
            logger.warning(f"Renders directory {self._render_dir} does not exist")
            ids = (
                p.stem.split("_feat")[0]
                for p in self._feats_dir.glob("*_feat.png")
                if not exclude_5000s or not p.stem.isnumeric() or int(p.stem) % 5000
            )

        if include_ids or exclude_ids:
            ids = set(ids)
        if include_ids:
            ids &= set(include_ids)
        if exclude_ids:
            ids -= set(exclude_ids)

        self.ids = sorted(ids)
        self.collate_fn = dd.PointmapDataset.collate_fn

    def _read_images(self, file_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        input_image = read_fuse_image(self._feats_dir / f"{file_id}_feat.png")

        # else:
        #     # rgb_image = torch.from_numpy(
        #     #     np.array(Image.open(self._render_dir / f"{file_id}_rgb.png"))
        #     # )
        #     rgb_image = read_image(self._render_dir / f"{file_id}_rgb.png")
        #     input_image = rgb_image

        # mask = torch.from_numpy(
        #     np.array(Image.open(self._mask_dir / f"{file_id}_mask.png"), dtype=np.float32) / 255.
        # )
        mask = read_mask(self._mask_dir / f"{file_id}_mask.png")
        input_image, mask = rescale_im_and_mask(
            input_image, mask, (self.resolution, self.resolution)
        )
        return input_image, mask

    @staticmethod
    def collate_fn(batch: list[tuple]) -> tuple:
        """Combines mulitple PointmapBatch of different examples into a single PointmapBatch"""
        return dd.PointmapDataset.collate_fn(batch)


class RasteredDataset(DualPmDataset):
    """
    dataset to be used if you pre-raster the pointmaps
    using scripts/raster_pointmaps.py
    """

    def __init__(self, *args, return_pcs: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        self._points_dir = self.root / f"pointmaps_{self.resolution}"
        if not self._points_dir.exists():
            raise FileNotFoundError(
                f"Points directory {self._points_dir} does not exist"
            )
        self.render_at_load = False

        self.return_pcs = return_pcs
        if self.return_pcs:
            self.shape_root = self.root / "shapes"
            self.shapes = {
                p.name: _read_shape(p / f"{p.stem}_shape.gltf")
                for p in self.shape_root.iterdir()
                if p.is_dir()
            }
            self.posed_pc_root = self.root / "pointclouds_posed"


    def __getitem__(self, idx: int):
        id_ = self.ids[idx]

        # load the pointmap
        pointmap = _read_pointmap_npz(self._points_dir / f"{id_}.npz")[
            :, :, : self.num_layers
        ]

        # match the expected shape of the rest of the code.. (NC, H, W)
        pointmap = einops.rearrange(pointmap, "h w n c -> (n c) h w")

        feats, feats_mask = None, None
        rgb_image = None
        input_image = None

        input_image, mask = self._read_images(id_)

        if self.return_pcs:
            meta = read_meta(self.root / "metadata" / f"{id_}_metadata.txt")
            shape_id = meta["model_name"]
            model = self.shapes[shape_id]
            posed_pc = trimesh.load(self.posed_pc_root / f"{id_}_posed.ply")
            render_args = dict(
                canonical_verts=model.vertices.to(torch.float32),
                posed_verts_camera_space=posed_pc.vertices.to(torch.float32),
                faces=model.faces,
            )
            return input_image, pointmap, mask, id_, render_args
        
        return input_image, pointmap, mask, id_


def _read_shape(path: Path) -> OneMeshGltf:
    gltf = GLTF2().load(path)
    return process_gltf(gltf)


def _transpose(x: torch.Tensor) -> torch.Tensor:
    return einops.rearrange(x, "... c r -> ... r c")


def _read_pose(path: Path) -> torch.Tensor:
    """
    read (num joints, 7)
    returns (num joints, 4, 4) as col major transforms
    """

    data = np.load(path)
    if path.suffix == '.npy':
        return torch.from_numpy(data).to(dtype=torch.float64)

    quat, pos = torch.from_numpy(data["poses"]).split([4, 3], dim=-1)
    rotmat = quaternion_to_matrix(quat)

    transform = torch.zeros(
        *quat.shape[:-1], 4, 4, device=quat.device, dtype=quat.dtype
    )
    transform[..., :3, :3] = rotmat
    transform[..., :3, 3] = pos
    transform[..., 3, 3] = 1
    return transform


def extract_joint_edges(joint_positions: torch.Tensor, model) -> np.ndarray:
    """
    Create edges between joints and their parents.

    Args:
        joint_positions: (num_joints, 3) tensor of joint positions
        nodes_parents_list: list of length num_joints, parent index for each joint (-1 for root)

    Returns:
        edges: (num_edges, 2, 3) numpy array of edges from joint -> parent
    """
    edges = []
    joint_positions_np = joint_positions.cpu().numpy() if torch.is_tensor(joint_positions) else joint_positions

    for child_idx in model.joints:
        child_idx = child_idx.item()
        # child_pos = joint_positions_np[child_idx]
        for parent_idx in model.nodes_parents_list[child_idx]:
            try:
                parent_idx = model.joints.tolist().index(parent_idx)
                # parent_pos = joint_positions_np[parent_idx]
                edges.append([child_idx, parent_idx])
            except ValueError:
                pass

    return np.array(edges)


class RasterizeDataset(DualPmDataset):
    """
    standard dataset, returns models to be rendered
    """

    def __init__(
            self,
            root: str | Path,
            image_size: int,
            num_layers: int,
            return_pcs: bool = False,
            render_at_load: bool = False,
            apply_pose_flag: bool = True,
            sensor_height: float = 36.0,
            pose_ext: str = "npz",
            return_vis: bool = True,
            **_,
        ):
        super().__init__(root, image_size, num_layers)
        self.shape_root = self.root / "shapes"
        self.shapes = {
            p.name: _read_shape(p / f"{p.stem}_shape.gltf")
            for p in self.shape_root.iterdir()
            if p.is_dir()
        }
        self.render_at_load = render_at_load
        self.return_pcs = return_pcs
        self.apply_pose_flag = apply_pose_flag
        self.sensor_height = sensor_height
        self.pose_ext = pose_ext
        self.return_vis = return_vis

        # self._points_dir = self.root / f"pointmaps_{self.resolution}"
        # if not self._points_dir.exists():
        #     raise FileNotFoundError(
        #         f"Points directory {self._points_dir} does not exist"
        #     )

    def apply_pose(self, pose: torch.Tensor, shape: OneMeshGltf) -> OneMeshGltf:
        shape.local_joint_transforms = _transpose(pose)

        verts, global_joint_transforms, *_ = skin_mesh(shape)
        return verts, global_joint_transforms

    def _get_render_args(self, file_id: str) -> dict:
        meta = read_meta(self.root / "metadata" / f"{file_id}_metadata.txt")
        focal_length = torch.tensor(meta["focal_length"], dtype=torch.float32)
        
        shape_id = meta["model_name"]
        model = self.shapes[shape_id]

        # TODO: use context manager below?
        view_matrix, camera_pose = read_camera(
            (self.root / "cameras" / f"{file_id}_camera.txt").open().read()
        )
        faces = model.faces

        if self.apply_pose_flag:
            pose = _read_pose(self.root / "poses" / f"{file_id}_pose.{self.pose_ext}")
            view_verts, global_joint_transforms = self.apply_pose(pose, model)
        else:
            view_verts = model.vertices

        if self.return_vis:
            vis_info = np.load(self.root / "visibility" / f"{file_id}_vis.npz")
            vis = ((vis_info["vis"] * vis_info["n_dot_cam"]) > 0.1).squeeze()
            vis = torch.from_numpy(vis)
        else:
            vis = None

        # mesh = Mesh(view_verts, faces)
        # args = {
        #     'v': mesh.v,
        #     'f': mesh.f,
        #     'cams': camera_pose[:3, 3:].T.numpy().astype(np.double),
        #     'n':  mesh.vn if hasattr(mesh, 'vn') else mesh.estimate_vertex_normals()
        # }
        # vis, n_dot_cam = visibility_compute(**args)
        # np.savez(
        #     self.root / "visibility" / f"{file_id}_vis.npz",
        #     vis=vis,
        #     n_dot_cam=n_dot_cam,
        # )

        joint_positions = global_joint_transforms[..., 3, :3]  # shape (num_joints, 3)

        # joints_np = joint_positions.cpu().numpy() if torch.is_tensor(joint_positions) else joint_positions
        # verts_np = view_verts.cpu().numpy() if torch.is_tensor(view_verts) else view_verts
        # dist_matrix = distance_matrix(joints_np, verts_np)  # shape: (num_joints, num_vertices)
        # nearest_distances = dist_matrix.min(axis=1)
        # joints_out = model.joints[nearest_distances > 0.3].tolist()
        # print(np.argwhere(nearest_distances > 0.3), joints_out, [model.node_names[i] for i in joints_out])

        return (
            view_verts.to(torch.float32),
            model.vertices.to(torch.float32),
            faces,
            view_matrix,
            focal_length,
            vis,
            shape_id,
            joint_positions.to(torch.float32),
        )

    def __getitem__(self, idx: int):
        file_id = self.ids[idx]

        view_verts, canonical_verts, faces, view_matrix, focal_length, vis, shape_id, joint_pos = (
            self._get_render_args(file_id)
        )

        render_args = dict(
            pose_verts=view_verts,
            canonical_verts=canonical_verts,
            faces=faces,
            model_view=view_matrix,
            focal_length=focal_length,
            vis=vis,
            shape_id=shape_id,
            joint_positions=joint_pos,
            # idx=idx,
        )

        if self.return_pcs:
            camera_space_verts = ut.apply_transform(view_verts, view_matrix)
            _cot_half_fov = ut.cot_half_fov(focal_length, self.sensor_height)
            projection = ut.perspective_matrix(_cot_half_fov)[0]
            clip_verts = ut.apply_projection(camera_space_verts, projection)
            verts_ndc = clip_verts / clip_verts[:, 3:]

            # Extract z-translation from model-view matrix (camera depth)
            camera_space_verts -= torch.tensor([0, 0, view_matrix[2, 3]])
            render_args["posed_verts_camera_space"] = camera_space_verts
            render_args["verts_ndc"] = verts_ndc


        # pointmap = _read_pointmap_npz(self._points_dir / f"{file_id}.npz")[
        #     :, :, : self.num_layers
        # ]
        # # match the expected shape of the rest of the code.. (NC, H, W)
        # model_targets = einops.rearrange(pointmap, "h w n c -> (n c) h w")

        model_targets = None
        if self.render_at_load:
            model_targets = einops.rearrange(
                self.renderer(**{k: v[None] for k, v in render_args.items()}),
                "b h w n c-> b (n c) h w",
            )

            render_args = None

        input_image, mask = self._read_images(file_id)

        # return (
        #     file_id,
        #     render_args,
        # )

        return (
            input_image,
            model_targets,
            mask,
            file_id,
            render_args,
        )

    def __len__(self) -> int:
        return len(self.ids)


class TestDataset(tud.Dataset):
    def __init__(
        self,
        image_dir: Path | None,
        mask_dir: Path,
        feat_dir: Path,
        image_size: tuple[int, int],
        include_ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
    ):
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.mask_dir = Path(mask_dir)
        self.feat_dir = Path(feat_dir) if feat_dir is not None else None

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.ids = self._find_ids(include_ids, exclude_ids)

    def _find_ids(
        self, include_ids: list[str] | None = None, exclude_ids: list[str] | None = None
    ):
        image_ids = None
        feat_ids = None
        if self.image_dir is not None:
            image_ids = set(
                p.stem.split("_rgb")[0] for p in self.image_dir.glob("*_rgb.png")
            )
        if self.feat_dir is not None:
            feat_ids = set(
                p.stem.split("_feat")[0] for p in self.feat_dir.glob("*_feat.png")
            )
        ids = set(p.stem.split("_mask")[0] for p in self.mask_dir.glob("*_mask.png"))

        if image_ids is not None:
            ids &= image_ids
        if feat_ids is not None:
            ids &= feat_ids
        if include_ids:
            ids &= set(include_ids)
        if exclude_ids:
            ids -= set(exclude_ids)

        return sorted(ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        file_id = self.ids[idx]
        if self.image_dir is not None:
            image = read_image(
                (self.image_dir / f"{file_id}_rgb.png").open("rb").read(),
            )

        feat = read_fuse_image(
            (self.feat_dir / f"{file_id}_feat.png").open("rb").read()
        )
        mask = read_mask((self.mask_dir / f"{file_id}_mask.png").open("rb").read())

        feat, mask = rescale_im_and_mask(feat, mask, self.image_size)

        return file_id, image, mask, feat

    @staticmethod
    def collate_fn(batch: list[tuple]) -> tuple:
        file_ids, images, masks, feats = list(zip(*batch, strict=True))
        images = torch.stack(images) if images[0] is not None else None
        masks, feats = torch.stack(masks), torch.stack(feats)
        return file_ids, images, masks, feats


class AnimodelDataset(tud.Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int,
        sam3d: bool = False,
        **kwargs,
    ):
        if isinstance(root, str):
            root = Path(root)
        self.root = root
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.features_dir = self.root / "features"
        self.image_size = (image_size, image_size)
        self.image_paths = sorted(list(self.image_dir.glob("*_rgb.png")))
        self.sam3d = sam3d

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        file_id = img_path.stem.split('_rgb')[0]

        if self.sam3d:
            input_image = torch.from_numpy(sam_load_image(self.image_dir / f"{file_id}_rgb.png"))
            mask = torch.from_numpy(sam_load_mask(self.mask_dir / f"{file_id}_mask.png"))
        else:
            input_image = read_fuse_image(self.features_dir / f"{file_id}_feat.png")
            mask = read_mask(self.mask_dir / f"{file_id}_mask.png")
            input_image, mask = rescale_im_and_mask(
                input_image, mask, self.image_size
            )

        render_args = dict()

        model_targets = None

        return input_image, model_targets, mask, file_id, render_args

    @staticmethod
    def collate_fn(batch: list[tuple]) -> tuple:
        return dd.PointmapDataset.collate_fn(batch)


class RealDataset(tud.Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int,
        image_ext: str = "jpg",
        **kwargs,
    ):
        if isinstance(root, str):
            root = Path(root)
        self.root = root
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.features_dir = self.root / "features"
        if not self.features_dir.exists():
            self.features_dir = None
        self.image_size = (image_size, image_size)
        self.image_ext = image_ext

        png_ids = [path.name.replace("_rgb.png", "") for path in self.image_dir.glob("*_rgb.png")]
        jpg_ids = [path.name.replace("_rgb.jpg", "") for path in self.image_dir.glob("*_rgb.jpg")]
        self.ids = sorted(list(set(png_ids + jpg_ids)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # img_path = self.image_paths[idx]
        # file_id = img_path.stem.split('_rgb')[0]
        file_id = self.ids[idx]

        input_image = read_fuse_image(self.features_dir / f"{file_id}_feat.png")

        # mask_path = self.mask_dir / f"{file_id}_mask.png"
        # mask_im = read_image(mask_path)
        # if mask_im.ndim == 3 and mask_im.shape[-1] == 4:
        #     mask = mask_im[..., 3]
        # else:
        #     mask = mask_im.mean(dim=-1) if mask_im.ndim == 3 else mask_im
        # mask = (mask > 0.5).float()

        mask = read_mask(self.mask_dir / f"{file_id}_mask.png")
        if mask.ndim == 3:
            mask = torch.amax(mask, dim=-1)
        input_image, mask = rescale_im_and_mask(
            input_image, mask, self.image_size
        )

        render_args = dict()

        model_targets = None

        return input_image, model_targets, mask, file_id, render_args

    @staticmethod
    def collate_fn(batch: list[tuple]) -> tuple:
        return dd.PointmapDataset.collate_fn(batch)
    

def sam_load_image(path):
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(np.uint8)
    return image

def sam_load_mask(path):
    mask = sam_load_image(path)
    mask = mask > 0
    if mask.ndim == 3:
        mask = np.amax(mask, axis=-1)
    return mask

class Sam3DDataset(DualPmDataset):
    """
    standard dataset, returns models to be rendered
    """

    def __init__(
            self,
            root: str | Path,
            image_size: int,
            num_layers: int,
            return_pcs: bool = False,
            apply_pose_flag: bool = True,
            sensor_height: float = 36.0,
            pose_ext: str = "npz",
            **_,
        ):
        super().__init__(root, image_size, num_layers)
        self.shape_root = self.root / "shapes"
        self.shapes = {
            p.name: _read_shape(p / f"{p.stem}_shape.gltf")
            for p in self.shape_root.iterdir()
            if p.is_dir()
        }
        self.return_pcs = return_pcs
        self.apply_pose_flag = apply_pose_flag
        self.pose_ext = pose_ext
        self.sensor_height = sensor_height

    def apply_pose(self, pose: torch.Tensor, shape: OneMeshGltf) -> OneMeshGltf:
        shape.local_joint_transforms = _transpose(pose)

        verts, global_joint_transforms, *_ = skin_mesh(shape)
        return verts, global_joint_transforms

    def _get_render_args(self, file_id: str) -> dict:
        meta = read_meta(self.root / "metadata" / f"{file_id}_metadata.txt")
        focal_length = torch.tensor(meta["focal_length"], dtype=torch.float32)
        
        shape_id = meta["model_name"]
        model = self.shapes[shape_id]

        # TODO: use context manager below?
        view_matrix, camera_pose = read_camera(
            (self.root / "cameras" / f"{file_id}_camera.txt").open().read()
        )
        faces = model.faces

        if self.apply_pose_flag:
            pose = _read_pose(self.root / "poses" / f"{file_id}_pose.{self.pose_ext}")
            view_verts, global_joint_transforms = self.apply_pose(pose, model)
        else:
            view_verts = model.vertices

        joint_positions = global_joint_transforms[..., 3, :3]  # shape (num_joints, 3)

        return (
            view_verts.to(torch.float32),
            model.vertices.to(torch.float32),
            faces,
            view_matrix,
            focal_length,
            shape_id,
            joint_positions.to(torch.float32),
        )

    def __getitem__(self, idx: int):
        file_id = self.ids[idx]

        view_verts, canonical_verts, faces, view_matrix, focal_length, shape_id, joint_pos = (
            self._get_render_args(file_id)
        )

        render_args = dict(
            pose_verts=view_verts,
            canonical_verts=canonical_verts,
            faces=faces,
            model_view=view_matrix,
            focal_length=focal_length,
            shape_id=shape_id,
            joint_positions=joint_pos,
            # idx=idx,
        )

        if self.return_pcs:
            camera_space_verts = ut.apply_transform(view_verts, view_matrix)
            _cot_half_fov = ut.cot_half_fov(focal_length, self.sensor_height)
            projection = ut.perspective_matrix(_cot_half_fov)[0]
            clip_verts = ut.apply_projection(camera_space_verts, projection)
            verts_ndc = clip_verts / clip_verts[:, 3:]

            # Extract z-translation from model-view matrix (camera depth)
            camera_space_verts -= torch.tensor([0, 0, view_matrix[2, 3]])
            render_args["posed_verts_camera_space"] = camera_space_verts
            render_args["verts_ndc"] = verts_ndc

        model_targets = None

        input_image = sam_load_image(self._render_dir / f"{file_id}_rgb.png")
        mask = sam_load_mask(self._mask_dir / f"{file_id}_mask.png")
        # input_image, mask = rescale_im_and_mask(
        #     input_image, mask, (self.resolution, self.resolution)
        # )

        return (
            torch.from_numpy(input_image),
            model_targets,
            torch.from_numpy(mask),
            file_id,
            render_args,
        )

    def __len__(self) -> int:
        return len(self.ids)


class Sam3DRealDataset(tud.Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int,
        image_ext: str = "jpg",
        **kwargs,
    ):
        if isinstance(root, str):
            root = Path(root)
        self.root = root
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.image_size = (image_size, image_size)

        self.image_ext = image_ext

        ids = [path.stem.split('_rgb')[0] for path in self.image_dir.glob(f"*_rgb.{self.image_ext}")]
        self.ids = sorted(ids)
        # self.image_paths = sorted(list(self.image_dir.glob("*_rgb.png")))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # img_path = self.image_paths[idx]
        # file_id = img_path.stem.split('_rgb')[0]
        file_id = self.ids[idx]

        input_image = sam_load_image(self.image_dir / f"{file_id}_rgb.{self.image_ext}")
        mask = sam_load_mask(self.mask_dir / f"{file_id}_mask.png")
        # input_image = input_image * mask[..., None]

        render_args = dict()
        model_targets = None
        return (
            torch.from_numpy(input_image),
            model_targets,
            torch.from_numpy(mask),
            file_id,
            render_args,
        )

    @staticmethod
    def collate_fn(batch: list[tuple]) -> tuple:
        return dd.PointmapDataset.collate_fn(batch)
