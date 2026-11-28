# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import KDTree

from bat3r.alt_dataset import _read_shape
from bat3r.refine_kinematic_chain import get_edges_and_lengths


def compute_geodesic_distances(
    verts: np.ndarray,
    faces: np.ndarray,
    proximity_threshold: float = 0.05,
) -> Tuple[np.ndarray, int]:
    """Compute all-pairs approximate geodesic distances on a mesh graph.

    Adds proximity bridge edges between non-adjacent vertices within
    `proximity_threshold` so UV seams and separate sub-meshes (e.g. eyes)
    remain connected.
    """
    edges, lens_sq = get_edges_and_lengths(
        torch.from_numpy(verts), torch.from_numpy(faces)
    )
    edges = edges.numpy()
    weights = np.sqrt(lens_sq.numpy())

    tree = KDTree(verts)
    raw_bridge_pairs = tree.query_pairs(r=proximity_threshold)

    bridge_pairs_normalized = {tuple(sorted(pair)) for pair in raw_bridge_pairs}
    existing_edge_set = {tuple(sorted((int(a), int(b)))) for a, b in edges}
    new_bridge_pairs = bridge_pairs_normalized - existing_edge_set

    if new_bridge_pairs:
        bridge_edges = np.array(list(new_bridge_pairs), dtype=np.int64)
        b_v_a = verts[bridge_edges[:, 0]]
        b_v_b = verts[bridge_edges[:, 1]]
        raw_bridge_dists = np.sqrt(np.sum((b_v_a - b_v_b) ** 2, axis=1))
        bridge_weights = np.maximum(raw_bridge_dists, 1e-6)

        final_rows = np.concatenate([edges[:, 0], bridge_edges[:, 0]])
        final_cols = np.concatenate([edges[:, 1], bridge_edges[:, 1]])
        final_data = np.concatenate([weights, bridge_weights])
        n_bridges = len(bridge_edges)
    else:
        final_rows = edges[:, 0]
        final_cols = edges[:, 1]
        final_data = weights
        n_bridges = 0

    n_verts = len(verts)
    graph = csr_matrix((final_data, (final_rows, final_cols)), shape=(n_verts, n_verts))
    geo_dists = shortest_path(graph, directed=False)
    return geo_dists, n_bridges


def save_compact_dists(out_path: Path, geo_dists: np.ndarray) -> None:
    np.savez_compressed(out_path, dists=geo_dists.astype(np.float16))


def process_gltf(gltf_path: Path, proximity_threshold: float, out_path: Path | None = None) -> Path:
    gltf_path = Path(gltf_path)
    if out_path is None:
        stem = gltf_path.stem.removesuffix("_shape")
        out_path = gltf_path.parent / f"{stem}_geo_dists.npz"

    print(f"Processing {gltf_path} (proximity_threshold={proximity_threshold})...")
    mesh = _read_shape(gltf_path)
    verts = mesh.vertices.numpy()
    faces = mesh.faces.numpy()

    geo_dists, n_bridges = compute_geodesic_distances(
        verts, faces, proximity_threshold=proximity_threshold
    )
    n_inf = int(np.isinf(geo_dists).sum())
    save_compact_dists(out_path, geo_dists)

    print(
        f"  Saved {out_path} | verts={len(verts)}, bridge_edges={n_bridges}, "
        f"disconnected_pairs={n_inf}"
    )
    if n_inf > 0:
        print(
            "  Warning: Mesh has disconnected components; consider increasing --proximity-threshold."
        )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute float16 compressed geodesic distance matrices (<shape>_geo_dists.npz) for .gltf meshes."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--shapes-dir",
        type=Path,
        help="Shapes directory containing <shape_id>/<shape_id>_shape.gltf subfolders.",
    )
    group.add_argument(
        "--gltf-path",
        type=Path,
        help="Path to a single <shape_id>_shape.gltf file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output .npz path (only valid with --gltf-path).",
    )
    parser.add_argument(
        "--proximity-threshold",
        type=float,
        default=0.05,
        help="Radius for KDTree bridge edges stitching seams/sub-meshes (default: 0.05; paper used 0.07 for horse, 0.05 for chimp & elephant).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gltf_path is not None:
        process_gltf(args.gltf_path, args.proximity_threshold, out_path=args.out)
        return

    if args.out is not None:
        raise ValueError("--out can only be used together with --gltf-path.")

    shape_dirs = sorted(p for p in args.shapes_dir.iterdir() if p.is_dir())
    if not shape_dirs:
        raise FileNotFoundError(f"No shape subdirectories found in {args.shapes_dir}")

    for shape_dir in shape_dirs:
        gltf_path = shape_dir / f"{shape_dir.name}_shape.gltf"
        if not gltf_path.exists():
            raise FileNotFoundError(f"Expected mesh file not found: {gltf_path}")
        process_gltf(gltf_path, args.proximity_threshold)


if __name__ == "__main__":
    main()
