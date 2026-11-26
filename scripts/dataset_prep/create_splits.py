# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml


def create_frame_splits(dataset_dir: Path, test_ratio: float = 0.2, seed: int = 42) -> None:
    images_dir = dataset_dir / "images"
    image_files = sorted(
        list(images_dir.glob("*_rgb.png")) + list(images_dir.glob("*_rgb.jpg")),
        key=lambda p: p.name,
    )
    print(f"Found {len(image_files)} frames. Splitting at frame level...")

    if not image_files:
        raise ValueError(f"No *_rgb.png or *_rgb.jpg files found in {images_dir}")

    all_frames = [f.stem.rsplit("_rgb", 1)[0] for f in image_files]
    num_frames = len(all_frames)

    rng = random.Random(seed)
    rng.shuffle(all_frames)

    num_test = max(1, int(num_frames * test_ratio)) if num_frames > 1 else 0
    test_ids = sorted(all_frames[:num_test])
    train_ids = sorted(all_frames[num_test:])

    _write_splits(dataset_dir, train_ids, test_ids)
    print(f"Split results:\n  Train: {len(train_ids)} frames\n  Test:  {len(test_ids)} frames")


def _extract_track_key(meta_data: dict, meta_file: Path) -> str:
    if "clip_id" in meta_data and "track_id" in meta_data:
        return f"{meta_data['clip_id']}/{meta_data['track_id']}"
    if "subdir" in meta_data and "track_id" in meta_data:
        return f"{meta_data['subdir']}/{meta_data['track_id']}"
    raise ValueError(
        f"Metadata file {meta_file} does not contain track keys "
        f"(expected clip_id+track_id or subdir+track_id, got keys {sorted(meta_data.keys())})."
    )


def create_track_splits(dataset_dir: Path, test_ratio: float = 0.2, seed: int = 42) -> None:
    metadata_dir = dataset_dir / "metadata"
    meta_files = sorted(metadata_dir.glob("*_metadata.txt"))
    print(f"Found {len(meta_files)} metadata files. Grouping by track...")

    if not meta_files:
        raise ValueError(f"No *_metadata.txt files found in {metadata_dir}")

    track_to_frames = defaultdict(list)
    for meta_file in meta_files:
        frame_id = meta_file.name.removesuffix("_metadata.txt")
        with open(meta_file, "r") as f:
            meta_data = yaml.safe_load(f) or {}
        track_key = _extract_track_key(meta_data, meta_file)
        track_to_frames[track_key].append(frame_id)

    unique_tracks = sorted(track_to_frames.keys())
    num_tracks = len(unique_tracks)
    print(f"Found {num_tracks} unique tracks.")

    rng = random.Random(seed)
    rng.shuffle(unique_tracks)

    num_test_tracks = max(1, int(num_tracks * test_ratio)) if num_tracks > 1 else 0
    test_tracks = set(unique_tracks[:num_test_tracks])

    train_ids = sorted(
        fid for track in unique_tracks if track not in test_tracks for fid in track_to_frames[track]
    )
    test_ids = sorted(
        fid for track in test_tracks for fid in track_to_frames[track]
    )

    _write_splits(dataset_dir, train_ids, test_ids)
    print(
        f"Split results:\n"
        f"  Train: {num_tracks - num_test_tracks} tracks, {len(train_ids)} frames\n"
        f"  Test:  {num_test_tracks} tracks, {len(test_ids)} frames"
    )


def _write_splits(dataset_dir: Path, train_ids, test_ids) -> None:
    with open(dataset_dir / "train_ids.json", "w") as f:
        json.dump(train_ids, f, indent=2)
    with open(dataset_dir / "test_ids.json", "w") as f:
        json.dump(test_ids, f, indent=2)
    print(f"Successfully created train_ids.json and test_ids.json in {dataset_dir}!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic train_ids.json and test_ids.json splits."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Preprocessed dataset directory containing images/ (and metadata/ for track mode).",
    )
    parser.add_argument(
        "--mode",
        choices=["frame", "track"],
        required=True,
        help="'frame' splits individual images randomly; 'track' keeps all frames of a track together.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of frames/tracks assigned to test_ids.json (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "frame":
        create_frame_splits(args.dataset_dir, test_ratio=args.test_ratio, seed=args.seed)
    else:
        create_track_splits(args.dataset_dir, test_ratio=args.test_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
