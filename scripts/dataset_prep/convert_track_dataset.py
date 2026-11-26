# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import yaml
from PIL import Image
from tqdm import tqdm


def check_disjoint_clips(input_dirs: Sequence[Path]) -> bool:
    clip_locations = defaultdict(list)

    for dir_path in input_dirs:
        p = Path(dir_path)
        if not p.is_dir():
            print(f"Warning: {p} is not a valid directory, skipping disjoint check.")
            continue

        for clip_dir in p.iterdir():
            if clip_dir.is_dir():
                clip_locations[clip_dir.name].append(str(p))

    duplicates = {
        clip: locations for clip, locations in clip_locations.items() if len(locations) > 1
    }

    if duplicates:
        print(f"Error: Found {len(duplicates)} overlapping clips!")
        for clip, locations in duplicates.items():
            print(f"  Clip '{clip}' found in: {', '.join(locations)}")
        return False

    print("Success: All top-level directories are disjoint. No overlapping clips found.")
    return True


def convert_dataset(
    input_dirs: Sequence[Path],
    output_dir: Path,
    frame_step: int = 5,
    dry_run: bool = False,
) -> int:
    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")

    if not check_disjoint_clips(input_dirs):
        sys.exit(1)

    output_dir = Path(output_dir)
    if not dry_run:
        images_dir = output_dir / "images"
        masks_dir = output_dir / "masks"
        metadata_dir = output_dir / "metadata"

        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    total_clips = 0
    total_tracks = 0
    frames_per_clip = {}
    frames_per_track = {}

    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        clip_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        total_clips += len(clip_dirs)

        for clip_dir in tqdm(clip_dirs, desc=f"Processing clips in {input_dir.name}"):
            track_dirs = sorted([d for d in clip_dir.iterdir() if d.is_dir()])
            total_tracks += len(track_dirs)
            clip_frame_count = 0

            for track_dir in track_dirs:
                rgb_files = sorted(track_dir.glob("*_rgb.png"))
                track_frame_count = 0
                valid_frame_counter = 0

                for rgb_file in rgb_files:
                    frame_idx = rgb_file.name.removesuffix("_rgb.png")

                    mask_file = track_dir / f"{frame_idx}_mask.png"
                    meta_file = track_dir / f"{frame_idx}_metadata.json"

                    if not mask_file.exists() or not meta_file.exists():
                        print(
                            f"Warning: Missing mask or metadata for {rgb_file.name} in {track_dir}, skipping."
                        )
                        continue

                    valid_frame_counter += 1

                    # Keep frames 1, 1 + K, 1 + 2K, ... (works for frame_step == 1 as well)
                    if (valid_frame_counter - 1) % frame_step != 0:
                        continue

                    if not dry_run:
                        try:
                            with Image.open(rgb_file) as img:
                                img.load()
                            with Image.open(mask_file) as mask_img:
                                mask_img.load()
                        except Exception as e:
                            print(
                                f"Warning: Failed to load image or mask for {rgb_file.name} in {track_dir}, skipping. Error: {e}"
                            )
                            continue

                    new_stem = f"{idx:06d}"

                    if not dry_run:
                        shutil.copy2(rgb_file, images_dir / f"{new_stem}_rgb.png")
                        shutil.copy2(mask_file, masks_dir / f"{new_stem}_mask.png")

                        with open(meta_file, "r") as f:
                            meta_data = json.load(f)

                        meta_data["track_id"] = track_dir.name
                        meta_data["clip_id"] = clip_dir.name
                        meta_data["frame_idx"] = frame_idx
                        meta_data["original_stem"] = (
                            f"{input_dir.name}_{clip_dir.name}_{track_dir.name}_{frame_idx}"
                        )

                        with open(metadata_dir / f"{new_stem}_metadata.txt", "w") as f:
                            yaml.dump(meta_data, f, sort_keys=False)

                    idx += 1
                    track_frame_count += 1
                    clip_frame_count += 1

                if track_frame_count > 0:
                    frames_per_track[f"{clip_dir.name}/{track_dir.name}"] = track_frame_count
            if clip_frame_count > 0:
                frames_per_clip[clip_dir.name] = clip_frame_count

    if dry_run:
        print("\n--- DRY RUN RESULTS ---")
        print(f"Number of different clips: {total_clips}")
        print(f"Number of tracks: {total_tracks}")
        print("\nFrames per clip:")
        for cid, count in frames_per_clip.items():
            print(f"  {cid}: {count}")
        print("\nFrames per track:")
        for tid, count in frames_per_track.items():
            print(f"  {tid}: {count}")
        print("\nTotal frames:", sum(frames_per_clip.values()))
    else:
        print(f"Conversion complete! Total valid samples processed: {idx}")

    return idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten Animal-in-Motion <clip_id>/<track_id> tracks into images/, masks/, metadata/."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more track root directories containing <clip_id>/<track_id>/ subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for flattened images/, masks/, and metadata/.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help="Extract every K-th valid frame per track (default: 5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count clips, tracks, and selected frames without copying files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dataset(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        frame_step=args.frame_step,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
