# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional, Sequence

import yaml
from tqdm import tqdm


def list_rgb_files(images_dir: Path):
    rgb_files = list(images_dir.glob("*_rgb.png")) + list(images_dir.glob("*_rgb.jpg"))
    return sorted(rgb_files, key=lambda p: p.name)


def merge_datasets(
    input_dirs: Sequence[Path],
    output_dir: Path,
    steps: Optional[Sequence[int]] = None,
) -> int:
    input_dirs = [Path(d) for d in input_dirs]
    output_dir = Path(output_dir)

    if steps is None:
        steps = [1] * len(input_dirs)
    elif len(steps) != len(input_dirs):
        raise ValueError(
            f"--steps length ({len(steps)}) must match --input-dirs length ({len(input_dirs)})"
        )
    elif any(s < 1 for s in steps):
        raise ValueError(f"All --steps values must be >= 1, got {steps}")

    for subdir in ("images", "masks", "metadata"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    current_id = 0
    out_train_ids = []
    out_val_ids = []
    out_test_ids = []

    for input_dir, step in zip(input_dirs, steps):
        print(f"Processing dataset: {input_dir.name} (step={step})")

        has_split_files = (input_dir / "train_ids.json").exists() and (
            input_dir / "test_ids.json"
        ).exists()
        src_train_ids = set()
        src_val_ids = set()
        src_test_ids = set()
        if has_split_files:
            with open(input_dir / "train_ids.json", "r") as f:
                src_train_ids = set(json.load(f))
            with open(input_dir / "test_ids.json", "r") as f:
                src_test_ids = set(json.load(f))
            if (input_dir / "val_ids.json").exists():
                with open(input_dir / "val_ids.json", "r") as f:
                    src_val_ids = set(json.load(f))

        meta_dir = input_dir / "metadata"
        has_metadata_dir = meta_dir.is_dir()

        img_files = list_rgb_files(input_dir / "images")[::step]

        for img_path in tqdm(img_files, desc=f"Copying {input_dir.name}"):
            old_id = img_path.stem.rsplit("_rgb", 1)[0]
            img_ext = img_path.suffix
            new_id = f"{current_id:06d}"

            src_mask = input_dir / "masks" / f"{old_id}_mask.png"
            src_meta = meta_dir / f"{old_id}_metadata.txt"

            if not src_mask.exists() or (has_metadata_dir and not src_meta.exists()):
                print(
                    f"Warning: Missing components for ID {old_id} in {input_dir.name}, skipping."
                )
                continue

            shutil.copy2(img_path, output_dir / "images" / f"{new_id}_rgb{img_ext}")
            shutil.copy2(src_mask, output_dir / "masks" / f"{new_id}_mask.png")

            dst_meta = output_dir / "metadata" / f"{new_id}_metadata.txt"
            if has_metadata_dir:
                shutil.copy2(src_meta, dst_meta)
            else:
                dummy_meta = {
                    "dataset": "image_dataset",
                    "original_stem": f"{input_dir.name}_{old_id}",
                }
                with open(dst_meta, "w") as f:
                    yaml.dump(dummy_meta, f, sort_keys=False)

            if has_split_files:
                if old_id in src_train_ids:
                    out_train_ids.append(new_id)
                elif old_id in src_val_ids:
                    out_val_ids.append(new_id)
                elif old_id in src_test_ids:
                    out_test_ids.append(new_id)
                else:
                    print(
                        f"Warning: ID {old_id} not found in split JSONs of {input_dir.name}"
                    )
            elif has_metadata_dir:
                with open(src_meta, "r") as f:
                    meta = yaml.safe_load(f) or {}
                split_val = str(meta.get("split", "")).lower()
                if "train" in split_val or "val" in split_val:
                    out_train_ids.append(new_id)
                elif "test" in split_val:
                    out_test_ids.append(new_id)
                else:
                    print(
                        f"Warning: Unknown split '{split_val}' for ID {old_id} in {input_dir.name}"
                    )
            else:
                print(
                    f"Warning: No split JSONs or metadata/ in {input_dir.name}; ID {new_id} unassigned."
                )

            current_id += 1

    with open(output_dir / "train_ids.json", "w") as f:
        json.dump(out_train_ids, f, indent=2)
    with open(output_dir / "test_ids.json", "w") as f:
        json.dump(out_test_ids, f, indent=2)
    if out_val_ids:
        with open(output_dir / "val_ids.json", "w") as f:
            json.dump(out_val_ids, f, indent=2)

    print(
        f"Merge complete! Total valid samples: {current_id} "
        f"(train={len(out_train_ids)}, test={len(out_test_ids)}"
        + (f", val={len(out_val_ids)}" if out_val_ids else "")
        + ")"
    )
    return current_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple preprocessed datasets into one sequential {idx:06d} dataset."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Input preprocessed dataset directories (in merge order).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output merged dataset directory.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help="Optional per-input-dir subsampling step (e.g. --steps 1 2 to keep every 2nd sample of the 2nd input).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_datasets(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
