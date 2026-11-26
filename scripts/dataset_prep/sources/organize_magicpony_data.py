# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.

import argparse
import json
import shutil
from pathlib import Path


def organize_magicpony_data(src_dir: Path, tgt_dir: Path) -> None:
    src_dir = Path(src_dir)
    tgt_dir = Path(tgt_dir)

    tgt_dir.mkdir(parents=True, exist_ok=True)
    (tgt_dir / "images").mkdir(exist_ok=True)
    (tgt_dir / "masks").mkdir(exist_ok=True)

    train_ids = []
    val_ids = []
    test_ids = []

    renaming_dict = {}
    id_to_path = {}
    counter = 0

    print(f"Processing files from {src_dir}...")

    # Sort to ensure deterministic order
    for file_path in sorted(src_dir.rglob("*")):
        if not file_path.is_file():
            continue

        rel_path = file_path.relative_to(src_dir)

        # Up to 2 subdirs deep -> files at depth <= 3 (root/split/seq/file)
        if len(rel_path.parts) > 3:
            continue

        name = file_path.name

        if file_path.stem.endswith("_rgb"):
            if file_path.suffix != ".jpg":
                print(f"Flagging non-jpg RGB file: {rel_path}")
                continue

            old_stem = name.rsplit("_rgb", 1)[0]
            key = str(rel_path.parent / old_stem)

            if key not in renaming_dict:
                renaming_dict[key] = f"{counter:06d}"
                counter += 1
            new_stem = renaming_dict[key]

            new_name = f"{new_stem}_rgb{file_path.suffix}"
            shutil.copy2(file_path, tgt_dir / "images" / new_name)

            if len(rel_path.parts) > 1:
                split_dir = rel_path.parts[0]
                if split_dir in ("train", "val", "test"):
                    if new_stem in id_to_path:
                        print(
                            f"Duplicate ID {new_stem} detected when appending to {split_dir}_ids!"
                        )
                        print(f"  Original: {id_to_path[new_stem]}")
                        print(f"  Current:  {file_path}")
                    else:
                        id_to_path[new_stem] = file_path

                if split_dir == "train":
                    train_ids.append(new_stem)
                elif split_dir == "val":
                    val_ids.append(new_stem)
                elif split_dir == "test":
                    test_ids.append(new_stem)
            else:
                print("Unexpected shallow path:", rel_path)

        elif name.endswith("_mask.png"):
            old_stem = name.rsplit("_mask", 1)[0]
            key = str(rel_path.parent / old_stem)

            if key not in renaming_dict:
                renaming_dict[key] = f"{counter:06d}"
                counter += 1
            new_stem = renaming_dict[key]

            new_name = f"{new_stem}_mask.png"
            shutil.copy2(file_path, tgt_dir / "masks" / new_name)

        elif (
            name.endswith("_box.txt")
            or name.endswith("_feat16.png")
            or name.endswith("_flow.png")
            or name.endswith("_metadata.json")
        ):
            continue

        else:
            print(f"Warning: Unknown file type encountered: {rel_path}")

    print(f"Train IDs: {len(train_ids)}")
    print(f"Val IDs: {len(val_ids)}")
    print(f"Test IDs: {len(test_ids)}")

    with (tgt_dir / "train_ids.json").open("w") as f:
        json.dump(sorted(train_ids), f)

    with (tgt_dir / "val_ids.json").open("w") as f:
        json.dump(sorted(val_ids), f)

    with (tgt_dir / "test_ids.json").open("w") as f:
        json.dump(sorted(test_ids), f)

    print("\nVerifying dataset integrity...")
    for split_name, id_list in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        print(f"Checking {split_name} ({len(id_list)} IDs)...")
        missing_rgb = []
        missing_mask = []

        unique_ids = set(id_list)
        if len(id_list) != len(unique_ids):
            print(
                f"  Warning: {split_name} list contains duplicates! "
                f"(Total: {len(id_list)}, Unique: {len(unique_ids)})"
            )

        for stem in id_list:
            rgb_path = tgt_dir / "images" / f"{stem}_rgb.jpg"
            mask_path = tgt_dir / "masks" / f"{stem}_mask.png"

            if not rgb_path.exists():
                missing_rgb.append(stem)
            if not mask_path.exists():
                missing_mask.append(stem)

        if missing_rgb:
            print(f"  Missing RGB images: {len(missing_rgb)}")
        if missing_mask:
            print(f"  Missing masks: {len(missing_mask)}")

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten MagicPony horse_combined dataset into images/, masks/, and split JSONs."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        required=True,
        help="Path to extracted MagicPony horse_combined directory (containing train/, val/, test/).",
    )
    parser.add_argument(
        "--tgt-dir",
        type=Path,
        required=True,
        help="Output directory for flattened images/, masks/, and {train,val,test}_ids.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    organize_magicpony_data(src_dir=args.src_dir, tgt_dir=args.tgt_dir)


if __name__ == "__main__":
    main()
