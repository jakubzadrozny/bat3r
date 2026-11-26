# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seg_utils import add_common_sam_args, run_bbox_sam_pipeline


def load_ap10k_items(
    annotation_files: Sequence[Path],
    image_dir: Path,
    target_category: str,
):
    items = []
    image_dir = Path(image_dir)

    for annotation_file in annotation_files:
        print(f"Loading annotations from {annotation_file}...")
        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        target_cat_id = None
        for cat in coco_data.get("categories", []):
            if cat["name"].lower() == target_category.lower():
                target_cat_id = cat["id"]

        if target_cat_id is None:
            available_cats = [c["name"] for c in coco_data.get("categories", [])]
            print(f"Warning: Category '{target_category}' not found in {annotation_file}!")
            print(f"Available categories: {available_cats}")
            continue

        print(f"Target Category ID: {target_cat_id}")
        image_map = {img["id"]: img["file_name"] for img in coco_data.get("images", [])}
        split_name = Path(annotation_file).stem
        annotations = coco_data.get("annotations", [])
        print(f"Total annotations in {split_name}: {len(annotations)}")

        for ann in annotations:
            if ann["category_id"] != target_cat_id:
                continue

            file_name = image_map.get(ann["image_id"])
            if not file_name:
                continue

            bbox = ann["bbox"]
            target_box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            image_path = image_dir / file_name

            items.append(
                {
                    "image_path": image_path,
                    "bbox": target_box,
                    "metadata": {
                        "dataset": "ap-10k",
                        "split": split_name,
                        "filename": file_name,
                        "ann_id": ann["id"],
                        "bbox": target_box,
                        "original_stem": f"{image_path.stem}_{ann['id']}",
                    },
                }
            )

    print(f"Found {len(items)} bboxes containing '{target_category}'.")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment AP-10K category instances with SAM 2.1 using COCO bbox annotations."
    )
    add_common_sam_args(
        parser,
        data_root_help="Path to AP-10K root containing annotations/ and data/.",
        default_min_area=10000,
        default_fix_inverted=False,
    )
    parser.add_argument(
        "--category",
        type=str,
        default="chimpanzee",
        help="Target AP-10K category name (default: chimpanzee).",
    )
    parser.add_argument(
        "--split-ver",
        type=int,
        default=1,
        help="AP-10K split version number (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_jsons = [
        args.data_root / "annotations" / f"ap10k-{split}-split{args.split_ver}.json"
        for split in ("train", "val", "test")
    ]
    items = load_ap10k_items(
        annotation_files=annotation_jsons,
        image_dir=args.data_root / "data",
        target_category=args.category,
    )
    run_bbox_sam_pipeline(
        items,
        args.output_dir,
        sam_model_name=args.sam_model,
        min_area=args.min_area,
        target_size=args.target_size,
        fix_inverted=args.fix_inverted,
        save_masked=not args.no_save_masked,
    )


if __name__ == "__main__":
    main()
