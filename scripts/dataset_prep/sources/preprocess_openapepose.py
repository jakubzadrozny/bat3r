# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seg_utils import add_common_sam_args, run_bbox_sam_pipeline


def load_openapepose_items(
    annotation_files: Sequence[Path],
    image_dir: Path,
    target_category: str,
):
    items = []
    image_dir = Path(image_dir)
    ann_id = 0

    for annotation_file in annotation_files:
        print(f"Loading annotations from {annotation_file}...")
        with open(annotation_file, "r") as f:
            oap_data = json.load(f)

        split_name = Path(annotation_file).stem
        annotations = oap_data["data"]
        print(f"Total annotations in {split_name}: {len(annotations)}")

        for ann in annotations:
            if ann["species"] != target_category:
                continue

            current_ann_id = ann_id
            ann_id += 1

            file_name = ann["file"]
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
                        "dataset": "openapepose",
                        "split": split_name,
                        "filename": file_name,
                        "ann_id": current_ann_id,
                        "bbox": target_box,
                        "original_stem": f"{image_path.stem}_{current_ann_id}",
                    },
                }
            )

    print(f"Found {len(items)} bboxes containing '{target_category}'.")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment OpenApePose species instances with SAM 2.1 using bbox annotations."
    )
    add_common_sam_args(
        parser,
        data_root_help="Path to OpenApePose root containing annotations/ and images/.",
        default_min_area=20000,
        default_fix_inverted=True,
    )
    parser.add_argument(
        "--category",
        type=str,
        default="Chimpanzee",
        help="Target OpenApePose species name (default: Chimpanzee).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_jsons = [
        args.data_root / "annotations" / f"oap_{split}.json"
        for split in ("train", "val", "test")
    ]
    items = load_openapepose_items(
        annotation_files=annotation_jsons,
        image_dir=args.data_root / "images",
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
