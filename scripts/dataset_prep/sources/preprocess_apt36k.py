# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seg_utils import add_common_sam_args, run_bbox_sam_pipeline


def get_bboxes_from_json(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    instances = {}
    for shape in data.get("shapes", []):
        if shape.get("shape_type") == "rectangle":
            track_id = shape["label"]
            points = shape["points"]
            xmin, ymin = points[0]
            xmax, ymax = points[1]
            instances[track_id] = [xmin, ymin, xmax, ymax]

    return instances


def load_apt36k_items(root_dir: Path):
    root_dir = Path(root_dir)
    items = []

    for subdir in sorted(root_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for image_path in sorted(subdir.glob("*.jpg")):
            json_path = image_path.with_suffix(".json")
            if not json_path.exists():
                continue
            bboxes = get_bboxes_from_json(json_path)
            for track_id, target_box in bboxes.items():
                items.append(
                    {
                        "image_path": image_path,
                        "bbox": target_box,
                        "metadata": {
                            "dataset": "apt-36k",
                            "subdir": subdir.name,
                            "filename": f"{image_path.stem}.jpg",
                            "track_id": track_id,
                            "bbox": target_box,
                            "original_stem": f"{image_path.stem}_{track_id}",
                        },
                    }
                )

    print(f"Found {len(items)} bboxes.")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment APT-36K video-clip instances with SAM 2.1 using Labelme bbox annotations."
    )
    add_common_sam_args(
        parser,
        data_root_help="Path to APT-36K category directory containing per-clip subfolders (e.g. .../7chimpanzee).",
        default_min_area=10000,
        default_fix_inverted=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_apt36k_items(args.data_root)
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
