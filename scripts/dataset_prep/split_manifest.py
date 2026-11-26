# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm


def build_raw_imgs_index(raw_imgs_dir: Path) -> List[str]:
    """Reproduce run_segmentation_yolo_world.py's deterministic sorted file index."""
    raw_imgs_dir = Path(raw_imgs_dir)
    if not raw_imgs_dir.is_dir():
        raise FileNotFoundError(f"raw_imgs directory does not exist: {raw_imgs_dir}")
    files = sorted(
        f
        for f in raw_imgs_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return [f.relative_to(raw_imgs_dir).as_posix() for f in files]


def parse_legacy_image_stem(original_stem: str) -> Tuple[int, int]:
    """Parse trailing 6-digit file index and optional box index from a legacy stem.

    Handles both single-detection ('elephant_data_image_000123' -> (123, 0))
    and multi-detection ('elephant_data_image_000123_1' -> (123, 1)) formats.
    """
    parts = original_stem.split("_")
    if len(parts[-1]) == 6 and parts[-1].isdigit():
        return int(parts[-1]), 0
    if (
        len(parts) >= 2
        and len(parts[-2]) == 6
        and parts[-2].isdigit()
        and parts[-1].isdigit()
    ):
        return int(parts[-2]), int(parts[-1])
    raise ValueError(
        f"Could not parse 6-digit file index (and optional box index) from legacy original_stem '{original_stem}'"
    )


def origin_key(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """Return a hashable canonical key uniquely identifying an origin record."""
    source = record["source"]
    if source in ("ap-10k", "openapepose"):
        return (source, str(record["filename"]), int(record["ann_id"]))
    if source == "apt-36k":
        return (source, str(record["subdir"]), str(record["filename"]), str(record["track_id"]))
    if source == "video_track":
        return (source, str(record["clip_id"]), str(record["track_id"]), str(record["frame_idx"]))
    if "source_file" in record:
        return (source, str(record["source_file"]), int(record["box_idx"]))
    raise ValueError(f"Unrecognized origin record schema: {record}")


def extract_origin_record(
    meta: Dict[str, Any],
    raw_imgs_index: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extract a canonical origin record from a metadata dict, failing loudly on missing keys."""
    if not isinstance(meta, dict):
        raise ValueError(f"Expected metadata dict, got {type(meta).__name__}")

    dataset = meta.get("dataset")

    if dataset in ("ap-10k", "openapepose"):
        return {
            "source": str(dataset),
            "filename": str(meta["filename"]),
            "ann_id": int(meta["ann_id"]),
        }

    if dataset == "apt-36k":
        return {
            "source": "apt-36k",
            "subdir": str(meta["subdir"]),
            "filename": str(meta["filename"]),
            "track_id": str(meta["track_id"]),
        }

    if "clip_id" in meta:
        return {
            "source": "video_track",
            "clip_id": str(meta["clip_id"]),
            "track_id": str(meta["track_id"]),
            "frame_idx": str(meta["frame_idx"]),
            "video_frame_id": int(meta["video_frame_id"]),
            "video_frame_time": float(meta["video_frame_time"]),
            "crop_box_xyxy": [int(v) for v in meta["crop_box_xyxy"]],
        }

    if "source_file" in meta:
        return {
            "source": str(meta["dataset"]),
            "source_file": str(meta["source_file"]),
            "box_idx": int(meta["box_idx"]),
        }

    if dataset == "image_dataset":
        original_stem = str(meta["original_stem"])
        if raw_imgs_index is None:
            raise ValueError(
                f"Legacy image_dataset entry (original_stem='{original_stem}') lacks 'source_file' "
                f"and requires --raw-imgs-dir."
            )
        file_idx, box_idx = parse_legacy_image_stem(original_stem)
        if file_idx >= len(raw_imgs_index):
            raise IndexError(
                f"Legacy file index {file_idx} from '{original_stem}' out of range "
                f"(raw_imgs has {len(raw_imgs_index)} files)"
            )
        return {
            "source": "image_dataset",
            "source_file": raw_imgs_index[file_idx],
            "box_idx": box_idx,
        }

    raise ValueError(f"Unsupported metadata schema: keys={sorted(meta.keys())}")


def _check_duplicates(
    occurrences: Dict[Tuple[Any, ...], List[Tuple[str, Any]]],
    context: str,
) -> None:
    """Raise ValueError listing all origin keys with more than one occurrence."""
    duplicates = [(key, occs) for key, occs in occurrences.items() if len(occs) > 1]
    if not duplicates:
        return

    lines = [f"Found {len(duplicates)} duplicate origin key(s) in {context}:"]
    for key, occs in duplicates[:20]:
        lines.append(f"  {key} -> {occs}")
    if len(duplicates) > 20:
        lines.append(f"  ... and {len(duplicates) - 20} more duplicate(s)")

    raise ValueError("\n".join(lines))


def _confirm_overwrite(paths: List[Path], overwrite: bool) -> None:
    """Prompt before overwriting existing output file(s) unless overwrite=True."""
    existing = [p for p in paths if p.exists()]
    if not existing or overwrite:
        return
    names = ", ".join(str(p) for p in existing)
    try:
        resp = input(f"Existing output file(s) found: {names}. Overwrite? [y/N]: ").strip().lower()
    except EOFError:
        resp = ""
    if resp not in ("y", "yes"):
        print("Aborted — existing files were not modified.")
        sys.exit(1)


def export_manifest(
    dataset_dir: Path,
    out_path: Path,
    raw_imgs_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    out_path = Path(out_path)
    metadata_dir = dataset_dir / "metadata"
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    _confirm_overwrite([out_path], overwrite=overwrite)

    raw_imgs_index = build_raw_imgs_index(raw_imgs_dir) if raw_imgs_dir is not None else None
    if raw_imgs_index is not None:
        print(f"Indexed {len(raw_imgs_index)} raw images from {raw_imgs_dir} for legacy stem recovery.")

    split_files = [
        ("train", dataset_dir / "train_ids.json"),
        ("test", dataset_dir / "test_ids.json"),
    ]
    val_file = dataset_dir / "val_ids.json"
    if val_file.exists():
        split_files.append(("val", val_file))

    splits: Dict[str, List[Dict[str, Any]]] = {}
    key_occurrences: Dict[Tuple[Any, ...], List[Tuple[str, str]]] = defaultdict(list)

    for split_name, split_file in split_files:
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file, "r") as f:
            ids = json.load(f)

        records: List[Dict[str, Any]] = []
        for sample_id in tqdm(ids, desc=f"Exporting {split_name}"):
            meta_path = metadata_dir / f"{sample_id}_metadata.txt"
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing metadata file for ID '{sample_id}': {meta_path}")
            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f)

            record = extract_origin_record(meta, raw_imgs_index=raw_imgs_index)
            key = origin_key(record)
            key_occurrences[key].append((split_name, str(sample_id)))
            records.append(record)

        splits[split_name] = records

    _check_duplicates(key_occurrences, context=f"dataset '{dataset_dir}'")

    manifest = {
        "format_version": 1,
        "counts": {k: len(v) for k, v in splits.items()},
        "splits": splits,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total = sum(manifest["counts"].values())
    split_summary = ", ".join(f"{k}={v}" for k, v in manifest["counts"].items())
    print(f"Exported split manifest to {out_path} (total={total}: {split_summary}).")

    source_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for split_name, records in splits.items():
        for rec in records:
            source_counts[rec["source"]][split_name] += 1

    print("Origin dataset summary:")
    for source_name in sorted(source_counts):
        per_split = source_counts[source_name]
        src_total = sum(per_split.values())
        breakdown = ", ".join(f"{sp}={per_split.get(sp, 0)}" for sp in splits)
        print(f"  {source_name}: {src_total} ({breakdown})")

    return manifest


def apply_manifest(
    dataset_dir: Path,
    manifest_path: Path,
    overwrite: bool = False,
) -> Dict[str, List[str]]:
    dataset_dir = Path(dataset_dir)
    metadata_dir = dataset_dir / "metadata"
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    manifest_splits: Dict[str, List[Dict[str, Any]]] = manifest["splits"]
    exact_lookup: Dict[Tuple[Any, ...], str] = {}
    manifest_occurrences: Dict[Tuple[Any, ...], List[Tuple[str, int]]] = defaultdict(list)

    for split_name, records in manifest_splits.items():
        for rec_idx, rec in enumerate(records):
            key = origin_key(rec)
            manifest_occurrences[key].append((split_name, rec_idx))
            exact_lookup[key] = split_name

    _check_duplicates(manifest_occurrences, context=f"manifest '{manifest_path}'")

    _confirm_overwrite(
        [dataset_dir / f"{split_name}_ids.json" for split_name in manifest_splits],
        overwrite=overwrite,
    )

    assigned: Dict[str, List[str]] = {split_name: [] for split_name in manifest_splits}
    matched_keys = set()
    unselected_ids: List[str] = []
    dataset_occurrences: Dict[Tuple[Any, ...], List[Tuple[str, str]]] = defaultdict(list)

    meta_files = sorted(metadata_dir.glob("*_metadata.txt"))
    for meta_file in tqdm(meta_files, desc="Applying manifest"):
        sample_id = meta_file.name.removesuffix("_metadata.txt")
        with open(meta_file, "r") as f:
            meta = yaml.safe_load(f)

        record = extract_origin_record(meta)
        key = origin_key(record)
        split_name = exact_lookup.get(key)
        dataset_occurrences[key].append((split_name or "unselected", sample_id))

        if split_name is not None:
            assigned[split_name].append(sample_id)
            matched_keys.add(key)
        else:
            unselected_ids.append(sample_id)

    _check_duplicates(dataset_occurrences, context=f"target dataset '{dataset_dir}'")

    for split_name, ids in assigned.items():
        ids.sort()
        with open(dataset_dir / f"{split_name}_ids.json", "w") as f:
            json.dump(ids, f, indent=2)

    missing_manifest_entries = len(exact_lookup) - len(matched_keys)
    counts_str = ", ".join(f"{k}={len(v)}" for k, v in assigned.items())
    print(f"Applied manifest {manifest_path} to {dataset_dir}: {counts_str}")
    if unselected_ids:
        print(
            f"  Unselected samples in dataset (not in manifest, e.g. extra frames): {len(unselected_ids)}"
        )
    if missing_manifest_entries > 0:
        print(
            f"  Manifest entries not found in dataset: {missing_manifest_entries}"
        )

    return assigned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and apply permutation-invariant origin-based train/test split manifests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export",
        help="Export a split manifest JSON from an existing dataset's metadata/ and split JSONs.",
    )
    export_parser.add_argument("--dataset-dir", type=Path, required=True)
    export_parser.add_argument("--out", type=Path, required=True, help="Output manifest JSON path.")
    export_parser.add_argument(
        "--raw-imgs-dir",
        type=Path,
        default=None,
        help="raw_imgs/ directory (required when exporting legacy image_dataset entries without source_file).",
    )
    export_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output manifest JSON without prompting.",
    )

    apply_parser = subparsers.add_parser(
        "apply",
        help="Reconstruct train_ids.json / test_ids.json on a recreated dataset using a split manifest.",
    )
    apply_parser.add_argument("--dataset-dir", type=Path, required=True)
    apply_parser.add_argument("--manifest", type=Path, required=True, help="Input manifest JSON path.")
    apply_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split JSON files without prompting.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "export":
        export_manifest(
            dataset_dir=args.dataset_dir,
            out_path=args.out,
            raw_imgs_dir=args.raw_imgs_dir,
            overwrite=args.overwrite,
        )
    elif args.command == "apply":
        apply_manifest(
            dataset_dir=args.dataset_dir,
            manifest_path=args.manifest,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
