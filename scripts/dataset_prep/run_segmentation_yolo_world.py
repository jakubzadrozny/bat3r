# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from ultralytics import SAM, YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seg_utils import extract_masked_crop, save_processed_sample


def yolo_sam_seg(
    input_image: Image.Image,
    source_file: str,
    yolo_model: YOLO,
    sam_model: SAM,
    output_dir: Path,
    *,
    device: str = "cpu",
    counter=None,
    lock=None,
    dataset_name: str = "image_dataset",
    min_area: int = 20000,
    target_size: int = 512,
    save_masked: bool = True,
) -> None:
    # --- STEP 1: DETECT WITH YOLO-WORLD ---
    yolo_results = yolo_model(input_image, verbose=False, device=device)

    if len(yolo_results[0].boxes) == 0:
        print(f"No objects detected in {source_file}!")
        return

    boxes = yolo_results[0].boxes.xyxy.cpu().numpy().tolist()

    # --- STEP 2: REFINE WITH SAM 2.1 ---
    sam_results = sam_model(input_image, bboxes=boxes, verbose=False, device=device)

    # --- STEP 3: POST-PROCESS AND SAVE ---
    if sam_results[0].masks is None:
        return

    for box_idx, mask_data in enumerate(sam_results[0].masks.data):
        target_box = boxes[box_idx]
        result = extract_masked_crop(
            input_image,
            mask_data,
            target_box,
            min_area=min_area,
            target_size=target_size,
            margin=0.1,
            morph_kernel_size=7,
            fix_inverted=True,
            log_name=source_file,
        )
        if result is None:
            continue

        final_img, final_mask = result

        with lock:
            current_id = counter.value
            counter.value += 1
        stem = f"{current_id:06d}"

        metadata = {
            "dataset": dataset_name,
            "source_file": source_file,
            "box_idx": int(box_idx),
            "bbox": [float(v) for v in target_box],
        }
        save_processed_sample(
            output_dir,
            stem,
            final_img,
            final_mask,
            save_masked=save_masked,
            metadata=metadata,
        )


def process_subset(
    files_subset,
    src_dir: Path,
    output_dir: Path,
    device: str,
    *,
    classes,
    yolo_model_name: str,
    sam_model_name: str,
    dataset_name: str,
    min_area: int,
    target_size: int,
    save_masked: bool,
    counter,
    lock,
    pos: int = 0,
    desc: str = "Processing",
) -> None:
    yolo_model = YOLO(yolo_model_name)
    yolo_model.set_classes(list(classes))
    sam_model = SAM(sam_model_name)

    for file_path in tqdm(files_subset, position=pos, desc=desc):
        source_file = file_path.relative_to(src_dir).as_posix()
        try:
            with Image.open(file_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                yolo_sam_seg(
                    img,
                    source_file,
                    yolo_model,
                    sam_model,
                    output_dir,
                    device=device,
                    counter=counter,
                    lock=lock,
                    dataset_name=dataset_name,
                    min_area=min_area,
                    target_size=target_size,
                    save_masked=save_masked,
                )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def worker(
    gpu_id: int,
    files_subset,
    src_dir: Path,
    output_dir: Path,
    classes,
    yolo_model_name: str,
    sam_model_name: str,
    dataset_name: str,
    min_area: int,
    target_size: int,
    save_masked: bool,
    counter,
    lock,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    torch.cuda.set_device(0)
    process_subset(
        files_subset,
        src_dir,
        output_dir,
        device,
        classes=classes,
        yolo_model_name=yolo_model_name,
        sam_model_name=sam_model_name,
        dataset_name=dataset_name,
        min_area=min_area,
        target_size=target_size,
        save_masked=save_masked,
        counter=counter,
        lock=lock,
        pos=gpu_id,
        desc=f"GPU {gpu_id}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and segment animal crops from <base_path>/raw_imgs using YOLO-World + SAM 2.1."
    )
    parser.add_argument("base_path", type=Path, help="Dataset base directory containing raw_imgs/.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["elephant"],
        help="Open-vocabulary class prompts for YOLO-World (default: elephant).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="image_dataset",
        help="Dataset label written to metadata files (default: image_dataset).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=20000,
        help="Minimum mask pixel area before cropping (default: 20000).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=512,
        help="Output square crop resolution (default: 512).",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8x-worldv2.pt",
        help="YOLO-World checkpoint name or path (default: yolov8x-worldv2.pt).",
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="sam2.1_l.pt",
        help="SAM checkpoint name or path (default: sam2.1_l.pt).",
    )
    parser.add_argument(
        "--no-save-masked",
        action="store_true",
        help="Skip saving RGBA previews to masked_images/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_masked = not args.no_save_masked

    src_dir = args.base_path / "raw_imgs"
    output_dir = args.base_path
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "masks").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)
    if save_masked:
        (output_dir / "masked_images").mkdir(exist_ok=True)

    files_to_process = sorted(
        f
        for f in src_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"Found {len(files_to_process)} files to process in {src_dir}.")

    if not files_to_process:
        return

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs.")
        if num_gpus > 0:
            mp.set_start_method("spawn", force=True)
            counter = mp.Value("i", 0)
            lock = mp.Lock()

            chunk_size = int(np.ceil(len(files_to_process) / num_gpus))
            processes = []

            for i in range(num_gpus):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(files_to_process))
                subset = files_to_process[start:end]
                if not subset:
                    continue

                p = mp.Process(
                    target=worker,
                    args=(
                        i,
                        subset,
                        src_dir,
                        output_dir,
                        args.classes,
                        args.yolo_model,
                        args.sam_model,
                        args.dataset_name,
                        args.min_area,
                        args.target_size,
                        save_masked,
                        counter,
                        lock,
                    ),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
            return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using fallback device: {device}")
    counter = mp.Value("i", 0)
    lock = mp.Lock()
    process_subset(
        files_to_process,
        src_dir,
        output_dir,
        device,
        classes=args.classes,
        yolo_model_name=args.yolo_model,
        sam_model_name=args.sam_model,
        dataset_name=args.dataset_name,
        min_area=args.min_area,
        target_size=args.target_size,
        save_masked=save_masked,
        counter=counter,
        lock=lock,
    )


if __name__ == "__main__":
    main()
