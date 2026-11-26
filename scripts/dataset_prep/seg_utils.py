# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.

import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


def resize_and_pad(
    img: Image.Image,
    target_size: int = 512,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Scale longer side to target_size and symmetrically zero-pad to a square."""
    w, h = img.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), resample)

    new_img = Image.new(img.mode, (target_size, target_size), 0)
    new_img.paste(img, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return new_img


def fix_inverted_mask(
    mask_uint8: np.ndarray,
    target_box: Sequence[float],
) -> Tuple[np.ndarray, bool]:
    """Flip SAM masks that segmented the background inside target_box.

    A normal object touches its bounding box only at extreme points, whereas an
    inverted mask activates most of the bounding-box perimeter.
    """
    orig_h, orig_w = mask_uint8.shape[:2]
    tx1, ty1 = max(0, int(target_box[0])), max(0, int(target_box[1]))
    tx2, ty2 = min(orig_w, int(target_box[2])), min(orig_h, int(target_box[3]))

    if tx2 <= tx1 or ty2 <= ty1:
        return mask_uint8, False

    border_pixels = np.concatenate(
        [
            mask_uint8[ty1, tx1:tx2],      # Top edge
            mask_uint8[ty2 - 1, tx1:tx2],  # Bottom edge
            mask_uint8[ty1:ty2, tx1],      # Left edge
            mask_uint8[ty1:ty2, tx2 - 1],  # Right edge
        ]
    )

    if len(border_pixels) > 0 and np.count_nonzero(border_pixels) > 0.5 * len(border_pixels):
        mask_uint8 = mask_uint8.copy()
        mask_box = mask_uint8[ty1:ty2, tx1:tx2]
        mask_uint8[ty1:ty2, tx1:tx2] = 255 - mask_box
        return mask_uint8, True

    return mask_uint8, False


def extract_masked_crop(
    input_image: Image.Image,
    mask_data: Any,
    target_box: Sequence[float],
    *,
    min_area: int = 10000,
    target_size: int = 512,
    margin: float = 0.1,
    morph_kernel_size: int = 7,
    fix_inverted: bool = False,
    log_name: str = "",
) -> Optional[Tuple[Image.Image, Image.Image]]:
    """Post-process a raw SAM mask and return padded (rgb, mask) crops, or None if skipped."""
    orig_w, orig_h = input_image.size

    if hasattr(mask_data, "cpu"):
        m = mask_data.cpu().numpy().astype(np.uint8)
    else:
        m = np.asarray(mask_data, dtype=np.uint8)

    # Normalize binary or 0-255 masks to {0, 1}
    m = (m > 0).astype(np.uint8)

    # Resize if SAM output differs from input resolution
    if m.shape != (orig_h, orig_w):
        m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    mask_uint8 = m * 255

    if fix_inverted:
        mask_uint8, was_inverted = fix_inverted_mask(mask_uint8, target_box)
        if was_inverted:
            print(f"Mask inverted inside bbox in {log_name}, flipping!")

    # Apply morphological closing to fill small gaps/occlusions
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    nonzero_area = int(np.count_nonzero(mask_uint8))
    if nonzero_area < min_area:
        print(f"skipping small mask ({nonzero_area} pixels) in {log_name}")
        return None

    rows = np.any(mask_uint8, axis=1)
    cols = np.any(mask_uint8, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Add margin around the tight mask bounding box
    h_box = ymax - ymin
    w_box = xmax - xmin

    ymin = max(0, int(ymin - h_box * margin))
    ymax = min(orig_h, int(ymax + h_box * margin))
    xmin = max(0, int(xmin - w_box * margin))
    xmax = min(orig_w, int(xmax + w_box * margin))

    if ymax <= ymin or xmax <= xmin:
        return None

    crop_img = input_image.crop((xmin, ymin, xmax, ymax))
    crop_mask = Image.fromarray(mask_uint8[ymin:ymax, xmin:xmax])

    final_img = resize_and_pad(crop_img, target_size, resample=Image.Resampling.LANCZOS)
    final_mask = resize_and_pad(crop_mask, target_size, resample=Image.Resampling.NEAREST)
    return final_img, final_mask


def save_processed_sample(
    output_dir: Path,
    stem: str,
    final_img: Image.Image,
    final_mask: Image.Image,
    *,
    save_masked: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Save processed RGB, mask, optional RGBA masked preview, and YAML metadata."""
    output_dir = Path(output_dir)
    final_img.save(output_dir / "images" / f"{stem}_rgb.png")
    final_mask.save(output_dir / "masks" / f"{stem}_mask.png")

    if save_masked:
        masked_img = final_img.convert("RGBA")
        masked_img.putalpha(final_mask)
        masked_img.save(output_dir / "masked_images" / f"{stem}_masked.png")

    if metadata is not None:
        with open(output_dir / "metadata" / f"{stem}_metadata.txt", "w") as mf:
            yaml.dump(dict(metadata), mf, sort_keys=False)


def add_common_sam_args(
    parser: argparse.ArgumentParser,
    *,
    data_root_help: str,
    default_min_area: int = 10000,
    default_fix_inverted: bool = False,
) -> argparse.ArgumentParser:
    """Register standard I/O and SAM 2.1 arguments for annotated dataset preprocessors."""
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=data_root_help,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for images/, masks/, masked_images/, and metadata/.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=default_min_area,
        help=f"Minimum mask pixel area before cropping (default: {default_min_area}).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=512,
        help="Output square crop resolution (default: 512).",
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="sam2.1_l.pt",
        help="SAM checkpoint name or path (default: sam2.1_l.pt).",
    )
    parser.add_argument(
        "--fix-inverted",
        action=argparse.BooleanOptionalAction,
        default=default_fix_inverted,
        help=f"Detect and flip inverted SAM masks inside the bbox (default: {default_fix_inverted}).",
    )
    parser.add_argument(
        "--no-save-masked",
        action="store_true",
        help="Skip saving RGBA previews to masked_images/.",
    )
    return parser


def _process_bbox_subset(
    items_subset: Sequence[Mapping[str, Any]],
    output_dir: Path,
    device: str,
    *,
    sam_model_name: str = "sam2.1_l.pt",
    min_area: int = 10000,
    target_size: int = 512,
    fix_inverted: bool = False,
    save_masked: bool = True,
    pos: int = 0,
    desc: str = "Processing",
    counter=None,
    lock=None,
) -> None:
    from ultralytics import SAM

    sam_model = SAM(sam_model_name)
    sam_model.to(device)
    output_dir = Path(output_dir)

    for item in tqdm(items_subset, position=pos, desc=desc):
        image_path = Path(item["image_path"])
        target_box = item["bbox"]
        meta_data = item["metadata"]

        if not image_path.exists():
            print(f"Warning: Image {image_path} does not exist.")
            continue

        try:
            with Image.open(image_path) as input_image:
                if input_image.mode != "RGB":
                    input_image = input_image.convert("RGB")

                sam_results = sam_model(
                    input_image, bboxes=[target_box], verbose=False, device=device
                )

                if sam_results[0].masks is None:
                    continue

                mask_data = sam_results[0].masks.data[0]
                result = extract_masked_crop(
                    input_image,
                    mask_data,
                    target_box,
                    min_area=min_area,
                    target_size=target_size,
                    margin=0.1,
                    morph_kernel_size=7,
                    fix_inverted=fix_inverted,
                    log_name=image_path.stem,
                )
                if result is None:
                    continue

                final_img, final_mask = result

                if counter is not None and lock is not None:
                    with lock:
                        current_id = counter.value
                        counter.value += 1
                    stem = f"{current_id:06d}"
                else:
                    stem = str(meta_data.get("original_stem", image_path.stem))

                save_processed_sample(
                    output_dir,
                    stem,
                    final_img,
                    final_mask,
                    save_masked=save_masked,
                    metadata=meta_data,
                )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def _sam_bbox_worker(
    gpu_id: int,
    items_subset: Sequence[Mapping[str, Any]],
    output_dir: Path,
    sam_model_name: str,
    min_area: int,
    target_size: int,
    fix_inverted: bool,
    save_masked: bool,
    counter,
    lock,
) -> None:
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    torch.cuda.set_device(0)
    _process_bbox_subset(
        items_subset,
        output_dir,
        device,
        sam_model_name=sam_model_name,
        min_area=min_area,
        target_size=target_size,
        fix_inverted=fix_inverted,
        save_masked=save_masked,
        pos=gpu_id,
        desc=f"GPU {gpu_id}",
        counter=counter,
        lock=lock,
    )


def run_bbox_sam_pipeline(
    items: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    sam_model_name: str = "sam2.1_l.pt",
    min_area: int = 10000,
    target_size: int = 512,
    fix_inverted: bool = False,
    save_masked: bool = True,
) -> None:
    """Run multi-GPU SAM 2.1 segmentation over normalized bbox annotation records.

    Each element of `items` must be a dict with keys:
      - `image_path`: Path or str to the source RGB image
      - `bbox`: [xmin, ymin, xmax, ymax] box prompt for SAM
      - `metadata`: dict written to `metadata/{stem}_metadata.txt`
    """
    if not items:
        return

    import torch
    import torch.multiprocessing as mp

    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    if save_masked:
        (output_dir / "masked_images").mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs.")
        if num_gpus > 0:
            mp.set_start_method("spawn", force=True)
            counter = mp.Value("i", 0)
            lock = mp.Lock()

            chunk_size = int(np.ceil(len(items) / num_gpus))
            processes = []
            for i in range(num_gpus):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(items))
                subset = items[start:end]
                if not subset:
                    continue

                p = mp.Process(
                    target=_sam_bbox_worker,
                    args=(
                        i,
                        subset,
                        output_dir,
                        sam_model_name,
                        min_area,
                        target_size,
                        fix_inverted,
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
    _process_bbox_subset(
        items,
        output_dir,
        device,
        sam_model_name=sam_model_name,
        min_area=min_area,
        target_size=target_size,
        fix_inverted=fix_inverted,
        save_masked=save_masked,
        counter=counter,
        lock=lock,
    )
