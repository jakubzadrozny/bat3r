# Dataset Preparation

This guide explains how to prepare real-world 2D image collections, 3D canonical shapes, geodesic distance matrices, and SD-DINO features for **BAT3R** — both for reproducing our three paper categories (**Horse**, **Chimp**, **Elephant**) and for onboarding a **new custom category**.

## Prerequisites

Install the optional dataset-preprocessing dependency (`ultralytics==8.4.21`) into your `bat3r` environment:

```bash
pip install -e ".[dataset-prep]"
```

For SD-DINO feature extraction (`dualpm_features/`), set up its separate Python 3.9 environment following [`dualpm_features/README.md`](../../dualpm_features/README.md).

---

## 1. Expected Dataset Directory Layout

Training and evaluation launchers (`scripts/launcher/`) expect each category dataset directory (`DATASET_PATH`) to follow:

```text
<category>_data/
├── images/
│   ├── 000000_rgb.png        # (.jpg for MagicPony horse)
│   └── ...
├── masks/
│   ├── 000000_mask.png
│   └── ...
├── features/                 # SD-DINO + PCA features (extracted via dualpm_features/)
│   ├── 000000_feat.png
│   └── ...
├── metadata/                 # Per-sample origin YAML (used by split_manifest.py / track splits)
│   ├── 000000_metadata.txt
│   └── ...
├── train_ids.json
├── test_ids.json
└── shapes/
    └── <shape_id>/           # e.g. horse_male-rd, chimp, elephant
        ├── <shape_id>_shape.gltf
        ├── <shape_id>_shape.bin
        └── <shape_id>_geo_dists.npz
```

> **Assets bundle (`jzadrozny/bat3r`)**: Download our released checkpoints, category PCA `.faiss` files, split manifests, and synthesized horse textures into `assets/` from Hugging Face Hub:
> ```bash
> hf download jzadrozny/bat3r --local-dir assets
> ```

---

## 2. Reproducing Paper Categories

### Horse
We use [MagicPony](https://3dmagicpony.github.io/)'s `horse_combined.zip` directly (no re-segmentation or split manifest needed - `organize_magicpony_data.py` reproduces the exact IDs and `train`/`test` splits):

1. Download and unzip [`horse_combined.zip`](https://download.cs.stanford.edu/viscam/3DAnimals/data/magicpony/horse_combined.zip).
2. Flatten into BAT3R format:
   ```bash
   python scripts/dataset_prep/sources/organize_magicpony_data.py \
     --src-dir /path/to/horse_combined \
     --tgt-dir /path/to/horse_data
   ```

### Chimp
Chimp combines three public datasets: **[AP-10K](https://github.com/AlexTheBad/AP-10K)**, **[APT-36K](https://github.com/pandorgan/APT-36K)** (`7chimpanzee` subset), and **[OpenApePose](https://github.com/desai-nisarg/OpenApePose)**.

1. Segment each source with SAM 2.1 using its bounding-box annotations:
   ```bash
   python scripts/dataset_prep/sources/preprocess_ap10k.py \
     --data-root /path/to/ap-10k \
     --output-dir /path/to/ap10k_ready

   python scripts/dataset_prep/sources/preprocess_apt36k.py \
     --data-root /path/to/AP-36k/7chimpanzee \
     --output-dir /path/to/apt36k_ready

   python scripts/dataset_prep/sources/preprocess_openapepose.py \
     --data-root /path/to/OpenApePose \
     --output-dir /path/to/openapepose_ready
   ```
2. Merge all three sources into one sequential `{idx:06d}` dataset:
   ```bash
   python scripts/dataset_prep/merge_datasets.py \
     --input-dirs /path/to/ap10k_ready /path/to/apt36k_ready /path/to/openapepose_ready \
     --output-dir /path/to/chimp_combined
   ```
   *(Warnings about missing splits for `apt36k_ready` can be ignored if you apply our split manifest in Step 3; otherwise run `create_splits.py --mode track` on `apt36k_ready` first.)*
3. Reconstruct our exact train/test split (invariant to `{idx:06d}` numbering) using `chimp_split_manifest.json` from `jzadrozny/bat3r`:
   ```bash
   python scripts/dataset_prep/split_manifest.py apply \
     --dataset-dir /path/to/chimp_combined \
     --manifest assets/split_manifests/chimp_split_manifest.json
   ```

### Elephant
Elephant combines still images from Kaggle with tracked video frames from **[Animal-in-Motion](https://github.com/briannlongzhao/Animal-in-Motion)**.

1. **Image subset**:
   - Download the [Sri Lankan Wild Elephant Dataset](https://www.kaggle.com/datasets/gunarakulangr/sri-lankan-wild-elephant-dataset) from Kaggle (unannotated `.jpg`/`.png` frames). Place the raw image files directly inside `/path/to/elephant_image/raw_imgs/` (keeping them flat inside `raw_imgs/` ensures filenames match `elephant_split_manifest.json`; `run_segmentation_yolo_world.py` also supports nested folders recursively for custom runs).
   - Run YOLO-World + SAM 2.1 segmentation:
     ```bash
     python scripts/dataset_prep/run_segmentation_yolo_world.py \
       /path/to/elephant_image \
       --classes elephant
     ```
2. **Video subset**:
   - Download and track the **171 YouTube videos** listed in [`elephant_youtube_ids.txt`](./elephant_youtube_ids.txt) using [Animal-in-Motion](https://github.com/briannlongzhao/Animal-in-Motion) (AiM's `YoutubeDownloader.download(video_id_to_query)` can download a list of video IDs directly, bypassing search scraping).
   - Flatten the resulting `<clip_id>/<track_id>/` tracks (subsampling every 5th frame):
     ```bash
     python scripts/dataset_prep/convert_track_dataset.py \
       --input-dirs /path/to/aim_tracks/elephant \
       --output-dir /path/to/elephant_video \
       --frame-step 5
     ```
3. **Merge & apply split manifest**:
   ```bash
   python scripts/dataset_prep/merge_datasets.py \
     --input-dirs /path/to/elephant_video /path/to/elephant_image \
     --output-dir /path/to/elephant_data

   python scripts/dataset_prep/split_manifest.py apply \
     --dataset-dir /path/to/elephant_data \
     --manifest assets/split_manifests/elephant_split_manifest.json
   ```
   *(If you are generating fresh splits instead of applying `elephant_split_manifest.json`, run `create_splits.py --mode track` on `elephant_video` and `create_splits.py --mode frame` on `elephant_image`, then pass `--steps 1 2` to `merge_datasets.py` to subsample the image set by 2× as in the paper.)*

---

## 3. Preparing a New Custom Category

1. **From raw images**: place raw photos under `/path/to/<cat>_data/raw_imgs/`, run open-vocabulary segmentation, and create a frame-level split:
   ```bash
   python scripts/dataset_prep/run_segmentation_yolo_world.py \
     /path/to/<cat>_data \
     --classes "<animal prompt>"

   python scripts/dataset_prep/create_splits.py \
     --dataset-dir /path/to/<cat>_data \
     --mode frame
   ```
2. **From video tracks**: run [Animal-in-Motion](https://github.com/briannlongzhao/Animal-in-Motion), flatten tracks with `convert_track_dataset.py`, and split by track so frames from the same track don't leak across train and test:
   ```bash
   python scripts/dataset_prep/convert_track_dataset.py \
     --input-dirs /path/to/aim_tracks/<cat> \
     --output-dir /path/to/<cat>_video \
     --frame-step 5

   python scripts/dataset_prep/create_splits.py \
     --dataset-dir /path/to/<cat>_video \
     --mode track
   ```
3. **Optional merge**: combine multiple prepared sources with `merge_datasets.py --input-dirs ... --output-dir ...`.
4. **Category config & Blender rig**: add a category YAML under `configs/category/<cat>.yaml` (following `horse.yaml`, `chimp.yaml`, or `elephant.yaml`) and run `scripts/rendering/prepare_blend.py` (see Section 4) to build `<cat>.blend` and canonical `<shape_id>_shape.gltf` from your rigged `.glb`/`.gltf` mesh.

---

## 4. 3D Shapes, Blender Scenes, Geodesic Distances & SD-DINO Features

### Canonical 3D Shapes (`.gltf`) & Blender Scenes (`.blend`)
We do not redistribute third-party 3D meshes. Download each rigged model from its original upstream source, then run our headless **Blender 4.5** scene builder (`scripts/rendering/prepare_blend.py`), which constructs the exact BAT3R Cycles scene (HDRI world shader, `Standard` / `Medium Contrast` color management, `IndexOB` mask compositor, light, camera), cleans/joins sub-meshes per `configs/category/<cat>.yaml`, writes `assets/<cat>.blend`, and optionally exports canonical `<shape_id>_shape.gltf` (+ `.bin`) into `<dataset>/shapes/`:

- **Chimp** — download [Chimp Rig on Sketchfab](https://sketchfab.com/3d-models/chimp-rig-9fd4896dc1fd4ff69ce855ba3d47298f) (`.glb`):
  ```bash
  blender --background --python scripts/rendering/prepare_blend.py -- \
    --category-config configs/category/chimp.yaml \
    --input-mesh /path/to/chimp_rig.glb \
    --out-blend assets/chimp.blend \
    --out-shapes-dir /path/to/chimp_combined/shapes
  ```
- **Elephant** — download [African Elephant on Sketchfab](https://sketchfab.com/3d-models/female-african-elephant-planet-zoo-fe5fcad06cad4cf79aca431b937510a2) (`.glb`; `elephant.yaml` automatically drops the `African_Elephant_Female_Hair` sub-mesh and joins `Skin` + `Eyes`):
  ```bash
  blender --background --python scripts/rendering/prepare_blend.py -- \
    --category-config configs/category/elephant.yaml \
    --input-mesh /path/to/female_african_elephant_planet_zoo.glb \
    --out-blend assets/elephant.blend \
    --out-shapes-dir /path/to/elephant_data/shapes
  ```
- **Horse** — download `horse_male-rd` and `horse_female-rd` `.gltf` shapes from the [DualPM](https://dualpm.github.io/) dataset (textures in `assets/textures/horse_{male,female}-rd/` from `jzadrozny/bat3r` are already vertically pre-oriented for DualPM's horse UVs):
  ```bash
  blender --background --python scripts/rendering/prepare_blend.py -- \
    --category-config configs/category/horse.yaml \
    --model horse_male-rd=/path/to/horse_male-rd_shape.gltf \
    --model horse_female-rd=/path/to/horse_female-rd_shape.gltf \
    --out-blend assets/horse.blend \
    --out-shapes-dir /path/to/horse_data/shapes
  ```
  *(For Horse, `--out-shapes-dir` is optional if DualPM's `horse_{male,female}-rd_shape.gltf` (+ `.bin`) files are already placed in `/path/to/horse_data/shapes/`.)*
- **Custom categories**: pass `--category-config configs/category/<cat>.yaml` with `--input-mesh` (single model) or repeatable `--model NAME=PATH` (multi-model), plus optional `--drop-submeshes <obj_or_mat_name...>` to discard unwanted sub-meshes before joining.

### Environment HDRIs (`assets/hdrs/`)
For synthetic data rendering (`scripts/rendering/`), download the CC0 Poly Haven HDRI set into `assets/hdrs/`:
```bash
python scripts/rendering/download_hdrs.py --out-dir assets/hdrs
```
*(When running `scripts/rendering/run_blender_runner.sh`, pass `--device METAL`, `--device CUDA`, or `--device CPU` after `ASSETS_DIR` if you are not on an NVIDIA OptiX GPU.)*

**Helpful tools for custom/unprepared meshes:**
- **Missing textures** (e.g. untextured meshes): [Text2Tex](https://github.com/daveredrum/Text2Tex) works well for generating plausible textures for synthetic rendering.
- **Un-rigged meshes**: [RigAnything](https://github.com/Isabella98Liu/RigAnything) can automatically generate a usable skeleton rig.

### Precomputing Geodesic Distances (`_geo_dists.npz`)
Pose fitting (`scripts/gen_train_set.py` and `scripts/infer.py`) requires a precomputed geodesic distance matrix `<shape_id>_geo_dists.npz` alongside each `<shape_id>_shape.gltf`:

```bash
# Paper proximity thresholds: 0.07 for horse; 0.05 for chimp & elephant
python scripts/dataset_prep/compute_geo_dists.py \
  --shapes-dir /path/to/<cat>_data/shapes \
  --proximity-threshold 0.05
```

Verify that `disconnected_pairs=0` in the output (if non-zero, something weird is going on - your mesh has disconnected components even after proximity patching! inspect or try increasing `--proximity-threshold` so separate sub-meshes or UV seams are bridged).

### SD-DINO Features & PCA (`dualpm_features/`)
Activate the separate `dualpm_features` Python 3.9 environment before running feature extraction:

- **Paper categories** (using our pre-trained `<category>.faiss` PCA from `jzadrozny/bat3r`; note `assets/pcas/horse.faiss` is identical to `dualpm_features/libs/horse_male_pca.faiss`):
  ```bash
  python dualpm_features/scripts/extract_distrib.py \
    configs=dualpm_features/configs/extract_feats.yml \
    args.test_root="[/path/to/<cat>_data]" \
    args.load_pca_path=assets/pcas/<cat>.faiss
  ```
  *(For **Horse**, also pass `args.img_postfix_test="[_rgb.jpg]"` since MagicPony RGB images are `.jpg`.)*

- **New custom categories**: train a category PCA `.faiss` on your real images mixed with canonical renders *(recommended - render canonical views first via `gen_canonical_dataset.py` + `run_blender_runner.sh`, or fall back to pure real images with `args.train_root="[/path/to/<cat>_data]"`)*, then extract features with `extract_distrib.py`:
  ```bash
  python -m dualpm_features.train_pca \
    configs=dualpm_features/configs/train_pca.yml \
    args.train_root="[/path/to/<cat>_canon_renders,/path/to/<cat>_data]" \
    args.results_info_root=/path/to/pcas \
    args.pca_filename=<cat>.faiss
  ```
