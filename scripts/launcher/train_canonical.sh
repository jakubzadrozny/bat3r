#!/usr/bin/env bash
# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.
#
# Stage 1: Canonical pretraining launcher (1-shot).
# Usage:
#   bash scripts/launcher/train_canonical.sh <config.env> [KEY=VAL ...]

set -euo pipefail
trap 'echo "❌ Interrupted"; exit 130' INT
trap 'echo "❌ Error at line $LINENO"; exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/utils.sh"

# ------------------
# CONFIG
# ------------------
load_launcher_config "$@"
CONFIG_FILE="$(to_abs_path "$1")"

: "${CATEGORY:?}"
: "${CANON_EXP_NAME:?}"
: "${DATASET_PATH:?}"
: "${MODELS_ROOT:?}"
: "${PCA_PATH:?}"
: "${RENDER_ASSETS_DIR:?}"

EXP_NAME="${CANON_EXP_NAME}"

DATASET_PATH="$(to_abs_path "$DATASET_PATH")"
MODELS_ROOT="$(to_abs_path "$MODELS_ROOT")"
PCA_PATH="$(to_abs_path "$PCA_PATH")"
RENDER_ASSETS_DIR="$(to_abs_path "$RENDER_ASSETS_DIR")"
if [[ -n "$EVAL_DATASET_PATH" ]]; then
    EVAL_DATASET_PATH="$(to_abs_path "$EVAL_DATASET_PATH")"
fi
if [[ -n "$ACCELERATE_CONFIG" ]]; then
    ACCELERATE_CONFIG="$(to_abs_path "$ACCELERATE_CONFIG")"
fi

EXP_ROOT="${MODELS_ROOT}/${EXP_NAME}"

# ------------------
# KERBEROS GUARD
# ------------------
setup_kerberos_guard

# ------------------
# INIT
# ------------------
init_conda
n_gpus=$(get_gpu_count "$ENV_BAT3R") || exit 1

echo "Category: $CATEGORY"
echo "Experiment: $EXP_NAME ($EXP_ROOT)"
echo "Using $n_gpus GPUs"

mkdir -p "$EXP_ROOT/${EXP_NAME}_0"
cp "$CONFIG_FILE" "$EXP_ROOT/run.env"
cp "$REPO_ROOT/configs/main.yaml" "$EXP_ROOT/"
cp "$REPO_ROOT/configs/infer.yaml" "$EXP_ROOT/"
cp "$REPO_ROOT/configs/real.yaml" "$EXP_ROOT/"
cp "$REPO_ROOT/configs/category/${CATEGORY}.yaml" "$EXP_ROOT/"

OLD_MODEL_PATH="$EXP_ROOT/${EXP_NAME}_0"
NEW_MODEL_PATH="$EXP_ROOT/${EXP_NAME}_1"
DATASET_PATH_ABS="$OLD_MODEL_PATH/output"

# ------------------
# PIPELINE
# ------------------
echo
echo "========== CANONICAL PRETRAINING =========="
date

if [[ $START_STEP -le 1 ]]; then
    # Step 1: generate canonical poses and camera views
    echo "Canonical, step 1"
    date
    safe_conda_activate "$ENV_BAT3R"
    cd "$REPO_ROOT"
    check_kerberos
    python scripts/gen_canonical_dataset.py --config-name real \
        category="${CATEGORY}" \
        log_dir="${OLD_MODEL_PATH}" \
        dataset_root="${DATASET_PATH}"
    rm -rf "$DATASET_PATH_ABS/shapes"
    ln -sfn "$DATASET_PATH/shapes" "$DATASET_PATH_ABS/shapes"
    safe_conda_deactivate
fi

if [[ $START_STEP -le 2 ]]; then
    # Step 2: Blender rendering
    echo "Canonical, step 2"
    date
    cd "$REPO_ROOT"
    check_kerberos
    bash "$REPO_ROOT/scripts/rendering/run_blender_runner.sh" \
        "$n_gpus" "$DATASET_PATH_ABS" "$CATEGORY" "$RENDER_ASSETS_DIR" || {
        echo "❌ Blender failed during canonical rendering"
        exit 1
    }
fi

if [[ $START_STEP -le 3 ]]; then
    # Step 3: extract SD-DINO features
    echo "Canonical, step 3"
    date
    safe_conda_activate "$ENV_SDDINO"
    cd "$REPO_ROOT/dualpm_features"
    check_kerberos
    python scripts/extract_distrib.py \
        configs=configs/extract_feats.yml \
        "args.test_root=[${DATASET_PATH_ABS}]" \
        "args.features_out_root=${DATASET_PATH_ABS}" \
        "args.load_pca_path=${PCA_PATH}"
    cd "$REPO_ROOT"
    safe_conda_deactivate
fi

if [[ $START_STEP -le 4 ]]; then
    # Step 4: train BAT3R on canonical renders
    echo "Canonical, step 4"
    date
    safe_conda_activate "$ENV_BAT3R"
    cd "$REPO_ROOT"
    check_kerberos

    ACCEL_ARGS=(--num_processes="${n_gpus}")
    if [[ -n "$ACCELERATE_CONFIG" ]]; then
        ACCEL_ARGS+=(--config_file "$ACCELERATE_CONFIG")
    fi

    VAL_DATASET="${EVAL_DATASET_PATH:-$DATASET_PATH_ABS}"
    accelerate launch "${ACCEL_ARGS[@]}" \
        scripts/train_distrib.py --config-name main \
        category="${CATEGORY}" \
        use_train_test_split=false \
        dataset_root="${DATASET_PATH_ABS}" \
        val_dataset_root="${VAL_DATASET}" \
        train_config.save_path="${NEW_MODEL_PATH}"
    safe_conda_deactivate
fi

if [[ $START_STEP -le 5 ]]; then
    # Step 5: run inference to get metrics
    echo "Canonical, step 5"
    date
    if [[ -n "$EVAL_DATASET_PATH" ]]; then
        safe_conda_activate "$ENV_BAT3R"
        cd "$REPO_ROOT"
        check_kerberos
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/infer.py --config-name infer \
            category="${CATEGORY}" \
            split="${EVAL_SPLIT}" \
            log_dir="${NEW_MODEL_PATH}" \
            dataset_root="${EVAL_DATASET_PATH}"
        safe_conda_deactivate
    else
        echo "Skipping Step 5 evaluation (EVAL_DATASET_PATH not set; canonical datasets have no train/test split JSON)."
    fi
fi

echo
echo "Canonical pretraining completed successfully."
