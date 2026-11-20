#!/usr/bin/env bash
# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.
#
# Stage 2: Iterative bootstrapping launcher (iterations 1..K).
# Usage:
#   bash scripts/launcher/train_bootstrap.sh <config.env> [KEY=VAL ...]

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
: "${BOOTSTRAP_EXP_NAME:?}"
: "${DATASET_PATH:?}"
: "${MODELS_ROOT:?}"
: "${PCA_PATH:?}"
: "${RENDER_ASSETS_DIR:?}"

EXP_NAME="${BOOTSTRAP_EXP_NAME}"
INITIAL_MODEL="${INITIAL_MODEL:-}"

DATASET_PATH="$(to_abs_path "$DATASET_PATH")"
MODELS_ROOT="$(to_abs_path "$MODELS_ROOT")"
PCA_PATH="$(to_abs_path "$PCA_PATH")"
RENDER_ASSETS_DIR="$(to_abs_path "$RENDER_ASSETS_DIR")"

if [[ -z "$INITIAL_MODEL" ]]; then
    : "${CANON_EXP_NAME:?Set either CANON_EXP_NAME or INITIAL_MODEL}"
    INITIAL_MODEL="${MODELS_ROOT}/${CANON_EXP_NAME}/${CANON_EXP_NAME}_1/${CANON_CKPT}"
elif [[ "$INITIAL_MODEL" != /* ]]; then
    INITIAL_MODEL="${MODELS_ROOT}/${INITIAL_MODEL}"
fi
if [[ ! -f "$INITIAL_MODEL" ]]; then
    echo "❌ Error: INITIAL_MODEL not found: $INITIAL_MODEL" >&2
    exit 1
fi
INITIAL_MODEL_CKPT="$(basename "$INITIAL_MODEL")"

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
echo "Initial model: $INITIAL_MODEL"
echo "Using $n_gpus GPUs"
echo "Running $K iterations (starting at iter=$START_ITER, step=$START_STEP)"

mkdir -p "$EXP_ROOT/${EXP_NAME}_0"
ln -sf "$INITIAL_MODEL" "$EXP_ROOT/${EXP_NAME}_0/"
cp "$CONFIG_FILE" "$EXP_ROOT/run.env"
cp "$REPO_ROOT/configs/main.yaml" "$EXP_ROOT/"
cp "$REPO_ROOT/configs/infer.yaml" "$EXP_ROOT/"
cp "$REPO_ROOT/configs/real.yaml" "$EXP_ROOT/"
cp "$REPO_ROOT/configs/category/${CATEGORY}.yaml" "$EXP_ROOT/"

# ------------------
# LOOP
# ------------------
for ((i=START_ITER; i<=K; i++)); do
    echo
    echo "========== ITERATION $i / $K =========="
    date

    OLD_MODEL_PATH="$EXP_ROOT/${EXP_NAME}_$((i-1))"
    NEW_MODEL_PATH="$EXP_ROOT/${EXP_NAME}_${i}"
    DATASET_PATH_ABS="$OLD_MODEL_PATH/output"

    if [[ $i -gt $START_ITER || $START_STEP -le 1 ]]; then
        # Step 1: inference with previous model on real-world training images
        echo "Iter ${i}, step 1"
        date
        safe_conda_activate "$ENV_BAT3R"
        cd "$REPO_ROOT"
        check_kerberos
        CKPT_ARGS=()
        if [[ $i -eq 1 ]]; then
            CKPT_ARGS+=("ckpt_name=${INITIAL_MODEL_CKPT}")
        fi
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/gen_train_set.py --config-name real \
            category="${CATEGORY}" \
            log_dir="${OLD_MODEL_PATH}" \
            dataset_root="${DATASET_PATH}" \
            +refinement_iter="$i" \
            ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}
        rm -rf "$DATASET_PATH_ABS/shapes"
        ln -sfn "$DATASET_PATH/shapes" "$DATASET_PATH_ABS/shapes"
        ln -sf "$DATASET_PATH/train_ids.json" "$DATASET_PATH_ABS/"
        if [[ -f "$DATASET_PATH/test_ids.json" ]]; then
            ln -sf "$DATASET_PATH/test_ids.json" "$DATASET_PATH_ABS/"
        fi
        safe_conda_deactivate
    fi

    if [[ $i -gt $START_ITER || $START_STEP -le 2 ]]; then
        # Step 2: Blender rendering
        echo "Iter ${i}, step 2"
        date
        cd "$REPO_ROOT"
        check_kerberos
        bash "$REPO_ROOT/scripts/rendering/run_blender_runner.sh" \
            "$n_gpus" "$DATASET_PATH_ABS" "$CATEGORY" "$RENDER_ASSETS_DIR" || {
            echo "❌ Blender failed on iteration $i"
            exit 1
        }
    fi

    if [[ $i -gt $START_ITER || $START_STEP -le 3 ]]; then
        # Step 3: extract SD-DINO features
        echo "Iter ${i}, step 3"
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

    if [[ $i -gt $START_ITER || $START_STEP -le 4 ]]; then
        # Step 4: train BAT3R on newly rendered dataset
        echo "Iter ${i}, step 4"
        date
        safe_conda_activate "$ENV_BAT3R"
        cd "$REPO_ROOT"
        check_kerberos

        ACCEL_ARGS=(--num_processes="${n_gpus}")
        if [[ -n "$ACCELERATE_CONFIG" ]]; then
            ACCEL_ARGS+=(--config_file "$ACCELERATE_CONFIG")
        fi

        SECONDARY_ARGS=()
        if [[ "$USE_PREV_DATASET" == "1" && $i -gt 1 ]]; then
            PREV_DATASET="$EXP_ROOT/${EXP_NAME}_$((i-2))/output"
            if [[ -d "$PREV_DATASET" ]]; then
                SECONDARY_ARGS+=("+secondary_dataset_root=${PREV_DATASET}")
            fi
        fi

        VAL_DATASET="${EVAL_DATASET_PATH:-$DATASET_PATH_ABS}"
        accelerate launch "${ACCEL_ARGS[@]}" \
            scripts/train_distrib.py --config-name main \
            category="${CATEGORY}" \
            use_train_test_split=true \
            dataset_root="${DATASET_PATH_ABS}" \
            ${SECONDARY_ARGS[@]+"${SECONDARY_ARGS[@]}"} \
            val_dataset_root="${VAL_DATASET}" \
            train_config.save_path="${NEW_MODEL_PATH}"
        safe_conda_deactivate
    fi

    if [[ $i -gt $START_ITER || $START_STEP -le 5 ]]; then
        # Step 5: run inference to get metrics
        echo "Iter ${i}, step 5"
        date
        safe_conda_activate "$ENV_BAT3R"
        cd "$REPO_ROOT"
        check_kerberos
        if [[ -n "$EVAL_DATASET_PATH" ]]; then
            STEP5_SPLIT="${EVAL_SPLIT}"
            STEP5_DATASET="${EVAL_DATASET_PATH}"
        else
            STEP5_SPLIT="train"
            STEP5_DATASET="${DATASET_PATH_ABS}"
        fi
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/infer.py --config-name infer \
            category="${CATEGORY}" \
            split="${STEP5_SPLIT}" \
            log_dir="${NEW_MODEL_PATH}" \
            dataset_root="${STEP5_DATASET}"
        safe_conda_deactivate
    fi
done

echo
echo "All iterations completed successfully."
