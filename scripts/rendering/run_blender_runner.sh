#!/usr/bin/env bash
# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.

# Usage:
#   ./run_blender_runner.sh <N_GPUS> <DATA_ROOT>

set -euo pipefail
trap 'echo "❌ Error at line $LINENO"; exit 1' ERR

if [ $# -ne 2 ]; then
    echo "Usage: $0 <N_GPUS> <DATA_ROOT>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cleanup() {
    echo "❌ Cleaning up Blender jobs"
    for pid in "${pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM

JOBS_PER_GPU=4

N_GPUS=$1
if [[ -z "$N_GPUS" || "$N_GPUS" -le 0 ]]; then
    echo "❌ Error: N_GPUS must be a positive integer."
    exit 1
fi
TOTAL_JOBS=$((N_GPUS * JOBS_PER_GPU))
DATA_ROOT=$2

N_CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
THREADS=$((N_CPUS / TOTAL_JOBS))
if [ "$THREADS" -lt 1 ]; then
    THREADS=1
fi

echo "Launching job with N_GPUS=$N_GPUS, TOTAL_JOBS=$TOTAL_JOBS"
echo "Threads per process: $THREADS"
echo "Data root: $DATA_ROOT"

# Handle CUDA_VISIBLE_DEVICES mapping to ensure we use the correct physical IDs
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    # If unset, assume 0 to N_GPUS-1
    GPU_LIST=($(seq 0 $((N_GPUS - 1))))
fi

pids=()

for i in $(seq 0 $((TOTAL_JOBS - 1))); do
    GPU_ID=$((i / JOBS_PER_GPU))
    PHYSICAL_GPU=${GPU_LIST[$GPU_ID]}
    echo "Launching job rank=$i on GPU=$GPU_ID (Physical: $PHYSICAL_GPU)"

    if [ "$i" -gt 0 ]; then
        # Ranks > 0 run silently
        CUDA_VISIBLE_DEVICES=$PHYSICAL_GPU OMP_NUM_THREADS=$THREADS blender horses.blend \
            --threads $THREADS \
            --background \
            --python "$SCRIPT_DIR/blender_render_posed.py" \
            -- $i $TOTAL_JOBS "$DATA_ROOT" \
            > /dev/null 2>&1 & pids+=($!)
    else
        # Rank 0 prints output normally
        CUDA_VISIBLE_DEVICES=$PHYSICAL_GPU OMP_NUM_THREADS=$THREADS blender horses.blend \
            --threads $THREADS \
            --background \
            --python "$SCRIPT_DIR/blender_render_posed.py" \
            -- $i $TOTAL_JOBS "$DATA_ROOT" \
            > /dev/null & pids+=($!)
    fi
done

FAIL=0
for pid in "${pids[@]}"; do
    wait "$pid" || FAIL=1
done

if [ "$FAIL" -ne 0 ]; then
    echo "❌ At least one Blender job failed."
    exit 1
fi
