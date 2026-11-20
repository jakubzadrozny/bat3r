#!/usr/bin/env bash
# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.

init_conda() {
    set +u
    local conda_home
    conda_home=$(conda info --base)
    # shellcheck source=/dev/null
    source "$conda_home/etc/profile.d/conda.sh"
    set -u
}

safe_conda_activate() {
    set +u
    conda activate "$1"
    set -u
}

safe_conda_deactivate() {
    set +u
    conda deactivate
    set -u
}

to_abs_path() {
    local p="$1"
    if [[ "$p" = /* ]]; then
        echo "$p"
    else
        echo "$(pwd)/$p"
    fi
}

load_launcher_config() {
    if [[ $# -lt 1 ]]; then
        echo "Usage: $0 <config.env> [KEY=VAL ...]" >&2
        return 1
    fi
    local config_file="$1"
    shift
    if [[ ! -f "$config_file" ]]; then
        echo "❌ Error: Config file not found: $config_file" >&2
        return 1
    fi
    # shellcheck source=/dev/null
    source "$config_file"

    local arg key val
    for arg in "$@"; do
        if [[ "$arg" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            val="${BASH_REMATCH[2]}"
            printf -v "$key" '%s' "$val"
        else
            echo "❌ Error: Invalid override '$arg' (expected KEY=VAL)" >&2
            return 1
        fi
    done
}

get_gpu_count() {
    local env_name="${1:-bat3r}"
    safe_conda_activate "$env_name"
    local count
    count=$(python - <<'EOF' 2>/dev/null
import torch
print(torch.cuda.device_count())
EOF
)
    safe_conda_deactivate
    if [[ -z "$count" || "$count" -le 0 ]]; then
        echo "❌ Error: Could not detect GPUs or n_gpus is invalid ($count)." >&2
        return 1
    fi
    echo "$count"
}

check_kerberos() {
    if [[ "${KERBEROS_ENABLED:-0}" == "0" ]]; then
        return 0
    fi
    echo "===== AUTH STATE $(date) ====="
    echo "PID=$$"
    echo "Using KRB5CCNAME=${KRB5CCNAME:-default}"
    echo "--- klist ---"
    klist -s || { echo "❌ Kerberos ticket missing or expired"; exit 1; }
    klist || echo "❌ klist failed"
    echo "--- tokens ---"
    tokens || echo "❌ tokens failed"
    echo "=============================="
}

setup_kerberos_guard() {
    if [[ "${KERBEROS_ENABLED:-0}" != "0" ]]; then
        krenew -K 60 -t &
        KRENEW_PID=$!
        trap 'kill "$KRENEW_PID" 2>/dev/null || true' EXIT
        check_kerberos
    fi
}
