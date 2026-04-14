#!/usr/bin/env bash
# Requires mikefarah/yq (YAML processor), not the Python "yq" package.
#
# Repo layout (this module): BaselineRunner.sh — entrypoint;
#   workflow/sh/baseline_runner_lib.sh — path/conda helpers (sourced here);
#   workflow/baseline_cli_utils.py — shared helpers for run_*.py;
#   workflow/conf/*.yaml — configs; workflow/run_*.py — method runners.
#
# Usage:
#   bash BaselineRunner.sh [--preflight] <path_to_config_file>
# Optional env:
#   TIME_BIN   - path to /usr/bin/time (default: /usr/bin/time)
#   CONDA_EXE  - if set, used to locate conda base (install root) when conda is not on PATH
#   BASELINE_PREFLIGHT_ENV - optional conda env used to run workflow/run_preflight.py

set -euo pipefail

PREFLIGHT_ENABLED="false"
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preflight)
            PREFLIGHT_ENABLED="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--preflight] <path_to_config_file>"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--preflight] <path_to_config_file>" >&2
            exit 1
            ;;
        *)
            if [[ -n "$CONFIG_FILE" ]]; then
                echo "Unexpected extra argument: $1" >&2
                echo "Usage: $0 [--preflight] <path_to_config_file>" >&2
                exit 1
            fi
            CONFIG_FILE="$1"
            shift
            ;;
    esac
done

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Usage: $0 [--preflight] <path_to_config_file>" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
# shellcheck source=workflow/sh/baseline_runner_lib.sh
source "${ROOT_DIR}/workflow/sh/baseline_runner_lib.sh"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found (relative to $ROOT_DIR): $CONFIG_FILE" >&2
    exit 1
fi
CONFIG_FILE="$(baseline_runner_abs_config_path "$CONFIG_FILE")"

if ! command -v yq >/dev/null 2>&1; then
    echo "yq is required but not found on PATH" >&2
    exit 1
fi

TIME_BIN="${TIME_BIN:-/usr/bin/time}"
if [[ ! -x "$TIME_BIN" ]]; then
    echo "Profiler binary not found or not executable: $TIME_BIN" >&2
    exit 1
fi

CONDA_BASE="$(baseline_runner_conda_base)" || {
    echo "conda not found: set CONDA_EXE or add conda to PATH" >&2
    exit 1
}
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
    echo "Missing conda.sh at $CONDA_SH" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_SH"

# --- Parse YAML ---
HOME_DIR=$(yq eval '.home' "$CONFIG_FILE")
DATASET_KEY=$(yq eval '.dataset' "$CONFIG_FILE")
CELL_LIST=$(yq eval '.cell' "$CONFIG_FILE")
GENE_LIST=$(yq eval '.gene' "$CONFIG_FILE")
REF_GENOME=$(yq eval '.ref-genome' "$CONFIG_FILE")
VERSION=$(yq eval '.version' "$CONFIG_FILE")
SEED=$(yq eval '.seed' "$CONFIG_FILE")

_raw_tmp=$(yq eval '.tmp-save // false' "$CONFIG_FILE")
case "${_raw_tmp,,}" in
    true|1|yes)  TMP_SAVE_PY="true";  TMP_SAVE_SH="true";  TMP_SAVE_R="TRUE" ;;
    false|0|no|"") TMP_SAVE_PY="false"; TMP_SAVE_SH="false"; TMP_SAVE_R="FALSE" ;;
    *)
        echo "Invalid tmp-save value in config: ${_raw_tmp}" >&2
        exit 1
        ;;
esac

# --- Validation (startup) ---
[[ -n "$HOME_DIR" && "$HOME_DIR" != "null" ]] || { echo "config: home is missing or null" >&2; exit 1; }
[[ -n "$DATASET_KEY" && "$DATASET_KEY" != "null" ]] || { echo "config: dataset is missing or null" >&2; exit 1; }
[[ -d "$HOME_DIR" ]] || { echo "home directory does not exist: $HOME_DIR" >&2; exit 1; }
[[ -f "$CELL_LIST" ]] || { echo "cell list not found: $CELL_LIST" >&2; exit 1; }
[[ -f "$GENE_LIST" ]] || { echo "gene list not found: $GENE_LIST" >&2; exit 1; }

LOG_DIR="${HOME_DIR}/benchmark/${VERSION}/log/"
NET_DIR="${HOME_DIR}/benchmark/${VERSION}/net/"
FIG_DIR="${HOME_DIR}/benchmark/${VERSION}/fig/"

METHOD_COUNT=$(yq eval '.methods | length' "$CONFIG_FILE")
if [[ "$METHOD_COUNT" -eq 0 ]]; then
    echo "No methods defined under 'methods:' in config" >&2
    exit 1
fi

for i in $(seq 0 $((METHOD_COUNT - 1))); do
    _ms=$(yq eval ".methods[$i].script" "$CONFIG_FILE")
    _ms="$(baseline_runner_resolve_method_script "$ROOT_DIR" "$_ms")"
    [[ -f "$_ms" ]] || { echo "Method script not found (methods[$i].script): $_ms" >&2; exit 1; }
done

mkdir -p "$HOME_DIR" "$LOG_DIR" "$NET_DIR" "$FIG_DIR"

PREFLIGHT_JSON=""
if [[ "$PREFLIGHT_ENABLED" == "true" ]]; then
    PREFLIGHT_JSON="${LOG_DIR}/preflight.json"
    PREFLIGHT_REPORT="${LOG_DIR}/preflight_report.md"
    PREFLIGHT_CMD=(python "${ROOT_DIR}/workflow/run_preflight.py" --config "$CONFIG_FILE" --json-out "$PREFLIGHT_JSON" --report-out "$PREFLIGHT_REPORT")

    echo "Running baseline preflight checks..."
    if [[ -n "${BASELINE_PREFLIGHT_ENV:-}" ]]; then
        conda run -n "$BASELINE_PREFLIGHT_ENV" "${PREFLIGHT_CMD[@]}"
    else
        "${PREFLIGHT_CMD[@]}"
    fi

    [[ -f "$PREFLIGHT_JSON" ]] || { echo "Preflight did not produce JSON output: $PREFLIGHT_JSON" >&2; exit 1; }
fi

# --- Run methods ---
for i in $(seq 0 $((METHOD_COUNT - 1))); do
    METHOD_NAME=$(yq eval ".methods[$i].name" "$CONFIG_FILE")
    METHOD_ENV=$(yq eval ".methods[$i].env" "$CONFIG_FILE")
    METHOD_SCRIPT=$(yq eval ".methods[$i].script" "$CONFIG_FILE")
    METHOD_SCRIPT="$(baseline_runner_resolve_method_script "$ROOT_DIR" "$METHOD_SCRIPT")"

    mapfile -t METHOD_EXTRA_ARGS < <(yq eval ".methods[$i].args // [] | .[]" "$CONFIG_FILE")

    METHOD_LOG="${LOG_DIR}/${METHOD_NAME}.log"

    if [[ "$PREFLIGHT_ENABLED" == "true" ]]; then
        METHOD_STATUS=$(yq eval ".method_status[\"${METHOD_NAME}\"].status // \"ok\"" "$PREFLIGHT_JSON")
        if [[ "$METHOD_STATUS" == "skip" ]]; then
            mapfile -t METHOD_SKIP_REASONS < <(yq eval ".method_status[\"${METHOD_NAME}\"].reasons[]" "$PREFLIGHT_JSON" 2>/dev/null || true)
            : > "$METHOD_LOG"
            {
                echo "SKIPPED by preflight: ${METHOD_NAME}"
                if [[ "${#METHOD_SKIP_REASONS[@]}" -eq 0 ]]; then
                    echo "No skip reason was recorded."
                else
                    printf '%s\n' "${METHOD_SKIP_REASONS[@]}"
                fi
            } | tee "$METHOD_LOG"
            continue
        fi
    fi

    echo "Running method: $METHOD_NAME..."

    conda activate "$METHOD_ENV"

    script_ext="${METHOD_SCRIPT##*.}"
    : > "$METHOD_LOG"

    case $script_ext in
        R)
            echo "Running R script: $METHOD_SCRIPT"
            "$TIME_BIN" -v Rscript "$METHOD_SCRIPT" \
                --home "$HOME_DIR" --dataset "$DATASET_KEY" --cell "$CELL_LIST" --gene "$GENE_LIST" \
                --version "$VERSION" --tmp-save "$TMP_SAVE_R" --seed "$SEED" --ref-genome "$REF_GENOME" \
                "${METHOD_EXTRA_ARGS[@]}" 2>&1 | tee "$METHOD_LOG"
            ;;
        py)
            echo "Running Python script: $METHOD_SCRIPT"
            "$TIME_BIN" -v python "$METHOD_SCRIPT" \
                --home "$HOME_DIR" --dataset "$DATASET_KEY" --cell "$CELL_LIST" --gene "$GENE_LIST" \
                --version "$VERSION" --tmp-save "$TMP_SAVE_PY" --seed "$SEED" --ref-genome "$REF_GENOME" \
                "${METHOD_EXTRA_ARGS[@]}" 2>&1 | tee "$METHOD_LOG"
            ;;
        sh)
            echo "Running Shell script: $METHOD_SCRIPT"
            "$TIME_BIN" -v bash "$METHOD_SCRIPT" "$HOME_DIR" "$DATASET_KEY" "$CELL_LIST" "$GENE_LIST" \
                "$VERSION" "$TMP_SAVE_SH" "$SEED" "$REF_GENOME" \
                "${METHOD_EXTRA_ARGS[@]}" 2>&1 | tee "$METHOD_LOG"
            ;;
        *)
            echo "Unsupported script type: $script_ext" >&2
            exit 1
            ;;
    esac

    conda deactivate
done

echo "BaselineRunner finished all methods."
