#!/bin/bash

CONFIG_FILE="$1"
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <path_to_config_file>"
    exit 1
fi

TIME_BIN="${TIME_BIN:-/usr/bin/time}"
if [ ! -x "$TIME_BIN" ]; then
    echo "Profiler binary not found or not executable: $TIME_BIN"
    exit 1
fi

# Parse YAML using yq
HOME_DIR=$(yq eval '.home' "$CONFIG_FILE")
DATASET_KEY=$(yq eval '.dataset' "$CONFIG_FILE")
CELL_LIST=$(yq eval '.cell' "$CONFIG_FILE")
GENE_LIST=$(yq eval '.gene' "$CONFIG_FILE")
REF_GENOME=$(yq eval '.ref-genome' "$CONFIG_FILE")
VERSION=$(yq eval '.version' "$CONFIG_FILE")
TMP_SAVE=$(yq eval '.tmp-save' "$CONFIG_FILE")
SEED=$(yq eval '.seed' "$CONFIG_FILE")

# Config log dir
LOG_DIR=$HOME_DIR/benchmark/$VERSION/log/
NET_DIR=$HOME_DIR/benchmark/$VERSION/net/
FIG_DIR=$HOME_DIR/benchmark/$VERSION/fig/

# Parse method list
METHODS=($(yq eval '.methods[].name' "$CONFIG_FILE"))

# Initialize associative arrays
declare -A METHOD_ENVS
declare -A METHOD_SCRIPTS

# Populate associative arrays
for i in $(seq 0 $(($(yq eval '.methods | length' "$CONFIG_FILE") - 1))); do
    METHOD_NAME=$(yq eval ".methods[$i].name" "$CONFIG_FILE")
    METHOD_ENVS["$METHOD_NAME"]=$(yq eval ".methods[$i].env" "$CONFIG_FILE")
    METHOD_SCRIPTS["$METHOD_NAME"]=$(yq eval ".methods[$i].script" "$CONFIG_FILE")
done

# Create output directory
mkdir -p "$HOME_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$NET_DIR"
mkdir -p "$FIG_DIR"

# Run methods
for METHOD_NAME in "${METHODS[@]}"; do
    METHOD_ENV=${METHOD_ENVS["$METHOD_NAME"]}
    METHOD_SCRIPT=${METHOD_SCRIPTS["$METHOD_NAME"]}
    METHOD_LOG="$LOG_DIR/$METHOD_NAME.log"

    echo "Running method: $METHOD_NAME..."

    # Activate Conda environment
    source activate "$METHOD_ENV"
    if [ $? -ne 0 ]; then
        echo "Failed to activate Conda environment: $METHOD_ENV"
        exit 1
    fi

    script_ext="${METHOD_SCRIPT##*.}"
    : > "$METHOD_LOG"

    case $script_ext in
        R)
            echo "Running R script: $METHOD_SCRIPT"
            "$TIME_BIN" -v Rscript "$METHOD_SCRIPT" --home "$HOME_DIR" --dataset "$DATASET_KEY" --cell "$CELL_LIST" --gene "$GENE_LIST" --version "$VERSION" --tmp-save "$TMP_SAVE" --seed "$SEED" --ref-genome "$REF_GENOME" 2>&1 | tee "$METHOD_LOG"
            ;;
        py)
            echo "Running Python script: $METHOD_SCRIPT"
            "$TIME_BIN" -v python "$METHOD_SCRIPT" --home "$HOME_DIR" --dataset "$DATASET_KEY" --cell "$CELL_LIST" --gene "$GENE_LIST" --version "$VERSION" --tmp-save "$TMP_SAVE" --seed "$SEED" --ref-genome "$REF_GENOME" 2>&1 | tee "$METHOD_LOG"
            ;;
        sh)
            echo "Running Shell script: $METHOD_SCRIPT"
            "$TIME_BIN" -v bash "$METHOD_SCRIPT" "$HOME_DIR" "$DATASET_KEY" "$CELL_LIST" "$GENE_LIST" "$VERSION" "$TMP_SAVE" "$SEED" "$REF_GENOME" 2>&1 | tee "$METHOD_LOG"
            ;;
        *)
            echo "Unsupported script type: $script_ext"
            exit 1
            ;;
    esac

    # Deactivate Conda environment
    conda deactivate
done
