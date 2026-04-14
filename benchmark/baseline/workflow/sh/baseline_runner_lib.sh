#!/usr/bin/env bash
# Sourced by BaselineRunner.sh (not intended to run standalone).

baseline_runner_conda_base() {
    local exe="${CONDA_EXE:-}"
    if [[ -n "$exe" && -x "$exe" ]]; then
        (cd "$(dirname "$exe")/.." && pwd)
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        conda info --base 2>/dev/null
        return 0
    fi
    return 1
}

baseline_runner_abs_config_path() {
    local cfg="$1"
    echo "$(cd "$(dirname "$cfg")" && pwd)/$(basename "$cfg")"
}

baseline_runner_resolve_method_script() {
    local root="$1"
    local script="$2"
    if [[ "$script" != /* ]]; then
        script="${root}/${script#./}"
    fi
    echo "$script"
}
