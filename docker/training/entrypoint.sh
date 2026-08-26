#!/bin/sh
set -eu

umask 0022

if [ ! -d /output ]; then
    echo "training runner: /output is missing" >&2
    exit 73
fi
if [ ! -w /output ]; then
    echo "training runner: /output is not writable by uid $(id -u), gid $(id -g)" >&2
    echo "mount a writable directory or run with --user matching its owner" >&2
    exit 73
fi

# CubeCL persists autotuning/kernel data below $HOME/.cache and currently
# assumes the parent exists. GPU containers put it on disposable /tmp tmpfs.
if [ -n "${HOME:-}" ]; then
    mkdir -p "$HOME/.cache"
fi
if [ -n "${XDG_CONFIG_HOME:-}" ]; then
    mkdir -p "$XDG_CONFIG_HOME"
fi
if [ -n "${XDG_RUNTIME_DIR:-}" ]; then
    mkdir -p "$XDG_RUNTIME_DIR"
    chmod 0700 "$XDG_RUNTIME_DIR"
fi

command_name=${1:-train}
case "$command_name" in
    train)
        if [ "$#" -gt 0 ]; then
            shift
        fi
        exec /usr/local/bin/train "$@"
        ;;
    sweep-plan)
        shift
        exec /usr/local/bin/rules-sweep "$@"
        ;;
    sweep-run)
        shift
        exec /usr/local/bin/rules-sweep-run "$@"
        ;;
    demonstrations)
        shift
        exec /usr/local/bin/demonstrations "$@"
        ;;
    behavior-clone)
        shift
        exec /usr/local/bin/behavior-clone "$@"
        ;;
    viability)
        shift
        exec /usr/local/bin/viability "$@"
        ;;
    viability-matrix)
        shift
        exec /usr/local/bin/viability-matrix "$@"
        ;;
    viability-gate)
        shift
        exec /usr/local/bin/viability-gate "$@"
        ;;
    viability-preflight)
        shift
        exec /usr/local/bin/viability-preflight "$@"
        ;;
    gpu-info)
        shift
        if ! command -v vulkaninfo >/dev/null 2>&1; then
            echo "training runner: gpu-info is available only in the GPU image" >&2
            exit 69
        fi
        exec vulkaninfo --summary "$@"
        ;;
    --help|-h)
        cat <<'EOF'
Usage:
  blob-training [train] [TRAIN_OPTIONS...]
  blob-training sweep-plan SWEEP_SPEC
  blob-training sweep-run MANIFEST [SWEEP_OPTIONS...]
  blob-training viability --config CONFIG --candidate PROFILE --opponent PROFILE --seeds SEEDS --output REPORT
  blob-training viability-matrix MANIFEST --candidates PROFILES --opponents PROFILES --output REPORT [MATRIX_OPTIONS...]
  blob-training viability-gate MATRIX --gates POLICY --output DECISION
  blob-training viability-preflight SWEEP_SPEC --candidates PROFILES --opponents PROFILES --gates POLICY [PREFLIGHT_OPTIONS...]
  blob-training gpu-info

Durable configs/checkpoints should be mounted under /config and /output.
EOF
        ;;
    --version|-V)
        exec /usr/local/bin/train --version
        ;;
    *)
        echo "training runner: unknown command '$command_name'" >&2
        echo "expected train, viability, viability-matrix, viability-gate, viability-preflight, sweep-plan, sweep-run, or gpu-info" >&2
        exit 64
        ;;
esac
