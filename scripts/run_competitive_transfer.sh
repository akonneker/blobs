#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 BEHAVIOR_CLONE_DIR BEHAVIOR_CLONE_METADATA_SHA256 FEEDING_EVALUATION_JSON FEEDING_ARTIFACT_HASH OUTPUT_DIRECTORY" >&2
    exit 2
fi

behavior_clone=$1
behavior_clone_sha256=$2
feeding_evaluation=$3
feeding_artifact_hash=$4
output_root=$5
train_bin=${BLOB_TRAIN_BIN:-target/release/train}
config=${BLOB_TRANSFER_CONFIG:-blob_rl/config/competitive_transfer.toml}

"$train_bin" \
    --config "$config" \
    --checkpoint-dir "$output_root" \
    --initial-policy "$behavior_clone" \
    --initial-policy-artifact-sha256 "$behavior_clone_sha256" \
    --initial-policy-qualification "$feeding_evaluation" \
    --initial-policy-qualification-hash "$feeding_artifact_hash"
