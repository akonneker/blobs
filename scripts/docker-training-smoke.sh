#!/bin/sh
set -eu

image=${TRAINING_IMAGE:-blobs-training:smoke}
revision=${BLOB_CODE_REVISION:-container-smoke}
run_id=${TRAINING_SMOKE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
output_root=${TRAINING_OUTPUT_DIR:-$(pwd)/training-output/smoke-${run_id}}

mkdir -p "$output_root"

docker build \
    --file Dockerfile.training \
    --build-arg "BLOB_CODE_REVISION=${revision}" \
    --tag "$image" \
    .

docker run --rm --init \
    --user "$(id -u):$(id -g)" \
    --env RAYON_NUM_THREADS="${TRAINING_THREADS:-2}" \
    --volume "$output_root:/output" \
    "$image" \
    train \
    --config /opt/blobs/config/default.toml \
    --num-envs 1 \
    --rollout-length 64 \
    --total-timesteps 64 \
    --eval-interval 0 \
    --checkpoint-interval 0 \
    --checkpoint-dir /output/checkpoints \
    --save-model /output/smoke-model

test -s "$output_root/checkpoints/metrics.csv"
test -s "$output_root/checkpoints/evaluation.csv"
test -s "$output_root/checkpoints/telemetry/summary.json"
test -s "$output_root/smoke-model.mpk"

echo "training runner smoke test passed; artifacts: $output_root"
