#!/bin/sh
set -eu

image=${TRAINING_GPU_IMAGE:-blobs-training:gpu-smoke}
revision=${BLOB_CODE_REVISION:-container-gpu-smoke}
run_id=${TRAINING_SMOKE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
output_root=${TRAINING_OUTPUT_DIR:-$(pwd)/training-output/gpu-smoke-${run_id}}

mkdir -p "$output_root"

docker build \
    --file Dockerfile.training \
    --target gpu \
    --build-arg "BLOB_CODE_REVISION=${revision}" \
    --tag "$image" \
    .

docker run --rm \
    --gpus all \
    --read-only \
    --security-opt no-new-privileges:true \
    --cap-drop ALL \
    --user "$(id -u):$(id -g)" \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    --tmpfs /tmp:size=1g,mode=1777 \
    --volume "$output_root:/output" \
    "$image" \
    gpu-info

docker run --rm --init \
    --gpus all \
    --read-only \
    --security-opt no-new-privileges:true \
    --cap-drop ALL \
    --user "$(id -u):$(id -g)" \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    --env WGPU_BACKEND=vulkan \
    --env WGPU_POWER_PREF=high \
    --env RAYON_NUM_THREADS="${TRAINING_THREADS:-2}" \
    --tmpfs /tmp:size=1g,mode=1777 \
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

echo "GPU training runner smoke test passed; artifacts: $output_root"
