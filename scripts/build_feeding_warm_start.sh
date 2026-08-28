#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 OUTPUT_DIRECTORY" >&2
    exit 2
fi

output_root=$1
demonstrations_bin=${BLOB_DEMONSTRATIONS_BIN:-target/release/demonstrations}
behavior_clone_bin=${BLOB_BEHAVIOR_CLONE_BIN:-target/release/behavior-clone}
feeding_evaluation_bin=${BLOB_FEEDING_EVALUATION_BIN:-target/release/feeding-evaluation}
config=${BLOB_FEEDING_CONFIG:-blob_rl/config/fast_learn.toml}
seeds=${BLOB_FEEDING_DEMONSTRATION_SEEDS:-6101,6201,6301,6401,6501,6601,6701,6801,6901,7001,7101,7201,7301,7401,7501,7601,7701,7801,7901,8001,8101,8201,8301,8401,8501,8601,8701,8801,8901,9001,9101,9201}
evaluation_seeds=${BLOB_FEEDING_EVALUATION_SEEDS:-900000101,900000201,900000301,900000401,900000501,900000601,900000701,900000801}
# With fast_learn's 12 founders this retains about sixteen decisions per cell
# and teaches the recurrent policy to remain feeding after move -> consume.
samples=${BLOB_FEEDING_DEMONSTRATION_SAMPLES:-6144}
epochs=${BLOB_FEEDING_CLONE_EPOCHS:-128}

"$demonstrations_bin" \
    --config "$config" \
    --teacher simple \
    --feeding-stage on-food \
    --seeds "$seeds" \
    --max-samples "$samples" \
    --output "$output_root/demonstrations/on-food"

"$demonstrations_bin" \
    --config "$config" \
    --teacher simple \
    --feeding-stage adjacent-food \
    --seeds "$seeds" \
    --max-samples "$samples" \
    --output "$output_root/demonstrations/adjacent-food"

"$behavior_clone_bin" \
    --config "$config" \
    --dataset "$output_root/demonstrations/on-food" \
    --dataset "$output_root/demonstrations/adjacent-food" \
    --epochs "$epochs" \
    --minibatch-size 256 \
    --recurrent-unroll-steps 16 \
    --validation-fraction 0.125 \
    --dataset-sampling balanced \
    --exact-round-trip-only \
    --action-balancing family \
    --action-balance-exponent 0.5 \
    --output "$output_root/behavior-clone"

"$feeding_evaluation_bin" \
    --config "$config" \
    --behavior-clone "$output_root/behavior-clone" \
    --seeds "$evaluation_seeds" \
    --require-pass \
    --output "$output_root/feeding-evaluation.json"
