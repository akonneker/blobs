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
contact_evaluation_bin=${BLOB_CONTACT_EVALUATION_BIN:-target/release/contact-evaluation}
config=${BLOB_CONTACT_CONFIG:-blob_rl/config/competitive_transfer_large.toml}
seeds=${BLOB_CONTACT_DEMONSTRATION_SEEDS:-6101,6201,6301,6401,6501,6601,6701,6801,6901,7001,7101,7201,7301,7401,7501,7601,7701,7801,7901,8001,8101,8201,8301,8401,8501,8601,8701,8801,8901,9001,9101,9201}
evaluation_seeds=${BLOB_FEEDING_EVALUATION_SEEDS:-900000101,900000201,900000301,900000401,900000501,900000601,900000701,900000801}
contact_evaluation_seeds=${BLOB_CONTACT_EVALUATION_SEEDS:-910000101,910000201,910000301,910000401}
feeding_samples=${BLOB_FEEDING_DEMONSTRATION_SAMPLES:-6144}
contact_samples=${BLOB_CONTACT_DEMONSTRATION_SAMPLES:-512}
feeding_epochs=${BLOB_FEEDING_CLONE_EPOCHS:-96}
contact_epochs=${BLOB_CONTACT_CLONE_EPOCHS:-128}
consolidation_epochs=${BLOB_CONTACT_CONSOLIDATION_EPOCHS:-32}
consolidation_weights=${BLOB_CONTACT_CONSOLIDATION_WEIGHTS:-4,4,1,1}

"$demonstrations_bin" \
    --config "$config" \
    --teacher simple \
    --feeding-stage on-food \
    --seeds "$seeds" \
    --max-samples "$feeding_samples" \
    --output "$output_root/demonstrations/on-food"

"$demonstrations_bin" \
    --config "$config" \
    --teacher simple \
    --feeding-stage adjacent-food \
    --seeds "$seeds" \
    --max-samples "$feeding_samples" \
    --output "$output_root/demonstrations/adjacent-food"

for energy in 60 100 180; do
    for opponent in aggressive defensive; do
        dataset="$output_root/demonstrations/contact-$energy-$opponent"
        "$demonstrations_bin" \
            --config "$config" \
            --teacher aggressive \
            --contact-energy "$energy" \
            --contact-opponent "$opponent" \
            --seeds "$seeds" \
            --max-samples "$contact_samples" \
            --output "$dataset"
    done
done

"$behavior_clone_bin" \
    --config "$config" \
    --dataset "$output_root/demonstrations/on-food" \
    --dataset "$output_root/demonstrations/adjacent-food" \
    --epochs "$feeding_epochs" \
    --minibatch-size 256 \
    --recurrent-unroll-steps 16 \
    --validation-fraction 0.125 \
    --dataset-sampling balanced \
    --exact-round-trip-only \
    --action-balancing family \
    --action-balance-exponent 0.5 \
    --output "$output_root/feeding-clone"

"$feeding_evaluation_bin" \
    --config "$config" \
    --behavior-clone "$output_root/feeding-clone" \
    --seeds "$evaluation_seeds" \
    --require-pass \
    --output "$output_root/feeding-evaluation-parent.json"

"$behavior_clone_bin" \
    --config "$config" \
    --initial-behavior-clone "$output_root/feeding-clone" \
    --dataset "$output_root/demonstrations/contact-60-aggressive" \
    --dataset "$output_root/demonstrations/contact-180-defensive" \
    --epochs "$contact_epochs" \
    --minibatch-size 128 \
    --learning-rate 0.0001 \
    --recurrent-unroll-steps 16 \
    --validation-fraction 0.125 \
    --dataset-sampling balanced \
    --exact-round-trip-only \
    --action-balancing family \
    --action-balance-exponent 0.5 \
    --action-balance-max-ratio 2 \
    --output "$output_root/contact-clone"

"$behavior_clone_bin" \
    --config "$config" \
    --initial-behavior-clone "$output_root/contact-clone" \
    --dataset "$output_root/demonstrations/on-food" \
    --dataset "$output_root/demonstrations/adjacent-food" \
    --dataset "$output_root/demonstrations/contact-60-aggressive" \
    --dataset "$output_root/demonstrations/contact-180-defensive" \
    --dataset-weight "$consolidation_weights" \
    --epochs "$consolidation_epochs" \
    --minibatch-size 256 \
    --learning-rate 0.0001 \
    --recurrent-unroll-steps 16 \
    --validation-fraction 0.125 \
    --dataset-sampling proportional \
    --exact-round-trip-only \
    --action-balancing family \
    --action-balance-exponent 0.5 \
    --action-balance-max-ratio 2 \
    --output "$output_root/behavior-clone"

"$feeding_evaluation_bin" \
    --config "$config" \
    --behavior-clone "$output_root/behavior-clone" \
    --seeds "$evaluation_seeds" \
    --require-pass \
    --output "$output_root/feeding-evaluation.json"

"$contact_evaluation_bin" \
    --config "$config" \
    --behavior-clone "$output_root/behavior-clone" \
    --seeds "$contact_evaluation_seeds" \
    --require-damage \
    --output "$output_root/contact-evaluation.json"
