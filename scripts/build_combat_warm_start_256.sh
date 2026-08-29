#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 OUTPUT_DIRECTORY [COMBAT_PARENT]" >&2
    exit 2
fi

output_root=$1
combat_parent=${2:-training-output/warmstarts/skirmish-schema35/behavior-clone}
demonstrations_bin=${BLOB_DEMONSTRATIONS_BIN:-target/release/demonstrations}
behavior_clone_bin=${BLOB_BEHAVIOR_CLONE_BIN:-target/release/behavior-clone}
feeding_teacher_evaluation_bin=${BLOB_FEEDING_TEACHER_EVALUATION_BIN:-target/release/feeding-teacher-evaluation}
feeding_evaluation_bin=${BLOB_FEEDING_EVALUATION_BIN:-target/release/feeding-evaluation}
feeding_evaluation_merge_bin=${BLOB_FEEDING_EVALUATION_MERGE_BIN:-target/release/feeding-evaluation-merge}
contact_evaluation_bin=${BLOB_CONTACT_EVALUATION_BIN:-target/release/contact-evaluation}
micro_combat_evaluation_bin=${BLOB_MICRO_COMBAT_EVALUATION_BIN:-target/release/micro-combat-evaluation}
config=${BLOB_COMBAT_WARM_START_256_CONFIG:-blob_rl/config/combat_warm_start_256.toml}
scenarios=${BLOB_MICRO_COMBAT_SCENARIOS:-blob_rl/config/micro_combat_scenarios.toml}
contact_dataset_root=${BLOB_CONTACT_DATASET_ROOT:-training-output/warmstarts/skirmish-schema35/demonstrations}
demonstration_seeds=${BLOB_FEEDING_DEMONSTRATION_SEEDS:-6101,6201,6301,6401,6501,6601,6701,6801,6901,7001,7101,7201,7301,7401,7501,7601,7701,7801,7901,8001,8101,8201,8301,8401,8501,8601,8701,8801,8901,9001,9101,9201}
feeding_evaluation_seeds=${BLOB_FEEDING_EVALUATION_SEEDS:-900000101,900000201,900000301,900000401,900000501,900000601,900000701,900000801}
contact_evaluation_seeds=${BLOB_CONTACT_EVALUATION_SEEDS:-910000101,910000201,910000301,910000401}
micro_evaluation_seeds=${BLOB_MICRO_COMBAT_EVALUATION_SEEDS:-920000101,920000201,920000301,920000401,920000501,920000601,920000701,920000801}
samples=${BLOB_FEEDING_DEMONSTRATION_SAMPLES:-32768}
consolidation_epochs=${BLOB_COMBAT_CONSOLIDATION_EPOCHS:-8}
# The exact contact datasets are only 544 samples combined versus 65,536
# feeding samples. These weights make combat roughly one third of each epoch
# instead of silently reducing it below one percent.
consolidation_weights=${BLOB_COMBAT_CONSOLIDATION_WEIGHTS:-1,1,64,64}
min_micro_attacks=${BLOB_MIN_MICRO_ATTACKS_PER_EPISODE:-0.25}

"$feeding_teacher_evaluation_bin" \
    --config "$config" \
    --teacher collision-aware-forager \
    --seeds "$feeding_evaluation_seeds" \
    --require-pass \
    --output "$output_root/teacher-feeding-evaluation.json"

"$demonstrations_bin" \
    --config "$config" \
    --teacher collision-aware-forager \
    --feeding-stage on-food \
    --seeds "$demonstration_seeds" \
    --max-samples "$samples" \
    --output "$output_root/demonstrations/on-food-256"

"$demonstrations_bin" \
    --config "$config" \
    --teacher collision-aware-forager \
    --feeding-stage adjacent-food \
    --seeds "$demonstration_seeds" \
    --max-samples "$samples" \
    --output "$output_root/demonstrations/adjacent-food-256"

"$behavior_clone_bin" \
    --config "$config" \
    --initial-behavior-clone "$combat_parent" \
    --dataset "$output_root/demonstrations/on-food-256" \
    --dataset "$output_root/demonstrations/adjacent-food-256" \
    --dataset "$contact_dataset_root/contact-60-aggressive" \
    --dataset "$contact_dataset_root/contact-180-defensive" \
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

IFS=',' read -r -a feeding_seed_array <<< "$feeding_evaluation_seeds"
feeding_merge_args=()
for seed in "${feeding_seed_array[@]}"; do
    shard="$output_root/feeding-evaluation-shards/seed-$seed.json"
    "$feeding_evaluation_bin" \
        --config "$config" \
        --behavior-clone "$output_root/behavior-clone" \
        --seeds "$seed" \
        --require-pass \
        --output "$shard"
    feeding_merge_args+=(--input "$shard")
done
"$feeding_evaluation_merge_bin" \
    "${feeding_merge_args[@]}" \
    --require-pass \
    --output "$output_root/feeding-evaluation.json"

"$contact_evaluation_bin" \
    --config "$config" \
    --behavior-clone "$output_root/behavior-clone" \
    --seeds "$contact_evaluation_seeds" \
    --require-damage \
    --output "$output_root/contact-evaluation.json"

"$micro_combat_evaluation_bin" \
    --config "$config" \
    --scenarios "$scenarios" \
    --behavior-clone "$output_root/behavior-clone" \
    --seeds "$micro_evaluation_seeds" \
    --require-attack-commitments-per-episode "$min_micro_attacks" \
    --output "$output_root/micro-combat-evaluation.json"
