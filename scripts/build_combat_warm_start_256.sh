#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 OUTPUT_DIRECTORY [COMBAT_PARENT]" >&2
    exit 2
fi

output_root=$1
combat_parent=${2:-}
demonstrations_bin=${BLOB_DEMONSTRATIONS_BIN:-target/release/demonstrations}
behavior_clone_bin=${BLOB_BEHAVIOR_CLONE_BIN:-target/release/behavior-clone}
feeding_layout_teacher_evaluation_bin=${BLOB_FEEDING_LAYOUT_TEACHER_EVALUATION_BIN:-target/release/feeding-layout-teacher-evaluation}
feeding_evaluation_bin=${BLOB_FEEDING_EVALUATION_BIN:-target/release/feeding-evaluation}
feeding_evaluation_merge_bin=${BLOB_FEEDING_EVALUATION_MERGE_BIN:-target/release/feeding-evaluation-merge}
feeding_layout_evaluation_bin=${BLOB_FEEDING_LAYOUT_EVALUATION_BIN:-target/release/feeding-layout-evaluation}
feeding_layout_evaluation_merge_bin=${BLOB_FEEDING_LAYOUT_EVALUATION_MERGE_BIN:-target/release/feeding-layout-evaluation-merge}
contact_evaluation_bin=${BLOB_CONTACT_EVALUATION_BIN:-target/release/contact-evaluation}
micro_combat_evaluation_bin=${BLOB_MICRO_COMBAT_EVALUATION_BIN:-target/release/micro-combat-evaluation}
config=${BLOB_COMBAT_WARM_START_256_CONFIG:-blob_rl/config/combat_warm_start_256.toml}
scenarios=${BLOB_MICRO_COMBAT_SCENARIOS:-blob_rl/config/micro_combat_scenarios.toml}
feeding_demonstration_seeds=${BLOB_FEEDING_DEMONSTRATION_SEEDS:-6101,6201,6301,6401,6501,6601,6701,6801}
contact_demonstration_seeds=${BLOB_CONTACT_DEMONSTRATION_SEEDS:-6101,6201,6301,6401,6501,6601,6701,6801,6901,7001,7101,7201,7301,7401,7501,7601,7701,7801,7901,8001,8101,8201,8301,8401,8501,8601,8701,8801,8901,9001,9101,9201}
feeding_evaluation_seeds=${BLOB_FEEDING_EVALUATION_SEEDS:-900000101,900000201,900000301,900000401,900000501,900000601,900000701,900000801}
feeding_layout_evaluation_seeds=${BLOB_FEEDING_LAYOUT_EVALUATION_SEEDS:-930000101,930000202}
feeding_layouts=${BLOB_FEEDING_LAYOUTS:-line,checkerboard,ring,loose-random,random}
contact_evaluation_seeds=${BLOB_CONTACT_EVALUATION_SEEDS:-910000101,910000201,910000301,910000401}
micro_evaluation_seeds=${BLOB_MICRO_COMBAT_EVALUATION_SEEDS:-920000101,920000201,920000301,920000401,920000501,920000601,920000701,920000801}
samples=${BLOB_FEEDING_DEMONSTRATION_SAMPLES:-32768}
contact_samples=${BLOB_CONTACT_DEMONSTRATION_SAMPLES:-512}
consolidation_epochs=${BLOB_COMBAT_CONSOLIDATION_EPOCHS:-32}
# Explicit weights are direct dataset mixture shares, independent of shard
# length. Each layout contributes 0.25 on-food plus 1.75 adjacent-food; the two
# 2.5-share contact shards retain one third of the epoch. This concentrates the
# feeding budget on the critical target-selection transition while retaining a
# smaller stationary-consume rehearsal.
consolidation_weights=${BLOB_COMBAT_CONSOLIDATION_WEIGHTS:-0.25,1.75,0.25,1.75,0.25,1.75,0.25,1.75,0.25,1.75,2.5,2.5}
action_balance_exponent=${BLOB_ACTION_BALANCE_EXPONENT:-1}
action_balance_max_ratio=${BLOB_ACTION_BALANCE_MAX_RATIO:-4}
min_micro_attacks=${BLOB_MIN_MICRO_ATTACKS_PER_EPISODE:-0.25}

"$feeding_layout_teacher_evaluation_bin" \
    --config "$config" \
    --teacher collision-aware-forager \
    --layouts "$feeding_layouts" \
    --seeds "$feeding_layout_evaluation_seeds" \
    --require-pass \
    --output "$output_root/teacher-feeding-layout-evaluation.json"

IFS=',' read -r -a feeding_layout_array <<< "$feeding_layouts"
samples_per_layout=$((samples / ${#feeding_layout_array[@]}))
if ((samples_per_layout == 0)); then
    echo "feeding demonstration sample budget must cover every layout" >&2
    exit 2
fi
feeding_dataset_args=()
for layout in "${feeding_layout_array[@]}"; do
    for stage in on-food adjacent-food; do
        dataset="$output_root/demonstrations/$stage-$layout-256"
        "$demonstrations_bin" \
            --config "$config" \
            --teacher collision-aware-forager \
            --feeding-stage "$stage" \
            --starting-layout "$layout" \
            --seeds "$feeding_demonstration_seeds" \
            --max-samples "$samples_per_layout" \
            --output "$dataset"
        feeding_dataset_args+=(--dataset "$dataset")
    done
done

for contact_spec in 60:aggressive 180:defensive; do
    energy=${contact_spec%%:*}
    opponent=${contact_spec##*:}
    dataset="$output_root/demonstrations/contact-$energy-$opponent"
    "$demonstrations_bin" \
        --config "$config" \
        --teacher aggressive \
        --contact-energy "$energy" \
        --contact-opponent "$opponent" \
        --seeds "$contact_demonstration_seeds" \
        --max-samples "$contact_samples" \
        --output "$dataset"
done

run_behavior_clone() {
    "$behavior_clone_bin" \
        --config "$config" \
        "$@" \
        "${feeding_dataset_args[@]}" \
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
        --action-balance-exponent "$action_balance_exponent" \
        --action-balance-max-ratio "$action_balance_max_ratio" \
        --output "$output_root/behavior-clone"
}
if [[ -n "$combat_parent" && "$combat_parent" != "none" ]]; then
    run_behavior_clone --initial-behavior-clone "$combat_parent"
else
    run_behavior_clone
fi

IFS=',' read -r -a feeding_seed_array <<< "$feeding_evaluation_seeds"
feeding_merge_args=()
for seed in "${feeding_seed_array[@]}"; do
    shard="$output_root/feeding-evaluation-shards/seed-$seed.json"
    "$feeding_evaluation_bin" \
        --config "$config" \
        --behavior-clone "$output_root/behavior-clone" \
        --seeds "$seed" \
        --output "$shard"
    feeding_merge_args+=(--input "$shard")
done
"$feeding_evaluation_merge_bin" \
    "${feeding_merge_args[@]}" \
    --require-pass \
    --output "$output_root/feeding-evaluation.json"

IFS=',' read -r -a feeding_layout_seed_array <<< "$feeding_layout_evaluation_seeds"
feeding_layout_merge_args=()
for layout in "${feeding_layout_array[@]}"; do
    for seed in "${feeding_layout_seed_array[@]}"; do
        shard="$output_root/feeding-layout-evaluation-shards/$layout-seed-$seed.json"
        "$feeding_layout_evaluation_bin" \
            --config "$config" \
            --behavior-clone "$output_root/behavior-clone" \
            --layouts "$layout" \
            --seeds "$seed" \
            --output "$shard"
        feeding_layout_merge_args+=(--input "$shard")
    done
done
"$feeding_layout_evaluation_merge_bin" \
    "${feeding_layout_merge_args[@]}" \
    --require-pass \
    --output "$output_root/feeding-layout-evaluation.json"

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
