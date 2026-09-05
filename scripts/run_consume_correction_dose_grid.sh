#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 OUTPUT_ROOT PARENT_CLONE REHEARSAL_DATASET_ROOT ON_FOOD_CORRECTIONS" >&2
    exit 2
fi

output_root=$1
parent_clone=$2
rehearsal_root=$3
on_food_corrections=$4

behavior_clone_bin=${BLOB_BEHAVIOR_CLONE_BIN:-target/release/behavior-clone}
feeding_evaluation_bin=${BLOB_FEEDING_EVALUATION_BIN:-target/release/feeding-evaluation}
feeding_evaluation_merge_bin=${BLOB_FEEDING_EVALUATION_MERGE_BIN:-target/release/feeding-evaluation-merge}
feeding_corrections_bin=${BLOB_FEEDING_POLICY_CORRECTIONS_BIN:-target/release/feeding-policy-corrections}
config=${BLOB_COMBAT_WARM_START_256_CONFIG:-blob_rl/config/combat_warm_start_256.toml}
dose_weights=${BLOB_CONSUME_CORRECTION_DOSE_WEIGHTS:-3.75,7.5,15}
evaluation_seeds=${BLOB_CONSUME_CORRECTION_EVALUATION_SEEDS:-1000000101,1000000202,1000000303,1000000404,1000000505,1000000606,1000000707,1000000808}
margin_probe_seed=${BLOB_CONSUME_CORRECTION_MARGIN_SEED:-1010000101}
epochs=${BLOB_CONSUME_CORRECTION_EPOCHS:-2}
epoch_samples=${BLOB_CONSUME_CORRECTION_EPOCH_SAMPLES:-14806}
optimizer_steps_per_epoch=${BLOB_CONSUME_CORRECTION_OPTIMIZER_STEPS_PER_EPOCH:-64}
learning_rate=${BLOB_CONSUME_CORRECTION_LEARNING_RATE:-0.00005}
training_seed=${BLOB_CONSUME_CORRECTION_TRAINING_SEED:-42}
action_kind_expert_only=${BLOB_CONSUME_ACTION_KIND_EXPERT_ONLY:-}
foraging_adapter_only=${BLOB_CONSUME_FORAGING_ADAPTER_ONLY:-0}
context_adapter_only=${BLOB_CONSUME_CONTEXT_ADAPTER_ONLY:-}
context_slot_adapter_only=${BLOB_CONSUME_CONTEXT_SLOT_ADAPTER_ONLY:-}
action_kind_margin=${BLOB_CONSUME_ACTION_KIND_MARGIN:-}
action_kind_margin_loss_weight=${BLOB_CONSUME_ACTION_KIND_MARGIN_LOSS_WEIGHT:-0}
target_query_head_only=${BLOB_CONSUME_TARGET_QUERY_HEAD_ONLY:-0}
effort_head_only=${BLOB_CONSUME_EFFORT_HEAD_ONLY:-0}

rehearsal_names=(
    on-food-line-256 adjacent-food-line-256
    on-food-checkerboard-256 adjacent-food-checkerboard-256
    on-food-ring-256 adjacent-food-ring-256
    on-food-loose-random-256 adjacent-food-loose-random-256
    on-food-random-256 adjacent-food-random-256
    contact-60-aggressive contact-180-defensive
)
rehearsal_weights=0.25,1.75,0.25,1.75,0.25,1.75,0.25,1.75,0.25,1.75,2.5,2.5
dataset_args=()
for name in "${rehearsal_names[@]}"; do
    dataset="$rehearsal_root/$name"
    if [[ ! -f "$dataset/manifest.json" || ! -f "$dataset/samples.mpk" ]]; then
        echo "missing rehearsal dataset $dataset" >&2
        exit 2
    fi
    dataset_args+=(--dataset "$dataset")
done
if [[ ! -f "$on_food_corrections/manifest.json" || ! -f "$on_food_corrections/samples.mpk" ]]; then
    echo "missing on-food correction dataset $on_food_corrections" >&2
    exit 2
fi

train_arm() {
    local arm=$1
    local weights=$2
    shift 2
    expert_only_args=()
    if [[ -n "$action_kind_expert_only" ]]; then
        expert_only_args+=(--action-kind-expert-only "$action_kind_expert_only")
    fi
    if [[ "$foraging_adapter_only" == "1" ]]; then
        expert_only_args+=(--foraging-adapter-only)
    fi
    if [[ -n "$context_adapter_only" ]]; then
        expert_only_args+=(--context-adapter-only "$context_adapter_only")
    fi
    if [[ -n "$context_slot_adapter_only" ]]; then
        expert_only_args+=(--context-slot-adapter-only "$context_slot_adapter_only")
    fi
    if [[ -n "$action_kind_margin" ]]; then
        expert_only_args+=(
            --action-kind-margin "$action_kind_margin"
            --action-kind-margin-loss-weight "$action_kind_margin_loss_weight"
        )
    fi
    if [[ "$target_query_head_only" == "1" ]]; then
        expert_only_args+=(--target-query-head-only)
    fi
    if [[ "$effort_head_only" == "1" ]]; then
        expert_only_args+=(--effort-head-only)
    fi
    "$behavior_clone_bin" \
        --config "$config" \
        --initial-behavior-clone "$parent_clone" \
        "${dataset_args[@]}" \
        "$@" \
        --dataset-weight "$weights" \
        --epoch-samples "$epoch_samples" \
        --optimizer-steps-per-epoch "$optimizer_steps_per_epoch" \
        --expert-routing foraging-interaction-exploration \
        --phase-gate-loss-weight 1 \
        "${expert_only_args[@]}" \
        --epochs "$epochs" \
        --minibatch-size 256 \
        --learning-rate "$learning_rate" \
        --seed "$training_seed" \
        --recurrent-unroll-steps 16 \
        --validation-fraction 0.125 \
        --dataset-sampling proportional \
        --exact-round-trip-only \
        --action-balancing family \
        --action-balance-exponent 1 \
        --action-balance-max-ratio 4 \
        --output "$output_root/$arm/behavior-clone"
}

arms=(control)
train_arm control "$rehearsal_weights"
IFS=',' read -r -a dose_weight_array <<< "$dose_weights"
for weight in "${dose_weight_array[@]}"; do
    arm="correction-weight-$weight"
    arms+=("$arm")
    train_arm "$arm" "$rehearsal_weights,$weight" --dataset "$on_food_corrections"
done

IFS=',' read -r -a evaluation_seed_array <<< "$evaluation_seeds"
for arm in "${arms[@]}"; do
    merge_args=()
    for seed in "${evaluation_seed_array[@]}"; do
        shard="$output_root/$arm/feeding-evaluation-shards/seed-$seed.json"
        "$feeding_evaluation_bin" \
            --config "$config" \
            --behavior-clone "$output_root/$arm/behavior-clone" \
            --seeds "$seed" \
            --output "$shard"
        merge_args+=(--input "$shard")
    done
    "$feeding_evaluation_merge_bin" \
        "${merge_args[@]}" \
        --output "$output_root/$arm/feeding-evaluation.json"
    "$feeding_corrections_bin" \
        --config "$config" \
        --behavior-clone "$output_root/$arm/behavior-clone" \
        --stage on-food \
        --seeds "$margin_probe_seed" \
        --max-samples 2048 \
        --minimum-policy-agreement-rate 0.8 \
        --output "$output_root/$arm/on-food-margin-probe"
done
