#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "usage: $0 OUTPUT_ROOT PARENT_CLONE REHEARSAL_DATASET_ROOT ON_FOOD_CORRECTIONS ADJACENT_FOOD_CORRECTIONS [ADDITIONAL_DATASETS]" >&2
    echo "ADDITIONAL_DATASETS is an optional comma-delimited list appended to both arms." >&2
    exit 2
fi

output_root=$1
parent_clone=$2
rehearsal_root=$3
on_food_corrections=$4
adjacent_food_corrections=$5
additional_datasets=${6:-}

behavior_clone_bin=${BLOB_BEHAVIOR_CLONE_BIN:-target/release/behavior-clone}
feeding_evaluation_bin=${BLOB_FEEDING_EVALUATION_BIN:-target/release/feeding-evaluation}
feeding_evaluation_merge_bin=${BLOB_FEEDING_EVALUATION_MERGE_BIN:-target/release/feeding-evaluation-merge}
config=${BLOB_COMBAT_WARM_START_256_CONFIG:-blob_rl/config/combat_warm_start_256.toml}
evaluation_seeds=${BLOB_FEEDING_CORRECTION_EVALUATION_SEEDS:-960000101,960000202,960000303,960000404,960000505,960000606,960000707,960000808}
epochs=${BLOB_FEEDING_CORRECTION_EPOCHS:-1}
epoch_samples=${BLOB_FEEDING_CORRECTION_EPOCH_SAMPLES:-14806}
optimizer_steps_per_epoch=${BLOB_FEEDING_CORRECTION_OPTIMIZER_STEPS_PER_EPOCH:-64}
learning_rate=${BLOB_FEEDING_CORRECTION_LEARNING_RATE:-0.00001}
correction_weight=${BLOB_FEEDING_CORRECTION_WEIGHT:-1.25}
correction_scope=${BLOB_FEEDING_CORRECTION_SCOPE:-both}
training_seed=${BLOB_FEEDING_CORRECTION_TRAINING_SEED:-42}
additional_weights=${BLOB_FEEDING_CORRECTION_ADDITIONAL_WEIGHTS:-}
action_kind_expert_only=${BLOB_FEEDING_ACTION_KIND_EXPERT_ONLY:-}
foraging_adapter_only=${BLOB_FEEDING_FORAGING_ADAPTER_ONLY:-0}
context_adapter_only=${BLOB_FEEDING_CONTEXT_ADAPTER_ONLY:-}
context_slot_adapter_only=${BLOB_FEEDING_CONTEXT_SLOT_ADAPTER_ONLY:-}
action_kind_margin=${BLOB_FEEDING_ACTION_KIND_MARGIN:-}
action_kind_margin_loss_weight=${BLOB_FEEDING_ACTION_KIND_MARGIN_LOSS_WEIGHT:-0}
target_query_head_only=${BLOB_FEEDING_TARGET_QUERY_HEAD_ONLY:-0}
effort_head_only=${BLOB_FEEDING_EFFORT_HEAD_ONLY:-0}

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

if [[ -n "$additional_datasets" ]]; then
    if [[ -z "$additional_weights" ]]; then
        echo "BLOB_FEEDING_CORRECTION_ADDITIONAL_WEIGHTS is required with ADDITIONAL_DATASETS" >&2
        exit 2
    fi
    IFS=',' read -r -a additional_dataset_array <<< "$additional_datasets"
    IFS=',' read -r -a additional_weight_array <<< "$additional_weights"
    if [[ ${#additional_dataset_array[@]} -ne ${#additional_weight_array[@]} ]]; then
        echo "additional dataset and weight counts differ" >&2
        exit 2
    fi
    for dataset in "${additional_dataset_array[@]}"; do
        if [[ ! -f "$dataset/manifest.json" || ! -f "$dataset/samples.mpk" ]]; then
            echo "missing additional rehearsal dataset $dataset" >&2
            exit 2
        fi
        dataset_args+=(--dataset "$dataset")
    done
    rehearsal_weights="$rehearsal_weights,$additional_weights"
fi

correction_datasets=("$on_food_corrections")
case "$correction_scope" in
    both)
        correction_datasets+=("$adjacent_food_corrections")
        ;;
    on-food)
        ;;
    *)
        echo "BLOB_FEEDING_CORRECTION_SCOPE must be both or on-food" >&2
        exit 2
        ;;
esac
for dataset in "${correction_datasets[@]}"; do
    if [[ ! -f "$dataset/manifest.json" || ! -f "$dataset/samples.mpk" ]]; then
        echo "missing correction dataset $dataset" >&2
        exit 2
    fi
done

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

train_arm control "$rehearsal_weights"
correction_args=()
correction_weights=$rehearsal_weights
for dataset in "${correction_datasets[@]}"; do
    correction_args+=(--dataset "$dataset")
    correction_weights="$correction_weights,$correction_weight"
done
train_arm correction "$correction_weights" "${correction_args[@]}"

IFS=',' read -r -a evaluation_seed_array <<< "$evaluation_seeds"
for arm in control correction; do
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
done
