#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
    echo "usage: $0 CONFIG PARENT PILOT_BRANCH PILOT_VALUE CONFIRM_BRANCH CONFIRM_VALUE OUTPUT_ROOT" >&2
    exit 2
fi

config=$1
parent=$2
pilot_branch=$3
pilot_value=$4
confirm_branch=$5
confirm_value=$6
output_root=$7

corrections_bin=${BLOB_COUNTERFACTUAL_EFFORT_CORRECTIONS_BIN:-target/release/counterfactual-effort-corrections}
clone_bin=${BLOB_BEHAVIOR_CLONE_BIN:-target/release/behavior-clone}
steps=${BLOB_EFFORT_AB_STEPS:-10}
learning_rate=${BLOB_EFFORT_AB_LEARNING_RATE:-0.005}
training_seed=${BLOB_EFFORT_AB_TRAINING_SEED:-1422000001}

if [[ -e "$output_root" ]]; then
    echo "refusing to replace output root $output_root" >&2
    exit 2
fi
mkdir -p "$output_root/datasets"

for shard in pilot confirm; do
    if [[ "$shard" == pilot ]]; then
        branch=$pilot_branch
        value=$pilot_value
    else
        branch=$confirm_branch
        value=$confirm_value
    fi
    for arm in control correction; do
        "$corrections_bin" \
            --counterfactual "$branch" \
            --value "$value" \
            --perspective conservative \
            --horizon-quanta 4096 \
            --arm "$arm" \
            --output "$output_root/datasets/$shard-$arm"
    done
done

train_arm() {
    local arm=$1
    "$clone_bin" \
        --config "$config" \
        --initial-behavior-clone "$parent" \
        --dataset "$output_root/datasets/pilot-$arm" \
        --dataset "$output_root/datasets/confirm-$arm" \
        --output "$output_root/$arm" \
        --epochs "$steps" \
        --minibatch-size 8 \
        --learning-rate "$learning_rate" \
        --seed "$training_seed" \
        --validation-fraction 0.25 \
        --dataset-sampling balanced \
        --dataset-weight 1,1 \
        --epoch-samples 8 \
        --optimizer-steps-per-epoch 1 \
        --effort-head-only \
        --recurrent-unroll-steps 1 \
        --action-balancing none
}

train_arm control
train_arm correction

echo "published paired effort-head A/B under $output_root"
