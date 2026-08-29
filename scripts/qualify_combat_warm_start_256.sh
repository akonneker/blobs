#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 OUTPUT_DIRECTORY [BEHAVIOR_CLONE]" >&2
    exit 2
fi

output_root=$1
behavior_clone=${2:-training-output/warmstarts/skirmish-schema35/behavior-clone}
feeding_evaluation_bin=${BLOB_FEEDING_EVALUATION_BIN:-target/release/feeding-evaluation}
feeding_evaluation_merge_bin=${BLOB_FEEDING_EVALUATION_MERGE_BIN:-target/release/feeding-evaluation-merge}
feeding_layout_evaluation_bin=${BLOB_FEEDING_LAYOUT_EVALUATION_BIN:-target/release/feeding-layout-evaluation}
feeding_layout_evaluation_merge_bin=${BLOB_FEEDING_LAYOUT_EVALUATION_MERGE_BIN:-target/release/feeding-layout-evaluation-merge}
contact_evaluation_bin=${BLOB_CONTACT_EVALUATION_BIN:-target/release/contact-evaluation}
micro_combat_evaluation_bin=${BLOB_MICRO_COMBAT_EVALUATION_BIN:-target/release/micro-combat-evaluation}
config=${BLOB_COMBAT_WARM_START_256_CONFIG:-blob_rl/config/combat_warm_start_256.toml}
scenarios=${BLOB_MICRO_COMBAT_SCENARIOS:-blob_rl/config/micro_combat_scenarios.toml}
feeding_seeds=${BLOB_FEEDING_EVALUATION_SEEDS:-900000101,900000201,900000301,900000401,900000501,900000601,900000701,900000801}
feeding_layout_seeds=${BLOB_FEEDING_LAYOUT_EVALUATION_SEEDS:-930000101,930000202}
feeding_layouts=${BLOB_FEEDING_LAYOUTS:-line,checkerboard,ring,loose-random,random}
contact_seeds=${BLOB_CONTACT_EVALUATION_SEEDS:-910000101,910000201,910000301,910000401}
micro_seeds=${BLOB_MICRO_COMBAT_EVALUATION_SEEDS:-920000101,920000201,920000301,920000401,920000501,920000601,920000701,920000801}
min_micro_attacks=${BLOB_MIN_MICRO_ATTACKS_PER_EPISODE:-0.25}

mkdir -p "$output_root"

IFS=',' read -r -a feeding_seed_array <<< "$feeding_seeds"
feeding_merge_args=()
for seed in "${feeding_seed_array[@]}"; do
    shard="$output_root/feeding-evaluation-shards/seed-$seed.json"
    "$feeding_evaluation_bin" \
        --config "$config" \
        --behavior-clone "$behavior_clone" \
        --seeds "$seed" \
        --output "$shard"
    feeding_merge_args+=(--input "$shard")
done
"$feeding_evaluation_merge_bin" \
    "${feeding_merge_args[@]}" \
    --require-pass \
    --output "$output_root/feeding-evaluation.json"

IFS=',' read -r -a feeding_layout_array <<< "$feeding_layouts"
IFS=',' read -r -a feeding_layout_seed_array <<< "$feeding_layout_seeds"
feeding_layout_merge_args=()
for layout in "${feeding_layout_array[@]}"; do
    for seed in "${feeding_layout_seed_array[@]}"; do
        shard="$output_root/feeding-layout-evaluation-shards/$layout-seed-$seed.json"
        "$feeding_layout_evaluation_bin" \
            --config "$config" \
            --behavior-clone "$behavior_clone" \
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
    --behavior-clone "$behavior_clone" \
    --seeds "$contact_seeds" \
    --require-damage \
    --output "$output_root/contact-evaluation.json"

"$micro_combat_evaluation_bin" \
    --config "$config" \
    --scenarios "$scenarios" \
    --behavior-clone "$behavior_clone" \
    --seeds "$micro_seeds" \
    --require-attack-commitments-per-episode "$min_micro_attacks" \
    --output "$output_root/micro-combat-evaluation.json"
