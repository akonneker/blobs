#!/usr/bin/env bash
set -u -o pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 CONFIG CONTROL_CLONE CORRECTION_CLONE OUTPUT_ROOT" >&2
    exit 2
fi

config=$1
control_clone=$2
correction_clone=$3
output_root=$4

feeding_evaluation_bin=${BLOB_FEEDING_EVALUATION_BIN:-target/release/feeding-evaluation}
feeding_merge_bin=${BLOB_FEEDING_EVALUATION_MERGE_BIN:-target/release/feeding-evaluation-merge}
container_image=${BLOB_EFFORT_AB_CONTAINER_IMAGE:-}
evaluation_seeds=${BLOB_EFFORT_AB_EVALUATION_SEEDS:-1423000101,1423000202,1423000303,1423000404,1423000505,1423000606,1423000707,1423000808}
require_pass=${BLOB_EFFORT_AB_REQUIRE_PASS:-0}
max_parallel=${BLOB_EFFORT_AB_MAX_PARALLEL:-1}

if [[ "$require_pass" != 0 && "$require_pass" != 1 ]]; then
    echo "BLOB_EFFORT_AB_REQUIRE_PASS must be 0 or 1" >&2
    exit 2
fi
if ! [[ "$max_parallel" =~ ^[1-9][0-9]*$ ]]; then
    echo "BLOB_EFFORT_AB_MAX_PARALLEL must be a positive integer" >&2
    exit 2
fi
IFS=',' read -r -a seed_array <<< "$evaluation_seeds"
if [[ ${#seed_array[@]} -eq 0 ]]; then
    echo "at least one evaluation seed is required" >&2
    exit 2
fi
mkdir -p "$output_root"

container_args=()
if [[ -n "$container_image" ]]; then
    for path in "$config" "$control_clone" "$correction_clone" "$output_root"; do
        if [[ "$path" != /* ]]; then
            echo "container qualification requires absolute paths: $path" >&2
            exit 2
        fi
    done
    container_user=${BLOB_EFFORT_AB_CONTAINER_USER:-$(id -u):$(id -g)}
    container_args=(
        --rm
        --init
        --read-only
        --security-opt no-new-privileges:true
        --cap-drop ALL
        --user "$container_user"
        --tmpfs /tmp:size=1g,mode=1777
        --volume "$config:$config:ro"
        --volume "$control_clone:$control_clone:ro"
        --volume "$correction_clone:$correction_clone:ro"
        --volume "$output_root:$output_root"
        --volume "$output_root:/output"
    )
fi

run_evaluation() {
    if [[ -n "$container_image" ]]; then
        docker run "${container_args[@]}" "$container_image" feeding-evaluation "$@"
    else
        "$feeding_evaluation_bin" "$@"
    fi
}

run_merge() {
    if [[ -n "$container_image" ]]; then
        docker run "${container_args[@]}" "$container_image" feeding-evaluation-merge "$@"
    else
        "$feeding_merge_bin" "$@"
    fi
}

arms=(control correction)
clones=("$control_clone" "$correction_clone")
total_shards=$((${#arms[@]} * ${#seed_array[@]}))
completed_shards=0
failed_shards=0
launched_shards=0
pending_pids=()
pending_labels=()
pending_logs=()
pending_starts=()
pending_ordinals=()

reap_first_shard() {
    local pid=${pending_pids[0]}
    local label=${pending_labels[0]}
    local log=${pending_logs[0]}
    local started=${pending_starts[0]}
    local ordinal=${pending_ordinals[0]}
    local status
    if wait "$pid"; then
        status=0
    else
        status=$?
    fi
    if [[ -s "$log" ]]; then
        sed "s|^|[$ordinal/$total_shards] |" "$log"
    fi
    if [[ $status -eq 0 ]]; then
        completed_shards=$((completed_shards + 1))
        echo "[$ordinal/$total_shards] $label complete in $((SECONDS - started))s"
    else
        failed_shards=$((failed_shards + 1))
        echo "[$ordinal/$total_shards] $label FAILED after $((SECONDS - started))s" >&2
    fi
    pending_pids=("${pending_pids[@]:1}")
    pending_labels=("${pending_labels[@]:1}")
    pending_logs=("${pending_logs[@]:1}")
    pending_starts=("${pending_starts[@]:1}")
    pending_ordinals=("${pending_ordinals[@]:1}")
}

for arm_index in "${!arms[@]}"; do
    arm=${arms[$arm_index]}
    clone=${clones[$arm_index]}
    mkdir -p "$output_root/$arm/shards"
    for seed in "${seed_array[@]}"; do
        shard="$output_root/$arm/shards/seed-$seed.json"
        log="$output_root/$arm/shards/seed-$seed.log"
        launched_shards=$((launched_shards + 1))
        ordinal=$launched_shards
        label="$arm seed $seed"
        echo "[$ordinal/$total_shards] $label started"
        started=$SECONDS
        run_evaluation \
            --config "$config" \
            --behavior-clone "$clone" \
            --seeds "$seed" \
            --resume \
            --output "$shard" >"$log" 2>&1 &
        pending_pids+=("$!")
        pending_labels+=("$label")
        pending_logs+=("$log")
        pending_starts+=("$started")
        pending_ordinals+=("$ordinal")
        if [[ ${#pending_pids[@]} -ge $max_parallel ]]; then
            reap_first_shard
        fi
    done
done

while [[ ${#pending_pids[@]} -gt 0 ]]; do
    reap_first_shard
done

if [[ $failed_shards -ne 0 ]]; then
    echo "$failed_shards shard(s) failed; rerun the same command to resume verified successes" >&2
    exit 1
fi

for arm_index in "${!arms[@]}"; do
    arm=${arms[$arm_index]}
    clone=${clones[$arm_index]}
    aggregate="$output_root/$arm/feeding-evaluation.json"
    if [[ -e "$aggregate" ]]; then
        if [[ "$require_pass" == 1 ]]; then
            run_evaluation \
                --config "$config" \
                --behavior-clone "$clone" \
                --seeds "$evaluation_seeds" \
                --resume \
                --require-pass \
                --output "$aggregate"
        else
            run_evaluation \
                --config "$config" \
                --behavior-clone "$clone" \
                --seeds "$evaluation_seeds" \
                --resume \
                --output "$aggregate"
        fi
        continue
    fi
    merge_args=()
    for seed in "${seed_array[@]}"; do
        merge_args+=(--input "$output_root/$arm/shards/seed-$seed.json")
    done
    if [[ "$require_pass" == 1 ]]; then
        run_merge \
            "${merge_args[@]}" \
            --require-pass \
            --output "$aggregate"
    else
        run_merge \
            "${merge_args[@]}" \
            --output "$aggregate"
    fi
done

echo "paired feeding qualification complete: $completed_shards/$total_shards shards"
