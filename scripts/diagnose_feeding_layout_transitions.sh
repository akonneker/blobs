#!/usr/bin/env bash
set -u -o pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 CONFIG BEHAVIOR_CLONE OUTPUT_ROOT" >&2
    exit 2
fi

config=$1
behavior_clone=$2
output_root=$3

diagnostics_bin=${BLOB_FEEDING_TRANSITION_DIAGNOSTICS_BIN:-target/release/feeding-transition-diagnostics}
container_image=${BLOB_FEEDING_TRANSITION_CONTAINER_IMAGE:-}
layouts=${BLOB_FEEDING_TRANSITION_LAYOUTS:-line,checkerboard,ring,loose-random,random}
seeds=${BLOB_FEEDING_TRANSITION_SEEDS:-1432000101,1432000202}
policy_randomness=${BLOB_FEEDING_TRANSITION_RANDOMNESS:-canonical}
max_parallel=${BLOB_FEEDING_TRANSITION_MAX_PARALLEL:-1}

if [[ "$policy_randomness" != zero && "$policy_randomness" != canonical ]]; then
    echo "BLOB_FEEDING_TRANSITION_RANDOMNESS must be zero or canonical" >&2
    exit 2
fi
if ! [[ "$max_parallel" =~ ^[1-9][0-9]*$ ]]; then
    echo "BLOB_FEEDING_TRANSITION_MAX_PARALLEL must be a positive integer" >&2
    exit 2
fi
IFS=',' read -r -a layout_array <<< "$layouts"
if [[ ${#layout_array[@]} -eq 0 ]]; then
    echo "feeding transition diagnostics require at least one layout" >&2
    exit 2
fi
mkdir -p "$output_root"

container_args=()
if [[ -n "$container_image" ]]; then
    for path in "$config" "$behavior_clone" "$output_root"; do
        if [[ "$path" != /* ]]; then
            echo "container diagnostics require absolute paths: $path" >&2
            exit 2
        fi
    done
    container_user=${BLOB_FEEDING_TRANSITION_CONTAINER_USER:-$(id -u):$(id -g)}
    container_args=(
        --rm
        --init
        --read-only
        --security-opt no-new-privileges:true
        --cap-drop ALL
        --user "$container_user"
        --tmpfs /tmp:size=1g,mode=1777
        --volume "$config:$config:ro"
        --volume "$behavior_clone:$behavior_clone:ro"
        --volume "$output_root:$output_root"
        --volume "$output_root:/output"
    )
fi

run_diagnostics() {
    if [[ -n "$container_image" ]]; then
        docker run "${container_args[@]}" "$container_image" feeding-transition-diagnostics "$@"
    else
        "$diagnostics_bin" "$@"
    fi
}

total_layouts=${#layout_array[@]}
completed_layouts=0
failed_layouts=0
launched_layouts=0
pending_pids=()
pending_labels=()
pending_logs=()
pending_starts=()
pending_ordinals=()

reap_first_layout() {
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
        sed "s|^|[$ordinal/$total_layouts] |" "$log"
    fi
    if [[ $status -eq 0 ]]; then
        completed_layouts=$((completed_layouts + 1))
        echo "[$ordinal/$total_layouts] $label complete in $((SECONDS - started))s"
    else
        failed_layouts=$((failed_layouts + 1))
        echo "[$ordinal/$total_layouts] $label FAILED after $((SECONDS - started))s" >&2
    fi
    pending_pids=("${pending_pids[@]:1}")
    pending_labels=("${pending_labels[@]:1}")
    pending_logs=("${pending_logs[@]:1}")
    pending_starts=("${pending_starts[@]:1}")
    pending_ordinals=("${pending_ordinals[@]:1}")
}

for layout in "${layout_array[@]}"; do
    output="$output_root/$layout.json"
    log="$output_root/$layout.log"
    launched_layouts=$((launched_layouts + 1))
    ordinal=$launched_layouts
    label="$layout transition diagnostics"
    echo "[$ordinal/$total_layouts] $label started"
    started=$SECONDS
    run_diagnostics \
        --config "$config" \
        --behavior-clone "$behavior_clone" \
        --layout "$layout" \
        --seeds "$seeds" \
        --policy-randomness "$policy_randomness" \
        --resume \
        --output "$output" >"$log" 2>&1 &
    pending_pids+=("$!")
    pending_labels+=("$label")
    pending_logs+=("$log")
    pending_starts+=("$started")
    pending_ordinals+=("$ordinal")
    if [[ ${#pending_pids[@]} -ge $max_parallel ]]; then
        reap_first_layout
    fi
done

while [[ ${#pending_pids[@]} -gt 0 ]]; do
    reap_first_layout
done

if [[ $failed_layouts -ne 0 ]]; then
    echo "$failed_layouts layout diagnostic(s) failed; rerun the same command to resume verified successes" >&2
    exit 1
fi

echo "feeding transition diagnostics complete: $completed_layouts/$total_layouts layouts"
