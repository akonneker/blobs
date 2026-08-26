#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 OUTPUT_DIRECTORY" >&2
    exit 2
fi

output_root=$1
trainer=${BLOB_DEMONSTRATIONS_BIN:-target/release/demonstrations}
base=blob_rl/config/objective_large_draw_smoke.toml
seeds=5101,5201,5301,5401,5501,5601,5701,5801
samples=${BLOB_DEMONSTRATION_SAMPLES:-100000}

generate() {
    local field=$1
    local cells=$2
    local loose_sources=$3
    local plants=$4
    "$trainer" \
        --config "$base" \
        --teacher colony \
        --seeds "$seeds" \
        --max-samples "$samples" \
        --world-size "$field" \
        --cells-per-team "$cells" \
        --num-scattered-energy "$loose_sources" \
        --num-plants "$plants" \
        --output "$output_root/field-$field"
}

generate 32 16 128 16
generate 64 64 512 64
generate 128 256 2048 256
generate 256 1024 8192 1024
