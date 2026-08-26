# Dockerized remote training

`Dockerfile.training` packages the RL trainer, rules-sweep planner, and bounded
sweep executor as two explicit targets:

- `cpu` (the default) uses Burn's portable `ndarray` backend and requires no
  GPU driver, Vulkan device, or display server.
- `gpu` uses Burn/CubeCL's WGPU Vulkan backend. It has an explicit NVIDIA
  Container Toolkit and host-driver contract and fails its preflight when no
  Vulkan device is available; it does not silently fall back to CPU.

The container has a deliberately small filesystem contract:

- `/config` contains read-only job configuration supplied by the operator.
- `/output` contains every durable metric, checkpoint, telemetry file, log,
  and final model.
- `/tmp` is disposable process-local scratch space.

Training artifacts retain the normal immutable checkpoint hashes and exact
update-boundary resume state. Artifact schema 8 records the numerical training
backend. Exact `--resume` is deliberately backend-bound; policy-only
`--load-model` remains the explicit path for moving weights between CPU and
GPU. Containerization does not change simulation, Mind isolation, evaluation,
or artifact-verification semantics.

## Build and identify the image

`Cargo.lock` is committed and the image build uses `cargo build --locked`. Pass
the source revision so immutable checkpoint metadata and the OCI image label
identify the code that produced a run:

```sh
docker build \
  --file Dockerfile.training \
  --target cpu \
  --build-arg BLOB_CODE_REVISION="$(git rev-parse HEAD)" \
  --tag blobs-training:local \
  .
```

Build scientific runs from a clean checkout. `BLOB_CODE_REVISION` is supplied
by the build operator and cannot prove that an uncommitted working tree matches
the named commit. Retain the resulting image digest with the experiment; pin
the Rust base image by digest as well if bit-for-bit image reconstruction is a
deployment requirement.

Pass the deployed digest to `rules-sweep-run` as
`--trainer-container-digest sha256:...`. The executor stores it in the immutable
execution contract alongside its independently calculated trainer-binary hash
and rejects retries that supply a different image or executable.

The image also contains the `demonstrations` and `behavior-clone` commands.
Mount immutable datasets and pretrained artifacts under `/output` (or another
explicit volume), then bind the printed behavior-cloning metadata digest in
the PPO training config's `[initial_policy]` table. This avoids relying on a
mutable model path when moving the curriculum between hosts.

For a registry-backed remote host, tag and push the exact image, then deploy by
digest when practical:

```sh
docker tag blobs-training:local registry.example.com/blobs/training:REVISION
docker push registry.example.com/blobs/training:REVISION
docker pull registry.example.com/blobs/training:REVISION
docker image inspect registry.example.com/blobs/training:REVISION \
  --format '{{index .RepoDigests 0}}'
```

The build is native to the builder's architecture. Use `docker buildx build`
with `--platform linux/amd64,linux/arm64` when a multi-architecture registry
image is required. The GPU target must match the architecture and driver stack
of its deployment host; it is not a portable multi-architecture promise.

## Build and verify the NVIDIA WGPU target

The GPU host needs an NVIDIA driver with Vulkan support, Docker Engine, and the
NVIDIA Container Toolkit configured for Docker. The image contains userspace
Vulkan dispatch libraries but no kernel driver and no CUDA toolkit. Build it
separately so backend selection is immutable and visible in the image tag:

```sh
docker build \
  --file Dockerfile.training \
  --target gpu \
  --build-arg BLOB_CODE_REVISION="$(git rev-parse HEAD)" \
  --tag blobs-training:gpu-local \
  .
```

Before spending time on a training job, verify the same constrained runtime
contract used by the smoke test:

```sh
mkdir -p training-output/gpu-preflight
docker run --rm \
  --gpus all \
  --read-only \
  --security-opt no-new-privileges:true \
  --cap-drop ALL \
  --user "$(id -u):$(id -g)" \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --tmpfs /tmp:size=1g,mode=1777 \
  --volume "$PWD/training-output/gpu-preflight:/output" \
  blobs-training:gpu-local gpu-info
```

The summary must name the intended physical GPU. `graphics` is included in
the NVIDIA driver capabilities because WGPU uses Vulkan; this is a headless
compute job and does not require an X server. CubeCL creates and autotunes
kernels on first use, so `/tmp` must be writable and the first short update is
not a representative steady-state benchmark. `/cache` stores device-namespaced,
checksum-validated autotune choices; Docker supplies an anonymous cache volume
unless the operator mounts a durable one. The Compose service uses the named
`blobs-gpu-kernel-cache` volume so later jobs on the same host can reuse them.
Do not copy this host/device-specific cache into scientific artifacts.

The trainer batches rollout policy inference across all environments. Set
`--num-envs` high enough to feed the device and use `--minibatch-size` in the
hundreds or thousands for PPO updates; the CPU-oriented default of 64 produces
many undersized GPU dispatches. Treat both as learning-dynamics parameters and
validate tuned values rather than changing them only for a favorable
throughput number.

## Launch one CPU training job

Create a host-owned output directory before starting the container. Matching
the container uid/gid to the directory owner avoids root-owned artifacts on a
Linux server:

```sh
mkdir -p training-output/run-42
docker run --detach \
  --name blobs-train-42 \
  --init \
  --cpus 8 \
  --memory 16g \
  --user "$(id -u):$(id -g)" \
  --env RAYON_NUM_THREADS=8 \
  --volume "$PWD/blob_rl/config:/config:ro" \
  --volume "$PWD/training-output/run-42:/output" \
  blobs-training:local \
  train \
  --config /config/default.toml \
  --seed 42 \
  --checkpoint-dir /output/checkpoints \
  --save-model /output/final-model
```

The detached container survives an SSH disconnect. Use normal Docker lifecycle
commands to inspect it:

```sh
docker logs --follow blobs-train-42
docker wait blobs-train-42
docker inspect blobs-train-42 --format '{{.State.ExitCode}}'
```

A zero exit code means the terminal model, metrics, final held-out evaluation,
and telemetry summary were published. Preserve the whole output directory,
not just the final model: checkpoint metadata binds the model, optimizer,
continuation state, complete expanded config, ruleset hash, and source revision.

The trainer publishes checkpoints only at clean update boundaries. Abrupt host
loss or `docker kill` can discard work since the newest checkpoint but cannot
partially publish a checkpoint. `docker stop` currently terminates the trainer;
it does not request an extra checkpoint before exit.

## Run a non-learning viability matchup

The CPU image also contains the baseline-versus-baseline runner. It does not
initialize a Burn model or optimizer:

```sh
docker run --rm --read-only \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:size=1g,mode=1777 \
  --volume "$PWD/blob_rl/config:/config:ro" \
  --volume "$PWD/training-output:/output" \
  blobs-training:local \
  viability \
  --config /config/default.toml \
  --candidate forager --opponent forager \
  --seeds 101,202,303 \
  --output /output/forager-vs-forager.json
```

The report is immutable and binds the profiles, seed suite, full initial
scenario, semantic and compiled rules hashes, telemetry cadence, and code
revision when the image was built with `BLOB_CODE_REVISION`.

After publishing a sweep plan into a durable mounted directory, the same image
can run the complete paired non-learning matrix with bounded CPU concurrency:

```sh
docker run --rm --read-only \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:size=1g,mode=1777 \
  --volume "$PWD/training-output:/output" \
  blobs-training:local \
  viability-matrix /output/sweeps/viability-v1/manifest.json \
  --candidates forager,random \
  --opponents wait,forager,aggressive \
  --baseline-variant baseline \
  --max-parallel 8 \
  --output /output/viability-v1-matrix.json
```

The manifest stores absolute paths, so planning and matrix execution must see
the sweep directory at the same absolute location. In containers, publish the
plan using its final `/output/...` location rather than copying a host-planned
manifest into a differently mounted path.

Evaluate the matrix policy in the same read-only runner:

```sh
docker run --rm --read-only \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:size=256m,mode=1777 \
  --volume "$PWD/training-output:/output" \
  --volume "$PWD/blob_rl/config:/config:ro" \
  blobs-training:local \
  viability-gate /output/viability-v1-matrix.json \
  --gates /config/viability_gates.toml \
  --output /output/viability-v1-decision.json
```

Exit status 0 means every configured gate passed. Status 2 means the decision
report was published but one or more scientific checks failed; other nonzero
statuses indicate invalid input or execution failure.

Launch the training executor from the same image only after the gate passes:

```sh
docker run --rm --read-only \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:size=1g,mode=1777 \
  --volume "$PWD/training-output:/output" \
  --volume "$PWD/blob_rl/config:/config:ro" \
  blobs-training:local \
  sweep-run /output/sweeps/viability-v1/manifest.json \
  --require-viability-gate /output/viability-v1-decision.json \
  --viability-matrix /output/viability-v1-matrix.json \
  --viability-gates /config/viability_gates.toml \
  --max-parallel 4 --threads-per-run 4
```

The executor freshly reproduces the decision before acquiring its sweep lock.
It launches nothing when any artifact changes, the gate fails, or the decision
belongs to a different sweep. Keep the planning, matrix, gate, and execution
steps on stable absolute mount paths and use the same image build for the gate
and executor.

The same sequence can be run as one restart-safe container command. The sweep
spec should use `/output/...` as its `output_dir`, and its base config must be
reachable at the same absolute path on every restart:

```sh
docker run --rm --read-only \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:size=1g,mode=1777 \
  --volume "$PWD/training-jobs:/jobs:ro" \
  --volume "$PWD/blob_rl/config:/config:ro" \
  --volume "$PWD/training-output:/output" \
  blobs-training:local \
  viability-preflight /jobs/viability.toml \
  --candidates forager,random \
  --opponents wait,forager,aggressive \
  --gates /config/viability_gates.toml \
  --baseline-variant baseline \
  --matrix-max-parallel 8
```

To continue into CPU training, append `--execute-training`,
`--training-max-parallel`, and `--threads-per-run`. Use the GPU image for the
same combined command when training with WGPU. Restarting the identical command
reuses verified plan, matrix, and decision artifacts; it never replaces a
stage whose inputs changed.

## Launch one GPU training job

Use one durable cache volume per compatible GPU host and keep it separate from
the run output:

```sh
mkdir -p training-output/gpu-42
docker volume create blobs-gpu-kernel-cache
docker run --detach \
  --name blobs-gpu-42 \
  --init \
  --gpus all \
  --read-only \
  --security-opt no-new-privileges:true \
  --cap-drop ALL \
  --user "$(id -u):$(id -g)" \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --env RAYON_NUM_THREADS=16 \
  --tmpfs /tmp:size=2g,mode=1777 \
  --volume blobs-gpu-kernel-cache:/cache \
  --volume "$PWD/blob_rl/config:/config:ro" \
  --volume "$PWD/training-output/gpu-42:/output" \
  blobs-training:gpu-local \
  train \
  --config /config/default.toml \
  --minibatch-size 1024 \
  --checkpoint-dir /output/checkpoints \
  --save-model /output/final-model
```

The 1,024-sample minibatch is a starting point for throughput experiments, not
a universal PPO recommendation. Record it in the expanded config and compare
learning quality as well as wall time.

## Resume on the same or another host

Copy or remount the complete output directory and start the same image. Select
the newest fully published checkpoint directory, increase the action target,
and keep training dynamics unchanged:

```sh
docker run --detach \
  --name blobs-train-42-resumed \
  --init \
  --cpus 8 \
  --memory 16g \
  --user "$(id -u):$(id -g)" \
  --env RAYON_NUM_THREADS=8 \
  --volume "$PWD/blob_rl/config:/config:ro" \
  --volume "$PWD/training-output/run-42:/output" \
  blobs-training:local \
  train \
  --config /config/default.toml \
  --total-timesteps 2000000 \
  --checkpoint-dir /output/checkpoints \
  --resume /output/checkpoints/checkpoint-00000100 \
  --save-model /output/final-model
```

Resume verifies every checkpoint digest and rejects incompatible configuration
changes, including changing the recorded CPU/WGPU numerical backend.
`--load-model` is intentionally different: it imports policy weights without
restoring exact optimizer, environment, RNG, league, or telemetry state, and
is the supported way to initialize one backend from weights trained by the
other.

## Docker Compose

The included Compose service uses an unprivileged uid, drops Linux
capabilities, enables `no-new-privileges`, and makes the container root
filesystem read-only. Export the host uid/gid and source revision before the
first launch:

```sh
mkdir -p training-output
export TRAINING_UID="$(id -u)"
export TRAINING_GID="$(id -g)"
export BLOB_CODE_REVISION="$(git rev-parse HEAD)"
export TRAINING_THREADS=8
docker compose --file compose.training.yaml up --build --detach trainer
docker compose --file compose.training.yaml logs --follow trainer
```

`TRAINING_CONFIG` selects a filename under `blob_rl/config`,
`TRAINING_OUTPUT_DIR` changes the host output mount, and `TRAINING_IMAGE`
selects a prebuilt registry image. Compose thread settings constrain Rayon but
do not enforce a CPU or memory quota; add deployment-specific limits or use
`docker run --cpus/--memory` on shared systems.

The GPU service has the same filesystem and security policy and adds the
NVIDIA/Vulkan device contract:

```sh
mkdir -p training-output-gpu
export TRAINING_UID="$(id -u)"
export TRAINING_GID="$(id -g)"
export BLOB_CODE_REVISION="$(git rev-parse HEAD)"
docker compose --file compose.training.gpu.yaml up --build --detach trainer-gpu
docker compose --file compose.training.gpu.yaml logs --follow trainer-gpu
```

Run one trainer per GPU unless devices are explicitly partitioned with
`NVIDIA_VISIBLE_DEVICES`. Multiple WGPU processes on one device duplicate
model and optimizer state, repeat kernel setup, and contend unpredictably.

## Rules sweeps

The entrypoint exposes `sweep-plan` and `sweep-run`. A container-side sweep
spec should use paths visible inside the container, for example:

```toml
base_config = "/config/default.toml"
output_dir = "/output/sweeps/viability-v1"
seeds = [101, 202, 303]

[[variants]]
name = "baseline"

[[variants]]
name = "lower-metabolism"
[variants.rules]
metabolism_rate_numerator = 1
metabolism_rate_denominator = 2048
```

Mount that spec and publish the immutable plan:

```sh
docker run --rm --init \
  --user "$(id -u):$(id -g)" \
  --volume "$PWD/blob_rl/config:/config:ro" \
  --volume "$PWD/jobs:/jobs:ro" \
  --volume "$PWD/training-output:/output" \
  blobs-training:local \
  sweep-plan /jobs/viability.toml
```

Then launch the bounded CPU sweep executor against the same output mount:

```sh
docker run --rm --init \
  --cpus 16 \
  --memory 32g \
  --user "$(id -u):$(id -g)" \
  --volume "$PWD/training-output:/output" \
  blobs-training:local \
  sweep-run /output/sweeps/viability-v1/manifest.json \
  --max-parallel 2 \
  --threads-per-run 8
```

Keep `max_parallel * threads_per_run` at or below the allocated CPU count.
Each run owns separate logs and immutable checkpoints. Rerunning the executor
recovers interrupted runs from their newest verified checkpoint; failed runs
require the explicit `--retry-failed` option.

The GPU image exposes the same sweep commands, but `--max-parallel 1` is the
safe default for one assigned GPU. Use process-level device partitioning for a
multi-GPU sweep rather than allowing several trainers to select the same
default adapter.

## Smoke test

The CPU smoke script builds the default backend, performs one tiny PPO update,
and checks that metrics and telemetry artifacts were written:

```sh
scripts/docker-training-smoke.sh
```

Set `TRAINING_IMAGE`, `TRAINING_THREADS`, or `TRAINING_OUTPUT_DIR` to override
its defaults.

On a configured NVIDIA host, the GPU smoke test first proves Vulkan device
visibility under the hardened runtime and then performs a real WGPU PPO update:

```sh
scripts/docker-training-gpu-smoke.sh
```

Override `TRAINING_GPU_IMAGE`, `TRAINING_THREADS`, or `TRAINING_OUTPUT_DIR` as
needed. A tiny smoke run validates integration, not GPU throughput; use a
larger rollout and several environments when measuring steady state.

On the initial `fitty` RTX 5090 validation, a warm-cache WGPU update of the
small default 128→64 policy reached roughly 796 decisions/second, while the
matching NdArray CPU job reached roughly 3,558. The GPU target is therefore an
opt-in development path, not the current throughput default. See
[Performance baseline](performance-baseline.md#containerized-wgpu-training-baseline)
for the exact fixture and cold-cache results.
