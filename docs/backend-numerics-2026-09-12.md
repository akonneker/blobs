# CPU training reciprocal defect — 2026-09-12

A reproducible numerical defect in the previous NdArray dependency configuration
biased training gradients on this ARM Mac. The configured backend now disables
NdArray SIMD while retaining its multithreaded matrix operations. Regression tests
cover both the numerical primitive and the actual masked training loss.

## Reproduction and cause

With Burn 0.20.1 and macerator 0.2.10, a tensor of fewer than 32 elements computes
`1 / 1 = 1`. At length 32 the previous build computes **0.998046875**. The NdArray
reciprocal implementation switches to SIMD at that boundary; macerator's ARM
implementation uses `vrecpeq` without a refinement step. Burn's logarithm backward
pass multiplies by that reciprocal. The forward loss can therefore look correct
while its gradient is biased.

A 32-row masked log-softmax loss at the exact uniform teacher distribution
produces a singleton-row logit gradient of **−0.00006103515625**, where the analytic
gradient is zero. At the experiment's batch size of 128 the value is
−0.0000152587890625. In the real first four-destination training batch, the
zero-initialized output head has maximum gradient 0.0005134226 and moves by
0.0049044746 after one AdamW update. This is not stochastic label noise: the
teacher-marginal labels are already equal to the uniform initial prediction.

The isolated one-row test passed because it never entered the vectorized path.
The failures and relevant dependency sources are retained in
`training-output/backend-numerics-2026-09-12/`, including `before.log`, the prior
manifest/lockfile, and dependency source copies. No registry files were modified.

## Fix and regression coverage

`blob_rl/Cargo.toml` selects Burn's `std` and `autodiff` features explicitly. It
removes the unused `train` feature: this project implements its own training loop,
and `burn-train` unconditionally enables NdArray's defaults. Direct NdArray
feature selection retains `std` and `multi-threads`, excluding `simd`. The GPU
backend's default features are selected separately so they remain available.
This also removes unused learner-framework dependencies from the lockfile.

`blob_rl/tests/backend_numerics.rs` compares reciprocal results with scalar
arithmetic at lengths 1, 8, 31, 32, 33, 128 and 256. It checks masked cross-entropy
gradients against an independent f64 analytic calculation, with uniform and
nonuniform predictions at batch sizes 1, 31, 32, 33 and 128. Both tests fail in the
old configuration and pass after the change. The learned-set example also checks
that four real batches and optimizer updates leave the optimal uniform policy's
output head exactly zero. The full workspace's all-feature dependency graph has
no `burn-ndarray/simd` feature.

These tests validate the configured CPU path. They do not establish a defect in
GPU execution, and do not claim that every prior training failure had this cause.

## Evidence implications

Earlier ARM NdArray experiments using these dependencies and affected tensor
sizes require a corrected rerun before their training failures can be interpreted
as architecture limits. Their measured behavior and exact mathematical witnesses
remain historical facts: the independent-score impossibility argument, and
bit-identical collisions in specific fitted max encoders, do not depend on
claiming the optimizer was correct. The learned-set and relational reports are
annotated accordingly; all failed runs remain retained.

The [corrected relational comparison](target-relational-2026-09-12.md) uses the same
models, data, initializations and update budget as its preceding run. The
manifest and lockfile change the training build's existing source fingerprint;
old artifacts must not be relabeled as runs of this corrected trainer. This fix
does not alter the previously frozen portable/WASM artifact or retroactively
qualify its feeding behavior.
