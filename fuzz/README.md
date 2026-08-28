# Maintained fuzz harnesses

The fuzz crate is intentionally outside the release workspace. It exercises
untrusted byte boundaries with tight allocation limits:

- `replay_formats` covers replay frames, archives, checkpoints, bundles,
  segments, manifests, match manifests, and attestations.
- `mind_abi` covers canonical Cap'n Proto Mind inputs and decisions and asserts
  semantic round trips for every accepted value.

Install the pinned tool and run a target locally:

```sh
rustup toolchain install nightly-2026-08-01 --profile minimal
cargo +nightly-2026-08-01 install cargo-fuzz --version 0.13.2 --locked
cargo +nightly-2026-08-01 fuzz run replay_formats
```

CI runs short bounded smoke jobs. Longer scheduled campaigns should retain
their corpus and crash directories as CI artifacts. A minimized finding must
become a deterministic regression test before it is considered resolved.
