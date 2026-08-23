# Blob Game

A programming game where teams of cells, controlled by WebAssembly modules, compete in a 2D world. The concept has lived in my mind for years, since I first saw [this blog post](https://phonons.wordpress.com/2010/06/01/cells-a-massively-multi-agent-python-programming-game/). I didn't really consult it in this implementation, since I had my own ideas. The primary difference is in sandboxing the minds, and removing any global interactions between agents. I think this will make for more interesting emergent behaviors eventually, but it's early days now. Everything is a janky mess.

## Overview

This project is a simulation of a 2D world where blobs compete for resources. Each blob is controlled by a "mind", which is a WebAssembly (WASM) module. The game is written in Rust and uses the `egui` library for the GUI. The minds are loaded at runtime by the game engine.

The core idea is to provide a platform for developing and testing different AI strategies for the blobs. The game is highly customizable, allowing you to change the world generation, cell properties, and more.

Currently the sample minds are AI-generated garbage. Better ones should be pushed in the next few days. I also n

## Project Structure

The project is structured as a Cargo workspace with the following main components:

-   `blob_game`: The main game engine and GUI. It's responsible for running the simulation, rendering the world, and loading the minds.
-   `blob_interface`: This crate defines the interface between the game and the minds. It uses Cap'n Proto for serialization to communicate with the WASM modules.
-   `minds/`: This directory contains several example minds, each with a different strategy (e.g., `aggressive_mind`, `explorer_mind`). These can be used as a starting point for creating your own minds.

## How to Build and Run

1.  **Build the minds:** Each mind is a separate crate that needs to be compiled to the `wasm32-unknown-unknown` target.

    ```bash
    cargo build --target wasm32-unknown-unknown -p <mind_name>
    ```

    For example, to build the `simple_mind`:

    ```bash
    cargo build --target wasm32-unknown-unknown -p simple_mind
    ```

    The compiled WASM file will be located at `target/wasm32-unknown-unknown/debug/<mind_name>.wasm`.

2.  **Run the game:** The `blob_game` executable takes the paths to the mind WASM files as arguments.

    To run the game with a GUI, use the `--gui` flag:

    ```bash
    cargo run -p blob_game -- --gui <path_to_mind1>.wasm <path_to_mind2>.wasm
    ```

    For example, to run a game with `simple_mind` and `aggressive_mind`:

    ```bash
    cargo run -p blob_game -- --gui target/wasm32-unknown-unknown/debug/simple_mind.wasm target/wasm32-unknown-unknown/debug/aggressive_mind.wasm
    ```

    You can also run the game in headless mode by omitting the `--gui` flag. For more options, run `cargo run -p blob_game -- --help`.

## How to Create a Mind

To create your own mind, you can start by copying one of the existing minds
(for example, `simple_mind`). A mind compiles to WASM and exports
`reference_mind_function`, which is called once for each ready cell.

The function receives canonical Cap'n Proto bytes for one isolated cell and
returns a complete action, optional anonymous signal, and explicit private-
memory operation. The schema and bounded converters live in `blob_interface`.

Your `Cargo.toml` should be configured to produce a `cdylib` library type:

```toml
[lib]
crate-type = ["cdylib"]
```

### Using Extism and Other Languages

Mind authors continue to use an [Extism PDK](https://extism.org/docs/category/pdk-documentation),
so they are not tied to the host's Rust implementation. The default host is
Extism. An experimental `memory.wasm_executor = "extism_compat"` host executes
the same byte-oriented PDK contract directly with Wasmtime and avoids much of
Extism's per-instance host setup. Its workers share one immutable engine,
compiled module, link plan, and deadline ticker per team pool. Every cell
decision still creates a fresh store, guest instance, guest memory, and host
byte arena. The fresh host arena keeps ordinary decision bytes and allocation
metadata inline, spilling to bounded heap storage for larger PDK calls; that
storage is never shared across decisions. Stock Extism keeps a worker-local
compiled descriptor because its
public compiled type can contain non-thread-safe host user data; no unsafe
sharing wrapper is used.

The restricted host admits the deterministic PDK memory/input/output imports
only. It deliberately rejects WASI, configuration, variables, HTTP, custom
host functions, and logging. A Mind using one of those capabilities is invalid,
not silently given a shared or persistent resource. Stock Extism remains the
compatibility oracle. Maintained AssemblyScript and TinyGo artifacts are
admitted and executed through both hosts in conformance testing.

Online submissions bind the named `extism_pdk_deterministic_v1` profile into
the signed verification manifest. Admission parses the artifact before
compilation and rejects modules whose typed imports, required export, memory,
or table shape falls outside that profile. Executor choice and worker count do
not enter the canonical match contract.

Extism publishes PDKs for languages including:

-   Rust
-   Go
-   Haskell
-   Zig
-   AssemblyScript (TypeScript-like)
-   C/C++

To create a Mind in another language, use that language's PDK to export
`reference_mind_function: () -> i32`, decode and encode
`blob_interface/interface/reference_mind.capnp`, and restrict imports to the
deterministic PDK memory contract. The same artifact can run on stock Extism;
it does not target a project-specific Wasmtime SDK.

Run `scripts/build_language_minds.sh` to build the maintained AssemblyScript
and Go canaries. The Go target is TinyGo `wasm-unknown`; ordinary
`GOOS=wasip1` output is rejected because WASI is outside the deterministic
profile.

## The `blob_interface` API

The communication boundary is Mind ABI v6 in
`blob_interface/interface/reference_mind.capnp`. It contains one cell's own
state, bounded anonymous local observations, action availability, explicit
private memory, and private random bytes. Outputs cover Wait, Move, Attack,
Guard, Consume, Split, Regurgitate, Excavate, and DepositTerrain. Local slot
visibility uses explicit presence bits plus inline scalar values, preserving
hidden-versus-visible-zero semantics without pointer-backed option objects.

## Contributing

Contributions are welcome! If you have an idea for a new feature or have found a bug, please open an issue or submit a pull request.
