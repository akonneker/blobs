# Blob Game

A programming game where teams of cells, controlled by WebAssembly modules, compete in a 2D world. The concept has lived in my mind for years, since I first saw [this blog post](https://phonons.wordpress.com/2010/06/01/cells-a-massively-multi-agent-python-programming-game/). I didn't really consult it in this implementation, since I had my own ideas. The primary difference is in sandboxing the minds, and removing any global interactions between agents. I think this will make for more interesting emergent behaviors eventually, but it's early days now. Everything is a janky mess.

## Overview

This project is a simulation of a 2D world where blobs compete for resources. Each blob is controlled by a "mind", which is a WebAssembly (WASM) module. The game is written in Rust and uses the `egui` library for the GUI. The minds are loaded at runtime by the game engine.

The core idea is to provide a platform for developing and testing different AI strategies for the blobs. The game is highly customizable, allowing you to change the world generation, cell properties, and more.


Currently the sample minds are AI-generated garbage. Better ones should be pushed in the next few days.

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

2.  **Create team configuration files (optional):** You can either use WASM files directly or create TOML configuration files for more control.

    Create a file like `team_simple.toml`:
    ```toml
    mind_path = "target/wasm32-unknown-unknown/debug/simple_mind.wasm"
    start_id = 1
    ```

    The `start_id` sets the initial marker value for all cells of that team, which can be used for team identification.

3.  **Run the game:** The `blob_game` executable accepts either team configuration files (.toml) or WASM files (.wasm) directly.

    To run the game with a GUI, use the `--gui` flag:

    ```bash
    # Using team configuration files
    cargo run -p blob_game -- --gui <team_config1>.toml <team_config2>.toml
    
    # Using WASM files directly
    cargo run -p blob_game -- --gui <mind1>.wasm <mind2>.wasm
    
    # Mixed usage
    cargo run -p blob_game -- --gui team_config.toml mind.wasm
    ```

    **Override team IDs:** You can override the start_id for teams using the `--team-ids` argument:

    ```bash
    # Override IDs for all teams
    cargo run -p blob_game -- --gui --team-ids 10,20 mind1.wasm mind2.wasm
    
    # Override only first team's ID, second gets random ID
    cargo run -p blob_game -- --gui --team-ids 42 mind1.wasm mind2.wasm
    ```

    For example, to run a game with two teams:

    ```bash
    cargo run -p blob_game -- --gui blob_game/config/team_simple.toml blob_game/config/team_aggressive.toml
    ```

    You can also run the game in headless mode by omitting the `--gui` flag. For more options, run `cargo run -p blob_game -- --help`.

## Team Configuration

Teams can be configured in two ways:

### 1. TOML Configuration Files
Create TOML files that specify both the mind path and start ID:

```toml
mind_path = "target/wasm32-unknown-unknown/debug/simple_mind.wasm"
start_id = 42
```

### 2. Direct WASM Files
You can pass WASM files directly. When using WASM files directly:
- If no `--team-ids` argument is provided, teams get random start IDs
- If `--team-ids` is provided, teams get the specified IDs (in order)
- If fewer IDs are provided than teams, remaining teams get random IDs

**Team configuration fields:**
- `mind_path`: Path to the compiled WASM file for the team's mind
- `start_id`: Initial marker ID for all cells of this team (used for team identification)

The `start_id` is important because minds can use it to distinguish between allied and enemy blobs by comparing marker values.

## How to Create a Mind

To create your own mind, you can start by copying one of the existing minds (e.g., `simple_mind`). A mind is a Rust crate that compiles to WASM and exposes a single function, `mind_function`, which is called by the game on each turn.

The `mind_function` function receives the current state of the blob and its surroundings as input (`MindInput`) and should return a `MindOutput` containing both an action and updated memory for the blob to perform. The `blob_interface` crate provides the necessary data structures and serialization functions.

**Important**: Your mind function must return a `MindOutput` struct that includes both the action and updated memory. Memory persists between actions, allowing your mind to maintain state across turns.

Your `Cargo.toml` should be configured to produce a `cdylib` library type:

```toml
[lib]
crate-type = ["cdylib"]
```

### Using Extism and Other Languages

The game uses the [Extism](https://extism.org/) plug-in system to load and run the WASM minds. This means that you are not limited to Rust for writing your minds. Any language that has an [Extism PDK (Plug-in Development Kit)](https://extism.org/docs/category/pdk-documentation) can be used. This includes:

-   Rust
-   Go
-   Haskell
-   Zig
-   AssemblyScript (TypeScript-like)
-   C/C++

To create a mind in another language, you will need to follow the instructions for that language's PDK to create a WASM module that exports a `mind_function` function. 

The use of Cap'n Proto also adds its own restrictions on language usage, but that could be swapped out for another serialization scheme.

## The `blob_interface` API

The communication between the game and the minds is defined in the `blob_interface` crate. The main data structures are:

-   `MindInput`: This struct contains all the information a mind receives on each turn. This includes:
    -   `BlobState`: The internal state of the blob (energy, memory, etc.).
    -   `BlobContext`: Information about the blob's immediate surroundings (elevation, energy, pheromones, etc.).
-   `MindOutput`: This struct contains what a mind returns on each turn:
    -   `Action`: The action the blob should perform (move, attack, eat, split, etc.).
    -   `Memory`: Updated memory state that will persist to the next turn (2048 bytes).

**Memory Persistence**: Each blob has 2048 bytes of persistent memory that survives across all actions. This allows minds to implement complex behaviors like:
- Directional movement patterns
- Resource tracking and exploration strategies  
- State machines for different behaviors
- Learning and adaptation over time

The memory is automatically preserved between actions, so your mind can maintain stateful information like current direction, discovered resources, or behavioral modes.

For more details, see the Cap'n Proto schema files in `blob_interface/interface`.

## Contributing

Contributions are welcome! If you have an idea for a new feature or have found a bug, please open an issue or submit a pull request.
