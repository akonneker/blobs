// src/lib.rs

// Declare modules to be part of the library crate.
// These are the core components of the game engine.
pub mod abi;
pub mod cell;
pub mod randomness;
pub mod reference_mind;
pub mod types;
pub mod world;

// Cap'n Proto generated files
pub mod reference_mind_capnp;

pub mod reference_mind_converter;
