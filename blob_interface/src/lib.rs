// src/lib.rs

// Declare modules to be part of the library crate.
// These are the core components of the game engine.
pub mod world;
pub mod cell;
pub mod types;

// Cap'n Proto generated files
pub mod mind_input_capnp;
pub mod mind_output_capnp;

pub mod mind_input_converter;
pub mod mind_output_converter;