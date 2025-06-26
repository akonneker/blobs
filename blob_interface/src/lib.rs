// src/lib.rs

// Declare modules to be part of the library crate.
// These are the core components of the game engine.
pub mod world;
pub mod cell;
pub mod types;

// Cap'n Proto generated files
pub mod action_capnp;
pub mod mind_input_capnp;

pub mod action_converter;
pub mod mind_input_converter;