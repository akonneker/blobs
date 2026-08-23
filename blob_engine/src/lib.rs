//! blob_engine — Canonical deterministic simulation engine.
//!
//! No WASM, no GUI dependencies. Designed for:
//! - High-performance RL training with native minds
//! - Reuse by `blob_game` (WASM sandbox) as the simulation backend

#[cfg(not(target_arch = "wasm32"))]
pub mod engine;
#[cfg(not(target_arch = "wasm32"))]
pub mod mind_runtime;
#[cfg(not(target_arch = "wasm32"))]
pub mod native_minds;
#[cfg(not(target_arch = "wasm32"))]
pub mod online_service;
#[cfg(not(target_arch = "wasm32"))]
pub mod replay_store;
pub mod resolution;
#[cfg(not(target_arch = "wasm32"))]
pub mod server_verification;
#[cfg(not(target_arch = "wasm32"))]
pub mod world_gen;
