//! blob_rl — Reinforcement learning training for blob minds using Burn.

pub mod action;
pub mod adjudication_sensitivity;
pub mod artifact;
pub mod behavior_cloning;
pub mod config;
pub mod control_matrix;
pub mod demonstration;
pub mod ecological_characterization;
pub mod env;
pub mod evaluation;
pub mod learned_signal_attribution;
pub mod model;
pub mod observation;
pub mod opponent;
pub mod ppo;
pub mod signal_policy_ablation;
pub mod signal_visibility_ablation;
pub mod sweep;
pub mod sweep_execution;
pub mod telemetry;
pub mod training;
pub mod viability;
pub mod viability_gate;
pub mod viability_matrix;
pub mod viability_preflight;

#[cfg(test)]
pub(crate) static BACKEND_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
