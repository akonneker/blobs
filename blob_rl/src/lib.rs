//! blob_rl — Reinforcement learning training for blob minds using Burn.

pub mod action;
pub mod adjudication_sensitivity;
pub mod artifact;
pub mod behavior_cloning;
pub mod checkpoint_evaluation;
pub mod checkpoint_promotion;
pub mod combat_horizon_evaluation;
pub mod competency_frontier;
pub mod config;
pub mod contact_evaluation;
pub mod control_matrix;
pub mod demonstration;
pub mod ecological_characterization;
pub mod env;
pub mod evaluation;
pub mod feeding_curriculum;
pub mod feeding_evaluation_artifact;
pub mod learned_signal_attribution;
pub mod match_explorer;
pub mod model;
pub mod observation;
pub mod opponent;
pub mod physics_calibration;
pub mod ppo;
pub mod scale_qualification;
pub mod scale_transition;
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
