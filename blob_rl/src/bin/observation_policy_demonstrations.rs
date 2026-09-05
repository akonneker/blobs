//! Convert a successful anonymous-observation combat oracle into immutable
//! behavior-cloning trajectories without consuming its sealed holdout seeds.

use std::fs;
use std::path::PathBuf;

use blob_rl::config::TrainingConfig;
use blob_rl::demonstration::{
    generate_observation_policy_demonstrations, publish_demonstrations,
    ObservationPolicyDemonstrationOptions,
};
use blob_rl::micro_combat::MicroCombatSuiteConfig;
use blob_rl::micro_combat_observation_policy::ObservationPolicySearchReport;
use clap::Parser;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(
    name = "observation-policy-demonstrations",
    about = "Replay a verified observation-policy oracle into behavior-cloning data"
)]
struct Args {
    /// Training TOML supplying the base rules and observation contract.
    #[arg(long)]
    config: PathBuf,

    /// Micro-combat scenario suite containing the report's named scenario.
    #[arg(long)]
    scenarios: PathBuf,

    /// Immutable observation-policy search report. Only synthesis seeds are
    /// replayed; its holdout partition remains sealed.
    #[arg(long)]
    policy_report: PathBuf,

    /// New immutable demonstration directory.
    #[arg(long)]
    output: PathBuf,

    #[arg(long, default_value_t = 65_536)]
    max_samples: usize,

    /// Require this many disjoint report holdouts, all covered and successful.
    #[arg(long, default_value_t = 32)]
    minimum_successful_holdout_seeds: usize,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn main() {
    let args = Args::parse();
    let config_bytes = fs::read(&args.config)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", args.config.display()));
    let config_text = std::str::from_utf8(&config_bytes)
        .unwrap_or_else(|error| panic!("{} is not UTF-8: {error}", args.config.display()));
    let config: TrainingConfig = toml::from_str(config_text)
        .unwrap_or_else(|error| panic!("failed to load {}: {error}", args.config.display()));
    config
        .validate()
        .unwrap_or_else(|error| panic!("invalid training configuration: {error}"));

    let suite_bytes = fs::read(&args.scenarios)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", args.scenarios.display()));
    let suite_text = std::str::from_utf8(&suite_bytes)
        .unwrap_or_else(|error| panic!("{} is not UTF-8: {error}", args.scenarios.display()));
    let suite = MicroCombatSuiteConfig::from_toml_str(suite_text)
        .unwrap_or_else(|error| panic!("failed to load scenario suite: {error}"));

    let report_bytes = fs::read(&args.policy_report)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", args.policy_report.display()));
    let report: ObservationPolicySearchReport = serde_json::from_slice(&report_bytes)
        .unwrap_or_else(|error| panic!("failed to decode observation-policy report: {error}"));
    let scenario = suite
        .scenarios
        .iter()
        .find(|scenario| scenario.name == report.scenario)
        .unwrap_or_else(|| panic!("scenario {} was not found", report.scenario));

    let (manifest, payload) = generate_observation_policy_demonstrations(
        &config,
        sha256(&config_bytes),
        scenario,
        &report,
        &ObservationPolicyDemonstrationOptions {
            source_report_sha256: sha256(&report_bytes),
            max_samples: args.max_samples,
            minimum_successful_holdout_seeds: args.minimum_successful_holdout_seeds,
        },
    )
    .unwrap_or_else(|error| panic!("failed to generate oracle demonstrations: {error}"));
    publish_demonstrations(&args.output, &manifest, &payload)
        .unwrap_or_else(|error| panic!("failed to publish demonstrations: {error}"));
    println!(
        "Published {} oracle samples from {} successful synthesis seeds to {} (wait={} guard={} move={} attack={})",
        manifest.samples,
        manifest.completed_episodes,
        args.output.display(),
        manifest.action_family_samples[blob_rl::action::PolicyActionFamily::Wait.index()],
        manifest.action_family_samples[blob_rl::action::PolicyActionFamily::Guard.index()],
        manifest.action_family_samples[blob_rl::action::PolicyActionFamily::Move.index()],
        manifest.action_family_samples[blob_rl::action::PolicyActionFamily::Attack.index()],
    );
}
