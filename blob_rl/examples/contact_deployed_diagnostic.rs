//! Read-only attack-opportunity probe and maintained aggressive positive control.
use blob_interface::reference_mind::*;
use blob_policy::action::action_is_commit_legal;
use blob_rl::{
    config::{OpponentProfile, TrainingConfig},
    contact_deployed::evaluate_contact_mind,
    deployed_artifact::{load_deployed_policy, read_bounded, sha},
    env::ActionBufferMind,
};
use clap::Parser;
use serde::Serialize;
use serde_json::json;
use std::{fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(
        long,
        required_unless_present = "aggressive_baseline",
        conflicts_with = "aggressive_baseline"
    )]
    export: Option<PathBuf>,
    #[arg(
        long,
        required_unless_present = "aggressive_baseline",
        conflicts_with = "aggressive_baseline"
    )]
    export_manifest_sha256: Option<String>,
    #[arg(long)]
    aggressive_baseline: bool,
    #[arg(long, value_delimiter = ',', required = true)]
    seeds: Vec<u64>,
    #[arg(long)]
    output: PathBuf,
}
#[derive(Default, Serialize)]
struct Opportunities {
    decisions: u64,
    legal_attack_available: u64,
    legal_occupied_attack_available: u64,
    attacks_selected: u64,
    moves_despite_legal_attack: u64,
    moves_despite_legal_occupied_attack: u64,
}
struct Probe {
    mind: Box<dyn ReferenceMind>,
    episodes: Vec<Opportunities>,
}
impl ReferenceMind for Probe {
    fn reset(&mut self) -> Result<(), String> {
        self.mind.reset()?;
        self.episodes.push(Opportunities::default());
        Ok(())
    }
    fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision {
        let decision = self.mind.decide(input);
        let legal_target = |slot: &LocalObservation| {
            [
                ReferenceEffort::Gentle,
                ReferenceEffort::Standard,
                ReferenceEffort::Burst,
            ]
            .into_iter()
            .any(|effort| {
                action_is_commit_legal(
                    input,
                    &ReferenceMindAction::Attack {
                        target_slot: slot.slot,
                        effort,
                        payload: 1,
                    },
                    decision.signal.is_some(),
                )
            })
        };
        let available = input.slots.iter().any(legal_target);
        let occupied = input
            .slots
            .iter()
            .any(|slot| slot.neighbor.is_some() && legal_target(slot));
        let counts = self
            .episodes
            .last_mut()
            .expect("episode reset before decisions");
        counts.decisions += 1;
        counts.legal_attack_available += u64::from(available);
        counts.legal_occupied_attack_available += u64::from(occupied);
        counts.attacks_selected += u64::from(matches!(
            decision.action,
            ReferenceMindAction::Attack { .. }
        ));
        counts.moves_despite_legal_attack +=
            u64::from(available && matches!(decision.action, ReferenceMindAction::Move { .. }));
        counts.moves_despite_legal_occupied_attack +=
            u64::from(occupied && matches!(decision.action, ReferenceMindAction::Move { .. }));
        decision
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.output.exists() {
        return Err("diagnostic output already exists".into());
    }
    if args.seeds.is_empty()
        || args
            .seeds
            .iter()
            .any(|seed| [1434999901, 1434999902].contains(seed))
    {
        return Err("empty or reserved diagnostic seeds".into());
    }
    let bytes = read_bounded(&args.config, 1024 * 1024)?;
    let config = TrainingConfig::from_toml_str(std::str::from_utf8(&bytes)?)?;
    config.validate()?;
    let (mind, identity): (Box<dyn ReferenceMind>, _) = if args.aggressive_baseline {
        (
            Box::new(ActionBufferMind::new_opponent(OpponentProfile::Aggressive)),
            json!({"baseline":"aggressive"}),
        )
    } else {
        let hash = args
            .export_manifest_sha256
            .as_ref()
            .ok_or("missing manifest SHA")?;
        let (policy, manifest) = load_deployed_policy(
            args.export.as_ref().ok_or("missing export")?,
            hash,
            &args.seeds,
        )?;
        (
            Box::new(policy),
            json!({"manifest_sha256":hash,"weights_sha256":manifest["weights_sha256"]}),
        )
    };
    let mut probe = Probe {
        mind,
        episodes: Vec::new(),
    };
    let evaluation = evaluate_contact_mind(&mut probe, &config, &args.seeds)?;
    assert_eq!(evaluation.episodes.len(), probe.episodes.len());
    let result = json!({"complete":true,"diagnostic_contract":"blob.contact.occupied-target-opportunity.v2","identity":identity,"source_config_sha256":sha(&bytes),
        "seeds":args.seeds,"evaluation":evaluation,"opportunities":probe.episodes,
        "scope":"Read-only legal payload-one attack availability with the returned signal cost reserved. Availability does not guarantee a tactically useful attack. Diagnostic sidecar, not a new promotion gate."});
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(args.output)?;
    serde_json::to_writer_pretty(&mut file, &result)?;
    Ok(())
}
