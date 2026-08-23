use blob_engine::engine::{CellConfig, Engine, ReplayStreamConfig};
use blob_engine::resolution::{
    ActionRequest, DurationRule, IntegrityMode, ReferenceRuleset, ReferenceSimulation, ReplayError,
    ReplayRecorder,
};

fn inert_rules() -> ReferenceRuleset {
    let uniform = DurationRule::new(1024, 0, 1);
    ReferenceRuleset {
        wait_duration: uniform,
        metabolism_rate_numerator: 0,
        diffusion_rate_numerator: 0,
        signal_decay_rate_numerator: 0,
        ..ReferenceRuleset::default()
    }
}

#[test]
fn on_demand_integrity_preserves_semantics_and_can_hash_the_result_later() {
    let mut verified = ReferenceSimulation::new(2, 1, inert_rules()).unwrap();
    let actor = verified
        .add_cell(verified.tile(0, 0).unwrap(), 10, 100, 7)
        .unwrap();
    verified.commit_action(actor, ActionRequest::Wait).unwrap();
    let mut on_demand = verified.clone();

    let verified_report = verified
        .resolve_next_batch_with_integrity(IntegrityMode::Verified)
        .unwrap();
    let on_demand_report = on_demand
        .resolve_next_batch_with_integrity(IntegrityMode::OnDemand)
        .unwrap();

    assert!(verified_report.is_verified());
    assert!(!on_demand_report.is_verified());
    assert_eq!(on_demand_report.pre_state_hash(), None);
    assert_eq!(on_demand_report.state_hash(), None);
    assert_eq!(on_demand_report.completed_at, verified_report.completed_at);
    assert_eq!(on_demand_report.outcomes, verified_report.outcomes);
    assert_eq!(on_demand_report.claims, verified_report.claims);
    assert_eq!(on_demand_report.deaths, verified_report.deaths);
    assert_eq!(on_demand_report.births, verified_report.births);
    assert_eq!(on_demand_report.delta, verified_report.delta);
    assert_eq!(on_demand.canonical_state(), verified.canonical_state());
    assert_eq!(on_demand.state_hash(), verified.state_hash());
    assert_eq!(verified_report.state_hash(), Some(on_demand.state_hash()));
}

#[test]
fn replay_recorder_rejects_an_on_demand_batch() {
    let mut simulation = ReferenceSimulation::new(1, 1, inert_rules()).unwrap();
    let actor = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let compiled = simulation.compiled_ruleset_hash();
    let initial = simulation.state_hash();
    simulation
        .commit_action(actor, ActionRequest::Wait)
        .unwrap();
    let report = simulation
        .resolve_next_batch_with_integrity(IntegrityMode::OnDemand)
        .unwrap();

    assert_eq!(
        ReplayRecorder::new(compiled, initial).record(&report),
        Err(ReplayError::UnverifiedBatch)
    );
}

#[test]
fn engine_prevents_replay_and_on_demand_mode_from_overlapping() {
    let config = CellConfig {
        starting_cells_per_team: 0,
        ..CellConfig::default()
    };
    let mut on_demand = Engine::new(2, 2, 10, config.clone(), Some(1), inert_rules());
    on_demand
        .set_reference_integrity_mode(IntegrityMode::OnDemand)
        .unwrap();
    assert!(on_demand.start_reference_replay_recording(1).is_err());
    assert!(on_demand
        .start_reference_replay_streaming(ReplayStreamConfig::default())
        .is_err());

    let mut recording = Engine::new(2, 2, 10, config, Some(1), inert_rules());
    recording.start_reference_replay_recording(1).unwrap();
    assert!(recording
        .set_reference_integrity_mode(IntegrityMode::OnDemand)
        .is_err());
    assert_eq!(
        recording.reference_integrity_mode(),
        IntegrityMode::Verified
    );
}
