use blob_engine::resolution::{
    ActionRequest, DurationRule, ReferenceCheckpoint, ReferenceReplayDriver, ReferenceRuleset,
    ReferenceSimulation, ReplayBatchEvent, ReplayCommitment, ReplayDriverError, ReplayLimits,
    ReplayRecorder, ReplaySegment, ReplaySegmentLimits,
};
use blob_interface::reference_mind::{ReferenceMemoryUpdate, ReferenceSignalEmission};

fn staggered_rules() -> ReferenceRuleset {
    let normal = DurationRule::new(1024, 0, 1);
    ReferenceRuleset {
        wait_duration: normal,
        move_duration: normal,
        attack_duration: normal,
        consume_duration: DurationRule::new(3072, 0, 1),
        split_duration: normal,
        regurgitate_duration: normal,
        digestion_rate_numerator: 0,
        metabolism_rate_numerator: 0,
        diffusion_rate_numerator: 0,
        ..ReferenceRuleset::default()
    }
}

fn commit(
    simulation: &mut ReferenceSimulation,
    actor: blob_engine::resolution::CellKey,
    request: ActionRequest,
) -> ReplayCommitment {
    let started_at = simulation.now();
    let receipt = simulation
        .commit_memory_update_with_signal(
            actor,
            request.clone(),
            None,
            ReferenceMemoryUpdate::Retain,
        )
        .unwrap();
    ReplayCommitment {
        actor,
        request,
        signal: None,
        memory_update: ReferenceMemoryUpdate::Retain,
        started_at,
        receipt,
    }
}

fn commit_with_memory(
    simulation: &mut ReferenceSimulation,
    actor: blob_engine::resolution::CellKey,
    request: ActionRequest,
    next_private_memory: Vec<u8>,
) -> ReplayCommitment {
    let started_at = simulation.now();
    let receipt = simulation
        .commit_decision(actor, request.clone(), next_private_memory.clone())
        .unwrap();
    ReplayCommitment {
        actor,
        request,
        signal: None,
        memory_update: ReferenceMemoryUpdate::Replace(next_private_memory),
        started_at,
        receipt,
    }
}

struct Fixture {
    first: ReplaySegment,
    second: ReplaySegment,
    events: Vec<ReplayBatchEvent>,
    final_state_hash: blob_engine::resolution::CanonicalHash,
}

fn delayed_action_fixture() -> Fixture {
    let mut simulation = ReferenceSimulation::new(2, 1, staggered_rules()).unwrap();
    let slow = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 20, 100, 1)
        .unwrap();
    let fast = simulation
        .add_cell(simulation.tile(1, 0).unwrap(), 20, 100, 2)
        .unwrap();
    let compiled = simulation.compiled_ruleset_hash();
    let initial = simulation.state_hash();
    let checkpoint_zero = ReferenceCheckpoint::from_simulation(&simulation);
    let mut recorder = ReplayRecorder::new(compiled, initial);
    let cursor_zero = recorder.cursor();
    let mut events = Vec::new();

    let mut commitments = vec![
        commit_with_memory(
            &mut simulation,
            slow,
            ActionRequest::Consume { amount: 1 },
            vec![4, 5, 6],
        ),
        commit(&mut simulation, fast, ActionRequest::Wait),
    ];
    commitments.sort_by_key(|commitment| (commitment.started_at, commitment.actor));
    events.push(
        recorder
            .record_with_commitments(&simulation.resolve_next_batch().unwrap(), commitments)
            .unwrap(),
    );

    events.push({
        let commitments = vec![commit(&mut simulation, fast, ActionRequest::Wait)];
        recorder
            .record_with_commitments(&simulation.resolve_next_batch().unwrap(), commitments)
            .unwrap()
    });
    let checkpoint_two = ReferenceCheckpoint::from_simulation(&simulation);

    events.push({
        let commitments = vec![commit(&mut simulation, fast, ActionRequest::Wait)];
        recorder
            .record_with_commitments(&simulation.resolve_next_batch().unwrap(), commitments)
            .unwrap()
    });

    assert_eq!(events[0].commitments.len(), 2);
    assert_eq!(events[0].outcomes.len(), 1);
    assert_eq!(events[0].commitments[0].actor, slow);
    assert_eq!(events[2].outcomes.len(), 2);
    assert!(events[2]
        .outcomes
        .iter()
        .any(|outcome| outcome.actor == slow));
    assert_eq!(
        simulation.cell(slow).unwrap().private_memory.as_ref(),
        [4, 5, 6]
    );

    let first = ReplaySegment::from_events(
        compiled,
        cursor_zero,
        checkpoint_zero,
        &events[..2],
        ReplaySegmentLimits::default(),
    )
    .unwrap();
    let second = ReplaySegment::from_events(
        compiled,
        first.end_cursor(),
        checkpoint_two,
        &events[2..],
        ReplaySegmentLimits::default(),
    )
    .unwrap();

    Fixture {
        first,
        second,
        events,
        final_state_hash: simulation.state_hash(),
    }
}

#[test]
fn authoritative_driver_replays_delayed_actions_across_segments() {
    let fixture = delayed_action_fixture();
    let mut driver = ReferenceReplayDriver::from_segment(&fixture.first).unwrap();
    assert_eq!(
        driver
            .apply_segment(&fixture.first, ReplayLimits::default())
            .unwrap(),
        2
    );
    assert_eq!(driver.cursor(), fixture.second.start_cursor());
    driver
        .apply_segment(&fixture.second, ReplayLimits::default())
        .unwrap();
    assert_eq!(driver.cursor(), fixture.second.end_cursor());
    assert_eq!(driver.simulation().state_hash(), fixture.final_state_hash);

    let mut seeked = ReferenceReplayDriver::from_segment(&fixture.second).unwrap();
    seeked
        .apply_segment(&fixture.second, ReplayLimits::default())
        .unwrap();
    assert_eq!(seeked.simulation().state_hash(), fixture.final_state_hash);
}

#[test]
fn authoritative_driver_supports_prefix_replay() {
    let fixture = delayed_action_fixture();
    let mut driver = ReferenceReplayDriver::from_segment(&fixture.first).unwrap();
    driver
        .apply_segment_prefix(&fixture.first, 1, ReplayLimits::default())
        .unwrap();
    assert_eq!(driver.cursor().next_sequence, 1);
    assert_eq!(
        driver.simulation().state_hash(),
        fixture.events[0].post_state_hash
    );
}

#[test]
fn authoritative_driver_rejects_a_tampered_commitment_receipt() {
    let fixture = delayed_action_fixture();
    let mut event = fixture.events[0].clone();
    event.commitments[0].receipt.effort_spent += 1;
    let actor = event.commitments[0].actor;
    let mut driver = ReferenceReplayDriver::from_segment(&fixture.first).unwrap();

    assert!(matches!(
        driver.apply_event(&event, ReplayLimits::default()),
        Err(ReplayDriverError::ReceiptMismatch {
            actor: actual_actor,
            ..
        }) if actual_actor == actor
    ));
}

#[test]
fn authoritative_driver_rejects_a_self_consistent_false_state_claim() {
    let fixture = delayed_action_fixture();
    let mut event = fixture.events[0].clone();
    event.post_state_hash = blob_engine::resolution::CanonicalHash::from_bytes([0x5a; 32]);
    let mut driver = ReferenceReplayDriver::from_segment(&fixture.first).unwrap();

    assert!(matches!(
        driver.apply_event(&event, ReplayLimits::default()),
        Err(ReplayDriverError::ReportMismatch("post-state hash"))
    ));
}

#[test]
fn authoritative_driver_replays_signal_commitments() {
    let mut rules = staggered_rules();
    rules.signal_emission_cost = 3;
    rules.signal_decay_rate_numerator = 1;
    rules.signal_decay_rate_denominator = 1024;
    let mut simulation = ReferenceSimulation::new(1, 1, rules).unwrap();
    let actor = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 1)
        .unwrap();
    let checkpoint = ReferenceCheckpoint::from_simulation(&simulation);
    let compiled = simulation.compiled_ruleset_hash();
    let mut recorder = ReplayRecorder::new(compiled, simulation.state_hash());
    let started_at = simulation.now();
    let request = ActionRequest::Wait;
    let signal = Some(ReferenceSignalEmission { channel: 2 });
    let next_private_memory = vec![7, 8, 9];
    let receipt = simulation
        .commit_decision_with_signal(actor, request.clone(), signal, next_private_memory.clone())
        .unwrap();
    let commitment = ReplayCommitment {
        actor,
        request,
        signal,
        memory_update: ReferenceMemoryUpdate::Replace(next_private_memory),
        started_at,
        receipt,
    };
    let event = recorder
        .record_with_commitments(&simulation.resolve_next_batch().unwrap(), vec![commitment])
        .unwrap();
    let segment = ReplaySegment::from_events(
        compiled,
        blob_engine::resolution::ReplayChainCursor::genesis(compiled, checkpoint.state_hash()),
        checkpoint,
        &[event],
        ReplaySegmentLimits::default(),
    )
    .unwrap();

    let mut driver = ReferenceReplayDriver::from_segment(&segment).unwrap();
    driver
        .apply_segment(&segment, ReplayLimits::default())
        .unwrap();

    assert_eq!(driver.simulation().state_hash(), simulation.state_hash());
    assert_eq!(
        driver
            .simulation()
            .cell(actor)
            .unwrap()
            .private_memory
            .as_ref(),
        [7, 8, 9]
    );
}
