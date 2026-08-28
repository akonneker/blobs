use blob_engine::resolution::{
    genesis_hash, verify_replay, ActionRequest, DurationRule, EffortTier, LocalSlot,
    ReferenceRuleset, ReferenceSimulation, ReplayArchive, ReplayArchiveLimits, ReplayBatchEvent,
    ReplayCommitment, ReplayError, ReplayLimits, ReplayRecorder, REPLAY_FORMAT_VERSION,
};
use blob_interface::reference_mind::ReferenceMemoryUpdate;
use sha2::{Digest, Sha256};

const WEST: LocalSlot = LocalSlot(3);
const EAST: LocalSlot = LocalSlot(4);

fn uniform_rules() -> ReferenceRuleset {
    let uniform = DurationRule::new(1024, 0, 1);
    ReferenceRuleset {
        wait_duration: uniform,
        move_duration: uniform,
        attack_duration: uniform,
        consume_duration: uniform,
        split_duration: uniform,
        regurgitate_duration: uniform,
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

fn two_batch_replay() -> (
    Vec<ReplayBatchEvent>,
    Vec<Vec<u8>>,
    blob_engine::resolution::CanonicalHash,
    blob_engine::resolution::CanonicalHash,
) {
    let mut simulation = ReferenceSimulation::new(3, 1, uniform_rules()).unwrap();
    let left = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 1)
        .unwrap();
    let right = simulation
        .add_cell(simulation.tile(2, 0).unwrap(), 10, 100, 2)
        .unwrap();
    let compiled = simulation.compiled_ruleset_hash();
    let initial = simulation.state_hash();
    let mut recorder = ReplayRecorder::new(compiled, initial);

    let commitments = vec![
        commit(
            &mut simulation,
            left,
            ActionRequest::Move {
                target: EAST,
                effort: EffortTier::Standard,
            },
        ),
        commit(
            &mut simulation,
            right,
            ActionRequest::Move {
                target: WEST,
                effort: EffortTier::Standard,
            },
        ),
    ];
    let first = recorder
        .record_with_commitments(&simulation.resolve_next_batch().unwrap(), commitments)
        .unwrap();

    let commitments = vec![
        commit(&mut simulation, left, ActionRequest::Wait),
        commit(&mut simulation, right, ActionRequest::Wait),
    ];
    let second = recorder
        .record_with_commitments(&simulation.resolve_next_batch().unwrap(), commitments)
        .unwrap();
    let events = vec![first, second];
    let frames = events.iter().map(ReplayBatchEvent::to_bytes).collect();
    (events, frames, compiled, initial)
}

#[test]
fn replay_frames_round_trip_and_verify_as_a_chain() {
    let (events, frames, compiled, initial) = two_batch_replay();

    for (event, frame) in events.iter().zip(&frames) {
        assert_eq!(ReplayBatchEvent::from_bytes(frame).unwrap(), *event);
    }
    let verified = verify_replay(&frames, compiled, initial, ReplayLimits::default()).unwrap();
    assert_eq!(verified.event_count, 2);
    assert_eq!(verified.final_event_hash, events[1].event_hash());
    assert_eq!(verified.final_state_hash, events[1].post_state_hash);
    assert_eq!(events[1].previous_event_hash, events[0].event_hash());
    assert_eq!(events[1].previous_state_hash, events[0].post_state_hash);
}

#[test]
fn replay_round_trip_preserves_exact_consume_intake() {
    let mut simulation = ReferenceSimulation::new(1, 1, uniform_rules()).unwrap();
    let tile = simulation.tile(0, 0).unwrap();
    simulation.tile_state_mut(tile).unwrap().plant_energy = 7;
    let actor = simulation.add_cell(tile, 10, 100, 1).unwrap();
    let compiled = simulation.compiled_ruleset_hash();
    let initial = simulation.state_hash();
    let mut recorder = ReplayRecorder::new(compiled, initial);
    let commitments = vec![commit(
        &mut simulation,
        actor,
        ActionRequest::Consume { amount: 7 },
    )];
    let event = recorder
        .record_with_commitments(&simulation.resolve_next_batch().unwrap(), commitments)
        .unwrap();

    assert_eq!(event.outcomes[0].consumed_energy, 7);
    let decoded = ReplayBatchEvent::from_bytes(&event.to_bytes()).unwrap();
    assert_eq!(decoded.outcomes[0].consumed_energy, 7);
    assert_eq!(decoded, event);
}

#[test]
fn replay_distinguishes_retain_from_replace_empty() {
    let (events, _, _, _) = two_batch_replay();
    let retained = &events[0];
    assert_eq!(
        retained.commitments[0].memory_update,
        ReferenceMemoryUpdate::Retain
    );
    assert_eq!(
        ReplayBatchEvent::from_bytes(&retained.to_bytes()).unwrap(),
        *retained
    );

    let mut replaced = retained.clone();
    replaced.commitments[0].memory_update = ReferenceMemoryUpdate::Replace(Vec::new());
    assert_ne!(retained.event_hash(), replaced.event_hash());
    assert_eq!(
        ReplayBatchEvent::from_bytes(&replaced.to_bytes())
            .unwrap()
            .commitments[0]
            .memory_update,
        ReferenceMemoryUpdate::Replace(Vec::new())
    );

    replaced.commitments[0].memory_update = ReferenceMemoryUpdate::Replace(vec![0; 2_048]);
    assert_eq!(replaced.to_bytes().len() - retained.to_bytes().len(), 2_056);
}

#[test]
fn replay_round_trip_preserves_full_split_request() {
    let mut simulation = ReferenceSimulation::new(2, 1, uniform_rules()).unwrap();
    let parent = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 1)
        .unwrap();
    let compiled = simulation.compiled_ruleset_hash();
    let initial = simulation.state_hash();
    let mut recorder = ReplayRecorder::new(compiled, initial);
    let request = ActionRequest::Split {
        target: EAST,
        child_allocation: 30,
        marker: 9,
        private_memory: vec![1, 2, 3, 5, 8],
    };
    let commitment = commit(&mut simulation, parent, request.clone());
    let event = recorder
        .record_with_commitments(&simulation.resolve_next_batch().unwrap(), vec![commitment])
        .unwrap();

    let decoded = ReplayBatchEvent::from_bytes(&event.to_bytes()).unwrap();
    assert_eq!(decoded.commitments[0].request, request);
    assert_eq!(decoded.outcomes[0].request, request);
    assert_eq!(decoded.outcomes[0].origin, simulation.tile(0, 0).unwrap());
    assert_eq!(decoded, event);
}

#[test]
fn replay_round_trip_preserves_explicit_signal_vector() {
    let mut rules = uniform_rules();
    rules.signal_emission_cost = 2;
    let mut simulation = ReferenceSimulation::new(1, 1, rules).unwrap();
    let actor = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 1)
        .unwrap();
    let mut recorder =
        ReplayRecorder::new(simulation.compiled_ruleset_hash(), simulation.state_hash());
    let request = ActionRequest::Signal {
        amounts: [2, 4, 0, 8],
    };
    let commitment = commit(&mut simulation, actor, request.clone());
    let event = recorder
        .record_with_commitments(&simulation.resolve_next_batch().unwrap(), vec![commitment])
        .unwrap();

    let decoded = ReplayBatchEvent::from_bytes(&event.to_bytes()).unwrap();
    assert_eq!(decoded.commitments[0].request, request);
    assert_eq!(decoded.outcomes[0].request, request);
    assert_eq!(decoded, event);
}

#[test]
fn replay_round_trip_preserves_damage_and_terrain_effects() {
    let mut simulation = ReferenceSimulation::new(2, 1, uniform_rules()).unwrap();
    let attacker = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 1)
        .unwrap();
    let victim = simulation
        .add_cell(simulation.tile(1, 0).unwrap(), 10, 100, 2)
        .unwrap();
    let mut recorder =
        ReplayRecorder::new(simulation.compiled_ruleset_hash(), simulation.state_hash());

    let attack = commit(
        &mut simulation,
        attacker,
        ActionRequest::Attack {
            target: EAST,
            payload: 20,
            effort: EffortTier::Standard,
        },
    );
    let attack_event = recorder
        .record_with_commitments(&simulation.resolve_next_batch().unwrap(), vec![attack])
        .unwrap();
    let decoded_attack = ReplayBatchEvent::from_bytes(&attack_event.to_bytes()).unwrap();
    let damage = decoded_attack.outcomes[0].attack_damage.as_ref().unwrap();
    assert_eq!(damage.victim, victim);
    assert_eq!(
        (
            damage.raw,
            damage.mitigated,
            damage.applied,
            damage.overkill
        ),
        (20, 0, 20, 0)
    );

    let excavate = commit(&mut simulation, attacker, ActionRequest::Excavate);
    let terrain_event = recorder
        .record_with_commitments(&simulation.resolve_next_batch().unwrap(), vec![excavate])
        .unwrap();
    let decoded_terrain = ReplayBatchEvent::from_bytes(&terrain_event.to_bytes()).unwrap();
    let terrain = decoded_terrain.outcomes[0].terrain_change.as_ref().unwrap();
    assert_eq!((terrain.elevation_before, terrain.elevation_after), (0, -1));
    assert_eq!(
        terrain.material_mass,
        simulation.rules().terrain_mass_per_elevation
    );
}

#[test]
fn tampering_reordering_and_chain_splicing_are_rejected() {
    let (events, frames, compiled, initial) = two_batch_replay();

    let mut tampered = frames[0].clone();
    tampered[32] ^= 0x80;
    assert!(matches!(
        ReplayBatchEvent::from_bytes(&tampered),
        Err(ReplayError::EventHashMismatch { .. })
    ));

    assert!(matches!(
        verify_replay(
            [&frames[1], &frames[0]],
            compiled,
            initial,
            ReplayLimits::default()
        ),
        Err(ReplayError::SequenceMismatch { .. })
    ));

    let mut spliced = events[1].clone();
    spliced.previous_event_hash = genesis_hash(compiled, initial);
    assert!(matches!(
        verify_replay(
            [&frames[0], &spliced.to_bytes()],
            compiled,
            initial,
            ReplayLimits::default()
        ),
        Err(ReplayError::PreviousEventHashMismatch { .. })
    ));

    let mut wrong_state_link = events[1].clone();
    wrong_state_link.previous_state_hash = initial;
    assert!(matches!(
        verify_replay(
            [&frames[0], &wrong_state_link.to_bytes()],
            compiled,
            initial,
            ReplayLimits::default()
        ),
        Err(ReplayError::PreviousStateHashMismatch { .. })
    ));
}

#[test]
fn decoder_enforces_frame_and_collection_limits_before_allocation() {
    let (_, frames, _, _) = two_batch_replay();
    let frame = &frames[0];
    let size_limits = ReplayLimits {
        max_frame_bytes: frame.len() - 1,
        ..ReplayLimits::default()
    };
    assert_eq!(
        ReplayBatchEvent::from_bytes_with_limits(frame, size_limits),
        Err(ReplayError::FrameTooLarge {
            actual: frame.len(),
            limit: frame.len() - 1,
        })
    );

    let count_limits = ReplayLimits {
        max_outcomes: 1,
        ..ReplayLimits::default()
    };
    assert!(matches!(
        ReplayBatchEvent::from_bytes_with_limits(frame, count_limits),
        Err(ReplayError::CountLimit {
            field: "outcomes",
            ..
        })
    ));
}

#[test]
fn noncanonical_collection_order_is_rejected_even_with_a_valid_frame_hash() {
    let (events, _, _, _) = two_batch_replay();
    let mut reordered = events[0].clone();
    reordered.outcomes.reverse();

    assert_eq!(
        ReplayBatchEvent::from_bytes(&reordered.to_bytes()),
        Err(ReplayError::NonCanonicalOrder("outcomes"))
    );
}

#[test]
fn replay_hash_golden_vectors() {
    let (events, frames, compiled, initial) = two_batch_replay();
    assert_eq!(
        (
            genesis_hash(compiled, initial).to_hex(),
            events[0].event_hash().to_hex(),
            events[1].event_hash().to_hex(),
            frames[0].len(),
        ),
        (
            "8bbd97f0a21804a9e05ebc9cbb7bce675bea949a699d7b38b255aa4b42dd3423".into(),
            "bb498a32d089550ad935c1f497a9913f153bf88de0dcd9945ddafd1e67ea6d7c".into(),
            "41de17c43a34b532f5ff7b5e61e15608b4d1dfb3bfb0b481b413245ec42ac914".into(),
            1020,
        )
    );
    assert_eq!(events[0].sequence, 0);
    assert_eq!(events[1].sequence, 1);
}

#[test]
fn recorder_rejects_a_report_from_another_ruleset() {
    let mut first = ReferenceSimulation::new(1, 1, uniform_rules()).unwrap();
    let actor = first
        .add_cell(first.tile(0, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let mut recorder = ReplayRecorder::new(first.compiled_ruleset_hash(), first.state_hash());

    let mut changed_rules = uniform_rules();
    changed_rules.bite_capacity += 1;
    let mut second = ReferenceSimulation::new(1, 1, changed_rules).unwrap();
    let other = second
        .add_cell(second.tile(0, 0).unwrap(), 10, 100, 0)
        .unwrap();
    second.commit_action(other, ActionRequest::Wait).unwrap();
    let other_report = second.resolve_next_batch().unwrap();

    first.commit_action(actor, ActionRequest::Wait).unwrap();
    assert!(matches!(
        recorder.record(&other_report),
        Err(ReplayError::RulesetHashMismatch { .. })
    ));
}

#[test]
fn empty_replay_verifies_to_the_genesis_and_initial_state() {
    let compiled = blob_engine::resolution::CanonicalHash::from_bytes([1; 32]);
    let initial = blob_engine::resolution::CanonicalHash::from_bytes([2; 32]);
    let frames: [&[u8]; 0] = [];
    let verified = verify_replay(frames, compiled, initial, ReplayLimits::default()).unwrap();

    assert_eq!(verified.event_count, 0);
    assert_eq!(verified.final_event_hash, genesis_hash(compiled, initial));
    assert_eq!(verified.final_state_hash, initial);
}

#[test]
fn indexed_archive_round_trips_and_seeks_without_scanning() {
    let (events, _, compiled, initial) = two_batch_replay();
    let archive =
        ReplayArchive::from_events(&events, compiled, initial, ReplayArchiveLimits::default())
            .unwrap();
    let decoded = ReplayArchive::from_bytes(archive.to_bytes()).unwrap();

    assert_eq!(decoded.event_count(), 2);
    assert_eq!(decoded.compiled_ruleset_hash(), compiled);
    assert_eq!(decoded.initial_state_hash(), initial);
    assert_eq!(
        decoded.event(0, ReplayLimits::default()).unwrap(),
        events[0]
    );
    assert_eq!(
        decoded.event(1, ReplayLimits::default()).unwrap(),
        events[1]
    );
    assert_eq!(
        decoded.event_bytes(2),
        Err(ReplayError::EventOutOfRange { index: 2, count: 2 })
    );
}

#[test]
fn archive_rejects_tampering_invalid_indexes_and_size_limits() {
    let (events, _, compiled, initial) = two_batch_replay();
    let archive =
        ReplayArchive::from_events(&events, compiled, initial, ReplayArchiveLimits::default())
            .unwrap();
    let mut tampered = archive.to_bytes().to_vec();
    tampered[100] ^= 1;
    assert!(matches!(
        ReplayArchive::from_bytes(&tampered),
        Err(ReplayError::ArchiveHashMismatch { .. })
    ));

    let mut invalid_index = archive.to_bytes().to_vec();
    // The first relative offset follows the fixed 82-byte archive header.
    invalid_index[82..90].copy_from_slice(&1_u64.to_le_bytes());
    rehash_archive(&mut invalid_index);
    assert!(matches!(
        ReplayArchive::from_bytes(&invalid_index),
        Err(ReplayError::InvalidArchiveIndex)
    ));

    let limits = ReplayArchiveLimits {
        max_archive_bytes: archive.to_bytes().len() - 1,
        ..ReplayArchiveLimits::default()
    };
    assert!(matches!(
        ReplayArchive::from_bytes_with_limits(archive.to_bytes(), limits),
        Err(ReplayError::ArchiveTooLarge { .. })
    ));
}

#[test]
fn replay_archive_golden_vector() {
    let (events, _, compiled, initial) = two_batch_replay();
    let archive =
        ReplayArchive::from_events(&events, compiled, initial, ReplayArchiveLimits::default())
            .unwrap();
    assert_eq!(
        (archive.archive_hash().to_hex(), archive.to_bytes().len()),
        (
            "ce6471f2905a5d6db54e779feb3ce029b7755db255843b888c702772d4ddc46b".into(),
            2108,
        )
    );
}

fn rehash_archive(bytes: &mut [u8]) {
    const DOMAIN: &[u8] = b"blob.replay.archive";
    let body_length = bytes.len() - 32;
    let mut hasher = Sha256::new();
    hasher.update((DOMAIN.len() as u64).to_le_bytes());
    hasher.update(DOMAIN);
    hasher.update(REPLAY_FORMAT_VERSION.to_le_bytes());
    hasher.update(&bytes[..body_length]);
    bytes[body_length..].copy_from_slice(&hasher.finalize());
}
