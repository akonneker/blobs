use blob_engine::resolution::{
    ActionRequest, CellKey, DeltaError, DurationRule, LocalSlot, ReferenceRuleset,
    ReferenceSimulation, ReplayArchive, ReplayArchiveLimits, ReplayBatchEvent, ReplayCommitment,
    ReplayError, ReplayLimits, ReplayRecorder,
};
use blob_interface::reference_mind::ReferenceMemoryUpdate;

const EAST: LocalSlot = LocalSlot(4);

fn split_event() -> (ReplayBatchEvent, ReplayArchive, ReferenceSimulation) {
    let uniform = DurationRule::new(1024, 0, 1);
    let rules = ReferenceRuleset {
        wait_duration: uniform,
        move_duration: uniform,
        attack_duration: uniform,
        consume_duration: uniform,
        split_duration: uniform,
        regurgitate_duration: uniform,
        ..ReferenceRuleset::default()
    };
    let mut simulation = ReferenceSimulation::new(2, 1, rules).unwrap();
    let parent = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 1)
        .unwrap();
    let compiled = simulation.compiled_ruleset_hash();
    let initial = simulation.state_hash();
    let mut recorder = ReplayRecorder::new(compiled, initial);
    let request = ActionRequest::Split {
        target: EAST,
        child_allocation: 30,
        marker: 7,
        private_memory: vec![1, 2, 3, 4],
    };
    let started_at = simulation.now();
    let next_private_memory = vec![9, 8, 7];
    let receipt = simulation
        .commit_decision(parent, request.clone(), next_private_memory.clone())
        .unwrap();
    let commitment = ReplayCommitment {
        actor: parent,
        request,
        signal: None,
        memory_update: ReferenceMemoryUpdate::Replace(next_private_memory),
        started_at,
        receipt,
    };
    let event = recorder
        .record_with_commitments(&simulation.resolve_next_batch().unwrap(), vec![commitment])
        .unwrap();
    let archive = ReplayArchive::from_events(
        std::slice::from_ref(&event),
        compiled,
        initial,
        ReplayArchiveLimits::default(),
    )
    .unwrap();
    (event, archive, simulation)
}

#[test]
fn every_truncated_frame_and_archive_prefix_is_rejected() {
    let (event, archive, _) = split_event();
    let frame = event.to_bytes();
    for length in 0..frame.len() {
        assert!(
            ReplayBatchEvent::from_bytes(&frame[..length]).is_err(),
            "accepted frame prefix of {length} bytes"
        );
    }
    for length in 0..archive.to_bytes().len() {
        assert!(
            ReplayArchive::from_bytes(&archive.to_bytes()[..length]).is_err(),
            "accepted archive prefix of {length} bytes"
        );
    }
}

#[test]
fn malformed_delta_order_and_unchanged_entries_are_rejected() {
    let (event, _, simulation) = split_event();
    assert!(event.delta.cells.len() >= 2);
    assert!(!event.delta.tiles.is_empty());

    let mut reordered = event.clone();
    reordered.delta.cells.swap(0, 1);
    assert_eq!(
        ReplayBatchEvent::from_bytes(&reordered.to_bytes()),
        Err(ReplayError::NonCanonicalOrder("delta cells"))
    );

    let mut unchanged = event.clone();
    unchanged.delta.tiles[0].after = unchanged.delta.tiles[0].before.clone();
    assert_eq!(
        ReplayBatchEvent::from_bytes(&unchanged.to_bytes()),
        Err(ReplayError::InvalidBatch(
            "delta contains an unchanged tile"
        ))
    );

    let post = simulation.canonical_state();
    let mut pre = post.clone();
    event.delta.apply_backward(&mut pre).unwrap();
    let mut malformed = event.delta.clone();
    malformed.cells.swap(0, 1);
    assert_eq!(
        malformed.apply_forward(&mut pre),
        Err(DeltaError::NonCanonicalOrder)
    );
    assert_ne!(pre, post);
}

#[test]
fn duplicate_canonical_resources_are_rejected() {
    let (event, _, _) = split_event();

    let mut duplicate_commitment = event.clone();
    duplicate_commitment
        .commitments
        .push(duplicate_commitment.commitments[0].clone());
    duplicate_commitment
        .commitments
        .sort_by_key(|commitment| (commitment.started_at, commitment.actor));
    assert_eq!(
        ReplayBatchEvent::from_bytes(&duplicate_commitment.to_bytes()),
        Err(ReplayError::NonCanonicalOrder("commitments"))
    );

    let mut duplicate_claim = event.clone();
    duplicate_claim.claims.push(duplicate_claim.claims[0]);
    duplicate_claim
        .claims
        .sort_by_key(|claim| (claim.key, claim.mode, claim.actor));
    assert_eq!(
        ReplayBatchEvent::from_bytes(&duplicate_claim.to_bytes()),
        Err(ReplayError::NonCanonicalOrder("claims"))
    );

    let mut duplicate_birth = event.clone();
    duplicate_birth.births.push(duplicate_birth.births[0]);
    duplicate_birth.births.sort();
    assert_eq!(
        ReplayBatchEvent::from_bytes(&duplicate_birth.to_bytes()),
        Err(ReplayError::NonCanonicalOrder("births"))
    );

    let mut duplicate_death = event;
    duplicate_death.deaths = vec![CellKey(9), CellKey(9)];
    assert_eq!(
        ReplayBatchEvent::from_bytes(&duplicate_death.to_bytes()),
        Err(ReplayError::NonCanonicalOrder("deaths"))
    );
}

#[test]
fn every_nested_collection_limit_is_enforced() {
    let (event, archive, _) = split_event();
    let frame = event.to_bytes();
    let cases = [
        ReplayLimits {
            max_commitments: 0,
            ..ReplayLimits::default()
        },
        ReplayLimits {
            max_outcomes: 0,
            ..ReplayLimits::default()
        },
        ReplayLimits {
            max_claims: 0,
            ..ReplayLimits::default()
        },
        ReplayLimits {
            max_births: 0,
            ..ReplayLimits::default()
        },
        ReplayLimits {
            max_delta_tiles: 0,
            ..ReplayLimits::default()
        },
        ReplayLimits {
            max_delta_cells: 0,
            ..ReplayLimits::default()
        },
        ReplayLimits {
            max_private_memory_bytes: 0,
            ..ReplayLimits::default()
        },
    ];
    for limits in cases {
        assert!(matches!(
            ReplayBatchEvent::from_bytes_with_limits(&frame, limits),
            Err(ReplayError::CountLimit { .. })
        ));
    }

    let limits = ReplayArchiveLimits {
        max_events: 0,
        ..ReplayArchiveLimits::default()
    };
    assert!(matches!(
        ReplayArchive::from_bytes_with_limits(archive.to_bytes(), limits),
        Err(ReplayError::CountLimit {
            field: "archive events",
            ..
        })
    ));
}

#[test]
fn bounded_decoders_do_not_panic_on_deterministic_garbage() {
    let mut state = 0x1234_5678_9abc_def0_u64;
    for case in 0..2_000_usize {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let length = (state as usize) % 1024;
        let mut bytes = vec![0_u8; length];
        for (index, byte) in bytes.iter_mut().enumerate() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(index as u64 + case as u64 + 1);
            *byte = (state >> 32) as u8;
        }
        let _ = ReplayBatchEvent::from_bytes_with_limits(&bytes, ReplayLimits::default());
        let _ = ReplayArchive::from_bytes_with_limits(&bytes, ReplayArchiveLimits::default());
    }
}

#[test]
fn empty_archive_round_trips_canonically() {
    let compiled = blob_engine::resolution::CanonicalHash::from_bytes([3; 32]);
    let initial = blob_engine::resolution::CanonicalHash::from_bytes([4; 32]);
    let archive =
        ReplayArchive::from_events(&[], compiled, initial, ReplayArchiveLimits::default()).unwrap();
    let decoded = ReplayArchive::from_bytes(archive.to_bytes()).unwrap();
    assert_eq!(decoded.event_count(), 0);
    assert_eq!(decoded.to_bytes(), archive.to_bytes());
}
