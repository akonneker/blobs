use blob_engine::resolution::{
    ActionRequest, DurationRule, ReferenceCheckpoint, ReferenceRuleset, ReferenceSimulation,
    ReplayArchive, ReplayArchiveLimits, ReplayBundle, ReplayBundleError, ReplayBundleLimits,
    ReplayRecorder,
};

fn uniform_rules() -> ReferenceRuleset {
    let duration = DurationRule::new(1024, 0, 1);
    ReferenceRuleset {
        wait_duration: duration,
        move_duration: duration,
        attack_duration: duration,
        consume_duration: duration,
        split_duration: duration,
        regurgitate_duration: duration,
        digestion_rate_numerator: 0,
        metabolism_rate_numerator: 0,
        diffusion_rate_numerator: 0,
        ..ReferenceRuleset::default()
    }
}

fn two_event_bundle_parts() -> (
    ReplayArchive,
    Vec<(u64, ReferenceCheckpoint)>,
    ReferenceCheckpoint,
) {
    let mut simulation = ReferenceSimulation::new(3, 1, uniform_rules()).unwrap();
    let left = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 1)
        .unwrap();
    let right = simulation
        .add_cell(simulation.tile(2, 0).unwrap(), 10, 100, 2)
        .unwrap();
    let genesis = ReferenceCheckpoint::from_simulation(&simulation);
    let compiled = simulation.compiled_ruleset_hash();
    let initial = simulation.state_hash();
    let mut recorder = ReplayRecorder::new(compiled, initial);

    simulation.commit_action(left, ActionRequest::Wait).unwrap();
    simulation
        .commit_action(right, ActionRequest::Wait)
        .unwrap();
    let first = recorder
        .record(&simulation.resolve_next_batch().unwrap())
        .unwrap();

    simulation.commit_action(left, ActionRequest::Wait).unwrap();
    simulation
        .commit_action(right, ActionRequest::Wait)
        .unwrap();
    let second = recorder
        .record(&simulation.resolve_next_batch().unwrap())
        .unwrap();
    let final_checkpoint = ReferenceCheckpoint::from_simulation(&simulation);
    let archive = ReplayArchive::from_events(
        &[first, second],
        compiled,
        initial,
        ReplayArchiveLimits::default(),
    )
    .unwrap();
    (
        archive,
        vec![(0, genesis.clone()), (2, final_checkpoint.clone())],
        genesis,
    )
}

#[test]
fn replay_bundle_round_trips_and_seeks_from_the_nearest_checkpoint() {
    let (archive, checkpoints, _) = two_event_bundle_parts();
    let bundle =
        ReplayBundle::from_parts(archive, checkpoints, ReplayBundleLimits::default()).unwrap();
    let decoded = ReplayBundle::from_bytes(bundle.to_bytes()).unwrap();

    assert_eq!(decoded.to_bytes(), bundle.to_bytes());
    assert_eq!(decoded.archive().event_count(), 2);
    assert_eq!(decoded.checkpoint_count(), 2);
    assert_eq!(decoded.seek(0).unwrap().events, 0..0);
    assert_eq!(decoded.seek(1).unwrap().events, 0..1);
    let final_seek = decoded.seek(2).unwrap();
    assert_eq!(final_seek.checkpoint_index, 1);
    assert_eq!(final_seek.checkpoint_applied_events, 2);
    assert_eq!(final_seek.events, 2..2);
    assert!(decoded.seek(3).is_err());
}

#[test]
fn replay_bundle_rejects_wrong_checkpoint_links_tampering_and_limits() {
    let (archive, checkpoints, genesis) = two_event_bundle_parts();
    assert!(matches!(
        ReplayBundle::from_parts(
            archive.clone(),
            vec![(0, genesis.clone()), (1, genesis)],
            ReplayBundleLimits::default(),
        ),
        Err(ReplayBundleError::CheckpointStateMismatch { sequence: 1 })
    ));

    let bundle =
        ReplayBundle::from_parts(archive, checkpoints, ReplayBundleLimits::default()).unwrap();
    let mut tampered = bundle.to_bytes().to_vec();
    tampered[24] ^= 0x80;
    assert!(ReplayBundle::from_bytes(&tampered).is_err());

    let limits = ReplayBundleLimits {
        max_checkpoints: 1,
        ..ReplayBundleLimits::default()
    };
    assert!(ReplayBundle::from_bytes_with_limits(bundle.to_bytes(), limits).is_err());

    let limits = ReplayBundleLimits {
        max_bundle_bytes: bundle.to_bytes().len() - 1,
        ..ReplayBundleLimits::default()
    };
    assert!(ReplayBundle::from_bytes_with_limits(bundle.to_bytes(), limits).is_err());
}
