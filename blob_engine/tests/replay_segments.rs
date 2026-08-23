use blob_engine::resolution::{
    ActionRequest, DurationRule, ReferenceCheckpoint, ReferenceRuleset, ReferenceSimulation,
    ReplayBatchEvent, ReplayChainCursor, ReplayManifest, ReplayManifestLimits, ReplayRecorder,
    ReplaySegment, ReplaySegmentDescriptor, ReplaySegmentLimits, ReplayStreamError,
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

fn two_segments() -> (ReplaySegment, ReplaySegment) {
    let mut simulation = ReferenceSimulation::new(3, 1, uniform_rules()).unwrap();
    let left = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 1)
        .unwrap();
    let right = simulation
        .add_cell(simulation.tile(2, 0).unwrap(), 10, 100, 2)
        .unwrap();
    let compiled = simulation.compiled_ruleset_hash();
    let initial = simulation.state_hash();
    let checkpoint_zero = ReferenceCheckpoint::from_simulation(&simulation);
    let mut recorder = ReplayRecorder::new(compiled, initial);
    let cursor_zero = recorder.cursor();
    let mut events: Vec<ReplayBatchEvent> = Vec::new();
    let mut checkpoint_two = None;

    for index in 0..4 {
        simulation.commit_action(left, ActionRequest::Wait).unwrap();
        simulation
            .commit_action(right, ActionRequest::Wait)
            .unwrap();
        events.push(
            recorder
                .record(&simulation.resolve_next_batch().unwrap())
                .unwrap(),
        );
        if index == 1 {
            checkpoint_two = Some(ReferenceCheckpoint::from_simulation(&simulation));
        }
    }

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
        checkpoint_two.unwrap(),
        &events[2..],
        ReplaySegmentLimits::default(),
    )
    .unwrap();
    (first, second)
}

#[test]
fn segments_round_trip_chain_and_support_manifest_seek() {
    let (first, second) = two_segments();
    assert_eq!(first.start_cursor().next_sequence, 0);
    assert_eq!(first.end_cursor().next_sequence, 2);
    assert_eq!(second.start_cursor(), first.end_cursor());
    assert_eq!(second.end_cursor().next_sequence, 4);
    assert_eq!(second.event(0, Default::default()).unwrap().sequence, 2);

    let decoded_first = ReplaySegment::from_bytes(first.to_bytes()).unwrap();
    let decoded_second = ReplaySegment::from_bytes(second.to_bytes()).unwrap();
    assert_eq!(decoded_first.segment_hash(), first.segment_hash());
    assert_eq!(decoded_second.to_bytes(), second.to_bytes());

    let manifest = ReplayManifest::from_segments(
        first.compiled_ruleset_hash(),
        first.start_cursor().previous_state_hash,
        vec![
            ReplaySegmentDescriptor::from_segment(&first),
            ReplaySegmentDescriptor::from_segment(&second),
        ],
        ReplayManifestLimits::default(),
    )
    .unwrap();
    let decoded_manifest = ReplayManifest::from_bytes(manifest.to_bytes()).unwrap();
    assert_eq!(decoded_manifest.to_bytes(), manifest.to_bytes());
    assert_eq!(decoded_manifest.segment_count(), 2);
    assert_eq!(decoded_manifest.event_count(), 4);
    assert_eq!(decoded_manifest.segment_for_event(0).unwrap(), 0);
    assert_eq!(decoded_manifest.segment_for_event(1).unwrap(), 0);
    assert_eq!(decoded_manifest.segment_for_event(2).unwrap(), 1);
    assert_eq!(decoded_manifest.segment_for_event(3).unwrap(), 1);
    assert_eq!(decoded_manifest.final_cursor(), second.end_cursor());
    decoded_manifest.verify_segment(0, &decoded_first).unwrap();
    decoded_manifest.verify_segment(1, &decoded_second).unwrap();
    assert!(decoded_manifest.segment_for_event(4).is_err());
}

#[test]
fn segment_and_manifest_reject_tampering_broken_chains_and_limits() {
    let (first, second) = two_segments();
    let mut tampered = first.to_bytes().to_vec();
    tampered[32] ^= 0x80;
    assert!(matches!(
        ReplaySegment::from_bytes(&tampered),
        Err(ReplayStreamError::IntegrityHashMismatch { .. })
    ));

    let wrong_checkpoint = first.checkpoint().clone();
    assert!(matches!(
        ReplaySegment::from_events(
            second.compiled_ruleset_hash(),
            second.start_cursor(),
            wrong_checkpoint,
            &[second.event(0, Default::default()).unwrap()],
            ReplaySegmentLimits::default(),
        ),
        Err(ReplayStreamError::CheckpointStateMismatch)
    ));

    let descriptors = vec![
        ReplaySegmentDescriptor::from_segment(&first),
        ReplaySegmentDescriptor::from_segment(&second),
    ];
    assert!(matches!(
        ReplayManifest::from_segments(
            first.compiled_ruleset_hash(),
            first.start_cursor().previous_state_hash,
            vec![descriptors[1], descriptors[0]],
            ReplayManifestLimits::default(),
        ),
        Err(ReplayStreamError::ManifestChain(_))
    ));

    let manifest = ReplayManifest::from_segments(
        first.compiled_ruleset_hash(),
        first.start_cursor().previous_state_hash,
        descriptors,
        ReplayManifestLimits::default(),
    )
    .unwrap();
    let unrelated_cursor = ReplayChainCursor {
        next_sequence: 0,
        ..second.start_cursor()
    };
    let unrelated = ReplaySegment::from_events(
        second.compiled_ruleset_hash(),
        unrelated_cursor,
        second.checkpoint().clone(),
        &[second.event(0, Default::default()).unwrap()],
        ReplaySegmentLimits::default(),
    );
    assert!(unrelated.is_err());

    let limits = ReplaySegmentLimits {
        max_events: 1,
        ..ReplaySegmentLimits::default()
    };
    assert!(matches!(
        ReplaySegment::from_bytes_with_limits(first.to_bytes(), limits),
        Err(ReplayStreamError::CountLimit { .. })
    ));
    let limits = ReplayManifestLimits {
        max_segments: 1,
        ..ReplayManifestLimits::default()
    };
    assert!(matches!(
        ReplayManifest::from_bytes_with_limits(manifest.to_bytes(), limits),
        Err(ReplayStreamError::CountLimit { .. })
    ));
}
