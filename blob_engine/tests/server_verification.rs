use blob_engine::engine::{CellConfig, Engine, ReplayStreamConfig, TickEvents};
use blob_engine::resolution::{
    mind_artifact_hash, BoundaryRule, CanonicalHash, MatchVerificationError,
    MatchVerificationLimits, MatchVerificationManifest, MindArtifactBinding, MindRuntimeProfile,
    NeighborhoodSpec, ReferenceRuleset, ReplayLimits, ReplayManifest, ReplaySegment,
    ReplaySegmentLimits,
};
use blob_engine::server_verification::{
    AttestedMatchManifest, ServerMatchVerifier, ServerSigningKey, ServerVerificationError,
};
use blob_interface::reference_mind::{
    ReferenceEffort, ReferenceMemoryUpdate, ReferenceMind, ReferenceMindAction,
    ReferenceMindDecision, ReferenceMindInput,
};
use blob_interface::types::TeamId;

struct WaitMind;

impl ReferenceMind for WaitMind {
    fn decide(&mut self, _input: &ReferenceMindInput) -> ReferenceMindDecision {
        ReferenceMindDecision {
            action: ReferenceMindAction::Wait,
            signal: None,
            memory_update: ReferenceMemoryUpdate::Retain,
        }
    }

    fn reset(&mut self) -> Result<(), String> {
        Ok(())
    }
}

struct EastMind;

impl ReferenceMind for EastMind {
    fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision {
        let action = input
            .slots
            .iter()
            .find(|slot| slot.dx == 1 && slot.dy == 0)
            .map_or(ReferenceMindAction::Wait, |slot| {
                ReferenceMindAction::Move {
                    target_slot: slot.slot,
                    effort: ReferenceEffort::Standard,
                }
            });
        ReferenceMindDecision {
            action,
            signal: None,
            memory_update: ReferenceMemoryUpdate::Retain,
        }
    }

    fn reset(&mut self) -> Result<(), String> {
        Ok(())
    }
}

fn cell_config() -> CellConfig {
    CellConfig {
        starting_cells_per_team: 2,
        ..CellConfig::default()
    }
}

fn engine(mind: impl ReferenceMind + 'static, match_secret: [u8; 32]) -> Engine {
    let config = cell_config();
    let rules = ReferenceRuleset {
        neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Wrap),
        child_core_mass: u64::from(config.min_energy),
        ..ReferenceRuleset::default()
    };
    let mut engine =
        Engine::new_with_match_secret(12, 12, 20, config, Some(73), match_secret, rules);
    engine.add_team_with_minds(TeamId(0), vec![mind]).unwrap();
    engine.initialize_reference_state().unwrap();
    engine
}

struct Fixture {
    segments: Vec<ReplaySegment>,
    replay_manifest: ReplayManifest,
    initial_runtime_hash: CanonicalHash,
    artifacts: Vec<MindArtifactBinding>,
    verification_manifest: MatchVerificationManifest,
    match_secret: [u8; 32],
}

fn fixture() -> Fixture {
    let match_secret = [0x31; 32];
    let mut client = engine(WaitMind, match_secret);
    let initial_runtime_hash = client.authoritative_state_hash().unwrap();
    client
        .start_reference_replay_streaming(ReplayStreamConfig {
            max_events_per_segment: 2,
            max_event_bytes_per_segment: usize::MAX / 2,
            segment_limits: ReplaySegmentLimits {
                max_segment_bytes: usize::MAX,
                ..ReplaySegmentLimits::default()
            },
            ..ReplayStreamConfig::default()
        })
        .unwrap();

    client.step(2, false).unwrap();
    let first = client.take_reference_replay_segment().unwrap();
    client.step(2, false).unwrap();
    let second = client.take_reference_replay_segment().unwrap();
    let replay_manifest = client.reference_replay_stream_manifest().unwrap();
    replay_manifest.verify_segment(0, &first).unwrap();
    replay_manifest.verify_segment(1, &second).unwrap();

    let artifacts = vec![MindArtifactBinding {
        slot: 0,
        artifact_hash: mind_artifact_hash(b"wait-mind.wasm.v1"),
    }];
    let verification_manifest = MatchVerificationManifest::from_replay(
        CanonicalHash::from_bytes([0x42; 32]),
        CanonicalHash::from_bytes(blob_interface::abi::reference_mind_abi_hash()),
        MindRuntimeProfile::ExtismPdkDeterministicV1,
        initial_runtime_hash,
        artifacts.clone(),
        &replay_manifest,
        MatchVerificationLimits::default(),
    )
    .unwrap();

    Fixture {
        segments: vec![first, second],
        replay_manifest,
        initial_runtime_hash,
        artifacts,
        verification_manifest,
        match_secret,
    }
}

fn start_server_verifier(fixture: &Fixture, server: &Engine) -> ServerMatchVerifier {
    ServerMatchVerifier::start(
        fixture.verification_manifest.clone(),
        fixture.replay_manifest.clone(),
        CanonicalHash::from_bytes(blob_interface::abi::reference_mind_abi_hash()),
        MindRuntimeProfile::ExtismPdkDeterministicV1,
        server.authoritative_state_hash().unwrap(),
        &fixture.artifacts,
        fixture.segments[0].clone(),
        server
            .reference_simulation()
            .unwrap()
            .compiled_ruleset_hash(),
        server.reference_simulation().unwrap().state_hash(),
    )
    .unwrap()
}

#[test]
fn server_reexecutes_every_mind_decision_before_signing() {
    let fixture = fixture();
    let mut server = engine(WaitMind, fixture.match_secret);
    let mut verifier = start_server_verifier(&fixture, &server);

    for _ in 0..2 {
        verifier
            .verify_tick(&server.tick(false).unwrap(), ReplayLimits::default())
            .unwrap();
    }
    verifier
        .advance_segment(
            fixture.segments[1].clone(),
            server
                .reference_simulation()
                .unwrap()
                .compiled_ruleset_hash(),
            server.reference_simulation().unwrap().state_hash(),
        )
        .unwrap();
    for _ in 0..2 {
        verifier
            .verify_tick(&server.tick(false).unwrap(), ReplayLimits::default())
            .unwrap();
    }

    let verified = verifier.finish().unwrap();
    assert_eq!(
        verified.manifest().manifest_hash(),
        fixture.verification_manifest.manifest_hash()
    );
    let signing_key = ServerSigningKey::from_seed([0x77; 32]).unwrap();
    let public_key = signing_key.public_key();
    let attested = AttestedMatchManifest::sign(verified, &signing_key).unwrap();
    attested.verify(&public_key).unwrap();

    let decoded = AttestedMatchManifest::from_bytes(
        attested.to_bytes(),
        2 * 1024 * 1024,
        MatchVerificationLimits::default(),
    )
    .unwrap();
    decoded.verify(&public_key).unwrap();
    assert_eq!(decoded.key_id(), signing_key.key_id());
    assert_eq!(
        decoded.manifest().manifest_hash(),
        fixture.verification_manifest.manifest_hash()
    );
    let wrong_key = ServerSigningKey::from_seed([0x78; 32]).unwrap();
    assert!(matches!(
        decoded.verify(&wrong_key.public_key()),
        Err(ServerVerificationError::SigningKeyMismatch)
    ));
    for length in 0..attested.to_bytes().len() {
        assert!(AttestedMatchManifest::from_bytes(
            &attested.to_bytes()[..length],
            2 * 1024 * 1024,
            MatchVerificationLimits::default(),
        )
        .is_err());
    }

    let mut tampered = attested.to_bytes().to_vec();
    *tampered.last_mut().unwrap() ^= 1;
    let tampered = AttestedMatchManifest::from_bytes(
        &tampered,
        2 * 1024 * 1024,
        MatchVerificationLimits::default(),
    )
    .unwrap();
    assert!(matches!(
        tampered.verify(&public_key),
        Err(ServerVerificationError::InvalidSignature)
    ));
}

#[test]
fn wrong_mind_artifact_runtime_and_commitments_are_rejected() {
    let fixture = fixture();

    let wrong_artifacts = vec![MindArtifactBinding {
        slot: 0,
        artifact_hash: mind_artifact_hash(b"east-mind.wasm.v1"),
    }];
    assert_eq!(
        fixture.verification_manifest.verify_context(
            CanonicalHash::from_bytes(blob_interface::abi::reference_mind_abi_hash()),
            MindRuntimeProfile::ExtismPdkDeterministicV1,
            fixture.initial_runtime_hash,
            &wrong_artifacts,
            &fixture.replay_manifest,
        ),
        Err(MatchVerificationError::ArtifactMismatch)
    );

    let wrong_secret_server = engine(WaitMind, [0x99; 32]);
    assert_eq!(
        fixture.verification_manifest.verify_context(
            CanonicalHash::from_bytes(blob_interface::abi::reference_mind_abi_hash()),
            MindRuntimeProfile::ExtismPdkDeterministicV1,
            wrong_secret_server.authoritative_state_hash().unwrap(),
            &fixture.artifacts,
            &fixture.replay_manifest,
        ),
        Err(MatchVerificationError::RuntimeMismatch)
    );
    assert_eq!(
        wrong_secret_server
            .reference_simulation()
            .unwrap()
            .state_hash(),
        fixture.segments[0].start_cursor().previous_state_hash
    );

    let mut wrong_mind_server = engine(EastMind, fixture.match_secret);
    let mut verifier = start_server_verifier(&fixture, &wrong_mind_server);
    let tick: TickEvents = wrong_mind_server.tick(false).unwrap();
    assert!(matches!(
        verifier.verify_tick(&tick, ReplayLimits::default()),
        Err(ServerVerificationError::CommitmentMismatch { sequence: 0 })
    ));
}

#[test]
fn manifests_are_bounded_canonical_and_integrity_checked() {
    let fixture = fixture();
    let bytes = fixture.verification_manifest.to_bytes();
    assert_eq!(
        fixture.verification_manifest.manifest_hash().to_hex(),
        "c7f51ca0fcb4d54cf9c33a8c2e0459c04bf7aa6d9bf0a26fbe59b629f1185389"
    );
    assert_eq!(bytes.len(), 365);
    let decoded = MatchVerificationManifest::from_bytes(bytes).unwrap();
    assert_eq!(decoded, fixture.verification_manifest);
    assert_eq!(
        decoded.mind_runtime_profile(),
        MindRuntimeProfile::ExtismPdkDeterministicV1
    );
    for length in 0..bytes.len() {
        assert!(MatchVerificationManifest::from_bytes(&bytes[..length]).is_err());
    }

    let mut tampered = bytes.to_vec();
    tampered[32] ^= 1;
    assert_eq!(
        MatchVerificationManifest::from_bytes(&tampered),
        Err(MatchVerificationError::HashMismatch)
    );
    assert!(matches!(
        MatchVerificationManifest::from_bytes_with_limits(
            bytes,
            MatchVerificationLimits {
                max_bytes: bytes.len() - 1,
                ..MatchVerificationLimits::default()
            }
        ),
        Err(MatchVerificationError::TooLarge { .. })
    ));
    assert!(matches!(
        MatchVerificationManifest::from_bytes_with_limits(
            bytes,
            MatchVerificationLimits {
                max_mind_artifacts: 0,
                ..MatchVerificationLimits::default()
            }
        ),
        Err(MatchVerificationError::TooManyArtifacts { .. })
    ));

    let duplicate = vec![fixture.artifacts[0], fixture.artifacts[0]];
    assert_eq!(
        MatchVerificationManifest::from_replay(
            CanonicalHash::from_bytes([1; 32]),
            CanonicalHash::from_bytes(blob_interface::abi::reference_mind_abi_hash()),
            MindRuntimeProfile::ExtismPdkDeterministicV1,
            fixture.initial_runtime_hash,
            duplicate,
            &fixture.replay_manifest,
            MatchVerificationLimits::default(),
        ),
        Err(MatchVerificationError::NonCanonicalArtifacts)
    );
}
