use std::fs;
use std::path::PathBuf;

use blob_engine::engine::{CellConfig, Engine, ReplayStreamConfig};
use blob_engine::online_service::{
    admit_match_submission, derive_verified_score, online_match_id_hash, FileLeaderboardStore,
    LeaderboardPublication, LeaderboardStoreLimits, OnlineServiceError, OnlineServiceLimits,
    ServerScorePolicy, StoredVerificationSession, SubmittedMindArtifact,
};
use blob_engine::replay_store::{FileReplayStore, ReplayArtifactStore, ReplayStoreLimits};
use blob_engine::resolution::{
    mind_artifact_hash, CanonicalHash, MatchVerificationLimits, MatchVerificationManifest,
    MindArtifactBinding, MindRuntimeProfile, ReferenceRuleset, ReferenceSimulation, ReplayLimits,
    ReplayManifest, ReplaySegment, ReplaySegmentLimits,
};
use blob_engine::server_verification::ServerSigningKey;
use blob_interface::reference_mind::{
    ReferenceMemoryUpdate, ReferenceMind, ReferenceMindAction, ReferenceMindDecision,
    ReferenceMindInput,
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

fn engine(match_secret: [u8; 32]) -> Engine {
    let mut engine = Engine::new_with_match_secret(
        8,
        8,
        20,
        CellConfig {
            starting_cells_per_team: 2,
            ..CellConfig::default()
        },
        Some(91),
        match_secret,
        ReferenceRuleset::default(),
    );
    engine
        .add_team_with_minds(TeamId(0), vec![WaitMind])
        .unwrap();
    engine.initialize_reference_state().unwrap();
    engine
}

struct Fixture {
    segments: Vec<ReplaySegment>,
    replay_manifest: ReplayManifest,
    verification_manifest: MatchVerificationManifest,
    artifact_bytes: Vec<u8>,
    match_id: Vec<u8>,
    match_secret: [u8; 32],
    initial_runtime_hash: CanonicalHash,
}

fn fixture() -> Fixture {
    let match_secret = [0x61; 32];
    let mut client = engine(match_secret);
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
    let artifact_bytes = minimal_wasm_mind();
    let artifacts = vec![MindArtifactBinding {
        slot: 0,
        artifact_hash: mind_artifact_hash(&artifact_bytes),
    }];
    let match_id = b"match-fixture-71".to_vec();
    let match_id_hash = online_match_id_hash(&match_id);
    let verification_manifest = MatchVerificationManifest::from_replay(
        match_id_hash,
        CanonicalHash::from_bytes(blob_interface::abi::reference_mind_abi_hash()),
        MindRuntimeProfile::ExtismPdkDeterministicV1,
        initial_runtime_hash,
        artifacts,
        &replay_manifest,
        MatchVerificationLimits::default(),
    )
    .unwrap();
    Fixture {
        segments: vec![first, second],
        replay_manifest,
        verification_manifest,
        artifact_bytes,
        match_id,
        match_secret,
        initial_runtime_hash,
    }
}

fn minimal_wasm_mind() -> Vec<u8> {
    let mut bytes = vec![
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
        0x03, 0x02, 0x01, 0x00, 0x07, 0x1b, 0x01, 0x17,
    ];
    bytes.extend_from_slice(b"reference_mind_function");
    bytes.extend_from_slice(&[0x00, 0x00, 0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x00, 0x0b]);
    bytes
}

fn temporary_directory(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "blob-online-service-{label}-{}",
        std::process::id()
    ))
}

struct PopulationScore;

impl ServerScorePolicy<()> for PopulationScore {
    fn policy_hash(&self) -> CanonicalHash {
        CanonicalHash::from_bytes([0x81; 32])
    }

    fn derive_score(
        &self,
        simulation: &ReferenceSimulation,
        _trusted_context: &(),
    ) -> Result<i64, String> {
        i64::try_from(simulation.cells().len()).map_err(|error| error.to_string())
    }
}

#[test]
fn admitted_submission_streams_from_store_and_publishes_atomically() {
    let fixture = fixture();
    let replay_root = temporary_directory("replays");
    let leaderboard_root = temporary_directory("leaderboard");
    let _ = fs::remove_dir_all(&replay_root);
    let _ = fs::remove_dir_all(&leaderboard_root);
    let replay_store = FileReplayStore::open(&replay_root, ReplayStoreLimits::default()).unwrap();
    for segment in &fixture.segments {
        replay_store.put_segment(segment).unwrap();
    }
    replay_store
        .publish_manifest("match-fixture-71", &fixture.replay_manifest, None)
        .unwrap();

    let accepted = admit_match_submission(
        fixture.verification_manifest.to_bytes(),
        fixture.replay_manifest.to_bytes(),
        vec![SubmittedMindArtifact {
            slot: 0,
            bytes: fixture.artifact_bytes.clone(),
        }],
        &fixture.match_id,
        CanonicalHash::from_bytes(blob_interface::abi::reference_mind_abi_hash()),
        MindRuntimeProfile::ExtismPdkDeterministicV1,
        fixture.initial_runtime_hash,
        OnlineServiceLimits::default(),
    )
    .unwrap();
    assert_eq!(accepted.artifacts()[0].bytes(), fixture.artifact_bytes);
    assert_eq!(
        accepted.artifacts()[0].artifact_hash(),
        mind_artifact_hash(&fixture.artifact_bytes)
    );

    let mut server = engine(fixture.match_secret);
    let mut session = StoredVerificationSession::start(
        accepted,
        &replay_store,
        server
            .reference_simulation()
            .unwrap()
            .compiled_ruleset_hash(),
        server.reference_simulation().unwrap().state_hash(),
    )
    .unwrap();
    for _ in 0..4 {
        let tick = server.tick(false).unwrap();
        session
            .verify_tick(
                &tick,
                server
                    .reference_simulation()
                    .unwrap()
                    .compiled_ruleset_hash(),
                server.reference_simulation().unwrap().state_hash(),
                ReplayLimits::default(),
            )
            .unwrap();
    }
    let verified = session.finish().unwrap();
    let season_hash = CanonicalHash::from_bytes([0x91; 32]);
    let scored = derive_verified_score(
        verified,
        server.reference_simulation().unwrap(),
        season_hash,
        &PopulationScore,
        &(),
    )
    .unwrap();
    let signing_key = ServerSigningKey::from_seed([0xa1; 32]).unwrap();
    let publication = LeaderboardPublication::sign(scored, &signing_key).unwrap();
    publication.verify(&signing_key.public_key()).unwrap();
    assert_eq!(publication.score(), 2);
    assert_eq!(
        publication.replay_manifest_hash(),
        fixture.replay_manifest.manifest_hash()
    );

    let decoded = LeaderboardPublication::from_bytes(
        publication.to_bytes(),
        2 * 1024 * 1024,
        MatchVerificationLimits::default(),
    )
    .unwrap();
    decoded.verify(&signing_key.public_key()).unwrap();
    for length in 0..publication.to_bytes().len() {
        assert!(LeaderboardPublication::from_bytes(
            &publication.to_bytes()[..length],
            2 * 1024 * 1024,
            MatchVerificationLimits::default(),
        )
        .is_err());
    }

    let leaderboard = FileLeaderboardStore::open(
        &leaderboard_root,
        signing_key.public_key(),
        LeaderboardStoreLimits::default(),
    )
    .unwrap();
    assert!(leaderboard.publish(&publication, &replay_store).unwrap());
    assert!(!leaderboard.publish(&publication, &replay_store).unwrap());
    let loaded = leaderboard
        .load(season_hash, publication.verification_manifest_hash())
        .unwrap()
        .unwrap();
    assert_eq!(loaded.to_bytes(), publication.to_bytes());
    fs::write(
        leaderboard_root
            .join("seasons")
            .join(season_hash.to_hex())
            .join("entries")
            .join(".publication-concurrent.tmp"),
        b"partial temporary publication",
    )
    .unwrap();
    let entries = leaderboard.leaderboard(season_hash, 10).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].score(), 2);

    let wrong_key = ServerSigningKey::from_seed([0xa2; 32]).unwrap();
    let wrong_store = FileLeaderboardStore::open(
        temporary_directory("wrong-key"),
        wrong_key.public_key(),
        LeaderboardStoreLimits::default(),
    )
    .unwrap();
    assert!(wrong_store.publish(&publication, &replay_store).is_err());

    fs::remove_dir_all(replay_root).unwrap();
    fs::remove_dir_all(leaderboard_root).unwrap();
    let _ = fs::remove_dir_all(temporary_directory("wrong-key"));
}

#[test]
fn admission_rejects_unbounded_or_rebound_artifacts_before_execution() {
    let fixture = fixture();
    let one_byte = OnlineServiceLimits {
        max_artifact_bytes: 1,
        max_total_artifact_bytes: 1,
        ..OnlineServiceLimits::default()
    };
    assert!(matches!(
        admit_match_submission(
            fixture.verification_manifest.to_bytes(),
            fixture.replay_manifest.to_bytes(),
            vec![SubmittedMindArtifact {
                slot: 0,
                bytes: fixture.artifact_bytes.clone(),
            }],
            &fixture.match_id,
            CanonicalHash::from_bytes(blob_interface::abi::reference_mind_abi_hash()),
            MindRuntimeProfile::ExtismPdkDeterministicV1,
            fixture.initial_runtime_hash,
            one_byte,
        ),
        Err(OnlineServiceError::ArtifactTooLarge { .. })
    ));

    assert!(matches!(
        admit_match_submission(
            fixture.verification_manifest.to_bytes(),
            fixture.replay_manifest.to_bytes(),
            vec![
                SubmittedMindArtifact {
                    slot: 1,
                    bytes: b"a".to_vec(),
                },
                SubmittedMindArtifact {
                    slot: 1,
                    bytes: b"b".to_vec(),
                },
            ],
            &fixture.match_id,
            CanonicalHash::from_bytes(blob_interface::abi::reference_mind_abi_hash()),
            MindRuntimeProfile::ExtismPdkDeterministicV1,
            fixture.initial_runtime_hash,
            OnlineServiceLimits::default(),
        ),
        Err(OnlineServiceError::ArtifactSlotsNotStrictlyOrdered)
    ));

    let mut different_valid_mind = minimal_wasm_mind();
    let immediate = different_valid_mind.len() - 2;
    different_valid_mind[immediate] = 1;
    assert!(admit_match_submission(
        fixture.verification_manifest.to_bytes(),
        fixture.replay_manifest.to_bytes(),
        vec![SubmittedMindArtifact {
            slot: 0,
            bytes: different_valid_mind,
        }],
        &fixture.match_id,
        CanonicalHash::from_bytes(blob_interface::abi::reference_mind_abi_hash()),
        MindRuntimeProfile::ExtismPdkDeterministicV1,
        fixture.initial_runtime_hash,
        OnlineServiceLimits::default(),
    )
    .is_err());

    assert!(matches!(
        admit_match_submission(
            fixture.verification_manifest.to_bytes(),
            fixture.replay_manifest.to_bytes(),
            vec![SubmittedMindArtifact {
                slot: 0,
                bytes: b"not WebAssembly".to_vec(),
            }],
            &fixture.match_id,
            CanonicalHash::from_bytes(blob_interface::abi::reference_mind_abi_hash()),
            MindRuntimeProfile::ExtismPdkDeterministicV1,
            fixture.initial_runtime_hash,
            OnlineServiceLimits::default(),
        ),
        Err(OnlineServiceError::ArtifactRuntimeProfile { slot: 0, .. })
    ));
}
