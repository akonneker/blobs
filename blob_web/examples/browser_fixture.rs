//! Deterministic, native-signed replay fixtures for the real Chromium/WASM tests.
//! The fixed signing seed is test data and must never be used by a service.

use std::path::PathBuf;

use blob_engine::engine::{CellConfig, Engine, ReplayStreamConfig};
use blob_engine::resolution::{
    mind_artifact_hash, CanonicalHash, MatchAttestation, MatchVerificationLimits,
    MatchVerificationManifest, MindArtifactBinding, MindRuntimeProfile, ReferenceRuleset,
    ReplayLimits, ReplayManifest, ReplaySegment, ReplaySegmentLimits,
};
use blob_engine::server_verification::{
    AttestedMatchManifest, ServerMatchVerifier, ServerSigningKey,
};
use blob_interface::reference_mind::{
    ReferenceMemoryUpdate, ReferenceMind, ReferenceMindAction, ReferenceMindDecision,
    ReferenceMindInput,
};
use blob_interface::types::TeamId;

struct WaitMind;

impl ReferenceMind for WaitMind {
    fn decide(&mut self, _: &ReferenceMindInput) -> ReferenceMindDecision {
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

fn engine(seed: u64) -> Engine {
    let config = CellConfig {
        starting_cells_per_team: 2,
        ..CellConfig::default()
    };
    let rules = ReferenceRuleset {
        child_core_mass: u64::from(config.min_energy),
        ..ReferenceRuleset::default()
    };
    let mut engine = Engine::new_with_match_secret(4, 4, 20, config, Some(seed), [0x31; 32], rules);
    engine
        .add_team_with_minds(TeamId(0), vec![WaitMind])
        .unwrap();
    engine.initialize_reference_state().unwrap();
    engine
}

fn record(engine: &mut Engine) -> (ReplayManifest, Vec<ReplaySegment>) {
    engine
        .start_reference_replay_streaming(ReplayStreamConfig {
            max_events_per_segment: 2,
            ..ReplayStreamConfig::default()
        })
        .unwrap();
    let segments = (0..2)
        .map(|_| {
            engine.step(2, false).unwrap();
            engine.take_reference_replay_segment().unwrap()
        })
        .collect();
    (engine.reference_replay_stream_manifest().unwrap(), segments)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("expected output directory")?,
    );
    std::fs::create_dir_all(&output)?;
    let mut client = engine(73);
    let initial_runtime_hash = client.authoritative_state_hash().unwrap();
    let (manifest, segments) = record(&mut client);
    let (other_manifest, _) = record(&mut engine(74));
    assert_ne!(manifest.manifest_hash(), other_manifest.manifest_hash());
    let abi = CanonicalHash::from_bytes(blob_interface::abi::reference_mind_abi_hash());
    let profile = MindRuntimeProfile::ExtismPdkDeterministicV1;
    let artifacts = vec![MindArtifactBinding {
        slot: 0,
        artifact_hash: mind_artifact_hash(b"browser-test-wait-mind"),
    }];
    let verification = |match_id| {
        MatchVerificationManifest::from_replay(
            CanonicalHash::from_bytes([match_id; 32]),
            abi,
            profile,
            initial_runtime_hash,
            artifacts.clone(),
            &manifest,
            MatchVerificationLimits::default(),
        )
        .unwrap()
    };
    let mut server = engine(73);
    let mut verifier = ServerMatchVerifier::start(
        verification(0x42),
        manifest.clone(),
        abi,
        profile,
        server.authoritative_state_hash().unwrap(),
        &artifacts,
        segments[0].clone(),
        server
            .reference_simulation()
            .unwrap()
            .compiled_ruleset_hash(),
        server.reference_simulation().unwrap().state_hash(),
    )?;
    for (index, segment) in segments.iter().enumerate() {
        if index > 0 {
            verifier.advance_segment(
                segment.clone(),
                server
                    .reference_simulation()
                    .unwrap()
                    .compiled_ruleset_hash(),
                server.reference_simulation().unwrap().state_hash(),
            )?;
        }
        for _ in 0..segment.event_count() {
            verifier.verify_tick(&server.tick(false)?, ReplayLimits::default())?;
        }
    }
    let key = ServerSigningKey::from_seed([0x77; 32])?;
    let signed = AttestedMatchManifest::sign(verifier.finish()?, &key)?;
    signed.verify(&key.public_key())?;
    let parsed = MatchAttestation::from_bytes(
        signed.to_bytes(),
        2 * 1024 * 1024,
        MatchVerificationLimits::default(),
    )?;
    let mut signature = *parsed.signature();
    signature[0] ^= 1;
    let bad_signature =
        MatchAttestation::from_signature(verification(0x42), key.key_id(), signature);
    let bad_message =
        MatchAttestation::from_signature(verification(0x43), key.key_id(), *parsed.signature());
    for (name, bytes) in [
        ("manifest.bin", manifest.to_bytes()),
        ("other-manifest.bin", other_manifest.to_bytes()),
        ("attestation.bin", signed.to_bytes()),
        ("bad-signature.bin", bad_signature.to_bytes()),
        ("bad-message.bin", bad_message.to_bytes()),
    ] {
        std::fs::write(output.join(name), bytes)?;
    }
    for (index, segment) in segments.iter().enumerate() {
        std::fs::write(
            output.join(format!("segment-{index}.bin")),
            segment.to_bytes(),
        )?;
    }
    std::fs::write(output.join("public-key.bin"), key.public_key())?;
    std::fs::write(
        output.join("wrong-key.bin"),
        ServerSigningKey::from_seed([0x78; 32])?.public_key(),
    )?;
    std::fs::write(output.join("expected.json"), format!(
        "{{\"manifestHash\":\"{}\",\"initialStateHash\":\"{}\",\"finalStateHash\":\"{}\",\"eventCount\":\"{}\",\"segmentCount\":{}}}\n",
        manifest.manifest_hash().to_hex(), manifest.initial_state_hash().to_hex(),
        server.reference_simulation().unwrap().state_hash().to_hex(),
        manifest.event_count(), segments.len(),
    ))?;
    // Re-parse the generated segments under the browser's ordinary limits.
    for segment in segments {
        ReplaySegment::from_bytes_with_limits(segment.to_bytes(), ReplaySegmentLimits::default())?;
    }
    Ok(())
}
