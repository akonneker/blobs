//! Artifact-backed native/WASM parity and signed server replay qualification.
//! Requires explicit exported weights and WASM; never silently skips a build.
#[path = "../src/extism_compat.rs"]
mod extism_compat;
#[path = "../../blob_policy/examples/sampled_target_fixture/mod.rs"]
#[allow(dead_code)]
mod fixture;

use blob_engine::{
    engine::{CellConfig, Engine, ReplayStreamConfig, StartingCellLayout},
    mind_runtime::inspect_mind_artifact,
    resolution::{
        CanonicalHash, MatchVerificationLimits, MatchVerificationManifest, MindArtifactBinding,
        MindRuntimeProfile, ReferenceRuleset, ReplayLimits, mind_artifact_hash,
    },
    server_verification::{AttestedMatchManifest, ServerMatchVerifier, ServerSigningKey},
};
use blob_interface::{
    reference_mind::{ReferenceMind, ReferenceMindDecision, ReferenceMindInput},
    reference_mind_converter::{
        ReferenceMindLimits, capnp_to_reference_mind_decision, reference_mind_input_to_capnp,
    },
    types::TeamId,
};
use blob_policy::composite::DeployedPolicy;
use blob_policy::runtime::{
    POLICY_BUILD_SCHEMA_VERSION, POLICY_EXPORT_SCHEMA_VERSION, POLICY_VERIFICATION_SCHEMA_VERSION,
    POLICY_WEIGHT_FORMAT_VERSION,
};
use extism::{Manifest, Plugin, PluginBuilder, Wasm};
use extism_manifest::MemoryOptions;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

type ParitySamples = Arc<Mutex<Vec<(Vec<u8>, ReferenceMindDecision)>>>;

struct WasmMind {
    executor: Arc<extism_compat::ExtismCompatExecutor>,
    samples: ParitySamples,
    policy: DeployedPolicy,
    decisions: Arc<AtomicUsize>,
    retention: Arc<Mutex<[usize; 3]>>,
}
impl ReferenceMind for WasmMind {
    fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision {
        let limits = ReferenceMindLimits::default();
        let bytes = reference_mind_input_to_capnp(input, limits).unwrap();
        let output = self
            .executor
            .call(bytes.clone(), limits.max_action_bytes)
            .expect("WASM inference failed");
        let actual = capnp_to_reference_mind_decision(&output, limits).unwrap();
        assert_eq!(
            actual,
            self.policy.try_decide(input).unwrap(),
            "native/WASM action, signal or memory mismatch"
        );
        if let DeployedPolicy::Composite(policy) = &self.policy {
            let parent = policy.parent.try_decide(input).unwrap();
            let mut restored = actual.clone();
            let mut counts = self.retention.lock().unwrap();
            match (&mut restored.action, &parent.action) {
                (
                    blob_interface::reference_mind::ReferenceMindAction::Move {
                        target_slot,
                        effort,
                    },
                    blob_interface::reference_mind::ReferenceMindAction::Move {
                        target_slot: before,
                        effort: previous,
                    },
                ) => {
                    assert_eq!(effort, previous, "parent effort changed");
                    counts[0] += 1;
                    counts[1] += usize::from(*target_slot != *before);
                    *target_slot = *before;
                }
                _ => counts[2] += 1,
            }
            assert_eq!(
                restored, parent,
                "inherited output changed beyond Move target"
            );
            assert!(blob_policy::action::action_is_commit_legal(
                input,
                &actual.action,
                actual.signal.is_some()
            ));
        }
        let n = self.decisions.fetch_add(1, Ordering::SeqCst);
        if n < 32 {
            self.samples
                .lock()
                .unwrap()
                .push((bytes.clone(), actual.clone()));
            // An unrelated invocation on the same compiled executor cannot affect
            // the original cell; every invocation has a fresh store and instance.
            let mut unrelated = input.clone();
            unrelated.private_memory = vec![0xa5; 31];
            self.executor
                .call(
                    reference_mind_input_to_capnp(&unrelated, limits).unwrap(),
                    limits.max_action_bytes,
                )
                .unwrap();
            assert_eq!(
                output,
                self.executor.call(bytes, limits.max_action_bytes).unwrap(),
                "cross-invocation state leaked"
            );
        }
        actual
    }
    fn reset(&mut self) -> Result<(), String> {
        Ok(())
    }
}
struct Runtime {
    executor: Arc<extism_compat::ExtismCompatExecutor>,
    samples: ParitySamples,
    policy: DeployedPolicy,
    decisions: Arc<AtomicUsize>,
    retention: Arc<Mutex<[usize; 3]>>,
}
impl Runtime {
    fn engine(
        &self,
        seed: u64,
        layout: StartingCellLayout,
        workers: usize,
        wasm: bool,
    ) -> Result<Engine, String> {
        let config = CellConfig {
            starting_cells_per_team: 4,
            ..CellConfig::default()
        };
        let rules = ReferenceRuleset {
            child_core_mass: u64::from(config.min_energy),
            ..ReferenceRuleset::default()
        };
        let mut engine =
            Engine::new_with_match_secret(12, 12, 20, config, Some(seed), [0x35; 32], rules);
        engine.set_starting_cell_layout(layout)?;
        engine.set_reference_mind_parallel_threshold(Some(1));
        engine.set_reference_parallel_threshold(Some(1));
        for team in 0..2 {
            if wasm {
                engine.add_team_with_minds(
                    TeamId(team),
                    (0..workers)
                        .map(|_| WasmMind {
                            executor: self.executor.clone(),
                            samples: self.samples.clone(),
                            policy: self.policy.clone(),
                            decisions: self.decisions.clone(),
                            retention: self.retention.clone(),
                        })
                        .collect(),
                )?;
            } else {
                engine.add_team_with_minds(TeamId(team), vec![self.policy.clone()])?;
            }
        }
        engine.initialize_reference_state()?;
        Ok(engine)
    }
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read_bounded(
    path: &std::path::Path,
    limit: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use std::io::Read;
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err("deployment artifact exceeds verification byte budget".into());
    }
    Ok(bytes)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let export = PathBuf::from(args.next().ok_or("expected export directory")?);
    let output = PathBuf::from(args.next().ok_or("expected new verification directory")?);
    if args.next().is_some() || output.exists() {
        return Err("expected two paths; verification output must be new".into());
    }
    let metadata_bytes = read_bounded(&export.join("export.json"), 1024 * 1024)?;
    let metadata: serde_json::Value = serde_json::from_slice(&metadata_bytes)?;
    let build: serde_json::Value =
        serde_json::from_slice(&read_bounded(&export.join("mind-build.json"), 1024 * 1024)?)?;
    let weights = read_bounded(
        &export.join("weights.bin"),
        blob_policy::runtime::MAX_WEIGHT_BYTES,
    )?;
    let wasm = read_bounded(&export.join("mind.wasm"), 64 * 1024 * 1024)?;
    if metadata["reference_mind_abi"]
        != serde_json::json!(blob_interface::abi::reference_mind_abi_hash())
    {
        return Err("deployment Mind ABI identity mismatch".into());
    }
    let policy = DeployedPolicy::from_bytes(&weights)?;
    if metadata["schema_version"] != POLICY_EXPORT_SCHEMA_VERSION
        || metadata["weight_format"] != POLICY_WEIGHT_FORMAT_VERSION
        || metadata["execution_contract"] != policy.execution_contract()
        || build["schema_version"] != POLICY_BUILD_SCHEMA_VERSION
    {
        return Err("unsupported deployment artifact contract".into());
    }
    if build["export_manifest_sha256"] != hash(&metadata_bytes)
        || metadata["weights_sha256"] != hash(&weights)
        || build["weights_sha256"] != hash(&weights)
        || build["wasm_sha256"] != hash(&wasm)
    {
        return Err("export/build artifact digest mismatch".into());
    }
    let reserved = [
        metadata.pointer("/source_seed_ledger/confirmation"),
        metadata.pointer("/source_training_config/confirmation_seeds"),
    ]
    .into_iter()
    .flatten()
    .filter_map(serde_json::Value::as_array)
    .flatten()
    .map(|seed| seed.as_u64().ok_or("invalid confirmation seed"))
    .collect::<Result<Vec<_>, _>>()?;
    blob_policy::qualification::check_confirmation_reservations(reserved)?;
    let profile = MindRuntimeProfile::ExtismPdkDeterministicV1;
    let inspection = inspect_mind_artifact(&wasm, profile)?;
    let manifest = Manifest::new([Wasm::data(wasm.clone())])
        .with_memory_options(
            MemoryOptions::new()
                .with_max_pages(1024)
                .with_max_var_bytes(0),
        )
        .disallow_all_hosts()
        .with_timeout(Duration::from_secs(2));
    let executor = Arc::new(extism_compat::ExtismCompatExecutor::new(
        &manifest,
        wasmtime::Config::new(),
    )?);
    assert!(
        executor.call(vec![0xff; 17], 1024).is_err(),
        "malformed input must fail"
    );
    let runtime = Runtime {
        executor,
        samples: Arc::new(Mutex::new(Vec::new())),
        policy,
        decisions: Arc::new(AtomicUsize::new(0)),
        retention: Arc::new(Mutex::new([0; 3])),
    };
    let mut boundary_mind = WasmMind {
        executor: runtime.executor.clone(),
        samples: runtime.samples.clone(),
        policy: runtime.policy.clone(),
        decisions: runtime.decisions.clone(),
        retention: runtime.retention.clone(),
    };
    let mut abi_fixture_decisions = 0;
    for count in 0..=32 {
        for variant in 0..4 {
            let mut input = fixture::input(count);
            input.current_tile.plant_energy = 0;
            input.current_tile.plant_capacity = 0;
            input.current_tile.loose_energy = 0;
            input.current_tile.diffuse_energy = 0;
            match variant {
                0 => {}
                1 => {
                    for slot in &mut input.slots {
                        slot.plant_energy = None;
                        slot.plant_capacity = None;
                        slot.loose_energy = None;
                        slot.neighbor = None;
                    }
                    input.action_space.move_targets &= 0x5555_5555;
                }
                2 => {
                    input.slots.reverse();
                    for (index, slot) in input.slots.iter_mut().enumerate() {
                        slot.slot = index as u8;
                    }
                }
                3 => {
                    input.self_state.assimilated_energy = 3;
                }
                _ => unreachable!(),
            }
            for word in [0_u64, 1, 1 << 63, u64::MAX] {
                let mut bytes = [0xa5; 32];
                bytes[..8].copy_from_slice(&word.to_le_bytes());
                input.randomness = blob_interface::randomness::PrivateRandom::from_bytes(bytes);
                boundary_mind.decide(&input);
                abi_fixture_decisions += 1;
            }
        }
    }
    // Public fixture key, never a service signing secret.
    let key = ServerSigningKey::from_seed([0x79; 32])?;
    let abi = CanonicalHash::from_bytes(blob_interface::abi::reference_mind_abi_hash());
    let artifacts = (0..2)
        .map(|slot| MindArtifactBinding {
            slot,
            artifact_hash: mind_artifact_hash(&wasm),
        })
        .collect::<Vec<_>>();
    fs::create_dir(&output)?;
    fs::write(output.join("fixture-public-key.bin"), key.public_key())?;
    let mut cases = Vec::new();
    for (layout_name, layout) in [
        ("line", StartingCellLayout::Line),
        ("checkerboard", StartingCellLayout::Checkerboard),
        ("ring", StartingCellLayout::Ring),
        ("loose-random", StartingCellLayout::LooseRandom),
        ("random", StartingCellLayout::Random),
    ] {
        for &seed in &blob_policy::qualification::DEVELOPMENT_SEEDS[..2] {
            let mut client = runtime.engine(seed, layout, 1, true)?;
            let mut native = runtime.engine(seed, layout, 1, false)?;
            let initial = client.authoritative_state_hash().unwrap();
            assert_eq!(initial, native.authoritative_state_hash().unwrap());
            client.start_reference_replay_streaming(ReplayStreamConfig {
                max_events_per_segment: 64,
                ..ReplayStreamConfig::default()
            })?;
            for _ in 0..64 {
                if client.cells.is_empty() {
                    break;
                }
                client.tick(false)?;
                native.tick(false)?;
                assert_eq!(
                    client.authoritative_state_hash(),
                    native.authoritative_state_hash(),
                    "native/WASM match diverged"
                );
            }
            client.flush_reference_replay_stream()?;
            let mut segments = Vec::new();
            while let Some(segment) = client.take_reference_replay_segment() {
                segments.push(segment);
            }
            let replay = client.reference_replay_stream_manifest()?;
            assert!(!segments.is_empty());
            let match_manifest = MatchVerificationManifest::from_replay(
                CanonicalHash::from_bytes(
                    Sha256::digest(format!("learned-mind-dev-{layout_name}-{seed}")).into(),
                ),
                abi,
                profile,
                initial,
                artifacts.clone(),
                &replay,
                MatchVerificationLimits::default(),
            )?;
            let mut server = runtime.engine(seed, layout, 4, true)?;
            let mut verifier = ServerMatchVerifier::start(
                match_manifest,
                replay.clone(),
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
            assert_eq!(
                client.authoritative_state_hash(),
                server.authoritative_state_hash()
            );
            let attestation = AttestedMatchManifest::sign(verifier.finish()?, &key)?;
            attestation.verify(&key.public_key())?;
            let case_dir = output.join(format!("{layout_name}-{seed}"));
            fs::create_dir(&case_dir)?;
            fs::write(case_dir.join("replay-manifest.bin"), replay.to_bytes())?;
            fs::write(case_dir.join("attestation.bin"), attestation.to_bytes())?;
            for (index, segment) in segments.iter().enumerate() {
                fs::write(
                    case_dir.join(format!("segment-{index}.bin")),
                    segment.to_bytes(),
                )?;
            }
            cases.push(serde_json::json!({"layout":layout_name,"seed":seed,"events":replay.event_count(),"final_state_hash":server.authoritative_state_hash().unwrap().to_hex(),"manifest_hash":replay.manifest_hash().to_hex()}));
        }
    }
    let compiled = PluginBuilder::new(manifest).compile()?;
    for (input, expected) in runtime.samples.lock().unwrap().iter() {
        let mut plugin = Plugin::new_from_compiled(&compiled)?;
        let bytes = plugin.call::<&Vec<u8>, Vec<u8>>("reference_mind_function", input)?;
        assert_eq!(
            &capnp_to_reference_mind_decision(&bytes, ReferenceMindLimits::default())?,
            expected,
            "stock/compatible executor mismatch"
        );
    }
    let decisions = runtime.decisions.load(Ordering::SeqCst);
    assert!(decisions >= 100);
    let retention = *runtime.retention.lock().unwrap();
    let report = serde_json::json!({"schema_version":POLICY_VERIFICATION_SCHEMA_VERSION,"wasm_sha256":hash(&wasm),"weights_sha256":hash(&weights),"decisions":decisions,"native_wasm_decision_and_memory_mismatches":0,"stock_extism_checked_decisions":32,"isolation_checks":32,"malformed_input_rejected":true,"admitted_imports":inspection.function_imports,"cases":cases,"worker_counts":[1,4],"memory_pages":1024,"invocation_timeout_seconds":2,"execution_contract":runtime.policy.execution_contract(),"abi_fixture_decisions":abi_fixture_decisions,"inherited_retention":{"applicable":matches!(runtime.policy,DeployedPolicy::Composite(_)),"move_decisions":retention[0],"changed_move_targets":retention[1],"non_move_decisions":retention[2],"other_output_mismatches":0},"qualification":"bounded 64-batch deployment fixtures and 0..32-slot ABI cases, not gameplay competence or deterministic fuel qualification"});
    fs::write(
        output.join("verification.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
