//! Freeze a verified cloning checkpoint for the portable learned Mind.
use crate::{
    behavior_cloning::BehaviorCloningArtifact, opponent::SnapshotPolicyMind,
    policy_artifact::load_behavior_clone,
};
use blob_engine::{
    engine::{CellConfig, Engine},
    resolution::ReferenceRuleset,
};
use blob_interface::{
    reference_mind::{
        ReferenceMemoryUpdate, ReferenceMind, ReferenceMindDecision, ReferenceMindInput,
    },
    types::TeamId,
};
use blob_policy::runtime::{
    FrozenPolicy, EXECUTION_CONTRACT, POLICY_EXPORT_SCHEMA_VERSION, POLICY_WEIGHT_FORMAT_VERSION,
};
use burn::{backend::NdArray, tensor::backend::Backend};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::Path,
    sync::{Arc, Mutex},
};

#[derive(Default, Serialize)]
struct Parity {
    decisions: usize,
    action_or_signal_mismatches: usize,
    memory_byte_mismatches: usize,
    max_memory_quantum_difference: u32,
}
struct Compare {
    burn: SnapshotPolicyMind<NdArray<f32>>,
    frozen: FrozenPolicy,
    report: Arc<Mutex<Parity>>,
}
impl ReferenceMind for Compare {
    fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision {
        let expected = self.burn.decide(input);
        let actual = self.frozen.try_decide(input).expect("frozen policy failed");
        let mut r = self.report.lock().unwrap();
        r.decisions += 1;
        if expected.action != actual.action || expected.signal != actual.signal {
            r.action_or_signal_mismatches += 1;
        }
        if expected.memory_update != actual.memory_update {
            r.memory_byte_mismatches += 1;
            match (&expected.memory_update, &actual.memory_update) {
                (ReferenceMemoryUpdate::Replace(a), ReferenceMemoryUpdate::Replace(b))
                    if a.len() == b.len() && a[..8] == b[..8] =>
                {
                    for (a, b) in a[8..].chunks_exact(2).zip(b[8..].chunks_exact(2)) {
                        let a = i16::from_le_bytes(a.try_into().unwrap()) as i32;
                        let b = i16::from_le_bytes(b.try_into().unwrap()) as i32;
                        r.max_memory_quantum_difference =
                            r.max_memory_quantum_difference.max(a.abs_diff(b));
                    }
                }
                _ => r.max_memory_quantum_difference = u32::MAX,
            }
        }
        actual
    }
    fn reset(&mut self) -> Result<(), String> {
        Ok(())
    }
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
/// Export a private verified snapshot and run bounded development fidelity gates.
/// This creates a new immutable directory only after the fidelity checks pass.
pub fn export_policy(
    artifact_dir: &Path,
    artifact_sha256: &str,
    output: &Path,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    if output.exists() {
        return Err("output already exists; exports are immutable".into());
    }
    // Verify and load a private snapshot of exactly the bytes read. The recorder
    // cannot reopen a mutable producer path after hash verification.
    let metadata =
        crate::artifact_io::read_bounded(&artifact_dir.join("behavior-cloning.json"), 1024 * 1024)?;
    if hash(&metadata) != artifact_sha256 {
        return Err("source metadata SHA-256 mismatch".into());
    }
    let artifact: BehaviorCloningArtifact = serde_json::from_slice(&metadata)?;
    blob_policy::runtime::layer_shapes(
        artifact.model.hidden1,
        artifact.model.hidden2,
        artifact.model.recurrent_size,
    )?;
    let model_bytes =
        crate::artifact_io::read_bounded(&artifact_dir.join("model.mpk"), 64 * 1024 * 1024)?;
    let snapshot = tempfile::tempdir()?;
    fs::write(snapshot.path().join("behavior-cloning.json"), &metadata)?;
    fs::write(snapshot.path().join("model.mpk"), &model_bytes)?;
    let device = Default::default();
    NdArray::<f32>::seed(&device, 721);
    let model = load_behavior_clone::<NdArray<f32>>(
        snapshot.path(),
        artifact_sha256,
        &artifact.model,
        &device,
    )?;
    blob_policy::qualification::check_confirmation_reservations(
        artifact.config.confirmation_seeds.iter().copied().chain(
            artifact
                .seed_ledger
                .iter()
                .flat_map(|ledger| ledger.confirmation.iter().copied()),
        ),
    )?;
    let frozen = model.export_frozen()?;
    let report = Arc::new(Mutex::new(Parity::default()));
    // Fixed deployment-development fixtures, never a gameplay promotion suite.
    let seeds = blob_policy::qualification::DEVELOPMENT_SEEDS;
    for seed in seeds {
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
        for team in 0..2 {
            engine.add_team_with_minds(
                TeamId(team),
                vec![Compare {
                    burn: SnapshotPolicyMind::new(model.clone(), device),
                    frozen: frozen.clone(),
                    report: report.clone(),
                }],
            )?;
        }
        engine.initialize_reference_state()?;
        for _ in 0..128 {
            engine.tick(false)?;
        }
    }
    let report = report.lock().unwrap();
    if report.decisions < 100
        || report.action_or_signal_mismatches != 0
        || report.max_memory_quantum_difference > 1
    {
        eprintln!("{}", serde_json::to_string_pretty(&*report)?);
        return Err("checkpoint failed portable deployment fidelity gate".into());
    }
    let weights = frozen.to_bytes();
    let manifest = serde_json::json!({
        "schema_version":POLICY_EXPORT_SCHEMA_VERSION,"weight_format":POLICY_WEIGHT_FORMAT_VERSION,"execution_contract":EXECUTION_CONTRACT,
        "source_artifact_sha256":artifact_sha256,"source_model_sha256":hash(&model_bytes),
        "source_schema":artifact.schema_version,"source_seed_ledger":artifact.seed_ledger,
        "source_training_config":artifact.config,"model":artifact.model,
        "export_build":env!("BLOB_CODE_REVISION"),"reference_mind_abi":blob_interface::abi::reference_mind_abi_hash(),
        "weights_sha256":hash(&weights),"weight_bytes":weights.len(),
        "development_seeds":seeds,"fidelity":&*report,
        "fidelity_gate":{"minimum_decisions":100,"action_or_signal_mismatches":0,"max_memory_quantum_difference":1},
        "qualification":"deployment development evidence; no gameplay promotion; WASM parity pending"
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    if manifest_bytes.len() > 1024 * 1024 {
        return Err("export manifest exceeds byte budget".into());
    }
    fs::create_dir(output)?;
    fs::write(output.join("weights.bin"), weights)?;
    fs::write(output.join("export.json"), manifest_bytes)?;
    Ok(serde_json::to_value(&*report)?)
}
