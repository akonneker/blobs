//! Deterministic, hash-bound behavior-cloning demonstrations.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use blob_interface::reference_mind::ReferenceMemoryUpdate;
use serde::{Deserialize, Serialize};

use crate::action::{
    decode_policy_choice, encode_decision, policy_action_family, PolicyActionFamily, NUM_ACTIONS,
    NUM_AMOUNT_CHOICES, NUM_SIGNAL_CHOICES, NUM_SIGNAL_STRENGTH_CHOICES,
};
use crate::config::{ScenarioProfile, TrainingConfig};
use crate::control_matrix::MaintainedMindProfile;
use crate::env::{BlobEnv, EpisodeOutcome};
use crate::observation::{Observation, OBS_DIM};
use crate::sweep::sha256;
use crate::viability::mind_abi_hash;

pub const DEMONSTRATION_SCHEMA_VERSION: u32 = 9;
const PAYLOAD_FILE: &str = "samples.mpk";
const MANIFEST_FILE: &str = "manifest.json";
const MAX_DATASET_BYTES: u64 = 4 * 1024 * 1024 * 1024;
static DATASET_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub struct DemonstrationOptions {
    pub teacher: MaintainedMindProfile,
    pub seeds: Vec<u64>,
    pub max_samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DemonstrationSample {
    /// Simulation seed that produced this decision. Validation partitions keep
    /// every decision from a source seed together to avoid trajectory leakage.
    pub source_seed: u64,
    /// Host-only trajectory key within `source_seed`. It is never part of the
    /// neural observation or Mind ABI; sequence training uses it to keep
    /// recurrent histories cell-private.
    pub source_cell: u64,
    pub observation: Vec<f32>,
    pub action_mask: Vec<bool>,
    pub action: u16,
    pub amount_mask: Vec<bool>,
    pub amount: u8,
    pub signal_mask: Vec<bool>,
    pub signal: u8,
    pub signal_strength_mask: Vec<bool>,
    pub signal_strength: u8,
    /// True only when the current policy decoder reproduces the teacher's
    /// complete action, signal, and memory update exactly.
    pub exact_round_trip: bool,
    pub memory_replacement: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DemonstrationPayload {
    pub schema_version: u32,
    pub samples: Vec<DemonstrationSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DemonstrationManifest {
    pub schema_version: u32,
    pub package_version: String,
    pub mind_abi_sha256: String,
    pub teacher: MaintainedMindProfile,
    pub seeds: Vec<u64>,
    pub source_config_sha256: String,
    pub semantic_ruleset_hash: String,
    pub compiled_ruleset_hash: String,
    pub scenario_hash: String,
    pub observation_dim: usize,
    pub action_count: usize,
    pub amount_choice_count: usize,
    pub signal_choice_count: usize,
    pub signal_strength_choice_count: usize,
    pub samples: usize,
    pub exact_round_trip_samples: usize,
    pub memory_replacement_samples: usize,
    /// Wait, guard, consume, move, attack, split, regurgitate, terrain, signal.
    pub action_family_samples: [usize; PolicyActionFamily::COUNT],
    pub completed_episodes: usize,
    pub wins: usize,
    pub losses: usize,
    pub timeouts: usize,
    pub payload_file: String,
    pub payload_sha256: String,
}

#[derive(Debug, Clone)]
pub struct LoadedDemonstrations {
    pub directory: PathBuf,
    pub manifest_sha256: String,
    pub manifest: DemonstrationManifest,
    pub payload: DemonstrationPayload,
}

fn validate_options(options: &DemonstrationOptions) -> Result<(), String> {
    if options.seeds.is_empty() || options.max_samples == 0 {
        return Err("demonstrations require seeds and a positive sample cap".into());
    }
    let mut unique = options.seeds.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != options.seeds.len() {
        return Err("demonstration seeds must be unique".into());
    }
    Ok(())
}

pub fn generate_demonstrations(
    config: &TrainingConfig,
    source_config_sha256: String,
    options: &DemonstrationOptions,
) -> Result<(DemonstrationManifest, DemonstrationPayload), String> {
    config.validate()?;
    validate_options(options)?;
    let scenario_hash = ScenarioProfile::from(&config.env).semantic_hash()?;
    let semantic_ruleset_hash = config.env.rules.semantic_hash().to_string();
    let mut samples = Vec::with_capacity(options.max_samples);
    let mut exact_round_trip_samples = 0usize;
    let mut memory_replacement_samples = 0usize;
    let mut action_family_samples = [0usize; PolicyActionFamily::COUNT];
    let mut completed_episodes = 0usize;
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut timeouts = 0usize;
    let mut compiled_ruleset_hash = None;

    let samples_per_seed = options.max_samples.div_ceil(options.seeds.len());
    'seeds: for seed in &options.seeds {
        let mut env = BlobEnv::new(config.env.clone(), config.reward.clone(), *seed);
        let mut seed_samples = 0usize;
        let compiled = env.compiled_ruleset_hash();
        if compiled_ruleset_hash
            .as_ref()
            .is_some_and(|expected| expected != &compiled)
        {
            return Err("identical demonstration configuration compiled differently".into());
        }
        compiled_ruleset_hash = Some(compiled);
        loop {
            let prepared = env.prepare_training_reference_inputs()?;
            let mut decisions = std::collections::HashMap::with_capacity(prepared.len());
            for (cell_id, input) in prepared {
                let decision = options.teacher.decide(&input);
                let choice = encode_decision(&decision, &input).ok_or_else(|| {
                    format!(
                        "teacher {} emitted an unencodable decision",
                        options.teacher
                    )
                })?;
                let observation = Observation::from_reference(&input);
                let amount_mask = observation.amount_mask(choice.action);
                let signal_mask = observation.signal_mask(choice.action, choice.amount);
                let signal_strength_mask =
                    observation.signal_strength_mask(choice.action, choice.amount, choice.signal);
                if choice.action >= NUM_ACTIONS
                    || !observation.action_mask[choice.action]
                    || choice.amount >= NUM_AMOUNT_CHOICES
                    || !amount_mask[choice.amount]
                    || choice.signal >= NUM_SIGNAL_CHOICES
                    || !signal_mask[choice.signal]
                    || choice.signal_strength >= NUM_SIGNAL_STRENGTH_CHOICES
                    || !signal_strength_mask[choice.signal_strength]
                {
                    return Err(format!(
                        "teacher {} emitted an action outside the policy mask",
                        options.teacher
                    ));
                }
                if samples.len() < options.max_samples && seed_samples < samples_per_seed {
                    let decoded = decode_policy_choice(choice, &input);
                    let exact_round_trip = decoded == decision;
                    exact_round_trip_samples += usize::from(exact_round_trip);
                    let memory_replacement =
                        matches!(decision.memory_update, ReferenceMemoryUpdate::Replace(_));
                    memory_replacement_samples += usize::from(memory_replacement);
                    let family = policy_action_family(choice.action)
                        .expect("encoded teacher decision belongs to a policy family");
                    action_family_samples[family.index()] += 1;
                    samples.push(DemonstrationSample {
                        source_seed: *seed,
                        source_cell: u64::try_from(cell_id.0).map_err(|_| {
                            "cell ID exceeds demonstration trajectory key".to_string()
                        })?,
                        observation: observation.data.to_vec(),
                        action_mask: observation.action_mask.to_vec(),
                        action: u16::try_from(choice.action)
                            .map_err(|_| "policy action catalog exceeds u16".to_string())?,
                        amount_mask: amount_mask.to_vec(),
                        amount: u8::try_from(choice.amount)
                            .map_err(|_| "policy amount catalog exceeds u8".to_string())?,
                        signal_mask: signal_mask.to_vec(),
                        signal: u8::try_from(choice.signal)
                            .map_err(|_| "policy signal catalog exceeds u8".to_string())?,
                        signal_strength_mask: signal_strength_mask.to_vec(),
                        signal_strength: u8::try_from(choice.signal_strength)
                            .map_err(|_| "policy signal-strength catalog exceeds u8".to_string())?,
                        exact_round_trip,
                        memory_replacement,
                    });
                    seed_samples += 1;
                }
                decisions.insert(cell_id, decision);
            }
            let result = env.step_with_training_decisions(decisions);
            if result.done {
                completed_episodes += 1;
                match result.outcome {
                    Some(EpisodeOutcome::Win) => wins += 1,
                    Some(EpisodeOutcome::Loss) => losses += 1,
                    Some(EpisodeOutcome::Timeout) => timeouts += 1,
                    Some(EpisodeOutcome::SafetyAbort) => {
                        return Err(format!(
                            "demonstration seed {seed} reached the decision-frontier safety limit"
                        ));
                    }
                    None => return Err("completed demonstration omitted its outcome".into()),
                }
                break;
            }
            if samples.len() >= options.max_samples {
                break 'seeds;
            }
            if seed_samples >= samples_per_seed {
                break;
            }
        }
        if samples.len() >= options.max_samples {
            break;
        }
    }
    if samples.is_empty() {
        return Err("demonstration suite produced no ready-cell decisions".into());
    }

    let payload = DemonstrationPayload {
        schema_version: DEMONSTRATION_SCHEMA_VERSION,
        samples,
    };
    let payload_bytes = rmp_serde::to_vec_named(&payload)
        .map_err(|error| format!("failed to encode demonstration payload: {error}"))?;
    let manifest = DemonstrationManifest {
        schema_version: DEMONSTRATION_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        mind_abi_sha256: mind_abi_hash(),
        teacher: options.teacher,
        seeds: options.seeds.clone(),
        source_config_sha256,
        semantic_ruleset_hash,
        compiled_ruleset_hash: compiled_ruleset_hash
            .expect("a nonempty demonstration has a compiled ruleset"),
        scenario_hash,
        observation_dim: OBS_DIM,
        action_count: NUM_ACTIONS,
        amount_choice_count: NUM_AMOUNT_CHOICES,
        signal_choice_count: NUM_SIGNAL_CHOICES,
        signal_strength_choice_count: NUM_SIGNAL_STRENGTH_CHOICES,
        samples: payload.samples.len(),
        exact_round_trip_samples,
        memory_replacement_samples,
        action_family_samples,
        completed_episodes,
        wins,
        losses,
        timeouts,
        payload_file: PAYLOAD_FILE.into(),
        payload_sha256: sha256(&payload_bytes),
    };
    Ok((manifest, payload))
}

fn validate_dataset(
    manifest: &DemonstrationManifest,
    payload: &DemonstrationPayload,
) -> Result<(), String> {
    if manifest.schema_version != DEMONSTRATION_SCHEMA_VERSION
        || payload.schema_version != DEMONSTRATION_SCHEMA_VERSION
        || manifest.observation_dim != OBS_DIM
        || manifest.action_count != NUM_ACTIONS
        || manifest.amount_choice_count != NUM_AMOUNT_CHOICES
        || manifest.signal_choice_count != NUM_SIGNAL_CHOICES
        || manifest.signal_strength_choice_count != NUM_SIGNAL_STRENGTH_CHOICES
        || manifest.mind_abi_sha256 != mind_abi_hash()
        || manifest.samples != payload.samples.len()
        || manifest.payload_file != PAYLOAD_FILE
    {
        return Err("demonstration schema or identity mismatch".into());
    }
    let mut exact = 0usize;
    let mut memory_replacements = 0usize;
    let mut action_families = [0usize; PolicyActionFamily::COUNT];
    for sample in &payload.samples {
        let action = usize::from(sample.action);
        let amount = usize::from(sample.amount);
        let signal = usize::from(sample.signal);
        let signal_strength = usize::from(sample.signal_strength);
        if !manifest.seeds.contains(&sample.source_seed)
            || sample.observation.len() != OBS_DIM
            || sample.action_mask.len() != NUM_ACTIONS
            || sample.amount_mask.len() != NUM_AMOUNT_CHOICES
            || sample.signal_mask.len() != NUM_SIGNAL_CHOICES
            || sample.signal_strength_mask.len() != NUM_SIGNAL_STRENGTH_CHOICES
            || sample.observation.iter().any(|value| !value.is_finite())
            || action >= NUM_ACTIONS
            || !sample.action_mask[action]
            || amount >= NUM_AMOUNT_CHOICES
            || !sample.amount_mask[amount]
            || signal >= NUM_SIGNAL_CHOICES
            || !sample.signal_mask[signal]
            || signal_strength >= NUM_SIGNAL_STRENGTH_CHOICES
            || !sample.signal_strength_mask[signal_strength]
        {
            return Err("demonstration sample is malformed or mask-inconsistent".into());
        }
        exact += usize::from(sample.exact_round_trip);
        memory_replacements += usize::from(sample.memory_replacement);
        let family = policy_action_family(action)
            .ok_or_else(|| "demonstration action has no policy family".to_string())?;
        action_families[family.index()] += 1;
    }
    if exact != manifest.exact_round_trip_samples
        || memory_replacements != manifest.memory_replacement_samples
        || action_families != manifest.action_family_samples
    {
        return Err("demonstration representability count mismatch".into());
    }
    Ok(())
}

pub fn publish_demonstrations(
    output: &Path,
    manifest: &DemonstrationManifest,
    payload: &DemonstrationPayload,
) -> Result<PathBuf, String> {
    if output.exists() {
        return Err(format!(
            "refusing to replace demonstration dataset {}",
            output.display()
        ));
    }
    validate_dataset(manifest, payload)?;
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "demonstration output needs a UTF-8 name".to_string())?;
    let nonce = DATASET_NONCE.fetch_add(1, Ordering::Relaxed);
    let staging = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    fs::create_dir(&staging)
        .map_err(|error| format!("failed to create {}: {error}", staging.display()))?;
    let result = (|| {
        let payload_bytes = rmp_serde::to_vec_named(payload)
            .map_err(|error| format!("failed to encode demonstrations: {error}"))?;
        if sha256(&payload_bytes) != manifest.payload_sha256 {
            return Err("demonstration payload changed after manifest creation".into());
        }
        let mut manifest_bytes = serde_json::to_vec_pretty(manifest)
            .map_err(|error| format!("failed to encode demonstration manifest: {error}"))?;
        manifest_bytes.push(b'\n');
        for (file_name, bytes) in [
            (PAYLOAD_FILE, payload_bytes.as_slice()),
            (MANIFEST_FILE, manifest_bytes.as_slice()),
        ] {
            let path = staging.join(file_name);
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
            file.write_all(bytes)
                .and_then(|()| file.sync_all())
                .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
        }
        File::open(&staging)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("failed to sync {}: {error}", staging.display()))?;
        fs::rename(&staging, output)
            .map_err(|error| format!("failed to publish {}: {error}", output.display()))?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("failed to sync {}: {error}", parent.display()))
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result.map(|()| output.to_path_buf())
}

pub fn load_demonstrations(directory: &Path) -> Result<LoadedDemonstrations, String> {
    let manifest_path = directory.join(MANIFEST_FILE);
    let manifest_length = fs::metadata(&manifest_path)
        .map_err(|error| format!("failed to inspect {}: {error}", manifest_path.display()))?
        .len();
    if manifest_length > 1024 * 1024 {
        return Err("demonstration manifest exceeds 1 MiB".into());
    }
    let manifest_bytes = fs::read(&manifest_path)
        .map_err(|error| format!("failed to read {}: {error}", manifest_path.display()))?;
    let manifest: DemonstrationManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("failed to decode {}: {error}", manifest_path.display()))?;
    if manifest.schema_version != DEMONSTRATION_SCHEMA_VERSION
        || manifest.payload_file != PAYLOAD_FILE
    {
        return Err("demonstration manifest schema or payload path mismatch".into());
    }
    let payload_path = directory.join(&manifest.payload_file);
    let length = fs::metadata(&payload_path)
        .map_err(|error| format!("failed to inspect {}: {error}", payload_path.display()))?
        .len();
    if length > MAX_DATASET_BYTES {
        return Err(format!(
            "demonstration payload exceeds {MAX_DATASET_BYTES} bytes"
        ));
    }
    let payload_bytes = fs::read(&payload_path)
        .map_err(|error| format!("failed to read {}: {error}", payload_path.display()))?;
    if sha256(&payload_bytes) != manifest.payload_sha256 {
        return Err("demonstration payload SHA-256 mismatch".into());
    }
    let payload: DemonstrationPayload = rmp_serde::from_slice(&payload_bytes)
        .map_err(|error| format!("failed to decode {}: {error}", payload_path.display()))?;
    validate_dataset(&manifest, &payload)?;
    Ok(LoadedDemonstrations {
        directory: directory.to_path_buf(),
        manifest_sha256: sha256(&manifest_bytes),
        manifest,
        payload,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> TrainingConfig {
        let mut config = TrainingConfig::default();
        config.env.world_size = 4;
        config.env.cells_per_team = 1;
        config.env.max_episode_len = 512;
        config.env.victory.sim_time_limit_quanta = 16_384;
        config.env.num_scattered_energy = 2;
        config.env.num_plants = 1;
        config
    }

    #[test]
    fn datasets_are_reproducible_hash_bound_and_report_colony_lossiness() {
        let options = DemonstrationOptions {
            teacher: MaintainedMindProfile::Colony,
            seeds: vec![11, 12],
            max_samples: 16,
        };
        let left =
            generate_demonstrations(&small_config(), "config-hash".into(), &options).unwrap();
        let right =
            generate_demonstrations(&small_config(), "config-hash".into(), &options).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.0.samples, 16);
        assert!(options.seeds.iter().all(|seed| {
            left.1
                .samples
                .iter()
                .any(|sample| sample.source_seed == *seed)
        }));
        assert!(left.0.memory_replacement_samples > 0);
        assert!(left.0.exact_round_trip_samples < left.0.samples);
        assert_eq!(
            left.0.action_family_samples.iter().sum::<usize>(),
            left.0.samples
        );

        let mut wrong_seed = left.1.clone();
        wrong_seed.samples[0].source_seed = 999;
        assert!(validate_dataset(&left.0, &wrong_seed).is_err());
        let mut wrong_families = left.0.clone();
        wrong_families.action_family_samples[PolicyActionFamily::Attack.index()] += 1;
        assert!(validate_dataset(&wrong_families, &left.1).is_err());

        let temporary = tempfile::tempdir().unwrap();
        let directory = temporary.path().join("dataset");
        publish_demonstrations(&directory, &left.0, &left.1).unwrap();
        let loaded = load_demonstrations(&directory).unwrap();
        assert_eq!(loaded.manifest, left.0);
        assert_eq!(loaded.payload, left.1);
        assert!(publish_demonstrations(&directory, &left.0, &left.1).is_err());

        let payload_path = directory.join(PAYLOAD_FILE);
        let mut bytes = fs::read(&payload_path).unwrap();
        bytes[0] ^= 1;
        fs::write(payload_path, bytes).unwrap();
        assert!(load_demonstrations(&directory)
            .unwrap_err()
            .contains("SHA-256"));
    }

    #[test]
    fn paired_contact_aggressive_teacher_produces_attack_dense_examples() {
        let mut config = small_config();
        config.env.world_size = 8;
        config.env.cells_per_team = 1;
        config.env.starting_cell_layout = blob_engine::engine::StartingCellLayout::PairedContact;
        config.env.initial_energy = 180;
        config.env.num_scattered_energy = 0;
        config.env.num_plants = 0;
        config.env.opponent = crate::config::OpponentProfile::Defensive;
        let options = DemonstrationOptions {
            teacher: MaintainedMindProfile::Aggressive,
            seeds: vec![21, 22, 23, 24],
            max_samples: 64,
        };

        let (manifest, payload) =
            generate_demonstrations(&config, "contact-config".into(), &options).unwrap();
        let attacks = manifest.action_family_samples[PolicyActionFamily::Attack.index()];
        let exact_attacks = payload
            .samples
            .iter()
            .filter(|sample| {
                sample.exact_round_trip
                    && policy_action_family(usize::from(sample.action))
                        == Some(PolicyActionFamily::Attack)
            })
            .count();
        assert!(attacks > 0);
        assert!(attacks * 2 >= payload.samples.len());
        assert!(exact_attacks > 0);
        assert!(exact_attacks * 2 >= manifest.exact_round_trip_samples);
    }
}
