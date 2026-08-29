//! Deterministic, immutable planning for paired ruleset sweeps.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use blob_engine::resolution::ReferenceSimulation;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{ScenarioProfile, TrainingConfig};

pub const RULES_SWEEP_SCHEMA_VERSION: u32 = 7;
static SWEEP_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RulesSweepSpec {
    pub base_config: String,
    pub output_dir: String,
    pub seeds: Vec<u64>,
    pub variants: Vec<RulesSweepVariant>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RulesSweepVariant {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// Strict partial override of `ReferenceRuleset`. Reward and optimizer
    /// settings cannot vary through this document.
    #[serde(default)]
    pub rules: toml::Table,
    /// Strict partial override of initial world/population parameters. This is
    /// deliberately separate from resolver rules and cannot alter rewards,
    /// PPO settings, or opponent selection.
    #[serde(default)]
    pub scenario: toml::Table,
    /// Strict partial override of the local-combat curriculum. This is kept
    /// separate from rules and scenario controls so cadence/mixture sweeps
    /// cannot change rewards, PPO, model capacity, or evaluation identity.
    #[serde(default)]
    pub combat_curriculum: toml::Table,
    /// Strict partial override of host-only frontier-teacher distillation.
    /// This enables paired retention experiments without exposing PPO or
    /// physics as incidental sweep variables.
    #[serde(default)]
    pub specialist_distillation: toml::Table,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RulesSweepRun {
    pub variant: String,
    pub replicate: usize,
    pub training_seed: u64,
    pub semantic_ruleset_hash: String,
    pub compiled_ruleset_hash: String,
    pub scenario_hash: String,
    /// Hash of the complete execution configuration with the artifact output
    /// path normalized away. The seed and host safety limits remain included.
    pub experiment_config_sha256: String,
    pub config_file_sha256: String,
    pub run_directory: String,
    pub config_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RulesSweepManifest {
    pub schema_version: u32,
    pub package_version: String,
    pub spec_file: String,
    pub spec_sha256: String,
    pub base_config_file: String,
    pub base_config_sha256: String,
    pub output_directory: String,
    pub seeds: Vec<u64>,
    pub variants: Vec<RulesSweepVariant>,
    pub runs: Vec<RulesSweepRun>,
}

pub(crate) fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn absolute_from(base: &Path, path: &str) -> Result<PathBuf, String> {
    if path.trim().is_empty() {
        return Err("sweep paths must not be empty".into());
    }
    let path = Path::new(path);
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(base.join(path))
    }
}

fn valid_variant_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name.bytes().enumerate().all(|(index, byte)| match byte {
            b'a'..=b'z' | b'0'..=b'9' => true,
            b'-' | b'_' => index > 0,
            _ => false,
        })
}

fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("failed to sync directory {}: {error}", path.display()))
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = File::create(path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

pub(crate) fn experiment_config_hash(config: &TrainingConfig) -> Result<String, String> {
    let mut normalized = config.clone();
    normalized.checkpoint_dir.clear();
    serde_json::to_vec(&normalized)
        .map(|bytes| sha256(&bytes))
        .map_err(|error| format!("failed to encode experiment configuration: {error}"))
}

/// Expand and atomically publish an immutable sweep plan. Every variant uses
/// the same ordered seed list, enabling paired comparisons without treating
/// replicate seeds as independent rule changes.
pub fn publish_rules_sweep(spec_file: &Path) -> Result<RulesSweepManifest, String> {
    let spec_file = fs::canonicalize(spec_file).map_err(|error| {
        format!(
            "failed to resolve sweep spec {}: {error}",
            spec_file.display()
        )
    })?;
    let spec_bytes = fs::read(&spec_file)
        .map_err(|error| format!("failed to read sweep spec {}: {error}", spec_file.display()))?;
    let spec: RulesSweepSpec = toml::from_str(
        std::str::from_utf8(&spec_bytes)
            .map_err(|error| format!("sweep spec is not UTF-8: {error}"))?,
    )
    .map_err(|error| format!("failed to parse sweep spec: {error}"))?;
    validate_spec(&spec)?;

    let spec_directory = spec_file.parent().unwrap_or_else(|| Path::new("."));
    let base_config_file = fs::canonicalize(absolute_from(spec_directory, &spec.base_config)?)
        .map_err(|error| format!("failed to resolve base config: {error}"))?;
    let requested_output = absolute_from(spec_directory, &spec.output_dir)?;
    let output_name = requested_output
        .file_name()
        .ok_or_else(|| "sweep output directory needs a name".to_string())?;
    let requested_parent = requested_output
        .parent()
        .ok_or_else(|| "sweep output directory has no parent".to_string())?;
    fs::create_dir_all(requested_parent).map_err(|error| {
        format!(
            "failed to create sweep output parent {}: {error}",
            requested_parent.display()
        )
    })?;
    let output_parent = fs::canonicalize(requested_parent).map_err(|error| {
        format!(
            "failed to resolve sweep output parent {}: {error}",
            requested_parent.display()
        )
    })?;
    let output_directory = output_parent.join(output_name);
    if output_directory.exists() {
        return Err(format!(
            "refusing to replace immutable sweep directory {}",
            output_directory.display()
        ));
    }
    let base_bytes = fs::read(&base_config_file).map_err(|error| {
        format!(
            "failed to read base config {}: {error}",
            base_config_file.display()
        )
    })?;
    let base_config = TrainingConfig::from_toml_str(
        std::str::from_utf8(&base_bytes)
            .map_err(|error| format!("base config is not UTF-8: {error}"))?,
    )?;
    base_config.validate()?;

    let nonce = SWEEP_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let output_name = output_directory
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "sweep output directory needs a UTF-8 name".to_string())?;
    let staging = output_parent.join(format!(".{output_name}.tmp-{}-{nonce}", std::process::id()));
    fs::create_dir(&staging).map_err(|error| {
        format!(
            "failed to create sweep staging directory {}: {error}",
            staging.display()
        )
    })?;

    let result = stage_sweep(
        &spec,
        &spec_file,
        &spec_bytes,
        &base_config_file,
        &base_bytes,
        &base_config,
        &output_directory,
        &staging,
    );
    let manifest = match result {
        Ok(manifest) => manifest,
        Err(error) => {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
    };
    if let Err(error) = sync_directory(&staging) {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    if let Err(error) = fs::rename(&staging, &output_directory) {
        let _ = fs::remove_dir_all(&staging);
        return Err(format!(
            "failed to publish sweep directory {}: {error}",
            output_directory.display()
        ));
    }
    sync_directory(&output_parent)?;
    Ok(manifest)
}

#[allow(clippy::too_many_arguments)]
fn stage_sweep(
    spec: &RulesSweepSpec,
    spec_file: &Path,
    spec_bytes: &[u8],
    base_config_file: &Path,
    base_bytes: &[u8],
    base_config: &TrainingConfig,
    output_directory: &Path,
    staging: &Path,
) -> Result<RulesSweepManifest, String> {
    let mut runs = Vec::with_capacity(spec.variants.len() * spec.seeds.len());
    let mut experiment_profiles = HashSet::new();
    for variant in &spec.variants {
        let variant_config = base_config
            .with_rules_override(&variant.rules)?
            .with_scenario_override(&variant.scenario)?
            .with_combat_curriculum_override(&variant.combat_curriculum)?
            .with_specialist_distillation_override(&variant.specialist_distillation)?;
        let semantic_hash = variant_config.env.rules.semantic_hash().to_string();
        let scenario_hash = ScenarioProfile::from(&variant_config.env).semantic_hash()?;
        let compiled_hash = ReferenceSimulation::new(
            variant_config.env.world_size,
            variant_config.env.world_size,
            variant_config.env.rules.clone(),
        )
        .map_err(|error| format!("variant {} is invalid: {error}", variant.name))?
        .compiled_ruleset_hash()
        .to_string();
        if !experiment_profiles.insert(experiment_config_hash(&variant_config)?) {
            return Err(format!(
                "variant {} duplicates another variant's complete expanded experiment",
                variant.name
            ));
        }

        for (replicate, &seed) in spec.seeds.iter().enumerate() {
            let relative_run = PathBuf::from(&variant.name).join(format!("seed-{seed}"));
            let staged_run = staging.join(&relative_run);
            let final_run = output_directory.join(&relative_run);
            fs::create_dir_all(&staged_run).map_err(|error| {
                format!(
                    "failed to create staged run {}: {error}",
                    staged_run.display()
                )
            })?;
            let mut config = variant_config.clone();
            config.seed = seed;
            config.checkpoint_dir = final_run.join("artifacts").to_string_lossy().into_owned();
            config.validate()?;
            let config_bytes = toml::to_string_pretty(&config)
                .map_err(|error| format!("failed to encode expanded run config: {error}"))?
                .into_bytes();
            let staged_config = staged_run.join("config.toml");
            write_synced(&staged_config, &config_bytes)?;
            runs.push(RulesSweepRun {
                variant: variant.name.clone(),
                replicate,
                training_seed: seed,
                semantic_ruleset_hash: semantic_hash.clone(),
                compiled_ruleset_hash: compiled_hash.clone(),
                scenario_hash: scenario_hash.clone(),
                experiment_config_sha256: experiment_config_hash(&config)?,
                config_file_sha256: sha256(&config_bytes),
                run_directory: final_run.to_string_lossy().into_owned(),
                config_file: final_run.join("config.toml").to_string_lossy().into_owned(),
            });
            sync_directory(&staged_run)?;
        }
        sync_directory(&staging.join(&variant.name))?;
    }

    let manifest = RulesSweepManifest {
        schema_version: RULES_SWEEP_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        spec_file: spec_file.to_string_lossy().into_owned(),
        spec_sha256: sha256(spec_bytes),
        base_config_file: base_config_file.to_string_lossy().into_owned(),
        base_config_sha256: sha256(base_bytes),
        output_directory: output_directory.to_string_lossy().into_owned(),
        seeds: spec.seeds.clone(),
        variants: spec.variants.clone(),
        runs,
    };
    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("failed to encode sweep manifest: {error}"))?;
    manifest_bytes.push(b'\n');
    write_synced(&staging.join("manifest.json"), &manifest_bytes)?;
    Ok(manifest)
}

fn validate_spec(spec: &RulesSweepSpec) -> Result<(), String> {
    if spec.seeds.len() < 3 {
        return Err("rules sweeps require at least three replicate seeds".into());
    }
    if spec.variants.is_empty() {
        return Err("rules sweeps require at least one variant".into());
    }
    let mut seeds = HashSet::new();
    if spec.seeds.iter().any(|seed| !seeds.insert(*seed)) {
        return Err("rules sweep seeds must be unique".into());
    }
    let mut names = HashSet::new();
    for variant in &spec.variants {
        if !valid_variant_name(&variant.name) {
            return Err(format!(
                "invalid rules variant name {}; use lowercase letters, digits, '-' or '_'",
                variant.name
            ));
        }
        if !names.insert(&variant.name) {
            return Err(format!("duplicate rules variant name {}", variant.name));
        }
        if variant
            .description
            .as_ref()
            .is_some_and(|description| description.trim().is_empty())
        {
            return Err(format!("variant {} has an empty description", variant.name));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_base(path: &Path) {
        fs::write(
            path,
            r#"
            seed = 1
            num_envs = 1
            rollout_length = 4
            total_timesteps = 8
            eval_interval = 0
            eval_episodes = 1
            checkpoint_interval = 0
            evaluation_opponents = ["wait"]

            [model]
            hidden1 = 8
            hidden2 = 8

            [env]
            world_size = 8
            cells_per_team = 1
            max_episode_len = 8
            num_scattered_energy = 2
            num_plants = 1

            [self_play]
            max_opponent_pool = 0
            "#,
        )
        .unwrap();
    }

    #[test]
    fn publishes_paired_hash_bound_expanded_runs() {
        let temporary = tempfile::tempdir().unwrap();
        let base = temporary.path().join("base.toml");
        let spec = temporary.path().join("sweep.toml");
        write_base(&base);
        fs::write(
            &spec,
            r#"
            base_config = "base.toml"
            output_dir = "planned"
            seeds = [11, 12, 13]

            [[variants]]
            name = "baseline"

            [[variants]]
            name = "fast-digestion"
            description = "Double digestion throughput"
            [variants.rules]
            digestion_rate_numerator = 2

            [[variants]]
            name = "more-plants"
            description = "Change only the initial ecology"
            [variants.scenario]
            num_plants = 4

            [[variants]]
            name = "short-cycle"
            description = "Change only the disabled curriculum profile"
            [variants.combat_curriculum]
            cycle_sim_time_quanta_per_env = 131072
            "#,
        )
        .unwrap();

        let manifest = publish_rules_sweep(&spec).unwrap();
        assert_eq!(manifest.runs.len(), 12);
        assert!(Path::new(&manifest.output_directory).is_absolute());
        assert!(manifest
            .runs
            .iter()
            .all(|run| Path::new(&run.config_file).is_absolute()));
        assert_eq!(
            manifest
                .runs
                .iter()
                .filter(|run| run.variant == "baseline")
                .map(|run| run.training_seed)
                .collect::<Vec<_>>(),
            vec![11, 12, 13]
        );
        let baseline = &manifest.runs[0];
        let changed = &manifest.runs[3];
        let scenario_changed = &manifest.runs[6];
        let curriculum_changed = &manifest.runs[9];
        assert_ne!(
            baseline.semantic_ruleset_hash,
            changed.semantic_ruleset_hash
        );
        assert_ne!(
            baseline.compiled_ruleset_hash,
            changed.compiled_ruleset_hash
        );
        assert_eq!(baseline.scenario_hash, changed.scenario_hash);
        assert_eq!(
            baseline.semantic_ruleset_hash,
            scenario_changed.semantic_ruleset_hash
        );
        assert_ne!(baseline.scenario_hash, scenario_changed.scenario_hash);
        let baseline_config = TrainingConfig::from_file(&baseline.config_file).unwrap();
        let changed_config = TrainingConfig::from_file(&changed.config_file).unwrap();
        let scenario_config = TrainingConfig::from_file(&scenario_changed.config_file).unwrap();
        let curriculum_config = TrainingConfig::from_file(&curriculum_changed.config_file).unwrap();
        assert_eq!(baseline_config.reward, changed_config.reward);
        assert_eq!(baseline_config.reward, scenario_config.reward);
        assert_eq!(baseline_config.ppo, scenario_config.ppo);
        assert_eq!(baseline_config.env.opponent, scenario_config.env.opponent);
        assert_eq!(changed_config.env.rules.digestion_rate_numerator, 2);
        assert_eq!(scenario_config.env.num_plants, 4);
        assert_eq!(
            curriculum_config
                .combat_curriculum
                .cycle_sim_time_quanta_per_env,
            131_072
        );
        assert_eq!(curriculum_config.env, baseline_config.env);
        assert_eq!(curriculum_config.reward, baseline_config.reward);
        assert_eq!(curriculum_config.ppo, baseline_config.ppo);
        assert!(temporary.path().join("planned/manifest.json").is_file());
        assert!(publish_rules_sweep(&spec)
            .unwrap_err()
            .contains("refusing to replace immutable sweep directory"));
    }

    #[test]
    fn rejects_rule_typos_before_publishing_any_output() {
        let temporary = tempfile::tempdir().unwrap();
        write_base(&temporary.path().join("base.toml"));
        let spec = temporary.path().join("sweep.toml");
        fs::write(
            &spec,
            r#"
            base_config = "base.toml"
            output_dir = "planned"
            seeds = [1, 2, 3]

            [[variants]]
            name = "typo"
            [variants.rules]
            bite_capcity = 9
            "#,
        )
        .unwrap();

        assert!(publish_rules_sweep(&spec)
            .unwrap_err()
            .contains("unknown field `bite_capcity`"));
        assert!(!temporary.path().join("planned").exists());
    }

    #[test]
    fn documented_combat_cadence_sweep_changes_only_schedule_and_is_paired() {
        let spec: RulesSweepSpec = toml::from_str(include_str!(
            "../config/combat_curriculum_cadence_sweep.toml"
        ))
        .unwrap();
        validate_spec(&spec).unwrap();
        assert_eq!(spec.seeds, vec![42, 43, 44]);
        assert_eq!(spec.variants.len(), 3);

        let base = TrainingConfig::from_toml_str(include_str!(
            "../config/competitive_transfer_large_skirmish.toml"
        ))
        .unwrap();
        let expanded = spec
            .variants
            .iter()
            .map(|variant| {
                base.with_rules_override(&variant.rules)
                    .and_then(|config| config.with_scenario_override(&variant.scenario))
                    .and_then(|config| {
                        config.with_combat_curriculum_override(&variant.combat_curriculum)
                    })
                    .map(|config| (variant.name.as_str(), config))
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        for (_, config) in &expanded {
            assert_eq!(config.env, base.env);
            assert_eq!(config.reward, base.reward);
            assert_eq!(config.ppo, base.ppo);
            assert_eq!(config.model, base.model);
            assert_eq!(config.evaluation_seed, base.evaluation_seed);
            assert_eq!(config.total_simulation_quanta_per_env, Some(524_288));
            assert_eq!(config.combat_curriculum.min_skirmish_kills_for_promotion, 1);
            let cycles = 524_288 / config.combat_curriculum.cycle_sim_time_quanta_per_env;
            assert_eq!(
                cycles * config.combat_curriculum.on_food_sim_time_quanta_per_cycle,
                65_536
            );
            assert_eq!(
                cycles
                    * config
                        .combat_curriculum
                        .adjacent_food_sim_time_quanta_per_cycle,
                65_536
            );
        }
        assert_eq!(
            expanded[0]
                .1
                .combat_curriculum
                .contact_sim_time_quanta_per_cycle,
            65_536
        );
        assert_eq!(
            expanded[1]
                .1
                .combat_curriculum
                .contact_sim_time_quanta_per_cycle,
            32_768
        );
        assert_eq!(
            expanded[2]
                .1
                .combat_curriculum
                .skirmish_sim_time_quanta_per_cycle,
            49_152
        );
    }

    #[test]
    fn documented_combat_cadence_holdout_is_strict_and_uses_new_seeds() {
        let spec: RulesSweepSpec = toml::from_str(include_str!(
            "../config/combat_curriculum_cadence_holdout_sweep.toml"
        ))
        .unwrap();
        validate_spec(&spec).unwrap();
        assert_eq!(spec.seeds, vec![45, 46, 47, 48, 49]);
        assert_eq!(spec.variants.len(), 2);

        let base = TrainingConfig::from_toml_str(include_str!(
            "../config/competitive_transfer_large_skirmish.toml"
        ))
        .unwrap();
        let expanded = spec
            .variants
            .iter()
            .map(|variant| {
                base.with_rules_override(&variant.rules)
                    .and_then(|config| config.with_scenario_override(&variant.scenario))
                    .and_then(|config| {
                        config.with_combat_curriculum_override(&variant.combat_curriculum)
                    })
                    .and_then(|config| {
                        config
                            .with_specialist_distillation_override(&variant.specialist_distillation)
                    })
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(expanded[0], base);
        for config in &expanded {
            assert!(!config.specialist_distillation.enabled);
            assert_eq!(config.env, base.env);
            assert_eq!(config.reward, base.reward);
            assert_eq!(config.ppo, base.ppo);
            assert_eq!(config.model, base.model);
            assert_eq!(config.evaluation_seed, base.evaluation_seed);
            assert_eq!(config.total_simulation_quanta_per_env, Some(524_288));
            let cycles = 524_288 / config.combat_curriculum.cycle_sim_time_quanta_per_env;
            assert_eq!(
                cycles * config.combat_curriculum.on_food_sim_time_quanta_per_cycle,
                65_536
            );
            assert_eq!(
                cycles
                    * config
                        .combat_curriculum
                        .adjacent_food_sim_time_quanta_per_cycle,
                65_536
            );
            assert_eq!(
                cycles * config.combat_curriculum.contact_sim_time_quanta_per_cycle,
                131_072
            );
            assert_eq!(
                cycles * config.combat_curriculum.skirmish_sim_time_quanta_per_cycle,
                131_072
            );
        }
        assert_eq!(
            expanded[0].combat_curriculum.cycle_sim_time_quanta_per_env,
            262_144
        );
        assert_eq!(
            expanded[1].combat_curriculum.cycle_sim_time_quanta_per_env,
            131_072
        );
    }

    #[test]
    fn documented_isolated_cadence_sweep_keeps_evaluation_horizons_fixed() {
        let spec: RulesSweepSpec = toml::from_str(include_str!(
            "../config/combat_curriculum_cadence_isolated_sweep.toml"
        ))
        .unwrap();
        validate_spec(&spec).unwrap();
        assert_eq!(spec.seeds, vec![45, 46, 47, 48, 49]);
        assert_eq!(spec.variants.len(), 2);

        let base = TrainingConfig::from_toml_str(include_str!(
            "../config/competitive_transfer_large_skirmish_frequent.toml"
        ))
        .unwrap();
        let expanded = spec
            .variants
            .iter()
            .map(|variant| {
                base.with_rules_override(&variant.rules)
                    .and_then(|config| config.with_scenario_override(&variant.scenario))
                    .and_then(|config| {
                        config.with_combat_curriculum_override(&variant.combat_curriculum)
                    })
                    .and_then(|config| {
                        config
                            .with_specialist_distillation_override(&variant.specialist_distillation)
                    })
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(expanded[0], base);
        for config in &expanded {
            assert_eq!(
                config
                    .combat_curriculum
                    .retention_episode_sim_time_limit_quanta,
                16_384
            );
            assert_eq!(
                config
                    .combat_curriculum
                    .contact_episode_sim_time_limit_quanta,
                16_384
            );
            assert_eq!(
                config
                    .combat_curriculum
                    .skirmish_episode_sim_time_limit_quanta,
                32_768
            );
            assert_eq!(
                config
                    .combat_curriculum
                    .contact_evaluation_sim_time_limit_quanta,
                16_384
            );
            assert_eq!(
                config
                    .combat_curriculum
                    .skirmish_evaluation_sim_time_limit_quanta,
                32_768
            );
            let cycles = 524_288 / config.combat_curriculum.cycle_sim_time_quanta_per_env;
            assert_eq!(
                cycles * config.combat_curriculum.on_food_sim_time_quanta_per_cycle,
                65_536
            );
            assert_eq!(
                cycles
                    * config
                        .combat_curriculum
                        .adjacent_food_sim_time_quanta_per_cycle,
                65_536
            );
            assert_eq!(
                cycles * config.combat_curriculum.contact_sim_time_quanta_per_cycle,
                131_072
            );
            assert_eq!(
                cycles * config.combat_curriculum.skirmish_sim_time_quanta_per_cycle,
                131_072
            );
        }
        assert_eq!(
            expanded[0].combat_curriculum.cycle_sim_time_quanta_per_env,
            131_072
        );
        assert_eq!(
            expanded[1].combat_curriculum.cycle_sim_time_quanta_per_env,
            262_144
        );
    }

    #[test]
    fn documented_specialist_sweep_changes_only_distillation() {
        let spec: RulesSweepSpec =
            toml::from_str(include_str!("../config/specialist_distillation_sweep.toml")).unwrap();
        validate_spec(&spec).unwrap();
        assert_eq!(spec.seeds, vec![42, 43, 44]);
        assert_eq!(spec.variants.len(), 2);

        let base = TrainingConfig::from_toml_str(include_str!(
            "../config/competitive_transfer_large_skirmish_frequent.toml"
        ))
        .unwrap();
        let expanded = spec
            .variants
            .iter()
            .map(|variant| {
                base.with_rules_override(&variant.rules)
                    .and_then(|config| config.with_scenario_override(&variant.scenario))
                    .and_then(|config| {
                        config.with_combat_curriculum_override(&variant.combat_curriculum)
                    })
                    .and_then(|config| {
                        config
                            .with_specialist_distillation_override(&variant.specialist_distillation)
                    })
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(!expanded[0].specialist_distillation.enabled);
        assert!(expanded[1].specialist_distillation.enabled);
        assert_eq!(expanded[1].specialist_distillation.ecology_coeff, 0.1);
        assert_eq!(expanded[1].specialist_distillation.combat_coeff, 0.1);

        let mut normalized = expanded[1].clone();
        normalized.specialist_distillation = expanded[0].specialist_distillation.clone();
        assert_eq!(normalized, expanded[0]);
    }

    #[test]
    fn documented_partitioned_teacher_sweep_reuses_the_exact_scientific_profile() {
        let spec: RulesSweepSpec = toml::from_str(include_str!(
            "../config/specialist_distillation_partitioned_inference_sweep.toml"
        ))
        .unwrap();
        validate_spec(&spec).unwrap();
        assert_eq!(spec.seeds, vec![42, 43, 44]);
        assert_eq!(spec.variants.len(), 1);
        let base = TrainingConfig::from_toml_str(include_str!(
            "../config/competitive_transfer_large_specialist_distillation.toml"
        ))
        .unwrap();
        let variant = &spec.variants[0];
        let expanded = base
            .with_rules_override(&variant.rules)
            .and_then(|config| config.with_scenario_override(&variant.scenario))
            .and_then(|config| config.with_combat_curriculum_override(&variant.combat_curriculum))
            .and_then(|config| {
                config.with_specialist_distillation_override(&variant.specialist_distillation)
            })
            .unwrap();
        assert_eq!(expanded.specialist_distillation.ecology_coeff, 0.10);
        assert_eq!(expanded.specialist_distillation.combat_coeff, 0.10);
        let mut normalized = expanded;
        normalized.specialist_distillation = base.specialist_distillation.clone();
        assert_eq!(normalized, base);
    }

    #[test]
    fn documented_specialist_coefficient_sweep_changes_only_coefficients() {
        let spec: RulesSweepSpec = toml::from_str(include_str!(
            "../config/specialist_distillation_coefficient_sweep.toml"
        ))
        .unwrap();
        validate_spec(&spec).unwrap();
        assert_eq!(spec.seeds, vec![42, 43, 44]);
        assert_eq!(spec.variants.len(), 3);

        let base = TrainingConfig::from_toml_str(include_str!(
            "../config/competitive_transfer_large_specialist_distillation.toml"
        ))
        .unwrap();
        let expanded = spec
            .variants
            .iter()
            .map(|variant| {
                base.with_rules_override(&variant.rules)
                    .and_then(|config| config.with_scenario_override(&variant.scenario))
                    .and_then(|config| {
                        config.with_combat_curriculum_override(&variant.combat_curriculum)
                    })
                    .and_then(|config| {
                        config
                            .with_specialist_distillation_override(&variant.specialist_distillation)
                    })
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        for (config, expected) in expanded.iter().zip([0.05, 0.10, 0.20]) {
            assert!(config.specialist_distillation.enabled);
            assert_eq!(config.specialist_distillation.ecology_coeff, expected);
            assert_eq!(config.specialist_distillation.combat_coeff, expected);

            let mut normalized = config.clone();
            normalized.specialist_distillation = base.specialist_distillation.clone();
            assert_eq!(normalized, base);
        }
    }

    #[test]
    fn documented_specialist_holdout_sweep_is_paired_and_uses_new_seeds() {
        let spec: RulesSweepSpec = toml::from_str(include_str!(
            "../config/specialist_distillation_holdout_sweep.toml"
        ))
        .unwrap();
        validate_spec(&spec).unwrap();
        assert_eq!(spec.seeds, vec![45, 46, 47, 48, 49]);
        assert_eq!(spec.variants.len(), 2);

        let base = TrainingConfig::from_toml_str(include_str!(
            "../config/competitive_transfer_large_skirmish_frequent.toml"
        ))
        .unwrap();
        let expanded = spec
            .variants
            .iter()
            .map(|variant| {
                base.with_rules_override(&variant.rules)
                    .and_then(|config| config.with_scenario_override(&variant.scenario))
                    .and_then(|config| {
                        config.with_combat_curriculum_override(&variant.combat_curriculum)
                    })
                    .and_then(|config| {
                        config
                            .with_specialist_distillation_override(&variant.specialist_distillation)
                    })
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(expanded[0], base);
        assert!(expanded[1].specialist_distillation.enabled);
        assert_eq!(expanded[1].specialist_distillation.ecology_coeff, 0.05);
        assert_eq!(expanded[1].specialist_distillation.combat_coeff, 0.05);
        let mut normalized = expanded[1].clone();
        normalized.specialist_distillation = expanded[0].specialist_distillation.clone();
        assert_eq!(normalized, expanded[0]);
    }

    #[test]
    fn documented_stage_capture_sweep_changes_only_distillation() {
        let spec: RulesSweepSpec = toml::from_str(include_str!(
            "../config/combat_retention_stage_capture_sweep.toml"
        ))
        .unwrap();
        validate_spec(&spec).unwrap();
        assert_eq!(spec.seeds, vec![47, 48, 49]);
        assert_eq!(spec.variants.len(), 2);

        let base = TrainingConfig::from_toml_str(include_str!(
            "../config/competitive_transfer_large_skirmish_stage_evaluation.toml"
        ))
        .unwrap();
        assert_eq!(base.eval_interval, 0);
        assert_eq!(base.checkpoint_interval, 0);
        assert_eq!(
            base.combat_curriculum
                .competency_evaluation_frontiers_sim_time_quanta_per_cycle,
            vec![65_536, 98_304]
        );

        let expanded = spec
            .variants
            .iter()
            .map(|variant| {
                base.with_rules_override(&variant.rules)
                    .and_then(|config| config.with_scenario_override(&variant.scenario))
                    .and_then(|config| {
                        config.with_combat_curriculum_override(&variant.combat_curriculum)
                    })
                    .and_then(|config| {
                        config
                            .with_specialist_distillation_override(&variant.specialist_distillation)
                    })
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(expanded[0], base);
        assert!(expanded[1].specialist_distillation.enabled);
        assert_eq!(expanded[1].specialist_distillation.ecology_coeff, 0.05);
        assert_eq!(expanded[1].specialist_distillation.combat_coeff, 0.05);
        let mut normalized = expanded[1].clone();
        normalized.specialist_distillation = expanded[0].specialist_distillation.clone();
        assert_eq!(normalized, expanded[0]);
    }

    #[test]
    fn documented_combat_precursor_sweep_changes_only_the_fallback_threshold() {
        let spec: RulesSweepSpec = toml::from_str(include_str!(
            "../config/specialist_distillation_precursor_sweep.toml"
        ))
        .unwrap();
        validate_spec(&spec).unwrap();
        assert_eq!(spec.seeds, vec![45, 46, 47, 48, 49]);
        assert_eq!(spec.variants.len(), 2);

        let base = TrainingConfig::from_toml_str(include_str!(
            "../config/competitive_transfer_large_specialist_distillation.toml"
        ))
        .unwrap();
        let expanded = spec
            .variants
            .iter()
            .map(|variant| {
                base.with_rules_override(&variant.rules)
                    .and_then(|config| config.with_scenario_override(&variant.scenario))
                    .and_then(|config| {
                        config.with_combat_curriculum_override(&variant.combat_curriculum)
                    })
                    .and_then(|config| {
                        config
                            .with_specialist_distillation_override(&variant.specialist_distillation)
                    })
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(expanded[0], base);
        assert_eq!(
            expanded[1]
                .specialist_distillation
                .combat_precursor_min_skirmish_damage,
            Some(1)
        );
        let mut normalized = expanded[1].clone();
        normalized
            .specialist_distillation
            .combat_precursor_min_skirmish_damage = None;
        assert_eq!(normalized, expanded[0]);
    }

    #[test]
    fn documented_signal_sweep_expands_every_strict_override() {
        let spec: RulesSweepSpec =
            toml::from_str(include_str!("../config/signal_rule_sweep.toml")).unwrap();
        validate_spec(&spec).unwrap();
        let base =
            TrainingConfig::from_toml_str(include_str!("../config/control_eval.toml")).unwrap();
        let expanded = spec
            .variants
            .iter()
            .map(|variant| {
                base.with_rules_override(&variant.rules)
                    .and_then(|config| config.with_scenario_override(&variant.scenario))
                    .and_then(|config| {
                        config.with_combat_curriculum_override(&variant.combat_curriculum)
                    })
                    .map(|config| (variant.name.as_str(), config))
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(expanded.len(), 12);
        let current_tile_only = expanded
            .iter()
            .find(|(name, _)| *name == "current-tile-only")
            .unwrap();
        assert_eq!(
            current_tile_only
                .1
                .env
                .rules
                .neighborhood
                .observations
                .signal
                .bits(),
            0
        );
        let cardinal = expanded
            .iter()
            .find(|(name, _)| *name == "cardinal-visibility")
            .unwrap();
        assert_eq!(
            cardinal.1.env.rules.neighborhood.observations.signal.bits(),
            90
        );
    }

    #[test]
    fn documented_starting_layout_sweep_covers_founder_and_assembly_controls() {
        let spec: RulesSweepSpec =
            toml::from_str(include_str!("../config/starting_layout_sweep.toml")).unwrap();
        validate_spec(&spec).unwrap();
        let base = TrainingConfig::from_toml_str(include_str!(
            "../config/objective_large_draw_smoke.toml"
        ))
        .unwrap();
        let expanded = spec
            .variants
            .iter()
            .map(|variant| {
                base.with_scenario_override(&variant.scenario)
                    .and_then(|config| {
                        config.with_combat_curriculum_override(&variant.combat_curriculum)
                    })
                    .map(|config| (variant.name.as_str(), config))
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(expanded.len(), 7);

        let single = expanded
            .iter()
            .find(|(name, _)| *name == "single-on-plant")
            .unwrap();
        assert_eq!(single.1.env.cells_per_team, 1);
        assert_eq!(
            single.1.env.resource_placement,
            crate::config::ResourcePlacement::OnAllCells
        );
        assert_eq!(
            single.1.env.starting_cell_layout,
            blob_engine::engine::StartingCellLayout::Block
        );

        let layouts = expanded
            .iter()
            .map(|(_, config)| config.env.starting_cell_layout)
            .collect::<HashSet<_>>();
        assert_eq!(layouts.len(), 6);
    }

    #[test]
    fn documented_resource_layout_sweep_crosses_independent_ecology_controls() {
        let spec: RulesSweepSpec =
            toml::from_str(include_str!("../config/resource_layout_sweep.toml")).unwrap();
        validate_spec(&spec).unwrap();
        let base = TrainingConfig::from_toml_str(include_str!(
            "../config/objective_large_draw_smoke.toml"
        ))
        .unwrap();
        let expanded = spec
            .variants
            .iter()
            .map(|variant| {
                base.with_scenario_override(&variant.scenario)
                    .and_then(|config| {
                        config.with_combat_curriculum_override(&variant.combat_curriculum)
                    })
                    .map(|config| (variant.name.as_str(), config))
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(expanded.len(), 8);

        let identities = expanded
            .iter()
            .map(|(_, config)| ScenarioProfile::from(&config.env).semantic_hash().unwrap())
            .collect::<HashSet<_>>();
        assert_eq!(identities.len(), expanded.len());
        assert!(expanded.iter().any(|(name, config)| {
            *name == "plant-islands"
                && matches!(
                    config.env.plant_layout,
                    blob_engine::world_gen::ResourceLayout::Islands { .. }
                )
                && matches!(
                    config.env.scattered_energy_layout,
                    blob_engine::world_gen::ResourceLayout::Uniform
                )
        }));
        assert!(expanded.iter().any(|(name, config)| {
            *name == "islands-and-corridors"
                && matches!(
                    config.env.plant_layout,
                    blob_engine::world_gen::ResourceLayout::Islands { .. }
                )
                && matches!(
                    config.env.scattered_energy_layout,
                    blob_engine::world_gen::ResourceLayout::Corridors { .. }
                )
        }));

        for (_, config) in expanded {
            for seed in &spec.seeds {
                crate::env::BlobEnv::new(config.env.clone(), config.reward.clone(), *seed);
            }
        }
    }

    #[test]
    fn corrected_micro_combat_ablation_initializes_every_stage_and_changes_only_rehearsal() {
        let spec: RulesSweepSpec =
            toml::from_str(include_str!("../config/micro_combat_ablation_256_v3.toml")).unwrap();
        validate_spec(&spec).unwrap();
        let base = TrainingConfig::from_toml_str(include_str!(
            "../config/micro_combat_ablation_256_v2_base.toml"
        ))
        .unwrap();
        base.validate().unwrap();
        let maintained = TrainingConfig::from_toml_str(include_str!(
            "../config/micro_combat_curriculum_256.toml"
        ))
        .unwrap();
        assert_eq!(
            base.combat_curriculum
                .micro_combat
                .suite
                .as_ref()
                .unwrap()
                .semantic_hash()
                .unwrap(),
            maintained
                .combat_curriculum
                .micro_combat
                .suite
                .as_ref()
                .unwrap()
                .semantic_hash()
                .unwrap()
        );
        let expanded = spec
            .variants
            .iter()
            .map(|variant| {
                base.with_scenario_override(&variant.scenario)
                    .and_then(|config| {
                        config.with_combat_curriculum_override(&variant.combat_curriculum)
                    })
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(spec.seeds.len(), 3);
        assert_eq!(expanded.len(), 2);
        assert_eq!(expanded[0].env.world_size, 256);
        assert_eq!(expanded[0].total_simulation_quanta_per_env, Some(1_048_576));
        assert!(!expanded[0].combat_curriculum.micro_combat.rollout_enabled);
        assert!(expanded[1].combat_curriculum.micro_combat.rollout_enabled);
        let mut normalized = expanded[1].clone();
        normalized.combat_curriculum.micro_combat.rollout_enabled = false;
        assert_eq!(normalized, expanded[0]);

        let stage_starts = [
            (crate::config::FeedingCurriculumStage::OnFood, 0),
            (crate::config::FeedingCurriculumStage::AdjacentFood, 32_768),
            (crate::config::FeedingCurriculumStage::Contact, 65_536),
            (crate::config::FeedingCurriculumStage::Skirmish, 131_072),
            (crate::config::FeedingCurriculumStage::Competitive, 196_608),
        ];
        for config in &expanded {
            for &seed in &spec.seeds {
                for &(stage, time) in &stage_starts {
                    let environment = config.rollout_environment(stage, time);
                    crate::env::BlobEnv::new(environment, config.reward.clone(), seed);
                }
            }
        }
    }

    #[test]
    fn action_acquisition_ablation_changes_only_the_attack_mixture() {
        let spec: RulesSweepSpec = toml::from_str(include_str!(
            "../config/micro_combat_action_acquisition_256.toml"
        ))
        .unwrap();
        validate_spec(&spec).unwrap();
        let base = TrainingConfig::from_toml_str(include_str!(
            "../config/micro_combat_ablation_256_v2_base.toml"
        ))
        .unwrap();
        let expanded = spec
            .variants
            .iter()
            .map(|variant| {
                base.with_scenario_override(&variant.scenario)
                    .and_then(|config| {
                        config.with_combat_curriculum_override(&variant.combat_curriculum)
                    })
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(expanded.len(), 2);
        for config in &expanded {
            assert!(config.combat_curriculum.micro_combat.rollout_enabled);
            assert_eq!(
                config
                    .combat_curriculum
                    .micro_combat
                    .min_attack_commitments_per_episode,
                0.25
            );
        }
        assert_eq!(
            expanded[0]
                .combat_curriculum
                .micro_combat
                .attack_action_kind_exploration_floor,
            0.0
        );
        assert_eq!(
            expanded[1]
                .combat_curriculum
                .micro_combat
                .attack_action_kind_exploration_floor,
            0.5
        );
        let mut normalized = expanded[1].clone();
        normalized
            .combat_curriculum
            .micro_combat
            .attack_action_kind_exploration_floor = 0.0;
        assert_eq!(normalized, expanded[0]);

        for config in &expanded {
            for &seed in &spec.seeds {
                for (stage, time) in [
                    (crate::config::FeedingCurriculumStage::OnFood, 0),
                    (crate::config::FeedingCurriculumStage::AdjacentFood, 32_768),
                    (crate::config::FeedingCurriculumStage::Contact, 65_536),
                    (crate::config::FeedingCurriculumStage::Skirmish, 131_072),
                    (crate::config::FeedingCurriculumStage::Competitive, 196_608),
                ] {
                    let environment = config.rollout_environment(stage, time);
                    crate::env::BlobEnv::new(environment, config.reward.clone(), seed);
                }
            }
        }
    }

    #[test]
    fn experiment_identity_ignores_only_the_artifact_destination() {
        let mut config = TrainingConfig {
            checkpoint_dir: "/first/output".into(),
            ..TrainingConfig::default()
        };
        let first = experiment_config_hash(&config).unwrap();
        config.checkpoint_dir = "/second/output".into();
        assert_eq!(experiment_config_hash(&config).unwrap(), first);
        config.seed = config.seed.wrapping_add(1);
        assert_ne!(experiment_config_hash(&config).unwrap(), first);
    }
}
