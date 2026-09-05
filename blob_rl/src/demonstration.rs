//! Deterministic, hash-bound behavior-cloning demonstrations.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use blob_interface::reference_mind::{
    ReferenceMemoryUpdate, ReferenceMindDecision, ReferenceMindInput,
};
use burn::prelude::Backend;
use serde::{Deserialize, Serialize};

use crate::action::{
    decode_policy_choice, decompose_policy_action, encode_decision, policy_action_family,
    PolicyActionFamily, PolicyActionKind, PolicyChoice, NUM_ACTIONS, NUM_AMOUNT_CHOICES,
    NUM_SIGNAL_CHOICES, NUM_SIGNAL_STRENGTH_CHOICES,
};
use crate::config::{ScenarioProfile, TrainingConfig};
use crate::control_matrix::MaintainedMindProfile;
use crate::env::{BlobEnv, EpisodeOutcome};
use crate::evaluation::greedy_policy_choices_with_kind_logits;
use crate::micro_combat::{MicroCombatScenario, MicroCombatSuiteConfig};
use crate::micro_combat_observation_policy::{
    observation_abstraction, observation_abstraction_sha256,
    policy_sha256 as observation_policy_sha256, select_abstract_rule,
    ObservationPolicySearchReport, MICRO_COMBAT_OBSERVATION_POLICY_SCHEMA_VERSION,
    MICRO_COMBAT_OBSERVATION_POLICY_SEARCH_REPORT_SCHEMA_VERSION,
};
use crate::micro_combat_planner::{mind_observation_sha256, SingleCellSearchSimulator};
use crate::model::PolicyValueNet;
use crate::observation::{
    Observation, CURRENT_TILE_LOOSE_ENERGY_FEATURE, CURRENT_TILE_PLANT_CAPACITY_FEATURE,
    CURRENT_TILE_PLANT_ENERGY_FEATURE, HEADER_FEATURES, OBS_DIM, SLOT_FEATURES,
    SLOT_NEIGHBOR_PRESENT_FEATURE, SLOT_PLANT_ENERGY_FEATURE,
};
use crate::sweep::sha256;
use crate::viability::mind_abi_hash;

pub const DEMONSTRATION_SCHEMA_VERSION: u32 = 18;
pub const POLICY_KIND_MARGIN_THRESHOLDS_MICROLOGITS: [u32; 6] =
    [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000];
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

#[derive(Debug, Clone)]
pub struct PolicyCorrectionOptions {
    pub teacher: MaintainedMindProfile,
    pub seeds: Vec<u64>,
    pub max_samples: usize,
    pub behavior_clone_metadata_sha256: String,
    pub behavior_clone_model_sha256: String,
    pub minimum_policy_agreement_rate: f64,
    pub label_mode: PolicyCorrectionLabelMode,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PolicyCorrectionLabelMode {
    #[default]
    FullTeacher,
    AdjacentConsumeControl,
    AdjacentConsumeTreatment,
    MoveTargetControl,
    MoveTargetTreatment,
}

impl PolicyCorrectionLabelMode {
    pub const fn is_targeted(self) -> bool {
        matches!(
            self,
            Self::AdjacentConsumeControl
                | Self::AdjacentConsumeTreatment
                | Self::MoveTargetControl
                | Self::MoveTargetTreatment
        )
    }

    const fn is_treatment(self) -> bool {
        matches!(
            self,
            Self::AdjacentConsumeTreatment | Self::MoveTargetTreatment
        )
    }
}

#[derive(Debug, Clone)]
pub struct ObservationPolicyDemonstrationOptions {
    /// SHA-256 of the immutable JSON search report supplying the policy.
    pub source_report_sha256: String,
    pub max_samples: usize,
    /// Minimum disjoint holdout seeds that must all be covered and successful
    /// before synthesis trajectories can be admitted as training data.
    pub minimum_successful_holdout_seeds: usize,
}

/// How the labeled observations were reached. Correction datasets deliberately
/// keep complete policy-induced trajectory prefixes rather than isolated
/// mistakes, so recurrent training never invents continuity across omitted
/// decisions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
// This is immutable, serialized artifact metadata instantiated once per
// dataset. Keeping correction diagnostics inline preserves a stable, flat JSON
// schema; enum storage size is not on a simulation or training hot path.
#[allow(clippy::large_enum_variant)]
pub enum DemonstrationCollection {
    TeacherRollout,
    GreedyPolicyCorrection {
        behavior_clone_metadata_sha256: String,
        behavior_clone_model_sha256: String,
        policy_disagreement_samples: usize,
        teacher_attack_policy_non_attack_samples: usize,
        /// Raw teacher decisions whose continuous parameters were projected to
        /// the nearest legal discrete policy choice before storage.
        projected_teacher_samples: usize,
        /// Complete discrete choice agreement on the exact input shared by the
        /// frozen policy and teacher.
        #[serde(default)]
        exact_policy_agreement_samples: usize,
        #[serde(default)]
        policy_agreement_ppm: u32,
        #[serde(default)]
        minimum_policy_agreement_ppm: u32,
        #[serde(default)]
        agreement_passed: bool,
        /// Policy-side Wait..Signal marginals and teacher-major confusion
        /// matrix. Schema 13 correction datasets decode with empty defaults.
        #[serde(default)]
        policy_action_family_samples: [usize; PolicyActionFamily::COUNT],
        #[serde(default)]
        action_family_confusion: Vec<usize>,
        /// Action-kind disagreements and the selected policy kind's logit
        /// advantage over the teacher kind, quantized at 1e-6 logit. Crossover
        /// counts use [`POLICY_KIND_MARGIN_THRESHOLDS_MICROLOGITS`] in order.
        #[serde(default)]
        action_kind_disagreement_samples: usize,
        #[serde(default)]
        policy_kind_advantage_micrologit_sum: u64,
        #[serde(default)]
        policy_kind_advantage_micrologit_max: u32,
        #[serde(default)]
        policy_kind_advantage_within_threshold_samples:
            [usize; POLICY_KIND_MARGIN_THRESHOLDS_MICROLOGITS.len()],
        #[serde(default)]
        action_family_policy_kind_advantage_micrologit_sums: Vec<u64>,
        #[serde(default)]
        action_family_kind_disagreement_samples: Vec<usize>,
        /// Schema 18 can retain complete recurrent prefixes while changing
        /// labels only at a declared action-kind or target-selection seam.
        #[serde(default)]
        label_mode: PolicyCorrectionLabelMode,
        #[serde(default)]
        teacher_action_family_samples: [usize; PolicyActionFamily::COUNT],
        #[serde(default)]
        eligible_correction_samples: usize,
        #[serde(default)]
        active_correction_samples: usize,
        /// Legacy schema-16 and earlier corrections projected zero here.
        /// Schema 17 requires `false`: deterministic greedy inference consumes
        /// the canonical, seed-reproducible cell-private random input.
        zero_randomness: bool,
    },
    /// Successful trajectories replayed from a policy synthesized only from
    /// anonymous Mind observations. Final holdout seeds in the source report
    /// are deliberately excluded from this training dataset.
    ObservationPolicyOracle {
        source_report_sha256: String,
        policy_sha256: String,
        scenario: String,
        zero_randomness: bool,
        coverage_complete: bool,
        objective_success: bool,
        #[serde(default)]
        holdout_seed_count: usize,
        #[serde(default)]
        minimum_successful_holdout_seeds: usize,
        #[serde(default)]
        holdout_coverage_complete: bool,
        #[serde(default)]
        holdout_objective_success: bool,
    },
    /// Sparse, exact-state supervision derived from a robust counterfactual
    /// frontier. The label changes only Move effort; its target remains the
    /// frozen policy's observation-local choice.
    CounterfactualEffortCorrection {
        source_counterfactual_sha256: String,
        source_value_sha256: String,
        behavior_clone_model_sha256: String,
        perspective: String,
        horizon_quanta: u64,
        recurrent_size: usize,
        minimum_effort: u8,
        active_correction: bool,
    },
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
    /// Present only for schema-15+ policy corrections. Keeping the chosen
    /// action and margin beside the teacher label makes aggregate diagnostics
    /// independently recomputable from the hash-bound payload.
    #[serde(default)]
    pub correction_policy_action: Option<u16>,
    #[serde(default)]
    pub correction_policy_kind_advantage_micrologits: Option<u32>,
    /// Actual maintained-teacher action before a targeted control/treatment
    /// decides whether that label is active. Schema 17 and earlier omit it.
    #[serde(default)]
    pub correction_teacher_action: Option<u16>,
    #[serde(default)]
    pub correction_eligible: bool,
    #[serde(default)]
    pub active_correction: bool,
    /// Exact cell-private recurrent state at an isolated counterfactual
    /// decision. Ordinary rollout datasets leave this empty and reconstruct
    /// memory from their complete trajectory prefixes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub initial_policy_memory: Vec<f32>,
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
    /// Human-readable teacher identity. Exact machine-verifiable provenance is
    /// carried by `collection`.
    pub teacher: String,
    pub collection: DemonstrationCollection,
    pub seeds: Vec<u64>,
    /// Hash of the source TOML before CLI scenario overrides.
    pub source_config_sha256: String,
    /// Hash of the validated effective config used to generate every sample.
    pub effective_config_sha256: String,
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

pub fn verify_feeding_correction_pair(
    control: &LoadedDemonstrations,
    treatment: &LoadedDemonstrations,
) -> Result<usize, String> {
    let common_manifest = |left: &DemonstrationManifest, right: &DemonstrationManifest| {
        left.schema_version == DEMONSTRATION_SCHEMA_VERSION
            && right.schema_version == DEMONSTRATION_SCHEMA_VERSION
            && left.mind_abi_sha256 == right.mind_abi_sha256
            && left.teacher == right.teacher
            && left.seeds == right.seeds
            && left.source_config_sha256 == right.source_config_sha256
            && left.effective_config_sha256 == right.effective_config_sha256
            && left.semantic_ruleset_hash == right.semantic_ruleset_hash
            && left.compiled_ruleset_hash == right.compiled_ruleset_hash
            && left.scenario_hash == right.scenario_hash
            && left.observation_dim == right.observation_dim
            && left.action_count == right.action_count
            && left.samples == right.samples
            && left.completed_episodes == right.completed_episodes
            && left.wins == right.wins
            && left.losses == right.losses
            && left.timeouts == right.timeouts
    };
    if !common_manifest(&control.manifest, &treatment.manifest) {
        return Err("feeding-correction pair does not share one trajectory identity".into());
    }
    let (
        DemonstrationCollection::GreedyPolicyCorrection {
            behavior_clone_metadata_sha256: control_metadata,
            behavior_clone_model_sha256: control_model,
            label_mode: control_mode,
            eligible_correction_samples: control_eligible,
            active_correction_samples: 0,
            ..
        },
        DemonstrationCollection::GreedyPolicyCorrection {
            behavior_clone_metadata_sha256: treatment_metadata,
            behavior_clone_model_sha256: treatment_model,
            label_mode: treatment_mode,
            eligible_correction_samples: treatment_eligible,
            active_correction_samples: treatment_active,
            ..
        },
    ) = (&control.manifest.collection, &treatment.manifest.collection)
    else {
        return Err("datasets are not a feeding control/treatment pair".into());
    };
    let modes_match = matches!(
        (*control_mode, *treatment_mode),
        (
            PolicyCorrectionLabelMode::AdjacentConsumeControl,
            PolicyCorrectionLabelMode::AdjacentConsumeTreatment
        ) | (
            PolicyCorrectionLabelMode::MoveTargetControl,
            PolicyCorrectionLabelMode::MoveTargetTreatment
        )
    );
    if control_metadata != treatment_metadata
        || control_model != treatment_model
        || !modes_match
        || control_eligible == &0
        || control_eligible != treatment_eligible
        || treatment_active != treatment_eligible
    {
        return Err("feeding-correction pair metadata or correction counts differ".into());
    }
    for (control_sample, treatment_sample) in control
        .payload
        .samples
        .iter()
        .zip(&treatment.payload.samples)
    {
        if control_sample.source_seed != treatment_sample.source_seed
            || control_sample.source_cell != treatment_sample.source_cell
            || control_sample.observation != treatment_sample.observation
            || control_sample.action_mask != treatment_sample.action_mask
            || control_sample.correction_policy_action != treatment_sample.correction_policy_action
            || control_sample.correction_teacher_action
                != treatment_sample.correction_teacher_action
            || control_sample.correction_policy_kind_advantage_micrologits
                != treatment_sample.correction_policy_kind_advantage_micrologits
            || control_sample.correction_eligible != treatment_sample.correction_eligible
            || control_sample.active_correction
            || treatment_sample.active_correction != treatment_sample.correction_eligible
            || control_sample.initial_policy_memory != treatment_sample.initial_policy_memory
        {
            return Err("feeding-correction pair changed its policy-induced trajectory".into());
        }
        let policy_action = control_sample
            .correction_policy_action
            .ok_or("feeding-correction control omitted its policy action")?;
        let teacher_action = control_sample
            .correction_teacher_action
            .ok_or("feeding-correction control omitted its teacher action")?;
        if control_sample.action != policy_action
            || treatment_sample.action
                != if treatment_sample.correction_eligible {
                    teacher_action
                } else {
                    policy_action
                }
        {
            return Err("feeding-correction pair changed more than its active labels".into());
        }
        if control_sample.amount_mask != treatment_sample.amount_mask
            || control_sample.amount != treatment_sample.amount
            || control_sample.signal_mask != treatment_sample.signal_mask
            || control_sample.signal != treatment_sample.signal
            || control_sample.signal_strength_mask != treatment_sample.signal_strength_mask
            || control_sample.signal_strength != treatment_sample.signal_strength
            || control_sample.exact_round_trip != treatment_sample.exact_round_trip
            || control_sample.memory_replacement != treatment_sample.memory_replacement
        {
            return Err("feeding-correction pair changed a non-action label field".into());
        }
    }
    Ok(*control_eligible)
}

fn validate_options(options: &DemonstrationOptions) -> Result<(), String> {
    validate_seed_sample_options(&options.seeds, options.max_samples)
}

fn validate_seed_sample_options(seeds: &[u64], max_samples: usize) -> Result<(), String> {
    if seeds.is_empty() || max_samples == 0 {
        return Err("demonstrations require seeds and a positive sample cap".into());
    }
    let mut unique = seeds.to_vec();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != seeds.len() {
        return Err("demonstration seeds must be unique".into());
    }
    Ok(())
}

fn sample_from_decision(
    source_seed: u64,
    source_cell: usize,
    input: &ReferenceMindInput,
    decision: &ReferenceMindDecision,
) -> Result<(DemonstrationSample, PolicyActionFamily), String> {
    let choice = encode_decision(decision, input)
        .ok_or_else(|| "teacher emitted an unencodable decision".to_string())?;
    let observation = Observation::from_reference(input);
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
        return Err("teacher emitted an action outside the policy mask".into());
    }
    let decoded = decode_policy_choice(choice, input);
    let family = policy_action_family(choice.action)
        .expect("encoded teacher decision belongs to a policy family");
    Ok((
        DemonstrationSample {
            source_seed,
            source_cell: u64::try_from(source_cell)
                .map_err(|_| "cell ID exceeds demonstration trajectory key".to_string())?,
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
            exact_round_trip: decoded == *decision,
            memory_replacement: matches!(decision.memory_update, ReferenceMemoryUpdate::Replace(_)),
            correction_policy_action: None,
            correction_policy_kind_advantage_micrologits: None,
            correction_teacher_action: None,
            correction_eligible: false,
            active_correction: false,
            initial_policy_memory: Vec::new(),
        },
        family,
    ))
}

pub fn generate_demonstrations(
    config: &TrainingConfig,
    source_config_sha256: String,
    options: &DemonstrationOptions,
) -> Result<(DemonstrationManifest, DemonstrationPayload), String> {
    config.validate()?;
    validate_options(options)?;
    let scenario_hash = ScenarioProfile::from(&config.env).semantic_hash()?;
    let effective_config_sha256 =
        sha256(&serde_json::to_vec(config).map_err(|error| {
            format!("failed to encode effective demonstration config: {error}")
        })?);
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
                if samples.len() < options.max_samples && seed_samples < samples_per_seed {
                    let (sample, family) =
                        sample_from_decision(*seed, cell_id.0, &input, &decision)
                            .map_err(|error| format!("teacher {} {error}", options.teacher))?;
                    exact_round_trip_samples += usize::from(sample.exact_round_trip);
                    memory_replacement_samples += usize::from(sample.memory_replacement);
                    action_family_samples[family.index()] += 1;
                    samples.push(sample);
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
        teacher: options.teacher.to_string(),
        collection: DemonstrationCollection::TeacherRollout,
        seeds: options.seeds.clone(),
        source_config_sha256,
        effective_config_sha256,
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

/// Roll out one verified greedy policy while labeling every visited state with
/// a maintained teacher. Keeping the full per-cell prefixes is essential: a
/// disagreement-only sample set would silently splice unrelated recurrent
/// states together during behavior cloning.
pub fn generate_policy_correction_demonstrations<B: Backend>(
    config: &TrainingConfig,
    source_config_sha256: String,
    options: &PolicyCorrectionOptions,
    model: &PolicyValueNet<B>,
    device: &B::Device,
) -> Result<(DemonstrationManifest, DemonstrationPayload), String>
where
    f32: From<B::FloatElem>,
{
    config.validate()?;
    validate_seed_sample_options(&options.seeds, options.max_samples)?;
    if !is_sha256(&options.behavior_clone_metadata_sha256)
        || !is_sha256(&options.behavior_clone_model_sha256)
    {
        return Err("policy correction requires valid behavior-clone hashes".into());
    }
    if !options.minimum_policy_agreement_rate.is_finite()
        || !(0.0..=1.0).contains(&options.minimum_policy_agreement_rate)
    {
        return Err("policy correction minimum agreement must be finite and in [0, 1]".into());
    }
    let scenario_hash = ScenarioProfile::from(&config.env).semantic_hash()?;
    let effective_config_sha256 = sha256(
        &serde_json::to_vec(config)
            .map_err(|error| format!("failed to encode effective correction config: {error}"))?,
    );
    let semantic_ruleset_hash = config.env.rules.semantic_hash().to_string();
    let mut samples = Vec::with_capacity(options.max_samples);
    let mut exact_round_trip_samples = 0usize;
    let mut memory_replacement_samples = 0usize;
    let mut action_family_samples = [0usize; PolicyActionFamily::COUNT];
    let mut policy_disagreement_samples = 0usize;
    let mut teacher_attack_policy_non_attack_samples = 0usize;
    let mut projected_teacher_samples = 0usize;
    let mut exact_policy_agreement_samples = 0usize;
    let mut policy_action_family_samples = [0usize; PolicyActionFamily::COUNT];
    let mut action_family_confusion = vec![0usize; PolicyActionFamily::COUNT.pow(2)];
    let mut action_kind_disagreement_samples = 0usize;
    let mut policy_kind_advantage_micrologit_sum = 0u64;
    let mut policy_kind_advantage_micrologit_max = 0u32;
    let mut policy_kind_advantage_within_threshold_samples =
        [0usize; POLICY_KIND_MARGIN_THRESHOLDS_MICROLOGITS.len()];
    let mut action_family_policy_kind_advantage_micrologit_sums =
        vec![0u64; PolicyActionFamily::COUNT.pow(2)];
    let mut action_family_kind_disagreement_samples =
        vec![0usize; PolicyActionFamily::COUNT.pow(2)];
    let mut teacher_action_family_samples = [0usize; PolicyActionFamily::COUNT];
    let mut eligible_correction_samples = 0usize;
    let mut active_correction_samples = 0usize;
    let mut completed_episodes = 0usize;
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut timeouts = 0usize;
    let mut compiled_ruleset_hash = None;
    let samples_per_seed = options.max_samples.div_ceil(options.seeds.len());

    'seeds: for seed in &options.seeds {
        let mut env = BlobEnv::new(config.env.clone(), config.reward.clone(), *seed);
        let compiled = env.compiled_ruleset_hash();
        if compiled_ruleset_hash
            .as_ref()
            .is_some_and(|expected| expected != &compiled)
        {
            return Err("identical correction configuration compiled differently".into());
        }
        compiled_ruleset_hash = Some(compiled);
        let mut seed_samples = 0usize;
        loop {
            let prepared = env.prepare_training_reference_inputs()?;
            let policy_observations = prepared
                .iter()
                .map(|(cell_id, input)| crate::env::PolicyObservation {
                    cell_id: *cell_id,
                    observation: Observation::from_reference(input),
                    private_memory: input.private_memory.clone(),
                })
                .collect::<Vec<_>>();
            let policy_choices =
                greedy_policy_choices_with_kind_logits(model, &policy_observations, device);
            if policy_choices.len() != prepared.len() {
                return Err("policy correction lost a ready-cell observation".into());
            }
            let step_policy_choices = policy_choices
                .iter()
                .map(|(cell_id, choice, memory, _)| (*cell_id, *choice, memory.clone()))
                .collect::<Vec<_>>();
            for ((cell_id, input), (policy_cell, policy_choice, _, action_kind_logits)) in
                prepared.iter().zip(&policy_choices)
            {
                if cell_id != policy_cell {
                    return Err("policy correction changed ready-cell ordering".into());
                }
                let raw_teacher_decision = options.teacher.decide(input);
                let teacher_choice =
                    encode_decision(&raw_teacher_decision, input).ok_or_else(|| {
                        format!(
                            "correction teacher {} emitted an unencodable decision",
                            options.teacher
                        )
                    })?;
                let teacher_decision = decode_policy_choice(teacher_choice, input);
                if samples.len() < options.max_samples && seed_samples < samples_per_seed {
                    let eligible = correction_eligible(
                        options.label_mode,
                        input,
                        *policy_choice,
                        teacher_choice,
                    );
                    let active = eligible && options.label_mode.is_treatment();
                    let policy_decision = decode_policy_choice(*policy_choice, input);
                    let label_decision =
                        if options.label_mode == PolicyCorrectionLabelMode::FullTeacher || active {
                            &teacher_decision
                        } else {
                            &policy_decision
                        };
                    let (mut sample, label_family) =
                        sample_from_decision(*seed, cell_id.0, input, label_decision).map_err(
                            |error| format!("correction teacher {} {error}", options.teacher),
                        )?;
                    let disagrees = teacher_choice != *policy_choice;
                    let policy_family = policy_action_family(policy_choice.action)
                        .ok_or("policy correction choice has no action family")?;
                    let teacher_family = policy_action_family(teacher_choice.action)
                        .ok_or("teacher correction choice has no action family")?;
                    let teacher_kind = decompose_policy_action(teacher_choice.action)
                        .ok_or("teacher correction choice has no action kind")?
                        .kind;
                    let policy_kind = decompose_policy_action(policy_choice.action)
                        .ok_or("policy correction choice has no action kind")?
                        .kind;
                    let kind_disagrees = teacher_kind != policy_kind;
                    let advantage = if kind_disagrees {
                        quantize_policy_kind_advantage(
                            action_kind_logits[policy_kind],
                            action_kind_logits[teacher_kind],
                        )?
                    } else {
                        0
                    };
                    sample.correction_policy_action = Some(
                        u16::try_from(policy_choice.action)
                            .map_err(|_| "policy action catalog exceeds u16")?,
                    );
                    sample.correction_policy_kind_advantage_micrologits = Some(advantage);
                    sample.correction_teacher_action = Some(
                        u16::try_from(teacher_choice.action)
                            .map_err(|_| "teacher action catalog exceeds u16")?,
                    );
                    sample.correction_eligible = eligible;
                    sample.active_correction = active;
                    policy_disagreement_samples += usize::from(disagrees);
                    action_kind_disagreement_samples += usize::from(kind_disagrees);
                    exact_policy_agreement_samples += usize::from(!disagrees);
                    projected_teacher_samples +=
                        usize::from(teacher_decision != raw_teacher_decision);
                    teacher_attack_policy_non_attack_samples += usize::from(
                        teacher_family == PolicyActionFamily::Attack
                            && policy_action_family(policy_choice.action)
                                != Some(PolicyActionFamily::Attack),
                    );
                    exact_round_trip_samples += usize::from(sample.exact_round_trip);
                    memory_replacement_samples += usize::from(sample.memory_replacement);
                    action_family_samples[label_family.index()] += 1;
                    teacher_action_family_samples[teacher_family.index()] += 1;
                    eligible_correction_samples += usize::from(eligible);
                    active_correction_samples += usize::from(active);
                    policy_action_family_samples[policy_family.index()] += 1;
                    action_family_confusion[teacher_family.index() * PolicyActionFamily::COUNT
                        + policy_family.index()] += 1;
                    if kind_disagrees {
                        policy_kind_advantage_micrologit_sum = policy_kind_advantage_micrologit_sum
                            .checked_add(u64::from(advantage))
                            .ok_or("policy correction margin sum overflowed")?;
                        policy_kind_advantage_micrologit_max =
                            policy_kind_advantage_micrologit_max.max(advantage);
                        for (count, threshold) in policy_kind_advantage_within_threshold_samples
                            .iter_mut()
                            .zip(POLICY_KIND_MARGIN_THRESHOLDS_MICROLOGITS)
                        {
                            *count += usize::from(advantage <= threshold);
                        }
                        let pair = teacher_family.index() * PolicyActionFamily::COUNT
                            + policy_family.index();
                        action_family_policy_kind_advantage_micrologit_sums[pair] =
                            action_family_policy_kind_advantage_micrologit_sums[pair]
                                .checked_add(u64::from(advantage))
                                .ok_or("policy correction pair margin sum overflowed")?;
                        action_family_kind_disagreement_samples[pair] += 1;
                    }
                    samples.push(sample);
                    seed_samples += 1;
                }
            }
            let result = env.step_with_policy_memory(&step_policy_choices);
            if result.done {
                completed_episodes += 1;
                match result.outcome {
                    Some(EpisodeOutcome::Win) => wins += 1,
                    Some(EpisodeOutcome::Loss) => losses += 1,
                    Some(EpisodeOutcome::Timeout) => timeouts += 1,
                    Some(EpisodeOutcome::SafetyAbort) => {
                        return Err(format!(
                            "policy-correction seed {seed} reached the decision-frontier safety limit"
                        ));
                    }
                    None => return Err("completed correction rollout omitted its outcome".into()),
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
        return Err("policy correction produced no ready-cell decisions".into());
    }
    if options.label_mode.is_targeted() && eligible_correction_samples == 0 {
        return Err(
            "targeted policy correction found no adjacent empty-tile Consume/teacher-Move states"
                .into(),
        );
    }

    let payload = DemonstrationPayload {
        schema_version: DEMONSTRATION_SCHEMA_VERSION,
        samples,
    };
    let payload_bytes = rmp_serde::to_vec_named(&payload)
        .map_err(|error| format!("failed to encode correction payload: {error}"))?;
    let policy_agreement_ppm = agreement_ppm(exact_policy_agreement_samples, payload.samples.len());
    let minimum_policy_agreement_ppm =
        (options.minimum_policy_agreement_rate * 1_000_000.0).round() as u32;
    let manifest = DemonstrationManifest {
        schema_version: DEMONSTRATION_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        mind_abi_sha256: mind_abi_hash(),
        teacher: options.teacher.to_string(),
        collection: DemonstrationCollection::GreedyPolicyCorrection {
            behavior_clone_metadata_sha256: options.behavior_clone_metadata_sha256.clone(),
            behavior_clone_model_sha256: options.behavior_clone_model_sha256.clone(),
            policy_disagreement_samples,
            teacher_attack_policy_non_attack_samples,
            projected_teacher_samples,
            exact_policy_agreement_samples,
            policy_agreement_ppm,
            minimum_policy_agreement_ppm,
            agreement_passed: agreement_meets(
                exact_policy_agreement_samples,
                payload.samples.len(),
                minimum_policy_agreement_ppm,
            ),
            policy_action_family_samples,
            action_family_confusion,
            action_kind_disagreement_samples,
            policy_kind_advantage_micrologit_sum,
            policy_kind_advantage_micrologit_max,
            policy_kind_advantage_within_threshold_samples,
            action_family_policy_kind_advantage_micrologit_sums,
            action_family_kind_disagreement_samples,
            label_mode: options.label_mode,
            teacher_action_family_samples,
            eligible_correction_samples,
            active_correction_samples,
            zero_randomness: false,
        },
        seeds: options.seeds.clone(),
        source_config_sha256,
        effective_config_sha256,
        semantic_ruleset_hash,
        compiled_ruleset_hash: compiled_ruleset_hash
            .expect("a nonempty correction dataset has a compiled ruleset"),
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

fn adjacent_consume_correction_eligible(
    input: &ReferenceMindInput,
    policy_choice: PolicyChoice,
    teacher_choice: PolicyChoice,
) -> bool {
    let Some(policy) = decompose_policy_action(policy_choice.action) else {
        return false;
    };
    let Some(teacher) = decompose_policy_action(teacher_choice.action) else {
        return false;
    };
    if policy.kind != PolicyActionKind::Consume.index()
        || teacher.kind != PolicyActionKind::Move.index()
        || input.current_tile.plant_capacity != 0
        || input.current_tile.plant_energy != 0
        || input.current_tile.loose_energy != 0
    {
        return false;
    }
    input.slots.get(teacher.target).is_some_and(|slot| {
        slot.reachable
            && slot.plant_energy.is_some_and(|energy| energy > 0)
            && slot.neighbor.is_none()
    })
}

fn move_target_correction_eligible(
    input: &ReferenceMindInput,
    policy_choice: PolicyChoice,
    teacher_choice: PolicyChoice,
) -> bool {
    let Some(policy) = decompose_policy_action(policy_choice.action) else {
        return false;
    };
    let Some(teacher) = decompose_policy_action(teacher_choice.action) else {
        return false;
    };
    if policy.kind != PolicyActionKind::Move.index()
        || teacher.kind != PolicyActionKind::Move.index()
        || policy.target == teacher.target
        || policy.effort != teacher.effort
        || input.current_tile.plant_capacity != 0
        || input.current_tile.plant_energy != 0
        || input.current_tile.loose_energy != 0
    {
        return false;
    }
    input.slots.get(teacher.target).is_some_and(|slot| {
        slot.reachable
            && slot.plant_energy.is_some_and(|energy| energy > 0)
            && slot.neighbor.is_none()
    })
}

fn correction_eligible(
    mode: PolicyCorrectionLabelMode,
    input: &ReferenceMindInput,
    policy_choice: PolicyChoice,
    teacher_choice: PolicyChoice,
) -> bool {
    match mode {
        PolicyCorrectionLabelMode::FullTeacher => false,
        PolicyCorrectionLabelMode::AdjacentConsumeControl
        | PolicyCorrectionLabelMode::AdjacentConsumeTreatment => {
            adjacent_consume_correction_eligible(input, policy_choice, teacher_choice)
        }
        PolicyCorrectionLabelMode::MoveTargetControl
        | PolicyCorrectionLabelMode::MoveTargetTreatment => {
            move_target_correction_eligible(input, policy_choice, teacher_choice)
        }
    }
}

fn adjacent_consume_sample_eligible(
    observation: &[f32],
    policy_action: usize,
    teacher_action: usize,
) -> bool {
    let Some(policy) = decompose_policy_action(policy_action) else {
        return false;
    };
    let Some(teacher) = decompose_policy_action(teacher_action) else {
        return false;
    };
    if policy.kind != PolicyActionKind::Consume.index()
        || teacher.kind != PolicyActionKind::Move.index()
        || observation
            .get(CURRENT_TILE_PLANT_CAPACITY_FEATURE)
            .is_none_or(|value| *value > 0.0)
        || observation
            .get(CURRENT_TILE_PLANT_ENERGY_FEATURE)
            .is_none_or(|value| *value > 0.0)
        || observation
            .get(CURRENT_TILE_LOOSE_ENERGY_FEATURE)
            .is_none_or(|value| *value > 0.0)
    {
        return false;
    }
    let base = HEADER_FEATURES.saturating_add(teacher.target.saturating_mul(SLOT_FEATURES));
    observation.get(base).is_some_and(|present| *present > 0.5)
        && observation
            .get(base + 1)
            .is_some_and(|reachable| *reachable > 0.5)
        && observation
            .get(base + SLOT_PLANT_ENERGY_FEATURE)
            .is_some_and(|energy| *energy > 0.0)
        && observation
            .get(base + SLOT_NEIGHBOR_PRESENT_FEATURE)
            .is_some_and(|neighbor| *neighbor <= 0.5)
}

fn move_target_sample_eligible(
    observation: &[f32],
    policy_action: usize,
    teacher_action: usize,
) -> bool {
    let Some(policy) = decompose_policy_action(policy_action) else {
        return false;
    };
    let Some(teacher) = decompose_policy_action(teacher_action) else {
        return false;
    };
    if policy.kind != PolicyActionKind::Move.index()
        || teacher.kind != PolicyActionKind::Move.index()
        || policy.target == teacher.target
        || policy.effort != teacher.effort
        || observation
            .get(CURRENT_TILE_PLANT_CAPACITY_FEATURE)
            .is_none_or(|value| *value > 0.0)
        || observation
            .get(CURRENT_TILE_PLANT_ENERGY_FEATURE)
            .is_none_or(|value| *value > 0.0)
        || observation
            .get(CURRENT_TILE_LOOSE_ENERGY_FEATURE)
            .is_none_or(|value| *value > 0.0)
    {
        return false;
    }
    let base = HEADER_FEATURES.saturating_add(teacher.target.saturating_mul(SLOT_FEATURES));
    observation.get(base).is_some_and(|present| *present > 0.5)
        && observation
            .get(base + 1)
            .is_some_and(|reachable| *reachable > 0.5)
        && observation
            .get(base + SLOT_PLANT_ENERGY_FEATURE)
            .is_some_and(|energy| *energy > 0.0)
        && observation
            .get(base + SLOT_NEIGHBOR_PRESENT_FEATURE)
            .is_some_and(|neighbor| *neighbor <= 0.5)
}

fn correction_sample_eligible(
    mode: PolicyCorrectionLabelMode,
    observation: &[f32],
    policy_action: usize,
    teacher_action: usize,
) -> bool {
    match mode {
        PolicyCorrectionLabelMode::FullTeacher => false,
        PolicyCorrectionLabelMode::AdjacentConsumeControl
        | PolicyCorrectionLabelMode::AdjacentConsumeTreatment => {
            adjacent_consume_sample_eligible(observation, policy_action, teacher_action)
        }
        PolicyCorrectionLabelMode::MoveTargetControl
        | PolicyCorrectionLabelMode::MoveTargetTreatment => {
            move_target_sample_eligible(observation, policy_action, teacher_action)
        }
    }
}

/// Replay a successful observation-valid combat policy into the ordinary
/// behavior-cloning format. This is intentionally limited to the report's
/// synthesis seeds: sealed holdout evidence must never become training data.
pub fn generate_observation_policy_demonstrations(
    config: &TrainingConfig,
    source_config_sha256: String,
    scenario: &MicroCombatScenario,
    report: &ObservationPolicySearchReport,
    options: &ObservationPolicyDemonstrationOptions,
) -> Result<(DemonstrationManifest, DemonstrationPayload), String> {
    config.validate()?;
    if !is_sha256(&source_config_sha256) || !is_sha256(&options.source_report_sha256) {
        return Err("observation-policy demonstrations require valid source hashes".into());
    }
    if options.max_samples < report.search_seeds.len() || report.search_seeds.is_empty() {
        return Err(
            "observation-policy demonstrations need at least one sample slot per synthesis seed"
                .into(),
        );
    }
    validate_observation_policy_report(report, scenario, options.minimum_successful_holdout_seeds)?;

    let suite = MicroCombatSuiteConfig {
        schema_version: crate::micro_combat::MICRO_COMBAT_SUITE_SCHEMA_VERSION,
        scenarios: vec![scenario.clone()],
    };
    suite.validate_against(&config.env)?;
    let scenario_hash = suite.semantic_hash()?;
    let effective_config_sha256 =
        sha256(&serde_json::to_vec(&(config, scenario)).map_err(|error| {
            format!("failed to encode effective observation-policy configuration: {error}")
        })?);
    let semantic_ruleset_hash = config.env.rules.semantic_hash().to_string();
    let policy_sha256 = observation_policy_sha256(&report.policy)?;
    let mut samples = Vec::with_capacity(options.max_samples);
    let mut exact_round_trip_samples = 0usize;
    let mut memory_replacement_samples = 0usize;
    let mut action_family_samples = [0usize; PolicyActionFamily::COUNT];
    let mut completed_episodes = 0usize;
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut timeouts = 0usize;
    let samples_per_seed = options.max_samples.div_ceil(report.search_seeds.len());
    let mut compiled_ruleset_hash = None;

    for seed in &report.search_seeds {
        let simulator =
            SingleCellSearchSimulator::new(&config.env, &config.reward, scenario, *seed)?;
        let mut state = simulator.root().clone();
        let mut decisions = 0usize;
        let mut seed_samples = 0usize;
        while !state.done() && decisions < report.config.max_decisions_per_particle {
            let frontier = simulator.frontier(&state, false)?;
            let exact_key = mind_observation_sha256(&frontier.input)?;
            let abstraction = observation_abstraction(
                &frontier.input,
                decisions,
                report.policy.observation_quantization_levels,
                &frontier.legal_choices,
            )?;
            let abstraction_sha256 = observation_abstraction_sha256(&abstraction)?;
            let (rule, _) = select_abstract_rule(
                &report.policy,
                &abstraction_sha256,
                &abstraction,
                report.policy.max_nearest_fallback_l1_distance,
            )
            .ok_or_else(|| {
                format!(
                    "observation policy does not cover synthesis seed {seed} at decision {decisions}"
                )
            })?;
            if rule.decision_depth != decisions {
                return Err("observation-policy rule depth did not replay".into());
            }
            if samples.len() < options.max_samples && seed_samples < samples_per_seed {
                // The oracle's private-memory replacement tracks its table
                // depth. The neural policy learns that sequence through its
                // own recurrent state, so only the physical choice is labeled.
                let decision = decode_policy_choice(rule.choice, &frontier.input);
                let (sample, family) = sample_from_decision(*seed, 0, &frontier.input, &decision)?;
                exact_round_trip_samples += usize::from(sample.exact_round_trip);
                memory_replacement_samples += usize::from(sample.memory_replacement);
                action_family_samples[family.index()] += 1;
                samples.push(sample);
                seed_samples += 1;
            }
            state = simulator.transition_observation_choice(
                &state,
                &exact_key,
                rule.choice,
                rule.next_private_memory.clone(),
            )?;
            decisions += 1;
        }
        let expected = report
            .search_replays
            .iter()
            .find(|replay| replay.seed == *seed)
            .ok_or_else(|| format!("observation-policy report omitted synthesis seed {seed}"))?;
        if !expected.coverage_complete
            || !expected.objective_success
            || expected.final_continuation_sha256 != state.continuation_sha256
            || !state.objective_success(scenario.objective)
        {
            return Err(format!(
                "observation-policy synthesis seed {seed} did not replay its successful evidence"
            ));
        }
        completed_episodes += 1;
        match state.terminal_outcome {
            Some(EpisodeOutcome::Win) => wins += 1,
            Some(EpisodeOutcome::Loss) => losses += 1,
            Some(EpisodeOutcome::Timeout) => timeouts += 1,
            Some(EpisodeOutcome::SafetyAbort) | None => {
                return Err(format!(
                    "observation-policy synthesis seed {seed} did not reach a valid terminal outcome"
                ));
            }
        }
        let compiled = state
            .checkpoint
            .canonical
            .canonical
            .compiled_ruleset_hash()
            .to_string();
        if compiled_ruleset_hash
            .as_ref()
            .is_some_and(|expected| expected != &compiled)
        {
            return Err("observation-policy scenarios compiled inconsistently".into());
        }
        compiled_ruleset_hash = Some(compiled);
    }
    if samples.is_empty() {
        return Err("observation-policy demonstrations produced no decisions".into());
    }

    let payload = DemonstrationPayload {
        schema_version: DEMONSTRATION_SCHEMA_VERSION,
        samples,
    };
    let payload_bytes = rmp_serde::to_vec_named(&payload)
        .map_err(|error| format!("failed to encode observation-policy payload: {error}"))?;
    let manifest = DemonstrationManifest {
        schema_version: DEMONSTRATION_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        mind_abi_sha256: mind_abi_hash(),
        teacher: "observation_policy_oracle".into(),
        collection: DemonstrationCollection::ObservationPolicyOracle {
            source_report_sha256: options.source_report_sha256.clone(),
            policy_sha256,
            scenario: scenario.name.clone(),
            zero_randomness: true,
            coverage_complete: true,
            objective_success: true,
            holdout_seed_count: report.holdout_seeds.len(),
            minimum_successful_holdout_seeds: options.minimum_successful_holdout_seeds,
            holdout_coverage_complete: report.holdout_covered_to_requested_horizon,
            holdout_objective_success: report.holdout_objective_success_count
                == report.holdout_seeds.len(),
        },
        seeds: report.search_seeds.clone(),
        source_config_sha256,
        effective_config_sha256,
        semantic_ruleset_hash,
        compiled_ruleset_hash: compiled_ruleset_hash
            .expect("successful observation-policy replay has a compiled ruleset"),
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

fn validate_observation_policy_report(
    report: &ObservationPolicySearchReport,
    scenario: &MicroCombatScenario,
    minimum_successful_holdout_seeds: usize,
) -> Result<(), String> {
    if report.schema_version != MICRO_COMBAT_OBSERVATION_POLICY_SEARCH_REPORT_SCHEMA_VERSION
        || report.policy.schema_version != MICRO_COMBAT_OBSERVATION_POLICY_SCHEMA_VERSION
        || report.scenario != scenario.name
        || report.objective != scenario.objective
        || report.policy_sha256 != observation_policy_sha256(&report.policy)?
        || report.policy_rule_count != report.policy.rules.len()
        || report.exact_observation_binding_count != report.policy.exact_observation_bindings.len()
        || report.search_replays.len() != report.search_seeds.len()
        || !report.search_covered_to_requested_horizon
        || !report.all_search_objective_success
        || report.search_objective_success_count != report.search_seeds.len()
        || minimum_successful_holdout_seeds == 0
        || report.holdout_seeds.len() < minimum_successful_holdout_seeds
        || !report.holdout_covered_to_requested_horizon
        || report.holdout_objective_success_count != report.holdout_seeds.len()
    {
        return Err("observation-policy report is incompatible or not successful".into());
    }
    let mut synthesis = report.search_seeds.clone();
    synthesis.sort_unstable();
    synthesis.dedup();
    if synthesis.len() != report.search_seeds.len()
        || synthesis
            .iter()
            .any(|seed| report.holdout_seeds.contains(seed))
        || report
            .search_replays
            .iter()
            .any(|replay| !synthesis.contains(&replay.seed))
    {
        return Err("observation-policy report seed partitions are malformed".into());
    }
    Ok(())
}

fn validate_dataset(
    manifest: &DemonstrationManifest,
    payload: &DemonstrationPayload,
) -> Result<(), String> {
    let maintained_teacher = [
        MaintainedMindProfile::Simple,
        MaintainedMindProfile::CollisionAwareForager,
        MaintainedMindProfile::Aggressive,
        MaintainedMindProfile::Defensive,
        MaintainedMindProfile::Explorer,
        MaintainedMindProfile::Colony,
        MaintainedMindProfile::ColonySignalDisabled,
        MaintainedMindProfile::ColonySignalOneQuantum,
        MaintainedMindProfile::ColonySignalSidecar,
        MaintainedMindProfile::ColonySignalCadenced,
    ]
    .iter()
    .any(|profile| profile.as_str() == manifest.teacher);
    let supported_schema = matches!(
        manifest.schema_version,
        11 | 12 | 13 | 14 | 15 | 16 | 17 | DEMONSTRATION_SCHEMA_VERSION
    ) && payload.schema_version == manifest.schema_version;
    if !supported_schema
        || manifest.observation_dim != OBS_DIM
        || manifest.action_count != NUM_ACTIONS
        || manifest.amount_choice_count != NUM_AMOUNT_CHOICES
        || manifest.signal_choice_count != NUM_SIGNAL_CHOICES
        || manifest.signal_strength_choice_count != NUM_SIGNAL_STRENGTH_CHOICES
        || manifest.mind_abi_sha256 != mind_abi_hash()
        || manifest.teacher.is_empty()
        || manifest.teacher.len() > 128
        || !is_sha256(&manifest.source_config_sha256)
        || !is_sha256(&manifest.effective_config_sha256)
        || manifest.samples != payload.samples.len()
        || manifest.payload_file != PAYLOAD_FILE
    {
        return Err("demonstration schema or identity mismatch".into());
    }
    if let DemonstrationCollection::GreedyPolicyCorrection {
        behavior_clone_metadata_sha256,
        behavior_clone_model_sha256,
        policy_disagreement_samples,
        teacher_attack_policy_non_attack_samples,
        projected_teacher_samples,
        exact_policy_agreement_samples,
        policy_agreement_ppm,
        minimum_policy_agreement_ppm,
        agreement_passed,
        policy_action_family_samples,
        action_family_confusion,
        action_kind_disagreement_samples,
        policy_kind_advantage_micrologit_sum,
        policy_kind_advantage_micrologit_max,
        policy_kind_advantage_within_threshold_samples,
        action_family_policy_kind_advantage_micrologit_sums,
        action_family_kind_disagreement_samples,
        label_mode,
        teacher_action_family_samples,
        eligible_correction_samples,
        active_correction_samples,
        zero_randomness,
    } = &manifest.collection
    {
        if !maintained_teacher
            || !is_sha256(behavior_clone_metadata_sha256)
            || !is_sha256(behavior_clone_model_sha256)
            || *policy_disagreement_samples > manifest.samples
            || *teacher_attack_policy_non_attack_samples > *policy_disagreement_samples
            || *projected_teacher_samples > manifest.samples
            || (manifest.schema_version >= 17 && *zero_randomness)
            || (manifest.schema_version < 17 && !*zero_randomness)
            || (manifest.schema_version < 18
                && (*label_mode != PolicyCorrectionLabelMode::FullTeacher
                    || teacher_action_family_samples
                        .iter()
                        .any(|count| *count != 0)
                    || *eligible_correction_samples != 0
                    || *active_correction_samples != 0))
            || (manifest.schema_version >= 18
                && (teacher_action_family_samples.iter().sum::<usize>() != manifest.samples
                    || *eligible_correction_samples > manifest.samples
                    || *active_correction_samples > *eligible_correction_samples
                    || match label_mode {
                        PolicyCorrectionLabelMode::FullTeacher => {
                            *eligible_correction_samples != 0 || *active_correction_samples != 0
                        }
                        PolicyCorrectionLabelMode::AdjacentConsumeControl => {
                            *eligible_correction_samples == 0 || *active_correction_samples != 0
                        }
                        PolicyCorrectionLabelMode::AdjacentConsumeTreatment => {
                            *eligible_correction_samples == 0
                                || *active_correction_samples != *eligible_correction_samples
                        }
                        PolicyCorrectionLabelMode::MoveTargetControl => {
                            *eligible_correction_samples == 0 || *active_correction_samples != 0
                        }
                        PolicyCorrectionLabelMode::MoveTargetTreatment => {
                            *eligible_correction_samples == 0
                                || *active_correction_samples != *eligible_correction_samples
                        }
                    }))
        {
            return Err("policy-correction collection identity is malformed".into());
        }
        if manifest.schema_version >= 14 {
            let expected_exact = manifest.samples - *policy_disagreement_samples;
            let expected_ppm = agreement_ppm(expected_exact, manifest.samples);
            let family_count = PolicyActionFamily::COUNT;
            let expected_teacher_marginals = if manifest.schema_version >= 18 {
                teacher_action_family_samples
            } else {
                &manifest.action_family_samples
            };
            let valid_matrix = action_family_confusion.len() == family_count * family_count
                && action_family_confusion.iter().sum::<usize>() == manifest.samples
                && (0..family_count).all(|teacher| {
                    let start = teacher * family_count;
                    action_family_confusion[start..start + family_count]
                        .iter()
                        .sum::<usize>()
                        == expected_teacher_marginals[teacher]
                })
                && (0..family_count).all(|policy| {
                    (0..family_count)
                        .map(|teacher| action_family_confusion[teacher * family_count + policy])
                        .sum::<usize>()
                        == policy_action_family_samples[policy]
                });
            let attack = PolicyActionFamily::Attack.index();
            let missed_teacher_attacks = (0..family_count)
                .filter(|policy| *policy != attack)
                .map(|policy| action_family_confusion[attack * family_count + policy])
                .sum::<usize>();
            if *exact_policy_agreement_samples != expected_exact
                || *policy_agreement_ppm != expected_ppm
                || *minimum_policy_agreement_ppm > 1_000_000
                || *agreement_passed
                    != agreement_meets(
                        expected_exact,
                        manifest.samples,
                        *minimum_policy_agreement_ppm,
                    )
                || policy_action_family_samples.iter().sum::<usize>() != manifest.samples
                || !valid_matrix
                || *teacher_attack_policy_non_attack_samples != missed_teacher_attacks
            {
                return Err("policy-correction agreement diagnostics are inconsistent".into());
            }
        }
        if manifest.schema_version >= 15 {
            let family_count = PolicyActionFamily::COUNT;
            let mut payload_policy_margins = [0usize; PolicyActionFamily::COUNT];
            let mut payload_teacher_margins = [0usize; PolicyActionFamily::COUNT];
            let mut payload_confusion = vec![0usize; family_count * family_count];
            let mut payload_pair_advantages = vec![0u64; family_count * family_count];
            let mut payload_pair_kind_disagreements = vec![0usize; family_count * family_count];
            let mut payload_kind_disagreements = 0usize;
            let mut payload_advantage_sum = 0u64;
            let mut payload_advantage_max = 0u32;
            let mut payload_within = [0usize; POLICY_KIND_MARGIN_THRESHOLDS_MICROLOGITS.len()];
            let mut payload_eligible = 0usize;
            let mut payload_active = 0usize;
            for sample in &payload.samples {
                let policy_action = sample
                    .correction_policy_action
                    .map(usize::from)
                    .ok_or("schema-15 policy correction omitted its policy action")?;
                let advantage = sample
                    .correction_policy_kind_advantage_micrologits
                    .ok_or("schema-15 policy correction omitted its action-kind margin")?;
                if policy_action >= NUM_ACTIONS
                    || sample.action_mask.len() != NUM_ACTIONS
                    || !sample.action_mask[policy_action]
                {
                    return Err(
                        "policy-correction payload contains an illegal policy action".into(),
                    );
                }
                let teacher_action = if manifest.schema_version >= 18 {
                    sample
                        .correction_teacher_action
                        .map(usize::from)
                        .ok_or("schema-18 policy correction omitted its teacher action")?
                } else {
                    usize::from(sample.action)
                };
                if teacher_action >= NUM_ACTIONS || !sample.action_mask[teacher_action] {
                    return Err(
                        "policy-correction payload contains an illegal teacher action".into(),
                    );
                }
                let teacher_family = policy_action_family(teacher_action)
                    .ok_or("policy-correction teacher action has no family")?;
                let policy_family = policy_action_family(policy_action)
                    .ok_or("policy-correction policy action has no family")?;
                let teacher_kind = decompose_policy_action(teacher_action)
                    .ok_or("policy-correction teacher action has no kind")?
                    .kind;
                let policy_kind = decompose_policy_action(policy_action)
                    .ok_or("policy-correction policy action has no kind")?
                    .kind;
                let pair = teacher_family.index() * family_count + policy_family.index();
                payload_policy_margins[policy_family.index()] += 1;
                payload_teacher_margins[teacher_family.index()] += 1;
                payload_confusion[pair] += 1;
                if manifest.schema_version >= 18 {
                    let expected_eligible = correction_sample_eligible(
                        *label_mode,
                        &sample.observation,
                        policy_action,
                        teacher_action,
                    );
                    if sample.correction_eligible != expected_eligible
                        || sample.active_correction && !sample.correction_eligible
                        || match label_mode {
                            PolicyCorrectionLabelMode::FullTeacher => {
                                sample.correction_eligible
                                    || sample.active_correction
                                    || usize::from(sample.action) != teacher_action
                            }
                            PolicyCorrectionLabelMode::AdjacentConsumeControl => {
                                sample.active_correction
                                    || usize::from(sample.action) != policy_action
                            }
                            PolicyCorrectionLabelMode::AdjacentConsumeTreatment
                            | PolicyCorrectionLabelMode::MoveTargetTreatment => {
                                sample.active_correction != sample.correction_eligible
                                    || usize::from(sample.action)
                                        != if sample.active_correction {
                                            teacher_action
                                        } else {
                                            policy_action
                                        }
                            }
                            PolicyCorrectionLabelMode::MoveTargetControl => {
                                sample.active_correction
                                    || usize::from(sample.action) != policy_action
                            }
                        }
                    {
                        return Err("targeted policy-correction label is inconsistent".into());
                    }
                    payload_eligible += usize::from(sample.correction_eligible);
                    payload_active += usize::from(sample.active_correction);
                } else if sample.correction_teacher_action.is_some()
                    || sample.correction_eligible
                    || sample.active_correction
                {
                    return Err("legacy policy correction contains schema-18 telemetry".into());
                }
                if teacher_kind != policy_kind {
                    payload_kind_disagreements += 1;
                    payload_pair_kind_disagreements[pair] += 1;
                    payload_advantage_sum = payload_advantage_sum
                        .checked_add(u64::from(advantage))
                        .ok_or("policy-correction payload margin sum overflowed")?;
                    payload_advantage_max = payload_advantage_max.max(advantage);
                    payload_pair_advantages[pair] = payload_pair_advantages[pair]
                        .checked_add(u64::from(advantage))
                        .ok_or("policy-correction payload pair margin sum overflowed")?;
                    for (count, threshold) in payload_within
                        .iter_mut()
                        .zip(POLICY_KIND_MARGIN_THRESHOLDS_MICROLOGITS)
                    {
                        *count += usize::from(advantage <= threshold);
                    }
                } else if advantage != 0 {
                    return Err(
                        "policy-correction same-kind choice has a nonzero kind margin".into(),
                    );
                }
            }
            if payload_policy_margins != *policy_action_family_samples
                || manifest.schema_version >= 18
                    && (payload_teacher_margins != *teacher_action_family_samples
                        || payload_eligible != *eligible_correction_samples
                        || payload_active != *active_correction_samples)
                || payload_confusion != *action_family_confusion
                || payload_kind_disagreements != *action_kind_disagreement_samples
                || payload_advantage_sum != *policy_kind_advantage_micrologit_sum
                || payload_advantage_max != *policy_kind_advantage_micrologit_max
                || payload_within != *policy_kind_advantage_within_threshold_samples
                || payload_pair_advantages != *action_family_policy_kind_advantage_micrologit_sums
                || payload_pair_kind_disagreements != *action_family_kind_disagreement_samples
            {
                return Err(
                    "policy-correction action-kind margin diagnostics are inconsistent".into(),
                );
            }
        }
    }
    if matches!(manifest.collection, DemonstrationCollection::TeacherRollout) && !maintained_teacher
    {
        return Err("teacher-rollout collection identity is malformed".into());
    }
    if let DemonstrationCollection::ObservationPolicyOracle {
        source_report_sha256,
        policy_sha256,
        scenario,
        zero_randomness,
        coverage_complete,
        objective_success,
        holdout_seed_count,
        minimum_successful_holdout_seeds,
        holdout_coverage_complete,
        holdout_objective_success,
    } = &manifest.collection
    {
        if manifest.schema_version < 12
            || manifest.teacher != "observation_policy_oracle"
            || !is_sha256(source_report_sha256)
            || !is_sha256(policy_sha256)
            || scenario.is_empty()
            || !zero_randomness
            || !coverage_complete
            || !objective_success
            || manifest.completed_episodes != manifest.seeds.len()
            || (manifest.schema_version >= 13
                && (*minimum_successful_holdout_seeds == 0
                    || *holdout_seed_count < *minimum_successful_holdout_seeds
                    || !holdout_coverage_complete
                    || !holdout_objective_success))
        {
            return Err("observation-policy collection identity is malformed".into());
        }
    }
    let counterfactual_correction = match &manifest.collection {
        DemonstrationCollection::CounterfactualEffortCorrection {
            source_counterfactual_sha256,
            source_value_sha256,
            behavior_clone_model_sha256,
            perspective,
            horizon_quanta,
            recurrent_size,
            minimum_effort,
            active_correction: _,
        } => {
            if manifest.schema_version != DEMONSTRATION_SCHEMA_VERSION
                || manifest.teacher != "counterfactual_effort"
                || !is_sha256(source_counterfactual_sha256)
                || !is_sha256(source_value_sha256)
                || !is_sha256(behavior_clone_model_sha256)
                || perspective.is_empty()
                || *horizon_quanta == 0
                || *recurrent_size == 0
                || usize::from(*minimum_effort) >= crate::action::NUM_POLICY_EFFORTS
                || manifest.completed_episodes != 0
                || manifest.wins != 0
                || manifest.losses != 0
                || manifest.timeouts != 0
            {
                return Err("counterfactual effort-correction identity is malformed".into());
            }
            Some((*recurrent_size, usize::from(*minimum_effort)))
        }
        _ => None,
    };
    let mut isolated_keys = std::collections::HashSet::new();
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
        if let Some((recurrent_size, minimum_effort)) = counterfactual_correction {
            let choice = decompose_policy_action(action)
                .ok_or("counterfactual correction action has no decomposition")?;
            if choice.kind != crate::action::PolicyActionKind::Move.index()
                || choice.effort != minimum_effort
                    && matches!(
                        manifest.collection,
                        DemonstrationCollection::CounterfactualEffortCorrection {
                            active_correction: true,
                            ..
                        }
                    )
                || sample.initial_policy_memory.len() != recurrent_size
                || sample
                    .initial_policy_memory
                    .iter()
                    .any(|value| !value.is_finite() || !(-1.0..=1.0).contains(value))
                || !isolated_keys.insert((sample.source_seed, sample.source_cell))
            {
                return Err("counterfactual effort-correction sample is malformed".into());
            }
        } else if !sample.initial_policy_memory.is_empty() {
            return Err("ordinary demonstration contains isolated policy memory".into());
        }
        if manifest.schema_version >= 15
            && !matches!(
                manifest.collection,
                DemonstrationCollection::GreedyPolicyCorrection { .. }
            )
            && (sample.correction_policy_action.is_some()
                || sample
                    .correction_policy_kind_advantage_micrologits
                    .is_some()
                || sample.correction_teacher_action.is_some()
                || sample.correction_eligible
                || sample.active_correction)
        {
            return Err("non-correction demonstration contains correction telemetry".into());
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

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn agreement_ppm(numerator: usize, denominator: usize) -> u32 {
    if denominator == 0 {
        0
    } else {
        u32::try_from((numerator as u128 * 1_000_000_u128 / denominator as u128).min(1_000_000))
            .expect("clamped agreement ppm fits u32")
    }
}

fn agreement_meets(numerator: usize, denominator: usize, minimum_ppm: u32) -> bool {
    denominator > 0
        && numerator as u128 * 1_000_000_u128 >= denominator as u128 * u128::from(minimum_ppm)
}

fn quantize_policy_kind_advantage(
    selected_policy_logit: f32,
    teacher_logit: f32,
) -> Result<u32, String> {
    if !selected_policy_logit.is_finite() || !teacher_logit.is_finite() {
        return Err("policy correction encountered a non-finite action-kind logit".into());
    }
    let advantage = f64::from(selected_policy_logit - teacher_logit);
    // The selected legal kind is maximal. Permit only microscopic negative
    // roundoff before quantizing the nonnegative crossover distance.
    if advantage < -1.0e-6 {
        return Err("greedy policy kind did not maximize its routed logits".into());
    }
    Ok((advantage.max(0.0) * 1_000_000.0)
        .round()
        .min(f64::from(u32::MAX)) as u32)
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
    if !matches!(
        manifest.schema_version,
        11 | 12 | 13 | 14 | 15 | 16 | DEMONSTRATION_SCHEMA_VERSION
    ) || manifest.payload_file != PAYLOAD_FILE
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
    use crate::micro_combat::MicroCombatObjective;
    use crate::micro_combat_observation_policy::{
        search_observation_policy, ObservationPolicySearchConfig,
    };
    use crate::observation::{OBS_RANDOMNESS_FEATURE_END, OBS_RANDOMNESS_FEATURE_START};
    use burn::backend::{Autodiff, NdArray};

    type TestBackend = Autodiff<NdArray<f32>>;

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
    fn move_target_correction_changes_only_same_effort_moves_to_visible_food() {
        use crate::action::{compose_policy_action, HierarchicalActionChoice};

        let teacher_target = 3;
        let policy_target = 5;
        let effort = 1;
        let teacher_action = compose_policy_action(HierarchicalActionChoice {
            kind: PolicyActionKind::Move.index(),
            target: teacher_target,
            effort,
        })
        .unwrap();
        let policy_action = compose_policy_action(HierarchicalActionChoice {
            kind: PolicyActionKind::Move.index(),
            target: policy_target,
            effort,
        })
        .unwrap();
        let mut observation = vec![0.0; OBS_DIM];
        let teacher_base = HEADER_FEATURES + teacher_target * SLOT_FEATURES;
        observation[teacher_base] = 1.0;
        observation[teacher_base + 1] = 1.0;
        observation[teacher_base + SLOT_PLANT_ENERGY_FEATURE] = 0.5;

        assert!(move_target_sample_eligible(
            &observation,
            policy_action,
            teacher_action
        ));
        assert!(correction_sample_eligible(
            PolicyCorrectionLabelMode::MoveTargetTreatment,
            &observation,
            policy_action,
            teacher_action,
        ));

        let other_effort = compose_policy_action(HierarchicalActionChoice {
            kind: PolicyActionKind::Move.index(),
            target: teacher_target,
            effort: 2,
        })
        .unwrap();
        assert!(!move_target_sample_eligible(
            &observation,
            policy_action,
            other_effort
        ));

        observation[teacher_base + SLOT_NEIGHBOR_PRESENT_FEATURE] = 1.0;
        assert!(!move_target_sample_eligible(
            &observation,
            policy_action,
            teacher_action
        ));
        observation[teacher_base + SLOT_NEIGHBOR_PRESENT_FEATURE] = 0.0;
        observation[CURRENT_TILE_PLANT_CAPACITY_FEATURE] = 1.0;
        assert!(!move_target_sample_eligible(
            &observation,
            policy_action,
            teacher_action
        ));
    }

    #[test]
    fn datasets_are_reproducible_hash_bound_and_report_colony_lossiness() {
        let options = DemonstrationOptions {
            teacher: MaintainedMindProfile::Colony,
            seeds: vec![11, 12],
            max_samples: 16,
        };
        let source_hash = sha256(b"source config");
        let left = generate_demonstrations(&small_config(), source_hash.clone(), &options).unwrap();
        let right = generate_demonstrations(&small_config(), source_hash, &options).unwrap();
        assert_eq!(left, right);
        let mut alternate_layout = small_config();
        alternate_layout.env.starting_cell_layout = blob_engine::engine::StartingCellLayout::Ring;
        let alternate = generate_demonstrations(
            &alternate_layout,
            left.0.source_config_sha256.clone(),
            &options,
        )
        .unwrap();
        assert_ne!(
            left.0.effective_config_sha256,
            alternate.0.effective_config_sha256
        );
        assert_ne!(left.0.scenario_hash, alternate.0.scenario_hash);
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

        // Schema 11 is an older maintained corpus format. Its serialized
        // teacher already decodes into the current string identity, so active
        // large datasets remain reusable. Oracle provenance began at schema
        // 12 and schema 13 adds sealed-holdout admission evidence.
        let mut legacy_manifest = left.0.clone();
        let mut legacy_payload = left.1.clone();
        legacy_manifest.schema_version = 11;
        legacy_payload.schema_version = 11;
        let legacy_payload_bytes = rmp_serde::to_vec_named(&legacy_payload).unwrap();
        legacy_manifest.payload_sha256 = sha256(&legacy_payload_bytes);
        validate_dataset(&legacy_manifest, &legacy_payload).unwrap();
        let legacy_root = tempfile::tempdir().unwrap();
        fs::write(
            legacy_root.path().join(MANIFEST_FILE),
            serde_json::to_vec_pretty(&legacy_manifest).unwrap(),
        )
        .unwrap();
        fs::write(legacy_root.path().join(PAYLOAD_FILE), legacy_payload_bytes).unwrap();
        assert_eq!(
            load_demonstrations(legacy_root.path())
                .unwrap()
                .manifest
                .schema_version,
            11
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
            generate_demonstrations(&config, sha256(b"contact config"), &options).unwrap();
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

    #[test]
    fn policy_corrections_are_reproducible_and_bind_the_rollout_policy() {
        let mut config = small_config();
        config.env.world_size = 8;
        config.env.cells_per_team = 1;
        config.env.starting_cell_layout = blob_engine::engine::StartingCellLayout::PairedContact;
        config.env.initial_energy = 100;
        config.env.num_scattered_energy = 0;
        config.env.num_plants = 0;
        config.env.opponent = crate::config::OpponentProfile::Defensive;
        let device = Default::default();
        TestBackend::seed(&device, 73);
        let model = crate::model::PolicyValueNetConfig {
            hidden1: config.model.hidden1,
            hidden2: config.model.hidden2,
            recurrent_size: config.model.recurrent_size,
        }
        .init::<TestBackend>(&device);
        let options = PolicyCorrectionOptions {
            teacher: MaintainedMindProfile::Aggressive,
            seeds: vec![31, 32],
            max_samples: 16,
            behavior_clone_metadata_sha256: "a".repeat(64),
            behavior_clone_model_sha256: "b".repeat(64),
            minimum_policy_agreement_rate: 0.5,
            label_mode: PolicyCorrectionLabelMode::FullTeacher,
        };
        let left = generate_policy_correction_demonstrations(
            &config,
            sha256(b"correction config"),
            &options,
            &model,
            &device,
        )
        .unwrap();
        let right = generate_policy_correction_demonstrations(
            &config,
            sha256(b"correction config"),
            &options,
            &model,
            &device,
        )
        .unwrap();
        assert_eq!(left, right);
        assert!((1..=16).contains(&left.0.samples));
        assert!(left.0.action_family_samples[PolicyActionFamily::Attack.index()] > 0);
        let DemonstrationCollection::GreedyPolicyCorrection {
            behavior_clone_metadata_sha256,
            behavior_clone_model_sha256,
            policy_disagreement_samples,
            teacher_attack_policy_non_attack_samples,
            projected_teacher_samples,
            exact_policy_agreement_samples,
            policy_agreement_ppm,
            minimum_policy_agreement_ppm,
            agreement_passed,
            policy_action_family_samples,
            action_family_confusion,
            action_kind_disagreement_samples,
            policy_kind_advantage_micrologit_sum,
            policy_kind_advantage_micrologit_max,
            policy_kind_advantage_within_threshold_samples,
            action_family_policy_kind_advantage_micrologit_sums,
            action_family_kind_disagreement_samples,
            zero_randomness,
            ..
        } = &left.0.collection
        else {
            panic!("expected policy-correction identity")
        };
        assert_eq!(behavior_clone_metadata_sha256, &"a".repeat(64));
        assert_eq!(behavior_clone_model_sha256, &"b".repeat(64));
        assert!(*policy_disagreement_samples <= left.0.samples);
        assert!(*teacher_attack_policy_non_attack_samples <= *policy_disagreement_samples);
        assert!(*projected_teacher_samples <= left.0.samples);
        assert_eq!(
            *exact_policy_agreement_samples + *policy_disagreement_samples,
            left.0.samples
        );
        assert_eq!(
            *policy_agreement_ppm,
            agreement_ppm(*exact_policy_agreement_samples, left.0.samples)
        );
        assert_eq!(*minimum_policy_agreement_ppm, 500_000);
        assert_eq!(
            *agreement_passed,
            agreement_meets(*exact_policy_agreement_samples, left.0.samples, 500_000)
        );
        assert_eq!(
            policy_action_family_samples.iter().sum::<usize>(),
            left.0.samples
        );
        assert_eq!(
            action_family_confusion.iter().sum::<usize>(),
            left.0.samples
        );
        assert_eq!(
            *action_kind_disagreement_samples,
            action_family_kind_disagreement_samples
                .iter()
                .sum::<usize>()
        );
        assert!(policy_kind_advantage_within_threshold_samples
            .windows(2)
            .all(|pair| pair[0] <= pair[1]));
        assert!(policy_kind_advantage_within_threshold_samples
            .iter()
            .all(|count| count <= action_kind_disagreement_samples));
        assert_eq!(
            action_family_policy_kind_advantage_micrologit_sums
                .iter()
                .sum::<u64>(),
            *policy_kind_advantage_micrologit_sum
        );
        assert!(
            u64::from(*policy_kind_advantage_micrologit_max)
                <= *policy_kind_advantage_micrologit_sum
        );
        assert!(left.1.samples.iter().all(|sample| {
            sample.correction_policy_action.is_some()
                && sample
                    .correction_policy_kind_advantage_micrologits
                    .is_some()
        }));
        assert!(!*zero_randomness);
        assert!(left.1.samples.iter().all(|sample| sample.observation
            [OBS_RANDOMNESS_FEATURE_START..OBS_RANDOMNESS_FEATURE_END]
            .iter()
            .any(|value| *value != 0.0)));
        validate_dataset(&left.0, &left.1).unwrap();

        let mut tampered = left.0.clone();
        let DemonstrationCollection::GreedyPolicyCorrection {
            behavior_clone_model_sha256,
            ..
        } = &mut tampered.collection
        else {
            unreachable!()
        };
        *behavior_clone_model_sha256 = "not-a-hash".into();
        assert!(validate_dataset(&tampered, &left.1).is_err());

        let mut tampered_diagnostics = left.0.clone();
        let DemonstrationCollection::GreedyPolicyCorrection {
            action_family_confusion,
            agreement_passed,
            ..
        } = &mut tampered_diagnostics.collection
        else {
            unreachable!()
        };
        action_family_confusion[0] = action_family_confusion[0].saturating_add(1);
        *agreement_passed = !*agreement_passed;
        assert!(validate_dataset(&tampered_diagnostics, &left.1).is_err());

        let mut tampered_payload = left.1.clone();
        tampered_payload.samples[0].correction_policy_kind_advantage_micrologits = Some(u32::MAX);
        assert!(validate_dataset(&left.0, &tampered_payload).is_err());
    }

    #[test]
    fn observation_policy_demonstrations_replay_success_and_seal_holdout() {
        let config = TrainingConfig::default();
        let scenario = MicroCombatScenario {
            name: "oracle-demonstration".into(),
            objective: MicroCombatObjective::Survival,
            world_size: 7,
            training_cells: 1,
            opponent_cells: 1,
            training_initial_energy: 100,
            opponent_initial_energy: 100,
            starting_layout: blob_engine::engine::StartingCellLayout::PairedContact,
            opponent: crate::config::OpponentProfile::Pursuer,
            sim_time_limit_quanta: 1,
        };
        let report = search_observation_policy(
            &config.env,
            &config.reward,
            &scenario,
            &[301],
            &[401],
            ObservationPolicySearchConfig {
                beam_width: 2,
                max_decisions_per_particle: 1,
                max_expansions: 256,
                ..ObservationPolicySearchConfig::default()
            },
        )
        .unwrap();
        assert!(report.all_search_objective_success);

        let insufficient_holdout = generate_observation_policy_demonstrations(
            &config,
            sha256(b"oracle source config"),
            &scenario,
            &report,
            &ObservationPolicyDemonstrationOptions {
                source_report_sha256: "c".repeat(64),
                max_samples: 8,
                minimum_successful_holdout_seeds: 2,
            },
        )
        .unwrap_err();
        assert!(insufficient_holdout.contains("incompatible or not successful"));

        let (manifest, payload) = generate_observation_policy_demonstrations(
            &config,
            sha256(b"oracle source config"),
            &scenario,
            &report,
            &ObservationPolicyDemonstrationOptions {
                source_report_sha256: "c".repeat(64),
                max_samples: 8,
                minimum_successful_holdout_seeds: 1,
            },
        )
        .unwrap();
        assert_eq!(manifest.seeds, vec![301]);
        assert!(!manifest.seeds.contains(&401));
        assert_eq!(manifest.completed_episodes, 1);
        assert_eq!(manifest.samples, 1);
        assert_eq!(manifest.exact_round_trip_samples, manifest.samples);
        assert_eq!(manifest.memory_replacement_samples, 0);
        assert_eq!(payload.samples[0].source_seed, 301);
        let DemonstrationCollection::ObservationPolicyOracle {
            source_report_sha256,
            policy_sha256,
            scenario: bound_scenario,
            zero_randomness,
            coverage_complete,
            objective_success,
            holdout_seed_count,
            minimum_successful_holdout_seeds,
            holdout_coverage_complete,
            holdout_objective_success,
        } = &manifest.collection
        else {
            panic!("expected observation-policy collection identity")
        };
        assert_eq!(source_report_sha256, &"c".repeat(64));
        assert_eq!(policy_sha256, &report.policy_sha256);
        assert_eq!(bound_scenario, &scenario.name);
        assert!(*zero_randomness && *coverage_complete && *objective_success);
        assert_eq!(*holdout_seed_count, 1);
        assert_eq!(*minimum_successful_holdout_seeds, 1);
        assert!(*holdout_coverage_complete && *holdout_objective_success);
        validate_dataset(&manifest, &payload).unwrap();

        let mut tampered = manifest.clone();
        let DemonstrationCollection::ObservationPolicyOracle {
            source_report_sha256,
            ..
        } = &mut tampered.collection
        else {
            unreachable!()
        };
        *source_report_sha256 = "not-a-hash".into();
        assert!(validate_dataset(&tampered, &payload).is_err());
    }
}
