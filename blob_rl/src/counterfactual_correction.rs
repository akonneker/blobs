//! Target-free effort corrections from immutable counterfactual evidence.
//!
//! Hidden branch futures may establish that minimum-effort Move is robustly
//! preferred, but they never select a direction. Every synthesized label keeps
//! the frozen policy's target and changes only its effort tier.

use std::path::{Path, PathBuf};

use crate::action::{
    compose_policy_action, decompose_policy_action, HierarchicalActionChoice, PolicyActionFamily,
    PolicyActionKind, NUM_ACTIONS, NUM_AMOUNT_CHOICES, NUM_SIGNAL_CHOICES,
    NUM_SIGNAL_STRENGTH_CHOICES,
};
use crate::counterfactual_branch::CounterfactualBranchArtifact;
use crate::counterfactual_value::{ActionArchetype, CounterfactualValueArtifact, ValuePerspective};
use crate::demonstration::{
    publish_demonstrations, DemonstrationCollection, DemonstrationManifest, DemonstrationPayload,
    DemonstrationSample, DEMONSTRATION_SCHEMA_VERSION,
};
use crate::model::decode_policy_memory;
use crate::observation::OBS_DIM;
use crate::sweep::sha256;
use crate::viability::mind_abi_hash;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CounterfactualEffortCorrectionOptions {
    pub perspective: ValuePerspective,
    pub horizon_quanta: u64,
    /// False creates an active control with the frozen policy's original
    /// effort. True changes only that effort to the robust minimum.
    pub active_correction: bool,
}

fn corrected_action(policy_action: usize, active: bool) -> Result<usize, String> {
    let policy = decompose_policy_action(policy_action)
        .ok_or("counterfactual correction policy action is outside the catalog")?;
    if policy.kind != PolicyActionKind::Move.index() {
        return Err("counterfactual correction source policy did not choose Move".into());
    }
    compose_policy_action(HierarchicalActionChoice {
        kind: policy.kind,
        target: policy.target,
        effort: if active { 0 } else { policy.effort },
    })
    .ok_or_else(|| "counterfactual correction action could not be composed".into())
}

fn perspective_name(perspective: ValuePerspective) -> &'static str {
    match perspective {
        ValuePerspective::Cell => "cell",
        ValuePerspective::Colony => "colony",
        ValuePerspective::Conservative => "conservative",
    }
}

pub fn build_counterfactual_effort_corrections(
    source: &CounterfactualBranchArtifact,
    value: &CounterfactualValueArtifact,
    options: CounterfactualEffortCorrectionOptions,
) -> Result<(DemonstrationManifest, DemonstrationPayload), String> {
    source.validate()?;
    value.validate_against(source)?;
    if options.horizon_quanta == 0 {
        return Err("counterfactual correction horizon must be positive".into());
    }
    let evaluation = value
        .report
        .evaluations
        .iter()
        .find(|evaluation| {
            evaluation.perspective == options.perspective
                && evaluation.requested_elapsed_quanta == options.horizon_quanta
        })
        .ok_or("counterfactual correction requested an absent value evaluation")?;
    let required_archetype = [ActionArchetype {
        kind: PolicyActionKind::Move.index(),
        effort: 0,
    }];
    if evaluation.states.len() != source.report.states.len()
        || evaluation
            .states
            .iter()
            .any(|state| state.pareto_archetypes.as_slice() != required_archetype)
    {
        return Err(
            "counterfactual correction requires a unanimous minimum-effort Move archetype".into(),
        );
    }

    let recurrent_size = source.config.model.recurrent_size;
    let mut samples = Vec::with_capacity(source.report.states.len());
    let mut changed = 0usize;
    for state in &source.report.states {
        if !evaluation.states.iter().any(|verdict| {
            verdict.seed == state.seed && verdict.state_ordinal == state.state_ordinal
        }) {
            return Err("counterfactual correction value states do not match the source".into());
        }
        let evidence = state
            .mind_evidence
            .as_ref()
            .ok_or("counterfactual source predates retained Mind evidence")?;
        let observation = evidence.observation()?;
        let action = corrected_action(state.policy_choice.action, options.active_correction)?;
        changed += usize::from(action != state.policy_choice.action);
        if !observation.action_mask[action] {
            return Err("counterfactual correction produced an illegal Move effort".into());
        }
        let amount = 0usize;
        let signal = 0usize;
        let signal_strength = 0usize;
        let amount_mask = observation.amount_mask(action);
        let signal_mask = observation.signal_mask(action, amount);
        let signal_strength_mask = observation.signal_strength_mask(action, amount, signal);
        samples.push(DemonstrationSample {
            source_seed: state.seed,
            // Counterfactual states are isolated recurrent decisions, not a
            // fabricated contiguous trajectory for the original cell.
            source_cell: u64::try_from(state.state_ordinal)
                .map_err(|_| "counterfactual state ordinal exceeds u64")?,
            observation: observation.to_vec(),
            action_mask: observation.action_mask.to_vec(),
            action: u16::try_from(action).map_err(|_| "policy action catalog exceeds u16")?,
            amount_mask: amount_mask.to_vec(),
            amount: u8::try_from(amount).expect("zero amount fits u8"),
            signal_mask: signal_mask.to_vec(),
            signal: u8::try_from(signal).expect("zero signal fits u8"),
            signal_strength_mask: signal_strength_mask.to_vec(),
            signal_strength: u8::try_from(signal_strength).expect("zero signal strength fits u8"),
            exact_round_trip: false,
            memory_replacement: false,
            correction_policy_action: None,
            correction_policy_kind_advantage_micrologits: None,
            correction_teacher_action: None,
            correction_eligible: false,
            active_correction: false,
            initial_policy_memory: decode_policy_memory(&evidence.private_memory, recurrent_size),
        });
    }
    if samples.is_empty() || options.active_correction && changed == 0 {
        return Err("counterfactual correction contains no effective labels".into());
    }

    let payload = DemonstrationPayload {
        schema_version: DEMONSTRATION_SCHEMA_VERSION,
        samples,
    };
    let payload_bytes = rmp_serde::to_vec_named(&payload)
        .map_err(|error| format!("failed to encode counterfactual corrections: {error}"))?;
    let mut seeds = payload
        .samples
        .iter()
        .map(|sample| sample.source_seed)
        .collect::<Vec<_>>();
    seeds.sort_unstable();
    seeds.dedup();
    let mut action_family_samples = [0usize; PolicyActionFamily::COUNT];
    action_family_samples[PolicyActionFamily::Move.index()] = payload.samples.len();
    let effective_config_sha256 = sha256(
        &serde_json::to_vec(&source.config)
            .map_err(|error| format!("failed to encode correction config: {error}"))?,
    );
    let manifest = DemonstrationManifest {
        schema_version: DEMONSTRATION_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").into(),
        mind_abi_sha256: mind_abi_hash(),
        teacher: "counterfactual_effort".into(),
        collection: DemonstrationCollection::CounterfactualEffortCorrection {
            source_counterfactual_sha256: source.artifact_hash.clone(),
            source_value_sha256: value.artifact_hash.clone(),
            behavior_clone_model_sha256: source.behavior_clone_model_sha256.clone(),
            perspective: perspective_name(options.perspective).into(),
            horizon_quanta: options.horizon_quanta,
            recurrent_size,
            minimum_effort: 0,
            active_correction: options.active_correction,
        },
        seeds,
        source_config_sha256: source.source_config_sha256.clone(),
        effective_config_sha256,
        semantic_ruleset_hash: source.config.env.rules.semantic_hash().to_string(),
        compiled_ruleset_hash: source.report.compiled_ruleset_hash.clone(),
        scenario_hash: source.report.scenario_hash.clone(),
        observation_dim: OBS_DIM,
        action_count: NUM_ACTIONS,
        amount_choice_count: NUM_AMOUNT_CHOICES,
        signal_choice_count: NUM_SIGNAL_CHOICES,
        signal_strength_choice_count: NUM_SIGNAL_STRENGTH_CHOICES,
        samples: payload.samples.len(),
        exact_round_trip_samples: 0,
        memory_replacement_samples: 0,
        action_family_samples,
        completed_episodes: 0,
        wins: 0,
        losses: 0,
        timeouts: 0,
        payload_file: "samples.mpk".into(),
        payload_sha256: sha256(&payload_bytes),
    };
    Ok((manifest, payload))
}

pub fn publish_counterfactual_effort_corrections(
    output: &Path,
    source: &CounterfactualBranchArtifact,
    value: &CounterfactualValueArtifact,
    options: CounterfactualEffortCorrectionOptions,
) -> Result<PathBuf, String> {
    let (manifest, payload) = build_counterfactual_effort_corrections(source, value, options)?;
    publish_demonstrations(output, &manifest, &payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_free_label_changes_only_move_effort() {
        let original = compose_policy_action(HierarchicalActionChoice {
            kind: PolicyActionKind::Move.index(),
            target: 6,
            effort: 2,
        })
        .unwrap();
        let corrected = corrected_action(original, true).unwrap();
        let original = decompose_policy_action(original).unwrap();
        let corrected = decompose_policy_action(corrected).unwrap();
        assert_eq!(corrected.kind, original.kind);
        assert_eq!(corrected.target, original.target);
        assert_eq!(corrected.effort, 0);
        assert_eq!(
            corrected_action(original.kind, true).unwrap_err(),
            "counterfactual correction source policy did not choose Move"
        );
    }
}
