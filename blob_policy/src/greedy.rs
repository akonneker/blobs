//! Canonical greedy tie-breaking and action-mask decoding.
use crate::action::{
    compose_policy_action, policy_action_kind_mask, policy_effort_mask, policy_target_mask,
    HierarchicalActionChoice, NUM_POLICY_EFFORTS, NUM_POLICY_TARGETS,
};
use crate::observation::Observation;

pub fn greedy_policy_action(
    kind_logits: &[f32],
    target_logits: &[f32],
    effort_logits: &[f32],
    observation: &Observation,
) -> usize {
    let kind = greedy_masked(
        kind_logits,
        &policy_action_kind_mask(&observation.action_mask),
    );
    let target_start = kind * NUM_POLICY_TARGETS;
    let target = greedy_masked(
        &target_logits[target_start..target_start + NUM_POLICY_TARGETS],
        &policy_target_mask(&observation.action_mask, kind),
    );
    let effort_start = kind * NUM_POLICY_EFFORTS;
    let effort = greedy_masked(
        &effort_logits[effort_start..effort_start + NUM_POLICY_EFFORTS],
        &policy_effort_mask(&observation.action_mask, kind, target),
    );
    compose_policy_action(HierarchicalActionChoice {
        kind,
        target,
        effort,
    })
    .expect("projected hierarchical masks must compose to a flat policy action")
}

pub fn greedy_amount(logits: &[f32], observation: &Observation, action: usize) -> usize {
    greedy_masked(logits, &observation.amount_mask(action))
}

pub fn greedy_signal(
    logits: &[f32],
    observation: &Observation,
    action: usize,
    amount: usize,
) -> usize {
    greedy_masked(logits, &observation.signal_mask(action, amount))
}

pub fn greedy_signal_strength(
    logits: &[f32],
    observation: &Observation,
    action: usize,
    amount: usize,
    signal: usize,
) -> usize {
    greedy_masked(
        logits,
        &observation.signal_strength_mask(action, amount, signal),
    )
}

fn greedy_masked<const N: usize>(logits: &[f32], mask: &[bool; N]) -> usize {
    logits
        .iter()
        .copied()
        .zip(mask)
        .enumerate()
        .filter(|(_, (logit, allowed))| **allowed && logit.is_finite())
        .max_by(|left, right| {
            left.1
                 .0
                .total_cmp(&right.1 .0)
                .then_with(|| right.0.cmp(&left.0))
        })
        .map_or(0, |(index, _)| index)
}
