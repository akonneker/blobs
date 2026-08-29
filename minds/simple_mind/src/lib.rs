use blob_interface::reference_mind::{
    ReferenceEffort, ReferenceMemoryUpdate, ReferenceMindAction, ReferenceMindDecision,
    ReferenceMindInput,
};
#[cfg(target_arch = "wasm32")]
use blob_interface::reference_mind_converter::{
    capnp_to_reference_mind_input, reference_mind_decision_to_capnp, ReferenceMindLimits,
};
use blob_mind_utils::{
    best_energy_slot, choose_slot, choose_slot_by_quantile, current_food, legal_action_or_wait,
    preferred_effort, random_best_energy_slot, safe_empty_slots,
};
#[cfg(target_arch = "wasm32")]
use extism_pdk::*;

fn choose_action(input: &ReferenceMindInput, randomize_energy_ties: bool) -> ReferenceMindAction {
    let standard = preferred_effort(
        &input.action_space,
        &[
            ReferenceEffort::Standard,
            ReferenceEffort::Gentle,
            ReferenceEffort::Burst,
        ],
    );
    if input.action_space.consume_enabled
        && input.action_space.max_consume_amount > 0
        && current_food(input) > 0
    {
        return ReferenceMindAction::Consume {
            amount: input.action_space.max_consume_amount,
        };
    }
    let food_target = if randomize_energy_ties {
        random_best_energy_slot(
            input,
            input.action_space.move_targets,
            input.randomness.sample_u64(0),
        )
    } else {
        best_energy_slot(input, input.action_space.move_targets)
    };
    if let Some(target_slot) = food_target {
        return ReferenceMindAction::Move {
            target_slot,
            effort: standard,
        };
    }
    if input.self_state.assimilated_energy < 50 && input.action_space.guard_enabled {
        return ReferenceMindAction::Guard { effort: standard };
    }
    let candidates = safe_empty_slots(input, input.action_space.move_targets);
    let fallback_target = if randomize_energy_ties {
        choose_slot_by_quantile(&candidates, input.randomness.sample_u64(0))
    } else {
        choose_slot(&candidates, input.randomness.sample_u64(0))
    };
    if let Some(target_slot) = fallback_target {
        return ReferenceMindAction::Move {
            target_slot,
            effort: standard,
        };
    }
    ReferenceMindAction::Wait
}

pub fn decide(input: &ReferenceMindInput) -> ReferenceMindDecision {
    ReferenceMindDecision {
        action: legal_action_or_wait(input, choose_action(input, false), false),
        signal: None,
        memory_update: ReferenceMemoryUpdate::Retain,
    }
}

/// Feeding teacher that preserves Simple's action priorities while using the
/// cell-private randomness input to disperse equal-energy movement targets.
pub fn decide_collision_aware(input: &ReferenceMindInput) -> ReferenceMindDecision {
    ReferenceMindDecision {
        action: legal_action_or_wait(input, choose_action(input, true), false),
        signal: None,
        memory_update: ReferenceMemoryUpdate::Retain,
    }
}

#[cfg(target_arch = "wasm32")]
#[plugin_fn]
pub fn reference_mind_function(bytes: Vec<u8>) -> FnResult<Vec<u8>> {
    let limits = ReferenceMindLimits::default();
    let input = capnp_to_reference_mind_input(&bytes, limits)
        .map_err(|error| Error::msg(error.to_string()))?;
    let decision = decide(&input);
    Ok(reference_mind_decision_to_capnp(&decision, limits)
        .map_err(|error| Error::msg(error.to_string()))?)
}
