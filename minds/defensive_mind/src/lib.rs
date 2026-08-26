use blob_interface::reference_mind::{
    ReferenceEffort, ReferenceMemoryUpdate, ReferenceMindAction, ReferenceMindDecision,
    ReferenceMindInput,
};
#[cfg(target_arch = "wasm32")]
use blob_interface::reference_mind_converter::{
    ReferenceMindLimits, capnp_to_reference_mind_input, reference_mind_decision_to_capnp,
};
use blob_mind_utils::{
    best_energy_slot, choose_slot, current_food, legal_action_or_wait, preferred_effort,
    safe_empty_slots, split_allocation,
};
#[cfg(target_arch = "wasm32")]
use extism_pdk::*;

fn choose_action(input: &ReferenceMindInput) -> ReferenceMindAction {
    let gentle = preferred_effort(
        &input.action_space,
        &[ReferenceEffort::Gentle, ReferenceEffort::Standard],
    );
    let standard = preferred_effort(
        &input.action_space,
        &[ReferenceEffort::Standard, ReferenceEffort::Gentle],
    );
    if input.self_state.assimilated_energy < 80
        && input.action_space.consume_enabled
        && input.action_space.max_consume_amount > 0
        && current_food(input) > 0
    {
        return ReferenceMindAction::Consume {
            amount: input.action_space.max_consume_amount,
        };
    }
    if input.self_state.assimilated_energy < 80
        && let Some(target_slot) = best_energy_slot(input, input.action_space.move_targets)
    {
        return ReferenceMindAction::Move {
            target_slot,
            effort: gentle,
        };
    }
    if input.self_state.assimilated_energy < 50 && input.action_space.guard_enabled {
        return ReferenceMindAction::Guard { effort: standard };
    }
    if input.self_state.assimilated_energy > 200 {
        let targets = safe_empty_slots(input, input.action_space.split_targets);
        if let (Some(target_slot), Some(child_allocation)) = (
            choose_slot(&targets, input.randomness.sample_u64(0)),
            split_allocation(input),
        ) {
            return ReferenceMindAction::Split {
                target_slot,
                child_allocation,
                marker: 3,
                private_memory: input.private_memory.clone(),
            };
        }
    }
    if input.action_space.guard_enabled {
        ReferenceMindAction::Guard { effort: standard }
    } else {
        ReferenceMindAction::Wait
    }
}

pub fn decide(input: &ReferenceMindInput) -> ReferenceMindDecision {
    ReferenceMindDecision {
        action: legal_action_or_wait(input, choose_action(input), false),
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
