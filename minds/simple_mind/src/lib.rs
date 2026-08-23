use blob_interface::reference_mind::{
    ReferenceEffort, ReferenceMemoryUpdate, ReferenceMindAction, ReferenceMindDecision,
    ReferenceMindInput,
};
use blob_interface::reference_mind_converter::{
    capnp_to_reference_mind_input, reference_mind_decision_to_capnp, ReferenceMindLimits,
};
use blob_mind_utils::{
    best_energy_slot, choose_slot, current_food, preferred_effort, safe_empty_slots,
};
use extism_pdk::*;

fn decide(input: &ReferenceMindInput) -> ReferenceMindAction {
    let standard = preferred_effort(
        &input.action_space,
        &[
            ReferenceEffort::Standard,
            ReferenceEffort::Gentle,
            ReferenceEffort::Burst,
        ],
    );
    if input.self_state.assimilated_energy < 50 && input.action_space.guard_enabled {
        return ReferenceMindAction::Guard { effort: standard };
    }
    if input.action_space.consume_enabled
        && input.action_space.max_consume_amount > 0
        && current_food(input) > 0
    {
        return ReferenceMindAction::Consume {
            amount: input.action_space.max_consume_amount,
        };
    }
    if let Some(target_slot) = best_energy_slot(input, input.action_space.move_targets) {
        return ReferenceMindAction::Move {
            target_slot,
            effort: standard,
        };
    }
    let candidates = safe_empty_slots(input, input.action_space.move_targets);
    if let Some(target_slot) = choose_slot(&candidates, input.randomness.sample_u64(0)) {
        return ReferenceMindAction::Move {
            target_slot,
            effort: standard,
        };
    }
    ReferenceMindAction::Wait
}

#[plugin_fn]
pub fn reference_mind_function(bytes: Vec<u8>) -> FnResult<Vec<u8>> {
    let limits = ReferenceMindLimits::default();
    let input = capnp_to_reference_mind_input(&bytes, limits)
        .map_err(|error| Error::msg(error.to_string()))?;
    let decision = ReferenceMindDecision {
        action: decide(&input),
        signal: None,
        memory_update: ReferenceMemoryUpdate::Retain,
    };
    Ok(reference_mind_decision_to_capnp(&decision, limits)
        .map_err(|error| Error::msg(error.to_string()))?)
}
