use blob_interface::reference_mind::{
    ReferenceActionSpace, ReferenceEffort, ReferenceMemoryUpdate, ReferenceMindAction,
    ReferenceMindDecision, ReferenceMindInput,
};
use blob_interface::reference_mind_converter::{
    ReferenceMindLimits, capnp_to_reference_mind_input, reference_mind_decision_to_capnp,
};
use blob_mind_utils::{
    attack_payload, available_slots, best_energy_slot, choose_slot, current_food, preferred_effort,
    split_allocation,
};
use extism_pdk::*;

fn decide(input: &ReferenceMindInput) -> ReferenceMindAction {
    let standard = preferred_effort(
        &input.action_space,
        &[ReferenceEffort::Standard, ReferenceEffort::Burst],
    );
    let burst = preferred_effort(
        &input.action_space,
        &[ReferenceEffort::Burst, ReferenceEffort::Standard],
    );
    if input.self_state.assimilated_energy < 40 {
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
    }

    let occupied_targets: Vec<_> = input
        .slots
        .iter()
        .filter(|slot| {
            slot.neighbor.is_some()
                && slot.reachable
                && ReferenceActionSpace::allows_target(input.action_space.attack_targets, slot.slot)
        })
        .map(|slot| slot.slot)
        .collect();
    if let (Some(target_slot), Some(payload)) = (
        choose_slot(&occupied_targets, input.randomness.sample_u64(1)),
        attack_payload(input, 8),
    ) {
        return ReferenceMindAction::Attack {
            target_slot,
            effort: burst,
            payload,
        };
    }

    if input.self_state.assimilated_energy > 150 {
        let targets = available_slots(input, input.action_space.split_targets, true);
        if let (Some(target_slot), Some(child_allocation)) = (
            choose_slot(&targets, input.randomness.sample_u64(2)),
            split_allocation(input),
        ) {
            return ReferenceMindAction::Split {
                target_slot,
                child_allocation,
                marker: 2,
                private_memory: Vec::new(),
            };
        }
    }

    if let Some(target_slot) = best_energy_slot(input, input.action_space.move_targets) {
        return ReferenceMindAction::Move {
            target_slot,
            effort: burst,
        };
    }
    let targets = available_slots(input, input.action_space.move_targets, true);
    if let Some(target_slot) = choose_slot(&targets, input.randomness.sample_u64(3)) {
        return ReferenceMindAction::Move {
            target_slot,
            effort: burst,
        };
    }
    if input.action_space.guard_enabled {
        ReferenceMindAction::Guard { effort: standard }
    } else {
        ReferenceMindAction::Wait
    }
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
