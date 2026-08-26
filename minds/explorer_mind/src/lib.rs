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

const MEMORY_BYTES: usize = 4;

fn read_u16(memory: &[u8], offset: usize) -> u16 {
    memory
        .get(offset..offset + 2)
        .and_then(|bytes| bytes.try_into().ok())
        .map(u16::from_le_bytes)
        .unwrap_or(0)
}

fn write_u16(memory: &mut [u8], offset: usize, value: u16) {
    if let Some(bytes) = memory.get_mut(offset..offset + 2) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
}

fn choose_action(input: &ReferenceMindInput, memory: &mut Vec<u8>) -> ReferenceMindAction {
    if input.action_space.max_private_memory_bytes as usize >= MEMORY_BYTES
        && memory.len() < MEMORY_BYTES
    {
        memory.resize(MEMORY_BYTES, 0);
    }
    let mut plants_seen = read_u16(memory, 0);
    let mut cursor = read_u16(memory, 2) as usize;

    if input.current_tile.plant_energy > 0 {
        plants_seen = plants_seen.saturating_add(1);
        write_u16(memory, 0, plants_seen);
    }

    if input.action_space.consume_enabled
        && input.action_space.max_consume_amount > 0
        && current_food(input) > 0
    {
        return ReferenceMindAction::Consume {
            amount: input.action_space.max_consume_amount,
        };
    }

    if input.self_state.assimilated_energy > 200 && plants_seen > 0 {
        let targets = safe_empty_slots(input, input.action_space.split_targets);
        if let (Some(target_slot), Some(child_allocation)) = (
            choose_slot(&targets, input.randomness.sample_u64(0)),
            split_allocation(input),
        ) {
            let mut child_memory = memory.clone();
            if !child_memory.is_empty() {
                let next_cursor = cursor.saturating_add(1) as u16;
                write_u16(&mut child_memory, 2, next_cursor);
            }
            return ReferenceMindAction::Split {
                target_slot,
                child_allocation,
                marker: 1,
                private_memory: child_memory,
            };
        }
    }

    let effort = preferred_effort(
        &input.action_space,
        &[ReferenceEffort::Gentle, ReferenceEffort::Standard],
    );
    if input.self_state.assimilated_energy < 40
        && let Some(target_slot) = best_energy_slot(input, input.action_space.move_targets)
    {
        return ReferenceMindAction::Move {
            target_slot,
            effort,
        };
    }

    let targets = safe_empty_slots(input, input.action_space.move_targets);
    if !targets.is_empty() {
        cursor %= targets.len();
        let target_slot = targets[cursor];
        cursor = (cursor + 1) % targets.len();
        write_u16(memory, 2, cursor as u16);
        return ReferenceMindAction::Move {
            target_slot,
            effort,
        };
    }

    ReferenceMindAction::Wait
}

pub fn decide(input: &ReferenceMindInput) -> ReferenceMindDecision {
    let mut memory = input.private_memory.clone();
    let action = choose_action(input, &mut memory);
    ReferenceMindDecision {
        action: legal_action_or_wait(input, action, false),
        signal: None,
        memory_update: ReferenceMemoryUpdate::Replace(memory),
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
