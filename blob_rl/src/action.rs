//! Fixed-shape policy catalog that decodes directly to reference actions.
//!
//! Target slots are reserved up to the ABI maximum so one model shape can be
//! reused across neighborhood rulesets. Per-observation masks prevent the
//! policy from sampling disabled efforts, actions, or unreachable slots.

use blob_interface::reference_mind::{
    ReferenceActionSpace, ReferenceEffort, ReferenceMemoryUpdate, ReferenceMindAction,
    ReferenceMindDecision, ReferenceMindInput, ReferenceSignalEmission, REFERENCE_MAX_LOCAL_SLOTS,
};

const EFFORT_COUNT: usize = 3;
const WAIT: usize = 0;
const GUARD_START: usize = 1;
const CONSUME: usize = GUARD_START + EFFORT_COUNT;
const MOVE_START: usize = CONSUME + 1;
const ATTACK_START: usize = MOVE_START + REFERENCE_MAX_LOCAL_SLOTS * EFFORT_COUNT;
const SPLIT_START: usize = ATTACK_START + REFERENCE_MAX_LOCAL_SLOTS * EFFORT_COUNT;
const REGURGITATE_START: usize = SPLIT_START + REFERENCE_MAX_LOCAL_SLOTS;
const EXCAVATE: usize = REGURGITATE_START + REFERENCE_MAX_LOCAL_SLOTS;
const DEPOSIT_TERRAIN: usize = EXCAVATE + 1;
const NUM_PHYSICAL_ACTIONS: usize = DEPOSIT_TERRAIN + 1;
const SIGNAL_VARIANTS: usize = 5;

/// Maximum fixed policy width. Invalid entries are masked for each cell.
pub const NUM_ACTIONS: usize = NUM_PHYSICAL_ACTIONS * SIGNAL_VARIANTS;

fn effort(index: usize) -> ReferenceEffort {
    match index {
        0 => ReferenceEffort::Gentle,
        1 => ReferenceEffort::Standard,
        _ => ReferenceEffort::Burst,
    }
}

fn effort_index(effort: ReferenceEffort) -> usize {
    match effort {
        ReferenceEffort::Gentle => 0,
        ReferenceEffort::Standard => 1,
        ReferenceEffort::Burst => 2,
    }
}

fn target_is_reachable(input: &ReferenceMindInput, slot: u8) -> bool {
    input
        .slots
        .get(slot as usize)
        .is_some_and(|observation| observation.slot == slot && observation.reachable)
}

fn split_allocation(input: &ReferenceMindInput) -> Option<u64> {
    let minimum_child = input
        .action_space
        .child_core_mass
        .checked_add(input.action_space.minimum_survival_energy)?;
    let expendable = input
        .self_state
        .assimilated_energy
        .saturating_sub(input.action_space.minimum_survival_energy);
    let allocation = (expendable / 3).max(minimum_child);
    (allocation <= expendable).then_some(allocation)
}

fn attack_payload(input: &ReferenceMindInput) -> Option<u64> {
    let expendable = input
        .self_state
        .assimilated_energy
        .saturating_sub(input.action_space.minimum_survival_energy);
    (expendable > 0).then(|| (expendable / 8).max(1))
}

/// Produces the sampling mask for one canonical decision frontier.
pub fn action_mask(input: &ReferenceMindInput) -> [bool; NUM_ACTIONS] {
    let mut physical = [false; NUM_PHYSICAL_ACTIONS];
    physical[WAIT] = input.action_space.wait_enabled;
    for effort_index in 0..EFFORT_COUNT {
        physical[GUARD_START + effort_index] = input.action_space.guard_enabled
            && input.action_space.supports_effort(effort(effort_index));
    }
    physical[CONSUME] =
        input.action_space.consume_enabled && input.action_space.max_consume_amount > 0;

    let can_attack = attack_payload(input).is_some();
    let can_split = split_allocation(input).is_some();
    for slot in 0..REFERENCE_MAX_LOCAL_SLOTS {
        let slot = slot as u8;
        let reachable = target_is_reachable(input, slot);
        for effort_index in 0..EFFORT_COUNT {
            let supported = input.action_space.supports_effort(effort(effort_index));
            physical[MOVE_START + slot as usize * EFFORT_COUNT + effort_index] = reachable
                && supported
                && ReferenceActionSpace::allows_target(input.action_space.move_targets, slot);
            physical[ATTACK_START + slot as usize * EFFORT_COUNT + effort_index] = reachable
                && supported
                && can_attack
                && ReferenceActionSpace::allows_target(input.action_space.attack_targets, slot);
        }
        physical[SPLIT_START + slot as usize] = reachable
            && can_split
            && ReferenceActionSpace::allows_target(input.action_space.split_targets, slot);
        physical[REGURGITATE_START + slot as usize] = reachable
            && input.self_state.gut_energy > 0
            && ReferenceActionSpace::allows_target(input.action_space.regurgitate_targets, slot);
    }
    physical[EXCAVATE] = input.action_space.excavate_enabled;
    physical[DEPOSIT_TERRAIN] = input.action_space.deposit_terrain_enabled
        && input.self_state.carried_material_mass >= input.action_space.terrain_mass_per_elevation;
    if !physical.iter().any(|allowed| *allowed) {
        physical[WAIT] = true;
    }
    let can_signal = input.action_space.signal_enabled
        && input.action_space.signal_emission_cost > 0
        && input.self_state.assimilated_energy
            >= input
                .action_space
                .signal_emission_cost
                .saturating_add(input.action_space.minimum_survival_energy);
    let mut mask = [false; NUM_ACTIONS];
    for (physical_id, allowed) in physical.into_iter().enumerate() {
        mask[physical_id * SIGNAL_VARIANTS] = allowed;
        if allowed && can_signal {
            for signal in 1..SIGNAL_VARIANTS {
                mask[physical_id * SIGNAL_VARIANTS + signal] = true;
            }
        }
    }
    mask
}

/// Decodes one policy choice into an exact reference action and preserves the
/// cell's explicit private memory.
pub fn decode_action(action_id: usize, input: &ReferenceMindInput) -> ReferenceMindDecision {
    let mask = action_mask(input);
    let valid = action_id < NUM_ACTIONS && mask[action_id];
    let physical_id = if valid {
        action_id / SIGNAL_VARIANTS
    } else {
        WAIT
    };
    let action = if physical_id == WAIT {
        ReferenceMindAction::Wait
    } else if physical_id < CONSUME {
        ReferenceMindAction::Guard {
            effort: effort(physical_id - GUARD_START),
        }
    } else if physical_id == CONSUME {
        ReferenceMindAction::Consume {
            amount: input.action_space.max_consume_amount,
        }
    } else if physical_id < ATTACK_START {
        let relative = physical_id - MOVE_START;
        ReferenceMindAction::Move {
            target_slot: (relative / EFFORT_COUNT) as u8,
            effort: effort(relative % EFFORT_COUNT),
        }
    } else if physical_id < SPLIT_START {
        let relative = physical_id - ATTACK_START;
        ReferenceMindAction::Attack {
            target_slot: (relative / EFFORT_COUNT) as u8,
            effort: effort(relative % EFFORT_COUNT),
            payload: attack_payload(input).unwrap_or(1),
        }
    } else if physical_id < REGURGITATE_START {
        ReferenceMindAction::Split {
            target_slot: (physical_id - SPLIT_START) as u8,
            child_allocation: split_allocation(input).unwrap_or(0),
            marker: input.self_state.marker,
            private_memory: input.private_memory.clone(),
        }
    } else if physical_id < EXCAVATE {
        ReferenceMindAction::Regurgitate {
            target_slot: (physical_id - REGURGITATE_START) as u8,
            amount: (input.self_state.gut_energy / 2).max(1),
        }
    } else if physical_id == EXCAVATE {
        ReferenceMindAction::Excavate
    } else {
        ReferenceMindAction::DepositTerrain
    };
    let signal =
        (valid && !action_id.is_multiple_of(SIGNAL_VARIANTS)).then(|| ReferenceSignalEmission {
            channel: (action_id % SIGNAL_VARIANTS - 1) as u8,
        });
    ReferenceMindDecision {
        action,
        signal,
        memory_update: ReferenceMemoryUpdate::Retain,
    }
}

pub fn encode_action(action: &ReferenceMindAction) -> Option<usize> {
    match action {
        ReferenceMindAction::Wait => Some(WAIT * SIGNAL_VARIANTS),
        ReferenceMindAction::Guard { effort } => {
            Some((GUARD_START + effort_index(*effort)) * SIGNAL_VARIANTS)
        }
        ReferenceMindAction::Consume { .. } => Some(CONSUME * SIGNAL_VARIANTS),
        ReferenceMindAction::Move {
            target_slot,
            effort,
        } if usize::from(*target_slot) < REFERENCE_MAX_LOCAL_SLOTS => Some(
            (MOVE_START + usize::from(*target_slot) * EFFORT_COUNT + effort_index(*effort))
                * SIGNAL_VARIANTS,
        ),
        ReferenceMindAction::Attack {
            target_slot,
            effort,
            ..
        } if usize::from(*target_slot) < REFERENCE_MAX_LOCAL_SLOTS => Some(
            (ATTACK_START + usize::from(*target_slot) * EFFORT_COUNT + effort_index(*effort))
                * SIGNAL_VARIANTS,
        ),
        ReferenceMindAction::Split { target_slot, .. }
            if usize::from(*target_slot) < REFERENCE_MAX_LOCAL_SLOTS =>
        {
            Some((SPLIT_START + usize::from(*target_slot)) * SIGNAL_VARIANTS)
        }
        ReferenceMindAction::Regurgitate { target_slot, .. }
            if usize::from(*target_slot) < REFERENCE_MAX_LOCAL_SLOTS =>
        {
            Some((REGURGITATE_START + usize::from(*target_slot)) * SIGNAL_VARIANTS)
        }
        ReferenceMindAction::Excavate => Some(EXCAVATE * SIGNAL_VARIANTS),
        ReferenceMindAction::DepositTerrain => Some(DEPOSIT_TERRAIN * SIGNAL_VARIANTS),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_interface::randomness::PrivateRandom;
    use blob_interface::reference_mind::{
        CurrentTileObservation, LocalObservation, ReferenceActionSpace, ReferenceSelfState,
        EFFORT_BURST_BIT, EFFORT_GENTLE_BIT, EFFORT_STANDARD_BIT,
    };

    fn input() -> ReferenceMindInput {
        ReferenceMindInput {
            self_state: ReferenceSelfState {
                core_mass: 10,
                assimilated_energy: 100,
                gut_energy: 20,
                metabolism_remainder: 0,
                carried_material_mass: 0,
                marker: 7,
                guarded: false,
                last_outcome: None,
            },
            current_tile: CurrentTileObservation {
                elevation: 0,
                plant_energy: 5,
                plant_capacity: 10,
                plant_growth_rate: 1,
                loose_energy: 0,
                diffuse_energy: 0,
                signal_energy: [0; 4],
            },
            slots: (0..8)
                .map(|slot| LocalObservation {
                    slot,
                    dx: 1,
                    dy: 0,
                    distance_cost_q10: 1024,
                    reachable: true,
                    elevation: Some(0),
                    plant_energy: Some(0),
                    plant_capacity: Some(0),
                    plant_growth_rate: Some(0),
                    loose_energy: Some(0),
                    diffuse_energy: Some(0),
                    signal_energy: Some([0; 4]),
                    neighbor: None,
                })
                .collect(),
            action_space: ReferenceActionSpace {
                wait_enabled: true,
                guard_enabled: true,
                consume_enabled: true,
                excavate_enabled: true,
                deposit_terrain_enabled: true,
                signal_enabled: true,
                move_targets: 0xff,
                attack_targets: 0xff,
                split_targets: 0xff,
                regurgitate_targets: 0xff,
                effort_mask: EFFORT_GENTLE_BIT | EFFORT_STANDARD_BIT | EFFORT_BURST_BIT,
                max_consume_amount: 12,
                gut_capacity: 64,
                max_private_memory_bytes: 2048,
                minimum_survival_energy: 1,
                child_core_mass: 10,
                metabolism_rate_numerator: 1,
                metabolism_rate_denominator: 1024,
                terrain_mass_per_elevation: 10,
                signal_emission_cost: 1,
            },
            private_memory: vec![1, 2, 3],
            randomness: PrivateRandom::ZERO,
        }
    }

    #[test]
    fn catalog_decodes_only_reference_actions_and_preserves_memory() {
        let input = input();
        let mask = action_mask(&input);
        for (action_id, allowed) in mask.into_iter().enumerate() {
            if allowed {
                let decision = decode_action(action_id, &input);
                assert_eq!(decision.memory_update, ReferenceMemoryUpdate::Retain);
                assert_eq!(
                    encode_action(&decision.action),
                    Some(action_id - action_id % SIGNAL_VARIANTS)
                );
                assert_eq!(
                    decision.signal.map(|signal| signal.channel),
                    (!action_id.is_multiple_of(SIGNAL_VARIANTS))
                        .then(|| (action_id % SIGNAL_VARIANTS - 1) as u8)
                );
            }
        }
    }

    #[test]
    fn masks_ruleset_disabled_and_unreachable_targets() {
        let mut input = input();
        input.slots[3].reachable = false;
        input.action_space.move_targets &= !(1 << 4);
        let mask = action_mask(&input);
        for effort_index in 0..EFFORT_COUNT {
            assert!(!mask[(MOVE_START + 3 * EFFORT_COUNT + effort_index) * SIGNAL_VARIANTS]);
            assert!(!mask[(MOVE_START + 4 * EFFORT_COUNT + effort_index) * SIGNAL_VARIANTS]);
        }
    }

    #[test]
    fn invalid_policy_choice_becomes_reference_wait() {
        let mut input = input();
        input.action_space.guard_enabled = false;
        assert_eq!(
            decode_action(GUARD_START * SIGNAL_VARIANTS, &input).action,
            ReferenceMindAction::Wait
        );
    }
}
