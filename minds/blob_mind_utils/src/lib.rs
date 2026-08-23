//! Shared helpers for Minds that consume the reference-native ABI.

use blob_interface::reference_mind::{
    LocalObservation, ReferenceActionSpace, ReferenceEffort, ReferenceMindInput,
};

pub fn current_food(input: &ReferenceMindInput) -> u64 {
    input
        .current_tile
        .plant_energy
        .saturating_add(input.current_tile.loose_energy)
}

pub fn observed_slot_energy(slot: &LocalObservation) -> u64 {
    slot.plant_energy
        .unwrap_or(0)
        .saturating_add(slot.loose_energy.unwrap_or(0))
        .saturating_add(if slot.plant_growth_rate.unwrap_or(0) > 0 {
            slot.diffuse_energy.unwrap_or(0)
        } else {
            0
        })
}

pub fn preferred_effort(
    space: &ReferenceActionSpace,
    preferences: &[ReferenceEffort],
) -> ReferenceEffort {
    preferences
        .iter()
        .copied()
        .find(|effort| space.supports_effort(*effort))
        .unwrap_or(ReferenceEffort::Standard)
}

pub fn available_slots(input: &ReferenceMindInput, mask: u32, require_empty: bool) -> Vec<u8> {
    input
        .slots
        .iter()
        .filter(|slot| {
            slot.reachable
                && ReferenceActionSpace::allows_target(mask, slot.slot)
                && (!require_empty || slot.neighbor.is_none())
        })
        .map(|slot| slot.slot)
        .collect()
}

pub fn safe_empty_slots(input: &ReferenceMindInput, mask: u32) -> Vec<u8> {
    input
        .slots
        .iter()
        .filter(|slot| {
            slot.reachable
                && slot.neighbor.is_none()
                && ReferenceActionSpace::allows_target(mask, slot.slot)
                && slot
                    .elevation
                    .is_none_or(|elevation| (elevation - input.current_tile.elevation).abs() <= 1)
        })
        .map(|slot| slot.slot)
        .collect()
}

pub fn best_energy_slot(input: &ReferenceMindInput, mask: u32) -> Option<u8> {
    input
        .slots
        .iter()
        .filter(|slot| {
            slot.reachable
                && slot.neighbor.is_none()
                && ReferenceActionSpace::allows_target(mask, slot.slot)
        })
        .max_by_key(|slot| (observed_slot_energy(slot), std::cmp::Reverse(slot.slot)))
        .filter(|slot| observed_slot_energy(slot) > 0)
        .map(|slot| slot.slot)
}

pub fn choose_slot(candidates: &[u8], random_word: u64) -> Option<u8> {
    (!candidates.is_empty()).then(|| candidates[random_word as usize % candidates.len()])
}

pub fn attack_payload(input: &ReferenceMindInput, divisor: u64) -> Option<u64> {
    let expendable = input
        .self_state
        .assimilated_energy
        .saturating_sub(input.action_space.minimum_survival_energy);
    (expendable > 0).then(|| (expendable / divisor.max(1)).max(1))
}

pub fn split_allocation(input: &ReferenceMindInput) -> Option<u64> {
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
