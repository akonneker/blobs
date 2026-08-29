//! Shared helpers for Minds that consume the reference-native ABI.

use blob_interface::reference_mind::{
    LocalObservation, ReferenceActionSpace, ReferenceEffort, ReferenceMindAction,
    ReferenceMindInput,
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

/// Choose uniformly among the locally visible vacant slots tied for the most
/// observed energy. The caller supplies a cell-private random word; no shared
/// state or team identity participates in the tie break.
pub fn random_best_energy_slot(
    input: &ReferenceMindInput,
    mask: u32,
    random_word: u64,
) -> Option<u8> {
    let best_energy = input
        .slots
        .iter()
        .filter(|slot| {
            slot.reachable
                && slot.neighbor.is_none()
                && ReferenceActionSpace::allows_target(mask, slot.slot)
        })
        .map(observed_slot_energy)
        .max()
        .filter(|energy| *energy > 0)?;
    let candidates = input
        .slots
        .iter()
        .filter(|slot| {
            slot.reachable
                && slot.neighbor.is_none()
                && ReferenceActionSpace::allows_target(mask, slot.slot)
                && observed_slot_energy(slot) == best_energy
        })
        .map(|slot| slot.slot)
        .collect::<Vec<_>>();
    choose_slot_by_quantile(&candidates, random_word)
}

pub fn choose_slot(candidates: &[u8], random_word: u64) -> Option<u8> {
    (!candidates.is_empty()).then(|| candidates[random_word as usize % candidates.len()])
}

/// Map a uniformly random word onto ordered candidates as contiguous ranges.
/// This remains uniform (up to unavoidable integer rounding), while making a
/// teacher's target a smooth threshold function of the exposed random input.
pub fn choose_slot_by_quantile(candidates: &[u8], random_word: u64) -> Option<u8> {
    if candidates.is_empty() {
        return None;
    }
    let index = (u128::from(random_word) * candidates.len() as u128) >> u64::BITS;
    Some(candidates[index as usize])
}

fn multiply_ratio_ceil(value: u128, numerator: u64, denominator: u64) -> Option<u64> {
    if denominator == 0 {
        return None;
    }
    let product = value.checked_mul(u128::from(numerator))?;
    let rounded =
        product.checked_add(u128::from(denominator).saturating_sub(1))? / u128::from(denominator);
    u64::try_from(rounded).ok()
}

fn effort_cost(input: &ReferenceMindInput, raw: u64, effort: ReferenceEffort) -> Option<u64> {
    multiply_ratio_ceil(
        u128::from(raw),
        u64::from(input.action_space.effort_cost_numerators[effort.index()]),
        u64::from(input.action_space.effort_cost_denominators[effort.index()]),
    )
}

fn fixed_effort_cost(input: &ReferenceMindInput, raw: u64, effort: ReferenceEffort) -> u64 {
    effort_cost(input, raw, effort).unwrap_or(u64::MAX)
}

fn move_effort_cost(input: &ReferenceMindInput, slot: u8, effort: ReferenceEffort) -> Option<u64> {
    let distance = input
        .slots
        .iter()
        .find(|observation| observation.slot == slot)?
        .distance_cost_q10;
    let mass = u128::from(input.self_state.core_mass)
        .checked_add(u128::from(input.self_state.assimilated_energy))?
        .checked_add(u128::from(input.self_state.gut_energy))?
        .checked_add(u128::from(input.self_state.carried_material_mass))?;
    let denominator = input
        .action_space
        .move_mass_units_per_effort
        .checked_mul(1024)?;
    let inertial = multiply_ratio_ceil(mass, u64::from(distance), denominator)?;
    effort_cost(
        input,
        input.action_space.move_effort_base.checked_add(inertial)?,
        effort,
    )
}

fn affordable(
    input: &ReferenceMindInput,
    effort_cost: u64,
    payload: u64,
    signal_cost: u64,
) -> bool {
    effort_cost
        .checked_add(payload)
        .and_then(|required| required.checked_add(signal_cost))
        .and_then(|required| required.checked_add(input.action_space.minimum_survival_energy))
        .is_some_and(|required| input.self_state.assimilated_energy >= required)
}

pub fn attack_payload(
    input: &ReferenceMindInput,
    effort: ReferenceEffort,
    divisor: u64,
) -> Option<u64> {
    let expendable = input
        .self_state
        .assimilated_energy
        .checked_sub(input.action_space.minimum_survival_energy)?
        .checked_sub(fixed_effort_cost(
            input,
            input.action_space.attack_effort_base,
            effort,
        ))?;
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
        .checked_sub(input.action_space.minimum_survival_energy)?
        .checked_sub(fixed_effort_cost(
            input,
            input.action_space.split_effort_base,
            ReferenceEffort::Standard,
        ))?;
    let allocation = (expendable / 3).max(minimum_child);
    (allocation <= expendable).then_some(allocation)
}

/// Checks all deterministic commit requirements visible through the anonymous
/// Mind ABI. A successful check does not promise resolution success: another
/// cell can still contest, frustrate, or interrupt the action.
pub fn action_is_commit_legal(
    input: &ReferenceMindInput,
    action: &ReferenceMindAction,
    emits_signal: bool,
) -> bool {
    let signal_amount = if emits_signal {
        input.action_space.signal_emission_cost
    } else {
        0
    };
    action_is_commit_legal_with_signal_amount(input, action, signal_amount)
}

/// Checks commit legality while pricing the exact conserved energy carried by
/// a sidecar signal. This is required for variable-strength emissions; the
/// boolean compatibility helper above represents one emission quantum.
pub fn action_is_commit_legal_with_signal_amount(
    input: &ReferenceMindInput,
    action: &ReferenceMindAction,
    signal_amount: u64,
) -> bool {
    if signal_amount > 0 && !input.action_space.signal_enabled {
        return false;
    }
    let quantum = input.action_space.signal_emission_cost;
    if signal_amount > 0 && (quantum == 0 || !signal_amount.is_multiple_of(quantum)) {
        return false;
    }
    let target_is_reachable = |slot: u8| {
        input
            .slots
            .iter()
            .any(|observation| observation.slot == slot && observation.reachable)
    };
    let (enabled, cost, payload) = match action {
        ReferenceMindAction::Wait => (input.action_space.wait_enabled, 0, 0),
        ReferenceMindAction::Move {
            target_slot,
            effort,
        } => (
            target_is_reachable(*target_slot)
                && input.action_space.supports_effort(*effort)
                && ReferenceActionSpace::allows_target(
                    input.action_space.move_targets,
                    *target_slot,
                ),
            move_effort_cost(input, *target_slot, *effort).unwrap_or(u64::MAX),
            0,
        ),
        ReferenceMindAction::Attack {
            target_slot,
            effort,
            payload,
        } => (
            *payload > 0
                && target_is_reachable(*target_slot)
                && input.action_space.supports_effort(*effort)
                && ReferenceActionSpace::allows_target(
                    input.action_space.attack_targets,
                    *target_slot,
                ),
            fixed_effort_cost(input, input.action_space.attack_effort_base, *effort),
            *payload,
        ),
        ReferenceMindAction::Guard { effort } => (
            input.action_space.guard_enabled && input.action_space.supports_effort(*effort),
            fixed_effort_cost(input, input.action_space.guard_effort_base, *effort),
            0,
        ),
        ReferenceMindAction::Consume { amount } => (
            input.action_space.consume_enabled
                && *amount > 0
                && *amount <= input.action_space.max_consume_amount,
            fixed_effort_cost(
                input,
                input.action_space.consume_effort_base,
                ReferenceEffort::Standard,
            ),
            0,
        ),
        ReferenceMindAction::Split {
            target_slot,
            child_allocation,
            private_memory,
            ..
        } => {
            let minimum_child = input
                .action_space
                .child_core_mass
                .checked_add(input.action_space.minimum_survival_energy);
            (
                minimum_child.is_some_and(|minimum| *child_allocation >= minimum)
                    && private_memory.len() <= input.action_space.max_private_memory_bytes as usize
                    && target_is_reachable(*target_slot)
                    && ReferenceActionSpace::allows_target(
                        input.action_space.split_targets,
                        *target_slot,
                    ),
                fixed_effort_cost(
                    input,
                    input.action_space.split_effort_base,
                    ReferenceEffort::Standard,
                ),
                *child_allocation,
            )
        }
        ReferenceMindAction::Regurgitate {
            target_slot,
            amount,
        } => (
            *amount > 0
                && *amount <= input.self_state.gut_energy
                && target_is_reachable(*target_slot)
                && ReferenceActionSpace::allows_target(
                    input.action_space.regurgitate_targets,
                    *target_slot,
                ),
            fixed_effort_cost(
                input,
                input.action_space.regurgitate_effort_base,
                ReferenceEffort::Standard,
            ),
            0,
        ),
        ReferenceMindAction::Signal { amounts } => {
            let total = amounts
                .iter()
                .try_fold(0_u64, |total, amount| total.checked_add(*amount));
            let quantum = input.action_space.signal_emission_cost;
            (
                signal_amount == 0
                    && input.action_space.signal_enabled
                    && total.is_some_and(|total| total > 0)
                    && amounts
                        .iter()
                        .all(|amount| *amount == 0 || quantum > 0 && *amount % quantum == 0),
                0,
                total.unwrap_or(u64::MAX),
            )
        }
        ReferenceMindAction::Excavate => (
            input.action_space.excavate_enabled,
            fixed_effort_cost(
                input,
                input.action_space.excavate_effort_base,
                ReferenceEffort::Standard,
            ),
            0,
        ),
        ReferenceMindAction::DepositTerrain => (
            input.action_space.deposit_terrain_enabled
                && input.self_state.carried_material_mass
                    >= input.action_space.terrain_mass_per_elevation,
            fixed_effort_cost(
                input,
                input.action_space.deposit_terrain_effort_base,
                ReferenceEffort::Standard,
            ),
            0,
        ),
    };
    enabled && affordable(input, cost, payload, signal_amount)
}

/// Keeps example Minds from producing deterministic resolver rejections when
/// a ruleset makes their preferred action unaffordable.
pub fn legal_action_or_wait(
    input: &ReferenceMindInput,
    action: ReferenceMindAction,
    emits_signal: bool,
) -> ReferenceMindAction {
    if action_is_commit_legal(input, &action, emits_signal) {
        return action;
    }
    let wait = ReferenceMindAction::Wait;
    if action_is_commit_legal(input, &wait, false) {
        return wait;
    }
    for effort in [
        ReferenceEffort::Gentle,
        ReferenceEffort::Standard,
        ReferenceEffort::Burst,
    ] {
        let guard = ReferenceMindAction::Guard { effort };
        if action_is_commit_legal(input, &guard, false) {
            return guard;
        }
    }
    if input.action_space.max_consume_amount > 0 {
        let consume = ReferenceMindAction::Consume {
            amount: input.action_space.max_consume_amount,
        };
        if action_is_commit_legal(input, &consume, false) {
            return consume;
        }
    }
    let excavate = ReferenceMindAction::Excavate;
    if action_is_commit_legal(input, &excavate, false) {
        return excavate;
    }
    // A valid ruleset must expose at least one action at every decision
    // frontier. Preserve the canonical inert fallback if the input violates
    // that invariant; the resolver will report the malformed frontier.
    wait
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_slot_choice_has_contiguous_balanced_ranges() {
        let candidates = [3, 7];
        assert_eq!(choose_slot_by_quantile(&candidates, 0), Some(3));
        assert_eq!(
            choose_slot_by_quantile(&candidates, (1_u64 << 63) - 1),
            Some(3)
        );
        assert_eq!(choose_slot_by_quantile(&candidates, 1_u64 << 63), Some(7));
        assert_eq!(choose_slot_by_quantile(&candidates, u64::MAX), Some(7));
        assert_eq!(choose_slot_by_quantile(&[], u64::MAX), None);
    }
}
