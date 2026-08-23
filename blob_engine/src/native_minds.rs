//! Native reference-Mind implementations for tests and benchmarks.

use blob_interface::reference_mind::{
    ReferenceActionSpace, ReferenceEffort, ReferenceMemoryUpdate, ReferenceMind,
    ReferenceMindAction, ReferenceMindDecision, ReferenceMindFactory, ReferenceMindInput,
};

fn effort(input: &ReferenceMindInput) -> ReferenceEffort {
    [
        ReferenceEffort::Standard,
        ReferenceEffort::Gentle,
        ReferenceEffort::Burst,
    ]
    .into_iter()
    .find(|effort| input.action_space.supports_effort(*effort))
    .unwrap_or(ReferenceEffort::Standard)
}

fn allowed_slots(input: &ReferenceMindInput, mask: u32, empty: bool) -> Vec<u8> {
    input
        .slots
        .iter()
        .filter(|slot| {
            slot.reachable
                && (!empty || slot.neighbor.is_none())
                && ReferenceActionSpace::allows_target(mask, slot.slot)
        })
        .map(|slot| slot.slot)
        .collect()
}

fn decision(_input: &ReferenceMindInput, action: ReferenceMindAction) -> ReferenceMindDecision {
    ReferenceMindDecision {
        action,
        signal: None,
        memory_update: ReferenceMemoryUpdate::Retain,
    }
}

/// A deterministic-random baseline over actions allowed by the current input.
#[derive(Default)]
pub struct RandomMind;

impl RandomMind {
    pub fn new() -> Self {
        Self
    }
}

impl ReferenceMind for RandomMind {
    fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision {
        let selected_effort = effort(input);
        let mut actions = vec![ReferenceMindAction::Wait];
        if input.action_space.guard_enabled {
            actions.push(ReferenceMindAction::Guard {
                effort: selected_effort,
            });
        }
        if input.action_space.consume_enabled && input.action_space.max_consume_amount > 0 {
            actions.push(ReferenceMindAction::Consume {
                amount: input.action_space.max_consume_amount,
            });
        }
        for slot in allowed_slots(input, input.action_space.move_targets, true) {
            actions.push(ReferenceMindAction::Move {
                target_slot: slot,
                effort: selected_effort,
            });
        }
        let payload = input
            .self_state
            .assimilated_energy
            .saturating_sub(input.action_space.minimum_survival_energy)
            / 8;
        if payload > 0 {
            for slot in allowed_slots(input, input.action_space.attack_targets, false) {
                actions.push(ReferenceMindAction::Attack {
                    target_slot: slot,
                    effort: selected_effort,
                    payload,
                });
            }
        }
        let minimum_child = input
            .action_space
            .child_core_mass
            .saturating_add(input.action_space.minimum_survival_energy);
        let allocation = input.self_state.assimilated_energy / 3;
        if allocation >= minimum_child {
            for slot in allowed_slots(input, input.action_space.split_targets, true) {
                actions.push(ReferenceMindAction::Split {
                    target_slot: slot,
                    child_allocation: allocation,
                    marker: input.self_state.marker,
                    private_memory: input.private_memory.clone(),
                });
            }
        }

        let index = input.randomness.sample_u64(0) as usize % actions.len();
        decision(input, actions.swap_remove(index))
    }

    fn reset(&mut self) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Default)]
pub struct RandomMindFactory;

impl RandomMindFactory {
    pub fn new() -> Self {
        Self
    }
}

impl ReferenceMindFactory for RandomMindFactory {
    type M = RandomMind;

    fn create(&self) -> Result<Self::M, String> {
        Ok(RandomMind)
    }
}

/// A simple aggressive baseline that consumes, splits, attacks, then explores.
#[derive(Default)]
pub struct AggressiveMind;

impl AggressiveMind {
    pub fn new() -> Self {
        Self
    }
}

impl ReferenceMind for AggressiveMind {
    fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision {
        let selected_effort = effort(input);
        let current_food = input
            .current_tile
            .plant_energy
            .saturating_add(input.current_tile.loose_energy);
        if input.action_space.consume_enabled
            && input.action_space.max_consume_amount > 0
            && (current_food > 0
                || input.self_state.assimilated_energy
                    < input.action_space.minimum_survival_energy.saturating_mul(2))
        {
            return decision(
                input,
                ReferenceMindAction::Consume {
                    amount: input.action_space.max_consume_amount,
                },
            );
        }

        let minimum_child = input
            .action_space
            .child_core_mass
            .saturating_add(input.action_space.minimum_survival_energy);
        let allocation = input.self_state.assimilated_energy / 3;
        if allocation >= minimum_child {
            if let Some(slot) = allowed_slots(input, input.action_space.split_targets, true).first()
            {
                return decision(
                    input,
                    ReferenceMindAction::Split {
                        target_slot: *slot,
                        child_allocation: allocation,
                        marker: input.self_state.marker,
                        private_memory: input.private_memory.clone(),
                    },
                );
            }
        }

        let payload = input
            .self_state
            .assimilated_energy
            .saturating_sub(input.action_space.minimum_survival_energy)
            / 5;
        if payload > 0 {
            if let Some(slot) = input.slots.iter().find(|slot| {
                slot.neighbor.is_some()
                    && slot.reachable
                    && ReferenceActionSpace::allows_target(
                        input.action_space.attack_targets,
                        slot.slot,
                    )
            }) {
                return decision(
                    input,
                    ReferenceMindAction::Attack {
                        target_slot: slot.slot,
                        effort: selected_effort,
                        payload,
                    },
                );
            }
        }

        if let Some(slot) = input
            .slots
            .iter()
            .filter(|slot| {
                slot.neighbor.is_none()
                    && slot.reachable
                    && ReferenceActionSpace::allows_target(
                        input.action_space.move_targets,
                        slot.slot,
                    )
            })
            .max_by_key(|slot| {
                slot.plant_energy
                    .unwrap_or(0)
                    .saturating_add(slot.loose_energy.unwrap_or(0))
                    .saturating_add(
                        slot.diffuse_energy
                            .unwrap_or(0)
                            .saturating_mul(u64::from(slot.plant_growth_rate.unwrap_or(0) > 0)),
                    )
            })
        {
            return decision(
                input,
                ReferenceMindAction::Move {
                    target_slot: slot.slot,
                    effort: selected_effort,
                },
            );
        }

        decision(input, ReferenceMindAction::Wait)
    }

    fn reset(&mut self) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Default)]
pub struct AggressiveMindFactory;

impl ReferenceMindFactory for AggressiveMindFactory {
    type M = AggressiveMind;

    fn create(&self) -> Result<Self::M, String> {
        Ok(AggressiveMind)
    }
}
