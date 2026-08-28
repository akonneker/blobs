//! Fixed-shape tensor projection of the canonical reference Mind input.

use blob_interface::randomness::PRIVATE_RANDOM_BYTES;
use blob_interface::reference_mind::{
    ReferenceActivity, ReferenceMindInput, ReferenceOutcomeStatus, ReferenceProgress,
    REFERENCE_MAX_LOCAL_SLOTS,
};

use crate::action::{
    policy_masks, NUM_ACTIONS, NUM_AMOUNT_CHOICES, NUM_SIGNAL_CHOICES, NUM_SIGNAL_STRENGTH_CHOICES,
};

const STATE_HEADER_FEATURES: usize = 38;
pub const OBS_RANDOMNESS_FEATURE_START: usize = STATE_HEADER_FEATURES;
pub const OBS_RANDOMNESS_FEATURE_END: usize = STATE_HEADER_FEATURES + PRIVATE_RANDOM_BYTES;
const HEADER_FEATURES: usize = STATE_HEADER_FEATURES + PRIVATE_RANDOM_BYTES;
const SLOT_FEATURES: usize = 33;
pub const OBS_DIM: usize = HEADER_FEATURES + REFERENCE_MAX_LOCAL_SLOTS * SLOT_FEATURES;

#[derive(Debug, Clone)]
pub struct Observation {
    pub data: [f32; OBS_DIM],
    pub action_mask: [bool; NUM_ACTIONS],
    /// Five conditional amount masks packed into the low bits per action.
    pub amount_choice_bits: [u8; NUM_ACTIONS],
    /// Five signal-strength bits packed for each action-amount tier.
    pub sidecar_strength_bits: [u32; NUM_ACTIONS],
    pub explicit_signal_strength_bits: [u8; 4],
}

fn amount(value: u64) -> f32 {
    (value as f64).ln_1p().min(32.0) as f32 / 32.0
}

fn marker(value: u32) -> f32 {
    value as f32 / u32::MAX as f32
}

fn outcome(status: ReferenceOutcomeStatus) -> f32 {
    match status {
        ReferenceOutcomeStatus::Success => 1.0 / 5.0,
        ReferenceOutcomeStatus::Frustrated => 2.0 / 5.0,
        ReferenceOutcomeStatus::Contested => 3.0 / 5.0,
        ReferenceOutcomeStatus::Interrupted => 4.0 / 5.0,
        ReferenceOutcomeStatus::Rejected => 1.0,
    }
}

fn activity(value: ReferenceActivity) -> f32 {
    match value {
        ReferenceActivity::Ready => 1.0 / 8.0,
        ReferenceActivity::Moving => 2.0 / 8.0,
        ReferenceActivity::AttackWindup => 3.0 / 8.0,
        ReferenceActivity::Guarding => 4.0 / 8.0,
        ReferenceActivity::Feeding => 5.0 / 8.0,
        ReferenceActivity::Splitting => 6.0 / 8.0,
        ReferenceActivity::ManipulatingTerrain => 7.0 / 8.0,
        ReferenceActivity::OtherBusy => 1.0,
    }
}

fn progress(value: ReferenceProgress) -> f32 {
    match value {
        ReferenceProgress::Early => 1.0 / 3.0,
        ReferenceProgress::Middle => 2.0 / 3.0,
        ReferenceProgress::Late => 1.0,
    }
}

impl Observation {
    pub fn from_reference(input: &ReferenceMindInput) -> Self {
        let mut data = [0.0; OBS_DIM];
        let state = &input.self_state;
        data[0] = amount(state.core_mass);
        data[1] = amount(state.assimilated_energy);
        data[2] = amount(state.gut_energy);
        data[3] = amount(state.carried_material_mass);
        data[4] = marker(state.marker);
        data[5] = f32::from(state.guarded);
        if let Some(last) = state.last_outcome {
            data[6] = 1.0;
            data[7] = outcome(last.status);
            data[8] = last
                .rejected_reason
                .map(|reason| (reason as u8 as f32 + 1.0) / 12.0)
                .unwrap_or(0.0);
        }
        data[9] = input.current_tile.elevation as f32 / i16::MAX as f32;
        data[10] = amount(input.current_tile.plant_energy);
        data[11] = amount(input.current_tile.loose_energy);
        data[12] = amount(input.current_tile.diffuse_energy);
        data[13] = f32::from(input.action_space.wait_enabled);
        data[14] = f32::from(input.action_space.guard_enabled);
        data[15] = f32::from(input.action_space.consume_enabled);
        data[16] = f32::from(input.action_space.effort_mask & 1 != 0);
        data[17] = f32::from(input.action_space.effort_mask & 2 != 0);
        data[18] = f32::from(input.action_space.effort_mask & 4 != 0);
        data[19] = amount(input.action_space.max_consume_amount);
        data[20] = amount(input.action_space.gut_capacity);
        data[21] = amount(u64::from(input.action_space.max_private_memory_bytes));
        data[22] = amount(input.action_space.minimum_survival_energy);
        data[23] = amount(input.action_space.child_core_mass);
        data[24] = amount(input.current_tile.plant_capacity);
        data[25] = amount(input.current_tile.plant_growth_rate);
        data[26] = amount(state.metabolism_remainder);
        data[27] = amount(input.action_space.metabolism_rate_numerator);
        data[28] = amount(input.action_space.metabolism_rate_denominator);
        for (index, energy) in input.current_tile.signal_energy.into_iter().enumerate() {
            data[29 + index] = amount(energy);
        }
        data[33] = f32::from(input.action_space.excavate_enabled);
        data[34] = f32::from(input.action_space.deposit_terrain_enabled);
        data[35] = f32::from(input.action_space.signal_enabled);
        data[36] = amount(input.action_space.terrain_mass_per_elevation);
        data[37] = amount(input.action_space.signal_emission_cost);
        for (index, byte) in input.randomness.as_bytes().iter().copied().enumerate() {
            data[OBS_RANDOMNESS_FEATURE_START + index] = f32::from(byte) / f32::from(u8::MAX);
        }

        for slot in &input.slots {
            if usize::from(slot.slot) >= REFERENCE_MAX_LOCAL_SLOTS {
                continue;
            }
            let base = HEADER_FEATURES + usize::from(slot.slot) * SLOT_FEATURES;
            data[base] = 1.0;
            data[base + 1] = f32::from(slot.reachable);
            data[base + 2] = slot.dx as f32 / i8::MAX as f32;
            data[base + 3] = slot.dy as f32 / i8::MAX as f32;
            data[base + 4] = slot.distance_cost_q10 as f32 / u16::MAX as f32;
            if let Some(elevation) = slot.elevation {
                data[base + 5] = 1.0;
                data[base + 6] = elevation as f32 / i16::MAX as f32;
            }
            if let Some(value) = slot.plant_energy {
                data[base + 7] = 1.0;
                data[base + 8] = amount(value);
            }
            if let Some(value) = slot.loose_energy {
                data[base + 9] = 1.0;
                data[base + 10] = amount(value);
            }
            if let Some(value) = slot.diffuse_energy {
                data[base + 11] = 1.0;
                data[base + 12] = amount(value);
            }
            if let Some(neighbor) = slot.neighbor {
                data[base + 13] = 1.0;
                if let Some(value) = neighbor.marker {
                    data[base + 14] = 1.0;
                    data[base + 15] = marker(value);
                }
                if let Some(value) = neighbor.apparent_mass_bucket {
                    data[base + 16] = 1.0;
                    data[base + 17] = value as f32 / u8::MAX as f32;
                }
                data[base + 18] = neighbor.activity.map(activity).unwrap_or(0.0);
                data[base + 19] = neighbor.progress.map(progress).unwrap_or(0.0);
            }
            data[base + 20] =
                f32::from(input.action_space.move_targets & (1_u32 << slot.slot) != 0);
            data[base + 21] =
                f32::from(input.action_space.attack_targets & (1_u32 << slot.slot) != 0);
            data[base + 22] =
                f32::from(input.action_space.split_targets & (1_u32 << slot.slot) != 0);
            data[base + 23] =
                f32::from(input.action_space.regurgitate_targets & (1_u32 << slot.slot) != 0);
            if let Some(value) = slot.plant_capacity {
                data[base + 24] = 1.0;
                data[base + 25] = amount(value);
            }
            if let Some(value) = slot.plant_growth_rate {
                data[base + 26] = 1.0;
                data[base + 27] = amount(value);
            }
            if let Some(signal) = slot.signal_energy {
                data[base + 28] = 1.0;
                for (index, energy) in signal.into_iter().enumerate() {
                    data[base + 29 + index] = amount(energy);
                }
            }
        }

        let masks = policy_masks(input);
        Self {
            data,
            action_mask: masks.actions,
            amount_choice_bits: masks.amount_choice_bits,
            sidecar_strength_bits: masks.sidecar_strength_bits,
            explicit_signal_strength_bits: masks.explicit_signal_strength_bits,
        }
    }

    pub fn amount_mask(&self, action: usize) -> [bool; NUM_AMOUNT_CHOICES] {
        let bits = self.amount_choice_bits.get(action).copied().unwrap_or(1);
        std::array::from_fn(|choice| bits & (1 << choice) != 0)
    }

    pub fn signal_mask(&self, action: usize, amount: usize) -> [bool; NUM_SIGNAL_CHOICES] {
        let mut mask = [false; NUM_SIGNAL_CHOICES];
        if action == NUM_ACTIONS - 1 {
            for (pattern, allowed) in mask.iter_mut().enumerate().skip(1) {
                *allowed =
                    self.explicit_signal_strength_bits[pattern.count_ones() as usize - 1] != 0;
            }
        } else {
            mask[0] = true;
            let shift = amount * NUM_SIGNAL_STRENGTH_CHOICES;
            if self
                .sidecar_strength_bits
                .get(action)
                .is_some_and(|bits| (*bits >> shift) & 0x1f != 0)
            {
                for pattern in [1, 2, 4, 8] {
                    mask[pattern] = true;
                }
            }
        }
        mask
    }

    pub fn signal_strength_mask(
        &self,
        action: usize,
        amount: usize,
        signal: usize,
    ) -> [bool; NUM_SIGNAL_STRENGTH_CHOICES] {
        if signal == 0 {
            return [true, false, false, false, false];
        }
        let bits = if action == NUM_ACTIONS - 1 && signal < NUM_SIGNAL_CHOICES {
            self.explicit_signal_strength_bits[signal.count_ones() as usize - 1]
        } else {
            self.sidecar_strength_bits.get(action).map_or(0, |bits| {
                ((*bits >> (amount * NUM_SIGNAL_STRENGTH_CHOICES)) & 0x1f) as u8
            })
        };
        std::array::from_fn(|choice| bits & (1 << choice) != 0)
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.data.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_interface::randomness::PrivateRandom;
    use blob_interface::reference_mind::{
        CurrentTileObservation, LocalObservation, ReferenceActionSpace, ReferenceSelfState,
        EFFORT_GENTLE_BIT, EFFORT_STANDARD_BIT,
    };

    fn input() -> ReferenceMindInput {
        ReferenceMindInput {
            self_state: ReferenceSelfState {
                core_mass: 10,
                assimilated_energy: 150,
                gut_energy: 4,
                metabolism_remainder: 100,
                carried_material_mass: 0,
                marker: 0,
                guarded: false,
                last_outcome: None,
            },
            current_tile: CurrentTileObservation {
                elevation: 2,
                plant_energy: 100,
                plant_capacity: 200,
                plant_growth_rate: 10,
                loose_energy: 0,
                diffuse_energy: 0,
                signal_energy: [1, 2, 3, 4],
            },
            slots: vec![LocalObservation {
                slot: 0,
                dx: -1,
                dy: -1,
                distance_cost_q10: 1448,
                reachable: true,
                elevation: Some(1),
                plant_energy: Some(50),
                plant_capacity: Some(150),
                plant_growth_rate: Some(8),
                loose_energy: Some(0),
                diffuse_energy: Some(0),
                signal_energy: Some([4, 3, 2, 1]),
                neighbor: None,
            }],
            action_space: ReferenceActionSpace {
                wait_enabled: true,
                guard_enabled: true,
                consume_enabled: true,
                excavate_enabled: true,
                deposit_terrain_enabled: true,
                signal_enabled: true,
                move_targets: 1,
                attack_targets: 1,
                split_targets: 1,
                regurgitate_targets: 1,
                effort_mask: EFFORT_GENTLE_BIT | EFFORT_STANDARD_BIT,
                max_consume_amount: 16,
                gut_capacity: 64,
                max_private_memory_bytes: 2048,
                minimum_survival_energy: 1,
                child_core_mass: 10,
                metabolism_rate_numerator: 1,
                metabolism_rate_denominator: 1024,
                terrain_mass_per_elevation: 10,
                signal_emission_cost: 1,
                effort_cost_numerators: [1, 1, 2],
                effort_cost_denominators: [2, 1, 1],
                move_effort_base: 1,
                move_mass_units_per_effort: 100,
                attack_effort_base: 2,
                guard_effort_base: 1,
                consume_effort_base: 1,
                split_effort_base: 2,
                regurgitate_effort_base: 1,
                excavate_effort_base: 2,
                deposit_terrain_effort_base: 2,
            },
            private_memory: Vec::new(),
            randomness: PrivateRandom::ZERO,
        }
    }

    #[test]
    fn reference_projection_has_fixed_shape_and_finite_values() {
        let observation = Observation::from_reference(&input());
        assert_eq!(observation.data.len(), OBS_DIM);
        assert!(observation.data.iter().all(|value| value.is_finite()));
        assert!(observation.action_mask.iter().any(|allowed| *allowed));
    }

    #[test]
    fn private_random_block_is_available_as_exact_cell_private_features() {
        let baseline = Observation::from_reference(&input());
        let mut randomized_input = input();
        let bytes = std::array::from_fn(|index| (index * 7 + 3) as u8);
        randomized_input.randomness = PrivateRandom::from_bytes(bytes);
        let randomized = Observation::from_reference(&randomized_input);

        assert_eq!(
            &baseline.data[..OBS_RANDOMNESS_FEATURE_START],
            &randomized.data[..OBS_RANDOMNESS_FEATURE_START]
        );
        for (index, byte) in bytes.into_iter().enumerate() {
            assert_eq!(
                randomized.data[OBS_RANDOMNESS_FEATURE_START + index],
                f32::from(byte) / f32::from(u8::MAX)
            );
        }
        assert_eq!(
            &baseline.data[HEADER_FEATURES..],
            &randomized.data[HEADER_FEATURES..]
        );
    }

    #[test]
    fn unavailable_slots_are_zero_padded() {
        let observation = Observation::from_reference(&input());
        let unavailable = HEADER_FEATURES + SLOT_FEATURES;
        assert!(observation.data[unavailable..unavailable + SLOT_FEATURES]
            .iter()
            .all(|value| *value == 0.0));
    }

    #[test]
    fn slot_action_masks_are_observable() {
        let observation = Observation::from_reference(&input());
        let base = HEADER_FEATURES;
        assert_eq!(observation.data[base + 20], 1.0);
        assert_eq!(observation.data[base + 21], 1.0);
        assert_eq!(observation.data[base + 22], 1.0);
        assert_eq!(observation.data[base + 23], 1.0);
    }
}
