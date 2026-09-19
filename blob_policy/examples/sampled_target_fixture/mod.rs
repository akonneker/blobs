//! Explicit untrained ABI canary fixture; never a learned policy.
use blob_interface::reference_mind::*;
#[cfg(not(target_arch = "wasm32"))]
pub fn base_input() -> ReferenceMindInput {
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
        randomness: blob_interface::randomness::PrivateRandom::ZERO,
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn input(count: usize) -> ReferenceMindInput {
    assert!(count <= 32);
    let mut input = base_input();
    let template = input.slots[0].clone();
    let geometry: Vec<_> = if count <= 8 {
        vec![
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ]
    } else {
        (-3..=3)
            .flat_map(|dy| (-3..=3).map(move |dx| (dx, dy)))
            .filter(|position| *position != (0, 0))
            .collect()
    };
    input.slots = geometry
        .into_iter()
        .take(count)
        .enumerate()
        .map(|(index, (dx, dy))| {
            let mut slot = template.clone();
            slot.slot = index as u8;
            slot.dx = dx;
            slot.dy = dy;
            slot.distance_cost_q10 = 1024;
            slot.plant_energy = Some((index % 17) as u64);
            slot
        })
        .collect();
    let mask = if count == 32 {
        u32::MAX
    } else {
        (1_u32 << count) - 1
    };
    input.action_space.move_targets = mask;
    input.action_space.attack_targets = mask;
    input.action_space.split_targets = mask;
    input.action_space.regurgitate_targets = mask;
    input
}

pub fn logits(input: &ReferenceMindInput) -> [f32; 32] {
    let mut scores = [0.; 32];
    for slot in &input.slots {
        if let Some(score) = scores.get_mut(usize::from(slot.slot)) {
            *score = (slot.plant_energy.unwrap_or(0).min(32) as f32 - 8.) / 4.;
        }
    }
    scores
}

/// An untrained visible-food utility exercises the decoder; the memory payload
/// reports every quantized weight for exact ABI/native/WASM numeric comparison.
/// This is a test Mind, not an exported learned model or ecological baseline.
pub fn decide(input: &ReferenceMindInput) -> Result<ReferenceMindDecision, String> {
    use blob_policy::{action::*, sampling::TargetDistribution};
    let distribution =
        TargetDistribution::from_input(input, PolicyActionKind::Move, &logits(input))?;
    let action = match distribution.sample(input.randomness.sample_u64(0)) {
        Some(target) => {
            let effort_mask = policy_effort_mask(
                &action_mask(input),
                PolicyActionKind::Move.index(),
                usize::from(target),
            );
            let effort = effort_mask
                .iter()
                .position(|allowed| *allowed)
                .ok_or("sampled target has no legal effort")?;
            let flat = compose_policy_action(HierarchicalActionChoice {
                kind: PolicyActionKind::Move.index(),
                target: usize::from(target),
                effort,
            })
            .ok_or("cannot compose sampled move")?;
            decode_action(flat, input).action
        }
        None => ReferenceMindAction::Wait,
    };
    let mut memory = b"SAMPLE01".to_vec();
    for (&slot, weight) in distribution
        .canonical_slots()
        .iter()
        .zip(distribution.categorical().weights())
    {
        memory.push(slot);
        memory.extend(weight.to_le_bytes());
    }
    if memory.len() > input.action_space.max_private_memory_bytes as usize {
        return Err("canary diagnostic memory exceeds observation budget".into());
    }
    Ok(ReferenceMindDecision {
        action,
        signal: None,
        memory_update: ReferenceMemoryUpdate::Replace(memory),
    })
}
