//! Bounded observation visibility, scratch reuse, ABI and anonymity contracts.

use blob_engine::resolution::{
    ActionRequest, BoundaryRule, CellKey, EffortTier, LocalSlot, NeighborhoodSpec,
    ObservationMasks, ReferenceRuleset, ReferenceSimulation, SimTime, SlotMask, TargetingAction,
    TileIndex,
};
use blob_interface::randomness::PrivateRandom;
use blob_interface::reference_mind::{ReferenceActivity, ReferenceProgress};
use blob_interface::reference_mind_converter::{
    ReferenceMindLimits, capnp_to_reference_mind_input, reference_mind_input_to_capnp,
};

fn byte(data: &[u8], index: usize) -> u8 {
    data.get(index).copied().unwrap_or(0)
}

fn mask(bits: u8, count: usize) -> SlotMask {
    SlotMask::from_slots(
        (0..count as u8)
            .filter(|slot| bits & (1 << slot) != 0)
            .map(LocalSlot),
    )
}

/// At most 25 tiles, eight relative slots and 16 bytes of private cell memory.
pub fn check(data: &[u8]) {
    check_with_coverage(data);
}

#[derive(Default, Debug)]
struct Coverage {
    activities: u8,
    progress: u8,
    accepted: u16,
    rejected: u16,
}

fn check_with_coverage(data: &[u8]) -> Coverage {
    let mut coverage = Coverage::default();
    if data.len() > 256 {
        return coverage;
    }
    let width = 1 + usize::from(byte(data, 0) % 5);
    let height = 1 + usize::from(byte(data, 1) % 5);
    let boundary = if byte(data, 2) & 1 == 0 {
        BoundaryRule::Bounded
    } else {
        BoundaryRule::Wrap
    };
    let count = usize::from(byte(data, 4) % 9);
    let mut spec = NeighborhoodSpec::moore_8(boundary);
    spec.slots.truncate(count);
    let masks: Vec<_> = (0..11)
        .map(|index| mask(byte(data, 5 + index), count))
        .collect();
    spec.observations = ObservationMasks {
        occupancy: masks[0],
        marker: masks[1],
        apparent_mass: masks[2],
        activity: masks[3],
        terrain: masks[4],
        energy: masks[5],
        signal: masks[6],
    };
    let actions = [
        TargetingAction::Move,
        TargetingAction::Attack,
        TargetingAction::Split,
        TargetingAction::Regurgitate,
    ];
    for (index, action) in actions.iter().enumerate() {
        spec.set_target_mask(*action, masks[7 + index]);
    }
    let rules = ReferenceRuleset {
        neighborhood: spec.clone(),
        diffusion_targets: SlotMask::empty(),
        apparent_mass_bucket_width: 1 + u64::from(byte(data, 16)),
        max_private_memory_bytes: 16,
        ..ReferenceRuleset::default()
    };
    let mut simulation = ReferenceSimulation::new(width, height, rules.clone()).unwrap();
    let origin = usize::from(byte(data, 3)) % (width * height);
    let observer = simulation
        .add_cell(TileIndex(origin), 20, 1000, 0x1234)
        .unwrap();
    for index in 0..width * height {
        let start = 17 + index * 8;
        let flag = byte(data, start);
        if index != origin && flag & 1 != 0 {
            simulation
                .add_cell(
                    TileIndex(index),
                    1 + u64::from(byte(data, start + 1)),
                    1000,
                    u32::from(byte(data, start + 2)),
                )
                .unwrap();
        }
        let tile = simulation.tile_state_mut(TileIndex(index)).unwrap();
        tile.elevation = i16::from(byte(data, start + 3) as i8);
        tile.plant_energy = u64::from(byte(data, start + 4));
        tile.plant_capacity = tile.plant_energy + 1;
        tile.plant_growth_rate = u64::from(byte(data, start + 5));
        tile.loose_energy = u64::from(byte(data, start + 6));
        tile.diffuse_energy = u64::from(byte(data, start + 7));
        tile.signal_energy =
            std::array::from_fn(|channel| u64::from(byte(data, start + 4 + channel)));
    }
    let mut state = simulation.canonical_state();
    for (_, cell) in &mut state.cells {
        let start = 17 + cell.position.0 * 8;
        cell.private_memory =
            vec![byte(data, start + 2); usize::from(byte(data, start + 1) % 17)].into();
        cell.gut_energy = u64::from(byte(data, start + 5));
        cell.carried_material_mass = u64::from(byte(data, start + 6));
        cell.guarded = byte(data, start) & 2 != 0;
    }
    let mut simulation =
        ReferenceSimulation::from_canonical_state(width, height, rules.clone(), state).unwrap();
    for (key, cell) in simulation.canonical_state().cells {
        let start = 17 + cell.position.0 * 8;
        let flag = byte(data, start);
        if key != observer && flag & 4 != 0 {
            let kind = (flag >> 4) % 10;
            let target = LocalSlot(byte(data, start + 2) % 12);
            let amount = u64::from(byte(data, start + 4));
            let effort = EffortTier::Standard;
            let action = match kind {
                0 => ActionRequest::Wait,
                1 => ActionRequest::Move { target, effort },
                2 => ActionRequest::Attack {
                    target,
                    effort,
                    payload: amount,
                },
                3 => ActionRequest::Guard { effort },
                4 => ActionRequest::Consume { amount },
                5 => ActionRequest::Split {
                    target,
                    child_allocation: rules.child_core_mass + amount,
                    marker: u32::from(byte(data, start + 2)),
                    private_memory: vec![byte(data, start + 1); 16],
                },
                6 => ActionRequest::Regurgitate { target, amount },
                7 => ActionRequest::Signal {
                    amounts: [rules.signal_emission_cost, 0, 0, 0],
                },
                8 => ActionRequest::Excavate,
                _ => ActionRequest::DepositTerrain,
            };
            if simulation.commit_action(key, action).unwrap().accepted {
                coverage.accepted |= 1 << kind;
            } else {
                coverage.rejected |= 1 << kind;
            }
        }
    }
    // Stay before every event, including metabolic death, so no commitment
    // resolves and the small-world bound cannot grow. The spare selector lies
    // after the 25 tile records and before the private-random bytes.
    if let Some(event) = simulation.next_event_time().unwrap() {
        let duration = event.0;
        let middle = duration.div_ceil(3);
        let late = (duration * 2).div_ceil(3);
        let elapsed = [
            0,
            middle.saturating_sub(1),
            middle,
            late.saturating_sub(1),
            late,
            duration.saturating_sub(1),
        ][usize::from(byte(data, 217) % 6)]
        .min(duration.saturating_sub(1));
        simulation.advance_clock_to(SimTime(elapsed)).unwrap();
    }
    let randomness =
        PrivateRandom::from_bytes(std::array::from_fn(|index| byte(data, 224 + index)));
    let input = simulation
        .reference_mind_input(observer, randomness)
        .unwrap();
    let canonical = simulation.canonical_state();
    let self_cell = simulation.cell(observer).unwrap();
    assert_eq!(input.randomness, randomness);
    assert_eq!(
        input.private_memory.as_slice(),
        self_cell.private_memory.as_ref()
    );
    assert_eq!(input.self_state.marker, self_cell.marker);
    assert_eq!(
        input.current_tile.signal_energy,
        canonical.tiles[origin].signal_energy
    );
    assert_eq!(input.slots.len(), count);
    assert_eq!(
        [
            input.action_space.move_targets,
            input.action_space.attack_targets,
            input.action_space.split_targets,
            input.action_space.regurgitate_targets
        ],
        [
            masks[7].bits(),
            masks[8].bits(),
            masks[9].bits(),
            masks[10].bits()
        ]
    );
    for (index, observed) in input.slots.iter().enumerate() {
        let slot = LocalSlot(index as u8);
        let offset = spec.slots[index];
        let x = (origin % width) as i32 + i32::from(offset.dx);
        let y = (origin / width) as i32 + i32::from(offset.dy);
        let target = if boundary == BoundaryRule::Wrap {
            Some(y.rem_euclid(height as i32) as usize * width + x.rem_euclid(width as i32) as usize)
        } else if x >= 0 && y >= 0 && x < width as i32 && y < height as i32 {
            Some(y as usize * width + x as usize)
        } else {
            None
        };
        let tile = target.map(|target| &canonical.tiles[target]);
        assert_eq!(
            (
                observed.slot,
                observed.dx,
                observed.dy,
                observed.distance_cost_q10
            ),
            (index as u8, offset.dx, offset.dy, offset.distance_cost_q10)
        );
        assert_eq!(observed.reachable, target.is_some());
        assert_eq!(
            observed.elevation,
            tile.filter(|_| masks[4].contains(slot))
                .map(|tile| tile.elevation)
        );
        let energy = tile.filter(|_| masks[5].contains(slot));
        assert_eq!(observed.plant_energy, energy.map(|tile| tile.plant_energy));
        assert_eq!(
            observed.plant_capacity,
            energy.map(|tile| tile.plant_capacity)
        );
        assert_eq!(
            observed.plant_growth_rate,
            energy.map(|tile| tile.plant_growth_rate)
        );
        assert_eq!(observed.loose_energy, energy.map(|tile| tile.loose_energy));
        assert_eq!(
            observed.diffuse_energy,
            energy.map(|tile| tile.diffuse_energy)
        );
        assert_eq!(
            observed.signal_energy,
            tile.filter(|_| masks[6].contains(slot))
                .map(|tile| tile.signal_energy)
        );
        // Any permitted cell cue reveals presence; occupancy is not an umbrella
        // permission for separately permitted marker/mass/activity observations.
        let neighbor = tile
            .and_then(|tile| tile.occupant)
            .filter(|key| *key != observer)
            .filter(|_| masks[..4].iter().any(|mask| mask.contains(slot)))
            .map(|key| simulation.cell(key).unwrap());
        assert_eq!(observed.neighbor.is_some(), neighbor.is_some());
        if let (Some(actual), Some(cell)) = (observed.neighbor, neighbor) {
            assert_eq!(
                actual.marker,
                masks[1].contains(slot).then_some(cell.marker)
            );
            let mass = u128::from(cell.core_mass)
                + u128::from(cell.assimilated_energy)
                + u128::from(cell.gut_energy)
                + u128::from(cell.carried_material_mass)
                + cell
                    .pending_action
                    .as_ref()
                    .map_or(0, |pending| u128::from(pending.payload_escrow));
            assert_eq!(
                actual.apparent_mass_bucket,
                masks[2].contains(slot).then_some(
                    (mass / u128::from(rules.apparent_mass_bucket_width)).min(255) as u8
                )
            );
            // Derive cues from the canonical commitment, not the projection's
            // helper. Rejections conceal the requested action behind OtherBusy.
            let (activity, progress) = match &cell.pending_action {
                None => (
                    if cell.guarded {
                        ReferenceActivity::Guarding
                    } else {
                        ReferenceActivity::Ready
                    },
                    None,
                ),
                Some(pending) => {
                    let activity = if pending.rejection.is_some() {
                        ReferenceActivity::OtherBusy
                    } else {
                        match pending.request {
                            ActionRequest::Move { .. } => ReferenceActivity::Moving,
                            ActionRequest::Attack { .. } => ReferenceActivity::AttackWindup,
                            ActionRequest::Guard { .. } => ReferenceActivity::Guarding,
                            ActionRequest::Consume { .. } => ReferenceActivity::Feeding,
                            ActionRequest::Split { .. } => ReferenceActivity::Splitting,
                            ActionRequest::Excavate | ActionRequest::DepositTerrain => {
                                ReferenceActivity::ManipulatingTerrain
                            }
                            ActionRequest::Wait
                            | ActionRequest::Regurgitate { .. }
                            | ActionRequest::Signal { .. } => ReferenceActivity::OtherBusy,
                        }
                    };
                    let duration = u128::from(pending.completes_at.0 - pending.started_at.0);
                    let elapsed = u128::from(canonical.now.0 - pending.started_at.0);
                    let progress = if 3 * elapsed < duration {
                        ReferenceProgress::Early
                    } else if 3 * elapsed < 2 * duration {
                        ReferenceProgress::Middle
                    } else {
                        ReferenceProgress::Late
                    };
                    (activity, Some(progress))
                }
            };
            assert_eq!(actual.activity, masks[3].contains(slot).then_some(activity));
            assert_eq!(
                actual.progress,
                masks[3].contains(slot).then_some(progress).flatten()
            );
            if let Some(activity) = actual.activity {
                coverage.activities |= 1 << activity as u8;
            }
            if let Some(progress) = actual.progress {
                coverage.progress |= 1 << progress as u8;
            }
        }
    }
    let limits = ReferenceMindLimits::default();
    let bytes = reference_mind_input_to_capnp(&input, limits).unwrap();
    assert_eq!(
        capnp_to_reference_mind_input(&bytes, limits).unwrap(),
        input
    );
    let batch = simulation.observation_batch();
    assert_eq!(
        batch.reference_mind_input(observer, randomness).unwrap(),
        input
    );

    // Pollute scratch from the full-visibility version of this world, then
    // require exact replacement including canonical bytes in both lookup paths.
    let mut visible_rules = rules.clone();
    visible_rules.neighborhood.observations = ObservationMasks::all(count);
    let visible =
        ReferenceSimulation::from_canonical_state(width, height, visible_rules, canonical.clone())
            .unwrap();
    let mut scratch = visible
        .reference_mind_input(observer, PrivateRandom::ZERO)
        .unwrap();
    scratch.private_memory.extend_from_slice(&[91, 92, 93]);
    let polluted = scratch.clone();
    simulation
        .reference_mind_input_into(observer, randomness, &mut scratch)
        .unwrap();
    assert_eq!(scratch, input);
    scratch = polluted;
    batch
        .reference_mind_input_into(observer, randomness, &mut scratch)
        .unwrap();
    assert_eq!(
        reference_mind_input_to_capnp(&scratch, limits).unwrap(),
        bytes
    );

    // Reassign every host identity and alter other cells' private memory. Neither
    // may change the observer's input after restoring pending canonical actions.
    let rename = |key: CellKey| CellKey(key.0 * 32 + 13);
    let mut renamed = canonical;
    for tile in &mut renamed.tiles {
        tile.occupant = tile.occupant.map(rename);
    }
    for (key, cell) in &mut renamed.cells {
        if *key != observer {
            cell.private_memory = vec![0x5a; 16].into();
        }
        *key = rename(*key);
    }
    renamed.next_cell_key = renamed.next_cell_key * 32 + 13;
    let renamed = ReferenceSimulation::from_canonical_state(width, height, rules, renamed).unwrap();
    let renamed_input = renamed
        .reference_mind_input(rename(observer), randomness)
        .unwrap();
    assert_eq!(
        reference_mind_input_to_capnp(&renamed_input, limits).unwrap(),
        bytes
    );
    coverage
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_activity_and_progress_with_accepted_and_rejected_commitments() {
        let mut data = [0; 256];
        data[..5].copy_from_slice(&[2, 2, 1, 4, 8]);
        data[5..16].fill(0xff);
        // One visible northern neighbor; its east target is a separate tile.
        let start = 17 + 8;
        data[start..start + 8].copy_from_slice(&[5, 19, 4, 0, 16, 32, 20, 0]);
        let expected = [
            ReferenceActivity::OtherBusy,
            ReferenceActivity::Moving,
            ReferenceActivity::AttackWindup,
            ReferenceActivity::Guarding,
            ReferenceActivity::Feeding,
            ReferenceActivity::Splitting,
            ReferenceActivity::OtherBusy,
            ReferenceActivity::OtherBusy,
            ReferenceActivity::ManipulatingTerrain,
            ReferenceActivity::ManipulatingTerrain,
        ];
        let mut activities = 0;
        for kind in 0..10_u8 {
            data[start] = 5 | (kind << 4);
            for phase in 0..6_u8 {
                data[217] = phase;
                let coverage = check_with_coverage(&data);
                assert_eq!(
                    coverage.accepted,
                    1 << kind,
                    "kind={kind}, phase={phase}: {coverage:?}"
                );
                assert_eq!(coverage.rejected, 0);
                assert_eq!(coverage.activities, 1 << expected[usize::from(kind)] as u8);
                assert_eq!(
                    coverage.progress,
                    1 << (phase / 2),
                    "kind={kind}, phase={phase}: {coverage:?}"
                );
                activities |= coverage.activities;
            }
        }
        for guarded in [false, true] {
            data[start] = if guarded { 3 } else { 1 };
            let coverage = check_with_coverage(&data);
            assert_eq!(coverage.accepted | coverage.rejected, 0);
            assert_eq!(coverage.progress, 0);
            assert_eq!(
                coverage.activities,
                1 << if guarded {
                    ReferenceActivity::Guarding
                } else {
                    ReferenceActivity::Ready
                } as u8
            );
            activities |= coverage.activities;
        }
        assert_eq!(activities, u8::MAX);
        for (kind, mask_index) in [(1, 12), (2, 13), (5, 14), (6, 15)] {
            data[start] = 5 | (kind << 4);
            for phase in 0..6_u8 {
                data[217] = phase;
                // Both a valid-but-forbidden slot and an out-of-range slot.
                for invalid_slot in [false, true] {
                    data[start + 2] = if invalid_slot { 11 } else { 4 };
                    data[mask_index] = if invalid_slot { 0xff } else { 0 };
                    let coverage = check_with_coverage(&data);
                    assert_eq!(coverage.accepted, 0);
                    assert_eq!(coverage.rejected, 1 << kind);
                    assert_eq!(coverage.activities, 1 << ReferenceActivity::OtherBusy as u8);
                    assert_eq!(coverage.progress, 1 << (phase / 2));
                }
                data[mask_index] = 0xff;
            }
        }
    }

    #[test]
    fn observation_visibility_and_anonymity_contract() {
        for fields in 0..128_u8 {
            for (width, height, wrap, count) in
                [(0, 0, 1, 8), (4, 4, 0, 8), (2, 2, 1, 8), (1, 3, 0, 0)]
            {
                let mut data = [0xff; 256];
                data[..5].copy_from_slice(&[width, height, wrap, fields, count]);
                for field in 0..7 {
                    data[5 + field] = if fields & (1 << field) != 0 { 0xff } else { 0 };
                }
                check(&data);
            }
        }
        let mut random = 0x3912_aced_0987_u64;
        for _ in 0..256 {
            let data: [u8; 256] = std::array::from_fn(|_| {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                random as u8
            });
            check(&data);
        }
    }
}
