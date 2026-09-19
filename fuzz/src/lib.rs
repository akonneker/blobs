//! Bounded semantic harness shared by coverage-guided and deterministic runs.

pub mod movement;
pub mod neighborhood;
pub mod observations;
pub mod wasm_admission;

use blob_engine::resolution::{
    ActionRequest, BatchReport, BoundaryRule, CellKey, CheckpointLimits, DecisionCommitment,
    DurationRule, EffortTier, LocalSlot, NeighborhoodSpec, ReferenceCheckpoint, ReferenceRuleset,
    ReferenceSimulation, SimTime, SimulationDelta, SimulationState,
};
use blob_interface::reference_mind::{ReferenceMemoryUpdate, ReferenceSignalEmission};

fn row_command(command: &[u8], index: usize, mixed: bool) -> [u8; 8] {
    let mut row: [u8; 8] = command.try_into().unwrap();
    if mixed {
        let index = index as u8; // At most 25 cells.
        row[2] = (command[2] % 10 + index) % 10;
        row[3] = command[3].wrapping_add(index);
        row[4] = command[4].wrapping_add(index);
        row[5] = command[5].wrapping_add(index * 3);
        row[7] ^= index;
    }
    row
}

/// Compare atomic admission with individually committing into a disposable
/// clone. A later fatal error may mutate that oracle, but must not mutate either
/// real branch. Semantic rejections remain successful minimum-Wait commitments.
fn check_ordered_batch(
    serial: &mut ReferenceSimulation,
    parallel: &mut ReferenceSimulation,
    decisions: &[DecisionCommitment],
) {
    let before = serial.canonical_state();
    let mut oracle = serial.clone();
    let ordered = decisions
        .windows(2)
        .all(|pair| pair[0].actor < pair[1].actor);
    let expected = ordered.then(|| {
        decisions
            .iter()
            .map(|decision| {
                oracle.commit_memory_update_with_signal(
                    decision.actor,
                    decision.request.clone(),
                    decision.signal,
                    decision.memory_update.clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()
    });
    let actual = serial.commit_decisions_ordered(decisions);
    assert_eq!(actual, parallel.commit_decisions_ordered(decisions));
    if let Some(expected) = expected {
        assert_eq!(actual, expected);
    } else {
        assert!(actual.is_err(), "unordered or duplicate actors must fail");
    }
    if actual.is_ok() {
        assert_eq!(serial.canonical_state(), oracle.canonical_state());
        assert_eq!(serial.state_hash(), oracle.state_hash());
    } else {
        assert_eq!(serial.canonical_state(), before);
        assert_eq!(parallel.canonical_state(), before);
    }
}

fn action(command: &[u8]) -> ActionRequest {
    let target = LocalSlot(command[3] % 12); // Includes invalid slots.
    let effort =
        [EffortTier::Low, EffortTier::Standard, EffortTier::High][usize::from(command[4] % 3)];
    let amount = u64::from(command[5]);
    match command[2] % 10 {
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
            child_allocation: amount,
            marker: u32::from(command[6]),
            private_memory: vec![command[7]; usize::from(command[6] % 17)],
        },
        6 => ActionRequest::Regurgitate { target, amount },
        7 => ActionRequest::Signal {
            amounts: [amount, 0, 0, 0],
        },
        8 => ActionRequest::Excavate,
        _ => ActionRequest::DepositTerrain,
    }
}

fn check_state(simulation: &ReferenceSimulation, total: u128) -> SimulationState {
    let state = simulation.canonical_state();
    assert_eq!(simulation.total_energy_equivalent(), total);
    assert_eq!(
        simulation.state_hash(),
        state.hash_with_compiled_ruleset(simulation.compiled_ruleset_hash())
    );
    assert!(state.cells.windows(2).all(|pair| pair[0].0 < pair[1].0));
    for (key, cell) in &state.cells {
        assert!(key.0 < state.next_cell_key);
        assert_eq!(state.tiles[cell.position.0].occupant, Some(*key));
    }
    for (index, tile) in state.tiles.iter().enumerate() {
        if let Some(key) = tile.occupant {
            assert_eq!(simulation.cell(key).unwrap().position.0, index);
        }
    }
    state
}

fn check_report(report: &BatchReport, simulation: &ReferenceSimulation) {
    let after = simulation.canonical_state();
    let mut reversed = after.clone();
    report.delta.apply_backward(&mut reversed).unwrap();
    // Reports begin after passive advancement, not at the previous clock time.
    assert_eq!(
        Some(reversed.hash_with_compiled_ruleset(simulation.compiled_ruleset_hash())),
        report.pre_state_hash()
    );
    report.delta.apply_forward(&mut reversed).unwrap();
    assert_eq!(reversed, after);
    assert_eq!(report.state_hash(), Some(simulation.state_hash()));
}

/// Up to 25 tiles, 64 commands, and 16 private-memory bytes per decision.
/// A header chooses topology, passive rates and equal/variable action durations;
/// eight-byte commands choose decisions, batched decisions, events, time or restore.
pub fn resolver_commands(data: &[u8]) {
    if data.len() < 4 {
        return;
    }
    let width = 2 + usize::from(data[0] % 4);
    let height = 2 + usize::from(data[1] % 4);
    let mut rules = ReferenceRuleset {
        neighborhood: NeighborhoodSpec::moore_8(if data[2] & 1 == 0 {
            BoundaryRule::Bounded
        } else {
            BoundaryRule::Wrap
        }),
        metabolism_rate_numerator: u64::from(data[2] >> 1 & 1),
        diffusion_rate_numerator: u64::from(data[2] >> 2 & 1),
        max_private_memory_bytes: 16,
        ..ReferenceRuleset::default()
    };
    if data[2] & 8 != 0 {
        let duration = DurationRule::new(1024, 0, 1);
        rules.wait_duration = duration;
        rules.move_duration = duration;
        rules.attack_duration = duration;
        rules.consume_duration = duration;
        rules.split_duration = duration;
        rules.regurgitate_duration = duration;
        rules.excavate_duration = duration;
        rules.deposit_terrain_duration = duration;
    }
    let mut serial = ReferenceSimulation::new(width, height, rules).unwrap();
    for index in 0..width * height {
        let tile = serial.tile(index % width, index / width).unwrap();
        let field = serial.tile_state_mut(tile).unwrap();
        field.loose_energy = u64::from(data[3]);
        field.diffuse_energy = u64::from(data[3]);
        field.plant_capacity = 256;
        field.plant_growth_rate = u64::from(data[2] >> 4 & 1);
        if index % 2 == 0 {
            serial
                .add_cell(tile, 10, 64 + u64::from(data[3]), index as u32)
                .unwrap();
        }
    }
    if data[2] & 32 != 0 {
        // Sparse historical keys put live pages, empty branches and the birth
        // frontier in the COW tree without requiring a long warm-up sequence.
        let mut state = serial.canonical_state();
        for (index, (key, cell)) in state.cells.iter_mut().enumerate() {
            *key = CellKey(index as u64 * 32);
            state.tiles[cell.position.0].occupant = Some(*key);
        }
        state.next_cell_key = 1024;
        serial =
            ReferenceSimulation::from_canonical_state(width, height, serial.rules().clone(), state)
                .unwrap();
    }
    serial.set_passive_parallel_thresholds(None, None);
    let mut parallel = serial.clone();
    parallel.set_passive_parallel_thresholds(Some(2), Some(2));
    let total = serial.total_energy_equivalent();
    check_state(&serial, total);
    for command in data[4..].chunks_exact(8).take(64) {
        let before = serial.canonical_state();
        match command[0] % 6 {
            0 => {
                // Usually choose a live cell; also admit unknown and busy actors.
                let key = if command[1] & 128 == 0 && !before.cells.is_empty() {
                    before.cells[usize::from(command[1]) % before.cells.len()].0
                } else {
                    CellKey(u64::from(command[1]))
                };
                let memory = vec![command[7]; usize::from(command[6] % 18)];
                assert_eq!(
                    serial.commit_decision(key, action(command), memory.clone()),
                    parallel.commit_decision(key, action(command), memory)
                );
            }
            1 => {
                // Each ready actor commits once; permuting these commitments must
                // leave both the snapshot and subsequent resolution unchanged.
                let keys: Vec<_> = before
                    .cells
                    .iter()
                    .filter(|(_, cell)| cell.is_ready_at(before.now))
                    .map(|(key, _)| *key)
                    .collect();
                let mut receipts = Vec::new();
                let requests: Vec<_> = keys
                    .iter()
                    .enumerate()
                    .map(|(index, _)| action(&row_command(command, index, data[2] & 64 != 0)))
                    .collect();
                for (key, request) in keys.iter().zip(&requests) {
                    receipts.push(serial.commit_action(*key, request.clone()));
                }
                for ((key, request), receipt) in keys.iter().zip(requests).zip(receipts).rev() {
                    assert_eq!(parallel.commit_action(*key, request), receipt);
                }
            }
            2 => {
                let left = serial.resolve_next_batch();
                let right = parallel.resolve_next_batch_parallel(2);
                assert_eq!(left, right);
                if let Ok(report) = left {
                    check_report(&report, &serial);
                }
            }
            3 => {
                let requested = if command[1] & 1 == 0 {
                    SimTime(before.now.0 + u64::from(command[5]) * 8)
                } else {
                    SimTime(before.now.0.saturating_sub(u64::from(command[5])))
                };
                assert_eq!(
                    serial.advance_clock_to(requested),
                    parallel.advance_clock_to(requested)
                );
            }
            4 => {
                let bytes = ReferenceCheckpoint::from_simulation(&serial).to_bytes();
                let checkpoint = ReferenceCheckpoint::from_bytes_with_limits(
                    &bytes,
                    CheckpointLimits {
                        max_checkpoint_bytes: 128 * 1024,
                        max_tiles: 25,
                        max_cells: 25,
                        max_cell_slots: 4096,
                        max_private_memory_bytes: 16,
                    },
                )
                .unwrap();
                assert_eq!(checkpoint.to_bytes(), bytes);
                // Restore only one branch so later events compare rebuilt indexes
                // with continuously maintained scheduling/hash metadata.
                serial = checkpoint.into_simulation().unwrap();
                serial.set_passive_parallel_thresholds(None, None);
            }
            _ => {
                let mut decisions: Vec<_> = before
                    .cells
                    .iter()
                    .filter(|(_, cell)| command[1] & 4 != 0 || cell.is_ready_at(before.now))
                    .enumerate()
                    .map(|(index, (actor, _))| {
                        let row = row_command(command, index, data[2] & 64 != 0);
                        DecisionCommitment {
                            actor: *actor,
                            request: action(&row),
                            signal: (command[4] & 128 != 0).then_some(ReferenceSignalEmission {
                                channel: row[7] % 6, // Includes invalid channels.
                                amount: u64::from(row[5]),
                            }),
                            memory_update: if command[6] & 128 != 0 {
                                ReferenceMemoryUpdate::Retain
                            } else {
                                ReferenceMemoryUpdate::Replace(vec![
                                    row[7];
                                    usize::from(command[6] % 18)
                                ])
                            },
                        }
                    })
                    .collect();
                match command[1] & 3 {
                    1 => decisions.reverse(),
                    2 if decisions.len() > 1 => {
                        let last = decisions.len() - 1;
                        decisions[last].actor = decisions[last - 1].actor;
                    }
                    3 => {
                        if let Some(last) = decisions.last_mut() {
                            last.actor = CellKey(before.next_cell_key + u64::from(command[1]));
                        }
                    }
                    _ => {}
                }
                check_ordered_batch(&mut serial, &mut parallel, &decisions);
            }
        }
        let after = check_state(&serial, total);
        assert_eq!(after, check_state(&parallel, total));
        let delta = SimulationDelta::between(&before, &after);
        assert_eq!(delta, SimulationDelta::between_parallel(&before, &after));
        let mut round_trip = before.clone();
        delta.apply_forward(&mut round_trip).unwrap();
        assert_eq!(round_trip, after);
        delta.apply_backward(&mut round_trip).unwrap();
        assert_eq!(round_trip, before);
        delta.apply_forward(&mut round_trip).unwrap();
        assert_eq!(round_trip, after);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_batch_rejections_are_atomic_and_semantic_rejections_commit() {
        let rules = ReferenceRuleset {
            max_private_memory_bytes: 16,
            ..ReferenceRuleset::default()
        };
        let mut baseline = ReferenceSimulation::new(3, 1, rules).unwrap();
        let actors: Vec<_> = (0..3)
            .map(|x| {
                let tile = baseline.tile(x, 0).unwrap();
                baseline.add_cell(tile, 10, 100, x as u32).unwrap()
            })
            .collect();
        let decisions = vec![
            DecisionCommitment {
                actor: actors[0],
                request: ActionRequest::Wait,
                signal: Some(ReferenceSignalEmission {
                    channel: 0,
                    amount: 8,
                }),
                memory_update: ReferenceMemoryUpdate::Replace(vec![1]),
            },
            DecisionCommitment {
                actor: actors[1],
                request: ActionRequest::Move {
                    target: LocalSlot(255),
                    effort: EffortTier::Standard,
                },
                signal: None,
                memory_update: ReferenceMemoryUpdate::Replace(vec![2]),
            },
            DecisionCommitment {
                actor: actors[2],
                request: ActionRequest::Guard {
                    effort: EffortTier::Standard,
                },
                signal: None,
                memory_update: ReferenceMemoryUpdate::Replace(vec![3]),
            },
        ];
        for failure in 0..5 {
            let mut invalid = decisions.clone();
            match failure {
                0 => invalid[2].memory_update = ReferenceMemoryUpdate::Replace(vec![9; 17]),
                1 => {
                    invalid[2].signal = Some(ReferenceSignalEmission {
                        channel: 4,
                        amount: 1,
                    })
                }
                2 => invalid[2].actor = CellKey(64),
                3 => invalid[2].actor = actors[1],
                _ => invalid.reverse(),
            }
            let mut serial = baseline.clone();
            let mut parallel = baseline.clone();
            check_ordered_batch(&mut serial, &mut parallel, &invalid);
            assert_eq!(serial.canonical_state(), baseline.canonical_state());
            assert_eq!(serial.state_hash(), baseline.state_hash());
            // Valid continuation proves the rejected batch left no scheduler,
            // signal, memory, escrow or hash residue.
            check_ordered_batch(&mut serial, &mut parallel, &decisions);
            for (index, actor) in actors.iter().enumerate() {
                let cell = serial.cell(*actor).unwrap();
                assert_eq!(cell.private_memory.as_ref(), &[index as u8 + 1]);
                assert_eq!(
                    cell.pending_action.as_ref().unwrap().rejection.is_some(),
                    index == 1
                );
            }
            assert_eq!(
                serial
                    .tile_state(serial.tile(0, 0).unwrap())
                    .unwrap()
                    .signal_energy[0],
                8
            );
            assert_eq!(
                serial.resolve_next_batch(),
                parallel.resolve_next_batch_parallel(2)
            );
            assert_eq!(
                check_state(&serial, baseline.total_energy_equivalent()),
                check_state(&parallel, baseline.total_energy_equivalent())
            );
        }
    }

    #[test]
    fn mixed_and_atomic_command_sequences() {
        for profile in [64, 71, 79, 95, 96, 103, 111, 127] {
            for kind in 0..10_u8 {
                for ordering in 0..8_u8 {
                    let mut data = vec![3, 3, profile, 100]; // 5x5, 13 actors.
                    for operation in [5, 4, 1, 5, 2, 4, 5, 2] {
                        data.extend_from_slice(&[operation, ordering, kind, 4, 1, 16, 8, 1]);
                    }
                    resolver_commands(&data);
                }
            }
        }
    }

    #[test]
    fn deterministic_command_sequences() {
        for seed in 0..128_u64 {
            let mut state = seed + 1;
            let data: Vec<_> = (0..516)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state as u8
                })
                .collect();
            resolver_commands(&data);
        }
    }
}
