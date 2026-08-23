use std::collections::{BTreeMap, BTreeSet};

use blob_engine::resolution::{
    ActionRequest, BoundaryRule, CellKey, DurationRule, EffortTier, LocalSlot, OutcomeStatus,
    ReferenceRuleset, ReferenceSimulation, RejectReason, SimTime, SimulationState,
};

const NORTH: LocalSlot = LocalSlot(1);
const EAST: LocalSlot = LocalSlot(4);

fn uniform_rules() -> ReferenceRuleset {
    let uniform = DurationRule::new(1024, 0, 1);
    ReferenceRuleset {
        wait_duration: uniform,
        move_duration: uniform,
        attack_duration: uniform,
        consume_duration: uniform,
        split_duration: uniform,
        regurgitate_duration: uniform,
        excavate_duration: uniform,
        deposit_terrain_duration: uniform,
        digestion_rate_numerator: 0,
        metabolism_rate_numerator: 0,
        diffusion_rate_numerator: 0,
        ..ReferenceRuleset::default()
    }
}

#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn range(&mut self, upper: u64) -> u64 {
        self.next() % upper
    }
}

fn initial_case(seed: u64) -> ReferenceSimulation {
    let mut simulation = ReferenceSimulation::new(6, 6, uniform_rules()).unwrap();
    let mut rng = SplitMix64(seed);
    for tile in 0..36 {
        let state = simulation
            .tile_state_mut(blob_engine::resolution::TileIndex(tile))
            .unwrap();
        state.plant_energy = rng.range(20);
        state.plant_growth_rate = rng.range(4);
        if state.plant_growth_rate > 0 {
            state.plant_capacity = state.plant_energy + 1 + rng.range(20);
        }
        state.loose_energy = rng.range(10);
        state.diffuse_energy = rng.range(10);
    }
    for tile in (0_usize..36).filter(|tile| (*tile).is_multiple_of(2)) {
        simulation
            .add_cell(
                blob_engine::resolution::TileIndex(tile),
                10,
                300,
                tile as u32,
            )
            .unwrap();
    }
    simulation
}

fn action_for(rng: &mut SplitMix64, actor: CellKey) -> ActionRequest {
    let slot = LocalSlot(rng.range(8) as u8);
    match rng.range(6) {
        0 => ActionRequest::Wait,
        1 => ActionRequest::Move {
            target: slot,
            effort: EffortTier::Standard,
        },
        2 => ActionRequest::Attack {
            target: slot,
            effort: EffortTier::Standard,
            payload: 1 + rng.range(24),
        },
        3 => ActionRequest::Guard {
            effort: EffortTier::Standard,
        },
        4 => ActionRequest::Consume {
            amount: 1 + rng.range(24),
        },
        _ => ActionRequest::Split {
            target: slot,
            child_allocation: 20,
            marker: actor.0 as u32,
            private_memory: vec![actor.0 as u8, rng.next() as u8],
        },
    }
}

fn assert_state_invariants(state: &SimulationState) {
    let cells: BTreeMap<_, _> = state
        .cells
        .iter()
        .map(|(cell, state)| (*cell, state))
        .collect();
    let mut occupied_cells = BTreeSet::new();
    for (index, tile) in state.tiles.iter().enumerate() {
        assert!(tile.plant_growth_remainder < 1024);
        if tile.plant_growth_rate > 0 {
            assert!(tile.plant_energy <= tile.plant_capacity);
        }
        if let Some(cell) = tile.occupant {
            assert!(
                occupied_cells.insert(cell),
                "cell {cell:?} occupies more than one tile"
            );
            assert_eq!(
                cells.get(&cell).unwrap().position.0,
                index,
                "tile and cell position disagree"
            );
        }
    }
    assert_eq!(occupied_cells.len(), cells.len());
    for (cell, value) in cells {
        assert_eq!(state.tiles[value.position.0].occupant, Some(cell));
        assert!(cell.0 < state.next_cell_key);
        if let Some(pending) = &value.pending_action {
            assert!(pending.started_at <= state.now);
            assert!(pending.completes_at > pending.started_at);
        }
    }
}

#[test]
fn randomized_batches_are_order_independent_conservative_and_reversible() {
    for seed in 0..64_u64 {
        let mut forward = initial_case(seed);
        let mut reverse = forward.clone();
        let compiled = forward.compiled_ruleset_hash();
        let mut rng = SplitMix64(seed ^ 0xa5a5_5a5a_d3c4_b2e1);

        for _round in 0..6 {
            let actions: Vec<_> = forward
                .cells()
                .iter()
                .filter(|(_, cell)| cell.is_ready_at(forward.now()))
                .map(|(actor, _)| (*actor, action_for(&mut rng, *actor)))
                .collect();
            if actions.is_empty() {
                break;
            }
            let energy_before = forward.total_energy_equivalent();
            for (actor, action) in &actions {
                forward.commit_action(*actor, action.clone()).unwrap();
            }
            for (actor, action) in actions.iter().rev() {
                reverse.commit_action(*actor, action.clone()).unwrap();
            }

            let forward_report = forward.resolve_next_batch().unwrap();
            let reverse_report = reverse.resolve_next_batch().unwrap();
            assert_eq!(forward_report, reverse_report, "seed {seed}");
            assert_eq!(forward.canonical_state(), reverse.canonical_state());
            assert_eq!(forward.total_energy_equivalent(), energy_before);
            assert_eq!(forward_report.state_hash(), Some(forward.state_hash()));

            let post = forward.canonical_state();
            let mut pre = post.clone();
            forward_report.delta.apply_backward(&mut pre).unwrap();
            assert_eq!(
                Some(pre.hash_with_compiled_ruleset(compiled)),
                forward_report.pre_state_hash()
            );
            forward_report.delta.apply_forward(&mut pre).unwrap();
            assert_eq!(pre, post);
            assert_state_invariants(&post);
        }
    }
}

#[test]
fn nonlethal_mass_loss_does_not_reschedule_a_pending_action() {
    let mut rules = uniform_rules();
    rules.move_duration = DurationRule::new(3072, 0, 1);
    let mut simulation = ReferenceSimulation::new(4, 2, rules).unwrap();
    let victim = simulation
        .add_cell(simulation.tile(1, 0).unwrap(), 10, 200, 0)
        .unwrap();
    let attacker = simulation
        .add_cell(simulation.tile(1, 1).unwrap(), 10, 200, 0)
        .unwrap();
    let move_receipt = simulation
        .commit_action(
            victim,
            ActionRequest::Move {
                target: EAST,
                effort: EffortTier::Standard,
            },
        )
        .unwrap();
    simulation
        .commit_action(
            attacker,
            ActionRequest::Attack {
                target: NORTH,
                effort: EffortTier::Standard,
                payload: 20,
            },
        )
        .unwrap();

    simulation.resolve_next_batch().unwrap();
    let pending = simulation
        .cell(victim)
        .unwrap()
        .pending_action
        .as_ref()
        .unwrap();
    assert_eq!(pending.completes_at, move_receipt.completes_at);
    assert_eq!(
        simulation.next_completion_time(),
        Some(move_receipt.completes_at)
    );
    let report = simulation.resolve_next_batch().unwrap();
    assert_eq!(report.outcomes[0].status, OutcomeStatus::Success);
    assert_eq!(
        simulation.cell(victim).unwrap().position,
        simulation.tile(2, 0).unwrap()
    );
}

#[test]
fn missed_attack_deposits_payload_without_losing_energy() {
    let mut simulation = ReferenceSimulation::new(2, 1, uniform_rules()).unwrap();
    let origin = simulation.tile(0, 0).unwrap();
    let target = simulation.tile(1, 0).unwrap();
    let attacker = simulation.add_cell(origin, 10, 100, 0).unwrap();
    let before = simulation.total_energy_equivalent();
    simulation
        .commit_action(
            attacker,
            ActionRequest::Attack {
                target: EAST,
                effort: EffortTier::Standard,
                payload: 15,
            },
        )
        .unwrap();
    let report = simulation.resolve_next_batch().unwrap();

    assert_eq!(report.outcomes[0].status, OutcomeStatus::Frustrated);
    assert_eq!(simulation.tile_state(target).unwrap().loose_energy, 15);
    assert_eq!(simulation.total_energy_equivalent(), before);
}

#[test]
fn representative_rejections_become_minimum_waits_without_payload_escrow() {
    let mut simulation = ReferenceSimulation::new(2, 1, uniform_rules()).unwrap();
    let actor = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 30, 0)
        .unwrap();
    let before = simulation.total_energy_equivalent();
    let receipt = simulation
        .commit_action(
            actor,
            ActionRequest::Attack {
                target: EAST,
                effort: EffortTier::Standard,
                payload: 30,
            },
        )
        .unwrap();

    assert!(!receipt.accepted);
    assert_eq!(receipt.rejection, Some(RejectReason::InsufficientEnergy));
    assert_eq!(receipt.payload_escrow, 0);
    assert_eq!(
        receipt.completes_at,
        SimTime(simulation.rules().time.decision_interval_floor)
    );
    let report = simulation.resolve_next_batch().unwrap();
    assert_eq!(
        report.outcomes[0].status,
        OutcomeStatus::Rejected(RejectReason::InsufficientEnergy)
    );
    assert_eq!(simulation.total_energy_equivalent(), before);
}

#[test]
fn one_cell_wrapping_targets_are_rejected_as_self_targets() {
    let rules = ReferenceRuleset {
        neighborhood: blob_engine::resolution::NeighborhoodSpec::moore_8(BoundaryRule::Wrap),
        ..uniform_rules()
    };
    let mut simulation = ReferenceSimulation::new(1, 1, rules).unwrap();
    let actor = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let receipt = simulation
        .commit_action(
            actor,
            ActionRequest::Move {
                target: EAST,
                effort: EffortTier::Standard,
            },
        )
        .unwrap();
    assert_eq!(receipt.rejection, Some(RejectReason::TargetsSelf));
}
