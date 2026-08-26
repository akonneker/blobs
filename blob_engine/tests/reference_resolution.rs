use std::collections::BTreeMap;

use blob_engine::resolution::{
    ActionKind, ActionRequest, ActivityCue, BoundaryRule, CellKey, DecisionCommitment,
    DiagonalCornerRule, DurationRule, EffortTier, LocalSlot, NeighborhoodSpec, OutcomeStatus,
    ReferenceRuleset, ReferenceSimulation, RejectReason, SimTime,
};
use blob_interface::reference_mind::ReferenceSignalEmission;

const WEST: LocalSlot = LocalSlot(3);
const EAST: LocalSlot = LocalSlot(4);
const NORTH: LocalSlot = LocalSlot(1);

fn uniform_rules() -> ReferenceRuleset {
    let uniform = DurationRule::new(1024, 0, 1);
    ReferenceRuleset {
        neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Bounded),
        wait_duration: uniform,
        move_duration: uniform,
        attack_duration: uniform,
        consume_duration: uniform,
        split_duration: uniform,
        regurgitate_duration: uniform,
        excavate_duration: uniform,
        deposit_terrain_duration: uniform,
        metabolism_rate_numerator: 0,
        diffusion_rate_numerator: 0,
        ..ReferenceRuleset::default()
    }
}

fn metabolic_rules() -> ReferenceRuleset {
    ReferenceRuleset {
        digestion_rate_numerator: 0,
        metabolism_rate_numerator: 0,
        bite_capacity: 16,
        gut_capacity: 64,
        ..uniform_rules()
    }
}

fn move_toward(target: LocalSlot) -> ActionRequest {
    ActionRequest::Move {
        target,
        effort: EffortTier::Standard,
    }
}

fn statuses(report: &blob_engine::resolution::BatchReport) -> BTreeMap<CellKey, OutcomeStatus> {
    report
        .outcomes
        .iter()
        .map(|outcome| (outcome.actor, outcome.status))
        .collect()
}

#[test]
fn canonical_checkpoint_restore_validates_and_preserves_the_hash() {
    let mut original = ReferenceSimulation::new(3, 2, uniform_rules()).unwrap();
    original
        .add_cell(original.tile(1, 1).unwrap(), 10, 100, 7)
        .unwrap();
    let state = original.canonical_state();
    let restored =
        ReferenceSimulation::from_canonical_state(3, 2, uniform_rules(), state.clone()).unwrap();
    assert_eq!(restored.canonical_state(), state);
    assert_eq!(restored.state_hash(), original.state_hash());

    let mut invalid = state;
    invalid.tiles[0].occupant = Some(CellKey(99));
    assert!(ReferenceSimulation::from_canonical_state(3, 2, uniform_rules(), invalid).is_err());

    let mut stranded = original.canonical_state();
    stranded.cells[0].1.ready_at = SimTime(1);
    assert!(ReferenceSimulation::from_canonical_state(3, 2, uniform_rules(), stranded).is_err());

    let mut invalid_growth = original.canonical_state();
    invalid_growth.tiles[0].plant_growth_rate = 1;
    invalid_growth.tiles[0].plant_capacity = 1;
    invalid_growth.tiles[0].plant_growth_remainder = 1024;
    assert!(
        ReferenceSimulation::from_canonical_state(3, 2, uniform_rules(), invalid_growth).is_err()
    );

    let mut invalid_metabolism = original.canonical_state();
    invalid_metabolism.cells[0].1.metabolism_remainder =
        uniform_rules().metabolism_rate_denominator;
    assert!(
        ReferenceSimulation::from_canonical_state(3, 2, uniform_rules(), invalid_metabolism)
            .is_err()
    );
}

#[test]
fn convoy_uses_only_vacancies_from_the_completion_snapshot() {
    let mut simulation = ReferenceSimulation::new(6, 1, uniform_rules()).unwrap();
    let a = simulation
        .add_cell(simulation.tile(1, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let b = simulation
        .add_cell(simulation.tile(2, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let c = simulation
        .add_cell(simulation.tile(3, 0).unwrap(), 10, 100, 0)
        .unwrap();

    for actor in [a, b, c] {
        simulation.commit_action(actor, move_toward(EAST)).unwrap();
    }
    let report = simulation.resolve_next_batch().unwrap();
    let outcomes = statuses(&report);

    assert_eq!(outcomes[&a], OutcomeStatus::Frustrated);
    assert_eq!(outcomes[&b], OutcomeStatus::Frustrated);
    assert_eq!(outcomes[&c], OutcomeStatus::Success);
    assert_eq!(
        simulation.cell(a).unwrap().position,
        simulation.tile(1, 0).unwrap()
    );
    assert_eq!(
        simulation.cell(b).unwrap().position,
        simulation.tile(2, 0).unwrap()
    );
    assert_eq!(
        simulation.cell(c).unwrap().position,
        simulation.tile(4, 0).unwrap()
    );
}

#[test]
fn blocked_convoy_has_no_recursive_failure_walk() {
    let mut simulation = ReferenceSimulation::new(6, 1, uniform_rules()).unwrap();
    let a = simulation
        .add_cell(simulation.tile(1, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let b = simulation
        .add_cell(simulation.tile(2, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let c = simulation
        .add_cell(simulation.tile(3, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let _blocker = simulation
        .add_cell(simulation.tile(4, 0).unwrap(), 10, 100, 0)
        .unwrap();

    for actor in [a, b, c] {
        simulation.commit_action(actor, move_toward(EAST)).unwrap();
    }
    let outcomes = statuses(&simulation.resolve_next_batch().unwrap());

    assert_eq!(outcomes[&a], OutcomeStatus::Frustrated);
    assert_eq!(outcomes[&b], OutcomeStatus::Frustrated);
    assert_eq!(outcomes[&c], OutcomeStatus::Frustrated);
}

#[test]
fn simultaneous_claims_on_an_empty_tile_are_contested() {
    let mut simulation = ReferenceSimulation::new(3, 1, uniform_rules()).unwrap();
    let left = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let right = simulation
        .add_cell(simulation.tile(2, 0).unwrap(), 10, 100, 0)
        .unwrap();
    simulation.commit_action(left, move_toward(EAST)).unwrap();
    simulation.commit_action(right, move_toward(WEST)).unwrap();

    let outcomes = statuses(&simulation.resolve_next_batch().unwrap());
    assert_eq!(outcomes[&left], OutcomeStatus::Contested);
    assert_eq!(outcomes[&right], OutcomeStatus::Contested);
    assert_eq!(
        simulation
            .tile_state(simulation.tile(1, 0).unwrap())
            .unwrap()
            .occupant,
        None
    );
}

#[test]
fn later_claim_is_frustrated_by_an_earlier_completion() {
    let mut simulation = ReferenceSimulation::new(3, 1, uniform_rules()).unwrap();
    let first = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let second = simulation
        .add_cell(simulation.tile(2, 0).unwrap(), 10, 100, 0)
        .unwrap();
    simulation.commit_action(first, move_toward(EAST)).unwrap();
    simulation.advance_clock_to(SimTime(64)).unwrap();
    simulation.commit_action(second, move_toward(WEST)).unwrap();

    let first_report = simulation.resolve_next_batch().unwrap();
    assert_eq!(statuses(&first_report)[&first], OutcomeStatus::Success);
    let second_report = simulation.resolve_next_batch().unwrap();
    assert_eq!(statuses(&second_report)[&second], OutcomeStatus::Frustrated);
}

#[test]
fn swaps_are_frustrated_from_the_snapshot() {
    let mut simulation = ReferenceSimulation::new(2, 1, uniform_rules()).unwrap();
    let left = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let right = simulation
        .add_cell(simulation.tile(1, 0).unwrap(), 10, 100, 0)
        .unwrap();
    simulation.commit_action(left, move_toward(EAST)).unwrap();
    simulation.commit_action(right, move_toward(WEST)).unwrap();

    let outcomes = statuses(&simulation.resolve_next_batch().unwrap());
    assert_eq!(outcomes[&left], OutcomeStatus::Frustrated);
    assert_eq!(outcomes[&right], OutcomeStatus::Frustrated);
}

#[test]
fn split_and_move_use_the_same_occupancy_claim() {
    let mut simulation = ReferenceSimulation::new(3, 1, uniform_rules()).unwrap();
    let parent = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let mover = simulation
        .add_cell(simulation.tile(2, 0).unwrap(), 10, 100, 0)
        .unwrap();
    simulation
        .commit_action(
            parent,
            ActionRequest::Split {
                target: EAST,
                child_allocation: 20,
                marker: 7,
                private_memory: vec![1, 2, 3],
            },
        )
        .unwrap();
    simulation.commit_action(mover, move_toward(WEST)).unwrap();

    let report = simulation.resolve_next_batch().unwrap();
    let outcomes = statuses(&report);
    assert_eq!(outcomes[&parent], OutcomeStatus::Contested);
    assert_eq!(outcomes[&mover], OutcomeStatus::Contested);
    assert!(report.births.is_empty());
    assert_eq!(simulation.cells().len(), 2);
}

#[test]
fn death_does_not_create_a_same_batch_move_vacancy() {
    let mut simulation = ReferenceSimulation::new(3, 2, uniform_rules()).unwrap();
    let blocker = simulation
        .add_cell(simulation.tile(1, 0).unwrap(), 10, 10, 0)
        .unwrap();
    let mover = simulation
        .add_cell(simulation.tile(2, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let attacker = simulation
        .add_cell(simulation.tile(1, 1).unwrap(), 10, 100, 0)
        .unwrap();
    simulation.commit_action(mover, move_toward(WEST)).unwrap();
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

    let report = simulation.resolve_next_batch().unwrap();
    assert_eq!(statuses(&report)[&mover], OutcomeStatus::Frustrated);
    assert!(report.deaths.contains(&blocker));
    assert_eq!(
        simulation.cell(mover).unwrap().position,
        simulation.tile(2, 0).unwrap()
    );
    assert_eq!(
        simulation
            .tile_state(simulation.tile(1, 0).unwrap())
            .unwrap()
            .occupant,
        None
    );
}

#[test]
fn same_time_attack_chain_does_not_cancel_outgoing_damage() {
    let mut simulation = ReferenceSimulation::new(3, 1, uniform_rules()).unwrap();
    let a = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let b = simulation
        .add_cell(simulation.tile(1, 0).unwrap(), 10, 20, 0)
        .unwrap();
    let c = simulation
        .add_cell(simulation.tile(2, 0).unwrap(), 10, 100, 0)
        .unwrap();
    simulation
        .commit_action(
            a,
            ActionRequest::Attack {
                target: EAST,
                effort: EffortTier::Standard,
                payload: 20,
            },
        )
        .unwrap();
    simulation
        .commit_action(
            b,
            ActionRequest::Attack {
                target: EAST,
                effort: EffortTier::Standard,
                payload: 5,
            },
        )
        .unwrap();

    let c_before = simulation.cell(c).unwrap().assimilated_energy;
    let report = simulation.resolve_next_batch().unwrap();
    assert!(report.deaths.contains(&b));
    assert_eq!(statuses(&report)[&b], OutcomeStatus::Success);
    assert_eq!(c_before - simulation.cell(c).unwrap().assimilated_energy, 5);
}

#[test]
fn simultaneous_damage_uses_deterministic_largest_remainder_attribution() {
    let mut simulation = ReferenceSimulation::new(3, 3, uniform_rules()).unwrap();
    let west = simulation
        .add_cell(simulation.tile(0, 1).unwrap(), 10, 100, 0)
        .unwrap();
    let north = simulation
        .add_cell(simulation.tile(1, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let east = simulation
        .add_cell(simulation.tile(2, 1).unwrap(), 10, 100, 0)
        .unwrap();
    let victim = simulation
        .add_cell(simulation.tile(1, 1).unwrap(), 10, 3, 1)
        .unwrap();

    for (actor, target, payload) in [
        (west, EAST, 1_u64),
        (north, LocalSlot(6), 2),
        (east, WEST, 3),
    ] {
        simulation
            .commit_action(
                actor,
                ActionRequest::Attack {
                    target,
                    effort: EffortTier::Standard,
                    payload,
                },
            )
            .unwrap();
    }
    simulation
        .commit_action(
            victim,
            ActionRequest::Guard {
                effort: EffortTier::Standard,
            },
        )
        .unwrap();

    // Guard effort leaves two health-energy available. Six raw damage becomes
    // three post-guard damage, of which two is applied and one is overkill.
    let report = simulation.resolve_next_batch().unwrap();
    let damage = report
        .outcomes
        .iter()
        .filter_map(|outcome| {
            outcome
                .attack_damage
                .as_ref()
                .map(|damage| (outcome.actor, damage.clone()))
        })
        .collect::<BTreeMap<_, _>>();

    assert_eq!(damage.len(), 3);
    assert_eq!(
        (
            damage[&west].raw,
            damage[&west].mitigated,
            damage[&west].applied,
            damage[&west].overkill
        ),
        (1, 0, 1, 0)
    );
    assert_eq!(
        (
            damage[&north].raw,
            damage[&north].mitigated,
            damage[&north].applied,
            damage[&north].overkill
        ),
        (2, 1, 1, 0)
    );
    assert_eq!(
        (
            damage[&east].raw,
            damage[&east].mitigated,
            damage[&east].applied,
            damage[&east].overkill
        ),
        (3, 2, 0, 1)
    );
    assert!(damage.values().all(|damage| {
        damage.victim == victim
            && damage.target_was_guarded
            && damage.raw == damage.mitigated + damage.applied + damage.overkill
    }));
    assert_eq!(damage.values().map(|damage| damage.raw).sum::<u64>(), 6);
    assert_eq!(
        damage.values().map(|damage| damage.mitigated).sum::<u64>(),
        3
    );
    assert_eq!(damage.values().map(|damage| damage.applied).sum::<u64>(), 2);
    assert_eq!(
        damage.values().map(|damage| damage.overkill).sum::<u64>(),
        1
    );
}

#[test]
fn guard_persists_after_completion_until_another_action_is_committed() {
    let mut simulation = ReferenceSimulation::new(1, 1, uniform_rules()).unwrap();
    let actor = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 0)
        .unwrap();
    simulation
        .commit_action(
            actor,
            ActionRequest::Guard {
                effort: EffortTier::Standard,
            },
        )
        .unwrap();
    assert!(simulation.cell(actor).unwrap().guarded);
    simulation.resolve_next_batch().unwrap();
    assert!(simulation.cell(actor).unwrap().guarded);
    simulation
        .commit_action(actor, ActionRequest::Wait)
        .unwrap();
    assert!(!simulation.cell(actor).unwrap().guarded);
}

fn build_order_independence_case(reverse_commit_order: bool) -> ReferenceSimulation {
    let mut simulation = ReferenceSimulation::new(5, 2, uniform_rules()).unwrap();
    let left = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 1)
        .unwrap();
    let right = simulation
        .add_cell(simulation.tile(2, 0).unwrap(), 10, 100, 2)
        .unwrap();
    let attacker = simulation
        .add_cell(simulation.tile(4, 1).unwrap(), 10, 100, 3)
        .unwrap();
    let target = simulation
        .add_cell(simulation.tile(4, 0).unwrap(), 10, 100, 4)
        .unwrap();
    let mut actions = vec![
        (left, move_toward(EAST)),
        (right, move_toward(WEST)),
        (
            attacker,
            ActionRequest::Attack {
                target: NORTH,
                effort: EffortTier::Standard,
                payload: 20,
            },
        ),
        (
            target,
            ActionRequest::Guard {
                effort: EffortTier::Standard,
            },
        ),
    ];
    if reverse_commit_order {
        actions.reverse();
    }
    for (actor, action) in actions {
        simulation.commit_action(actor, action).unwrap();
    }
    simulation
}

#[test]
fn result_is_independent_of_commit_iteration_order() {
    let mut forward = build_order_independence_case(false);
    let mut reverse = build_order_independence_case(true);

    let forward_report = forward.resolve_next_batch().unwrap();
    let reverse_report = reverse.resolve_next_batch().unwrap();
    assert_eq!(forward_report, reverse_report);
    assert_eq!(forward.canonical_state(), reverse.canonical_state());
}

#[test]
fn adversarial_batch_conserves_energy_equivalent() {
    let mut simulation = build_order_independence_case(false);
    let before = simulation.total_energy_equivalent();
    simulation.resolve_next_batch().unwrap();
    assert_eq!(simulation.total_energy_equivalent(), before);
}

#[test]
fn greater_mass_increases_move_duration_and_effort() {
    let mut simulation = ReferenceSimulation::new(5, 1, ReferenceRuleset::default()).unwrap();
    let light = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 40, 0)
        .unwrap();
    let heavy = simulation
        .add_cell(simulation.tile(3, 0).unwrap(), 10, 240, 0)
        .unwrap();

    let light_receipt = simulation.commit_action(light, move_toward(EAST)).unwrap();
    let heavy_receipt = simulation.commit_action(heavy, move_toward(EAST)).unwrap();

    assert!(heavy_receipt.completes_at > light_receipt.completes_at);
    assert!(heavy_receipt.effort_spent > light_receipt.effort_spent);
    assert_eq!(
        light_receipt.completes_at.0 % simulation.rules().time.completion_bucket,
        0
    );
    assert_eq!(
        heavy_receipt.completes_at.0 % simulation.rules().time.completion_bucket,
        0
    );
}

#[test]
fn diagonal_corner_policy_is_applied_from_the_snapshot() {
    let mut rules = uniform_rules();
    rules.neighborhood.diagonal_corner_rule = DiagonalCornerRule::BlockIfEitherOrthogonalOccupied;
    let mut simulation = ReferenceSimulation::new(3, 3, rules).unwrap();
    let mover = simulation
        .add_cell(simulation.tile(1, 1).unwrap(), 10, 100, 0)
        .unwrap();
    let _east_blocker = simulation
        .add_cell(simulation.tile(2, 1).unwrap(), 10, 100, 0)
        .unwrap();
    simulation
        .commit_action(mover, move_toward(LocalSlot(2)))
        .unwrap();

    let report = simulation.resolve_next_batch().unwrap();
    assert_eq!(statuses(&report)[&mover], OutcomeStatus::Frustrated);
    assert_eq!(
        simulation.cell(mover).unwrap().position,
        simulation.tile(1, 1).unwrap()
    );
}

#[test]
fn successful_split_transfers_escrow_without_creating_energy() {
    let mut simulation = ReferenceSimulation::new(2, 1, uniform_rules()).unwrap();
    let parent = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let before = simulation.total_energy_equivalent();
    simulation
        .commit_action(
            parent,
            ActionRequest::Split {
                target: EAST,
                child_allocation: 30,
                marker: 9,
                private_memory: vec![4, 5, 6],
            },
        )
        .unwrap();

    let report = simulation.resolve_next_batch().unwrap();
    let (_, child) = report.births[0];
    assert_eq!(simulation.cell(child).unwrap().core_mass, 10);
    assert_eq!(simulation.cell(child).unwrap().assimilated_energy, 20);
    assert_eq!(simulation.cell(child).unwrap().marker, 9);
    assert_eq!(simulation.total_energy_equivalent(), before);
}

#[test]
fn earlier_death_interrupts_an_actor_bound_future_action() {
    let mut rules = uniform_rules();
    rules.split_duration = DurationRule::new(2048, 0, 1);
    let mut simulation = ReferenceSimulation::new(3, 2, rules).unwrap();
    let victim = simulation
        .add_cell(simulation.tile(1, 0).unwrap(), 10, 35, 0)
        .unwrap();
    let attacker = simulation
        .add_cell(simulation.tile(1, 1).unwrap(), 10, 100, 0)
        .unwrap();
    let before = simulation.total_energy_equivalent();
    simulation
        .commit_action(
            victim,
            ActionRequest::Split {
                target: EAST,
                child_allocation: 20,
                marker: 1,
                private_memory: Vec::new(),
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

    let report = simulation.resolve_next_batch().unwrap();
    let interrupted = report
        .outcomes
        .iter()
        .find(|outcome| outcome.actor == victim)
        .unwrap();
    assert_eq!(interrupted.status, OutcomeStatus::Interrupted);
    assert!(report.births.is_empty());
    assert!(simulation.cell(victim).is_none());
    assert_eq!(simulation.total_energy_equivalent(), before);
    assert_eq!(simulation.next_completion_time(), None);
}

#[test]
fn consume_moves_snapshot_energy_into_the_gut_plant_first() {
    let mut simulation = ReferenceSimulation::new(1, 1, metabolic_rules()).unwrap();
    let tile = simulation.tile(0, 0).unwrap();
    simulation.tile_state_mut(tile).unwrap().plant_energy = 12;
    simulation.tile_state_mut(tile).unwrap().loose_energy = 12;
    let actor = simulation.add_cell(tile, 10, 100, 0).unwrap();
    let before = simulation.total_energy_equivalent();

    simulation
        .commit_action(actor, ActionRequest::Consume { amount: 30 })
        .unwrap();
    let report = simulation.resolve_next_batch().unwrap();

    assert_eq!(statuses(&report)[&actor], OutcomeStatus::Success);
    assert_eq!(simulation.cell(actor).unwrap().gut_energy, 16);
    assert_eq!(simulation.tile_state(tile).unwrap().plant_energy, 0);
    assert_eq!(simulation.tile_state(tile).unwrap().loose_energy, 8);
    assert_eq!(simulation.total_energy_equivalent(), before);
}

#[test]
fn committed_consume_is_exposed_as_local_feeding_activity() {
    let mut simulation = ReferenceSimulation::new(2, 1, metabolic_rules()).unwrap();
    let feeder_tile = simulation.tile(0, 0).unwrap();
    simulation.tile_state_mut(feeder_tile).unwrap().plant_energy = 16;
    let feeder = simulation.add_cell(feeder_tile, 10, 100, 0).unwrap();
    let observer = simulation
        .add_cell(simulation.tile(1, 0).unwrap(), 10, 100, 0)
        .unwrap();

    simulation
        .commit_action(feeder, ActionRequest::Consume { amount: 16 })
        .unwrap();

    assert_eq!(
        simulation.neighbor_cues(observer).unwrap()[usize::from(WEST.0)]
            .unwrap()
            .activity,
        Some(ActivityCue::Feeding)
    );
}

#[test]
fn consume_is_limited_by_remaining_gut_capacity() {
    let mut rules = metabolic_rules();
    rules.gut_capacity = 20;
    let mut simulation = ReferenceSimulation::new(1, 1, rules).unwrap();
    let tile = simulation.tile(0, 0).unwrap();
    simulation.tile_state_mut(tile).unwrap().plant_energy = 40;
    let actor = simulation.add_cell(tile, 10, 100, 0).unwrap();

    simulation
        .commit_action(actor, ActionRequest::Consume { amount: 16 })
        .unwrap();
    simulation.resolve_next_batch().unwrap();
    simulation
        .commit_action(actor, ActionRequest::Consume { amount: 16 })
        .unwrap();
    simulation.resolve_next_batch().unwrap();

    assert_eq!(simulation.cell(actor).unwrap().gut_energy, 20);
    assert_eq!(simulation.tile_state(tile).unwrap().plant_energy, 20);
}

#[test]
fn digestion_is_invariant_to_event_time_partitioning() {
    let mut rules = uniform_rules();
    rules.digestion_rate_numerator = 1;
    rules.digestion_rate_denominator = 256;
    let mut direct = ReferenceSimulation::new(1, 1, rules).unwrap();
    let tile = direct.tile(0, 0).unwrap();
    direct.tile_state_mut(tile).unwrap().plant_energy = 16;
    let actor = direct.add_cell(tile, 10, 100, 0).unwrap();
    direct
        .commit_action(actor, ActionRequest::Consume { amount: 16 })
        .unwrap();
    direct.resolve_next_batch().unwrap();
    let mut partitioned = direct.clone();
    let start = direct.now().0;

    direct.advance_clock_to(SimTime(start + 1024)).unwrap();
    partitioned.advance_clock_to(SimTime(start + 128)).unwrap();
    partitioned.advance_clock_to(SimTime(start + 512)).unwrap();
    partitioned.advance_clock_to(SimTime(start + 1024)).unwrap();

    assert_eq!(direct.canonical_state(), partitioned.canonical_state());
    assert_eq!(direct.cell(actor).unwrap().gut_energy, 12);
    assert_eq!(direct.cell(actor).unwrap().assimilated_energy, 103);
}

#[test]
fn metabolism_is_partition_invariant_and_conserves_energy() {
    let mut rules = uniform_rules();
    rules.metabolism_rate_numerator = 1;
    rules.metabolism_rate_denominator = 1024;
    let mut direct = ReferenceSimulation::new(1, 1, rules).unwrap();
    let tile = direct.tile(0, 0).unwrap();
    let actor = direct.add_cell(tile, 10, 10, 0).unwrap();
    let before = direct.total_energy_equivalent();
    let mut partitioned = direct.clone();

    direct.advance_clock_to(SimTime(1024)).unwrap();
    partitioned.advance_clock_to(SimTime(128)).unwrap();
    partitioned.advance_clock_to(SimTime(512)).unwrap();
    partitioned.advance_clock_to(SimTime(1024)).unwrap();

    assert_eq!(direct.canonical_state(), partitioned.canonical_state());
    assert_eq!(direct.cell(actor).unwrap().assimilated_energy, 9);
    assert_eq!(direct.tile_state(tile).unwrap().diffuse_energy, 1);
    assert_eq!(direct.total_energy_equivalent(), before);
}

#[test]
fn metabolic_exhaustion_is_a_scheduled_event_that_interrupts_future_work() {
    let mut rules = uniform_rules();
    rules.metabolism_rate_numerator = 1;
    rules.metabolism_rate_denominator = 1;
    let mut simulation = ReferenceSimulation::new(1, 1, rules).unwrap();
    let tile = simulation.tile(0, 0).unwrap();
    let actor = simulation.add_cell(tile, 10, 5, 0).unwrap();
    let before = simulation.total_energy_equivalent();
    simulation
        .commit_action(actor, ActionRequest::Wait)
        .unwrap();

    assert_eq!(simulation.next_completion_time(), Some(SimTime(1024)));
    assert_eq!(simulation.next_event_time().unwrap(), Some(SimTime(5)));
    let report = simulation.resolve_next_batch().unwrap();

    assert_eq!(report.completed_at, SimTime(5));
    assert_eq!(report.deaths, vec![actor]);
    assert_eq!(statuses(&report)[&actor], OutcomeStatus::Interrupted);
    assert_eq!(simulation.tile_state(tile).unwrap().diffuse_energy, 5);
    assert_eq!(simulation.tile_state(tile).unwrap().loose_energy, 10);
    assert_eq!(simulation.total_energy_equivalent(), before);
}

#[test]
fn action_completing_at_exact_metabolic_exhaustion_still_resolves() {
    let mut rules = uniform_rules();
    rules.metabolism_rate_numerator = 1;
    rules.metabolism_rate_denominator = 1024;
    let mut simulation = ReferenceSimulation::new(1, 1, rules).unwrap();
    let tile = simulation.tile(0, 0).unwrap();
    let actor = simulation.add_cell(tile, 10, 1, 0).unwrap();
    simulation
        .commit_action(actor, ActionRequest::Wait)
        .unwrap();

    let report = simulation.resolve_next_batch().unwrap();

    assert_eq!(report.completed_at, SimTime(1024));
    assert_eq!(statuses(&report)[&actor], OutcomeStatus::Success);
    assert_eq!(report.deaths, vec![actor]);
}

#[test]
fn excavation_and_deposition_conserve_terrain_mass_energy() {
    let mut simulation = ReferenceSimulation::new(1, 1, uniform_rules()).unwrap();
    let tile = simulation.tile(0, 0).unwrap();
    let actor = simulation.add_cell(tile, 10, 100, 0).unwrap();
    let before = simulation.total_energy_equivalent();

    simulation
        .commit_action(actor, ActionRequest::Excavate)
        .unwrap();
    let excavated = simulation.resolve_next_batch().unwrap();
    assert_eq!(statuses(&excavated)[&actor], OutcomeStatus::Success);
    let excavated_change = excavated.outcomes[0].terrain_change.as_ref().unwrap();
    assert_eq!(excavated_change.tile, tile);
    assert_eq!(excavated_change.elevation_before, 0);
    assert_eq!(excavated_change.elevation_after, -1);
    assert_eq!(
        excavated_change.material_mass,
        simulation.rules().terrain_mass_per_elevation
    );
    assert!(excavated.outcomes[0].attack_damage.is_none());
    assert_eq!(simulation.tile_state(tile).unwrap().elevation, -1);
    assert_eq!(
        simulation.cell(actor).unwrap().carried_material_mass,
        simulation.rules().terrain_mass_per_elevation
    );
    assert_eq!(simulation.total_energy_equivalent(), before);

    simulation
        .commit_action(actor, ActionRequest::DepositTerrain)
        .unwrap();
    let deposited = simulation.resolve_next_batch().unwrap();
    assert_eq!(statuses(&deposited)[&actor], OutcomeStatus::Success);
    let deposited_change = deposited.outcomes[0].terrain_change.as_ref().unwrap();
    assert_eq!(deposited_change.tile, tile);
    assert_eq!(deposited_change.elevation_before, -1);
    assert_eq!(deposited_change.elevation_after, 0);
    assert_eq!(
        deposited_change.material_mass,
        simulation.rules().terrain_mass_per_elevation
    );
    assert_eq!(simulation.tile_state(tile).unwrap().elevation, 0);
    assert_eq!(simulation.cell(actor).unwrap().carried_material_mass, 0);
    assert_eq!(simulation.total_energy_equivalent(), before);
}

#[test]
fn local_signal_is_committed_with_the_action_and_decays_to_diffuse_energy() {
    use blob_interface::reference_mind::ReferenceSignalEmission;

    let mut rules = uniform_rules();
    rules.signal_emission_cost = 3;
    rules.signal_decay_rate_numerator = 1;
    rules.signal_decay_rate_denominator = 1024;
    let mut simulation = ReferenceSimulation::new(2, 1, rules).unwrap();
    let actor_tile = simulation.tile(0, 0).unwrap();
    let actor = simulation.add_cell(actor_tile, 10, 100, 0).unwrap();
    let observer = simulation
        .add_cell(simulation.tile(1, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let before = simulation.total_energy_equivalent();

    simulation
        .commit_decision_with_signal(
            actor,
            ActionRequest::Wait,
            Some(ReferenceSignalEmission {
                channel: 2,
                amount: 6,
            }),
            Vec::new(),
        )
        .unwrap();

    assert_eq!(simulation.cell(actor).unwrap().assimilated_energy, 94);
    assert_eq!(
        simulation.tile_state(actor_tile).unwrap().signal_energy[2],
        6
    );
    let observed = simulation
        .reference_mind_input(
            observer,
            blob_interface::randomness::PrivateRandom::from_bytes([0; 32]),
        )
        .unwrap();
    assert_eq!(observed.slots[3].signal_energy.unwrap()[2], 6);
    simulation.resolve_next_batch().unwrap();
    assert_eq!(
        simulation.tile_state(actor_tile).unwrap().signal_energy[2],
        5
    );
    assert_eq!(simulation.tile_state(actor_tile).unwrap().diffuse_energy, 1);
    assert_eq!(simulation.total_energy_equivalent(), before);

    let mut direct = simulation.clone();
    let mut partitioned = simulation.clone();
    direct.advance_clock_to(SimTime(2048)).unwrap();
    partitioned.advance_clock_to(SimTime(1536)).unwrap();
    partitioned.advance_clock_to(SimTime(2048)).unwrap();
    assert_eq!(direct.canonical_state(), partitioned.canonical_state());
    assert_eq!(direct.tile_state(actor_tile).unwrap().signal_energy[2], 4);
    assert_eq!(direct.tile_state(actor_tile).unwrap().diffuse_energy, 2);

    let before_invalid = simulation.state_hash();
    assert!(matches!(
        simulation.commit_decision_with_signal(
            observer,
            ActionRequest::Wait,
            Some(ReferenceSignalEmission {
                channel: 4,
                amount: 3,
            }),
            Vec::new(),
        ),
        Err(blob_engine::resolution::CommitError::InvalidSignal(_))
    ));
    assert_eq!(simulation.state_hash(), before_invalid);
}

#[test]
fn signal_cost_is_paid_before_primary_action_validation() {
    use blob_interface::reference_mind::ReferenceSignalEmission;

    let mut rules = uniform_rules();
    rules.signal_emission_cost = 3;
    let mut simulation = ReferenceSimulation::new(2, 1, rules).unwrap();
    let actor_tile = simulation.tile(0, 0).unwrap();
    let actor = simulation.add_cell(actor_tile, 10, 5, 0).unwrap();
    simulation
        .add_cell(simulation.tile(1, 0).unwrap(), 10, 100, 0)
        .unwrap();

    let receipt = simulation
        .commit_decision_with_signal(
            actor,
            ActionRequest::Attack {
                target: LocalSlot(4),
                payload: 1,
                effort: EffortTier::Standard,
            },
            Some(ReferenceSignalEmission {
                channel: 1,
                amount: 3,
            }),
            Vec::new(),
        )
        .unwrap();

    assert!(!receipt.accepted);
    assert_eq!(receipt.rejection, Some(RejectReason::InsufficientEnergy));
    assert_eq!(simulation.cell(actor).unwrap().assimilated_energy, 2);
    assert_eq!(
        simulation.tile_state(actor_tile).unwrap().signal_energy[1],
        3
    );
}

#[test]
fn explicit_signal_vector_is_variable_and_terrain_edits_erase_its_information() {
    let mut rules = uniform_rules();
    rules.signal_emission_cost = 2;
    rules.signal_decay_rate_numerator = 1;
    rules.signal_decay_rate_denominator = 1024;
    rules.diffusion_rate_numerator = 0;
    let mut simulation = ReferenceSimulation::new(1, 1, rules).unwrap();
    let tile = simulation.tile(0, 0).unwrap();
    let actor = simulation.add_cell(tile, 10, 100, 0).unwrap();
    let before = simulation.total_energy_equivalent();

    let receipt = simulation
        .commit_action(
            actor,
            ActionRequest::Signal {
                amounts: [2, 4, 0, 6],
            },
        )
        .unwrap();
    assert!(receipt.accepted);
    assert_eq!(simulation.cell(actor).unwrap().assimilated_energy, 88);
    assert_eq!(
        simulation.tile_state(tile).unwrap().signal_energy,
        [2, 4, 0, 6]
    );
    let report = simulation.resolve_next_batch().unwrap();
    assert_eq!(report.outcomes[0].action, ActionKind::Signal);
    assert_eq!(report.outcomes[0].status, OutcomeStatus::Success);

    simulation.advance_clock_to(SimTime(1536)).unwrap();
    assert!(simulation.tile_state(tile).unwrap().signal_decay_remainder[0] > 0);
    simulation
        .commit_action(actor, ActionRequest::Excavate)
        .unwrap();
    simulation.resolve_next_batch().unwrap();

    let state = simulation.tile_state(tile).unwrap();
    assert_eq!(state.signal_energy, [0; 4]);
    assert_eq!(state.signal_decay_remainder, [0; 4]);
    assert!(state.diffuse_energy >= 12);

    simulation
        .commit_action(
            actor,
            ActionRequest::Signal {
                amounts: [0, 6, 0, 0],
            },
        )
        .unwrap();
    simulation.resolve_next_batch().unwrap();
    assert!(simulation.tile_state(tile).unwrap().signal_energy[1] > 0);
    simulation
        .commit_action(actor, ActionRequest::DepositTerrain)
        .unwrap();
    simulation.resolve_next_batch().unwrap();
    let state = simulation.tile_state(tile).unwrap();
    assert_eq!(state.signal_energy, [0; 4]);
    assert_eq!(state.signal_decay_remainder, [0; 4]);
    assert_eq!(simulation.total_energy_equivalent(), before);
}

#[test]
fn signal_deposits_reject_zero_nonquantized_and_double_emission_atomically() {
    use blob_interface::reference_mind::ReferenceMemoryUpdate;

    let mut rules = uniform_rules();
    rules.signal_emission_cost = 3;
    let mut simulation = ReferenceSimulation::new(1, 1, rules).unwrap();
    let tile = simulation.tile(0, 0).unwrap();
    let actor = simulation.add_cell(tile, 10, 100, 0).unwrap();
    let before = simulation.state_hash();

    for decision in [
        DecisionCommitment {
            actor,
            request: ActionRequest::Wait,
            signal: Some(ReferenceSignalEmission {
                channel: 0,
                amount: 0,
            }),
            memory_update: ReferenceMemoryUpdate::Retain,
        },
        DecisionCommitment {
            actor,
            request: ActionRequest::Wait,
            signal: Some(ReferenceSignalEmission {
                channel: 0,
                amount: 4,
            }),
            memory_update: ReferenceMemoryUpdate::Retain,
        },
        DecisionCommitment {
            actor,
            request: ActionRequest::Signal {
                amounts: [3, 0, 0, 0],
            },
            signal: Some(ReferenceSignalEmission {
                channel: 1,
                amount: 3,
            }),
            memory_update: ReferenceMemoryUpdate::Retain,
        },
    ] {
        assert!(matches!(
            simulation.commit_decisions_ordered(&[decision]),
            Err(blob_engine::resolution::CommitError::InvalidSignal(_))
        ));
        assert_eq!(simulation.state_hash(), before);
    }
}

#[test]
fn synchronous_diffusion_is_partition_invariant_and_conservative() {
    let mut rules = uniform_rules();
    rules.diffusion_rate_numerator = 1;
    rules.diffusion_rate_denominator = 2;
    rules.diffusion_interval_quanta = 1024;
    let mut direct = ReferenceSimulation::new(3, 1, rules).unwrap();
    let center = direct.tile(1, 0).unwrap();
    direct.tile_state_mut(center).unwrap().diffuse_energy = 20;
    let before = direct.total_energy_equivalent();
    let mut partitioned = direct.clone();

    assert_eq!(direct.next_event_time().unwrap(), Some(SimTime(1024)));
    let report = direct.resolve_next_batch().unwrap();
    assert!(report.outcomes.is_empty());
    partitioned.advance_clock_to(SimTime(512)).unwrap();
    partitioned.resolve_next_batch().unwrap();

    assert_eq!(direct.canonical_state(), partitioned.canonical_state());
    assert_eq!(
        direct
            .tile_state(direct.tile(0, 0).unwrap())
            .unwrap()
            .diffuse_energy,
        5
    );
    assert_eq!(direct.tile_state(center).unwrap().diffuse_energy, 10);
    assert_eq!(
        direct
            .tile_state(direct.tile(2, 0).unwrap())
            .unwrap()
            .diffuse_energy,
        5
    );
    assert_eq!(direct.total_energy_equivalent(), before);

    let mut subquantum = direct.clone();
    for index in 0..3 {
        let tile = subquantum.tile(index, 0).unwrap();
        subquantum.tile_state_mut(tile).unwrap().diffuse_energy = 0;
    }
    subquantum.tile_state_mut(center).unwrap().diffuse_energy = 1;
    assert_eq!(subquantum.next_diffusion_event_time().unwrap(), None);
}

#[test]
fn digestion_can_sustain_a_cell_at_a_conservative_metabolic_barrier() {
    let mut rules = uniform_rules();
    rules.wait_duration = DurationRule::new(2048, 0, 1);
    rules.digestion_rate_numerator = 1;
    rules.digestion_rate_denominator = 1024;
    rules.metabolism_rate_numerator = 1;
    rules.metabolism_rate_denominator = 1024;
    let mut initial = ReferenceSimulation::new(1, 1, rules.clone()).unwrap();
    let tile = initial.tile(0, 0).unwrap();
    let actor = initial.add_cell(tile, 10, 1, 0).unwrap();
    let mut state = initial.canonical_state();
    state.cells[0].1.gut_energy = 1;
    let mut simulation = ReferenceSimulation::from_canonical_state(1, 1, rules, state).unwrap();
    let before = simulation.total_energy_equivalent();
    simulation
        .commit_action(actor, ActionRequest::Wait)
        .unwrap();

    assert_eq!(simulation.next_completion_time(), Some(SimTime(2048)));
    assert_eq!(simulation.next_event_time().unwrap(), Some(SimTime(1024)));
    let barrier = simulation.resolve_next_batch().unwrap();

    assert_eq!(barrier.completed_at, SimTime(1024));
    assert!(barrier.outcomes.is_empty());
    assert!(barrier.deaths.is_empty());
    assert_eq!(simulation.cell(actor).unwrap().assimilated_energy, 1);
    assert_eq!(simulation.cell(actor).unwrap().gut_energy, 0);
    assert_eq!(simulation.tile_state(tile).unwrap().diffuse_energy, 1);
    assert_eq!(simulation.total_energy_equivalent(), before);

    let completion = simulation.resolve_next_batch().unwrap();
    assert_eq!(completion.completed_at, SimTime(2048));
    assert_eq!(statuses(&completion)[&actor], OutcomeStatus::Success);
    assert_eq!(completion.deaths, vec![actor]);
    assert_eq!(simulation.total_energy_equivalent(), before);
}

#[test]
fn plant_growth_conserves_energy_and_is_invariant_to_time_partitioning() {
    let mut direct = ReferenceSimulation::new(1, 1, uniform_rules()).unwrap();
    let tile = direct.tile(0, 0).unwrap();
    {
        let plant = direct.tile_state_mut(tile).unwrap();
        plant.plant_energy = 4;
        plant.plant_capacity = 20;
        plant.plant_growth_rate = 3;
        plant.diffuse_energy = 10;
    }
    let before = direct.total_energy_equivalent();
    let mut partitioned = direct.clone();

    direct.advance_clock_to(SimTime(1024)).unwrap();
    partitioned.advance_clock_to(SimTime(128)).unwrap();
    partitioned.advance_clock_to(SimTime(512)).unwrap();
    partitioned.advance_clock_to(SimTime(1024)).unwrap();

    assert_eq!(direct.canonical_state(), partitioned.canonical_state());
    assert_eq!(direct.tile_state(tile).unwrap().plant_energy, 7);
    assert_eq!(direct.tile_state(tile).unwrap().diffuse_energy, 7);
    assert_eq!(direct.total_energy_equivalent(), before);
}

#[test]
fn plant_growth_stops_at_capacity_without_banking_progress() {
    let mut simulation = ReferenceSimulation::new(1, 1, uniform_rules()).unwrap();
    let tile = simulation.tile(0, 0).unwrap();
    {
        let plant = simulation.tile_state_mut(tile).unwrap();
        plant.plant_energy = 9;
        plant.plant_capacity = 10;
        plant.plant_growth_rate = 5;
        plant.diffuse_energy = 20;
    }

    simulation.advance_clock_to(SimTime(1024)).unwrap();
    let plant = simulation.tile_state(tile).unwrap();
    assert_eq!(plant.plant_energy, 10);
    assert_eq!(plant.diffuse_energy, 19);
    assert_eq!(plant.plant_growth_remainder, 0);
}

#[test]
fn regurgitate_moves_gut_escrow_to_adjacent_loose_energy() {
    let mut simulation = ReferenceSimulation::new(2, 1, metabolic_rules()).unwrap();
    let origin = simulation.tile(0, 0).unwrap();
    let target = simulation.tile(1, 0).unwrap();
    simulation.tile_state_mut(origin).unwrap().plant_energy = 16;
    let actor = simulation.add_cell(origin, 10, 100, 0).unwrap();
    simulation
        .commit_action(actor, ActionRequest::Consume { amount: 16 })
        .unwrap();
    simulation.resolve_next_batch().unwrap();
    let before = simulation.total_energy_equivalent();

    simulation
        .commit_action(
            actor,
            ActionRequest::Regurgitate {
                target: EAST,
                amount: 10,
            },
        )
        .unwrap();
    let report = simulation.resolve_next_batch().unwrap();

    assert_eq!(statuses(&report)[&actor], OutcomeStatus::Success);
    assert_eq!(simulation.cell(actor).unwrap().gut_energy, 6);
    assert_eq!(simulation.tile_state(target).unwrap().loose_energy, 10);
    assert_eq!(simulation.total_energy_equivalent(), before);
}

#[test]
fn regurgitate_rejects_without_gut_energy_and_creates_no_escrow() {
    let mut simulation = ReferenceSimulation::new(2, 1, metabolic_rules()).unwrap();
    let actor = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let before = simulation.total_energy_equivalent();

    let receipt = simulation
        .commit_action(
            actor,
            ActionRequest::Regurgitate {
                target: EAST,
                amount: 1,
            },
        )
        .unwrap();

    assert!(!receipt.accepted);
    assert_eq!(receipt.rejection, Some(RejectReason::InsufficientGutEnergy));
    assert_eq!(receipt.payload_escrow, 0);
    simulation.resolve_next_batch().unwrap();
    assert_eq!(simulation.total_energy_equivalent(), before);
}

#[test]
fn frustrated_regurgitate_returns_payload_to_the_gut() {
    let mut simulation = ReferenceSimulation::new(2, 1, metabolic_rules()).unwrap();
    let origin = simulation.tile(0, 0).unwrap();
    let target = simulation.tile(1, 0).unwrap();
    simulation.tile_state_mut(origin).unwrap().plant_energy = 16;
    simulation.tile_state_mut(target).unwrap().elevation = 10;
    let actor = simulation.add_cell(origin, 10, 100, 0).unwrap();
    simulation
        .commit_action(actor, ActionRequest::Consume { amount: 16 })
        .unwrap();
    simulation.resolve_next_batch().unwrap();
    let before = simulation.total_energy_equivalent();

    simulation
        .commit_action(
            actor,
            ActionRequest::Regurgitate {
                target: EAST,
                amount: 10,
            },
        )
        .unwrap();
    let report = simulation.resolve_next_batch().unwrap();

    assert_eq!(statuses(&report)[&actor], OutcomeStatus::Frustrated);
    assert_eq!(simulation.cell(actor).unwrap().gut_energy, 16);
    assert_eq!(simulation.tile_state(target).unwrap().loose_energy, 0);
    assert_eq!(simulation.total_energy_equivalent(), before);
}

#[test]
fn consume_cannot_take_same_batch_regurgitated_energy() {
    let mut simulation = ReferenceSimulation::new(2, 1, metabolic_rules()).unwrap();
    let consumer_tile = simulation.tile(0, 0).unwrap();
    let donor_tile = simulation.tile(1, 0).unwrap();
    simulation.tile_state_mut(donor_tile).unwrap().plant_energy = 16;
    let consumer = simulation.add_cell(consumer_tile, 10, 100, 0).unwrap();
    let donor = simulation.add_cell(donor_tile, 10, 100, 0).unwrap();
    simulation
        .commit_action(donor, ActionRequest::Consume { amount: 16 })
        .unwrap();
    simulation.resolve_next_batch().unwrap();
    let before = simulation.total_energy_equivalent();

    simulation
        .commit_action(consumer, ActionRequest::Consume { amount: 16 })
        .unwrap();
    simulation
        .commit_action(
            donor,
            ActionRequest::Regurgitate {
                target: WEST,
                amount: 16,
            },
        )
        .unwrap();
    let report = simulation.resolve_next_batch().unwrap();
    let outcomes = statuses(&report);

    assert_eq!(outcomes[&consumer], OutcomeStatus::Frustrated);
    assert_eq!(outcomes[&donor], OutcomeStatus::Success);
    assert_eq!(simulation.cell(consumer).unwrap().gut_energy, 0);
    assert_eq!(
        simulation.tile_state(consumer_tile).unwrap().loose_energy,
        16
    );
    assert_eq!(simulation.total_energy_equivalent(), before);
}

fn build_regurgitate_consume_case(reverse_commit_order: bool) -> ReferenceSimulation {
    let mut simulation = ReferenceSimulation::new(2, 1, metabolic_rules()).unwrap();
    let consumer_tile = simulation.tile(0, 0).unwrap();
    let donor_tile = simulation.tile(1, 0).unwrap();
    simulation.tile_state_mut(donor_tile).unwrap().plant_energy = 16;
    let consumer = simulation.add_cell(consumer_tile, 10, 100, 0).unwrap();
    let donor = simulation.add_cell(donor_tile, 10, 100, 0).unwrap();
    simulation
        .commit_action(donor, ActionRequest::Consume { amount: 16 })
        .unwrap();
    simulation.resolve_next_batch().unwrap();

    let mut actions = vec![
        (consumer, ActionRequest::Consume { amount: 16 }),
        (
            donor,
            ActionRequest::Regurgitate {
                target: WEST,
                amount: 16,
            },
        ),
    ];
    if reverse_commit_order {
        actions.reverse();
    }
    for (actor, action) in actions {
        simulation.commit_action(actor, action).unwrap();
    }
    simulation
}

#[test]
fn regurgitate_consume_batch_is_commit_order_independent() {
    let mut forward = build_regurgitate_consume_case(false);
    let mut reverse = build_regurgitate_consume_case(true);

    let forward_report = forward.resolve_next_batch().unwrap();
    let reverse_report = reverse.resolve_next_batch().unwrap();

    assert_eq!(forward_report, reverse_report);
    assert_eq!(forward.canonical_state(), reverse.canonical_state());
}

#[test]
fn earlier_death_spills_and_interrupts_regurgitation_escrow() {
    let mut rules = metabolic_rules();
    rules.regurgitate_duration = DurationRule::new(2048, 0, 1);
    let mut simulation = ReferenceSimulation::new(3, 1, rules).unwrap();
    let attacker_tile = simulation.tile(0, 0).unwrap();
    let victim_tile = simulation.tile(1, 0).unwrap();
    let target_tile = simulation.tile(2, 0).unwrap();
    simulation.tile_state_mut(victim_tile).unwrap().plant_energy = 16;
    let attacker = simulation.add_cell(attacker_tile, 10, 200, 0).unwrap();
    let victim = simulation.add_cell(victim_tile, 10, 100, 0).unwrap();
    simulation
        .commit_action(victim, ActionRequest::Consume { amount: 16 })
        .unwrap();
    simulation.resolve_next_batch().unwrap();
    let before = simulation.total_energy_equivalent();

    simulation
        .commit_action(
            victim,
            ActionRequest::Regurgitate {
                target: EAST,
                amount: 10,
            },
        )
        .unwrap();
    simulation
        .commit_action(
            attacker,
            ActionRequest::Attack {
                target: EAST,
                effort: EffortTier::Standard,
                payload: 100,
            },
        )
        .unwrap();
    let report = simulation.resolve_next_batch().unwrap();
    let interrupted = report
        .outcomes
        .iter()
        .find(|outcome| outcome.actor == victim)
        .unwrap();

    assert_eq!(interrupted.status, OutcomeStatus::Interrupted);
    assert!(report.deaths.contains(&victim));
    assert_eq!(simulation.tile_state(target_tile).unwrap().loose_energy, 0);
    assert_eq!(simulation.next_completion_time(), None);
    assert_eq!(simulation.total_energy_equivalent(), before);
}
