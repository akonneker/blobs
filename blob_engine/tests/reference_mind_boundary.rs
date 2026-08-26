use blob_engine::resolution::{
    ActionRequest, BoundaryRule, CellKey, CommitError, EffortTier, LocalSlot, ReferenceRuleset,
    ReferenceSimulation,
};
use blob_interface::randomness::PrivateRandom;
use blob_interface::reference_mind::{
    ReferenceActivity, ReferenceEffort, ReferenceMindAction, ReferenceProgress,
};
use blob_interface::reference_mind_converter::{
    reference_mind_input_to_capnp, ReferenceMindLimits,
};

const WEST: LocalSlot = LocalSlot(3);
const EAST_INDEX: usize = 4;

fn projected_simulation(
    observer_x: usize,
    observer_y: usize,
    with_distant_dummy: bool,
) -> (ReferenceSimulation, CellKey) {
    let rules = ReferenceRuleset {
        neighborhood: blob_engine::resolution::NeighborhoodSpec::moore_8(BoundaryRule::Bounded),
        ..ReferenceRuleset::default()
    };
    let mut simulation = ReferenceSimulation::new(7, 7, rules).unwrap();
    if with_distant_dummy {
        simulation
            .add_cell(simulation.tile(0, 0).unwrap(), 99, 777, 0xffff)
            .unwrap();
    }
    let observer_tile = simulation.tile(observer_x, observer_y).unwrap();
    let observer = simulation.add_cell(observer_tile, 10, 100, 17).unwrap();
    let neighbor = simulation
        .add_cell(
            simulation.tile(observer_x + 1, observer_y).unwrap(),
            12,
            100,
            23,
        )
        .unwrap();
    {
        let tile = simulation.tile_state_mut(observer_tile).unwrap();
        tile.elevation = -3;
        tile.plant_energy = 21;
        tile.plant_capacity = 90;
        tile.plant_growth_rate = 6;
        tile.loose_energy = 8;
        tile.diffuse_energy = 5;
    }
    simulation
        .commit_action(
            neighbor,
            ActionRequest::Attack {
                target: WEST,
                effort: EffortTier::Standard,
                payload: 7,
            },
        )
        .unwrap();
    (simulation, observer)
}

#[test]
fn projection_is_relative_anonymous_and_complete_for_local_decisions() {
    let random = PrivateRandom::from_bytes([7; 32]);
    let (first, first_observer) = projected_simulation(2, 2, false);
    let (second, second_observer) = projected_simulation(4, 4, true);
    assert_ne!(first_observer, second_observer);

    let first_input = first.reference_mind_input(first_observer, random).unwrap();
    let second_input = second
        .reference_mind_input(second_observer, random)
        .unwrap();
    assert_eq!(first_input, second_input);
    assert_eq!(first_input.current_tile.elevation, -3);
    assert_eq!(first_input.current_tile.plant_energy, 21);
    assert_eq!(first_input.current_tile.plant_capacity, 90);
    assert_eq!(first_input.current_tile.plant_growth_rate, 6);
    assert_eq!(first_input.current_tile.loose_energy, 8);
    assert_eq!(first_input.current_tile.diffuse_energy, 5);
    assert_eq!(first_input.current_tile.signal_energy, [0; 4]);
    assert_eq!(first_input.self_state.metabolism_remainder, 0);
    assert_eq!(
        first_input.action_space.metabolism_rate_numerator,
        first.rules().metabolism_rate_numerator
    );
    assert_eq!(
        first_input.action_space.metabolism_rate_denominator,
        first.rules().metabolism_rate_denominator
    );
    assert!(first_input.action_space.excavate_enabled);
    assert!(!first_input.action_space.deposit_terrain_enabled);
    assert!(first_input.action_space.signal_enabled);

    let offsets: Vec<_> = first_input
        .slots
        .iter()
        .map(|slot| (slot.dx, slot.dy))
        .collect();
    assert_eq!(
        offsets,
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
    );
    let neighbor = first_input.slots[EAST_INDEX].neighbor.unwrap();
    assert_eq!(neighbor.marker, Some(23));
    assert_eq!(neighbor.activity, Some(ReferenceActivity::AttackWindup));
    assert_eq!(neighbor.progress, Some(ReferenceProgress::Early));

    let limits = ReferenceMindLimits::default();
    assert_eq!(
        reference_mind_input_to_capnp(&first_input, limits).unwrap(),
        reference_mind_input_to_capnp(&second_input, limits).unwrap()
    );
}

#[test]
fn every_reference_action_maps_without_defaults_or_loss() {
    let cases = [
        (ReferenceMindAction::Wait, ActionRequest::Wait),
        (
            ReferenceMindAction::Move {
                target_slot: 1,
                effort: ReferenceEffort::Gentle,
            },
            ActionRequest::Move {
                target: LocalSlot(1),
                effort: EffortTier::Low,
            },
        ),
        (
            ReferenceMindAction::Attack {
                target_slot: 2,
                effort: ReferenceEffort::Burst,
                payload: 33,
            },
            ActionRequest::Attack {
                target: LocalSlot(2),
                effort: EffortTier::High,
                payload: 33,
            },
        ),
        (
            ReferenceMindAction::Guard {
                effort: ReferenceEffort::Standard,
            },
            ActionRequest::Guard {
                effort: EffortTier::Standard,
            },
        ),
        (
            ReferenceMindAction::Consume { amount: 9 },
            ActionRequest::Consume { amount: 9 },
        ),
        (
            ReferenceMindAction::Split {
                target_slot: 5,
                child_allocation: 44,
                marker: 51,
                private_memory: vec![1, 2, 3],
            },
            ActionRequest::Split {
                target: LocalSlot(5),
                child_allocation: 44,
                marker: 51,
                private_memory: vec![1, 2, 3],
            },
        ),
        (
            ReferenceMindAction::Regurgitate {
                target_slot: 6,
                amount: 12,
            },
            ActionRequest::Regurgitate {
                target: LocalSlot(6),
                amount: 12,
            },
        ),
        (
            ReferenceMindAction::Signal {
                amounts: [2, 4, 0, 8],
            },
            ActionRequest::Signal {
                amounts: [2, 4, 0, 8],
            },
        ),
        (ReferenceMindAction::Excavate, ActionRequest::Excavate),
        (
            ReferenceMindAction::DepositTerrain,
            ActionRequest::DepositTerrain,
        ),
    ];
    for (wire_action, expected) in cases {
        assert_eq!(ActionRequest::from(wire_action), expected);
    }
}

#[test]
fn current_tile_is_not_smuggled_into_target_slots() {
    let (simulation, observer) = projected_simulation(2, 2, false);
    let input = simulation
        .reference_mind_input(observer, PrivateRandom::ZERO)
        .unwrap();
    assert!(input.slots.iter().all(|slot| slot.dx != 0 || slot.dy != 0));
    assert_eq!(input.slots.len(), simulation.neighborhood().slot_count());
}

#[test]
fn reusable_projection_clears_every_previous_observer_value() {
    let (first, first_observer) = projected_simulation(2, 2, false);
    let mut scratch = first
        .reference_mind_input(first_observer, PrivateRandom::from_bytes([1; 32]))
        .unwrap();
    assert!(scratch.slots.iter().any(|slot| slot.neighbor.is_some()));
    scratch.private_memory.extend_from_slice(&[9, 8, 7]);

    let rules = ReferenceRuleset {
        neighborhood: blob_engine::resolution::NeighborhoodSpec::moore_8(BoundaryRule::Bounded),
        ..ReferenceRuleset::default()
    };
    let mut second = ReferenceSimulation::new(3, 3, rules).unwrap();
    let second_observer = second
        .add_cell(second.tile(1, 1).unwrap(), 7, 41, 55)
        .unwrap();
    let random = PrivateRandom::from_bytes([2; 32]);
    let expected = second
        .reference_mind_input(second_observer, random)
        .unwrap();

    second
        .reference_mind_input_into(second_observer, random, &mut scratch)
        .unwrap();

    assert_eq!(scratch, expected);
    assert!(scratch.private_memory.is_empty());
    assert!(scratch.slots.iter().all(|slot| slot.neighbor.is_none()));
}

#[test]
fn batch_observation_lookup_matches_the_direct_indexed_path() {
    let (simulation, observer) = projected_simulation(2, 2, true);
    let randomness = PrivateRandom::from_bytes([7; 32]);
    let expected = simulation
        .reference_mind_input(observer, randomness)
        .unwrap();
    let batch = simulation.observation_batch();
    let actual = batch.reference_mind_input(observer, randomness).unwrap();
    assert_eq!(actual, expected);

    let mut reused = simulation
        .reference_mind_input(observer, PrivateRandom::ZERO)
        .unwrap();
    reused.private_memory.extend_from_slice(&[1, 2, 3]);
    batch
        .reference_mind_input_into(observer, randomness, &mut reused)
        .unwrap();
    assert_eq!(reused, expected);

    let rules = simulation.rules().clone();
    let mut sparse_state = simulation.canonical_state();
    sparse_state.next_cell_key = 1_000_000;
    let sparse = ReferenceSimulation::from_canonical_state(7, 7, rules, sparse_state).unwrap();
    let sparse_expected = sparse.reference_mind_input(observer, randomness).unwrap();
    let sparse_batch = sparse.observation_batch();
    assert_eq!(
        sparse_batch
            .reference_mind_input(observer, randomness)
            .unwrap(),
        sparse_expected
    );
}

#[test]
fn decision_memory_is_atomic_bounded_and_advances_on_rejection() {
    let rules = ReferenceRuleset {
        max_private_memory_bytes: 3,
        ..ReferenceRuleset::default()
    };
    let mut simulation = ReferenceSimulation::new(1, 1, rules).unwrap();
    let actor = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 0)
        .unwrap();
    let before = simulation.state_hash();
    assert_eq!(
        simulation.commit_decision(actor, ActionRequest::Wait, vec![0; 4]),
        Err(CommitError::PrivateMemoryTooLarge {
            actual: 4,
            limit: 3,
        })
    );
    assert_eq!(simulation.state_hash(), before);

    let receipt = simulation
        .commit_decision(
            actor,
            ActionRequest::Move {
                target: LocalSlot(4),
                effort: EffortTier::Standard,
            },
            vec![1, 2, 3],
        )
        .unwrap();
    assert!(!receipt.accepted);
    assert_eq!(
        simulation.cell(actor).unwrap().private_memory.as_ref(),
        [1, 2, 3]
    );
}
