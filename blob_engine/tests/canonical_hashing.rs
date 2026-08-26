use blob_engine::resolution::{
    ActionRequest, BoundaryRule, EffortTier, LocalSlot, NeighborhoodSpec, ReferenceRuleset,
    ReferenceSimulation, SlotMask, TargetingAction,
};

const WEST: LocalSlot = LocalSlot(3);
const EAST: LocalSlot = LocalSlot(4);
const NORTH: LocalSlot = LocalSlot(1);

fn assert_cached_hash_matches_oracle(simulation: &ReferenceSimulation) {
    assert_eq!(
        simulation.state_hash(),
        simulation
            .canonical_state()
            .hash_with_compiled_ruleset(simulation.compiled_ruleset_hash())
    );
}

fn build_order_case(reverse: bool) -> ReferenceSimulation {
    let mut simulation = ReferenceSimulation::new(5, 2, ReferenceRuleset::default()).unwrap();
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
        (
            left,
            ActionRequest::Move {
                target: EAST,
                effort: EffortTier::Standard,
            },
        ),
        (
            right,
            ActionRequest::Move {
                target: WEST,
                effort: EffortTier::Standard,
            },
        ),
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
    if reverse {
        actions.reverse();
    }
    for (actor, action) in actions {
        simulation.commit_action(actor, action).unwrap();
    }
    simulation
}

#[test]
fn ruleset_hash_covers_parameters_neighborhood_and_compiled_dimensions() {
    let base = ReferenceSimulation::new(3, 2, ReferenceRuleset::default()).unwrap();
    let same = ReferenceSimulation::new(3, 2, ReferenceRuleset::default()).unwrap();
    assert_eq!(base.semantic_ruleset_hash(), same.semantic_ruleset_hash());
    assert_eq!(base.compiled_ruleset_hash(), same.compiled_ruleset_hash());

    let mut changed_parameter = ReferenceRuleset::default();
    changed_parameter.bite_capacity += 1;
    let changed_parameter = ReferenceSimulation::new(3, 2, changed_parameter).unwrap();
    assert_ne!(
        base.semantic_ruleset_hash(),
        changed_parameter.semantic_ruleset_hash()
    );

    let mut changed_mask = ReferenceRuleset::default();
    changed_mask
        .neighborhood
        .set_target_mask(TargetingAction::Regurgitate, SlotMask::from_slots([NORTH]));
    let changed_mask = ReferenceSimulation::new(3, 2, changed_mask).unwrap();
    assert_ne!(
        base.semantic_ruleset_hash(),
        changed_mask.semantic_ruleset_hash()
    );

    let changed_dimensions = ReferenceSimulation::new(4, 2, ReferenceRuleset::default()).unwrap();
    assert_eq!(
        base.semantic_ruleset_hash(),
        changed_dimensions.semantic_ruleset_hash()
    );
    assert_ne!(
        base.compiled_ruleset_hash(),
        changed_dimensions.compiled_ruleset_hash()
    );

    let wrapped_rules = ReferenceRuleset {
        neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Wrap),
        ..ReferenceRuleset::default()
    };
    let wrapped = ReferenceSimulation::new(3, 2, wrapped_rules).unwrap();
    assert_ne!(
        base.semantic_ruleset_hash(),
        wrapped.semantic_ruleset_hash()
    );
}

#[test]
fn state_hash_tracks_authoritative_state_and_batch_reports() {
    let mut simulation = ReferenceSimulation::new(2, 1, ReferenceRuleset::default()).unwrap();
    let tile = simulation.tile(0, 0).unwrap();
    assert_cached_hash_matches_oracle(&simulation);
    let initial = simulation.state_hash();
    simulation.tile_state_mut(tile).unwrap().plant_energy = 7;
    assert_cached_hash_matches_oracle(&simulation);
    assert_ne!(initial, simulation.state_hash());
    simulation.tile_state_mut(tile).unwrap().plant_energy = 0;
    assert_eq!(initial, simulation.state_hash());

    let actor = simulation.add_cell(tile, 10, 100, 7).unwrap();
    assert_cached_hash_matches_oracle(&simulation);
    let ready = simulation.state_hash();
    simulation
        .commit_action(actor, ActionRequest::Wait)
        .unwrap();
    assert_cached_hash_matches_oracle(&simulation);
    let pending = simulation.state_hash();
    assert_ne!(ready, pending);

    let report = simulation.resolve_next_batch().unwrap();
    assert_cached_hash_matches_oracle(&simulation);
    assert_eq!(
        report.compiled_ruleset_hash,
        simulation.compiled_ruleset_hash()
    );
    assert_eq!(report.state_hash(), Some(simulation.state_hash()));
    assert_ne!(Some(pending), report.state_hash());
}

#[test]
fn state_hash_is_independent_of_commit_iteration_order() {
    let mut forward = build_order_case(false);
    let mut reverse = build_order_case(true);
    assert_eq!(forward.state_hash(), reverse.state_hash());

    let forward_report = forward.resolve_next_batch().unwrap();
    let reverse_report = reverse.resolve_next_batch().unwrap();
    assert_eq!(forward_report, reverse_report);
    assert_eq!(forward.state_hash(), reverse.state_hash());
}

#[test]
fn incremental_pages_match_oracle_across_tile_and_cell_boundaries() {
    let rules = ReferenceRuleset {
        digestion_rate_numerator: 0,
        metabolism_rate_numerator: 0,
        signal_decay_rate_numerator: 0,
        diffusion_rate_numerator: 0,
        ..ReferenceRuleset::default()
    };
    let mut simulation = ReferenceSimulation::new(32, 32, rules).unwrap();
    let mut actors = Vec::new();
    for index in 0..300 {
        actors.push(
            simulation
                .add_cell(simulation.tile(index % 32, index / 32).unwrap(), 10, 100, 0)
                .unwrap(),
        );
        if matches!(index, 255 | 256) {
            assert_cached_hash_matches_oracle(&simulation);
        }
    }
    simulation
        .tile_state_mut(simulation.tile(0, 9).unwrap())
        .unwrap()
        .loose_energy = 17;
    assert_cached_hash_matches_oracle(&simulation);

    for (index, actor) in actors.into_iter().enumerate() {
        if index == 257 {
            simulation
                .commit_decision(actor, ActionRequest::Wait, vec![0x5a; 2_048])
                .unwrap();
        } else {
            simulation
                .commit_action(actor, ActionRequest::Wait)
                .unwrap();
        }
    }
    assert_cached_hash_matches_oracle(&simulation);
    simulation.resolve_next_batch().unwrap();
    assert_cached_hash_matches_oracle(&simulation);
}

#[test]
fn canonical_hash_golden_vectors() {
    let mut simulation = ReferenceSimulation::new(3, 2, ReferenceRuleset::default()).unwrap();
    let attacker_tile = simulation.tile(0, 0).unwrap();
    let target_tile = simulation.tile(1, 0).unwrap();
    simulation.tile_state_mut(target_tile).unwrap().elevation = -2;
    {
        let tile = simulation.tile_state_mut(target_tile).unwrap();
        tile.plant_energy = 11;
        tile.plant_capacity = 40;
        tile.plant_growth_rate = 3;
        tile.plant_growth_remainder = 17;
        tile.loose_energy = 5;
    }
    let attacker = simulation.add_cell(attacker_tile, 10, 100, 7).unwrap();
    simulation.add_cell(target_tile, 12, 80, 9).unwrap();
    simulation
        .commit_action(
            attacker,
            ActionRequest::Attack {
                target: EAST,
                effort: EffortTier::Standard,
                payload: 20,
            },
        )
        .unwrap();

    let semantic = simulation.semantic_ruleset_hash().to_hex();
    let compiled = simulation.compiled_ruleset_hash().to_hex();
    let pending = simulation.state_hash().to_hex();
    let report = simulation.resolve_next_batch().unwrap();
    assert_eq!(
        (
            semantic,
            compiled,
            pending,
            report.state_hash().expect("verified report").to_hex(),
        ),
        (
            "63bb834b45315558ece15250608de71b9df007e18c45952751b424a4f96e389f".into(),
            "abb178db1afdf1e162d8d2c48fac0569d275747ef0dd6a17c9092e6d7cae5012".into(),
            "ce3864c933e36461b0248522e903ca0c6fa445093f510c55b16ded5a35e3eb9b".into(),
            "cd886383cdba6e8ff4aaa2df2c2a77fe59eb12c86e1a2aac17a1d07697fb0058".into(),
        )
    );
}
