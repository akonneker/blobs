use blob_engine::resolution::{
    ActionRequest, DeltaError, DurationRule, LocalSlot, ReferenceRuleset, ReferenceSimulation,
};

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
        metabolism_rate_numerator: 0,
        diffusion_rate_numerator: 0,
        ..ReferenceRuleset::default()
    }
}

#[test]
fn batch_delta_reverses_and_reapplies_the_exact_hashed_state() {
    let mut simulation = ReferenceSimulation::new(2, 1, uniform_rules()).unwrap();
    let parent = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 1)
        .unwrap();
    simulation
        .commit_action(
            parent,
            ActionRequest::Split {
                target: EAST,
                child_allocation: 30,
                marker: 9,
                private_memory: vec![1, 2, 3],
            },
        )
        .unwrap();
    let report = simulation.resolve_next_batch().unwrap();
    let compiled = simulation.compiled_ruleset_hash();
    let post = simulation.canonical_state();

    let mut reconstructed_pre = post.clone();
    report.delta.apply_backward(&mut reconstructed_pre).unwrap();
    assert_eq!(
        Some(reconstructed_pre.hash_with_compiled_ruleset(compiled)),
        report.pre_state_hash()
    );
    report.delta.apply_forward(&mut reconstructed_pre).unwrap();
    assert_eq!(reconstructed_pre, post);
    assert_eq!(
        Some(reconstructed_pre.hash_with_compiled_ruleset(compiled)),
        report.state_hash()
    );
}

#[test]
fn delta_rejects_the_wrong_source_state_without_partial_mutation() {
    let mut simulation = ReferenceSimulation::new(2, 1, uniform_rules()).unwrap();
    let actor = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 1)
        .unwrap();
    simulation
        .commit_action(actor, ActionRequest::Wait)
        .unwrap();
    let report = simulation.resolve_next_batch().unwrap();
    let mut post = simulation.canonical_state();
    report.delta.apply_backward(&mut post).unwrap();
    post.cells[0].1.assimilated_energy += 1;
    let corrupted = post.clone();

    assert!(matches!(
        report.delta.apply_forward(&mut post),
        Err(DeltaError::CellMismatch(_))
    ));
    assert_eq!(post, corrupted);
}

#[test]
fn unchanged_resources_are_omitted_and_delta_order_is_canonical() {
    let mut simulation = ReferenceSimulation::new(4, 1, uniform_rules()).unwrap();
    let actor = simulation
        .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 1)
        .unwrap();
    simulation
        .commit_action(
            actor,
            ActionRequest::Split {
                target: EAST,
                child_allocation: 30,
                marker: 2,
                private_memory: Vec::new(),
            },
        )
        .unwrap();
    let report = simulation.resolve_next_batch().unwrap();

    assert!(report
        .delta
        .tiles
        .windows(2)
        .all(|pair| pair[0].tile < pair[1].tile));
    assert!(report
        .delta
        .cells
        .windows(2)
        .all(|pair| pair[0].cell < pair[1].cell));
    assert!(report
        .delta
        .tiles
        .iter()
        .all(|delta| delta.before != delta.after));
    assert!(report
        .delta
        .cells
        .iter()
        .all(|delta| delta.before != delta.after));
    assert!(report.delta.tiles.iter().all(|delta| delta.tile.0 < 2));
}
