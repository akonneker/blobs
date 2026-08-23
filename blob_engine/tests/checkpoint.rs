use blob_engine::resolution::{
    ActionRequest, BoundaryRule, CheckpointLimits, DurationRule, EffortTier, LocalSlot,
    NeighborhoodSpec, ReferenceCheckpoint, ReferenceRuleset, ReferenceSimulation,
};

fn checkpoint_rules() -> ReferenceRuleset {
    let duration = DurationRule::new(1024, 0, 1);
    ReferenceRuleset {
        neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Wrap),
        wait_duration: duration,
        move_duration: duration,
        attack_duration: duration,
        consume_duration: duration,
        split_duration: duration,
        regurgitate_duration: duration,
        digestion_rate_numerator: 0,
        metabolism_rate_numerator: 0,
        diffusion_rate_numerator: 0,
        ..ReferenceRuleset::default()
    }
}

fn pending_simulation() -> ReferenceSimulation {
    let mut simulation = ReferenceSimulation::new(4, 2, checkpoint_rules()).unwrap();
    let actor = simulation
        .add_cell(simulation.tile(1, 0).unwrap(), 10, 100, 3)
        .unwrap();
    let waiter = simulation
        .add_cell(simulation.tile(3, 1).unwrap(), 10, 80, 7)
        .unwrap();
    {
        let plant = simulation
            .tile_state_mut(simulation.tile(0, 1).unwrap())
            .unwrap();
        plant.plant_energy = 25;
        plant.plant_capacity = 50;
        plant.plant_growth_rate = 2;
        plant.plant_growth_remainder = 11;
        plant.diffuse_energy = 5;
    }
    simulation
        .commit_action(
            actor,
            ActionRequest::Move {
                target: LocalSlot(4),
                effort: EffortTier::High,
            },
        )
        .unwrap();
    simulation
        .commit_action(waiter, ActionRequest::Wait)
        .unwrap();
    simulation
}

#[test]
fn canonical_checkpoint_round_trips_pending_state_and_continuation() {
    let mut uninterrupted = pending_simulation();
    let checkpoint = ReferenceCheckpoint::from_simulation(&uninterrupted);
    assert_eq!(
        checkpoint.checkpoint_hash().to_hex(),
        "291ff1211b9fe64d7d5da5747e3691685f0088a3d78524d13bf90d9438bb097a"
    );
    let bytes = checkpoint.to_bytes();
    let decoded = ReferenceCheckpoint::from_bytes(&bytes).unwrap();

    assert_eq!(decoded.to_bytes(), bytes);
    assert_eq!(decoded.width(), 4);
    assert_eq!(decoded.height(), 2);
    assert_eq!(decoded.state_hash(), uninterrupted.state_hash());

    let mut resumed = decoded.into_simulation().unwrap();
    let direct_report = uninterrupted.resolve_next_batch().unwrap();
    let resumed_report = resumed.resolve_next_batch().unwrap();
    assert_eq!(resumed_report, direct_report);
    assert_eq!(resumed.canonical_state(), uninterrupted.canonical_state());
    assert_eq!(resumed.state_hash(), uninterrupted.state_hash());
}

#[test]
fn checkpoint_rejects_tampering_truncation_and_configured_limits() {
    let checkpoint = ReferenceCheckpoint::from_simulation(&pending_simulation());
    let bytes = checkpoint.to_bytes();

    let mut tampered = bytes.clone();
    tampered[32] ^= 0x40;
    assert!(ReferenceCheckpoint::from_bytes(&tampered).is_err());

    for length in 0..bytes.len() {
        assert!(ReferenceCheckpoint::from_bytes(&bytes[..length]).is_err());
    }

    let limits = CheckpointLimits {
        max_tiles: 2,
        ..CheckpointLimits::default()
    };
    assert!(ReferenceCheckpoint::from_bytes_with_limits(&bytes, limits).is_err());

    let limits = CheckpointLimits {
        max_checkpoint_bytes: bytes.len() - 1,
        ..CheckpointLimits::default()
    };
    assert!(ReferenceCheckpoint::from_bytes_with_limits(&bytes, limits).is_err());
}
