use blob_interface::randomness::PrivateRandom;
use blob_interface::reference_mind::{
    CurrentTileObservation, EFFORT_BURST_BIT, EFFORT_GENTLE_BIT, EFFORT_STANDARD_BIT,
    LOCAL_NEIGHBOR_BIT, LOCAL_NEIGHBOR_MARKER_BIT, LOCAL_NEIGHBOR_PROGRESS_BIT, LocalObservation,
    ReferenceActionSpace, ReferenceActivity, ReferenceEffort, ReferenceMemoryUpdate,
    ReferenceMindAction, ReferenceMindDecision, ReferenceMindInput, ReferenceNeighbor,
    ReferenceOutcome, ReferenceOutcomeStatus, ReferenceProgress, ReferenceRejectReason,
    ReferenceSelfState, ReferenceSignalEmission,
};
use blob_interface::reference_mind_converter::{
    ReferenceMindLimits, capnp_to_reference_mind_decision, capnp_to_reference_mind_input,
    reference_mind_decision_to_capnp, reference_mind_input_to_capnp,
};

#[test]
#[ignore = "release-only owned observation decode benchmark"]
fn benchmark_owned_observation_decode() {
    const WARMUP: usize = 5_000;
    const ITERATIONS: usize = 100_000;
    let limits = ReferenceMindLimits::default();
    let mut input = fixture();
    let occupied = input.slots[0].clone();
    let empty = input.slots[1].clone();
    input.slots = (0..8)
        .map(|slot| {
            let mut observation = if slot % 2 == 0 {
                occupied.clone()
            } else {
                empty.clone()
            };
            observation.slot = slot;
            observation
        })
        .collect();
    input.action_space.move_targets = 0xff;
    input.action_space.attack_targets = 0xff;
    input.action_space.split_targets = 0xff;
    input.action_space.regurgitate_targets = 0xff;
    let bytes = reference_mind_input_to_capnp(&input, limits).unwrap();
    for _ in 0..WARMUP {
        std::hint::black_box(capnp_to_reference_mind_input(&bytes, limits).unwrap());
    }
    let started = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(capnp_to_reference_mind_input(&bytes, limits).unwrap());
    }
    let elapsed = started.elapsed();
    println!(
        "owned observation decode: {:.0} ns/input ({} bytes)",
        elapsed.as_nanos() as f64 / ITERATIONS as f64,
        bytes.len()
    );
}

fn fixture() -> ReferenceMindInput {
    ReferenceMindInput {
        self_state: ReferenceSelfState {
            core_mass: 11,
            assimilated_energy: 97,
            gut_energy: 13,
            metabolism_remainder: 511,
            carried_material_mass: 5,
            marker: 0x1234_5678,
            guarded: true,
            last_outcome: Some(ReferenceOutcome {
                status: ReferenceOutcomeStatus::Rejected,
                rejected_reason: Some(ReferenceRejectReason::InsufficientEnergy),
            }),
        },
        current_tile: CurrentTileObservation {
            elevation: -2,
            plant_energy: 21,
            plant_capacity: 80,
            plant_growth_rate: 7,
            loose_energy: 8,
            diffuse_energy: 3,
            signal_energy: [1, 2, 3, 4],
        },
        slots: vec![
            LocalObservation {
                slot: 0,
                dx: -1,
                dy: -1,
                distance_cost_q10: 1448,
                reachable: true,
                elevation: Some(-1),
                plant_energy: Some(34),
                plant_capacity: Some(100),
                plant_growth_rate: Some(5),
                loose_energy: None,
                diffuse_energy: Some(1),
                signal_energy: Some([4, 3, 2, 1]),
                neighbor: Some(ReferenceNeighbor {
                    marker: Some(9),
                    apparent_mass_bucket: Some(4),
                    activity: Some(ReferenceActivity::AttackWindup),
                    progress: Some(ReferenceProgress::Middle),
                }),
            },
            LocalObservation {
                slot: 1,
                dx: 0,
                dy: -1,
                distance_cost_q10: 1024,
                reachable: false,
                elevation: None,
                plant_energy: None,
                plant_capacity: None,
                plant_growth_rate: None,
                loose_energy: None,
                diffuse_energy: None,
                signal_energy: None,
                neighbor: None,
            },
        ],
        action_space: ReferenceActionSpace {
            wait_enabled: true,
            guard_enabled: true,
            consume_enabled: true,
            excavate_enabled: true,
            deposit_terrain_enabled: true,
            signal_enabled: true,
            move_targets: 0b01,
            attack_targets: 0b11,
            split_targets: 0b01,
            regurgitate_targets: 0b10,
            effort_mask: EFFORT_GENTLE_BIT | EFFORT_STANDARD_BIT | EFFORT_BURST_BIT,
            max_consume_amount: 16,
            gut_capacity: 64,
            max_private_memory_bytes: 2048,
            minimum_survival_energy: 1,
            child_core_mass: 10,
            metabolism_rate_numerator: 1,
            metabolism_rate_denominator: 1024,
            terrain_mass_per_elevation: 10,
            signal_emission_cost: 1,
        },
        private_memory: vec![1, 2, 3, 4],
        randomness: PrivateRandom::from_bytes([0xa5; 32]),
    }
}

#[test]
fn representative_input_round_trips_exactly() {
    let limits = ReferenceMindLimits::default();
    let input = fixture();
    let encoded = reference_mind_input_to_capnp(&input, limits).unwrap();
    assert_eq!(
        capnp_to_reference_mind_input(&encoded, limits).unwrap(),
        input
    );
}

#[test]
fn every_action_round_trips_exactly() {
    let actions = [
        ReferenceMindAction::Wait,
        ReferenceMindAction::Move {
            target_slot: 2,
            effort: ReferenceEffort::Gentle,
        },
        ReferenceMindAction::Attack {
            target_slot: 3,
            effort: ReferenceEffort::Burst,
            payload: 55,
        },
        ReferenceMindAction::Guard {
            effort: ReferenceEffort::Standard,
        },
        ReferenceMindAction::Consume { amount: 7 },
        ReferenceMindAction::Split {
            target_slot: 4,
            child_allocation: 23,
            marker: 99,
            private_memory: vec![8, 9],
        },
        ReferenceMindAction::Regurgitate {
            target_slot: 5,
            amount: 6,
        },
        ReferenceMindAction::Excavate,
        ReferenceMindAction::DepositTerrain,
    ];
    let limits = ReferenceMindLimits::default();
    for (index, action) in actions.into_iter().enumerate() {
        let decision = ReferenceMindDecision {
            action,
            signal: None,
            memory_update: ReferenceMemoryUpdate::Replace(vec![index as u8, 0xa5]),
        };
        let encoded = reference_mind_decision_to_capnp(&decision, limits).unwrap();
        assert_eq!(
            capnp_to_reference_mind_decision(&encoded, limits).unwrap(),
            decision
        );
    }

    let signaled = ReferenceMindDecision {
        action: ReferenceMindAction::Wait,
        signal: Some(ReferenceSignalEmission { channel: 3 }),
        memory_update: ReferenceMemoryUpdate::Replace(vec![4, 3, 2, 1]),
    };
    let encoded = reference_mind_decision_to_capnp(&signaled, limits).unwrap();
    assert_eq!(
        capnp_to_reference_mind_decision(&encoded, limits).unwrap(),
        signaled
    );

    let retained = ReferenceMindDecision {
        action: ReferenceMindAction::Wait,
        signal: None,
        memory_update: ReferenceMemoryUpdate::Retain,
    };
    let encoded = reference_mind_decision_to_capnp(&retained, limits).unwrap();
    assert_eq!(
        capnp_to_reference_mind_decision(&encoded, limits).unwrap(),
        retained
    );
}

#[test]
fn malformed_and_noncanonical_inputs_are_rejected() {
    let limits = ReferenceMindLimits::default();
    let encoded = reference_mind_input_to_capnp(&fixture(), limits).unwrap();
    for end in (0..encoded.len()).step_by(8) {
        assert!(
            capnp_to_reference_mind_input(&encoded[..end], limits).is_err(),
            "truncated input of {end} bytes was accepted"
        );
    }

    let mut noncanonical = fixture();
    noncanonical.slots[1].slot = 7;
    assert!(reference_mind_input_to_capnp(&noncanonical, limits).is_err());

    let mut impossible_target = fixture();
    impossible_target.action_space.move_targets |= 1 << 8;
    assert!(reference_mind_input_to_capnp(&impossible_target, limits).is_err());

    let mut inconsistent_outcome = fixture();
    inconsistent_outcome.self_state.last_outcome = Some(ReferenceOutcome {
        status: ReferenceOutcomeStatus::Success,
        rejected_reason: Some(ReferenceRejectReason::InvalidSlot),
    });
    assert!(reference_mind_input_to_capnp(&inconsistent_outcome, limits).is_err());

    let mut zero_metabolism_denominator = fixture();
    zero_metabolism_denominator
        .action_space
        .metabolism_rate_denominator = 0;
    assert!(reference_mind_input_to_capnp(&zero_metabolism_denominator, limits).is_err());

    let mut invalid_metabolism_remainder = fixture();
    invalid_metabolism_remainder.self_state.metabolism_remainder = 1024;
    assert!(reference_mind_input_to_capnp(&invalid_metabolism_remainder, limits).is_err());

    let mut remainder_without_energy = fixture();
    remainder_without_energy.self_state.assimilated_energy = 0;
    assert!(reference_mind_input_to_capnp(&remainder_without_energy, limits).is_err());
}

#[test]
fn configured_resource_limits_are_enforced() {
    let limits = ReferenceMindLimits {
        max_slots: 1,
        max_private_memory_bytes: 4,
        ..ReferenceMindLimits::default()
    };
    assert!(reference_mind_input_to_capnp(&fixture(), limits).is_err());

    let decision = ReferenceMindDecision {
        action: ReferenceMindAction::Split {
            target_slot: 0,
            child_allocation: 20,
            marker: 0,
            private_memory: vec![0; 5],
        },
        signal: None,
        memory_update: ReferenceMemoryUpdate::Retain,
    };
    assert!(reference_mind_decision_to_capnp(&decision, limits).is_err());

    let oversized_state = ReferenceMindDecision {
        action: ReferenceMindAction::Wait,
        signal: None,
        memory_update: ReferenceMemoryUpdate::Replace(vec![0; 5]),
    };
    assert!(reference_mind_decision_to_capnp(&oversized_state, limits).is_err());
}

#[test]
fn malformed_randomness_length_is_rejected() {
    use blob_interface::reference_mind_capnp::reference_mind_input;
    use capnp::serialize;

    let mut message = capnp::message::Builder::new_default();
    message
        .init_root::<reference_mind_input::Builder>()
        .set_randomness(&[1, 2, 3]);
    let mut bytes = Vec::new();
    serialize::write_message(&mut bytes, &message).unwrap();
    assert!(capnp_to_reference_mind_input(&bytes, ReferenceMindLimits::default()).is_err());
}

#[test]
fn compact_visibility_encoding_rejects_noncanonical_hidden_values() {
    use blob_interface::reference_mind_capnp::reference_mind_input;

    fn encoded(visibility: u16, hidden_plant_energy: u64) -> Vec<u8> {
        let mut message = capnp::message::Builder::new_default();
        {
            let mut root = message.init_root::<reference_mind_input::Builder>();
            root.reborrow()
                .init_self_state()
                .reborrow()
                .init_last_outcome()
                .set_none(());
            root.reborrow().init_current_tile().init_signal_energy(4);
            let mut slot = root.reborrow().init_slots(1).get(0);
            slot.set_slot(0);
            slot.set_visibility(visibility);
            slot.set_plant_energy(hidden_plant_energy);
            let mut action = root.reborrow().init_action_space();
            action.set_metabolism_rate_denominator(1);
            action.set_terrain_mass_per_elevation(1);
            root.set_private_memory(&[]);
            root.set_randomness(&[0; 32]);
        }
        capnp::serialize::write_message_to_words(&message)
    }

    let limits = ReferenceMindLimits::default();
    for bytes in [
        encoded(1 << 15, 0),
        encoded(0, 1),
        encoded(LOCAL_NEIGHBOR_MARKER_BIT, 0),
        encoded(LOCAL_NEIGHBOR_BIT | LOCAL_NEIGHBOR_PROGRESS_BIT, 0),
    ] {
        assert!(capnp_to_reference_mind_input(&bytes, limits).is_err());
    }
}

#[test]
fn retained_memory_with_replacement_bytes_is_rejected() {
    use blob_interface::reference_mind_capnp::reference_mind_decision;
    use capnp::serialize;

    let mut message = capnp::message::Builder::new_default();
    {
        let mut root = message.init_root::<reference_mind_decision::Builder>();
        root.reborrow().init_action().set_wait(());
        root.set_next_private_memory(&[1]);
        root.set_retain_private_memory(true);
        root.reborrow().init_signal().set_none(());
    }
    let mut bytes = Vec::new();
    serialize::write_message(&mut bytes, &message).unwrap();
    assert!(capnp_to_reference_mind_decision(&bytes, ReferenceMindLimits::default()).is_err());
}

#[test]
fn schema_contains_no_identity_global_clock_or_seed_channel() {
    let schema = include_str!("../interface/reference_mind.capnp");
    for forbidden in [
        "cellId",
        "publicId",
        "teamId",
        "absoluteCoordinate",
        "globalTime",
        "populationCount",
        "sharedSeed",
        "randomSeed",
        "inbox",
        "senderId",
    ] {
        assert!(
            !schema.contains(forbidden),
            "reference Mind ABI contains forbidden channel {forbidden}"
        );
    }
}
